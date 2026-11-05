"""
unet_backbone.py — the BDM's denoising network (paper Sec 3.3.1, Fig. 2):
a U-Net with skip connections, sinusoidal timestep embedding, and
FlashAttention blocks at the configured resolutions. Predicts the noise
epsilon_theta(x_t, t) used by the bridge diffusion process.

Config knobs (configs/model_*.yaml -> bdm.unet):
    in_channels, base_channels, channel_mults, use_flash_attention,
    attention_resolutions
"""
import math
import torch
import torch.nn as nn

from .flash_attention import FlashSelfAttention2d


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding, as in DDPM/Transformer positional encodings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect')
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='reflect')
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1, padding_mode='reflect')

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNetBackbone(nn.Module):
    """
    Args mirror configs/model_*.yaml -> bdm.unet, e.g.:
        in_channels: 3, base_channels: 64, channel_mults: [1,2,4,8],
        use_flash_attention: true, attention_resolutions: [16]

    Forward returns the predicted noise map, plus the final decoder
    feature map (needed by BoundaryHead for the auxiliary edge branch).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = None,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        use_flash_attention: bool = True,
        attention_resolutions=(16,),
        input_resolution: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.input_resolution = input_resolution
        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_emb_base = base_channels

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1, padding_mode='reflect')

        chans = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attn = nn.ModuleList()

        res = input_resolution
        in_ch = base_channels
        skip_chs = [base_channels]
        for i, out_ch in enumerate(chans):
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_emb_dim))
            self.down_attn.append(
                FlashSelfAttention2d(out_ch, num_heads=num_heads)
                if use_flash_attention and res in attention_resolutions
                else nn.Identity()
            )
            in_ch = out_ch
            skip_chs.append(in_ch)
            if i < len(chans) - 1:
                self.down_samples.append(Downsample(in_ch))
                res //= 2
            else:
                self.down_samples.append(nn.Identity())

        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = (
            FlashSelfAttention2d(in_ch, num_heads=num_heads)
            if use_flash_attention
            else nn.Identity()
        )
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        rev_chans = list(reversed(chans))
        for i, out_ch in enumerate(rev_chans):
            skip_ch = skip_chs.pop()
            self.up_blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim))
            self.up_attn.append(
                FlashSelfAttention2d(out_ch, num_heads=num_heads)
                if use_flash_attention and res in attention_resolutions
                else nn.Identity()
            )
            in_ch = out_ch
            if i < len(rev_chans) - 1:
                self.up_samples.append(Upsample(in_ch))
                res *= 2
            else:
                self.up_samples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1, padding_mode='reflect')
        self.act = nn.SiLU()
        self.final_feature_channels = in_ch

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_emb_base))

        h = self.stem(x)
        skips = [h]
        for block, attn, down in zip(self.down_blocks, self.down_attn, self.down_samples):
            h = block(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block, attn, up in zip(self.up_blocks, self.up_attn, self.up_samples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = attn(h)
            h = up(h)

        decoder_features = h
        out = self.out_conv(self.act(self.out_norm(h)))
        return out, decoder_features
