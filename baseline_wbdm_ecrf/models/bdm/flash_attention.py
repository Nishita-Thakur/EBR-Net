"""
flash_attention.py — QKV self-attention block for the BDM U-Net
(paper Sec 3.3.1 / Fig. 2 "QKVFlashAttention"), using PyTorch's
scaled_dot_product_attention which dispatches to a FlashAttention
kernel on supported hardware. This reduces the O(N^2) memory of naive
attention to O(N), which is what makes attention tractable at the
resolutions used here (paper: attention_resolutions == [16]).

Falls back gracefully (same numerical result, just no fused kernel) on
CPU or older GPUs, since torch's SDPA handles kernel selection
internally.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashSelfAttention2d(nn.Module):
    """Self-attention over spatial positions of a (B, C, H, W) feature map."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            # (B,C,H,W) -> (B, num_heads, H*W, head_dim)
            t = t.view(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2).contiguous()

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # Uses FlashAttention / memory-efficient kernels under the hood
        # when available (CUDA + fp16/bf16); O(N) memory instead of O(N^2).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return residual + out
