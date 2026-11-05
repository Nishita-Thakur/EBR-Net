"""model.py - GRU-Net-style replacement for the diffusion pipeline:

 pretrained encoder (timm) -> global-context transformer at stride 32 -> scSE U-Net decoder
 -> deep-supervision heads -> boundary head -> full-resolution residual refinement
 (the learned, differentiable replacement for ECRF: it sees raw RGB + coarse logits at full res)
"""
import torch, torch.nn as nn, torch.nn.functional as F, timm


def cba(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class SCSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 4), 1), nn.SiLU(),
                                 nn.Conv2d(max(c // r, 4), c, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x): return x * self.cse(x) + x * self.sse(x)


class GlobalContext(nn.Module):
    """Self-attention over the stride-32 map (12x12 tokens at 384px: negligible cost)."""
    def __init__(self, c, d=256, heads=8, depth=2):
        super().__init__()
        self.inp, self.out = nn.Conv2d(c, d, 1), nn.Conv2d(d, c, 1)
        self.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)  # conditional positional encoding
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True,
                                                                norm_first=True, activation="gelu")
                                     for _ in range(depth)])

    def forward(self, x):
        z = self.inp(x); z = z + self.pos(z)
        B, C, H, W = z.shape
        t = z.flatten(2).transpose(1, 2)
        for b in self.blocks: t = b(t)
        return x + self.out(t.transpose(1, 2).reshape(B, C, H, W))


class UpBlock(nn.Module):
    def __init__(self, i, s, o):
        super().__init__()
        self.conv = nn.Sequential(cba(i + s, o), cba(o, o)); self.att = SCSE(o)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.att(self.conv(torch.cat([x, skip], 1)))


class Refine(nn.Module):
    """Full-res residual refinement: RGB + decoder feats + coarse logit -> logit correction."""
    def __init__(self, feat_c, mid=32):
        super().__init__()
        self.f = cba(feat_c, 16, 1)
        self.body = nn.Sequential(cba(3 + 16 + 1, mid), cba(mid, mid, d=2), cba(mid, mid, d=4), cba(mid, mid, d=1))
        self.out = nn.Conv2d(mid, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # starts as identity on coarse logits

    def forward(self, img, feat, coarse):
        sz = img.shape[-2:]
        f = F.interpolate(self.f(feat), size=sz, mode="bilinear", align_corners=False)
        return coarse + self.out(self.body(torch.cat([img, f, coarse.detach()], 1)))


class SegNet(nn.Module):
    def __init__(self, encoder="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, dec=(256, 128, 64, 48),
                 use_refiner=True, use_aux=True):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        ch = self.enc.feature_info.channels()           # strides 2,4,8,16,32
        self.ctx = GlobalContext(ch[4])
        self.up3 = UpBlock(ch[4], ch[3], dec[0]); self.up2 = UpBlock(dec[0], ch[2], dec[1])
        self.up1 = UpBlock(dec[1], ch[1], dec[2]); self.up0 = UpBlock(dec[2], ch[0], dec[3])
        self.aux = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in dec])
        self.use_refiner, self.use_aux = use_refiner, use_aux
        if use_aux:
            self.bnd = nn.Sequential(cba(dec[3], 24), nn.Conv2d(24, 1, 1))
        if use_refiner:
            self.refine = Refine(dec[3])

    def forward(self, img):
        f = self.enc(img)
        d3 = self.up3(self.ctx(f[4]), f[3]); d2 = self.up2(d3, f[2])
        d1 = self.up1(d2, f[1]); d0 = self.up0(d1, f[0])
        up = lambda t: F.interpolate(t, size=img.shape[-2:], mode="bilinear", align_corners=False)
        coarse = up(self.aux[3](d0))
        final = self.refine(img, d0, coarse) if self.use_refiner else coarse
        if not self.training:
            return {"logits": final}
        out = {"logits": final}
        if self.use_refiner:
            out["coarse"] = coarse
        if self.use_aux:
            out["aux"] = [self.aux[i](d) for i, d in enumerate((d3, d2, d1))]
            out["boundary"] = up(self.bnd(d0))
        return out
