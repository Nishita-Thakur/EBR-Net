"""zoo_models.py - comparison architectures trained inside the sls_v2 pipeline.
  unet   : the classic U-Net (Ronneberger et al., 2015), 4 down/up levels, 64..1024 channels, batch norm, same-size padding.
  unetpp : UNet++ (Zhou et al., 2018/2020): nested dense skip pathways, filters 32..512, batch norm, random initialisation,
           final output X(0,4) only (no deep supervision, so that the loss is the same as for every other model).
  attunet: Attention U-Net (Oktay et al., 2018): the U-Net above (64..1024 channels) with additive attention gates on the four skip
           connections (gating signal = up-sampled decoder feature, intermediate channels = half of the skip channels).
  fatnet : FAT-Net (Wu et al., 2022): the OFFICIAL model file (git clone https://github.com/SZUcsh/FAT-Net) is loaded
           from the path in $FATNET_PY. Its ImageNet-pretrained ResNet-34 and DeiT-tiny parts are fetched with current
           torchvision / timm calls (the original file uses the deprecated `pretrained=True` and torch.hub); the
           architecture itself is unchanged. FAT-Net needs a 224x224 input (14x14 transformer grid)."""
import importlib.util, os
import torch, torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(in_ch, c[0])] + [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in range(4)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x)); skips.append(x)
        x = skips.pop()
        for i in reversed(range(4)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.out(x)


class UNetPP(nn.Module):
    def __init__(self, in_ch=3, f=(32, 64, 128, 256, 512)):
        super().__init__()
        self.f = f; self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.n = nn.ModuleDict()
        for j in range(5):                       # column j (j=0 backbone)
            for i in range(5 - j):               # row i (depth)
                cin = (in_ch if i == 0 else f[i - 1]) if j == 0 else f[i] * j + f[i + 1]
                self.n[f"{i}_{j}"] = DoubleConv(cin, f[i])
        self.out = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        X = {}
        for j in range(5):
            for i in range(5 - j):
                if j == 0:
                    X[(i, j)] = self.n[f"{i}_{j}"](x if i == 0 else self.pool(X[(i - 1, 0)]))
                else:
                    X[(i, j)] = self.n[f"{i}_{j}"](torch.cat([X[(i, k)] for k in range(j)] + [self.up(X[(i + 1, j - 1)])], 1))
        return self.out(X[(0, 4)])


class AttGate(nn.Module):
    def __init__(self, fg, fl, fi):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(fg, fi, 1), nn.BatchNorm2d(fi))
        self.wx = nn.Sequential(nn.Conv2d(fl, fi, 1), nn.BatchNorm2d(fi))
        self.psi = nn.Sequential(nn.Conv2d(fi, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.wg(g) + self.wx(x)))


class AttUNet(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(in_ch, c[0])] + [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.att = nn.ModuleList([AttGate(c[i], c[i], c[i] // 2) for i in range(4)])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in range(4)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x)); skips.append(x)
        x = skips.pop()
        for i in reversed(range(4)):
            g = self.up[i](x)
            x = self.dec[i](torch.cat([self.att[i](g, skips[i]), g], 1))
        return self.out(x)


def build_fatnet(pretrained=True):
    import timm, torchvision.models as tvm
    path = os.environ.get("FATNET_PY", "/workspace/FAT-Net/FAT_Net.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run  git clone https://github.com/SZUcsh/FAT-Net.git /workspace/FAT-Net")
    spec = importlib.util.spec_from_file_location("fat_net_official", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    hub_load, r34 = torch.hub.load, tvm.resnet34
    torch.hub.load = lambda *a, **k: timm.create_model("deit_tiny_distilled_patch16_224", pretrained=pretrained)
    def _r34(*a, **k):
        m = r34(weights=None)
        if pretrained:                       # torchvision ImageNet weights, fetched through the Hugging Face hub (as timm does)
            m.load_state_dict(timm.create_model("resnet34.tv_in1k", pretrained=True).state_dict())
        return m
    tvm.resnet34 = _r34
    try:
        return mod.FAT_Net(n_channels=3, n_classes=1)
    finally:
        torch.hub.load, tvm.resnet34 = hub_load, r34


class Wrap(nn.Module):
    """Gives every comparison model the interface of the proposed network: model(x) -> {"logits": ...}."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return {"logits": self.m(x)}


DEFAULT_SIZE = {"unet": 256, "unetpp": 256, "attunet": 256, "fatnet": 224}
PRETRAINED_PREFIX = {"unet": (), "unetpp": (), "attunet": (), "fatnet": ("firstconv", "firstbn", "encoder", "patch_embed", "transformers")}


def build_model(arch, pretrained=True):
    if arch == "unet": return Wrap(UNet())
    if arch == "unetpp": return Wrap(UNetPP())
    if arch == "attunet": return Wrap(AttUNet())
    if arch == "fatnet": return Wrap(build_fatnet(pretrained))
    raise ValueError(arch)
