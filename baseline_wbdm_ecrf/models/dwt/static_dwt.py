"""
static_dwt.py — reproduces the ORIGINAL paper's DWT block exactly
(Section 3.2): 2D Haar DWT, LL-only extraction, no gating.

The LL subband retains ~99.87% of spectral energy (paper Fig. 4) and is
used as a signal-purification step: it suppresses high-frequency noise
(hair, texture artifacts) while preserving the lesion's structural and
boundary information.

Used by configs/model_baseline.yaml (dwt.type == "static").
"""
import torch
import torch.nn as nn


def _haar_filters(dtype, device):
    """Return the 4 Haar analysis filters (LL, LH, HL, HH), each 2x2,
    normalized so that a flat region maps to itself in LL."""
    ll = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device) * 0.25
    lh = torch.tensor([[1, 1], [-1, -1]], dtype=dtype, device=device) * 0.5
    hl = torch.tensor([[1, -1], [1, -1]], dtype=dtype, device=device) * 0.5
    hh = torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device) * 0.5
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)


class StaticDWT(nn.Module):
    """Single-level 2D Haar DWT, depthwise across channels.

    forward(x) -> dict with keys 'LL','LH','HL','HH', each (B,C,H/2,W/2).
    """

    def __init__(self, channels: int = 3, wavelet: str = "haar"):
        super().__init__()
        if wavelet != "haar":
            raise NotImplementedError(
                f"Only the Haar wavelet is implemented (got '{wavelet}'); "
                "the paper's Sec 3.2 uses Haar exclusively."
            )
        self.channels = channels
        filters = _haar_filters(torch.float32, torch.device("cpu"))
        # (4,1,2,2) -> repeat per input channel for grouped conv
        filters = filters.repeat(channels, 1, 1, 1)  # (4*C,1,2,2)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        assert C == self.channels, f"expected {self.channels} channels, got {C}"
        filters = self.filters.to(dtype=x.dtype, device=x.device)
        # grouped conv: each input channel produces 4 output channels (LL,LH,HL,HH)
        out = nn.functional.conv2d(x, filters, stride=2, groups=C)
        out = out.view(B, C, 4, H // 2, W // 2)
        subbands = {
            "LL": out[:, :, 0],
            "LH": out[:, :, 1],
            "HL": out[:, :, 2],
            "HH": out[:, :, 3],
        }
        return subbands

    def extract_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns only the LL subband (B, C, H/2, W/2),
        matching the paper's baseline input to the BDM."""
        return self.forward(x)["LL"]
