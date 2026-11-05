"""
boundary_head.py — auxiliary boundary-prediction branch off the BDM
U-Net decoder (architecturenew.docx, item 3). A lightweight conv head
that turns the final decoder feature map into a 1-channel boundary
logit map, supervised against the boundary pseudo-GT produced by
data/boundary_labels.py::mask_to_boundary.

Only active when configs/model_adaptive.yaml -> bdm.boundary_head.enabled
is true; the baseline config (bdm.boundary_head.enabled: false) never
constructs this module.
"""
import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, hidden_channels: int = None):
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, out_channels, H, W); apply sigmoid outside
        (kept as logits so boundary_bce_dice can use BCEWithLogits)."""
        return self.net(decoder_features)
