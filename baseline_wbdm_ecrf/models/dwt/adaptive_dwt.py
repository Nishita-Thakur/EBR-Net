"""
adaptive_dwt.py — YOUR modified DWT block (architecturenew.docx, items 1-2):
keeps all four Haar subbands (LL/LH/HL/HH) instead of discarding LH/HL/HH,
and fuses them with a learned per-subband/per-channel gate (SEFrequencyGate)
instead of raw concatenation.

Used by configs/model_adaptive.yaml (dwt.type == "adaptive").
"""
import torch
import torch.nn as nn

from .static_dwt import StaticDWT
from .frequency_gate import SEFrequencyGate


class AdaptiveDWT(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        wavelet: str = "haar",
        keep_subbands=("LL", "LH", "HL", "HH"),
        reduction_ratio: int = 8,
        init_bias_toward_LL: bool = True,
        init_prior=(0.9987, 0.0004, 0.0004, 0.0005),
    ):
        super().__init__()
        self.dwt = StaticDWT(channels=channels, wavelet=wavelet)
        self.keep_subbands = list(keep_subbands)
        self.gate = SEFrequencyGate(
            channels=channels,
            reduction_ratio=reduction_ratio,
            init_bias_toward_LL=init_bias_toward_LL,
            init_prior=init_prior,
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            fused: (B, C, H/2, W/2) — gated fusion of the kept subbands,
                same channel count as the input, ready for the BDM U-Net.
            aux: dict with 'gate' (B,4,C) gate weights and 'subbands' (the
                raw subband dict), useful for logging/regularization.
        """
        subbands = self.dwt(x)
        for k in list(subbands.keys()):
            if k not in self.keep_subbands:
                subbands[k] = torch.zeros_like(subbands[k])
        fused, gate = self.gate(subbands)
        aux = {"gate": gate, "subbands": subbands}
        return fused, aux

    def regularization_loss(self, aux: dict, epoch: int, warmup_epochs: int, weight: float) -> torch.Tensor:
        """Prior-KL regularization (model_adaptive.yaml -> dwt.gate.regularization).

        IMPORTANT: this is a genuine linear ramp down to zero across
        [0, warmup_epochs), not a hard cutoff. The previous implementation
        applied full weight for epoch < warmup_epochs and then EXACTLY
        ZERO regularization for every epoch after that -- with
        warmup_epochs=20 out of a 200-epoch run, that meant the gate had
        no anchoring pressure at all for 90% of training, free to drift
        away from the empirically-justified, mostly-LL energy prior based
        purely on noisy, indirect gradients from the downstream diffusion
        loss. A true linear ramp keeps meaningful (if decreasing)
        regularization pressure for the whole warmup window instead of
        releasing it all at once partway through.
        """
        if epoch >= warmup_epochs:
            return torch.tensor(0.0, device=aux["gate"].device)
        kl = self.gate.prior_kl_loss(aux["gate"])
        ramp = 1.0 - (epoch / max(warmup_epochs, 1))
        return weight * ramp * kl