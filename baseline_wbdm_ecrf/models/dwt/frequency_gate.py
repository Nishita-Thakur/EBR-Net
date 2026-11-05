"""
frequency_gate.py — learned per-subband, per-channel gate for the
adaptive DWT block (see architecturenew.docx, item 1-2, and
configs/model_adaptive.yaml -> dwt.gate).

The paper's own ablation (Table 6) shows that naively concatenating all
four Haar subbands crashes performance (Dice 94.45% -> 40.23%) because
LH/HL/HH are dominated by hair/texture noise. Instead of feeding the raw
subbands, we squeeze-excite-gate each subband so the network can learn
to suppress noisy high-frequency content while still letting genuine
edge information through when it is present.

The gate is initialized close to the paper's own measured energy
distribution (Fig. 4: ~99.87% LL, ~0.04% each of LH/HL/HH) via
`init_prior`, and is softly regularized toward that prior early in
training (`prior_kl`) so it does not need to rediscover the paper's
result from scratch; the regularization is relaxed after `warmup_epochs`
so the gate can still learn to deviate where it helps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEFrequencyGate(nn.Module):
    """Squeeze-excite style gate producing one weight per (subband, channel).

    Input: dict of 4 subband tensors, each (B, C, H, W).
    Output: (fused (B, C, H, W), gate_weights (B, 4, C)) where fused is the
    weighted sum of subbands and gate_weights are the sigmoid gate values
    used for the KL-to-prior regularizer.
    """

    SUBBAND_ORDER = ["LL", "LH", "HL", "HH"]

    def __init__(
        self,
        channels: int = 3,
        reduction_ratio: int = 8,
        init_bias_toward_LL: bool = True,
        init_prior=(0.9987, 0.0004, 0.0004, 0.0005),
    ):
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction_ratio, 4)

        # one shared squeeze-excite trunk over the 4*C stacked subbands
        self.fc1 = nn.Linear(4 * channels, hidden)
        self.fc2 = nn.Linear(hidden, 4 * channels)

        self.register_buffer("init_prior", torch.tensor(init_prior, dtype=torch.float32))

        if init_bias_toward_LL:
            self._init_bias_from_prior()

    def _init_bias_from_prior(self):
        # bias the gate logits so sigmoid(logit) ~= init_prior at t=0,
        # per subband, broadcast across channels.
        prior = self.init_prior.clamp(1e-4, 1 - 1e-4)
        logits = torch.log(prior / (1 - prior))  # inverse sigmoid
        bias = logits.repeat_interleave(self.channels)  # (4*C,)
        with torch.no_grad():
            self.fc2.bias.copy_(bias)
            self.fc2.weight.mul_(0.01)  # start near-constant output

    def forward(self, subbands: dict):
        B, C, H, W = subbands["LL"].shape
        stacked = torch.stack([subbands[k] for k in self.SUBBAND_ORDER], dim=1)  # (B,4,C,H,W)

        # squeeze: global average pool each subband/channel
        squeezed = stacked.mean(dim=(3, 4))  # (B,4,C)
        squeezed_flat = squeezed.view(B, 4 * C)

        hidden = F.relu(self.fc1(squeezed_flat))
        gate_logits = self.fc2(hidden).view(B, 4, C)
        gate = torch.sigmoid(gate_logits)  # (B,4,C) -- kept raw for KL regularization/logging

        # IMPORTANT: prior_kl_loss() normalizes `gate` before computing KL, so
        # that penalty only constrains the *relative* proportions between
        # subbands -- it does not stop the network from shrinking all four
        # raw sigmoids together (which barely changes the normalized
        # proportions, so KL stays low) while silently weakening the actual
        # fused signal fed into the diffusion bridge. Normalize explicitly
        # here, on the path that actually builds `fused`, so the total
        # weight across subbands is always 1 (matching the baseline's
        # implicit weight of 1.0 on LL) and the network can only reallocate
        # weight between subbands, not shrink overall signal energy for free.
        eps = 1e-8
        gate_norm = gate / (gate.sum(dim=1, keepdim=True) + eps)

        gated = stacked * gate_norm.unsqueeze(-1).unsqueeze(-1)  # (B,4,C,H,W)
        fused = gated.sum(dim=1)  # (B,C,H,W)

        return fused, gate

    def prior_kl_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """KL(gate_distribution || init_prior), averaged over batch/channel.

        `gate` is treated as an unnormalized energy share per subband; we
        normalize across the subband dim before computing KL, mirroring
        the paper's Fig. 4 energy-share analysis.
        """
        eps = 1e-8
        gate_dist = gate / (gate.sum(dim=1, keepdim=True) + eps)  # (B,4,C)
        prior = self.init_prior.clamp(eps, 1.0).view(1, 4, 1)
        prior = prior / prior.sum()
        kl = (gate_dist.clamp(eps, 1.0) * (gate_dist.clamp(eps, 1.0).log() - prior.log()))
        # KL divergence is mathematically >= 0 (Gibbs' inequality); when the
        # gate distribution sits almost exactly on the prior (as at init),
        # float32 rounding can push the sum a hair below zero. Clamp it out.
        return kl.sum(dim=1).mean().clamp(min=0.0)