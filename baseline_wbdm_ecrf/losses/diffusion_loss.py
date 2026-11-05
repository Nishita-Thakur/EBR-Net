"""
diffusion_loss.py — the BDM training objective (paper Eq. 9): simplified
ELBO reduces to an MSE between the predicted and target noise terms.
BrownianBridgeDiffusion.training_losses() already computes this target
internally; this module exposes it as a standalone function so it can be
reused/tested outside the model class (e.g. in tests/test_bridge_diffusion.py).
"""
import torch
import torch.nn.functional as F


def diffusion_mse_loss(pred_noise: torch.Tensor, target: torch.Tensor, correction_coeff: float = 1.0) -> torch.Tensor:
    """MSE(target, pred_noise), optionally scaled by the theoretical
    Brownian-bridge correction coefficient C from Eq. 9 (Li et al. 2023,
    BBDM). C defaults to 1.0 since it is treated as a constant scale on
    an already-normalized loss in most BBDM implementations."""
    return correction_coeff * F.mse_loss(pred_noise, target)
