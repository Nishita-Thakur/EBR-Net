"""
loss_weighting.py — combines the diffusion loss, boundary loss, and DWT
gate regularization into the model's total loss, using the fixed weights
from configs/model_adaptive.yaml -> loss_weights:
    diffusion: 1.0, boundary: 0.3, gate_regularization: 0.01

Kept as a small standalone class (rather than inlined in the trainer) so
alternative weighting strategies (e.g. uncertainty-based or learned
weights) can be swapped in later without touching training/trainer.py.
"""
import torch


class LossWeighter:
    def __init__(self, diffusion: float = 1.0, boundary: float = 0.3, gate_regularization: float = 0.01):
        self.weights = {
            "diffusion_loss": diffusion,
            "boundary_loss": boundary,
            "gate_regularization_loss": gate_regularization,
        }

    def combine(self, losses: dict) -> (torch.Tensor, dict):
        """
        Args:
            losses: dict possibly containing 'diffusion_loss',
                'boundary_loss', 'gate_regularization_loss' tensors.
        Returns:
            total: scalar tensor to backprop
            components: dict of the (unweighted) per-term scalars for logging
        """
        total = 0.0
        components = {}
        for key, weight in self.weights.items():
            if key in losses and losses[key] is not None:
                term = losses[key]
                total = total + weight * term
                components[key] = term.detach()
        if isinstance(total, float):
            raise ValueError("No recognized loss terms found in `losses` dict.")
        return total, components

    @classmethod
    def from_config(cls, loss_weights_cfg: dict) -> "LossWeighter":
        return cls(
            diffusion=loss_weights_cfg.get("diffusion", 1.0),
            boundary=loss_weights_cfg.get("boundary", 0.3),
            gate_regularization=loss_weights_cfg.get("gate_regularization", 0.01),
        )
