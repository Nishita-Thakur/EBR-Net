"""
schedulers.py — LR scheduling and the DWT gate's KL-regularization
warmup schedule (configs/model_adaptive.yaml -> dwt.gate.regularization.warmup_epochs).

base.yaml's training config uses a flat Adam LR (1e-4) with no explicit
scheduler in the paper; we provide a couple of standard, opt-in options
for longer/careful reproduction runs.
"""
import math
import torch


def build_lr_scheduler(optimizer: torch.optim.Optimizer, name: str = "constant", total_epochs: int = 200, warmup_epochs: int = 0):
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    if name == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 4), gamma=0.5)

    raise ValueError(f"Unknown scheduler: {name}")


class GateWarmupSchedule:
    """Tracks whether the DWT gate's prior-KL regularization is still
    active for a given epoch, per model_adaptive.yaml ->
    dwt.gate.regularization.warmup_epochs. The model itself already
    reads `epoch` directly (see models/wbdm_ecrf_adaptive.py); this
    helper exists mainly for logging/plotting the effective weight."""

    def __init__(self, weight: float, warmup_epochs: int):
        self.weight = weight
        self.warmup_epochs = warmup_epochs

    def weight_at(self, epoch: int) -> float:
        return self.weight if epoch < self.warmup_epochs else 0.0
