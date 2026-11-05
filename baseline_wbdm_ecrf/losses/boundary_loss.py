"""
boundary_loss.py — supervises the auxiliary boundary head
(models/bdm/boundary_head.py) against the boundary pseudo-GT from
data/boundary_labels.py. Config: model_adaptive.yaml ->
bdm.boundary_head.loss == "boundary_bce_dice".

Combines BCE-with-logits (pixelwise classification) with a soft Dice
loss (region-overlap, robust to the boundary's class imbalance — edges
are a small fraction of pixels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred_prob.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def boundary_bce_dice_loss(boundary_logits: torch.Tensor, boundary_gt: torch.Tensor, bce_weight: float = 0.5, dice_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_gt)
    dice = soft_dice_loss(torch.sigmoid(boundary_logits), boundary_gt)
    return bce_weight * bce + dice_weight * dice
