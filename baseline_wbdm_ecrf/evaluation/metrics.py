"""
metrics.py — Dice, IoU, ACC, SE, SP (paper Eqs. 19-23), plus HD95/ASSD
boundary-distance metrics used in Tables 3 and 7.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _confusion(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    dims = tuple(range(1, pred_bin.dim()))
    tp = (pred_bin * target_bin).sum(dim=dims)
    fp = (pred_bin * (1 - target_bin)).sum(dim=dims)
    fn = ((1 - pred_bin) * target_bin).sum(dim=dims)
    tn = ((1 - pred_bin) * (1 - target_bin)).sum(dim=dims)
    return tp, fp, fn, tn


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 20: Dice = 2TP / (2TP + FP + FN), per-sample."""
    tp, fp, fn, _ = _confusion(pred, target)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 21: IoU = TP / (TP + FP + FN), per-sample."""
    tp, fp, fn, _ = _confusion(pred, target)
    return (tp + eps) / (tp + fp + fn + eps)


def accuracy_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 19: ACC = (TP+TN) / (TP+TN+FP+FN)."""
    tp, fp, fn, tn = _confusion(pred, target)
    return (tp + tn + eps) / (tp + tn + fp + fn + eps)


def sensitivity_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 22: SE = TP / (TP + FN)."""
    tp, fp, fn, _ = _confusion(pred, target)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 23: SP = TN / (TN + FP)."""
    tp, fp, fn, tn = _confusion(pred, target)
    return (tn + eps) / (tn + fp + eps)


def _surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Return (N,2) array of boundary-pixel coordinates via erosion diff."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_mask)
    boundary = binary_mask & ~eroded
    return np.argwhere(boundary)


def hausdorff_distance_95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """HD95: 95th percentile of the symmetric surface-distance distribution."""
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    if not pred_b.any() or not gt_b.any():
        return float("nan")

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    pred_pts = _surface_points(pred_b)
    gt_pts = _surface_points(gt_b)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    d_pred_to_gt = dt_gt[pred_pts[:, 0], pred_pts[:, 1]]
    d_gt_to_pred = dt_pred[gt_pts[:, 0], gt_pts[:, 1]]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def average_symmetric_surface_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """ASSD: mean of the symmetric surface-distance distribution."""
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    if not pred_b.any() or not gt_b.any():
        return float("nan")

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    pred_pts = _surface_points(pred_b)
    gt_pts = _surface_points(gt_b)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    d_pred_to_gt = dt_gt[pred_pts[:, 0], pred_pts[:, 1]]
    d_gt_to_pred = dt_pred[gt_pts[:, 0], gt_pts[:, 1]]
    return float(np.concatenate([d_pred_to_gt, d_gt_to_pred]).mean())
