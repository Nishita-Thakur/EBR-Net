"""
postprocess.py — standalone post-processing utilities used both inside
models/ecrf (dilation + Gaussian smoothing, paper Sec 3.4) and for
turning a soft probability mask into a final binary PNG-ready mask.
"""
import numpy as np
import torch
import torch.nn.functional as F


def binarize(pred_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (pred_mask > threshold).float()


def largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component — a common
    clinical post-processing step to remove small spurious blobs (e.g.
    residual hair-artifact false positives, paper Sec 4.7)."""
    from scipy.ndimage import label

    labeled, num_features = label(binary_mask)
    if num_features == 0:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()
    return (labeled == largest_label).astype(binary_mask.dtype)


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(binary_mask).astype(binary_mask.dtype)


def to_uint8_png(binary_mask: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) float/binary tensor -> uint8 (H,W) array in {0,255}."""
    m = binary_mask.squeeze().detach().cpu().numpy()
    return (m > 0.5).astype(np.uint8) * 255


def full_postprocess_pipeline(pred_mask: torch.Tensor, threshold: float = 0.5, keep_largest_only: bool = True, fill: bool = True) -> np.ndarray:
    """End-to-end: soft prob map -> binarize -> (optional) largest CC -> (optional) fill holes -> uint8."""
    binary = binarize(pred_mask, threshold)
    mask_np = binary.squeeze().detach().cpu().numpy().astype(np.uint8)
    if keep_largest_only:
        mask_np = largest_connected_component(mask_np)
    if fill:
        mask_np = fill_holes(mask_np)
    return mask_np * 255
