"""
boundary_labels.py — derives a boundary-map pseudo ground-truth from the
binary segmentation mask, since ISIC does not ship boundary annotations.

Used to supervise the adaptive model's auxiliary boundary head
(models/bdm/boundary_head.py) and the boundary_loss.

Two options are provided:
  - sobel: cheap, differentiable-friendly, produces a thicker gradient band
  - morphological: mask XOR eroded-mask, produces a thin, crisp 1px ring
                    (closer to what "edge of the lesion" should mean)

Both are intentionally simple — do not depend on OpenCV's Canny to keep
the dependency footprint small; swap in cv2.Canny if you want a sharper
edge and don't mind the dependency.
"""
import torch
import torch.nn.functional as F


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def mask_to_boundary_sobel(mask: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Args:
        mask: (1, H, W) or (B, 1, H, W) binary float tensor in {0,1}
    Returns:
        boundary: same shape, float tensor in {0,1}
    """
    squeeze_back = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        squeeze_back = True

    kx, ky = _sobel_kernels(mask.device, mask.dtype)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    grad_mag = grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
    boundary = (grad_mag > threshold).float()

    return boundary.squeeze(0) if squeeze_back else boundary


def mask_to_boundary_morphological(mask: torch.Tensor, erosion_iters: int = 1) -> torch.Tensor:
    """Boundary = mask - erode(mask), using a 3x3 min-pool as erosion."""
    squeeze_back = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        squeeze_back = True

    eroded = mask
    for _ in range(erosion_iters):
        eroded = -F.max_pool2d(-eroded, kernel_size=3, stride=1, padding=1)
    boundary = (mask - eroded).clamp(min=0.0, max=1.0)

    return boundary.squeeze(0) if squeeze_back else boundary


def mask_to_boundary(mask: torch.Tensor, method: str = "morphological", **kwargs) -> torch.Tensor:
    if method == "sobel":
        return mask_to_boundary_sobel(mask, **kwargs)
    elif method == "morphological":
        return mask_to_boundary_morphological(mask, **kwargs)
    else:
        raise ValueError(f"Unknown boundary method: {method}")