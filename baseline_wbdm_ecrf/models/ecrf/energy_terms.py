"""
energy_terms.py — the ECRF energy function shared by both static and
adaptive ECRF variants (paper Sec 3.4, Eqs. 15-18).

E(x) = sum_i Psi_u(x_i) + sum_{i<j} Psi_p(x_i, x_j)               [Eq. 15]

Psi_u(x_i) = -log P(x_i)                                          (unary)
Psi_p(x_i,x_j) = exp( -S/2*sigma_s^2 - color_diff/2*sigma_r^2
                       - (1-SSIM)/2*sigma_ssim^2 )                 [Eq. 16]
S(x_i,x_j) = squared Euclidean pixel-coordinate distance           [Eq. 17]
mu_c(x) = mean color over local region R(x), window_size x window_size [Eq. 18]
"""
import torch
import torch.nn.functional as F


def unary_potential(prior_prob: float, shape, device, dtype=torch.float32) -> torch.Tensor:
    """Psi_u(x_i) = -log P(x_i); P(x_i) is the prior prob. of the correct
    label, set to 0.9 by default per the paper (binary classification)."""
    p = torch.full(shape, prior_prob, device=device, dtype=dtype)
    return -torch.log(p.clamp(min=1e-8))


def spatial_term(coords_i: torch.Tensor, coords_j: torch.Tensor) -> torch.Tensor:
    """Eq. 17: squared Euclidean distance between two (..., 2) coordinate
    tensors (row, col)."""
    return ((coords_i - coords_j) ** 2).sum(dim=-1)


def local_mean_color(image: torch.Tensor, window_size: int) -> torch.Tensor:
    """Eq. 18: mu_c(x) — mean color within an window_size x window_size
    neighborhood of each pixel, per channel, via a box-filter (avg pool)."""
    pad = window_size // 2
    return F.avg_pool2d(image, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)


def color_term(mu_c_i: torch.Tensor, mu_c_j: torch.Tensor) -> torch.Tensor:
    """(mu_c(x_i) - mu_c(x_j))^2, summed over channels."""
    return ((mu_c_i - mu_c_j) ** 2).sum(dim=1 if mu_c_i.dim() == 4 else -1)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, C1=0.01 ** 2, C2=0.03 ** 2) -> torch.Tensor:
    """Standard windowed SSIM, returned as a per-pixel map (B,1,H,W)
    (averaged over channels), used as the structural-similarity potential
    Psi_p's SSIM(R1,R2) term."""
    device, dtype = img1.device, img1.dtype
    channels = img1.shape[1]
    g1d = _gaussian_window(window_size, sigma=1.5, device=device, dtype=dtype)
    window = (g1d[:, None] @ g1d[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean(dim=1, keepdim=True)  # (B,1,H,W)


def ssim_term(image: torch.Tensor, pred_mask: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM(R1,R2), where R1/R2 are local regions of the image and the
    (broadcast-to-image-channels) predicted mask, per Eq. 16's third term."""
    mask_rgb = pred_mask.expand(-1, image.shape[1], -1, -1) if pred_mask.shape[1] == 1 else pred_mask
    s = ssim_map(image, mask_rgb, window_size=window_size)
    return 1.0 - s
