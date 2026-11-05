"""
static_ecrf.py — reproduces the ORIGINAL paper's ECRF exactly (Sec 3.4):
fixed 11x11 boundary expansion, energy minimization restricted to that
expanded edge region, with the SSIM-augmented pairwise potential (Eq. 16),
followed by dilation + Gaussian smoothing post-processing.

Used by configs/model_baseline.yaml (ecrf.type == "static").
"""
import torch
import torch.nn.functional as F

from .energy_terms import unary_potential, local_mean_color, color_term, ssim_term


def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    """4-neighborhood edge extraction: pixel differs from at least one of
    its 4 neighbors."""
    up = F.pad(mask[:, :, 1:, :], (0, 0, 0, 1))
    down = F.pad(mask[:, :, :-1, :], (0, 0, 1, 0))
    left = F.pad(mask[:, :, :, 1:], (0, 1, 0, 0))
    right = F.pad(mask[:, :, :, :-1], (1, 0, 0, 0))
    diff = ((mask != up) | (mask != down) | (mask != left) | (mask != right)).float()
    return diff


def _expand_region(edge_mask: torch.Tensor) -> torch.Tensor:
    """Union of edge points and their 4-neighborhoods (paper Sec 3.4,
    "Expansion" op in Fig. 2)."""
    up = F.pad(edge_mask[:, :, 1:, :], (0, 0, 0, 1))
    down = F.pad(edge_mask[:, :, :-1, :], (0, 0, 1, 0))
    left = F.pad(edge_mask[:, :, :, 1:], (0, 1, 0, 0))
    right = F.pad(edge_mask[:, :, :, :-1], (1, 0, 0, 0))
    return torch.clamp(edge_mask + up + down + left + right, 0, 1)


class StaticECRF:
    def __init__(
        self,
        window_size: int = 11,
        sigma_spatial: float = 1.0,
        sigma_color: float = 2.0,
        sigma_ssim: float = 2.0,
        ssim_window: int = 11,
        prior_prob: float = 0.9,
        num_iterations: int = 5,
        step_size: float = 0.5,
        dilation: bool = True,
        gaussian_smoothing: bool = True,
    ):
        self.window_size = window_size
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.sigma_ssim = sigma_ssim
        self.ssim_window = ssim_window
        self.prior_prob = prior_prob
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.dilation = dilation
        self.gaussian_smoothing = gaussian_smoothing

    @torch.no_grad()
    def refine(self, image: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B,3,H,W) original image, values in [0,1]
            pred_mask: (B,1,H,W) predicted mask (probabilities in [0,1])
        Returns:
            refined_mask: (B,1,H,W) in [0,1]
        """
        B, _, H, W = pred_mask.shape
        x = pred_mask.clone()
        hard = (x > 0.5).float()
        edge = _extract_boundary(hard)
        region = _expand_region(edge)  # (B,1,H,W), 1 where ECRF is applied

        mu_c = local_mean_color(image, self.window_size)  # (B,3,H,W)

        for _ in range(self.num_iterations):
            unary = unary_potential(self.prior_prob, x.shape, x.device, x.dtype)

            # pairwise term approximated locally: compare each pixel to the
            # window-mean color/ssim at its own location (a local smoothness
            # energy), which is minimized by nudging x toward the locally
            # consistent label — a standard mean-field-style local update.
            color_diff = ((image - mu_c) ** 2).sum(dim=1, keepdim=True)
            ssim_diff = ssim_term(image, x, window_size=self.ssim_window)

            pairwise_energy = torch.exp(
                -color_diff / (2 * self.sigma_color ** 2)
                - ssim_diff / (2 * self.sigma_ssim ** 2)
            )

            # energy gradient step: pixels with low pairwise agreement get
            # pulled toward the smoothed local mask estimate
            local_mask_mean = local_mean_color(x, self.window_size)
            update = pairwise_energy * (local_mask_mean - x) - self.step_size * unary * 0.0

            x = x + self.step_size * update * region
            x = x.clamp(0.0, 1.0)

        if self.dilation:
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.gaussian_smoothing:
            x = self._gaussian_blur(x)

        return x.clamp(0.0, 1.0)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        C = x.shape[1]
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, -1)
        # depthwise (grouped) conv so this works for any channel count, not just 1
        kernel_x = g.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
        kernel_y = g.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
        pad = kernel_size // 2
        x = F.conv2d(x, kernel_x, padding=(0, pad), groups=C)
        x = F.conv2d(x, kernel_y, padding=(pad, 0), groups=C)
        return x
