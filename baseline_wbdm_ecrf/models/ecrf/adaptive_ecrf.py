"""
adaptive_ecrf.py — YOUR modified ECRF (architecturenew.docx, item 5):
the boundary-expansion window size is modulated per-pixel-region by the
uncertainty map from models/bdm/{deterministic,stochastic}_uncertainty.py,
instead of the paper's fixed 11x11 window.

window_choices + uncertainty_thresholds (configs/model_adaptive.yaml ->
ecrf): low uncertainty -> smallest window (cheap, confident regions),
high uncertainty -> largest window (more context where the model is
unsure) — directly targeting the two failure modes from the paper's
Sec 4.7 (hair artifacts, low-contrast boundaries).
"""
import torch
import torch.nn.functional as F

from .static_ecrf import StaticECRF, _extract_boundary, _expand_region
from .energy_terms import local_mean_color, ssim_term


class AdaptiveECRF:
    def __init__(
        self,
        window_choices=(11, 21, 31),
        uncertainty_thresholds=(0.33, 0.66),
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
        assert len(window_choices) == len(uncertainty_thresholds) + 1
        self.window_choices = list(window_choices)
        self.uncertainty_thresholds = list(uncertainty_thresholds)
        self.sigma_color = sigma_color
        self.sigma_ssim = sigma_ssim
        self.ssim_window = ssim_window
        self.prior_prob = prior_prob
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.dilation = dilation
        self.gaussian_smoothing = gaussian_smoothing

        # one StaticECRF-style refiner per window size choice, sharing
        # hyperparameters but differing in `window_size`
        self._refiners = {
            w: StaticECRF(
                window_size=w,
                sigma_spatial=sigma_spatial,
                sigma_color=sigma_color,
                sigma_ssim=sigma_ssim,
                ssim_window=ssim_window,
                prior_prob=prior_prob,
                num_iterations=num_iterations,
                step_size=step_size,
                dilation=False,       # dilation/smoothing applied once at the end
                gaussian_smoothing=False,
            )
            for w in self.window_choices
        }

    def _window_bucket(self, uncertainty_region_mean: float) -> int:
        for i, thr in enumerate(self.uncertainty_thresholds):
            if uncertainty_region_mean < thr:
                return self.window_choices[i]
        return self.window_choices[-1]

    @torch.no_grad()
    def refine(self, image: torch.Tensor, pred_mask: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B,3,H,W) in [0,1]
            pred_mask: (B,1,H,W) predicted mask probabilities
            uncertainty: (B,1,H,W) in [0,1] from DeterministicUncertainty
                or StochasticUncertainty
        Returns:
            refined_mask: (B,1,H,W)
        """
        B = pred_mask.shape[0]
        hard = (pred_mask > 0.5).float()
        edge = _extract_boundary(hard)
        base_region = _expand_region(edge)

        refined = pred_mask.clone()

        # bucket uncertainty into discrete levels and run the matching
        # window-size refiner only where that bucket applies, restricted
        # to the (edge-)expanded region — keeps cost proportional to
        # boundary length + uncertainty, not the whole image.
        for w in self.window_choices:
            bucket_mask = self._bucket_membership_mask(uncertainty, w)
            region = base_region * bucket_mask
            if region.sum() == 0:
                continue
            candidate = self._refiners[w].refine(image, pred_mask)
            refined = torch.where(region.bool(), candidate, refined)

        if self.dilation:
            refined = F.max_pool2d(refined, kernel_size=3, stride=1, padding=1)
        if self.gaussian_smoothing:
            refined = StaticECRF._gaussian_blur(refined)

        return refined.clamp(0.0, 1.0)

    def _bucket_membership_mask(self, uncertainty: torch.Tensor, window: int) -> torch.Tensor:
        idx = self.window_choices.index(window)
        lo = 0.0 if idx == 0 else self.uncertainty_thresholds[idx - 1]
        hi = 1.0 if idx == len(self.window_choices) - 1 else self.uncertainty_thresholds[idx]
        if idx == len(self.window_choices) - 1:
            return ((uncertainty >= lo) & (uncertainty <= hi)).float()
        return ((uncertainty >= lo) & (uncertainty < hi)).float()
