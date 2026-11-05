"""
deterministic_uncertainty.py — cheap, default uncertainty estimate
(architecturenew.docx, item 4 / configs/model_adaptive.yaml ->
uncertainty.deterministic).

Uncertainty is estimated as the disagreement between (a) the boundary
implied by thresholding the predicted mask and (b) the model's own
predicted boundary map from BoundaryHead. Pixels where the two
disagree (within a small dilation tolerance) are flagged uncertain;
this directly targets the two failure modes the paper documents in
Sec 4.7 (hair artifacts, low-contrast fading boundaries).
"""
import torch
import torch.nn.functional as F

from data.boundary_labels import mask_to_boundary_morphological


class DeterministicUncertainty:
    def __init__(self, dilation_px: int = 2, mask_threshold: float = 0.5):
        self.dilation_px = dilation_px
        self.mask_threshold = mask_threshold

    @staticmethod
    def _dilate(x: torch.Tensor, px: int) -> torch.Tensor:
        if px <= 0:
            return x
        k = 2 * px + 1
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=px)

    def __call__(self, pred_mask: torch.Tensor, boundary_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mask: (B,1,H,W) in [0,1] (or {0,1}) — thresholded/soft mask
            boundary_logits: (B,1,H,W) raw logits from BoundaryHead
        Returns:
            uncertainty: (B,1,H,W) in [0,1], higher = more uncertain
        """
        hard_mask = (pred_mask > self.mask_threshold).float()
        mask_boundary = mask_to_boundary_morphological(hard_mask, erosion_iters=1)

        pred_boundary = (torch.sigmoid(boundary_logits) > 0.5).float()

        mask_boundary_dilated = self._dilate(mask_boundary, self.dilation_px)
        pred_boundary_dilated = self._dilate(pred_boundary, self.dilation_px)

        # xor_disagreement: boundary present in one dilated map but not
        # covered by the other -> disagreement
        disagreement = torch.clamp(
            (mask_boundary * (1 - pred_boundary_dilated))
            + (pred_boundary * (1 - mask_boundary_dilated)),
            0.0,
            1.0,
        )
        return disagreement
