"""
stochastic_uncertainty.py — optional, costlier uncertainty estimate
(configs/model_adaptive.yaml -> uncertainty.stochastic).

Since BDM sampling converges in very few steps (paper Sec 3.3.2 / Table 9
shows T_sample=3 already near-optimal), we can afford to draw several
independent samples and use the pixelwise variance across samples as an
uncertainty map — the same idea as Zbinden et al. / Amit et al.'s
multi-annotator diffusion consensus, but far cheaper here because each
sample only needs `sampling_steps` (default: reuse T_sample) diffusion
steps rather than a full T_train trajectory.
"""
import torch


class StochasticUncertainty:
    def __init__(self, num_samples: int = 5, sampling_steps: int = 3):
        self.num_samples = num_samples
        self.sampling_steps = sampling_steps

    @torch.no_grad()
    def __call__(self, bdm, xT_image: torch.Tensor):
        """
        Args:
            bdm: a BrownianBridgeDiffusion instance (models/bdm/bridge_diffusion.py)
            xT_image: (B,C,H,W) conditioning image (the DWT-fused input)
        Returns:
            mean_mask: (B,C,H,W) average of the sampled masks
            uncertainty: (B,1,H,W) per-pixel variance, averaged over channels
        """
        samples = []
        for _ in range(self.num_samples):
            x0_hat, _ = bdm.sample(xT_image, num_steps=self.sampling_steps, sigma=1.0)
            samples.append(x0_hat)

        stacked = torch.stack(samples, dim=0)  # (S,B,C,H,W)
        mean_mask = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)  # (B,C,H,W)
        uncertainty = variance.mean(dim=1, keepdim=True)

        # normalize to [0,1] per-sample for downstream thresholding in ECRF
        u_min = uncertainty.amin(dim=(2, 3), keepdim=True)
        u_max = uncertainty.amax(dim=(2, 3), keepdim=True)
        uncertainty = (uncertainty - u_min) / (u_max - u_min + 1e-8)

        return mean_mask, uncertainty
