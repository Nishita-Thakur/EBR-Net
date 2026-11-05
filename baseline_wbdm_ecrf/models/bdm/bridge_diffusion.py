"""
bridge_diffusion.py — the Brownian Bridge Diffusion Model (BDM) core
(paper Sec 3.3, Eqs. 6-14). Learns a direct stochastic transformation
from the source image domain (x_T) to the target mask domain (x_0),
instead of noise -> mask.

Forward process (Eq. 7):
    q(x_t | x_0, x_T) = N((1-m_t) x_0 + m_t x_T, delta_t I)
    m_t = t/T,  delta_t = 2 s (m_t - m_t^2)                     [Eq. 7]

Training objective (Eq. 9, simplified ELBO): predict the noise
epsilon_theta(x_t, t) that reconstructs m_t (x_T - x_0) + sqrt(delta_t) eps.

Sampling (Sec 3.3.2, Eqs. 10-14): DDIM-style accelerated sampling over a
subsequence of T_sample steps (default 3), re-estimating x_0_hat at each
step and using it (with the fixed anchor x_T) to compute x_{t-1}.
"""
import math
import torch
import torch.nn as nn

from .unet_backbone import UNetBackbone
from .boundary_head import BoundaryHead


class BrownianBridgeDiffusion(nn.Module):
    def __init__(
        self,
        unet_kwargs: dict,
        T_train: int = 1000,
        T_sample: int = 3,
        s_variance_scale: float = 1.0,
        boundary_head_enabled: bool = False,
        boundary_head_out_channels: int = 1,
    ):
        super().__init__()
        self.unet = UNetBackbone(**unet_kwargs)
        self.T_train = T_train
        self.T_sample = T_sample
        self.s = s_variance_scale

        self.boundary_head_enabled = boundary_head_enabled
        if boundary_head_enabled:
            self.boundary_head = BoundaryHead(
                self.unet.final_feature_channels, out_channels=boundary_head_out_channels
            )
        else:
            self.boundary_head = None

    # ---------- forward-process quantities (Eq. 7) ----------
    def m_t(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() / self.T_train

    def delta_t(self, t: torch.Tensor) -> torch.Tensor:
        m = self.m_t(t)
        return 2.0 * self.s * (m - m ** 2)

    def q_sample(self, x0: torch.Tensor, xT: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Sample x_t ~ q(x_t | x_0, x_T) via Eq. 7 / the reparameterized
        form x_t = x_0 + m_t (x_T - x_0) + sqrt(delta_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        m = self.m_t(t).view(-1, 1, 1, 1)
        delta = self.delta_t(t).clamp(min=1e-12).view(-1, 1, 1, 1)
        x_t = x0 + m * (xT - x0) + delta.sqrt() * noise
        return x_t, noise

    # ---------- training loss (Eq. 9) ----------
    def training_losses(self, x0_mask: torch.Tensor, xT_image: torch.Tensor, boundary_gt: torch.Tensor = None):
        B = x0_mask.shape[0]
        t = torch.randint(1, self.T_train + 1, (B,), device=x0_mask.device)
        x_t, noise = self.q_sample(x0_mask, xT_image, t)

        pred_noise, decoder_features = self.unet(x_t, t)

        # target: m_t (x_T - x_0) + sqrt(delta_t) * eps   (matches Eq. 9's argument)
        m = self.m_t(t).view(-1, 1, 1, 1)
        delta = self.delta_t(t).clamp(min=1e-12).view(-1, 1, 1, 1)
        target = m * (xT_image - x0_mask) + delta.sqrt() * noise

        diffusion_loss = torch.nn.functional.mse_loss(pred_noise, target)

        losses = {"diffusion_loss": diffusion_loss}
        boundary_logits = None
        if self.boundary_head_enabled and boundary_gt is not None:
            boundary_logits = self.boundary_head(decoder_features.detach())
            # resized to match GT resolution if needed
            if boundary_logits.shape[-2:] != boundary_gt.shape[-2:]:
                boundary_logits = torch.nn.functional.interpolate(
                    boundary_logits, size=boundary_gt.shape[-2:], mode="bilinear", align_corners=False
                )
            losses["boundary_logits"] = boundary_logits

        return losses, x_t

    # ---------- sampling (Sec 3.3.2, Eqs. 10-14) ----------
    @torch.no_grad()
    def sample(self, xT_image: torch.Tensor, num_steps: int = None, sigma: float = 0.0):
        """Accelerated DDIM-style sampling: at each step, re-estimate x0_hat
        from the model's noise prediction and the fixed anchor xT, then
        step to x_{t-1} via Eq. 14."""
        num_steps = num_steps or self.T_sample
        device = xT_image.device
        B = xT_image.shape[0]

        # subsequence of timesteps {T, ..., 1}, evenly spaced, length num_steps
        seq = torch.linspace(self.T_train, 1, num_steps, device=device).round().long()
        seq = torch.unique_consecutive(seq)

        x_t = xT_image.clone()  # anchored at t=T: x_T = image
        boundary_logits = None
        decoder_features = None

        for i, t_val in enumerate(seq):
            t = torch.full((B,), int(t_val.item()), device=device, dtype=torch.long)
            pred_noise, decoder_features = self.unet(x_t, t)

            m_t = self.m_t(t).view(-1, 1, 1, 1)
            delta_t = self.delta_t(t).clamp(min=1e-12).view(-1, 1, 1, 1)

            # Eq. 13: recover eps, then x0_hat
            # pred_noise approximates m_t (xT - x0) + sqrt(delta_t) eps directly
            # (network target from training_losses), so we invert algebraically:
            x0_hat = x_t - pred_noise
            x0_hat = x0_hat.clamp(-1.0, 1.0)

            if i == len(seq) - 1:
                x_t = x0_hat
                break

            t_prev = torch.full((B,), int(seq[i + 1].item()), device=device, dtype=torch.long)
            m_prev = self.m_t(t_prev).view(-1, 1, 1, 1)
            delta_prev = self.delta_t(t_prev).clamp(min=1e-12).view(-1, 1, 1, 1)

            noise = torch.randn_like(x_t) if sigma > 0 else torch.zeros_like(x_t)
            mean = (1 - m_prev) * x0_hat + m_prev * xT_image
            x_t = mean + (sigma * delta_prev.sqrt()) * noise

        if self.boundary_head_enabled and decoder_features is not None:
            boundary_logits = self.boundary_head(decoder_features)

        return x_t, boundary_logits
