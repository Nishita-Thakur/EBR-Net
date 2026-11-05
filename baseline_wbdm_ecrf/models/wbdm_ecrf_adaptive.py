"""
wbdm_ecrf_adaptive.py — top-level module wiring DWT -> BDM -> uncertainty
-> ECRF into a single pipeline, matching configs/model_adaptive.yaml.

This is the "adaptive" counterpart to the paper's fixed baseline
(configs/model_baseline.yaml): adaptive multi-frequency DWT with a
learned gate, a BDM with an auxiliary boundary head, a deterministic (or
stochastic) uncertainty map, and an uncertainty-modulated ECRF.

`build_model_from_config` also supports building the paper's exact
baseline when given model_baseline.yaml, by dispatching on
cfg["dwt"]["type"] / cfg["ecrf"]["type"] == "static".
"""
import torch
import torch.nn as nn

from models.dwt import StaticDWT, AdaptiveDWT
from models.ecrf import StaticECRF, AdaptiveECRF
from models.bdm import BrownianBridgeDiffusion, DeterministicUncertainty, StochasticUncertainty


class WBDMECRFAdaptive(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg.get("_data", {})
        channels = data_cfg.get("channels", 3)
        image_size = data_cfg.get("image_size", 256)

        # ---- DWT block ----
        dwt_cfg = cfg["dwt"]
        if dwt_cfg["type"] == "adaptive":
            gate_cfg = dwt_cfg.get("gate", {})
            self.dwt = AdaptiveDWT(
                channels=channels,
                wavelet=dwt_cfg.get("wavelet", "haar"),
                keep_subbands=dwt_cfg.get("keep_subbands", ["LL", "LH", "HL", "HH"]),
                reduction_ratio=gate_cfg.get("reduction_ratio", 8),
                init_bias_toward_LL=gate_cfg.get("init_bias_toward_LL", True),
                init_prior=gate_cfg.get("init_prior", [0.9987, 0.0004, 0.0004, 0.0005]),
            )
            self._dwt_reg_cfg = gate_cfg.get("regularization", {})
        elif dwt_cfg["type"] == "static":
            self.dwt = StaticDWT(channels=channels, wavelet=dwt_cfg.get("wavelet", "haar"))
            self._dwt_reg_cfg = None
        else:
            raise ValueError(f"Unknown dwt.type: {dwt_cfg['type']}")
        self.dwt_type = dwt_cfg["type"]

        # ---- BDM block ----
        bdm_cfg = cfg["bdm"]
        unet_cfg = dict(bdm_cfg["unet"])
        unet_cfg["input_resolution"] = image_size // 2  # DWT halves H,W
        boundary_cfg = bdm_cfg.get("boundary_head", {"enabled": False})
        diffusion_cfg = cfg.get("_diffusion", {})
        self.bdm = BrownianBridgeDiffusion(
            unet_kwargs=unet_cfg,
            T_train=diffusion_cfg.get("T_train", 1000),
            T_sample=diffusion_cfg.get("T_sample", 3),
            s_variance_scale=diffusion_cfg.get("s_variance_scale", 1.0),
            boundary_head_enabled=boundary_cfg.get("enabled", False),
            boundary_head_out_channels=boundary_cfg.get("out_channels", 1),
        )

        # ---- Uncertainty block ----
        unc_cfg = cfg.get("uncertainty", {"enabled": False})
        self.uncertainty_enabled = unc_cfg.get("enabled", False)
        self.uncertainty_type = unc_cfg.get("type", "deterministic")
        if self.uncertainty_enabled:
            if self.uncertainty_type == "deterministic":
                det_cfg = unc_cfg.get("deterministic", {})
                self.uncertainty_estimator = DeterministicUncertainty(
                    dilation_px=det_cfg.get("dilation_px", 2)
                )
            elif self.uncertainty_type == "stochastic":
                sto_cfg = unc_cfg.get("stochastic", {})
                self.uncertainty_estimator = StochasticUncertainty(
                    num_samples=sto_cfg.get("num_samples", 5),
                    sampling_steps=sto_cfg.get("sampling_steps", diffusion_cfg.get("T_sample", 3)),
                )
            else:
                raise ValueError(f"Unknown uncertainty.type: {self.uncertainty_type}")
        else:
            self.uncertainty_estimator = None

        # ---- ECRF block ----
        ecrf_cfg = cfg["ecrf"]
        pp = ecrf_cfg.get("postprocess", {})
        if ecrf_cfg["type"] == "adaptive":
            self.ecrf = AdaptiveECRF(
                window_choices=ecrf_cfg.get("window_choices", [11, 21, 31]),
                uncertainty_thresholds=ecrf_cfg.get("uncertainty_thresholds", [0.33, 0.66]),
                sigma_spatial=ecrf_cfg.get("sigma_spatial", 1.0),
                sigma_color=ecrf_cfg.get("sigma_color", 2.0),
                sigma_ssim=ecrf_cfg.get("sigma_ssim", 2.0),
                ssim_window=ecrf_cfg.get("ssim_window", 11),
                prior_prob=ecrf_cfg.get("prior_prob", 0.9),
                dilation=pp.get("dilation", True),
                gaussian_smoothing=pp.get("gaussian_smoothing", True),
            )
        elif ecrf_cfg["type"] == "static":
            self.ecrf = StaticECRF(
                window_size=ecrf_cfg.get("window_size", 11),
                sigma_spatial=ecrf_cfg.get("sigma_spatial", 1.0),
                sigma_color=ecrf_cfg.get("sigma_color", 2.0),
                sigma_ssim=ecrf_cfg.get("sigma_ssim", 2.0),
                ssim_window=ecrf_cfg.get("ssim_window", 11),
                prior_prob=ecrf_cfg.get("prior_prob", 0.9),
                dilation=pp.get("dilation", True),
                gaussian_smoothing=pp.get("gaussian_smoothing", True),
            )
        else:
            raise ValueError(f"Unknown ecrf.type: {ecrf_cfg['type']}")
        self.ecrf_type = ecrf_cfg["type"]

    # ---------------- training ----------------
    def compute_dwt(self, image: torch.Tensor):
        if self.dwt_type == "adaptive":
            fused, aux = self.dwt(image)
            return fused, aux
        else:
            return self.dwt.extract_ll(image), {}

    def training_step(self, image: torch.Tensor, mask: torch.Tensor, boundary_gt: torch.Tensor = None, epoch: int = 0):
        xT, dwt_aux = self.compute_dwt(image)
        # mask/boundary are at full resolution; downsample to match DWT output
        target_hw = xT.shape[-2:]
        x0 = torch.nn.functional.interpolate(mask, size=target_hw, mode="nearest")
        # q_sample mixes x0 and xT elementwise, so x0 must carry the same
        # channel count as the DWT-domain xT (mask is otherwise silently
        # broadcast, which produces a multi-channel x0_hat at sample time).
        if x0.shape[1] != xT.shape[1]:
            x0 = x0.repeat(1, xT.shape[1] // x0.shape[1], 1, 1)
        boundary_gt_ds = None
        if boundary_gt is not None:
            boundary_gt_ds = torch.nn.functional.interpolate(boundary_gt, size=target_hw, mode="nearest")

        losses, _ = self.bdm.training_losses(x0, xT, boundary_gt_ds)

        if self.dwt_type == "adaptive" and self._dwt_reg_cfg:
            reg = self.dwt.regularization_loss(
                dwt_aux,
                epoch=epoch,
                warmup_epochs=self._dwt_reg_cfg.get("warmup_epochs", 20),
                weight=self._dwt_reg_cfg.get("weight", 0.01),
            )
            losses["gate_regularization_loss"] = reg

        return losses

    # ---------------- inference ----------------
    @torch.no_grad()
    def predict(self, image: torch.Tensor, num_sample_steps: int = None):
        xT, _ = self.compute_dwt(image)
        pred_mask, boundary_logits = self.bdm.sample(xT, num_steps=num_sample_steps)
        # sample() returns x0_hat in the same (repeated) channel count as xT;
        # collapse back to the single-channel mask domain by averaging.
        if pred_mask.shape[1] > 1:
            pred_mask = pred_mask.mean(dim=1, keepdim=True)
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = pred_mask.clamp(0.0, 1.0)

        boundary_prob = None
        if boundary_logits is not None:
            boundary_prob = torch.sigmoid(boundary_logits)
            boundary_prob = torch.nn.functional.interpolate(boundary_prob, size=image.shape[-2:], mode="bilinear", align_corners=False)

        uncertainty = None
        if self.uncertainty_enabled:
            if self.uncertainty_type == "deterministic" and boundary_logits is not None:
                boundary_logits_full = torch.nn.functional.interpolate(
                    boundary_logits, size=image.shape[-2:], mode="bilinear", align_corners=False
                )
                uncertainty = self.uncertainty_estimator(pred_mask, boundary_logits_full)
            elif self.uncertainty_type == "stochastic":
                pred_mask, uncertainty = self.uncertainty_estimator(self.bdm, xT)
                pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
                uncertainty = torch.nn.functional.interpolate(uncertainty, size=image.shape[-2:], mode="bilinear", align_corners=False)

        if self.ecrf_type == "adaptive":
            if uncertainty is None:
                uncertainty = torch.zeros_like(pred_mask)
            refined = self.ecrf.refine(image, pred_mask, uncertainty)
        else:
            refined = self.ecrf.refine(image, pred_mask)

        return {
            "pred_mask": pred_mask,
            "refined_mask": refined,
            "boundary_prob": boundary_prob,
            "uncertainty": uncertainty,
        }


def build_model_from_config(full_cfg: dict) -> WBDMECRFAdaptive:
    """full_cfg is the merge of base.yaml and a model_*.yaml (see
    training/train.py for how they're merged), with base.yaml's `data`
    and `diffusion` sections stashed under `_data` / `_diffusion` keys."""
    cfg = dict(full_cfg["model"])
    cfg["_data"] = full_cfg.get("data", {})
    cfg["_diffusion"] = full_cfg.get("diffusion", {})
    return WBDMECRFAdaptive(cfg)
