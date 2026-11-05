"""
End-to-end shape tests: builds the full WBDMECRFAdaptive pipeline (and
the static baseline) from tiny configs and checks tensor shapes flow
correctly through DWT -> BDM -> uncertainty -> ECRF.
"""
import copy
import torch

from models import build_model_from_config


def _tiny_full_cfg(dwt_type="adaptive", ecrf_type="adaptive", uncertainty_enabled=True, boundary_enabled=True):
    return {
        "data": {"image_size": 32, "channels": 3},
        "diffusion": {"T_train": 50, "T_sample": 2, "s_variance_scale": 1.0},
        "model": {
            "experiment_name": "test",
            "dwt": {
                "type": dwt_type, "wavelet": "haar", "keep_subbands": ["LL", "LH", "HL", "HH"],
                "gate": {"reduction_ratio": 2, "init_bias_toward_LL": True,
                         "init_prior": [0.9987, 0.0004, 0.0004, 0.0005],
                         "regularization": {"weight": 0.01, "warmup_epochs": 5}},
            } if dwt_type == "adaptive" else {"type": "static", "wavelet": "haar", "keep_subbands": ["LL"]},
            "bdm": {
                "unet": {"in_channels": 3, "base_channels": 8, "channel_mults": [1, 2],
                         "use_flash_attention": True, "attention_resolutions": [8]},
                "boundary_head": {"enabled": boundary_enabled, "out_channels": 1, "loss": "boundary_bce_dice"},
                "bridge": {"type": "brownian_bridge"},
            },
            "uncertainty": {
                "enabled": uncertainty_enabled, "type": "deterministic",
                "deterministic": {"metric": "xor_disagreement", "dilation_px": 1},
            },
            "ecrf": {
                "type": ecrf_type,
                "window_choices": [3, 5, 7], "uncertainty_thresholds": [0.33, 0.66],
                "window_size": 3,
                "sigma_spatial": 1.0, "sigma_color": 2.0, "sigma_ssim": 2.0, "ssim_window": 5,
                "prior_prob": 0.9,
                "postprocess": {"dilation": True, "gaussian_smoothing": True},
            } if ecrf_type == "adaptive" else {
                "type": "static", "window_size": 3, "sigma_spatial": 1.0, "sigma_color": 2.0,
                "sigma_ssim": 2.0, "ssim_window": 5, "prior_prob": 0.9,
                "postprocess": {"dilation": True, "gaussian_smoothing": True},
            },
            "loss_weights": {"diffusion": 1.0, "boundary": 0.3, "gate_regularization": 0.01},
        },
    }


def test_adaptive_model_training_step():
    cfg = _tiny_full_cfg()
    model = build_model_from_config(cfg)
    image = torch.rand(2, 3, 32, 32)
    mask = torch.randint(0, 2, (2, 1, 32, 32)).float()
    boundary_gt = torch.randint(0, 2, (2, 1, 32, 32)).float()
    losses = model.training_step(image, mask, boundary_gt, epoch=0)
    assert "diffusion_loss" in losses
    assert torch.isfinite(losses["diffusion_loss"])


def test_adaptive_model_predict_shapes():
    cfg = _tiny_full_cfg()
    model = build_model_from_config(cfg)
    image = torch.rand(1, 3, 32, 32)
    out = model.predict(image)
    assert out["pred_mask"].shape == (1, 1, 32, 32)
    assert out["refined_mask"].shape == (1, 1, 32, 32)
    assert out["uncertainty"] is not None
    assert out["uncertainty"].shape == (1, 1, 32, 32)


def test_baseline_model_predict_shapes():
    cfg = _tiny_full_cfg(dwt_type="static", ecrf_type="static", uncertainty_enabled=False, boundary_enabled=False)
    model = build_model_from_config(cfg)
    image = torch.rand(1, 3, 32, 32)
    out = model.predict(image)
    assert out["pred_mask"].shape == (1, 1, 32, 32)
    assert out["refined_mask"].shape == (1, 1, 32, 32)
    assert out["boundary_prob"] is None
    assert out["uncertainty"] is None
