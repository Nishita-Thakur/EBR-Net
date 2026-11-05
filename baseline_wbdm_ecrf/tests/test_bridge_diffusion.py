"""Unit tests for models/bdm/* (paper Sec 3.3, Eqs. 6-14)."""
import torch

from models.bdm.bridge_diffusion import BrownianBridgeDiffusion
from models.bdm.flash_attention import FlashSelfAttention2d
from losses.diffusion_loss import diffusion_mse_loss


def _tiny_unet_kwargs():
    return dict(
        in_channels=3, base_channels=8, channel_mults=(1, 2),
        use_flash_attention=True, attention_resolutions=(8,),
        input_resolution=16, num_heads=2,
    )


def test_flash_attention_preserves_shape():
    attn = FlashSelfAttention2d(channels=8, num_heads=2)
    x = torch.randn(2, 8, 8, 8)
    out = attn(x)
    assert out.shape == x.shape


def test_m_t_and_delta_t_endpoint_behavior():
    """Eq. 6 endpoint consistency: as t->0, m_t->0; as t->T, m_t->1;
    delta_t -> 0 at both endpoints."""
    bdm = BrownianBridgeDiffusion(_tiny_unet_kwargs(), T_train=100, T_sample=3)
    t0 = torch.tensor([1])
    tT = torch.tensor([100])
    assert bdm.m_t(t0).item() < 0.02
    assert abs(bdm.m_t(tT).item() - 1.0) < 1e-6
    assert bdm.delta_t(t0).item() >= 0.0
    assert bdm.delta_t(tT).item() >= 0.0


def test_q_sample_shape():
    bdm = BrownianBridgeDiffusion(_tiny_unet_kwargs(), T_train=100, T_sample=3)
    x0 = torch.rand(2, 3, 16, 16)
    xT = torch.rand(2, 3, 16, 16)
    t = torch.randint(1, 100, (2,))
    x_t, noise = bdm.q_sample(x0, xT, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_training_losses_runs_and_is_finite():
    bdm = BrownianBridgeDiffusion(
        _tiny_unet_kwargs(), T_train=100, T_sample=3,
        boundary_head_enabled=True, boundary_head_out_channels=1,
    )
    x0 = torch.rand(2, 3, 16, 16)
    xT = torch.rand(2, 3, 16, 16)
    boundary_gt = torch.randint(0, 2, (2, 1, 16, 16)).float()

    losses, x_t = bdm.training_losses(x0, xT, boundary_gt)
    assert torch.isfinite(losses["diffusion_loss"])
    assert "boundary_logits" in losses
    assert losses["boundary_logits"].shape == (2, 1, 16, 16)


def test_sample_produces_correct_shape():
    bdm = BrownianBridgeDiffusion(_tiny_unet_kwargs(), T_train=100, T_sample=3)
    xT = torch.rand(2, 3, 16, 16)
    x0_hat, _ = bdm.sample(xT, num_steps=3)
    assert x0_hat.shape == xT.shape
    assert torch.isfinite(x0_hat).all()


def test_diffusion_mse_loss_matches_functional_mse():
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    loss = diffusion_mse_loss(pred, target)
    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.isclose(loss, expected)
