"""Unit tests for models/ecrf/* (paper Sec 3.4, Eqs. 15-18)."""
import torch

from models.ecrf.energy_terms import unary_potential, spatial_term, local_mean_color, ssim_map, ssim_term
from models.ecrf.static_ecrf import StaticECRF, _extract_boundary, _expand_region
from models.ecrf.adaptive_ecrf import AdaptiveECRF


def test_unary_potential_is_negative_log_prior():
    u = unary_potential(0.9, (2, 1, 4, 4), device="cpu")
    expected = -torch.log(torch.tensor(0.9))
    assert torch.allclose(u, torch.full_like(u, expected.item()))


def test_spatial_term_zero_for_same_point():
    c = torch.tensor([[1.0, 2.0]])
    assert spatial_term(c, c).item() == 0.0


def test_local_mean_color_flat_image_is_identity():
    img = torch.full((1, 3, 16, 16), 0.42)
    mu = local_mean_color(img, window_size=11)
    assert torch.allclose(mu, img, atol=1e-5)


def test_ssim_map_identical_images_is_one():
    img = torch.rand(1, 3, 16, 16)
    s = ssim_map(img, img, window_size=7)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3)


def test_extract_boundary_flat_mask_has_no_edges():
    mask = torch.zeros(1, 1, 8, 8)
    edge = _extract_boundary(mask)
    assert edge.sum().item() == 0


def test_extract_boundary_and_expand_region_shapes():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1.0
    edge = _extract_boundary(mask)
    assert edge.sum().item() > 0
    region = _expand_region(edge)
    assert region.shape == mask.shape
    # expanded region must contain at least as many pixels as the edge itself
    assert region.sum().item() >= edge.sum().item()


def test_static_ecrf_refine_shape_and_range():
    image = torch.rand(1, 3, 32, 32)
    pred_mask = torch.rand(1, 1, 32, 32)
    ecrf = StaticECRF(window_size=11, num_iterations=2)
    refined = ecrf.refine(image, pred_mask)
    assert refined.shape == pred_mask.shape
    assert refined.min() >= 0.0 and refined.max() <= 1.0


def test_adaptive_ecrf_refine_shape_and_range():
    image = torch.rand(1, 3, 32, 32)
    pred_mask = torch.rand(1, 1, 32, 32)
    uncertainty = torch.rand(1, 1, 32, 32)
    ecrf = AdaptiveECRF(window_choices=[11, 21, 31], uncertainty_thresholds=[0.33, 0.66], num_iterations=1)
    refined = ecrf.refine(image, pred_mask, uncertainty)
    assert refined.shape == pred_mask.shape
    assert refined.min() >= 0.0 and refined.max() <= 1.0
