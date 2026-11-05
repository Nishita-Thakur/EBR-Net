"""Unit tests for models/dwt/*."""
import torch

from models.dwt.static_dwt import StaticDWT
from models.dwt.adaptive_dwt import AdaptiveDWT
from models.dwt.frequency_gate import SEFrequencyGate


def test_static_dwt_output_shapes():
    x = torch.randn(2, 3, 32, 32)
    dwt = StaticDWT(channels=3)
    subbands = dwt(x)
    assert set(subbands.keys()) == {"LL", "LH", "HL", "HH"}
    for k, v in subbands.items():
        assert v.shape == (2, 3, 16, 16), f"{k} shape mismatch: {v.shape}"


def test_static_dwt_ll_matches_average_pool_on_flat_input():
    # for a spatially constant image, LL should equal the input value
    # (Haar LL filter is a normalized average).
    x = torch.ones(1, 3, 8, 8) * 0.7
    dwt = StaticDWT(channels=3)
    ll = dwt.extract_ll(x)
    assert torch.allclose(ll, torch.full_like(ll, 0.7), atol=1e-5)


def test_frequency_gate_output_shapes():
    subbands = {k: torch.randn(2, 3, 16, 16) for k in ["LL", "LH", "HL", "HH"]}
    gate = SEFrequencyGate(channels=3)
    fused, g = gate(subbands)
    assert fused.shape == (2, 3, 16, 16)
    assert g.shape == (2, 4, 3)
    assert torch.all((g >= 0) & (g <= 1))


def test_frequency_gate_initial_bias_favors_LL():
    subbands = {k: torch.randn(4, 3, 16, 16) for k in ["LL", "LH", "HL", "HH"]}
    gate = SEFrequencyGate(channels=3, init_bias_toward_LL=True)
    _, g = gate(subbands)
    mean_per_subband = g.mean(dim=(0, 2))  # (4,)
    assert mean_per_subband[0] > mean_per_subband[1]
    assert mean_per_subband[0] > mean_per_subband[2]
    assert mean_per_subband[0] > mean_per_subband[3]


def test_adaptive_dwt_forward_and_regularization():
    x = torch.randn(2, 3, 32, 32)
    dwt = AdaptiveDWT(channels=3)
    fused, aux = dwt(x)
    assert fused.shape == (2, 3, 16, 16)
    reg_active = dwt.regularization_loss(aux, epoch=0, warmup_epochs=20, weight=0.01)
    reg_inactive = dwt.regularization_loss(aux, epoch=25, warmup_epochs=20, weight=0.01)
    assert reg_active.item() >= 0.0
    assert reg_inactive.item() == 0.0
