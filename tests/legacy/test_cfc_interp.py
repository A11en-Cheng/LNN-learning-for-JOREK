"""Unit tests for the Phase-3 CfC interpolator."""

from __future__ import annotations

import pytest
import torch

from src.models.cfc_interp import (
    CfCInterpolator,
    deterministic_fractions,
    sample_split_fractions,
)


def _model() -> CfCInterpolator:
    torch.manual_seed(0)
    return CfCInterpolator(latent_dim=8, hidden_size=8, dt_scale=50.0)


def test_step_shapes_and_finiteness():
    model = _model()
    state = torch.randn(16, 9)
    controls = torch.randn(16, 5)
    prediction, latent, hidden = model.step(state, None, 50.0, controls)
    assert prediction.shape == (16, 9)
    assert latent.shape == (16, 8)
    assert torch.isfinite(prediction).all()
    assert hidden is not None


def test_two_substep_is_differentiable_and_alpha_is_validated():
    model = _model()
    state = torch.randn(16, 9, requires_grad=True)
    out = model.two_substep(state, None, 0.3, 50.0, torch.randn(16, 5))
    assert out["prediction"].shape == (16, 9)
    assert out["mid_latent"].shape == (16, 8)
    out["prediction"].square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    with pytest.raises(ValueError):
        model.two_substep(state.detach(), None, 0.0, 50.0, None)


def test_hidden_state_carries_and_dt_scales():
    model = _model()
    state = torch.randn(16, 9)
    small, _, hidden = model.step(state, None, 5.0, None)
    large, _, _ = model.step(state, hidden, 500.0, None)
    assert not torch.allclose(small, large, atol=1e-6)


def test_split_fractions_are_positive_and_sum_to_one():
    torch.manual_seed(0)
    random_split = sample_split_fractions(8, 5, mode="random")
    assert random_split.shape == (8, 5)
    assert (random_split > 0).all()
    assert torch.allclose(random_split.sum(dim=1), torch.ones(8), atol=1e-5)
    equal_split = sample_split_fractions(3, 4, mode="equal")
    assert torch.allclose(equal_split, torch.full((3, 4), 0.25))
    single = sample_split_fractions(2, 1)
    assert torch.allclose(single, torch.ones(2, 1))


def test_deterministic_fractions_windows():
    assert deterministic_fractions(1) == [1.0]
    assert deterministic_fractions(2, 0.25) == [0.25, 0.75]
    assert deterministic_fractions(2) == [0.5, 0.5]
    four = deterministic_fractions(4)
    assert four == [0.25] * 4
    last_weighted = deterministic_fractions(4, 0.4)
    assert pytest.approx(sum(last_weighted), rel=1e-12) == 1.0
    assert last_weighted[-1] == pytest.approx(0.4)


def test_multi_substep_matches_single_step_when_trivial_and_validates():
    model = _model()
    state = torch.randn(16, 9)
    controls = torch.randn(16, 5)
    one = model.multi_substep(state, None, [1.0], 50.0, controls)["prediction"]
    reference, _, _ = model.step(state, None, 50.0, controls)
    assert torch.allclose(one, reference, atol=1e-6)
    out = model.multi_substep(state, None, [0.9, 0.05, 0.05], 50.0, controls)
    assert out["prediction"].shape == (16, 9)
    assert out["mid_latent"].shape == (16, 8)
    with pytest.raises(ValueError):
        model.multi_substep(state, None, [0.5, 0.4], 50.0, controls)
    with pytest.raises(ValueError):
        model.multi_substep(state, None, [0.0, 1.0], 50.0, controls)


def test_multi_substep_per_window_fractions_use_each_window_dt():
    """A per-window split must apply that window's own dt (ragged intervals)."""
    model = _model()
    state = torch.randn(2, 9)
    dt = torch.tensor([1.0, 16.0])
    batch_index = torch.tensor([0, 1])
    mixed = model.multi_substep(
        state, None, [[1.0], [1.0]], dt, None, batch_index
    )["prediction"]
    expected = torch.cat(
        [
            model.step(state[0:1], None, 1.0)[0],
            model.step(state[1:2], None, 16.0)[0],
        ]
    )
    assert torch.allclose(mixed, expected, atol=1e-6)
