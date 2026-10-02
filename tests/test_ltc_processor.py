"""Regression guards for the time-consistent LTC processor (pitfall #43)."""

from __future__ import annotations

import torch

from src.models.cfc_interp import CfCInterpolator
from src.models.lnn_processor import JorekLNNProcessor, JorekLTCProcessor


def _relative(value: torch.Tensor, scale: float) -> float:
    return float(value.norm()) / max(scale, 1e-12)


def test_ltc_processor_is_identity_at_zero_timespan():
    torch.manual_seed(0)
    processor = JorekLTCProcessor(latent_dim=8, hidden_size=8)
    latent = torch.randn(16, 8)
    controls = torch.randn(16, 5)
    out_zero, state_zero = processor(latent, None, None, time_steps=0.0, controls=controls)
    assert torch.allclose(state_zero, torch.zeros_like(state_zero), atol=1e-7)
    # A zero-length integration must not move the state from its initial value.
    initial = torch.randn(16, 8)
    _, state_after = processor(latent, None, initial, time_steps=0.0, controls=controls)
    assert torch.allclose(state_after, initial, atol=1e-7)
    assert out_zero.shape == (16, 8)


def test_ltc_processor_is_an_approximate_semigroup():
    """Splitting dt must change the answer by much less than the update itself."""
    torch.manual_seed(0)
    processor = JorekLTCProcessor(latent_dim=16, hidden_size=16)
    latent = torch.randn(32, 16)
    controls = torch.randn(32, 5)
    dt = 0.1
    with torch.no_grad():
        full, _ = processor(latent, None, None, time_steps=dt, controls=controls)
        update = float((full - latent).norm())
        for alpha in (0.25, 0.5, 0.75):
            mid, mid_state = processor(latent, None, None, time_steps=alpha * dt, controls=controls)
            split, _ = processor(
                mid, None, mid_state, time_steps=(1.0 - alpha) * dt, controls=controls
            )
            assert _relative(split - full, update) < 0.05


def test_ncps_processor_is_not_a_propagator():
    """Documents *why* the default cell is a problem (pitfall #43): advancing by
    ts=0 already moves the state by a full update's worth."""
    torch.manual_seed(0)
    processor = JorekLNNProcessor(latent_dim=16, hidden_size=16)
    latent = torch.randn(32, 16)
    controls = torch.randn(32, 5)
    dt = 0.1
    with torch.no_grad():
        _, state_dt = processor(latent, None, None, time_steps=dt, controls=controls)
        _, state_zero = processor(latent, None, None, time_steps=0.0, controls=controls)
        update = float(state_dt[0].norm())
        assert update > 0.0
        assert _relative(state_zero[0], update) > 0.5


def test_interpolator_cell_flag_round_trips_state_dict():
    for cell in ("ncps", "ltc"):
        torch.manual_seed(0)
        model = CfCInterpolator(latent_dim=8, hidden_size=8, cell=cell)
        state_dict = model.state_dict()
        reloaded = CfCInterpolator(latent_dim=8, hidden_size=8, cell=cell)
        reloaded.load_state_dict(state_dict)
        assert reloaded.cell == cell
    try:
        CfCInterpolator(cell="gru")
    except ValueError as error:
        assert "cell must be" in str(error)
    else:  # pragma: no cover - the flag must reject unknown cells
        raise AssertionError("unknown cell must raise")
