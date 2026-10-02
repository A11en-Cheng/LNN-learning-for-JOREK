"""Guards for the LTC propagator: identity, semigroup, rate sign, accumulator."""

from __future__ import annotations

import torch

from src.models.cfc_interp import CfCInterpolator
from src.models.ltc_cell import LTCCell


def _cell(hidden=16, input_dim=8, **kwargs) -> LTCCell:
    torch.manual_seed(0)
    return LTCCell(input_dim, hidden, backbone_units=32, **kwargs).eval()


def _state(cell, hidden, input_dim, dtype=torch.float32):
    x = torch.randn(32, input_dim, dtype=dtype)
    h = torch.randn(32, hidden, dtype=dtype)
    return x, h


def test_zero_timespan_is_exactly_the_identity():
    cell = _cell()
    x, h = _state(cell, 16, 8)
    with torch.no_grad():
        out = cell(x, h, torch.tensor(0.0))
    assert torch.equal(out, h)


def test_semigroup_is_machine_precision_in_float64_and_float32():
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        cell = _cell().to(dtype)
        x, h = _state(cell, 16, 8, dtype)
        with torch.no_grad():
            one = cell(x, h, torch.tensor(1.0, dtype=dtype))
            mid = cell(x, h, torch.tensor(0.3, dtype=dtype))
            two = cell(x, mid, torch.tensor(0.7, dtype=dtype))
            update = float((one - h).norm())
        defect = float((one - two).norm())
        assert defect <= tol * max(update, 1.0), (dtype, defect, update)


def test_state_dependent_coefficients_are_not_exactly_a_semigroup():
    """Documents the trade-off: h-dependent rate/target re-evaluated at a split
    point only compose to O(ts^2); the default keeps them input-only."""
    cell = _cell(state_dependent=True)
    x, h = _state(cell, 16, 8)
    with torch.no_grad():
        one = cell(x, h, torch.tensor(1.0))
        mid = cell(x, h, torch.tensor(0.3))
        two = cell(x, mid, torch.tensor(0.7))
        update = float((one - h).norm())
    assert float((one - two).norm()) / max(update, 1e-12) > 1e-6


def test_rate_is_strictly_positive_for_extreme_inputs():
    cell = _cell()
    with torch.no_grad():
        cell.rate_net.bias.fill_(-40.0)
    x, h = _state(cell, 16, 8)
    rate, target = cell.coefficients(x, h)
    assert (rate > 0).all()
    assert (target.abs() <= 1).all()


def test_multi_substep_is_split_invariant_for_the_default_cell():
    """The whole point of the rewrite: the trained integrator must not depend on
    how the recorded interval happens to be chopped."""
    torch.manual_seed(0)
    model = CfCInterpolator(latent_dim=16, hidden_size=16, backbone_units=32, cell="ltc").eval()
    state = torch.randn(24, 9)
    controls = torch.randn(24, 5)
    with torch.no_grad():
        one = model.multi_substep(state, None, [1.0], 5.0, controls)["prediction"]
        two = model.multi_substep(state, None, [0.37, 0.63], 5.0, controls)["prediction"]
        many = model.multi_substep(state, None, [0.1] * 10, 5.0, controls)["prediction"]
    scale = float((one - state).norm())
    assert float((two - one).norm()) <= 1e-5 * max(scale, 1.0)
    assert float((many - one).norm()) <= 1e-5 * max(scale, 1.0)


def test_source_integral_is_exact_for_a_constant_source_and_additive():
    torch.manual_seed(0)
    model = CfCInterpolator(
        latent_dim=16, hidden_size=16, backbone_units=32, cell="ltc", predict_source=True
    ).eval()
    for parameter in model.source_head.parameters():
        parameter.data.zero_()
    model.source_head[-1].bias.data.fill_(0.2)
    state = torch.randn(24, 9)
    dt = 1.5
    with torch.no_grad():
        whole = model.advance_with_source_integral(state, None, dt, num_substeps=5)
        first = model.advance_with_source_integral(state, None, 0.3 * dt, num_substeps=3)
        second = model.advance_with_source_integral(
            state, first["hidden"], 0.7 * dt, num_substeps=3
        )
    expected = torch.full_like(whole["source_integral"], 0.2 * dt)
    assert torch.allclose(whole["source_integral"], expected)
    assert torch.allclose(
        first["source_integral"] + second["source_integral"],
        whole["source_integral"], atol=1e-6,
    )
    assert len(whole["source_rates"]) == 5


def test_source_integral_requires_the_head():
    model = CfCInterpolator(latent_dim=16, hidden_size=16, backbone_units=32, cell="ltc")
    try:
        model.advance_with_source_integral(torch.randn(8, 9), None, 1.0)
    except ValueError as error:
        assert "predict_source" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing source head must raise")


def test_input_dt_feature_enters_the_encoder_without_breaking_the_semigroup():
    """dt must be visible to the encoder (pitfalls #48) and must stay a
    per-interval constant so the propagator is still split invariant."""
    torch.manual_seed(0)
    model = CfCInterpolator(
        latent_dim=16, hidden_size=16, backbone_units=32, cell="ltc", input_dt=True
    ).eval()
    state = torch.randn(24, 9)
    controls = torch.randn(24, 5)
    assert model.encoder[0].in_features == 9 + 1
    with torch.no_grad():
        one = model.multi_substep(state, None, [1.0], 50.0, controls)["prediction"]
        split = model.multi_substep(state, None, [0.3, 0.7], 50.0, controls)["prediction"]
        short = model.multi_substep(state, None, [1.0], 5.0, controls)["prediction"]
    scale = float((one - state).norm())
    assert float((one - split).norm()) <= 1e-5 * max(scale, 1.0)
    # the feature must actually change the answer when dt changes
    assert float((one - short).norm()) > 1e-4 * max(scale, 1.0)
