"""Untrained pure-fluid head isolation, not new-data or physics qualification."""
from copy import deepcopy

import pytest
import torch

from tests.test_native_temporal import gp_temporal_fixture


def fluid_fixture():
    from src.models.native_features import NativeFeaturePath
    from src.models.native_pure_fluid import NativePureFluidField
    field, _, window = gp_temporal_fixture()
    statistics = field.feature_path.statistics()
    statistics['scenario_enabled'] = False
    field.feature_path = NativeFeaturePath(field.feature_path.evaluator, statistics)
    # A fresh isolated fixture, not an in-place conversion of a trained model.
    field.scenario_film = None
    return NativePureFluidField(field), window


def test_isolated_head_has_no_source_or_spi_parameters():
    model, window = fluid_fixture()
    from src.models.native_torch_rollout import rollout_field_window
    result = rollout_field_window(model, window, teacher_mask=(False, False))
    assert len(result.states) == 2
    assert not any('source' in n or 'scenario' in n for n, _ in model.named_parameters())
    assert model.checkpoint_configuration['task_scope'] == 'pure_fluid_candidate'
    assert model.checkpoint_configuration['source_policy'] == 'not_accepted_not_zero_imputed'
    assert model.checkpoint_configuration != model.field.checkpoint_configuration
    assert all(s.values.shape == (9, 4, 1, 4) for s in result.states)


@pytest.mark.parametrize('name', [
    'source_encoding', 'marker_feedback', 'projection_deltas', 'shard_source',
    'source_time', 'field_event',
])
def test_source_keywords_rejected_even_on_zero_time(name):
    model, window = fluid_fixture()
    from src.models.native_torch_rollout import initialize_field_state
    s = initialize_field_state(model, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    before = s.values.clone()
    with pytest.raises(TypeError, match='unexpected keyword'):
        model(s.values, s.raw_deltas, dt_si=0., previous_dt_si=s.previous_dt_si,
            scheme=s.scheme, memory=s.memory, **{name: None})
    assert torch.equal(s.values, before)


def test_coupled_scenario_and_incomplete_candidates_are_refused():
    from src.models.native_pure_fluid import NativePureFluidField
    from tests.test_native_joint_source import joint_fixture
    from tests.test_native_field_step import make_step
    field, _, _ = gp_temporal_fixture()
    with pytest.raises(ValueError, match='SPI scenario'):
        NativePureFluidField(field)
    with pytest.raises(ValueError, match='NativeFieldStep'):
        NativePureFluidField(joint_fixture()[0])
    with pytest.raises(ValueError, match='temporal|gradient'):
        NativePureFluidField(make_step())


def test_autoregressive_causality_bptt_and_exact_noop():
    model, window = fluid_fixture()
    from src.models.native_torch_rollout import initialize_field_state, advance_field, rollout_field_window
    s = initialize_field_state(model, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    s.values.requires_grad_(); s.raw_deltas.requires_grad_(); s.memory.requires_grad_()
    a = rollout_field_window(model, window, initial_state=s, teacher_mask=(False, False))
    assert a.inputs[1].values is a.states[0].values
    changed = deepcopy(window)
    for target in changed.targets:
        target.values.fill(999.); target.raw_deltas.fill(888.)
    b = rollout_field_window(model, changed, initial_state=s, teacher_mask=(False, False))
    assert torch.equal(a.states[-1].values, b.states[-1].values)
    a.states[-1].values.square().sum().backward()
    for tensor in (s.values, s.raw_deltas, s.memory):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.temporal_core.parameters())
    assert advance_field(model, s, dt_si=0.) is s


def test_new_model_resume_and_old_scope_rejection(tmp_path):
    from src.models.native_torch_rollout import initialize_field_state, advance_field
    from src.models.marker_state import save_model_rollout, load_model_rollout
    from src.models.native_features import select_feature_parameters
    model, window = fluid_fixture()
    params = select_feature_parameters(window.current.parameters,
        scenario_enabled=False, physics_enabled=True)
    s = initialize_field_state(model, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    a = advance_field(model, s, dt_si=window.time_plan_si[0], parameters=params)
    expected = advance_field(model, a, dt_si=window.time_plan_si[1], parameters=params)
    path = tmp_path / 'fluid.pt'
    save_model_rollout(path, a, model)
    fresh, _ = fluid_fixture()
    restored = load_model_rollout(path, fresh)
    actual = advance_field(fresh, restored, dt_si=window.time_plan_si[1], parameters=params)
    for name in ('values', 'raw_deltas', 'memory'):
        assert torch.equal(getattr(expected, name), getattr(actual, name))
    before = {n: t.clone() for n, t in model.field.state_dict().items()}
    with pytest.raises(ValueError, match='configuration'):
        load_model_rollout(path, model.field)
    assert all(torch.equal(t, model.field.state_dict()[n]) for n, t in before.items())
    old_state = initialize_field_state(model.field, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    old_path = tmp_path / 'old-field-only.pt'
    save_model_rollout(old_path, old_state, model.field)
    with pytest.raises(ValueError, match='configuration'):
        load_model_rollout(old_path, model)


def test_reenabled_spi_branch_fails_before_forward():
    model, window = fluid_fixture()
    from src.models.native_torch_rollout import initialize_field_state
    s = initialize_field_state(model, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    model.field.scenario_film = torch.nn.Linear(4, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match='SPI scenario'):
        model(s.values, s.raw_deltas, dt_si=window.time_plan_si[0],
            previous_dt_si=s.previous_dt_si, scheme=s.scheme, memory=s.memory)


@pytest.mark.parametrize('dt', [0., .001])
def test_spi_parameter_routes_are_rejected_before_time_update(dt):
    model, window = fluid_fixture()
    from src.models.native_features import select_feature_parameters
    from src.models.native_torch_rollout import initialize_field_state
    s = initialize_field_state(model, window.current.state,
        previous_dt_si=window.current.previous_dt_si, scheme=window.scheme)
    spi = select_feature_parameters(window.current.parameters,
        scenario_enabled=True, physics_enabled=True)
    with pytest.raises(ValueError, match='non-SPI parameters'):
        model(s.values, s.raw_deltas, dt_si=dt, previous_dt_si=s.previous_dt_si,
            scheme=s.scheme, memory=s.memory, feature_parameters=spi)
