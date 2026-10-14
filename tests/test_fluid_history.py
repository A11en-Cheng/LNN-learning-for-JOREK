"""History transport and scalar Gears component tests, not nine-field coupling."""
from copy import deepcopy
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from src.data.marker_checkpoint import initialize_native_checkpoint, reference_rollout_from_checkpoint
from src.models.marker_state import (
    FluidSolveResult,
    fluid_time_coefficients,
    load_model_rollout,
    reference_fluid_event,
    reference_particle_then_fluid,
    save_model_rollout,
)
from tests.test_marker_state import native_fixture


def checkpoint(tmp_path, delta=.25):
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = native_fixture(tmp_path)
    args['namelist_path'].write_text(
        "tstep_n=(/ .5, 1., .25 /)\nnstep_n=(/ 1, 1, 1 /)\n"
        "tstep_particles=1d-9\ntime_evol_scheme='Gears'\n"
    )
    with h5py.File(args['fluid_path'], 'a') as f:
        f['deltas'][...] = delta
    return args, initialize_native_checkpoint(**args)


def no_shard_update(shards, old_field, time, dt):
    return shards


def particle_component(marker, target, old, current, shards):
    # Only a prescribed-field/event component. Do not claim fluid-source closure.
    marker.clock.sim_time_si = target
    return marker, {}


def scalar_gears_solver(t_norm, calls):
    def solve(u, shards, feedback, dt, previous_dt, *, fluid_deltas, time_coefficients):
        calls.append((fluid_deltas.copy(), dt, previous_dt, time_coefficients))
        # Unit forcing per JOREK time in an independent scalar mass equation.
        delta = (dt/t_norm + time_coefficients.zeta * time_coefficients.delta_scale * fluid_deltas) / (1+time_coefficients.zeta)
        return FluidSolveResult(u+delta, delta)
    return solve


def independent_bdf2_increment(old_delta, h, k):
    # Unequal-step polynomial differentiation of a scalar y'(t)=1 equation:
    # (1+2r)*dy_new = h*(1+r) + r^2*dy_old, r=h/k.
    r = h/k
    return (h*(1+r) + r*r*old_delta)/(1+2*r)


def test_same_current_field_different_native_history_reaches_solver(tmp_path):
    _, a = checkpoint(tmp_path/'a', .25)
    _, b = checkpoint(tmp_path/'b', .75)
    np.testing.assert_array_equal(a.fluid_values, b.fluid_values)
    states = [reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm)) for cp in (a, b)]
    # Previous field endpoints are intentionally identical: do not reconstruct
    # or substitute history from them, even when the shapes happen to match.
    for s in states:
        s.fields_previous = s.fields_current.copy()
    calls = []
    solved = [reference_fluid_event(s, solve_fluid=scalar_gears_solver(a.t_norm, calls),
                                    update_shards=no_shard_update) for s in states]
    for idx, old_delta in enumerate((.25, .75)):
        np.testing.assert_array_equal(calls[idx][0], np.full(a.fluid_values.shape, old_delta))
        assert calls[idx][2] == calls[idx][1]  # JOREK first process step override
        assert calls[idx][3].zeta == .5
        expected = (1.+old_delta)/3.  # h=k=.5, scalar equation, hand computed
        np.testing.assert_allclose(solved[idx].fluid_deltas, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(solved[idx].fields_current, a.fluid_values+expected)
        # Program restart zeros ONLY marker projection history.
        assert not solved[idx].marker.feedback.deltas.any()
    assert not np.array_equal(solved[0].fields_current, solved[1].fields_current)
    np.testing.assert_array_equal(states[0].fluid_deltas, a.fluid_deltas)  # input not mutated


def test_variable_step_history_and_serialized_reference_resume(tmp_path):
    _, cp = checkpoint(tmp_path/'native')
    initial = reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm))
    calls = []
    solve = scalar_gears_solver(cp.t_norm, calls)
    first = reference_fluid_event(initial, solve_fluid=solve, update_shards=no_shard_update)
    module = torch.nn.Identity()
    saved = tmp_path/'reference.pt'
    save_model_rollout(saved, first, module)
    restored = load_model_rollout(saved, module)
    np.testing.assert_array_equal(restored.fluid_deltas, first.fluid_deltas)
    assert restored.fluid_dt_previous_si == first.fluid_dt_previous_si
    assert restored.time_evol_scheme == 'Gears' and restored.plan_cursor == 1
    kwargs = dict(particle_step=particle_component, solve_fluid=solve, update_shards=no_shard_update)
    direct, _ = reference_particle_then_fluid(first, **kwargs)
    resumed, _ = reference_particle_then_fluid(restored, **kwargs)
    np.testing.assert_array_equal(direct.fields_current, resumed.fields_current)
    np.testing.assert_array_equal(direct.fluid_deltas, resumed.fluid_deltas)
    assert direct.fluid_dt_previous_si == resumed.fluid_dt_previous_si == cp.t_norm
    expected = independent_bdf2_increment(5./12., 1., .5)
    assert expected == pytest.approx(14./15.)
    np.testing.assert_allclose(direct.fluid_deltas, expected, rtol=0, atol=1e-15)
    assert calls[1][3].zeta == pytest.approx(2./3.)
    assert calls[1][3].delta_scale == 2.
    np.testing.assert_array_equal(calls[1][0], first.fluid_deltas)  # raw, not pre-scaled
    third, _ = reference_particle_then_fluid(direct, **kwargs)
    expected_third = independent_bdf2_increment(expected, .25, 1.)
    np.testing.assert_allclose(third.fluid_deltas, expected_third, rtol=0, atol=1e-15)
    assert calls[-1][3].zeta == pytest.approx(.2)
    assert calls[-1][3].delta_scale == .25


def test_nodal_increment_is_not_inferred_from_rounded_field_difference(tmp_path):
    _, cp = checkpoint(tmp_path/'native')
    state = reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm))
    state.fields_current.fill(1e16)
    state.marker.feedback.deltas.fill(123.)
    marker_history = state.marker.feedback.deltas.copy()
    def solve(u, shards, feedback, dt, previous_dt, *, fluid_deltas, time_coefficients):
        # IEEE float64 addition rounds the +1 away, but node%deltas remains 1.
        return FluidSolveResult(u+1., np.ones_like(u))
    def shard_update(shards, u, time, dt):
        np.testing.assert_array_equal(u, state.fields_current)
        return shards
    result = reference_fluid_event(state, solve_fluid=solve, update_shards=shard_update)
    np.testing.assert_array_equal(result.fields_current, result.fields_previous)
    np.testing.assert_array_equal(result.fluid_deltas, np.ones_like(result.fields_current))
    np.testing.assert_array_equal(result.marker.feedback.deltas, marker_history)


@pytest.mark.parametrize('bad', ['missing', 'shape', 'nonfinite'])
def test_native_history_missing_or_invalid_fails(tmp_path, bad):
    args, cp = checkpoint(tmp_path/'native')
    with h5py.File(args['fluid_path'], 'a') as f:
        del f['deltas']
        if bad == 'shape':
            f['deltas'] = np.zeros(9)
        elif bad == 'nonfinite':
            f['deltas'] = np.full(cp.fluid_values.shape, np.nan)
    # Even a saved previous endpoint cannot undo rounded addition or recover
    # constrained/frozen nodal increments. Do not silently subtract endpoints.
    previous = tmp_path/'previous.h5'
    with h5py.File(previous, 'w') as f:
        f['t_now'] = 1.5; f['t_norm'] = cp.t_norm; f['values'] = cp.fluid_values-1
    with pytest.raises(ValueError, match='fluid node deltas'):
        initialize_native_checkpoint(**args, previous_fluid_path=previous)


def test_missing_history_or_previous_step_never_defaults_to_zero(tmp_path):
    _, cp = checkpoint(tmp_path/'native')
    ready = reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm))
    ready.marker.clock.startup_pending = False
    ready.fluid_dt_previous_si = .5*cp.t_norm
    invalid_history = replace(ready, fluid_deltas=None)
    def forbidden(*args, **kwargs):
        pytest.fail('solver called without valid history')
    with pytest.raises(ValueError, match='missing fluid node deltas'):
        reference_fluid_event(invalid_history, solve_fluid=forbidden, update_shards=no_shard_update)
    with pytest.raises(ValueError, match='previous fluid timestep'):
        reference_fluid_event(replace(ready, fluid_dt_previous_si=None), solve_fluid=forbidden, update_shards=no_shard_update)
    # Old model checkpoint with omitted history also fails on restore.
    path = tmp_path/'old.pt'
    module = torch.nn.Identity()
    save_model_rollout(path, ready, module)
    record = torch.load(path, weights_only=False)
    record['state'].fluid_deltas = None
    torch.save(record, path)
    with pytest.raises(ValueError, match='missing fluid node deltas'):
        load_model_rollout(path, module)


def test_continuous_checkpoint_uses_saved_positive_tstep_not_duplicate_output_times(tmp_path):
    _, cp = checkpoint(tmp_path/'native')
    cp.marker.clock.startup_pending = False
    cp.marker.clock.field_start_si = cp.marker.clock.field_end_si  # duplicate output
    state = reference_rollout_from_checkpoint(cp, continuous_remaining_plan_si=(cp.t_norm,))
    calls = []
    reference_fluid_event(state, solve_fluid=scalar_gears_solver(cp.t_norm, calls), update_shards=no_shard_update)
    assert calls[0][2] == cp.provenance['tstep_h5']*cp.t_norm
    assert calls[0][3].delta_scale == 2.


@pytest.mark.parametrize('scheme,theta,zeta', [('Gears', 1., 2./3.), ('implicit Euler', 1., 0.), ('Crank-Nicholson', .5, 0.)])
def test_time_coefficients_match_jorek(scheme, theta, zeta):
    got = fluid_time_coefficients(scheme, 2., 1.)
    assert got.theta == theta and got.zeta == pytest.approx(zeta)
    assert got.delta_scale == 2.
