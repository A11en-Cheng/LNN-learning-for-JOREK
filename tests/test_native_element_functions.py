"""Function-cache candidates cross the same sampling/source-event seam."""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from src.models.native_element_functions import (
    CompiledElementFunctions, NativeElementFunctionSource, build_kernel,
)
from src.models.native_source_time import NativeSourceTimeSampler
from tests.test_native_locator_events import event_fixture, assert_runtime_exact
from tests.test_native_marker_sampling import window


@pytest.fixture(scope='session')
def element_kernel(tmp_path_factory):
    return CompiledElementFunctions(build_kernel(tmp_path_factory.mktemp('element-functions')/'kernel.so'))


def assert_runtime_close(actual, expected):
    for name in actual.cloud.array_names():
        a, b = getattr(actual.cloud, name), getattr(expected.cloud, name)
        assert a.dtype == b.dtype
        if a.dtype.kind in 'iu' or name in ('weight', 't_birth'):
            np.testing.assert_array_equal(a, b)
        elif name == 'V_prev':
            # The fixture has a 1e8 radial speed and a mathematically zero Z
            # speed formed by cancellation. Judge vector roundoff, not a ratio
            # against its 1e-9 cancellation residue; report components separately.
            from tools.native_replay_metrics import relative_metrics
            assert relative_metrics(b, a)['relative_l2'] < 2e-12
        else:
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-13, err_msg=name)
    assert actual.cloud.group == expected.cloud.group
    for name in ('values', 'deltas'):
        np.testing.assert_allclose(getattr(actual.feedback, name), getattr(expected.feedback, name),
                                   rtol=2e-11, atol=2e-13)
    assert actual.clock == expected.clock


@pytest.mark.parametrize('layout', ['C', 'F', 'reversed'])
def test_random_native_cubic_values_gradients_and_signed_geometry(element_kernel, layout):
    source, _, fields = event_fixture(backend='batched_v1')
    rng = np.random.default_rng(4810)
    fields = fields + rng.normal(0, .01, fields.shape)
    feedback = rng.normal(0, .01, (5, 4, 1, source.mesh.n_nodes))
    if layout == 'F':
        fields, feedback = np.asfortranarray(fields), np.asfortranarray(feedback)
    elif layout == 'reversed':
        fields, feedback = (a[..., ::-1].copy()[..., ::-1] for a in (fields, feedback))
    # Preserve the represented geometry/fields while exercising signed sizes.
    sizes = source.mesh.size.copy()
    sizes[1::2] *= -1
    source.mesh.coordinates[:, 1::2] *= -1
    fields[:, 1::2] *= -1
    feedback[:, 1::2] *= -1
    source.mesh.size[:] = sizes
    fields.flags.writeable = feedback.flags.writeable = False
    cache = NativeElementFunctionSource(source, kernel=element_kernel).sampling_cache()
    e = np.arange(257) % len(source.mesh.gauss)
    st = rng.uniform(0, 1, (len(e), 2))
    expected = source.mesh._marker_points(e, st, fields, feedback)
    actual = cache._marker_points(e, st, fields, feedback)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)
    cache._marker_points(e, st, fields, feedback)
    assert cache.stats['field_builds'] == cache.stats['feedback_builds'] == 1
    assert cache.stats['sample_points'] == 514


@pytest.mark.parametrize('static', [True, False])
def test_complete_window_history_and_failure_retry(element_kernel, static):
    source, initial, fields, sampler = window(static=static)
    before = deepcopy(initial)
    candidate = NativeElementFunctionSource(source, kernel=element_kernel)
    expected, _ = source.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    actual, _ = candidate.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_close(actual, expected)
    assert_runtime_exact(initial, before)
    assert candidate.last_stats['field_builds'] == (1 if static else 4)
    atomic = source.atomic
    def fail(*args):
        raise RuntimeError('prescribed function-cache failure')
    source.atomic = fail
    with pytest.raises(RuntimeError, match='function-cache failure'):
        candidate.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_exact(initial, before)
    source.atomic = atomic
    retried, _ = candidate.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_close(retried, expected)


@pytest.mark.parametrize('scenario', ['normal', 'crossing', 'lost'])
def test_complete_locator_events_are_not_replaced(element_kernel, scenario):
    source, initial, fields = event_fixture(backend='batched_v1', scenario=scenario)
    expected, _ = source.advance(initial, 1e-9, fields, fields)
    actual, _ = NativeElementFunctionSource(source, kernel=element_kernel).advance(initial, 1e-9, fields, fields)
    assert_runtime_close(actual, expected)


def test_two_events_fresh_checkpoint_and_mutated_input(element_kernel):
    source, initial, fields, sampler = window(static=True)
    candidate = NativeElementFunctionSource(source, kernel=element_kernel)
    first, _ = candidate.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    first.clock.field_start_si, first.clock.field_end_si = 1e-9, 2e-9
    fields[4, 0] *= 1.02
    first.feedback.values[3, 0] = .03
    first.feedback.values[4, 0] = .01
    sampler = NativeSourceTimeSampler(fields, fields*.001, time_prev_si=1e-9,
        time_now_si=2e-9, tstep_si=1e-9, static=True)
    expected, _ = source.advance(first, 2e-9, fields, fields, time_sampler=sampler)
    continued, _ = candidate.advance(first, 2e-9, fields, fields, time_sampler=sampler)
    restored = pickle.loads(pickle.dumps(first))
    resumed, _ = NativeElementFunctionSource(source, kernel=element_kernel).advance(
        restored, 2e-9, fields, fields, time_sampler=sampler)
    assert_runtime_close(continued, expected)
    assert_runtime_exact(resumed, continued)


def test_budget_and_bad_query_fail_without_commit(element_kernel):
    source, initial, fields, sampler = window(static=True)
    before = deepcopy(initial)
    with pytest.raises(MemoryError, match='function cache budget'):
        NativeElementFunctionSource(source, kernel=element_kernel, max_cache_bytes=1).advance(
            initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_exact(initial, before)
    fields.flags.writeable = False
    feedback = initial.feedback.values.copy(); feedback.flags.writeable = False
    cache = NativeElementFunctionSource(source, kernel=element_kernel).sampling_cache()
    for e, st in ((np.array([-1]), np.array([[.2, .3]])),
                  (np.array([0]), np.array([[np.nan, .3]]))):
        with pytest.raises(ValueError, match='element/local'):
            cache._marker_points(e, st, fields, feedback)


def test_nearby_composition_keeps_saved_neighbours_and_source_unchanged(element_kernel):
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    from tests.test_native_nearby_locator import saved_chain_neighbours
    from tests.test_native_relocation_work import near_edge_event
    source, initial, fields = near_edge_event()
    nearby = NativeNearbyMarkerSource(source, neighbours=saved_chain_neighbours())
    candidate = NativeElementFunctionSource(nearby, kernel=element_kernel)
    expected, _ = nearby.advance(initial, 1e-9, fields, fields)
    actual, _ = candidate.advance(initial, 1e-9, fields, fields)
    assert_runtime_close(actual, expected)
    assert source.mesh is nearby._mesh
    assert '_marker_points' not in source.mesh.__dict__


def test_two_complete_torch_events_keep_cross_step_gradients(element_kernel):
    import torch
    from tests.test_base_coupled_training import event_fixture as coupled_fixture
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import TorchRecomputedSourceEvents
    model, source, shard, runtime = coupled_fixture(n=2)
    # The fixture defaults to scalar; select the existing native batched path.
    from src.models.native_coupling import NativeMarkerSource
    source = NativeMarkerSource(source.mesh, atomic=source.atomic,
        atomic_version=source.configuration['atomic'], potentials_ev=source.potentials_ev,
        F0=source.F0, diffusion_ju=source.diffusion_ju, spatial_backend='batched_v1')
    source = NativeElementFunctionSource(source, kernel=element_kernel)
    model.source_configuration = deepcopy(source.configuration)
    runtime.native_context.configuration['model'] = model.checkpoint_configuration
    adapter = TorchRecomputedSourceEvents(model, source, shard)
    initial = adapter.initialize(runtime)
    initial.values.requires_grad_()
    np.random.seed(419)
    first, _ = adapter.advance(initial, StepControl(1e-9, 1e-9, 2e-9, 'external_control', 'fixture'))
    second, _ = adapter.advance(first, StepControl(2e-9, 1e-9, 3e-9, 'external_control', 'fixture'))
    second.values.square().sum().backward()
    assert initial.values.grad is not None and initial.values.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal_core.parameters())
    assert adapter.gradient_scope == 'torch_field_bptt_stop_gradient_numpy_source_shard'


def test_independent_bezier_degree_value_and_compatible_physical_derivatives(element_kernel):
    source, initial, fields = event_fixture(backend='batched_v1')
    fields[:] = 0
    # One native derivative degree: B_left(s)*A_left(t), not Hermite's B/3.
    node = source.mesh.vertex[0, 0]
    fields[7, 1, 0, node] = 1/source.mesh.size[1, 0, 0]
    fields.flags.writeable = False
    feedback = np.zeros_like(initial.feedback.values); feedback.flags.writeable = False
    st = np.random.default_rng(651).uniform(0, 1, (101, 2))
    s, t = st.T
    value = 3*s*(1-s)**2*(1-t)**2*(1+2*t)
    fs = 3*(s-1)*(3*s-1)*(1-t)**2*(1+2*t)
    ft = 3*s*(1-s)**2*6*t*(t-1)
    rs, rt = np.ones_like(s), .1*(1-2*t)
    zs, zt = .12*t*(1-t), 1+.12*s*(1-2*t)
    jac = rs*zt-rt*zs
    cache = NativeElementFunctionSource(source, kernel=element_kernel).sampling_cache()
    values, gradients, _, rz = cache._marker_points(np.zeros(len(st), dtype=int), st, fields, feedback)
    np.testing.assert_allclose(values[:, 7], value, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(gradients[:, 7, 0], (zt*fs-zs*ft)/jac, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(gradients[:, 7, 1], (-rt*fs+rs*ft)/jac, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(rz[:, 0], 2+s+.1*t*(1-t), rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(rz[:, 1], t+.12*s*t*(1-t), rtol=2e-14, atol=2e-15)


def test_complete_fine_substeps_keep_charge_deposition_and_transport_history(element_kernel):
    from src.models.native_coupling import NativeMarkerSource
    from src.models.marker_transport import LocalSubcycling, initialize_transport_history
    from tests.test_native_gauss import native_coefficients
    old, initial, fields, _ = window(static=True)
    fields[7, :, 0] = native_coefficients(lambda s,t: .5+.1*s,
        lambda s,t: .1, lambda s,t: 0., lambda s,t: 0.)
    policy = LocalSubcycling(trigger_indicator=1e-12, min_level=2, max_level=4)
    initial = initialize_transport_history(initial, policy, previous_dt_si=initial.clock.particle_dt_si)
    source = NativeMarkerSource(old.mesh, atomic=old.atomic, atomic_version=old.configuration['atomic'],
        potentials_ev=old.potentials_ev, F0=old.F0, diffusion_ju=.001,
        spatial_backend='batched_v1', time_sampling='jorek_raw_deltas_v1', field_static=True,
        transport_policy=policy)
    sampler = NativeSourceTimeSampler(fields, np.zeros_like(fields), time_prev_si=0.,
        time_now_si=1e-9, tstep_si=1e-9, static=True)
    expected, diag = source.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    candidate = NativeElementFunctionSource(source, kernel=element_kernel)
    actual, got = candidate.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_close(actual, expected)
    np.testing.assert_array_equal(actual.transport_history.level, expected.transport_history.level)
    np.testing.assert_array_equal(actual.transport_history.previous_dt_si, expected.transport_history.previous_dt_si)
    np.testing.assert_array_equal(actual.transport_history.lifetime, expected.transport_history.lifetime)
    assert expected.transport_history.level[expected.cloud.alive].min() == 2
    assert candidate.last_stats['sample_points'] >= 16
    assert got['n_steps'] == diag['n_steps'] == 4


def test_narrow_marker_coordinates_are_explicitly_unsupported(element_kernel):
    source, initial, fields, sampler = window(static=True)
    initial.cloud.st = initial.cloud.st.astype(np.float32)
    before = deepcopy(initial)
    with pytest.raises(ValueError, match='float64 local'):
        NativeElementFunctionSource(source, kernel=element_kernel).advance(
            initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_exact(initial, before)
