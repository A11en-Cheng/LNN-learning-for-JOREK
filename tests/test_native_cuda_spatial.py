"""CPU differential coverage for the explicit FP64 CUDA spatial adapter."""
from unittest.mock import patch

import numpy as np
import pytest

from src.models.native_cuda_spatial import NativeCudaSpatial
from tests.test_native_locator_latency import locator_mesh


@pytest.mark.parametrize('count', [0, 1, 37])
@pytest.mark.parametrize('fields', [1, 5, 9])
def test_points_matches_all_five_native_outputs(count, fields):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    rng = np.random.default_rng(278)
    e = np.arange(count, dtype=np.int32) % 2
    st = rng.uniform(0., 1., (count, 2))
    values = rng.normal(size=(fields, 4, 1, mesh.n_nodes))
    actual = backend.points(e, st, values)
    expected = mesh.points(e, st, values)
    for got, want in zip(actual, expected):
        assert isinstance(got, np.ndarray) and got.dtype == np.float64
        np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('hint_mode', ['none', 'correct', 'crossing', 'invalid'])
def test_locator_elements_and_coordinates_match_cpu(hint_mode):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    rng = np.random.default_rng(392)
    st = np.vstack((rng.uniform(.05, .95, (32, 2)),
                    [[0., 0.], [1., 1.], [0., .5], [1., .5], [1e-11, .5]]))
    e = np.arange(len(st)) % 2
    rz = mesh.points(e, st, mesh.coordinates[:1])[2]
    positions = np.vstack((np.column_stack((rz, np.ones(len(st)))), [9., 9., 0.]))
    kwargs = {}
    if hint_mode != 'none':
        hints = np.r_[e+1, 1]
        if hint_mode == 'crossing':
            hints = 3-hints
        elif hint_mode == 'invalid':
            hints[::2] = 0
            hints[1::2] = 99
        kwargs = dict(element_hints=hints, local_hints=rng.uniform(-.2, 1.2, (len(hints), 2)))
    before = {k: v.copy() for k, v in kwargs.items()}
    expected = mesh.locate_batch(positions, **kwargs)
    actual = backend.locate_batch(positions, **kwargs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-12)
    for key, value in before.items():
        np.testing.assert_array_equal(kwargs[key], value)


def test_threshold_and_boundary_ambiguity_uses_original_cpu_hints():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    st = np.array([[.5, .5], [0., .5]])
    rz = mesh.points(np.array([0, 0]), st, mesh.coordinates[:1])[2]
    rz[0, 0] += 1e-10
    positions = np.column_stack((rz, np.zeros(2)))
    kwargs = dict(element_hints=np.ones(2, dtype=int), local_hints=st.copy())
    expected = mesh.locate(positions, **kwargs)
    with patch.object(mesh, 'locate', wraps=mesh.locate) as fallback:
        actual = backend.locate_batch(positions, **kwargs)
    assert fallback.call_count == 1
    np.testing.assert_array_equal(fallback.call_args.kwargs['local_hints'], st)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_fields_and_particle_positions_are_never_cached_by_identity():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    e = np.array([0]); st = np.array([[.2, .3]])
    values = np.ones((1, 4, 1, mesh.n_nodes))
    first = backend.points(e, st, values)[0]
    values *= 2
    second = backend.points(e, st, values)[0]
    np.testing.assert_array_equal(second, 2*first)
    st[:] = [.7, .8]
    backend.reset_event_cache(event_binding={'event': 2})
    for got, want in zip(backend.points(e, st, values), mesh.points(e, st, values)):
        np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('name', ['coordinates', 'size', 'vertex', 'indices'])
def test_mutated_geometry_is_rejected_before_resident_data_can_go_stale(name):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    getattr(mesh, name).flat[0] += 1
    with pytest.raises(ValueError, match='binding'):
        backend.points(np.array([0]), np.array([[.5, .5]]), mesh.coordinates[:1])


def test_configuration_binding_and_device_are_explicit():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    assert backend.configuration['version'] == 'native-cuda-spatial-fp64-v1'
    assert backend.configuration['test_only'] is True
    assert backend.configuration['geometry'] == mesh.configuration['geometry']
    with pytest.raises(ValueError, match='device'):
        NativeCudaSpatial(mesh, device='meta')
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='CUDA'):
            NativeCudaSpatial(mesh)
    mesh.configuration['locator'] = 'changed'
    with pytest.raises(ValueError, match='binding'):
        backend.reset_event_cache()


@pytest.mark.parametrize('e,st', [([-1], [[.5, .5]]), ([2], [[.5, .5]]),
                                ([.5], [[.5, .5]]), ([0], [[np.nan, .5]]),
                                ([0], [[.5, .5, .5]])])
def test_invalid_point_inputs_match_cpu_guard(e, st):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    for sampler in (mesh, backend):
        with pytest.raises(ValueError, match='element/local'):
            sampler.points(np.asarray(e), np.asarray(st), mesh.coordinates[:1])


def test_empty_locator_and_nonfinite_positions_and_hints():
    backend = NativeCudaSpatial(locator_mesh(), device='cpu')
    out = backend.locate_batch(np.empty((0, 3)), element_hints=np.empty(0, dtype=int),
                               local_hints=np.empty((0, 2)))
    assert out[0].shape == (0,) and out[1].shape == (0, 2)
    with pytest.raises(ValueError, match='positions'):
        backend.locate_batch([[np.nan, 0., 0.]])
    with pytest.raises(ValueError, match='hints'):
        backend.locate_batch([[2.5, .5, 0.]], element_hints=[1], local_hints=[[np.nan, .5]])


def test_event_snapshots_replace_slots_validate_once_and_clear_on_exit():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    values = np.ones((5, 4, 1, mesh.n_nodes))
    e = np.array([0]); st = np.array([[.2, .3]])
    with patch.object(backend, '_binding_digest', wraps=backend._binding_digest) as digest:
        with backend.event_scope({'event': 1}):
            backend.freeze_fields('field', values)
            initial = backend.points_frozen('field', e, st)[0]
            values *= 2
            # The uploaded snapshot is owned and immune to caller mutation.
            np.testing.assert_array_equal(backend.points_frozen('field', e, st)[0], initial)
            backend.freeze_fields('field', values)
            np.testing.assert_array_equal(backend.points_frozen('field', e, st)[0], 2*initial)
            backend.freeze_fields('feedback', values[:1])
            assert len(backend._fields) == 2
            position = mesh.points(e, st, values)[2]
            backend.locate_batch(np.column_stack((position, [0.])),
                                 element_hints=[1], local_hints=[[.5, .5]])
            assert digest.call_count == 1
        assert digest.call_count == 1
    assert backend._fields == {}
    with pytest.raises(ValueError, match='active spatial event'):
        backend.points_frozen('field', e, st)
    with backend.event_scope({'event': 2}):
        assert backend._fields == {}
        with pytest.raises(ValueError, match='missing frozen'):
            backend.points_frozen('field', e, st)


def test_event_exception_and_failed_replacement_cannot_reuse_old_fields():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    with pytest.raises(RuntimeError, match='event failure'):
        with backend.event_scope(1):
            backend.freeze_fields('field', mesh.coordinates)
            with pytest.raises(ValueError, match='nonnested'):
                with backend.event_scope(2):
                    pass
            with pytest.raises(ValueError, match='coefficients'):
                backend.freeze_fields('field', np.ones((4, 4)))
            assert 'field' not in backend._fields
            backend.freeze_fields('field', mesh.coordinates)
            raise RuntimeError('event failure')
    assert not backend._in_event and backend._fields == {}
    mesh.size.flat[0] += .1
    with pytest.raises(ValueError, match='binding'):
        with backend.event_scope(3):
            pass


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_fields_match_native_validation_and_invalidate_snapshot(bad):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    values = np.ones((9, 4, 1, mesh.n_nodes))
    invalid = values.copy()
    invalid[8, 3, 0, -1] = bad  # Even an unsampled node must be validated.
    e = np.array([0]); st = np.array([[.2, .3]])
    with pytest.raises(ValueError, match='coefficients'):
        mesh.validate_fields(invalid, 9)
    with pytest.raises(ValueError, match='finite'):
        backend.points(e, st, invalid)
    with backend.event_scope(1):
        backend.freeze_fields('field', values)
        with pytest.raises(ValueError, match='finite'):
            backend.freeze_fields('field', invalid)
        with pytest.raises(ValueError, match='missing frozen'):
            backend.points_frozen('field', e, st)


@pytest.mark.parametrize('fault', ['singular', 'negative_jacobian', 'nonfinite', 'negative_radius'])
def test_geometry_guards_and_batch_scalar_recovery_match_cpu(fault):
    mesh = locator_mesh()
    if fault == 'singular':
        mesh.coordinates[1, :, :, :4] = 0.
    elif fault == 'negative_jacobian':
        mesh.coordinates[1, :, :, :4] *= -1
    elif fault == 'nonfinite':
        mesh.coordinates[1, 1, 0, :4] = np.nan
    else:
        mesh.coordinates[0, 0, 0, :4] -= 100.
    backend = NativeCudaSpatial(mesh, device='cpu')
    with pytest.raises(ValueError, match='geometry'):
        backend.points(np.array([0, 1]), np.full((2, 2), .5), mesh.coordinates[:1])
    positions = np.array([[2.4, .3, 0.], [3.4, .3, 0.]])
    kwargs = dict(element_hints=np.array([1, 2]), local_hints=np.full((2, 2), .5))
    expected = mesh.locate_batch(positions, **kwargs)
    actual = backend.locate_batch(positions, **kwargs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-12)


def test_damping_overshoot_and_noncontiguous_field_inputs():
    from src.models.native_coupling import NativeSourceMesh
    from tests.test_native_gauss import native_coefficients, rectangle_geometry

    geometry = rectangle_geometry()
    x = geometry.coordinates.copy()
    x[0, :, 0] = native_coefficients(lambda s, t: 2+.05*s+s**3,
                                    lambda s, t: .05+3*s*s,
                                    lambda s, t: 0., lambda s, t: 0.)
    mesh = NativeSourceMesh(x, np.ones((4, 4, 1)), np.arange(4)[:, None],
                            np.arange(1, 17).reshape(4, 4),
                            units=locator_mesh().units, projection='none')
    backend = NativeCudaSpatial(mesh, device='cpu')
    positions = np.array([[2+.05*.4+.4**3, .5, 0.]])
    kwargs = dict(element_hints=[1], local_hints=[[.05, .5]])
    expected = mesh.locate_batch(positions, **kwargs)
    actual = backend.locate_batch(positions, **kwargs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-12)
    fields = np.arange(144., dtype=float).reshape(9, 4, 1, 4)[::-1, :, :, ::-1]
    for got, want in zip(backend.points(np.array([0]), actual[1], fields),
                         mesh.points(np.array([0]), actual[1], fields)):
        np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('count', [0, 1, 37])
def test_frozen_many_matches_separate_calls_and_evaluates_geometry_once(count):
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    rng = np.random.default_rng(719)
    e = np.arange(count, dtype=np.int32) % 2
    st = rng.uniform(.05, .95, (count, 2))
    with backend.event_scope(1):
        backend.freeze_fields('field', rng.normal(size=(9, 4, 1, mesh.n_nodes)))
        backend.freeze_fields('feedback', rng.normal(size=(5, 4, 1, mesh.n_nodes)))
        keys = ('field', 'feedback', 'field')
        expected = tuple(backend.points_frozen(key, e, st) for key in keys)
        with patch.object(backend, '_geometry', wraps=backend._geometry) as geometry, \
                patch.object(backend, '_inputs', wraps=backend._inputs) as inputs:
            actual = backend.points_frozen_many(keys, e, st)
        assert geometry.call_count == inputs.call_count == 1
        assert len(actual) == len(keys)
        for got_outputs, want_outputs in zip(actual, expected):
            assert len(got_outputs) == 5
            for got, want in zip(got_outputs, want_outputs):
                np.testing.assert_array_equal(got, want)
        # Per-call sharing must never retain the old particle coordinates.
        moved = st.copy(); moved[:, 0] += .01
        changed = backend.points_frozen_many(keys, e, moved)
        for got_outputs, key in zip(changed, keys):
            for got, want in zip(got_outputs, backend.points_frozen(key, e, moved)):
                np.testing.assert_array_equal(got, want)


def test_frozen_many_requires_active_scope_and_all_named_slots():
    mesh = locator_mesh()
    backend = NativeCudaSpatial(mesh, device='cpu')
    e = np.array([0]); st = np.array([[.2, .3]])
    with pytest.raises(ValueError, match='active spatial event'):
        backend.points_frozen_many(('field',), e, st)
    with backend.event_scope(1):
        backend.freeze_fields('field', mesh.coordinates)
        with patch.object(backend, '_geometry', wraps=backend._geometry) as geometry:
            with pytest.raises(ValueError, match='missing frozen'):
                backend.points_frozen_many(('field', 'missing'), e, st)
            assert geometry.call_count == 0
        with pytest.raises(ValueError, match='nonempty'):
            backend.points_frozen_many((), e, st)
        with pytest.raises(ValueError, match='keys'):
            backend.points_frozen_many('field', e, st)
