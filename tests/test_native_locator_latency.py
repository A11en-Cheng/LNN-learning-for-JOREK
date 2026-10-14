"""Geometry location must not interpolate dummy fields during Newton trials."""
from unittest.mock import patch

import numpy as np
import pytest

from src.models.native_coupling import NativeSourceMesh
from src.models.native_state import NativeUnits
from tests.test_native_gauss import native_coefficients


class FullFieldLocatorReference(NativeSourceMesh):
    """Retain the original full-field geometry evaluation as a differential oracle."""

    def _location_geometry(self, e, st):
        result = self.point(e, st, np.zeros((1, 4, 1, self.n_nodes)))
        return result[2], result[4]

    def _location_geometries(self, e, st):
        result = self.points(e, st, np.zeros((1, 4, 1, self.n_nodes)))
        return result[2], result[4]


def locator_mesh(mesh_type=NativeSourceMesh):
    """Two adjacent curved cells, with signed, nonunit native size factors."""
    coordinates = []
    sizes = []
    for offset in (0., 1.):
        r = native_coefficients(
            lambda s, t: 2 + offset + s + .1*t*(1-t),
            lambda s, t: 1., lambda s, t: .1*(1-2*t), lambda s, t: 0.)
        z = native_coefficients(
            lambda s, t: t + .12*(offset+s)*t*(1-t),
            lambda s, t: .12*t*(1-t),
            lambda s, t: 1 + .12*(offset+s)*(1-2*t),
            lambda s, t: .12*(1-2*t))
        size = np.arange(1., 17.).reshape(4, 4)/8
        size[1::2, ::2] *= -1
        coordinates.append(np.stack((r, z))[:, :, None, :]/size[None, :, None, :])
        sizes.append(size)
    return mesh_type(
        np.concatenate(coordinates, axis=-1), np.stack(sizes, axis=-1),
        np.arange(8).reshape(2, 4).T, np.arange(1, 33).reshape(4, 8),
        units=NativeUnits(1e-7, 8e19, 1., 5/3), projection='none')


@pytest.mark.parametrize('method', ['locate', 'locate_batch'])
def test_public_locator_does_no_dummy_field_evaluations(method):
    mesh = locator_mesh()
    # Correct hint, wrong-cell hint (scalar fallback), and out-of-mesh point.
    st = np.array([[.23, .41], [.67, .19]])
    rz = mesh.points(np.array([0, 1]), st, mesh.coordinates[:1])[2]
    positions = np.vstack((np.column_stack((rz, [0., .7])), [9., 9., 0.]))
    with patch.object(mesh, 'point', wraps=mesh.point) as scalar, \
            patch.object(mesh, 'points', wraps=mesh.points) as batch, \
            patch.object(np, 'einsum', wraps=np.einsum) as contractions:
        elements, local = getattr(mesh, method)(
            positions, element_hints=np.ones(3, dtype=int),
            local_hints=np.full((3, 2), .5))
    np.testing.assert_array_equal(elements, [1, 2, 0])
    np.testing.assert_allclose(local[:2], st, rtol=0, atol=2e-10)
    # Count actual discarded field contractions, even if moved into a helper.
    # Geometry has two coordinate channels; the dummy field has one channel.
    field_contractions = sum(
        (call.args[0] == 'vd,fvd->f' and call.args[2].shape[0] == 1)
        or (call.args[0] == 'bvd,bfvd->bf' and call.args[2].shape[1] == 1)
        for call in contractions.call_args_list)
    assert field_contractions == 0, (
        'Newton location contracted dummy field coefficients', field_contractions)
    assert (scalar.call_count, batch.call_count) == (0, 0), (
        'Newton location evaluated dummy field values/gradients',
        scalar.call_count, batch.call_count)


@pytest.mark.parametrize('method', ['locate', 'locate_batch'])
@pytest.mark.parametrize('hint_mode', ['none', 'correct', 'crossing', 'invalid'])
def test_public_locator_matches_full_field_reference_exactly(method, hint_mode):
    mesh = locator_mesh()
    reference = locator_mesh(FullFieldLocatorReference)
    rng = np.random.default_rng(721)
    st = np.vstack((rng.uniform(0., 1., (24, 2)),
                    [[0., 0.], [1., 1.], [0., .5], [1., .5],
                     [.5, 0.], [.5, 1.], [1e-11, .5], [1-1e-11, .5]]))
    elements = np.arange(len(st)) % 2
    rz = mesh.points(elements, st, mesh.coordinates[:1])[2]
    positions = np.vstack((np.column_stack((rz, rng.normal(size=len(st)))),
                           [[9., 9., 0.], [2.-5e-10, .5, 0.]]))
    kwargs = {}
    if hint_mode != 'none':
        hints = np.r_[elements+1, 1, 1]
        if hint_mode == 'crossing':
            hints = 3-hints
        elif hint_mode == 'invalid':
            hints[::2] = 0
            hints[1::2] = 99
        kwargs = dict(element_hints=hints,
                      local_hints=rng.uniform(-.5, 1.5, (len(positions), 2)))
    before = {name: value.copy() for name, value in kwargs.items()}
    expected = getattr(reference, method)(positions, **kwargs)
    actual = getattr(mesh, method)(positions, **kwargs)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    assert np.all(actual[0][:24] > 0)
    assert actual[0][-1] == actual[0][-2] == 0
    for name, value in before.items():
        np.testing.assert_array_equal(kwargs[name], value)


@pytest.mark.parametrize('method', ['locate', 'locate_batch'])
def test_public_locator_preserves_convergence_threshold_exactly(method):
    mesh = locator_mesh()
    reference = locator_mesh(FullFieldLocatorReference)
    hints = np.full((2, 2), .5)
    rz = mesh.points(np.zeros(2, dtype=int), hints, mesh.coordinates[:1])[2]
    rz[:, 0] += [.5e-10, 2e-10]
    positions = np.column_stack((rz, np.zeros(2)))
    kwargs = dict(element_hints=np.ones(2, dtype=int), local_hints=hints)
    expected = getattr(reference, method)(positions, **kwargs)
    actual = getattr(mesh, method)(positions, **kwargs)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(actual[0], [1, 1])
    np.testing.assert_array_equal(actual[1][0], hints[0])
    assert not np.array_equal(actual[1][1], hints[1])


@pytest.mark.parametrize('method', ['locate', 'locate_batch'])
def test_public_locator_preserves_rejected_damping_trials_exactly(method):
    from tests.native_locator_reference import FrozenLocatorMesh

    # Positive-Jacobian cubic: the first full and half Newton steps overshoot.
    r = native_coefficients(lambda s, t: 2+.05*s+s**3,
                            lambda s, t: .05+3*s*s,
                            lambda s, t: 0., lambda s, t: 0.)
    z = native_coefficients(lambda s, t: t, lambda s, t: 0.,
                            lambda s, t: 1., lambda s, t: 0.)
    results = []
    for mesh_type in (FrozenLocatorMesh, NativeSourceMesh):
        mesh = mesh_type(
            np.stack((r, z))[:, :, None, :], np.ones((4, 4, 1)),
            np.arange(4)[:, None], np.arange(1, 17).reshape(4, 4),
            units=NativeUnits(1e-7, 8e19, 1., 5/3), projection='none')
        results.append(getattr(mesh, method)(
            np.array([[2+.05*.4+.4**3, .5, 0.]]),
            element_hints=np.array([1]), local_hints=np.array([[.05, .5]])))
    for got, want in zip(results[1], results[0]):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(results[1][0], [1])
    # Preserve the adversarial public result, not redundant basis evaluations.
    np.testing.assert_allclose(results[1][1], [[.4, .5]], rtol=0, atol=2e-10)


@pytest.mark.parametrize('method', ['locate', 'locate_batch'])
@pytest.mark.parametrize('fault', ['singular', 'negative_jacobian', 'nonfinite', 'negative_radius'])
def test_public_locator_empty_and_invalid_geometry_match_reference(method, fault):
    mesh = locator_mesh()
    reference = locator_mesh(FullFieldLocatorReference)
    for current in (mesh, reference):
        empty = getattr(current, method)(
            np.empty((0, 3)), element_hints=np.empty(0, dtype=int),
            local_hints=np.empty((0, 2)))
        assert empty[0].shape == (0,) and empty[1].shape == (0, 2)
        # Keep search bounds so the invalid first candidate reaches its guard.
        if fault == 'singular':
            current.coordinates[1, :, :, :4] = 0.
        elif fault == 'negative_jacobian':
            current.coordinates[1, :, :, :4] *= -1
        elif fault == 'nonfinite':
            current.coordinates[1, 1, 0, :4] = np.nan
        else:
            current.coordinates[0, 0, 0, :4] -= 100.
    positions = np.array([[2.4, .3, 0.], [3.4, .3, 0.]])
    kwargs = dict(element_hints=np.array([1, 2]), local_hints=np.full((2, 2), .5))
    expected = getattr(reference, method)(positions, **kwargs)
    actual = getattr(mesh, method)(positions, **kwargs)
    np.testing.assert_array_equal(actual[0], [0, 2])
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('backend', ['scalar', 'batched_v1'])
def test_small_source_event_matches_full_field_locator_exactly(backend):
    from copy import deepcopy
    from tests.test_native_coupling import native_fixture

    mesh, source, state, values = native_fixture()
    source.spatial_backend = backend
    reference = deepcopy(source)
    reference.mesh = FullFieldLocatorReference(
        mesh.coordinates, mesh.size, mesh.vertex, mesh.indices, units=mesh.units)
    expected, _ = reference.advance(state, 1e-9, values, values)
    actual, _ = source.advance(state, 1e-9, values, values)
    for name in state.cloud.array_names():
        np.testing.assert_array_equal(getattr(actual.cloud, name), getattr(expected.cloud, name))
    np.testing.assert_array_equal(actual.feedback.values, expected.feedback.values)
    np.testing.assert_array_equal(actual.feedback.deltas, expected.feedback.deltas)
    assert actual.feedback.time_si == expected.feedback.time_si
    assert actual.clock == expected.clock
