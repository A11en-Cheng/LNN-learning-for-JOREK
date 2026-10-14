"""Exact CPU oracles for shared sampling and geometry-only batch deposition."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

import src.models.native_coupling as coupling
from src.models.marker_state import MarkerCloud
from tests.test_native_locator_latency import locator_mesh


def legacy_points(mesh, e, st, values):
    """Frozen pre-sharing arithmetic, independent of future production helpers."""
    e = np.asarray(e); st = np.asarray(st); a = np.asarray(values)
    if (e.ndim != 1 or e.dtype.kind not in 'iu' or (e < 0).any()
            or (e >= len(mesh.gauss)).any() or st.shape != (len(e), 2)
            or not np.isfinite(st).all()):
        raise ValueError('invalid batch element/local coordinates')
    h, hs, ht = coupling._cubic_basis(st[:, 0], st[:, 1])
    sizes = mesh.size[:, :, e].transpose(2, 1, 0)
    b = h*sizes; bs = hs*sizes; bt = ht*sizes; ids = mesh.vertex[:, e].T
    x = mesh.coordinates[:, :, 0, :][:, :, ids].transpose(2, 0, 3, 1)
    xyz = np.einsum('bvd,bfvd->bf', b, x)
    xs = np.einsum('bvd,bfvd->bf', bs, x)
    xt = np.einsum('bvd,bfvd->bf', bt, x)
    jac = xs[:, 0]*xt[:, 1]-xt[:, 0]*xs[:, 1]
    if not np.isfinite(jac).all() or (jac <= 0).any() or (xyz[:, 0] <= 0).any():
        raise ValueError('invalid native point geometry')
    local = a[:, :, 0, :][:, :, ids].transpose(2, 0, 3, 1)
    val = np.einsum('bvd,bfvd->bf', b, local)
    ds = np.einsum('bvd,bfvd->bf', bs, local)
    dt = np.einsum('bvd,bfvd->bf', bt, local)
    grad = np.stack(((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/jac[:, None],
                     (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/jac[:, None]), axis=-1)
    return val, grad, xyz, b, np.stack((xs, xt), axis=-1)


def legacy_deposit(mesh, cloud, ids, load):
    """Original dummy-field sampler and accumulation order."""
    e = cloud.i_elm[ids].astype(np.int64)-1
    basis = legacy_points(mesh, e, cloud.st[ids], np.zeros((1, 4, 1, mesh.n_nodes)))[3]
    dofs = mesh.inverse[:, mesh.vertex[:, e].T].transpose(1, 2, 0).ravel()
    contributions = (load[:, :, None, None]*basis[:, None, :, :]).transpose(1, 0, 2, 3).reshape(5, -1)
    rhs = np.zeros((5, mesh.n_dofs))
    for channel in range(5):
        np.add.at(rhs[channel], dofs, contributions[channel])
    return rhs


class LegacyBatchMesh(coupling.NativeSourceMesh):
    def points(self, e, st, values):
        return legacy_points(self, e, st, values)

    def points_many(self, e, st, fields):
        return tuple(self.points(e, st, values) for values in fields)

    def _marker_points(self, e, st, fields, feedback):
        field_point, feedback_point = self.points_many(e, st, (fields, feedback))
        return field_point[0], field_point[1], feedback_point[0], field_point[2]

    def deposit_batch(self, cloud, ids, load):
        return legacy_deposit(self, cloud, ids, load)


@pytest.mark.parametrize('count', [0, 1, 31])
def test_frozen_oracle_matches_existing_public_points(count):
    mesh = locator_mesh()
    rng = np.random.default_rng(402)
    e = np.arange(count, dtype=np.int64) % 2
    st = rng.uniform(0., 1., (count, 2))
    for width in (1, 5, 9):
        a = rng.normal(size=(width, 4, 1, mesh.n_nodes))
        for got, want in zip(mesh.points(e, st, a), legacy_points(mesh, e, st, a)):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('count', [0, 1, 31])
def test_public_points_many_is_exact_and_evaluates_geometry_once(count):
    mesh = locator_mesh()
    rng = np.random.default_rng(404)
    e = np.arange(count, dtype=np.int64) % 2
    st = rng.uniform(0., 1., (count, 2))
    fields = (rng.normal(size=(9, 4, 1, mesh.n_nodes)),
              rng.normal(size=(5, 4, 1, mesh.n_nodes))[:, :, :, ::-1])
    expected = tuple(legacy_points(mesh, e, st, a) for a in fields)
    with patch.object(coupling, '_cubic_basis', wraps=coupling._cubic_basis) as basis, \
            patch.object(np, 'einsum', wraps=np.einsum) as contractions:
        actual = mesh.points_many(e, st, fields)
    assert basis.call_count == 1
    geometry_ops = sum(c.args[0] == 'bvd,bfvd->bf' and c.args[2].shape[1] == 2
                       for c in contractions.call_args_list)
    assert geometry_ops == 3  # RZ, dRZ/ds, dRZ/dt, each once for both fields.
    for got_outputs, want_outputs in zip(actual, expected):
        for got, want in zip(got_outputs, want_outputs):
            np.testing.assert_array_equal(got, want)
    assert len(actual) == len(fields)
    # Reusing the same inputs by identity must not cache their previous contents.
    st[:, 0] *= .7
    fields[0][...] *= 2
    for got_outputs, a in zip(mesh.points_many(e, st, fields), fields):
        for got, want in zip(got_outputs, legacy_points(mesh, e, st, a)):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('empty', [False, True])
def test_public_deposit_batch_exact_repeated_dofs_without_dummy_field_work(empty):
    original = locator_mesh()
    indices = original.indices.copy(); indices[3, -1] = indices[3, 0]
    mesh = coupling.NativeSourceMesh(original.coordinates, original.size, original.vertex,
                                     indices, units=original.units, projection='none')
    rng = np.random.default_rng(439)
    cloud = MarkerCloud.empty(10, 7)
    cloud.i_elm[:] = np.arange(7) % 2+1
    cloud.st[:] = rng.uniform(0., 1., (7, 2))
    ids = np.array([], dtype=int) if empty else np.array([5, 2, 5, 0, 2])
    loads = rng.normal(size=(len(ids), 5))
    expected = legacy_deposit(mesh, cloud, ids, loads)
    with patch.object(mesh, 'points', wraps=mesh.points) as fields, \
            patch.object(np, 'einsum', wraps=np.einsum) as contractions:
        actual = mesh.deposit_batch(cloud, ids, loads)
    np.testing.assert_array_equal(actual, expected)
    dummy_ops = sum(c.args[0] == 'bvd,bfvd->bf' and c.args[2].shape[1] == 1
                   for c in contractions.call_args_list)
    assert dummy_ops == 0, f'deposition contracted a discarded field {dummy_ops} times'
    assert fields.call_count == 0


@pytest.mark.parametrize('fault', ['singular', 'negative_jacobian', 'nonfinite', 'negative_radius'])
def test_shared_sampling_and_deposition_keep_geometry_guards(fault):
    mesh = locator_mesh()
    if fault == 'singular':
        mesh.coordinates[1, :, :, :4] = 0.
    elif fault == 'negative_jacobian':
        mesh.coordinates[1, :, :, :4] *= -1
    elif fault == 'nonfinite':
        mesh.coordinates[1, 1, 0, :4] = np.nan
    else:
        mesh.coordinates[0, 0, 0, :4] -= 100.
    cloud = MarkerCloud.empty(10, 1); cloud.i_elm[:] = 1; cloud.st[:] = .5
    with pytest.raises(ValueError, match='geometry'):
        legacy_points(mesh, np.array([0]), cloud.st, mesh.coordinates)
    with pytest.raises(ValueError, match='geometry'):
        mesh.points_many(np.array([0]), cloud.st, (mesh.coordinates,))
    with pytest.raises(ValueError, match='geometry'):
        mesh.deposit_batch(cloud, np.array([0]), np.ones((1, 5)))


@pytest.mark.parametrize('e,st', [([-1], [[.5, .5]]), ([2], [[.5, .5]]),
                                ([.5], [[.5, .5]]), ([0], [[np.nan, .5]]),
                                ([0], [[.5, .5, .5]])])
def test_points_many_preserves_element_local_input_guards(e, st):
    mesh = locator_mesh()
    with pytest.raises(ValueError, match='element/local'):
        mesh.points_many(np.asarray(e), np.asarray(st), (mesh.coordinates,))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_shared_sampling_validates_every_full_field(bad):
    mesh = locator_mesh()
    good = np.ones((5, 4, 1, mesh.n_nodes))
    invalid = good.copy(); invalid[4, 3, 0, -1] = bad
    with pytest.raises(ValueError, match='coefficients'):
        mesh.points_many(np.array([0]), np.array([[.5, .5]]), (good, invalid))
    with pytest.raises(ValueError, match='coefficients'):
        mesh.points(np.array([0]), np.array([[.5, .5]]), invalid)


def test_shared_sampling_rejects_empty_field_group_and_wrong_layout():
    mesh = locator_mesh()
    e = np.array([0]); st = np.array([[.5, .5]])
    with pytest.raises(ValueError, match='nonempty'):
        mesh.points_many(e, st, ())
    with pytest.raises(ValueError, match='coefficients'):
        mesh.points_many(e, st, (np.ones((5, 4, 2, mesh.n_nodes)),))


@pytest.mark.parametrize('cubic', [False, True])
def test_complete_cpu_source_event_matches_frozen_full_sampling_exactly(cubic):
    from tests.test_native_coupling import native_fixture

    mesh, original, state, u = native_fixture()
    if cubic:
        rng = np.random.default_rng(483)
        u += rng.normal(scale=1e-7, size=u.shape)
        state.feedback.values[:, 1:] = rng.normal(scale=1e-10, size=(5, 3, 1, 4))
        state.feedback.values[3, 0] = 1e-8
        state.feedback.values[4, 0] = 1e-9
        state.feedback.deltas[:] = rng.normal(scale=1e-10, size=state.feedback.deltas.shape)
    source = coupling.NativeMarkerSource(
        mesh, atomic=original.atomic, atomic_version=original.configuration['atomic'],
        potentials_ev=original.potentials_ev, F0=original.F0,
        diffusion_ju=original.diffusion_ju, spatial_backend='batched_v1')
    reference = deepcopy(source)
    reference.mesh = LegacyBatchMesh(mesh.coordinates, mesh.size, mesh.vertex,
                                      mesh.indices, units=mesh.units)
    expected, expected_diagnostic = reference.advance(state, 1e-9, u, u)
    actual, actual_diagnostic = source.advance(state, 1e-9, u, u)
    for name in state.cloud.array_names():
        np.testing.assert_array_equal(getattr(actual.cloud, name), getattr(expected.cloud, name))
    np.testing.assert_array_equal(actual.feedback.values, expected.feedback.values)
    np.testing.assert_array_equal(actual.feedback.deltas, expected.feedback.deltas)
    for name in ('energy', 'rad', 'moments'):
        np.testing.assert_array_equal(actual_diagnostic[name], expected_diagnostic[name])
    assert actual.clock == expected.clock
