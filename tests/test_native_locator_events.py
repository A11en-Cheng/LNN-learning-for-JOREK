"""Locator changes observed only through complete public source events."""
from copy import deepcopy

import numpy as np
import pytest

from src.models.native_coupling import NativeSourceMesh, NativeMarkerSource
from src.models.native_state import NativeUnits
from src.models.marker_feedback import MU_ZERO, ATOMIC_MASS_UNIT
from src.models.marker_state import MarkerCloud, MarkerRuntime, ProjectionFeedback, MarkerClock
from tests.native_locator_reference import FrozenLocatorMesh
from tests.test_native_locator_latency import locator_mesh
from tests.test_native_gauss import native_coefficients


def event_fixture(mesh_type=NativeSourceMesh, backend="scalar", *, scenario="normal", dtype=np.float64):
    geometry = locator_mesh()
    units = NativeUnits(np.sqrt(MU_ZERO*ATOMIC_MASS_UNIT*8e19), 8e19, 1., 5/3)
    mesh = mesh_type(geometry.coordinates, geometry.size, geometry.vertex,
                     geometry.indices, units=units, projection="dense")

    def atomic(te, ne):
        scd = np.zeros((len(ne), 10)); scd[:, 0] = 1e-12
        return scd, np.zeros_like(scd), np.zeros((len(ne), 11))

    source = NativeMarkerSource(mesh, atomic=atomic, atomic_version="fixture-SCD",
        potentials_ev=np.full(10, 20.), F0=-1., diffusion_ju=0., spatial_backend=backend)
    cloud = MarkerCloud.empty(10, 3)
    st = {"normal": [.3, .4], "crossing": [.95, .4], "lost": [.95, .4]}[scenario]
    e = 1 if scenario == "lost" else 0
    cloud.st = cloud.st.astype(dtype)
    cloud.st[0] = st
    cloud.i_elm[0] = e+1; cloud.i_life[0] = 4; cloud.weight[0] = 1e12
    # Integer st is an accepted input representation: use its actual corner.
    cloud.x[0, :2] = mesh.point(e, cloud.st[0], mesh.coordinates[:1])[2]
    velocity_scale = 1e8
    cloud.V_prev[0, 0] = cloud.x[0, 0]*velocity_scale
    fields = np.zeros((9, 4, 1, mesh.n_nodes))
    fields[4, 0] = 1.; fields[7, 0] = .5; fields[8, 0] = .01
    # u=-a*Z*t_norm gives radial velocity a*R, using full cubic Z.
    fields[1] = -velocity_scale*units.t_norm_si*mesh.coordinates[1]
    feedback = np.zeros((5, 4, 1, mesh.n_nodes))
    state = MarkerRuntime(cloud, ProjectionFeedback(feedback, feedback.copy(), 0.),
                          MarkerClock(0., 0., 1e-9, 0., 1e-9))
    return source, state, fields


def assert_runtime_exact(actual, expected):
    for name in actual.cloud.array_names():
        a, b = getattr(actual.cloud, name), getattr(expected.cloud, name)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b, err_msg=name)
    np.testing.assert_array_equal(actual.cloud.alive, expected.cloud.alive)
    assert actual.cloud.group == expected.cloud.group
    np.testing.assert_array_equal(actual.feedback.values, expected.feedback.values)
    np.testing.assert_array_equal(actual.feedback.deltas, expected.feedback.deltas)
    assert actual.feedback.time_si == expected.feedback.time_si
    assert actual.feedback.layout == expected.feedback.layout
    for name, value in vars(actual.feedback).items():
        if not isinstance(value, np.ndarray):
            assert value == getattr(expected.feedback, name)
    assert actual.clock == expected.clock


def test_frozen_original_source_event_baseline():
    source, state, fields = event_fixture(FrozenLocatorMesh)
    before = deepcopy(state)
    result, diagnostic = source.advance(state, 1e-9, fields, fields)
    assert result.cloud.i_elm[0] == 1
    assert result.cloud.x[0, 0] > state.cloud.x[0, 0]
    assert result.cloud.P_imp[0, 1] > 0
    assert np.any(result.feedback.values)
    assert diagnostic["n_steps"] == 1
    assert_runtime_exact(state, before)


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
@pytest.mark.parametrize("scenario,expected_element", [("normal", 1), ("crossing", 2), ("lost", 0)])
def test_source_event_curved_cells_exact_frozen_locator(backend, scenario, expected_element):
    reference, state, fields = event_fixture(FrozenLocatorMesh, backend, scenario=scenario)
    candidate, candidate_state, candidate_fields = event_fixture(backend=backend, scenario=scenario)
    before = deepcopy(candidate_state)
    expected, old_diagnostic = reference.advance(state, 1e-9, fields, fields)
    actual, new_diagnostic = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    assert actual.cloud.i_elm[0] == expected_element
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(candidate_state, before)
    for key in ("energy", "rad", "moments", "n_steps", "gather_time_seconds"):
        np.testing.assert_array_equal(new_diagnostic[key], old_diagnostic[key])


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_source_event_preserves_narrow_local_coordinate_writeback(backend, dtype):
    reference, state, fields = event_fixture(FrozenLocatorMesh, backend, dtype=dtype)
    candidate, candidate_state, candidate_fields = event_fixture(backend=backend, dtype=dtype)
    # MarkerCloud.validate does not reject finite integer or float32 st.
    candidate_state.cloud.validate()
    expected, _ = reference.advance(state, 1e-9, fields, fields)
    actual, _ = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    assert actual.cloud.st.dtype == dtype
    assert_runtime_exact(actual, expected)


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_two_source_events_observe_geometry_change_without_stale_cache(backend):
    reference, state, fields = event_fixture(FrozenLocatorMesh, backend)
    candidate, candidate_state, candidate_fields = event_fixture(backend=backend)
    expected, _ = reference.advance(state, 1e-9, fields, fields)
    actual, _ = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    assert_runtime_exact(actual, expected)
    unchanged = deepcopy(candidate)
    for source, runtime in ((reference, expected), (candidate, actual)):
        # Valid geometry mutation between calls, not a new physical-run claim.
        # No search bounds/factors are rebuilt: compare the old API semantics.
        source.mesh.coordinates[1] *= 1.01
        runtime.clock.field_start_si = 1e-9
        runtime.clock.field_end_si = 2e-9
    before = deepcopy(actual)
    expected2, _ = reference.advance(expected, 2e-9, fields, fields)
    actual2, _ = candidate.advance(actual, 2e-9, candidate_fields, candidate_fields)
    unchanged2, _ = unchanged.advance(actual, 2e-9, candidate_fields, candidate_fields)
    assert_runtime_exact(actual2, expected2)
    assert_runtime_exact(actual, before)
    assert actual2.cloud.alive[0]
    assert not np.array_equal(actual2.cloud.st, unchanged2.cloud.st)


class StatefulGeometry:
    """Observable evaluator extension: third evaluation raises, just once."""
    failure_type = RuntimeError

    def _tick_geometry(self):
        self.geometry_calls = getattr(self, "geometry_calls", 0) + 1
        if self.geometry_calls == 3:
            raise self.failure_type("stateful third geometry evaluation")

    def _location_geometry(self, e, st):
        self._tick_geometry()
        return super()._location_geometry(e, st)

    def _location_geometries(self, e, st):
        self._tick_geometry()
        return super()._location_geometries(e, st)


class StatefulNativeMesh(StatefulGeometry, NativeSourceMesh):
    pass


class StatefulFrozenMesh(StatefulGeometry, FrozenLocatorMesh):
    pass


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_source_stateful_geometry_override_keeps_propagated_exception_and_rollback(backend):
    # The third evaluation is the next main iteration after an accepted trial.
    # Reusing a stateful override would suppress this public event failure.
    for mesh_type in (StatefulFrozenMesh, StatefulNativeMesh):
        source, state, fields = event_fixture(mesh_type, backend)
        before = deepcopy(state)
        with pytest.raises(RuntimeError, match="stateful third geometry evaluation"):
            source.advance(state, 1e-9, fields, fields)
        assert_runtime_exact(state, before)


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_source_stateful_geometry_valueerror_keeps_old_fallback_policy(backend):
    reference, state, fields = event_fixture(StatefulFrozenMesh, backend)
    candidate, candidate_state, candidate_fields = event_fixture(StatefulNativeMesh, backend)
    reference.mesh.failure_type = candidate.mesh.failure_type = ValueError
    expected, old_diag = reference.advance(state, 1e-9, fields, fields)
    actual, new_diag = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    assert_runtime_exact(actual, expected)
    # Scalar fails this candidate; batch failure falls back to scalar search.
    assert actual.cloud.i_elm[0] == (0 if backend == "scalar" else 1)
    for key in ("energy", "rad", "moments"):
        np.testing.assert_array_equal(new_diag[key], old_diag[key])


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_source_instance_geometry_override_keeps_exception_semantics(backend):
    for mesh_type in (FrozenLocatorMesh, NativeSourceMesh):
        source, state, fields = event_fixture(mesh_type, backend)
        before = deepcopy(state)
        name = "_location_geometry" if backend == "scalar" else "_location_geometries"
        original = getattr(source.mesh, name)
        calls = 0

        def stateful(e, st):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("instance third geometry evaluation")
            return original(e, st)

        setattr(source.mesh, name, stateful)
        with pytest.raises(RuntimeError, match="instance third geometry evaluation"):
            source.advance(state, 1e-9, fields, fields)
        assert_runtime_exact(state, before)


def mixed_event_fixture(mesh_type, backend):
    source, state, fields = event_fixture(mesh_type, backend)
    mesh = source.mesh
    # At t=0 the poloidal velocity is zero, but parallel toroidal motion
    # remains nonzero. That row finishes immediately while others need Newton.
    potential = native_coefficients(lambda s, t: t*t, lambda s, t: 0.,
                                    lambda s, t: 2*t, lambda s, t: 0.)
    for e in range(2):
        fields[1, :, 0, mesh.vertex[:, e]] = (
            -1e8*mesh.units.t_norm_si*potential/mesh.size[:, :, e]).T
    fields[6, 0] = 1e-3
    cloud = MarkerCloud.empty(10, 7)
    ids = np.array([0, 1, 3, 4, 6])
    elements = np.array([0, 0, 0, 1, 1])
    local = np.array([[.2, 0.], [.3, .4], [.95, .4], [.3, .6], [.95, .4]])
    cloud.i_elm[ids] = elements+1
    cloud.i_life[ids] = np.arange(1, 6)
    cloud.weight[ids] = np.arange(1, 6)*1e11
    cloud.st[ids] = local
    for slot, e, st in zip(ids, elements, local):
        value, gradient, rz, *_ = mesh.point(int(e), st, fields)
        cloud.x[slot] = [*rz, .1*slot]
        cloud.V_prev[slot] = [
            -gradient[1, 1]*rz[0]/mesh.units.t_norm_si,
            gradient[1, 0]*rz[0]/mesh.units.t_norm_si,
            -value[6]/mesh.units.t_norm_si/rz[0],
        ]
    state.cloud = cloud
    return source, state, fields


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_source_mixed_live_batch_partial_completion_crossing_and_loss(backend):
    reference, state, fields = mixed_event_fixture(FrozenLocatorMesh, backend)
    candidate, candidate_state, candidate_fields = mixed_event_fixture(NativeSourceMesh, backend)
    before = deepcopy(candidate_state)
    expected, old_diag = reference.advance(state, 1e-9, fields, fields)
    actual, new_diag = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    np.testing.assert_array_equal(actual.cloud.i_elm, [1, 1, 0, 2, 2, 0, 0])
    np.testing.assert_array_equal(actual.cloud.st[0], candidate_state.cloud.st[0])
    assert actual.cloud.x[0, 2] != candidate_state.cloud.x[0, 2]
    assert not np.array_equal(actual.cloud.st[1], candidate_state.cloud.st[1])
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(candidate_state, before)
    for key in ("energy", "rad", "moments", "n_steps", "gather_time_seconds"):
        np.testing.assert_array_equal(new_diag[key], old_diag[key])


def stiff_event_fixture(mesh_type, backend):
    source, _, _ = event_fixture(mesh_type, backend)
    r = native_coefficients(lambda s, t: 2+.05*s+s**3,
                            lambda s, t: .05+3*s*s,
                            lambda s, t: 0., lambda s, t: 0.)
    z = native_coefficients(lambda s, t: t, lambda s, t: 0.,
                            lambda s, t: 1., lambda s, t: 0.)
    source.mesh = mesh_type(np.stack((r, z))[:, :, None, :],
        np.ones((4, 4, 1)), np.arange(4)[:, None], np.arange(1, 17).reshape(4, 4),
        units=source.mesh.units, projection="dense")
    source.configuration["mesh"] = source.mesh.configuration
    mesh = source.mesh
    start_r, target_r = 2+.05*.05+.05**3, 2+.05*.4+.4**3
    rate = (target_r-start_r)/(start_r*1e-9)
    fields = np.zeros((9, 4, 1, 4))
    fields[4, 0] = 1.; fields[7, 0] = .5; fields[8, 0] = .01
    fields[1] = -rate*mesh.units.t_norm_si*mesh.coordinates[1]
    cloud = MarkerCloud.empty(10, 2)
    cloud.i_elm[0] = 1; cloud.i_life[0] = 2; cloud.weight[0] = 1e12
    cloud.st[0] = [.05, .5]
    cloud.x[0] = [start_r, .5, 0.]
    cloud.V_prev[0, 0] = rate*start_r
    feedback = np.zeros((5, 4, 1, 4))
    state = MarkerRuntime(cloud, ProjectionFeedback(feedback, feedback.copy(), 0.),
                          MarkerClock(0., 0., 1e-9, 0., 1e-9))
    return source, state, fields


@pytest.mark.parametrize("backend", ["scalar", "batched_v1"])
def test_source_stiff_cubic_requires_damping_and_matches_frozen_locator(backend):
    reference, state, fields = stiff_event_fixture(FrozenLocatorMesh, backend)
    candidate, candidate_state, candidate_fields = stiff_event_fixture(NativeSourceMesh, backend)
    before = deepcopy(candidate_state)
    expected, old_diag = reference.advance(state, 1e-9, fields, fields)
    actual, new_diag = candidate.advance(candidate_state, 1e-9, candidate_fields, candidate_fields)
    # Analytic map proves damping is needed; no private-call-count oracle.
    radius = lambda s: 2+.05*s+s**3
    target_r = actual.cloud.x[0, 0]
    error = abs(radius(.05)-target_r)
    step = (radius(.05)-target_r)/(.05+3*.05**2)
    assert abs(radius(np.clip(.05-step, 0, 1))-target_r) > error
    assert abs(radius(np.clip(.05-.5*step, 0, 1))-target_r) > error
    assert abs(radius(np.clip(.05-.25*step, 0, 1))-target_r) < error
    assert actual.cloud.i_elm[0] == 1
    np.testing.assert_allclose(actual.cloud.st[0], [.4, .5], rtol=0, atol=2e-10)
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(candidate_state, before)
    for key in ("energy", "rad", "moments", "n_steps", "gather_time_seconds"):
        np.testing.assert_array_equal(new_diag[key], old_diag[key])
