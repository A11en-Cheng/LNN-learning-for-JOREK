"""Experimental lookup is observed through the agreed public source event."""
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_coupling import native_fixture


def test_lookup_source_affine_constant_field_event_matches_reference():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    source.spatial_backend = "batched_v1"
    before = deepcopy(state)
    expected, _ = source.advance(state, 1e-9, fields, fields)
    lookup = ExperimentalLookupMesh(mesh, grid_points=9)
    candidate = make_lookup_source(source, lookup)
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    np.testing.assert_allclose(actual.cloud.x, expected.cloud.x, rtol=1e-13, atol=0)
    np.testing.assert_allclose(actual.cloud.P_imp, expected.cloud.P_imp, rtol=1e-13, atol=0)
    np.testing.assert_allclose(actual.feedback.values, expected.feedback.values, rtol=1e-12, atol=1e-20)
    np.testing.assert_array_equal(actual.cloud.i_elm, expected.cloud.i_elm)
    np.testing.assert_array_equal(state.cloud.x, before.cloud.x)
    assert actual.clock == expected.clock
    assert candidate.configuration != source.configuration


def test_lookup_source_cubic_sampling_error_decreases_with_resolution():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source
    from tests.test_native_gauss import native_coefficients

    mesh, source, state, fields = native_fixture()
    source.spatial_backend = "batched_v1"
    fields[4, :, 0] = native_coefficients(lambda s,t: 1+s**3+.2*t**3,
        lambda s,t: 3*s*s, lambda s,t: .6*t*t, lambda s,t: 0.)
    expected, _ = source.advance(state, 1e-9, fields, fields)
    # Independently calculated n_e=(1+.3^3+.2*.4^3-.5)*8e19.
    assert expected.cloud.P_imp[0, 1] == pytest.approx(.043184, abs=1e-14)
    errors = []
    for resolution in (5, 17):
        candidate = make_lookup_source(source, ExperimentalLookupMesh(mesh, grid_points=resolution))
        actual, _ = candidate.advance(state, 1e-9, fields, fields)
        errors.append(abs(actual.cloud.P_imp[0, 1]-expected.cloud.P_imp[0, 1]))
    assert 0 < errors[1] < errors[0]/4


@pytest.mark.parametrize("sampling", ["bilinear", "native_cubic_confirmed"])
def test_lookup_source_rebuilds_sampling_after_same_input_array_changes(sampling):
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    candidate = make_lookup_source(source, ExperimentalLookupMesh(mesh, grid_points=9, sampling=sampling))
    first, _ = candidate.advance(state, 1e-9, fields, fields)
    fields[4, 0] = 2.
    second, _ = candidate.advance(state, 1e-9, fields, fields)
    assert first.cloud.P_imp[0, 1] == pytest.approx(.04, abs=1e-14)
    assert second.cloud.P_imp[0, 1] == pytest.approx(.12, abs=1e-14)
    assert state.cloud.P_imp[0, 0] == 1.


@pytest.mark.parametrize("scenario,expected_element", [("crossing", 2), ("lost", 0)])
@pytest.mark.parametrize("sampling", ["bilinear", "native_cubic_confirmed"])
def test_lookup_source_crossing_and_loss_preserve_identity_and_clock(scenario, expected_element, sampling):
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source
    from tests.test_native_locator_events import event_fixture

    source, state, fields = event_fixture(backend="batched_v1", scenario=scenario)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(source.mesh, grid_points=17, sampling=sampling))
    before = deepcopy(state)
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    expected, _ = source.advance(state, 1e-9, fields, fields)
    assert actual.cloud.i_elm[0] == expected_element
    assert actual.clock == expected.clock
    np.testing.assert_array_equal(actual.cloud.i_life, expected.cloud.i_life)
    np.testing.assert_array_equal(actual.cloud.weight, expected.cloud.weight)
    np.testing.assert_array_equal(state.cloud.x, before.cloud.x)


@pytest.mark.parametrize("sampling", ["bilinear", "native_cubic_confirmed"])
def test_lookup_source_failed_atomic_event_does_not_commit_cloud_or_feedback(sampling):
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    candidate = make_lookup_source(source, ExperimentalLookupMesh(mesh, grid_points=9, sampling=sampling))
    def failing_atomic(te, ne):
        raise RuntimeError("prescribed atomic failure")
    candidate.atomic = failing_atomic
    before = deepcopy(state)
    with pytest.raises(RuntimeError, match="prescribed atomic failure"):
        candidate.advance(state, 1e-9, fields, fields)
    np.testing.assert_array_equal(state.cloud.x, before.cloud.x)
    np.testing.assert_array_equal(state.cloud.P_imp, before.cloud.P_imp)
    np.testing.assert_array_equal(state.feedback.values, before.feedback.values)
    assert state.clock == before.clock
    candidate.atomic = source.atomic
    fields[4, 0] = 2.
    retried, _ = candidate.advance(state, 1e-9, fields, fields)
    assert retried.cloud.P_imp[0, 1] == pytest.approx(.12, abs=1e-14)
    assert state.cloud.P_imp[0, 0] == 1.


@pytest.mark.parametrize("sampling", ["bilinear", "native_cubic_confirmed"])
def test_lookup_source_rejects_mesh_change_after_geometry_table_creation(sampling):
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    candidate = make_lookup_source(source, ExperimentalLookupMesh(mesh, grid_points=9, sampling=sampling))
    mesh.coordinates[0, 0, 0] += .1
    with pytest.raises(ValueError, match="geometry changed"):
        candidate.advance(state, 1e-9, fields, fields)


@pytest.mark.parametrize("sampling", ["bilinear", "native_cubic_confirmed"])
def test_lookup_source_consumes_nonzero_persisted_charge_feedback(sampling):
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    source.spatial_backend = "batched_v1"
    state.feedback.values[3, 0] = .4
    state.feedback.values[4, 0] = .1
    state.feedback.deltas[3, 0] = .02
    expected, _ = source.advance(state, 1e-9, fields, fields)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(mesh, grid_points=9, sampling=sampling))
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    # mean charge=4, n_imp=.5*8e19/20=2e18, n_e=4e19+4*2e18.
    assert actual.cloud.P_imp[0, 1] == pytest.approx(.048, abs=1e-14)
    np.testing.assert_allclose(actual.feedback.deltas, expected.feedback.deltas, rtol=1e-12, atol=1e-20)
    np.testing.assert_array_equal(state.feedback.values[3, 0, 0], [.4]*4)


def test_confirmed_cubic_source_consumes_exact_cubic_density_without_dense_field_approximation():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source
    from tests.test_native_gauss import native_coefficients

    mesh, source, state, fields = native_fixture()
    fields[4, :, 0] = native_coefficients(lambda s,t: 1+s**3+.2*t**3,
        lambda s,t: 3*s*s, lambda s,t: .6*t*t, lambda s,t: 0.)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(
        mesh, grid_points=5, sampling="native_cubic_confirmed"))
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    # Independent density/electron conversion at the fixture's s=.3,t=.4.
    assert actual.cloud.P_imp[0, 1] == pytest.approx(.043184, abs=1e-14)
    np.testing.assert_array_equal(state.cloud.P_imp[0], [1.]+[0.]*10)


def test_confirmed_cubic_source_curved_crossing_matches_native_event_geometry():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source
    from tests.test_native_locator_events import event_fixture

    source, state, fields = event_fixture(backend="batched_v1", scenario="crossing")
    expected, _ = source.advance(state, 1e-9, fields, fields)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(
        source.mesh, grid_points=3, sampling="native_cubic_confirmed"))
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    np.testing.assert_array_equal(actual.cloud.i_elm, expected.cloud.i_elm)
    np.testing.assert_allclose(actual.cloud.x, expected.cloud.x, rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual.cloud.st, expected.cloud.st, rtol=0, atol=2e-9)
    assert actual.clock == expected.clock


def test_confirmed_cubic_first_atomic_sample_matches_native_floating_point_density():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    mesh, source, state, fields = native_fixture()
    source.spatial_backend = "batched_v1"
    fields[4, :, 0] = np.random.default_rng(7).uniform(.1, 2., (4, 4))
    atomic = source.atomic
    samples = []
    def observed_atomic(te, ne):
        samples.append((te.copy(), ne.copy()))
        return atomic(te, ne)
    source.atomic = observed_atomic
    source.advance(state, 1e-9, fields, fields)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(
        mesh, grid_points=5, sampling="native_cubic_confirmed"))
    candidate.advance(state, 1e-9, fields, fields)
    np.testing.assert_array_equal(samples[1][1], samples[0][1])


def test_confirmed_cubic_first_velocity_keeps_native_geometry_arithmetic():
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source
    from tests.test_native_locator_events import event_fixture

    source, state, fields = event_fixture(backend="batched_v1", scenario="normal")
    expected, _ = source.advance(state, 1e-9, fields, fields)
    candidate = make_lookup_source(source, ExperimentalLookupMesh(
        source.mesh, grid_points=5, sampling="native_cubic_confirmed"))
    actual, _ = candidate.advance(state, 1e-9, fields, fields)
    np.testing.assert_array_equal(actual.cloud.V_prev, expected.cloud.V_prev)
