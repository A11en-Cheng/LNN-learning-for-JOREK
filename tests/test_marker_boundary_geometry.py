"""Independent analytic evidence at the public cubic boundary-event seam.

No source integration, transport scheduling, mocked roots, or private geometry
calls: endpoint jets build real NativeSourceMesh cells and point() verifies
each returned event against the actual full cubic map.
"""
from copy import deepcopy

import numpy as np
import pytest

from src.models.marker_boundary_exit import BoundaryExit, NativeMarkerBoundaryExit
from src.models.marker_state import MarkerCloud
from src.models.native_coupling import NativeSourceMesh
from src.models.native_state import NativeUnits
from tests.test_native_gauss import native_coefficients


def analytic_mesh(curvature=0.):
    """R=2+s+a*(t-t**3), Z=t; det J=1, genuinely cubic curved faces."""
    coordinates = np.zeros((2, 4, 1, 4))
    coordinates[0, :, 0] = native_coefficients(
        lambda s, t: 2+s+curvature*(t-t**3),
        lambda s, t: 1., lambda s, t: curvature*(1-3*t*t),
        lambda s, t: 0.)
    coordinates[1, :, 0] = native_coefficients(
        lambda s, t: t, lambda s, t: 0., lambda s, t: 1.,
        lambda s, t: 0.)
    return NativeSourceMesh(
        coordinates, np.ones((4, 4, 1)), np.arange(4)[:, None],
        np.arange(1, 17).reshape(4, 4),
        units=NativeUnits(1., 1e20, 1., 5/3), projection='none')


def marker_cloud(mesh, local, previous, *, element=1, phi=.7):
    local = np.atleast_2d(np.asarray(local, dtype=np.float64))
    cloud = MarkerCloud.empty(capacity=len(local))
    cloud.st[:] = local
    cloud.i_elm[:] = element
    cloud.i_life[:] = np.arange(1, len(local)+1)
    cloud.weight[:] = 2.
    cloud.V_prev[:] = np.asarray(previous, dtype=np.float64)
    cloud.x[:, 2] = phi
    for i, st in enumerate(local):
        cloud.x[i, :2] = mesh.point(element-1, st, mesh.coordinates)[2]
    cloud.validate()
    return cloud


def two_rectangle_mesh():
    """Conforming cells R in [2,3] and [3,4], with actual shared node IDs."""
    coordinates = np.zeros((2, 4, 1, 6))
    coordinates[0, 0, 0] = [2., 3., 3., 2., 4., 4.]
    coordinates[1, 0, 0] = [0., 0., 1., 1., 0., 1.]
    coordinates[0, 1, 0] = 1/3
    coordinates[1, 2, 0] = 1/3
    signed_size = np.array([[1, 1, 1, 1], [1, -1, -1, 1],
                            [1, 1, -1, -1], [1, -1, 1, -1]])
    mesh = NativeSourceMesh(
        coordinates, np.repeat(signed_size[:, :, None], 2, axis=2),
        np.array([[0, 1], [1, 4], [2, 5], [3, 2]]),
        np.arange(1, 25).reshape(4, 6),
        units=NativeUnits(1., 1e20, 1., 5/3), projection='none')
    neighbours = np.zeros((4, 2), np.int32)
    neighbours[1, 0], neighbours[3, 1] = 2, 1
    # Check the independent fixture through point(), not a locator/root seam.
    for element in range(2):
        for st in ([0., 0.], [1., 1.], [.3, .7]):
            np.testing.assert_allclose(mesh.point(element, st, mesh.coordinates)[2],
                                       [2+element+st[0], st[1]], rtol=0, atol=1e-14)
    return mesh, neighbours


def assert_verified_event(mesh, event, *, dt, element=-1):
    assert isinstance(event, BoundaryExit)
    assert event.element == element
    assert event.edge in range(4)
    assert 0 < event.dt_si <= dt
    assert np.isfinite([event.dt_si, event.residual_m]).all()
    assert np.isfinite(event.x).all() and event.x.shape == (3,)
    assert np.isfinite(event.st).all() and event.st.shape == (2,)
    assert ((event.st >= 0) & (event.st <= 1)).all()
    coordinate, value = ((1, 0.), (0, 1.), (1, 1.), (0, 0.))[event.edge]
    assert event.st[coordinate] == value
    actual_rz = mesh.point(-element-1, event.st, mesh.coordinates)[2]
    residual = np.linalg.norm(actual_rz-event.x[:2])
    assert residual <= NativeMarkerBoundaryExit.tolerance_m
    assert 0 <= event.residual_m <= NativeMarkerBoundaryExit.tolerance_m


@pytest.mark.parametrize('st,velocity,edge,time,expected_st', [
    ([.2, .4], [2., 0., 0.], 1, .4, [1., .4]),
    ([.8, .4], [-2., 0., 0.], 3, .4, [0., .4]),
    ([.4, .2], [0., 2., 0.], 2, .4, [.4, 1.]),
    ([.4, .8], [0., -2., 0.], 0, .4, [.4, 0.]),
])
def test_rectangle_exit_has_analytic_duration_on_each_face(st, velocity, edge, time, expected_st):
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, st, velocity)
    before = deepcopy(cloud)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    provider.validate(mesh)
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([velocity]), 1., np.ones(1))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.)
    assert event.edge == edge
    assert event.dt_si == pytest.approx(time, rel=0, abs=1e-12)
    np.testing.assert_allclose(event.st, expected_st, rtol=0, atol=1e-12)
    np.testing.assert_allclose(event.x, [2+expected_st[0], expected_st[1], .7], rtol=0, atol=1e-12)
    for name in cloud.array_names():
        np.testing.assert_array_equal(getattr(cloud, name), getattr(before, name))


@pytest.mark.parametrize('st,velocity,dt,time', [
    ([.2, .5], [2., 0., 0.], 1., .4),
    ([.75, .25], [49/40, 1., 0.], .5, .25),
])
def test_cubic_curved_edge_not_vertex_chord(st, velocity, dt, time):
    # Both endpoints of the right face have R=3, but at t=.5 it has
    # R=3+.4*(.5-.5**3)=3.15. The second ray changes t as it moves.
    mesh = analytic_mesh(curvature=.4)
    cloud = marker_cloud(mesh, st, velocity)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([velocity]), dt, np.array([dt]))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=dt)
    assert event.edge == 1
    assert event.dt_si == pytest.approx(time, rel=0, abs=1e-12)
    np.testing.assert_allclose(event.st, [1., .5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(event.x, [3.15, .5, .7], rtol=0, atol=1e-12)
    assert abs(event.x[0]-3.) > .1


@pytest.mark.parametrize('previous_dt', [1., .5])
def test_direction_history_exit_uses_quadratic_not_linear_ray(previous_dt):
    # Unit speeds with orthogonal directions give
    # R(h)=2.5+h+h**2/(2*h_prev), Z(h)=.75-h**2/(2*h_prev).
    # Solve R=3 independently; the linear-ray exit would be h=.5.
    time = np.sqrt(previous_dt**2+previous_dt)-previous_dt
    expected_z = .75-time**2/(2*previous_dt)
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.5, .75], [0., 1., 0.])
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([[1., 0., 0.]]),
                           1., np.array([previous_dt]))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.)
    assert event.edge == 1
    assert event.dt_si == pytest.approx(time, rel=0, abs=1e-12)
    assert abs(event.dt_si-.5) > .08
    np.testing.assert_allclose(event.st, [1., expected_z], rtol=0, atol=1e-12)
    np.testing.assert_allclose(event.x, [3., expected_z, .7], rtol=0, atol=1e-12)


def test_pure_toroidal_current_velocity_still_has_quadratic_history_exit():
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.5, .4], [-1., 0., 0.])
    # Both 3D speeds are one, but current RZ velocity is zero. Direction
    # history gives R(h)=2.5+h**2/2, Z=.4, phi=.7+(h+h**2/2)/2.5.
    # The first right-face exit is h=1, before proposed h=1.1.
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([[0., 0., 1.]]),
                           1.1, np.ones(1))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.1)
    assert event.edge == 1
    assert event.dt_si == pytest.approx(1., rel=0, abs=1e-12)
    np.testing.assert_allclose(event.st, [1., .4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(event.x, [3., .4, 1.3], rtol=0, atol=1e-12)


def test_terminal_position_includes_finite_toroidal_push():
    mesh = analytic_mesh()
    velocity = [2., 0., .4]
    cloud = marker_cloud(mesh, [.2, .4], velocity)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([velocity]), 1., np.ones(1))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.)
    assert event.dt_si == pytest.approx(.4, rel=0, abs=1e-12)
    # Cylindrical angular advance uses the starting R=2.2, not the exit R.
    assert event.x[2] == pytest.approx(.7+.4*.4/2.2, rel=0, abs=1e-12)


@pytest.mark.parametrize('unknown', [-1, -7])
def test_negative_neighbour_does_not_certify_absorption(unknown):
    mesh = analytic_mesh()
    neighbours = np.zeros((4, 1), np.int32)
    neighbours[1, 0] = unknown
    cloud = marker_cloud(mesh, [.2, .4], [2., 0., 0.])
    provider = NativeMarkerBoundaryExit(mesh, neighbours)
    assert provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                          1., np.ones(1)) == {}


def test_entirely_unknown_topology_has_no_exterior_certificate():
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.2, .4], [2., 0., 0.])
    provider = NativeMarkerBoundaryExit(mesh, np.full((4, 1), -1, np.int32))
    assert provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                          1., np.ones(1)) == {}


def test_internal_face_precedes_downstream_outer_boundary():
    mesh, neighbours = two_rectangle_mesh()
    cloud = marker_cloud(mesh, [.8, .4], [2., 0., 0.])
    provider = NativeMarkerBoundaryExit(mesh, neighbours)
    # The first face at R=3 is internal (h=.1); true outer R=4 is h=.6.
    # This provider certifies only the current cell, not a multi-cell path.
    assert provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                          1., np.ones(1)) == {}


@pytest.mark.parametrize('first_neighbour', [2, -7])
def test_earlier_internal_or_unknown_hit_not_skipped_for_later_exterior(first_neighbour):
    if first_neighbour > 0:
        mesh, neighbours = two_rectangle_mesh()
    else:
        mesh = analytic_mesh()
        neighbours = np.zeros((4, 1), np.int32)
        neighbours[1, 0] = first_neighbour
    cloud = marker_cloud(mesh, [.999, .2], [1., 0., 0.])
    # Speed=sqrt(1.04), previous interval=.2. R first crosses the right
    # face, then re-enters; Z=.2+h+2.5*h**2 reaches exterior top at h=.4.
    radial_quadratic = (.2-np.sqrt(1.04))/.4
    discriminant = .2**2+4*radial_quadratic*.001
    first_time = (-.2+np.sqrt(discriminant))/(2*radial_quadratic)
    assert 0 < first_time < .01 < .4
    assert 2 < 2.999+.2*.4+radial_quadratic*.4**2 < 3
    provider = NativeMarkerBoundaryExit(mesh, neighbours)
    assert provider.exits(mesh, cloud, np.array([0]), np.array([[.2, 1., 0.]]),
                          .5, np.array([.2])) == {}


def assert_fail_closed(call):
    """Unsupported geometry may refuse explicitly or return no certificate."""
    try:
        result = call()
    except ValueError:
        return
    assert result is None or result == {}


@pytest.mark.parametrize('velocity', [
    [1., 1., 0.], [-1., 1., 0.], [-1., -1., 0.], [1., -1., 0.],
])
def test_exact_corner_is_not_arbitrarily_assigned_to_one_face(velocity):
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.5, .5], velocity)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    assert_fail_closed(lambda: provider.exits(
        mesh, cloud, np.array([0]), np.array([velocity]), 1., np.ones(1)))


def test_motion_coincident_with_edge_has_no_proven_outward_exit():
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.2, 0.], [.5, 0., 0.])
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    assert_fail_closed(lambda: provider.exits(
        mesh, cloud, np.array([0]), np.array([[.5, 0., 0.]]), 1., np.ones(1)))


def test_quadratic_touches_exterior_then_returns_inside_not_absorbed():
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.1, .84], [0., 1., 0.])
    # Unit current speed: R=2.1+.6*h+3*h**2, Z=.84+.8*h-h**2.
    # Top is a double root at h=.4, with zero outward speed. At .45 the
    # path is again inside and has never crossed either radial face.
    assert .84+.8*.4-.4**2 == pytest.approx(1., abs=1e-15)
    assert 2.1+.6*.45+3*.45**2 < 3
    assert .84+.8*.45-.45**2 < 1
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    assert_fail_closed(lambda: provider.exits(
        mesh, cloud, np.array([0]), np.array([[.6, .8, 0.]]), .45, np.array([.1])))


def test_vertical_ray_tangent_to_cubic_exterior_cannot_certify_loss():
    # Right face R=3-.4*(t-t**3) has its minimum at t=1/sqrt(3).
    # A vertical ray at that minimum remains inside before AND after contact:
    # numerical splitting of the double root must not turn it into absorption.
    curvature = -.4
    touch_t = 1/np.sqrt(3.)
    radius = 3+curvature*(touch_t-touch_t**3)
    start_t = .2
    start_s = radius-(2+curvature*(start_t-start_t**3))
    mesh = analytic_mesh(curvature)
    cloud = marker_cloud(mesh, [start_s, start_t], [0., 1., 0.])
    assert 0 < start_s < 1
    assert start_t < touch_t < start_t+.5
    for t in [touch_t-.01, touch_t+.01]:
        assert 0 < radius-(2+curvature*(t-t**3)) < 1
    np.testing.assert_allclose(mesh.point(0, [1., touch_t], mesh.coordinates)[2],
                               [radius, touch_t], rtol=0, atol=1e-14)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    assert_fail_closed(lambda: provider.exits(
        mesh, cloud, np.array([0]), np.array([[0., 1., 0.]]), .5, np.ones(1)))


def test_event_index_is_batch_relative_with_noncontiguous_cloud_slots():
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [[.2, .3], [.5, .5], [.5, .75]],
                         [[.1, 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    before = deepcopy(cloud)
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([2, 0]),
                           np.array([[1., 0., 0.], [.1, 0., 0.]]), 1., np.array([.5, 2.]))
    assert set(exits) == {0}  # event for cloud slot 2, not slot 0
    time = (np.sqrt(3.)-1)/2
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.)
    assert event.dt_si == pytest.approx(time, rel=0, abs=1e-12)
    np.testing.assert_allclose(event.st, [1., .75-time**2], rtol=0, atol=1e-12)
    for name in cloud.array_names():
        np.testing.assert_array_equal(getattr(cloud, name), getattr(before, name))


def test_terminal_element_identity_is_negative_current_cell_not_first_cell():
    mesh, neighbours = two_rectangle_mesh()
    cloud = marker_cloud(mesh, [.8, .4], [2., 0., 0.], element=2)
    provider = NativeMarkerBoundaryExit(mesh, neighbours)
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                           1., np.ones(1))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1., element=-2)
    assert event.edge == 1
    assert event.dt_si == pytest.approx(.1, rel=0, abs=1e-12)
    np.testing.assert_allclose(event.x, [4., .4, .7], rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', ['coordinates', 'size', 'vertex', 'indices'])
def test_public_validation_rejects_changed_geometry_arrays(name):
    mesh = analytic_mesh()
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    provider.validate(mesh)
    # Exercise all inputs used by geometry/configuration, even if the cached
    # configuration digest itself has not been refreshed after mutation.
    getattr(mesh, name).flat[0] += 1
    with pytest.raises(ValueError, match='geometry changed'):
        provider.validate(mesh)


def test_public_validation_rejects_changed_geometry_configuration():
    mesh = analytic_mesh()
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    mesh.configuration['geometry'] = 'different-geometry'
    with pytest.raises(ValueError, match='geometry changed'):
        provider.validate(mesh)


def test_public_validation_rejects_replaced_active_topology():
    mesh = analytic_mesh()
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    assert not provider.neighbours.flags.writeable
    # The documented active table is immutable, but replacement also needs
    # detection before any certificate can be relied upon.
    provider.neighbours = provider.neighbours.copy()
    provider.neighbours[0, 1] = -7
    with pytest.raises(ValueError, match='topology changed'):
        provider.validate(mesh)


def test_saved_topology_is_owned_not_aliased_to_callers_table():
    mesh = analytic_mesh()
    saved = np.zeros((4, 1), np.int32)
    provider = NativeMarkerBoundaryExit(mesh, saved)
    configuration = deepcopy(provider.configuration)
    saved[1, 0] = -7
    provider.validate(mesh)
    assert provider.configuration == configuration
    cloud = marker_cloud(mesh, [.2, .4], [2., 0., 0.])
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                           1., np.ones(1))
    assert set(exits) == {0}
    assert_verified_event(mesh, exits[0], dt=1.)


def test_missing_saved_topology_cannot_infer_exterior_from_geometry():
    with pytest.raises(ValueError, match='saved neighbours required'):
        NativeMarkerBoundaryExit(analytic_mesh(), None)


def test_nonreciprocal_internal_neighbour_table_is_rejected():
    mesh, neighbours = two_rectangle_mesh()
    neighbours[3, 1] = 0
    with pytest.raises(ValueError, match='not reciprocal'):
        NativeMarkerBoundaryExit(mesh, neighbours)


@pytest.mark.parametrize('st', [[-1e-3, .4], [1.001, .4], [.2, np.nan]])
def test_invalid_current_cell_coordinate_never_becomes_exit(st):
    mesh = analytic_mesh()
    cloud = marker_cloud(mesh, [.2, .4], [2., 0., 0.])
    cloud.st[0] = st
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    with pytest.raises(ValueError, match='inside current-cell coordinate'):
        provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]), 1., np.ones(1))


@pytest.mark.parametrize('curvature', [0., .4])
def test_certificate_refines_inside_hint_without_changing_sampled_state(curvature):
    mesh = analytic_mesh(curvature)
    cloud = marker_cloud(mesh, [.2, .5], [2., 0., 0.])
    # The physical position has an exact inverse in this same cell. A looser
    # accepted locator hint is evidence neither of death nor of changed fields.
    cloud.st[0] += [-.015, .025]
    before = deepcopy(cloud)
    hint_rz = mesh.point(0, cloud.st[0], mesh.coordinates)[2]
    assert np.linalg.norm(hint_rz-cloud.x[0, :2]) > 1e-3
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    exits = provider.exits(mesh, cloud, np.array([0]), np.array([[2., 0., 0.]]),
                           1., np.ones(1))
    assert set(exits) == {0}
    event = exits[0]
    assert_verified_event(mesh, event, dt=1.)
    assert event.dt_si == pytest.approx(.4, rel=0, abs=1e-12)
    np.testing.assert_allclose(event.st, [1., .5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(event.x, [3+.375*curvature, .5, .7], rtol=0, atol=1e-12)
    for name in cloud.array_names():
        np.testing.assert_array_equal(getattr(cloud, name), getattr(before, name))


@pytest.mark.parametrize('curvature', [0., .4])
def test_inside_hint_cannot_certify_already_outside_physical_position(curvature):
    mesh = analytic_mesh(curvature)
    cloud = marker_cloud(mesh, [.2, .4], [2., 0., 0.])
    cloud.x[0, 0] = 3+curvature*(.4-.4**3)+.1
    # Move back through the cell so the path's bounds reach exterior faces;
    # an outside origin with a stale inside hint is not an exit certificate.
    provider = NativeMarkerBoundaryExit(mesh, np.zeros((4, 1), np.int32))
    with pytest.raises(ValueError, match='position/geometry mismatch|outside current cell'):
        provider.exits(mesh, cloud, np.array([0]), np.array([[-2., 0., 0.]]), 1., np.ones(1))
