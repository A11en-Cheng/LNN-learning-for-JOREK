"""Diagnostic wrappers must not disable native geometry reuse or change outputs."""
import numpy as np
import pytest


def test_locator_probe_preserves_public_results_and_restores_mesh():
    from tools.diagnose_native_locator_work import LocatorProbe
    from tests.test_native_locator_latency import locator_mesh
    mesh = locator_mesh()
    st = np.array([[.23,.41],[.67,.19]])
    rz = mesh.points(np.array([0,1]),st,mesh.coordinates[:1])[2]
    positions = np.vstack((np.column_stack((rz,[0.,.7])),[9.,9.,0.]))
    kwargs = dict(element_hints=np.ones(3,dtype=int),local_hints=np.full((3,2),.5))
    expected = mesh.locate_batch(positions,**kwargs)
    original_tree = mesh._location_tree
    before = mesh.__dict__.copy()
    with LocatorProbe(mesh) as probe:
        assert '_location_geometry' not in mesh.__dict__
        assert '_location_geometries' not in mesh.__dict__
        actual = mesh.locate_batch(positions,**kwargs)
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(got,want)
    assert mesh._location_tree is original_tree
    assert mesh.__dict__.keys() == before.keys()
    summary = probe.summary()
    assert summary['batch_calls'] == 1 and summary['point_visits'] == 3
    assert summary['fallback_point_visits'] == 2
    assert summary['fallback_changed_element'] == 1
    assert summary['fallback_unresolved'] == 1
    assert summary['query_point_calls'] == 2
    assert sum(summary['candidate_count_histogram'].values()) == 2
    assert summary['batch_seconds'] >= summary['fallback_seconds'] >= 0


def test_locator_probe_restores_on_failure():
    from tools.diagnose_native_locator_work import LocatorProbe
    from tests.test_native_locator_latency import locator_mesh
    mesh = locator_mesh(); before = mesh.__dict__.copy()
    with pytest.raises(ValueError):
        with LocatorProbe(mesh):
            mesh.locate_batch(np.array([[np.nan,0.,0.]]))
    assert mesh.__dict__.keys() == before.keys()
    assert mesh._location_tree is before['_location_tree']
