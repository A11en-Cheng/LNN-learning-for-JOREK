"""Optional batch execution must retain the complete public source event."""
import numpy as np
from tests.test_native_coupling import native_fixture


def test_batched_source_event_matches_scalar_cloud_history_and_projection():
    from src.models.native_coupling import NativeMarkerSource
    mesh,source,state,u=native_fixture()
    batch=NativeMarkerSource(mesh,atomic=source.atomic,atomic_version='fixture_prescribed_SCD_m3_s',
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=source.diffusion_ju,
        spatial_backend='batched_v1')
    reference,_=source.advance(state,1e-9,u,u)
    actual,_=batch.advance(state,1e-9,u,u)
    for name in state.cloud.array_names():
        np.testing.assert_allclose(getattr(actual.cloud,name),getattr(reference.cloud,name),rtol=2e-12,atol=0.)
    np.testing.assert_allclose(actual.feedback.values,reference.feedback.values,rtol=2e-12,atol=1e-25)
    np.testing.assert_allclose(actual.feedback.deltas,reference.feedback.deltas,rtol=2e-12,atol=1e-25)
    assert actual.clock==reference.clock
    assert batch.configuration!=source.configuration


def test_batch_cubic_values_gradients_and_repeated_deposits_match_scalar():
    from src.models.marker_state import MarkerCloud
    mesh,_,_,_=native_fixture()
    rng=np.random.default_rng(713)
    st=rng.uniform(.1,.9,(19,2));elements=np.zeros(19,dtype=int)
    values=rng.normal(size=(9,4,1,4))
    batch=mesh.points(elements,st,values)
    scalar=list(zip(*(mesh.point(0,p,values) for p in st)))
    for actual,expected in zip(batch,scalar):
        np.testing.assert_allclose(actual,np.asarray(expected),rtol=2e-13,atol=1e-13)
    cloud=MarkerCloud.empty(10,19);cloud.i_elm[:]=1;cloud.st[:]=st
    loads=rng.normal(size=(19,5));ids=np.arange(19)
    np.testing.assert_allclose(mesh.deposit_batch(cloud,ids,loads),mesh.deposit(cloud,ids,loads),rtol=2e-13,atol=1e-13)


def test_batch_hint_location_preserves_boundary_and_missing_point_behavior():
    mesh,_,_,_=native_fixture()
    positions=np.array([[2.2,.3,0.],[2.9,.8,0.],[9.,9.,0.],[2.,0.,0.]])
    hints=np.ones(4,dtype=int);local=np.full((4,2),.5)
    expected=mesh.locate(positions,element_hints=hints,local_hints=local)
    actual=mesh.locate_batch(positions,element_hints=hints,local_hints=local)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(actual[1],expected[1],rtol=0,atol=2e-12)
