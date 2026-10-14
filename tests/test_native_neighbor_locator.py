"""Neighbor experiment observed through the agreed complete Source.advance seam."""
from copy import deepcopy

import numpy as np
import pytest


def shared_event(scenario='crossing',n_cells=2):
    from tests.test_native_locator_events import event_fixture
    from src.models.native_coupling import NativeSourceMesh
    source,state,_ = event_fixture(backend='batched_v1',scenario=scenario)
    # Independent global endpoint jets: R=2+X+.1*t*(1-t),
    # Z=t+.12*X*t*(1-t), two conforming cells X in [0,1] and [1,2].
    X=np.array([0.,1.,1.,0.]+[float(k) for k in range(2,n_cells+1) for _ in range(2)])
    t=np.array([0.,0.,1.,1.]+[v for _ in range(2,n_cells+1) for v in (0.,1.)])
    n_nodes=len(X)
    coords=np.zeros((2,4,1,n_nodes))
    coords[0,:,0,:]=np.stack((2+X+.1*t*(1-t),np.ones(n_nodes)/3,.1*(1-2*t)/3,np.zeros(n_nodes)))
    coords[1,:,0,:]=np.stack((t+.12*X*t*(1-t),.12*t*(1-t)/3,
                            (1+.12*X*(1-2*t))/3,.12*(1-2*t)/9))
    size=np.repeat(np.array([[1,1,1,1],[1,-1,-1,1],
                             [1,1,-1,-1],[1,-1,1,-1]])[:,:,None],n_cells,axis=2)
    vertex=np.array([[0,1,2,3]]+[[1 if k==1 else 2*k,2*k+2,2*k+3,2 if k==1 else 2*k+1]
                                for k in range(1,n_cells)]).T
    mesh=NativeSourceMesh(coords,size,vertex,
        np.arange(1,4*n_nodes+1).reshape(4,n_nodes),units=source.mesh.units,projection='dense')
    source.mesh=mesh
    source.configuration['mesh']=deepcopy(mesh.configuration)
    e=int(state.cloud.i_elm[0])-1
    state.cloud.x[0,:2]=mesh.point(e,state.cloud.st[0],coords[:1])[2]
    state.cloud.V_prev[0,0]=state.cloud.x[0,0]*1e8
    fields=np.zeros((9,4,1,n_nodes)); fields[4,0]=1.; fields[7,0]=.5; fields[8,0]=.01
    fields[1]=-1e8*mesh.units.t_norm_si*coords[1]
    state.feedback.values=np.zeros((5,4,1,n_nodes)); state.feedback.deltas=state.feedback.values.copy()
    return source,state,fields


def test_complete_source_event_crosses_curved_shared_edge_without_global_search():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event(); before=deepcopy(state)
    expected,old_diag=native.advance(state,1e-9,fields,fields)
    experimental=NativeNeighborMarkerSource(native)
    actual,diag=experimental.advance(state,1e-9,fields,fields)
    assert expected.cloud.i_elm[0] == actual.cloud.i_elm[0] == 2
    for name in ('x','weight','P_imp','V_prev','i_elm','i_life','t_birth'):
        np.testing.assert_array_equal(getattr(actual.cloud,name),getattr(expected.cloud,name))
    np.testing.assert_allclose(actual.cloud.st,expected.cloud.st,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(actual.feedback.values,expected.feedback.values)
    np.testing.assert_array_equal(actual.feedback.deltas,expected.feedback.deltas)
    assert actual.clock == expected.clock
    assert diag['gather_time_seconds'] == old_diag['gather_time_seconds']
    assert experimental.last_stats['neighbor_accepted'] == 1
    assert experimental.last_stats['global_fallback_points'] == 0
    assert_runtime_exact(state,before)


def test_complete_event_lost_marker_retains_global_boundary_policy():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event('lost')
    expected,expected_diag=native.advance(state,1e-9,fields,fields)
    experiment=NativeNeighborMarkerSource(native)
    actual,diag=experiment.advance(state,1e-9,fields,fields)
    assert expected.cloud.i_elm[0] == 0
    assert_runtime_exact(actual,expected)
    assert diag['gather_time_seconds'] == expected_diag['gather_time_seconds']
    assert diag['sampling'] == expected_diag['sampling']
    assert experiment.last_stats['global_fallback_points'] == 1
    assert experiment.last_stats['neighbor_accepted'] == 0


def test_experiment_rejects_narrow_marker_coordinates_instead_of_changing_rounding():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event()
    state.cloud.st=state.cloud.st.astype(np.float32)
    before=deepcopy(state)
    experiment=NativeNeighborMarkerSource(native)
    with pytest.raises(ValueError,match='float64'):
        experiment.advance(state,1e-9,fields,fields)
    assert_runtime_exact(state,before)


def test_reversed_neighbor_axes_preserve_curved_source_event():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from src.models.native_coupling import NativeSourceMesh
    native,state,fields=shared_event()
    mesh=native.mesh
    vertex=mesh.vertex.copy(); vertex[:,1]=[5,2,1,4]
    size=mesh.size.copy(); size[1:3,:,1]*=-1
    native.mesh=NativeSourceMesh(mesh.coordinates,size,vertex,mesh.indices,
                                units=mesh.units,projection='dense')
    native.configuration['mesh']=deepcopy(native.mesh.configuration)
    expected,_=native.advance(state,1e-9,fields,fields)
    experiment=NativeNeighborMarkerSource(native)
    actual,_=experiment.advance(state,1e-9,fields,fields)
    assert actual.cloud.i_elm[0] == expected.cloud.i_elm[0] == 2
    np.testing.assert_allclose(actual.cloud.st,expected.cloud.st,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(actual.cloud.x,expected.cloud.x)
    np.testing.assert_array_equal(actual.feedback.values,expected.feedback.values)
    assert experiment.last_stats['neighbor_accepted'] == 1
    assert experiment.last_stats['global_fallback_points'] == 0


def test_disconnected_topology_uses_original_global_search_exactly():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import event_fixture,assert_runtime_exact
    native,state,fields=event_fixture(backend='batched_v1',scenario='crossing')
    experiment=NativeNeighborMarkerSource(native)
    expected,_=native.advance(state,1e-9,fields,fields)
    actual,_=experiment.advance(state,1e-9,fields,fields)
    assert_runtime_exact(actual,expected)
    assert actual.cloud.i_elm[0] == 2
    assert experiment.last_stats['global_fallback_points'] == 1
    assert experiment.last_stats['neighbor_accepted'] == 0


@pytest.mark.parametrize('name',['coordinates','size','vertex','indices'])
def test_mutating_static_mesh_rejects_event_without_partial_commit(name):
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event()
    experiment=NativeNeighborMarkerSource(native)
    before=deepcopy(state)
    getattr(native.mesh,name).flat[0]+=1
    with pytest.raises(ValueError,match='mesh changed'):
        experiment.advance(state,1e-9,fields,fields)
    assert_runtime_exact(state,before)


def test_two_source_events_and_fresh_adapter_resume_preserve_projection_history():
    import pickle
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event()
    experiment=NativeNeighborMarkerSource(native)
    first,_=experiment.advance(state,1e-9,fields,fields)
    first.clock.field_start_si=1e-9; first.clock.field_end_si=2e-9
    restored=pickle.loads(pickle.dumps(first))
    continued,_=experiment.advance(first,2e-9,fields,fields)
    resumed,_=NativeNeighborMarkerSource(native).advance(restored,2e-9,fields,fields)
    assert_runtime_exact(continued,resumed)
    assert np.any(continued.feedback.deltas)
    expected,_=native.advance(first,2e-9,fields,fields)
    assert_runtime_exact(continued,expected)
    assert experiment.configuration != native.configuration
    assert experiment.configuration['experimental_neighbor_locator']['production_qualified'] is False


def test_geometry_override_added_after_construction_is_not_silently_bypassed():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event()
    experiment=NativeNeighborMarkerSource(native)
    native.mesh._location_geometries=native.mesh._location_geometries
    before=deepcopy(state)
    with pytest.raises(ValueError,match='overridden'):
        experiment.advance(state,1e-9,fields,fields)
    assert_runtime_exact(state,before)


@pytest.mark.parametrize('max_hops',[1,2,4])
def test_two_cell_crossing_is_bounded_and_falls_back_when_budget_expires(max_hops):
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    native,state,fields=shared_event(n_cells=3)
    fields[1]*=4; state.cloud.V_prev*=4
    expected,_=native.advance(state,1e-9,fields,fields)
    experiment=NativeNeighborMarkerSource(native,max_hops=max_hops)
    actual,_=experiment.advance(state,1e-9,fields,fields)
    assert actual.cloud.i_elm[0] == expected.cloud.i_elm[0] == 3
    np.testing.assert_array_equal(actual.cloud.x,expected.cloud.x)
    np.testing.assert_allclose(actual.cloud.st,expected.cloud.st,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(actual.cloud.P_imp,expected.cloud.P_imp)
    np.testing.assert_array_equal(actual.feedback.values,expected.feedback.values)
    assert experiment.last_stats['neighbor_accepted'] == (max_hops>=2)
    assert experiment.last_stats['global_fallback_points'] == (max_hops==1)


def test_neighbor_adapter_preserves_two_coupled_events_and_cross_step_backpropagation():
    import torch
    from src.models.native_coupling import NativeMarkerSource
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tests.test_native_production_torch_integration import fixture,control,AnalyticEquilibrium
    model,legacy,shard,runtime=fixture()
    source=NativeMarkerSource(legacy.mesh,atomic=legacy.atomic,
        atomic_version=legacy.configuration['atomic'],potentials_ev=legacy.potentials_ev,
        F0=legacy.F0,diffusion_ju=legacy.diffusion_ju,time_sampling=legacy.time_sampling,
        field_static=legacy.field_static,spatial_backend='batched_v1')
    experiment=NativeNeighborMarkerSource(source)
    model.source_configuration=deepcopy(experiment.configuration)
    runtime.native_context.configuration['model']=model.checkpoint_configuration
    with torch.no_grad():model.field.decoder.weight.fill_(1e-4)
    adapter=TorchRecomputedSourceEvents(model,experiment,shard,equilibrium_provider=AnalyticEquilibrium())
    initial=adapter.initialize(runtime)
    for value in (initial.values,initial.raw_deltas,initial.memory):value.requires_grad_()
    np.random.seed(1931)
    first,_=adapter.advance(initial,control())
    for value in (first.values,first.raw_deltas,first.memory):value.retain_grad()
    second,_=adapter.advance(first,control(1))
    (second.values.square().sum()+second.raw_deltas.square().sum()+second.memory.square().sum()).backward()
    for value in (initial.values,initial.raw_deltas,initial.memory,first.values,first.raw_deltas,first.memory):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert second.runtime.marker.cloud.alive.sum()==2
    assert experiment.last_stats['point_visits']>0


def test_shared_neighbor_outer_edge_retains_original_candidate_order():
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields=shared_event()
    target=native.mesh.point(1,np.array([.2,0.]),fields[:1])[2]
    velocity=(target-state.cloud.x[0,:2])/1e-9
    a,b=velocity/state.cloud.x[0,0]
    fields[1]=native.mesh.units.t_norm_si*(-a*native.mesh.coordinates[1]+b*native.mesh.coordinates[0])
    state.cloud.V_prev[0,:2]=velocity
    experiment=NativeNeighborMarkerSource(native)
    expected,_=native.advance(state,1e-9,fields,fields)
    actual,_=experiment.advance(state,1e-9,fields,fields)
    assert expected.cloud.i_elm[0] == 2
    assert_runtime_exact(actual,expected)
    assert experiment.last_stats['edge_ambiguous'] == 1
    assert experiment.last_stats['global_fallback_points'] == 1


@pytest.mark.parametrize('static',[True,False])
@pytest.mark.parametrize('time_sampling',['jorek_raw_deltas_v1','endpoint_reference'])
def test_native_saved_history_and_sampling_schedule_are_unchanged(static,time_sampling):
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from src.models.native_source_time import NativeSourceTimeSampler
    from tests.test_native_marker_sampling import window
    from tests.test_native_locator_events import assert_runtime_exact
    native,state,fields,sampler=window(static=static,time_sampling=time_sampling)
    if sampler is not None:
        sampler=NativeSourceTimeSampler(fields,fields*.02,time_prev_si=0.,time_now_si=1e-9,
                                      tstep_si=1e-9,static=static)
    experiment=NativeNeighborMarkerSource(native)
    expected,expected_diag=native.advance(state,1e-9,fields*.99,fields,time_sampler=sampler)
    actual,diag=experiment.advance(state,1e-9,fields*.99,fields,time_sampler=sampler)
    assert_runtime_exact(actual,expected)
    assert diag['sampling'] == expected_diag['sampling']
    assert diag['n_steps'] == expected_diag['n_steps'] == 4
