"""Complete-source regression for JOREK-style edge-limited nearby search."""
import cProfile
import pstats
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_relocation_work import near_edge_event


def saved_chain_neighbours(n=2):
    neighbors = np.zeros((4, n), dtype=np.int32)
    neighbors[1, :-1] = np.arange(2, n+1)
    neighbors[3, 1:] = np.arange(1, n)
    return neighbors


def test_complete_source_crossing_uses_nearby_without_global_restart():
    source, state, fields = near_edge_event()
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    source = NativeNearbyMarkerSource(source, neighbours=saved_chain_neighbours())
    profile = cProfile.Profile()
    profile.enable()
    out, _ = source.advance(state, 1e-9, fields, fields)
    profile.disable()
    assert out.cloud.i_elm[0] == 2
    restarts = sum(v[1] for k, v in pstats.Stats(profile).stats.items()
                   if k[2] == '_locate')
    assert restarts == 0, f'crossing restarted global candidate search {restarts} times'
    assert source.last_stats['edge_crossings'] == 1
    assert source.last_stats['global_fallback_points'] == 0


@pytest.mark.parametrize('cells,target',[ (2, [1.2,.3]), (4,[3.2,.3]) ])
def test_curved_crossing_and_multi_element_moves(cells,target):
    from src.models.native_nearby_locator import NativeNearbyLocator
    from tests.test_native_neighbor_locator import shared_event
    source,_,_=shared_event(n_cells=cells)
    e=int(target[0]);st=np.array([target[0]-e,target[1]])
    x=np.r_[source.mesh._location_geometry(e,st)[0],0.][None]
    locator=NativeNearbyLocator(source.mesh,saved_chain_neighbours(cells))
    elm,local=locator.locate_batch(x,element_hints=np.array([1]),local_hints=np.array([[.2,.3]]))
    assert elm[0]==e+1
    np.testing.assert_allclose(local[0],st,rtol=0,atol=1e-12)
    assert locator.last_stats['global_fallback_points']==0
    assert locator.last_stats['edge_crossings']==e


def test_saved_zero_neighbor_preserves_negative_lost_id_and_boundary_coordinates():
    from src.models.native_nearby_locator import NativeNearbyLocator
    source,_,_=near_edge_event()
    locator=NativeNearbyLocator(source.mesh,np.zeros((4,2),dtype=np.int32))
    target=np.array([[4.2,.3,0.]])
    elm,st=locator.locate_batch(target,element_hints=np.array([1]),local_hints=np.array([[.5,.3]]))
    assert elm[0]==-1
    assert st[0,0]==pytest.approx(1.)
    assert locator.last_ifail[0]==-1
    assert locator.last_stats['global_fallback_points']==0


def test_negative_neighbor_uses_explicit_existing_global_fallback():
    from src.models.native_nearby_locator import NativeNearbyLocator
    source,_,_=near_edge_event()
    neighbors=saved_chain_neighbours();neighbors[1,0]=-1;neighbors[3,1]=-1
    locator=NativeNearbyLocator(source.mesh,neighbors)
    x=np.r_[source.mesh._location_geometry(1,np.array([.2,.3]))[0],0.][None]
    elm,st=locator.locate_batch(x,element_hints=np.array([1]),local_hints=np.array([[.5,.3]]))
    assert elm[0]==2
    np.testing.assert_allclose(st,[[.2,.3]],rtol=0,atol=1e-10)
    assert locator.last_stats['negative_neighbour_points']==1
    assert locator.last_stats['global_fallback_points']==1


def test_two_events_and_restored_adapter_preserve_history_and_rng():
    import pickle
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    source,state,fields=near_edge_event();before=deepcopy(state)
    nearby=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    first,_=nearby.advance(state,1e-9,fields,fields)
    first.clock.field_start_si=1e-9;first.clock.field_end_si=2e-9
    restored=pickle.loads(pickle.dumps(first))
    continued,_=nearby.advance(first,2e-9,fields,fields)
    resumed,_=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours()).advance(restored,2e-9,fields,fields)
    assert_runtime_exact(continued,resumed)
    assert_runtime_exact(state,before)
    assert np.any(continued.feedback.deltas)
    assert nearby.configuration!=source.configuration


@pytest.mark.parametrize('mutation',['coordinates','size','vertex','indices'])
def test_static_mesh_mutation_fails_without_partial_state(mutation):
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    from tests.test_native_locator_events import assert_runtime_exact
    source,state,fields=near_edge_event();before=deepcopy(state)
    nearby=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    getattr(source.mesh,mutation).flat[0]+=1
    with pytest.raises(ValueError,match='changed'):
        nearby.advance(state,1e-9,fields,fields)
    assert_runtime_exact(state,before)


@pytest.mark.parametrize('bad',[
    np.zeros((2,4),dtype=np.int32),np.zeros((4,2)),
    np.full((4,2),3,dtype=np.int32),np.array([[0,0],[2,0],[0,0],[0,0]])])
def test_invalid_saved_neighbours_are_not_rebuilt_silently(bad):
    from src.models.native_nearby_locator import NativeNearbyLocator
    source,_,_=near_edge_event()
    with pytest.raises(ValueError,match='neighbours'):
        NativeNearbyLocator(source.mesh,bad)


def test_lost_full_source_stops_future_substep_visits():
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    source,state,fields=near_edge_event();fields[1]*=1000;state.cloud.V_prev*=1000
    nearby=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    out,_=nearby.advance(state,1e-9,fields,fields)
    assert out.cloud.i_elm[0]==-2
    assert not out.cloud.alive[0]
    assert nearby.last_stats['lost_points']==1


def test_empty_batch_and_invalid_hint_use_explicit_global_boundary():
    from src.models.native_nearby_locator import NativeNearbyLocator
    source,_,_=near_edge_event()
    locator=NativeNearbyLocator(source.mesh,saved_chain_neighbours())
    elm,st=locator.locate_batch(np.empty((0,3)),element_hints=np.empty(0,dtype=np.int32),local_hints=np.empty((0,2)))
    assert elm.shape==(0,) and st.shape==(0,2)
    x=np.r_[source.mesh._location_geometry(0,np.array([.4,.3]))[0],0.][None]
    elm,st=locator.locate_batch(x,element_hints=np.array([0]),local_hints=np.array([[.1,.1]]))
    assert elm[0]==1
    assert locator.last_stats['invalid_hint_points']==locator.last_stats['global_fallback_points']==1


def test_input_positions_hints_and_neighbours_are_owned_not_mutated():
    from src.models.native_nearby_locator import NativeNearbyLocator
    source,state,_=near_edge_event()
    n=saved_chain_neighbours();locator=NativeNearbyLocator(source.mesh,n)
    x=state.cloud.x.copy();hints=state.cloud.i_elm.copy();st=state.cloud.st.copy()
    expected=tuple(a.copy() for a in (x,hints,st,n))
    locator.locate_batch(x,element_hints=hints,local_hints=st)
    for a,b in zip((x,hints,st,n),expected):np.testing.assert_array_equal(a,b)
    n[:]=0
    np.testing.assert_array_equal(locator.neighbours.T,saved_chain_neighbours())
    assert not locator.neighbours.flags.writeable


def test_nearby_adapter_preserves_coupled_two_events_and_cross_step_backpropagation():
    import torch
    from src.models.native_coupling import NativeMarkerSource
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tests.test_native_production_torch_integration import fixture,control,AnalyticEquilibrium
    model,legacy,shard,runtime=fixture()
    source=NativeMarkerSource(legacy.mesh,atomic=legacy.atomic,
        atomic_version=legacy.configuration['atomic'],potentials_ev=legacy.potentials_ev,
        F0=legacy.F0,diffusion_ju=legacy.diffusion_ju,time_sampling=legacy.time_sampling,
        field_static=legacy.field_static,spatial_backend='batched_v1')
    nearby=NativeNearbyMarkerSource(source,neighbours=np.zeros(source.mesh.vertex.shape,dtype=np.int32))
    model.source_configuration=deepcopy(nearby.configuration)
    runtime.native_context.configuration['model']=model.checkpoint_configuration
    with torch.no_grad():model.field.decoder.weight.fill_(1e-4)
    adapter=TorchRecomputedSourceEvents(model,nearby,shard,equilibrium_provider=AnalyticEquilibrium())
    initial=adapter.initialize(runtime)
    for value in (initial.values,initial.raw_deltas,initial.memory):value.requires_grad_()
    np.random.seed(1931)
    first,_=adapter.advance(initial,control())
    for value in (first.values,first.raw_deltas,first.memory):value.retain_grad()
    second,_=adapter.advance(first,control(1))
    (second.values.square().sum()+second.raw_deltas.square().sum()+second.memory.square().sum()).backward()
    for value in (initial.values,initial.raw_deltas,initial.memory,first.values,first.raw_deltas,first.memory):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert nearby.last_stats['point_visits']>0
