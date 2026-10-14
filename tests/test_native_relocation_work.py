"""Deterministic work-budget repro at the complete marker source seam."""
import cProfile
import pstats
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_neighbor_locator import shared_event
from tests.native_relocation_reference import reference_locator
from tests.test_native_locator_events import assert_runtime_exact


def near_edge_event():
    source,state,fields=shared_event()
    state.cloud.st[0]=[.999,.2]
    state.cloud.x[0,:2]=source.mesh.point(0,state.cloud.st[0],source.mesh.coordinates[:1])[2]
    state.cloud.V_prev[0,0]=state.cloud.x[0,0]*5e5
    fields[1]=-5e5*source.mesh.units.t_norm_si*source.mesh.coordinates[1]
    return source,state,fields


def test_complete_source_near_edge_relocation_work_budget():
    source,state,fields=near_edge_event()
    profile=cProfile.Profile()
    profile.enable()
    result,_=source.advance(state,1e-9,fields,fields)
    profile.disable()
    assert result.cloud.i_elm[0]==2
    calls=sum(value[1] for key,value in pstats.Stats(profile).stats.items()
              if key[2] in ('_location_geometry','_location_geometries'))
    # Avoid a flaky wall-clock threshold: native geometry work is deterministic.
    # Trace minimization found 13 distinct required geometries: 9 hint
    # trials and 4 in the selected next cell. The original did 22 (9 repeats).
    assert calls<=13, f'near-edge relocation repeated geometry {calls} times (budget 13)'


def test_measured_original_work_and_complete_state_exact():
    source,state,fields=near_edge_event()
    before=deepcopy(state)
    profile=cProfile.Profile()
    with reference_locator():
        profile.enable()
        expected,old_diag=source.advance(state,1e-9,fields,fields)
        profile.disable()
    calls=sum(v[1] for key,v in pstats.Stats(profile).stats.items()
              if key[2] in ('_location_geometry','_location_geometries'))
    assert calls==22
    actual,new_diag=source.advance(state,1e-9,fields,fields)
    assert_runtime_exact(actual,expected)
    assert_runtime_exact(state,before)
    for key in ('energy','rad','moments','n_steps','gather_time_seconds'):
        np.testing.assert_array_equal(new_diag[key],old_diag[key])


def test_native_scalar_and_batch_geometry_are_bitwise_identical():
    source,_,_=shared_event(n_cells=5)
    rng=np.random.default_rng(779)
    e=rng.integers(0,5,401)
    st=rng.uniform(0,1,(len(e),2))
    st[:4]=[[0,0],[1,1],[0,.7],[1,.3]]
    xyz,jac=source.mesh._location_geometries(e,st)
    for i,(element,local) in enumerate(zip(e,st)):
        a,b=source.mesh._location_geometry(int(element),local)
        np.testing.assert_array_equal(a,xyz[i])
        np.testing.assert_array_equal(b,jac[i])


@pytest.mark.parametrize('dtype',[np.float64,np.float32,np.int32])
def test_bulk_mixed_candidates_and_hint_dtypes_match_pinned_reference(dtype):
    source,_,_=shared_event(n_cells=5)
    mesh=source.mesh;rng=np.random.default_rng(831)
    current=rng.integers(0,5,301);local=rng.uniform(0,1,(len(current),2))
    positions=np.zeros((len(current),3))
    positions[:,:2]=mesh._location_geometries(current,local)[0]
    hints=rng.integers(0,7,len(current));st=rng.uniform(0,1,(len(current),2)).astype(dtype)
    with reference_locator():
        expected=mesh.locate_batch(positions,element_hints=hints,local_hints=st)
    actual=mesh.locate_batch(positions,element_hints=hints,local_hints=st)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_public_scalar_override_is_not_bypassed():
    source,state,fields=near_edge_event()
    original=source.mesh.locate;calls=[]
    def observed(*args,**kwargs):
        calls.append(len(args[0]))
        return original(*args,**kwargs)
    source.mesh.locate=observed
    out,_=source.advance(state,1e-9,fields,fields)
    assert out.cloud.i_elm[0]==2
    assert calls==[1]


def test_two_events_after_geometry_change_do_not_reuse_previous_call_cache():
    source,state,fields=near_edge_event()
    out,_=source.advance(state,1e-9,fields,fields)
    source.mesh.coordinates[1]*=1.01
    out.clock.field_start_si=1e-9;out.clock.field_end_si=2e-9
    with reference_locator():expected,_=source.advance(out,2e-9,fields,fields)
    actual,_=source.advance(out,2e-9,fields,fields)
    assert_runtime_exact(actual,expected)
