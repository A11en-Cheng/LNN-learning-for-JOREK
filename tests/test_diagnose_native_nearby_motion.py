"""Observation must not alter the public source trajectory or its inputs."""
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_locator_events import assert_runtime_exact
from tests.test_native_nearby_locator import saved_chain_neighbours
from tests.test_native_relocation_work import near_edge_event
from tools.diagnose_native_nearby_motion import concentration, refine_baseline_mesh, select_cloud, traced_advance


@pytest.mark.parametrize('nearby',[False,True])
def test_tracing_returns_actual_callbacks_without_perturbing_state(nearby):
    source,state,fields=near_edge_event(); before=deepcopy(state)
    if nearby:
        from src.models.native_nearby_locator import NativeNearbyMarkerSource
        source=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    expected,_=source.advance(state,1e-9,fields,fields)
    actual,diag,trace=traced_advance(source,state,1e-9,fields,fields,None)
    assert_runtime_exact(actual,expected)
    assert_runtime_exact(state,before)
    assert len(trace)==diag['n_steps']==1
    row=trace[0];detail=row['detail']
    reconstructed=detail['exb']+detail['parallel']
    reconstructed[:,:2]+=detail['diffusion']
    np.testing.assert_array_equal(reconstructed,row['velocity'])
    np.testing.assert_array_equal(row['next_x'],actual.cloud.x[row['ids']])
    np.testing.assert_array_equal(row['V_prev'],state.cloud.V_prev[row['ids']])


def test_selected_cloud_owns_arrays_and_preserves_history():
    _,state,_=near_edge_event(); selected=select_cloud(state,np.array([0]))
    for key in state.cloud.array_names():
        np.testing.assert_array_equal(getattr(selected.cloud,key),getattr(state.cloud,key)[[0]])
    assert selected.clock==state.clock
    selected.cloud.V_prev[0,0]+=1
    assert selected.cloud.V_prev[0,0]!=state.cloud.V_prev[0,0]


def test_selection_keeps_per_slot_transport_history_and_rejects_fake_import_interval():
    from src.models.marker_transport import LocalSubcycling,initialize_transport_history
    from tools.diagnose_native_nearby_motion import replay
    _,state,_=near_edge_event()
    state=initialize_transport_history(state,LocalSubcycling(),previous_dt_si=state.clock.particle_dt_si)
    selected=select_cloud(state,np.array([0]))
    selected.transport_history.validate(selected.cloud)
    assert len(selected.transport_history.level)==1
    selected.transport_history.level[0]=6
    assert state.transport_history.level[0]==0
    with pytest.raises(ValueError,match='original native velocity history interval'):
        replay({},local_subcycling=True,dt_divisor=2)


def test_increment_metric_reports_error_concentration_not_background_state():
    _,state,_=near_edge_event(); reference=deepcopy(state);actual=deepcopy(state)
    reference.cloud.x[0]+=np.array([1.,0.,0.])
    actual.cloud.x[0]+=np.array([1.1,0.,0.])
    metric=concentration(reference,actual,state,'x',state.cloud.alive)
    assert metric['relative_l2']==pytest.approx(.1)
    assert metric['error_energy_fractions']['1']==pytest.approx(1.)


def test_diagnostic_refinement_preserves_cell_and_reduces_inverse_residual():
    source,_,_=near_edge_event();mesh=source.mesh
    local=np.array([[.5,.3]])
    target=np.r_[mesh._location_geometry(0,local[0])[0],0.][None]
    hints=dict(element_hints=np.array([1]),local_hints=local+1e-11)
    old,st=mesh.locate_batch(target,**hints)
    assert np.linalg.norm(mesh._location_geometry(0,st[0])[0]-target[0,:2])>1e-14
    actual,new=refine_baseline_mesh(mesh).locate_batch(target,**hints)
    np.testing.assert_array_equal(actual,old)
    assert np.linalg.norm(mesh._location_geometry(0,new[0])[0]-target[0,:2])<=1e-14
    assert 'locate_batch' not in mesh.__dict__


def test_refinement_rejects_out_of_domain_false_acceptance(monkeypatch):
    source,_,_=near_edge_event();mesh=source.mesh
    target=np.r_[mesh._location_geometry(1,np.array([.5,.3]))[0],0.][None]
    monkeypatch.setattr(mesh,'locate_batch',lambda *args,**kwargs:(np.array([1]),np.array([[.9,.3]])))
    with pytest.raises(ValueError,match='left accepted element'):
        refine_baseline_mesh(mesh).locate_batch(target)


def test_summary_relative_does_not_invent_zero_denominator_precision():
    from tools.analyze_native_nearby_motion import relative
    assert relative([1.],[0.]) is None
    assert relative([12.],[12.],origin=[10.])==0.
