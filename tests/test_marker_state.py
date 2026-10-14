"""Component verification with prescribed fields; NOT self-generated closed loop."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import h5py
import pytest
import torch
from src.models.marker_state import (MarkerCloud, MarkerRuntime, MarkerClock,
    ProjectionFeedback, charge_step, push_velocity, advance_window, execute_startup,
    save_model_rollout, load_model_rollout)
from src.data.marker_checkpoint import initialize_native_checkpoint, read_marker_snapshot


def cloud(n=1, z=1):
    c=MarkerCloud.empty(z, n)
    c.x[:]=[2.,0.,0.]; c.i_elm[:]=1; c.weight[:]=2.
    c.V_prev[:]=[1.,0.,0.]; c.i_life[:]=1
    return c


def runtime():
    return MarkerRuntime(cloud(),ProjectionFeedback(np.zeros((1,5)),np.zeros((1,5)),0.,"test"),
                         MarkerClock(0.,0.,.1,0.,1.))


def advance(s,target,births=None):
    def sample(t,c,ids,previous):
        n=len(ids)
        return dict(velocity=np.tile([1.,0.,0.],(n,1)),ne_si=np.ones(n),
                    scd=np.full((n,1),2.),acd=np.full((n,1),.5),
                    radiation_w_per_atom=np.tile([3.,7.],(n,1)))
    def locate(x):
        return np.ones(len(x),dtype=int),np.zeros((len(x),2))
    def deposit(c,ids,load):
        return load.sum(0,keepdims=True)
    return advance_window(s,target,sample=sample,locate=locate,deposit=deposit,
                          project=lambda x:x,potentials_ev=np.array([10.]),
                          t_norm=1.,central_density=1.,births=births)


def test_independent_jorek_charge_push_and_old_moment_formula():
    # Hand-computed master equation: .7*(1-.2)+.3*.05=.575.
    p=charge_step(np.array([[.7,.3]]),np.array([[2.]]),np.array([[.5]]),np.array([1.]),.1)
    np.testing.assert_allclose(p,[[.575,.425]],rtol=0,atol=1e-15)
    # Upstream uses CURRENT speed with OLD direction (not old vector AB2).
    result=push_velocity(np.array([[2.,0.,.3]]),np.array([[0.,3.,0.]]),np.array([[2.,0.,0.]]),.1)
    np.testing.assert_allclose(result,[[2.3,-.1,.3]],atol=1e-15)
    s=runtime(); out,d=advance(s,.1)
    np.testing.assert_allclose(out.cloud.P_imp,[[.8,.2]])
    # OLD neutral charge -> zero charge moments, nonzero ionization energy.
    np.testing.assert_array_equal(d['moments'][0],[0.,0.,.2])
    assert d['energy'][0] == pytest.approx(.2*2*10*1.602176634e-19)
    assert d['rad'][0] == pytest.approx(3*2*.1)
    assert out.cloud.x[0,0] == pytest.approx(2.1)


def test_window_persistence_and_model_roundtrip(tmp_path):
    from src.models.source_module import AugmentedState
    module=torch.nn.Linear(1,1)
    s=runtime(); first,_=advance(s,.25)  # 2 substeps, .05 residual carried
    assert first.clock.step_rest_time == pytest.approx(.05)
    saved=AugmentedState(torch.ones(1,1,9),torch.ones(1,1,16),torch.ones(1,1,dtype=torch.bool),
                         torch.tensor([[.7]]),1,.25,first,torch.zeros(1,1,9),0.)
    path=tmp_path/'rollout.pt'; save_model_rollout(path,saved,module)
    restored=load_model_rollout(path,module)
    direct,_=advance(first,.5); resumed,_=advance(restored.marker_state,.5)
    # Same platform/dtype, lossless serialization, bitwise equality expected.
    for key in MarkerCloud.array_names():
        np.testing.assert_array_equal(getattr(direct.cloud,key),getattr(resumed.cloud,key))
    np.testing.assert_array_equal(direct.feedback.values,resumed.feedback.values)
    np.testing.assert_array_equal(direct.feedback.deltas,resumed.feedback.deltas)
    np.testing.assert_array_equal(restored.h_residual,saved.h_residual)
    assert direct.clock == resumed.clock
    assert direct.cloud.P_imp[0,1] > first.cloud.P_imp[0,1] > 0
    np.testing.assert_array_equal(first.cloud.V_prev,[[1.,0.,0.]])
    np.testing.assert_allclose(direct.feedback.deltas,direct.feedback.values-first.feedback.values)
    assert direct.cloud.weight[0] == 2
    assert direct.cloud.x[0,0] > first.cloud.x[0,0]
    # Pure transition leaves the input untouched.
    assert s.clock.sim_time_si == 0.


def test_births_reuse_lost_slots_without_resetting_live_cloud():
    s=runtime(); s.cloud=cloud(2); s.cloud.i_elm[1]=-1; s.cloud.i_life[1]=5
    first,_=advance(s,.1)
    born=cloud(); born.weight[:]=3.; born.V_prev[:]=[2.,0.,0.]
    second,d=advance(first,.2,born)
    assert second.cloud.i_life.tolist()==[1,6]
    assert second.cloud.weight.tolist()==[2.,3.]
    assert second.cloud.P_imp[0,1] > second.cloud.P_imp[1,1]
    assert second.cloud.P_imp[1,1] == pytest.approx(.2)
    assert d['moments'][0,1] > 0 and d['moments'][1,1] == 0
    with pytest.raises(ValueError,match='capacity'):
        second.cloud.insert(born)
    # No births on subsequent windows; both live particles continue evolving.
    final,_=advance(second,.4)
    assert final.cloud.alive.sum()==2
    assert (final.cloud.x[:,0]>second.cloud.x[:,0]).all()


def test_restart_zero_projection_and_startup_event_are_distinct():
    s,_=advance(runtime(),.25)
    restarted=s.jorek_restart()
    np.testing.assert_array_equal(restarted.cloud.P_imp,s.cloud.P_imp)
    assert restarted.clock.step_rest_time == 0
    assert restarted.clock.startup_pending
    assert not restarted.feedback.values.any() and not restarted.feedback.deltas.any()
    with pytest.raises(ValueError,match='startup'):
        advance(restarted,.5)
    calls=[]
    def fluid_step(feedback,time):
        assert not feedback.values.any()
        calls.append(('fluid_solve_then_shard_update',time))
        return time,time+.3
    ready=execute_startup(restarted,fluid_step)
    after,_=advance(ready,.5)
    assert calls==[('fluid_solve_then_shard_update',.25)]
    assert after.cloud.P_imp[0,1]>s.cloud.P_imp[0,1]
    continuous,_=advance(s,.5)
    assert continuous.clock.particle_time_si != after.clock.particle_time_si
    with pytest.raises(ValueError,match='pending'):
        execute_startup(ready,fluid_step)


def test_feedback_is_passed_to_particle_sampling_and_deposit_precedes_motion():
    s=runtime(); s.feedback.values[:]=7
    seen=[]
    def sample(t,c,ids,previous):
        seen.append((t,c.x.copy(),previous.values.copy()))
        return dict(velocity=np.tile([1.,0.,0.],(len(ids),1)),ne_si=np.ones(len(ids)),
                    scd=np.zeros((len(ids),1)),acd=np.zeros((len(ids),1)),
                    radiation_w_per_atom=np.zeros((len(ids),2)))
    deposited=[]
    def deposit(c,ids,load):
        deposited.append(c.x.copy()); return load.sum(0,keepdims=True)
    out,_=advance_window(s,.2,sample=sample,locate=lambda x:(np.ones(len(x),int),np.zeros((len(x),2))),
        deposit=deposit,project=lambda x:x,potentials_ev=[10.],t_norm=1,central_density=1)
    assert [a[0] for a in seen]==[0.,.1]
    assert all((a[2]==7).all() for a in seen)
    assert deposited[0][0,0]==2. and deposited[1][0,0]==pytest.approx(2.1)
    assert out.cloud.x[0,0]==pytest.approx(2.2)


def native_fixture(tmp_path):
    from conftest import write_native_frame, write_macroscopic_times
    tn=np.sqrt(4e-7*np.pi*1.66053906660e-27*.8e20)
    fluid=tmp_path/'jorek000011.h5'
    write_native_frame(fluid,index=11,t_now=2.,t_norm=tn,tstep=.5)
    with h5py.File(fluid,'a') as f:
        f['deltas']=np.full(f['values'].shape,.25)
        for k,v in [('central_density',.8),('central_mass',1.)]:
            if k in f: del f[k]
            f[k]=v
    times=tmp_path/'macroscopic_vars.dat'; write_macroscopic_times(times,{10:1.5,11:2.})
    nl=tmp_path/'namelist'; nl.write_text('tstep_n=(/ .5, 1. /)\nnstep_n=(/ 10, 20 /)\ntstep_particles=1d-9\n')
    part=tmp_path/'arbitrary_name.h5'; c=cloud(z=10)
    with h5py.File(part,'w') as f:
        f['time']=tn*1.5
        g=f.create_group('groups/001')
        g['type']=np.bytes_('particle_marker'); g['Z']=10; g['mass']=20.1797; g['adas_suffix']=np.bytes_('96_ne')
        for k in c.array_names(): g['v' if k=='V_prev' else k]=getattr(c,k)
    return dict(fluid_path=fluid,marker_path=part,event_index=11,times_path=times,namelist_path=nl,boundary='jorek_restart')


def test_native_checkpoint_fields_times_units_and_errors(tmp_path):
    kw=native_fixture(tmp_path); cp=initialize_native_checkpoint(**kw)
    assert cp.marker.clock.sim_time_si == pytest.approx(1.5*cp.t_norm)
    assert cp.fluid_time_jorek==2 and cp.marker.clock.startup_pending
    np.testing.assert_array_equal(cp.marker.cloud.V_prev,[[1,0,0]])
    assert cp.marker.cloud.P_imp.shape==(1,11)
    assert cp.marker.feedback.values.shape==(5,4,1,6)
    assert cp.provenance['imported_step_origin']=='native_hdf5_tstep'
    assert cp.provenance['marker']['path'].endswith('arbitrary_name.h5')
    with pytest.raises(ValueError,match='missing marker'):
        initialize_native_checkpoint(**{**kw,'marker_path':None})
    with pytest.raises(ValueError,match='saved clock'):
        initialize_native_checkpoint(**{**kw,'boundary':'continuous'})
    with h5py.File(kw['marker_path'],'a') as f: f['time'][...]=2*cp.t_norm
    with pytest.raises(ValueError,match='internal /time'):
        initialize_native_checkpoint(**kw)
    with h5py.File(kw['marker_path'],'a') as f:
        f['time'][...]=1.5*cp.t_norm; del f['groups/001/v']
    with pytest.raises(ValueError,match='required marker field v'):
        initialize_native_checkpoint(**kw)


REAL=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')
@pytest.mark.skipif(not (REAL/'part04110.h5').exists(),reason='real phaseA data unavailable')
def test_real_phase_a_checkpoint_read():
    cp=initialize_native_checkpoint(REAL/'jorek004110.h5',REAL/'part04110.h5',
        event_index=4110,times_path=REAL/'macroscopic_vars.dat',
        namelist_path=REAL/'jorek_namelist_spitest',boundary='jorek_restart')
    assert len(cp.marker.cloud.weight)==1000000
    assert cp.marker.cloud.alive.sum()>0
    assert cp.marker.cloud.P_imp.shape==(1000000,11)
    with h5py.File(REAL/'jorek004110.h5','r') as f:
        np.testing.assert_array_equal(cp.fluid_deltas, f['deltas'][...])
    from src.data.marker_checkpoint import reference_rollout_from_checkpoint
    rollout = reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm))
    np.testing.assert_array_equal(rollout.fluid_deltas, cp.fluid_deltas)
    # Direct independent raw-field check, includes native v -> V_prev mapping.
    with h5py.File(REAL/'part04110.h5','r') as f:
        ids=np.flatnonzero(cp.marker.cloud.alive)[:10]
        np.testing.assert_array_equal(cp.marker.cloud.V_prev[ids],f['groups/001/v'][ids])
        assert cp.marker.clock.sim_time_si==float(f['time'][0])
    assert cp.marker.clock.sim_time_si/cp.t_norm < cp.fluid_time_jorek
    assert not cp.marker.feedback.values.any()


def test_reference_driver_obeys_startup_and_window_event_order(tmp_path):
    from src.data.marker_checkpoint import reference_rollout_from_checkpoint
    from src.models.marker_state import reference_fluid_event, reference_particle_then_fluid
    cp=initialize_native_checkpoint(**native_fixture(tmp_path))
    state=reference_rollout_from_checkpoint(cp,control_plan_si=(.5*cp.t_norm,cp.t_norm,.25*cp.t_norm))
    calls=[]
    def solve(u,shards,feedback,dt,previous_dt,*,fluid_deltas,time_coefficients):
        calls.append(('solve',u.copy(),feedback.values.copy(),dt))
        from src.models.marker_state import FluidSolveResult
        return FluidSolveResult(u+1,np.ones_like(u))
    def shards(p,u,time,dt):
        calls.append(('shards',u.copy(),time))
        return p
    def particle(m,target,old,new,p):
        calls.append(('particle',old.copy(),new.copy()))
        assert target==m.clock.field_end_si
        m.clock.sim_time_si=target
        m.feedback.values[:]=2.
        return m,{'component_only':True}
    with pytest.raises(ValueError,match='startup'):
        reference_particle_then_fluid(state,particle_step=particle,solve_fluid=solve,update_shards=shards)
    ready=reference_fluid_event(state,solve_fluid=solve,update_shards=shards)
    assert not calls[0][2].any()
    assert calls[1][2]==cp.marker.clock.sim_time_si
    np.testing.assert_array_equal(calls[0][1],calls[1][1])
    result,_=reference_particle_then_fluid(ready,particle_step=particle,solve_fluid=solve,update_shards=shards)
    assert [c[0] for c in calls]==['solve','shards','particle','solve','shards']
    assert (calls[3][2]==2).all()
    assert result.plan_cursor==2
    assert result.fluid_time_si>ready.fluid_time_si


def test_continuous_native_checkpoint_needs_and_keeps_history(tmp_path):
    kw=native_fixture(tmp_path); native=initialize_native_checkpoint(**kw)
    tn=native.t_norm
    from conftest import write_native_frame,write_native_projection
    previous=tmp_path/'previous.h5'
    write_native_frame(previous,index=10,t_now=1.5,t_norm=tn,tstep=.5)
    projection=tmp_path/'feedback.h5'; write_native_projection(projection,t_seconds=1.5*tn)
    clock=MarkerClock(1.5*tn,1e-10,1e-9,1.5*tn,2.*tn)
    delta=np.ones((5,4,1,6))*.2
    cp=initialize_native_checkpoint(**{**kw,'boundary':'continuous'},clock=clock,
        feedback_path=projection,feedback_deltas=delta,previous_fluid_path=previous)
    assert cp.marker.clock.step_rest_time==1e-10
    assert not cp.marker.clock.startup_pending
    assert cp.marker.feedback.values.any()
    np.testing.assert_array_equal(cp.marker.feedback.deltas,delta)
