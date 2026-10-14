"""Public prepare/sample and actual birth integration; no training."""
from copy import deepcopy
from math import pi

import numpy as np
import pytest

from src.models.native_birth_positions import NativeTabulatedBirthSampler
from tests.test_native_production_birth import birth_fixture
from tests.test_native_production_density import source_fixture


def prepare(*, cells=8, dv=.001, geometry='axisymmetric_n0', phi_cells=None, shards=None):
    kernel, s = source_fixture()
    if shards is not None:
        s = shards
    sampler = NativeTabulatedBirthSampler(geometry=geometry,
        n_tor=1 if geometry=='axisymmetric_n0' else 3,
        cells_per_width=cells, phi_cells=phi_cells)
    norm=float(np.sum(s['ablation_atoms_s']))*kernel.mesh.units.t_norm_si/kernel.mesh.units.n_norm_m3
    table = sampler.prepare(kernel,s,time_si=1.,bounds=[[2.35,.35],[2.65,.65]],dv=dv,norm=norm)
    return sampler, table, kernel, s


def independent_vertex_masses(table, kernel, s, dv):
    """Literal Gaussian/analytic-volume formula, not sampler or density query."""
    r=table.r_edges;z=table.z_edges
    widths=np.maximum(s['radius']*kernel.ratio,kernel.minimum)
    result=np.zeros((len(r),len(z),len(table.phi_edges)))
    total=np.sum(s['ablation_atoms_s'])
    for j in range(len(s['radius'])):
        if s['radius'][j]<=0:continue
        center=s['positions'][j]
        pol=np.exp(-((r[:,None]-center[0])**2+(z[None,:]-center[1])**2)/widths[j]**2)
        # Literal mass source (not number source); then total-source gate.
        if result.shape[-1]==2:
            volume=pi*center[0]*widths[j]**2*kernel.toroidal_integral
            result+=s['ablation_atoms_s'][j]/total*pol[:,:,None]/volume
        else:
            phi=table.phi_edges
            angle=np.abs((phi-center[2]+pi)%(2*pi)-pi)
            tor=np.exp(-(angle/kernel.deltaphi)**2)
            volume=pi*center[0]*widths[j]**2*kernel.toroidal_integral
            result+=s['ablation_atoms_s'][j]/total*pol[:,:,None]*tor[None,None,:]/volume
    result*=20/kernel.mesh.units.central_mass
    if result.shape[-1]==2:
        from scipy.integrate import quad
        # Independent adaptive integral of the continuous total gate; source
        # centers in these fixtures share phi. Mean BEFORE clip is incorrect.
        peak=result[:,:,0]*dv
        def angular_integral(p):
            crossing=min(pi,kernel.deltaphi*np.sqrt(np.log(max(p,1.))))
            return quad(lambda angle:min(p*np.exp(-(angle/kernel.deltaphi)**2),1.),
                        0,pi,points=[crossing] if 0<crossing<pi else None)[0]/pi
        values=np.array([angular_integral(p) for p in peak.ravel()]).reshape(peak.shape)
        acceptance=np.repeat(values[:,:,None],2,axis=2).astype(np.float32).astype(float)
    else:acceptance=np.maximum(result*dv,0).astype(np.float32).astype(float)
    volume=.5*np.diff(table.r_edges**2)[:,None,None]*np.diff(table.z_edges)[None,:,None]*np.diff(table.phi_edges)[None,None,:]
    clipped=np.minimum(acceptance,1)
    nr,nz,np_=table.probabilities.shape
    average=sum(clipped[i:i+nr,j:j+nz,k:k+np_] for i in (0,1) for j in (0,1) for k in (0,1))/8
    mass=average*volume
    return mass/mass.sum()


@pytest.mark.parametrize('dv',[.001,.2])
def test_total_source_plateau_matches_independent_literal_formula(dv):
    _,s=source_fixture()
    s={k:np.repeat(v,2,axis=0) if isinstance(v,np.ndarray) else v for k,v in s.items()}
    s['positions'][1,:2]=[2.53,.47]
    _,table,kernel,s=prepare(dv=dv,shards=s)
    np.testing.assert_allclose(table.probabilities,independent_vertex_masses(table,kernel,s,dv),rtol=3e-7,atol=1e-12)
    assert table.diagnostics['max_acceptance_before_clip']>1 if dv==.2 else table.diagnostics['max_acceptance_before_clip']<1
    assert table.cdf[-1]==1 and np.all(np.diff(table.cdf)>=0)


def test_saturation_is_of_sum_not_sum_of_clipped_shards():
    _,s=source_fixture()
    s={k:np.repeat(v,2,axis=0) if isinstance(v,np.ndarray) else v for k,v in s.items()}
    _,table,_,_=prepare(dv=.3,shards=s)
    # Identical components overlap; saturation must be evaluated on their sum.
    assert table.diagnostics['saturated_cells']==0  # Narrow phi retains unsaturated angular tails.
    assert table.diagnostics['max_acceptance_before_clip']>1
    assert table.acceptance_nodes.max()<1


def test_uniform_plateau_samples_exact_cylindrical_volume_not_uniform_R():
    _,table,_,_=prepare(dv=1e30)
    assert table.diagnostics['saturated_probability']==pytest.approx(1)
    x=table.sample(60000,rng=np.random.default_rng(417))
    r0,r1=table.r_edges[[0,-1]]
    expected=2/3*(r1**3-r0**3)/(r1**2-r0**2)
    assert x[:,0].mean()==pytest.approx(expected,abs=.001)
    assert x[:,1].mean()==pytest.approx(.5,abs=.001)
    assert x[:,2].mean()==pytest.approx(pi,abs=.025)
    assert np.all((x[:,0]>=r0)&(x[:,0]<r1))


def test_sample_histogram_and_rng_replay_match_table():
    _,table,_,_=prepare(cells=2,dv=.15)
    points=table.sample(100000,rng=np.random.default_rng(7))
    actual=np.histogram2d(points[:,0],points[:,1],bins=(table.r_edges,table.z_edges))[0]/len(points)
    expected=table.probabilities[:,:,0]
    assert np.sum(np.abs(actual-expected))<.025
    np.random.seed(47);saved=deepcopy(np.random.get_state())
    first=table.sample(15);after=deepcopy(np.random.get_state())
    np.random.set_state(saved);second=table.sample(15)
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(after[1],np.random.get_state()[1])
    assert after[2:]==np.random.get_state()[2:]
    assert table.sample(0).shape==(0,3)


def test_n0_has_no_shard_phi_dependence_and_continuum_not_one_plane_average():
    _,table,kernel,s=prepare()
    shifted=deepcopy(s);shifted['positions'][:,2]+=2.3
    _,other,_,_=prepare(shards=shifted)
    np.testing.assert_array_equal(table.probabilities,other.probabilities)
    assert kernel.angles.shape==(1,)
    assert table.diagnostics['max_acceptance_before_clip']==other.diagnostics['max_acceptance_before_clip']


def test_future_3d_periodic_kernel_and_total_platform():
    _,table,kernel,s=prepare(geometry='periodic_phi',phi_cells=32,dv=.02)
    np.testing.assert_allclose(table.probabilities,independent_vertex_masses(table,kernel,s,.02),rtol=3e-7,atol=1e-12)
    angular=table.probabilities.sum(axis=(0,1))
    assert angular[0]==pytest.approx(angular[-1],rel=1e-14)
    assert angular[0]>1000*angular[len(angular)//2]
    points=table.sample(20000,rng=np.random.default_rng(77))
    # Saturation broadens phi; check the table moment, not an old Gaussian >.7.
    nodes=table.acceptance_nodes
    rz=.5*np.diff(table.r_edges**2)[:,None]*np.diff(table.z_edges)[None,:]
    vertices=np.sum(rz[:,:,None]*(nodes[:-1,:-1]+nodes[1:,:-1]+nodes[:-1,1:]+nodes[1:,1:])/4,axis=(0,1))
    x,w=np.polynomial.legendre.leggauss(16)
    phi=(table.phi_edges[:-1,None]+table.phi_edges[1:,None])/2+np.diff(table.phi_edges)[:,None]/2*x
    density=vertices[:-1,None]*(1-x)/2+vertices[1:,None]*(1+x)/2
    numerator=np.sum(density*np.cos(phi)*w*np.diff(table.phi_edges)[:,None]/2)
    denominator=np.sum((vertices[:-1]+vertices[1:])/2*np.diff(table.phi_edges))
    assert np.cos(points[:,2]).mean()==pytest.approx(numerator/denominator,abs=.008)


def test_refinement_reduces_continuous_source_bin_probability_error():
    tables=[prepare(cells=k)[1] for k in (2,4,8,16)]
    fine=tables[-1].probabilities[:,:,0]
    errors=[]
    for k,table in zip((2,4,8),tables[:-1]):
        n=16//k;nr,nz,_=table.probabilities.shape
        truth=fine.reshape(nr,n,nz,n).sum(axis=(1,3))
        errors.append(np.sum(np.abs(table.probabilities[:,:,0]-truth)))
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.01


def test_prepared_distribution_is_owned_readonly_snapshot():
    _,table,kernel,s=prepare()
    before=table.sample(30,rng=np.random.default_rng(1))
    s['positions'][:]=100;s['ablation_atoms_s'][:]=0
    np.testing.assert_array_equal(before,table.sample(30,rng=np.random.default_rng(1)))
    for array in (table.r_edges,table.z_edges,table.phi_edges,table.probabilities,table.cdf,table.acceptance_nodes):
        assert not array.flags.writeable


def test_n0_marginal_integrates_total_plateau_before_angular_averaging():
    from scipy.integrate import quad
    _,table,kernel,s=prepare(dv=.01)
    ir=int(np.argmin(abs(table.r_edges-2.5)));iz=int(np.argmin(abs(table.z_edges-.5)))
    point=np.array([[table.r_edges[ir],table.z_edges[iz],0.]])
    norm=s['ablation_atoms_s'].sum()*kernel.mesh.units.t_norm_si/kernel.mesh.units.n_norm_m3
    peak=kernel.rates_at(s,point,time_si=1.)[0]*.01/norm
    assert peak>1
    expected=quad(lambda phi:min(peak*np.exp(-(phi/kernel.deltaphi)**2),1.),0,pi,
        points=[kernel.deltaphi*np.sqrt(np.log(peak))])[0]/pi
    assert table.acceptance_nodes[ir,iz,0]==pytest.approx(expected,abs=5e-8)
    wrong=min(peak*kernel.toroidal_integral/(2*pi),1.)
    assert abs(expected-wrong)>.1


def test_periodic_phi_births_are_explicitly_supported_with_H1_native_fields():
    from src.models.native_source_birth import NativeProductionBirths
    old,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    sampler=NativeTabulatedBirthSampler(geometry='periodic_phi',n_tor=1,phi_cells=32)
    births=NativeProductionBirths(old.kernel,physical_particles=2e13,n_particle_slots=2,
        n_cpu=1,rank=0,R_geo=2.5,n_plane=1,max_proposals=100,position_sampler=sampler)
    np.random.seed(914)
    cloud=births.births(s,marker,2e-9,u,u,source)
    assert len(cloud.weight)==2 and cloud.alive.all()
    assert births.configuration['position_sampler']['geometry']=='periodic_phi'


def test_old_mass_contract_configuration_fails_before_rng_consumption():
    births,s,source,marker,u= birth_fixture();marker.cloud.i_elm[:]=0
    births.configuration.pop('acceptance')
    before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='native-mass version'):
        births.births(s,marker,2e-9,u,u,source)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]


@pytest.mark.parametrize('kwargs',[
    dict(geometry='axisymmetric_n0',n_tor=2),dict(geometry='periodic_phi',n_tor=1,phi_cells=1),
    dict(geometry='axisymmetric_n0',n_tor=1,phi_cells=8),
    dict(geometry='periodic_phi',n_tor=3),dict(geometry='guess',n_tor=1),
    dict(geometry='axisymmetric_n0',n_tor=1,cells_per_width=1)])
def test_geometry_and_resolution_must_be_explicit(kwargs):
    with pytest.raises(ValueError):NativeTabulatedBirthSampler(**kwargs)


def test_budget_zero_source_and_invalid_bounds_fail_without_rng_consumption():
    kernel,s=source_fixture()
    sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,max_cells=2)
    before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='cell budget'):
        sampler.prepare(kernel,s,time_si=1.,bounds=[[2.35,.35],[2.65,.65]],dv=.1,norm=1.)
    s['ablation_atoms_s'][:]=0
    with pytest.raises(ValueError,match='zero active source'):
        sampler.prepare(kernel,s,time_si=1.,bounds=[[2.35,.35],[2.65,.65]],dv=.1,norm=1.)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    with pytest.raises(ValueError,match='bounds'):
        sampler.prepare(kernel,s,time_si=1.,bounds=[[-1,0],[1,1]],dv=.1,norm=1.)


def tabulated_birth_fixture(max_proposals=100):
    from src.models.native_source_birth import NativeProductionBirths
    old,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,cells_per_width=8)
    births=NativeProductionBirths(old.kernel,physical_particles=2e13,n_particle_slots=2,
        n_cpu=1,rank=0,R_geo=2.5,n_plane=1,max_proposals=max_proposals,position_sampler=sampler)
    return births,s,source,marker,u,old


def test_actual_births_keep_count_weight_fields_and_distinct_checkpoint_configuration():
    from src.models.native_state import configuration_digest
    births,s,source,marker,u,old=tabulated_birth_fixture()
    np.random.seed(201);saved=deepcopy(np.random.get_state())
    cloud=births.births(s,marker,2e-9,u,u,source)
    assert len(cloud.weight)==2 and cloud.alive.all()
    np.testing.assert_array_equal(cloud.weight,[12500000047104.,12500000047104.])
    assert np.isfinite(cloud.V_prev).all()
    np.random.set_state(saved)
    recovered=births.births(s,marker,2e-9,u,u,source)
    for key in cloud.array_names():np.testing.assert_array_equal(getattr(cloud,key),getattr(recovered,key))
    assert configuration_digest(births.configuration)!=configuration_digest(old.configuration)
    assert old.position_sampler is None and 'position_sampler' not in old.configuration


def test_debug_seed_replays_births_without_reseeding_each_call():
    births,s,source,marker,u,_=tabulated_birth_fixture()
    saved=deepcopy(np.random.get_state())
    try:
        np.random.seed(491)
        first=births.births(s,marker,2e-9,u,u,source)
        # The next call continues the random stream instead of starting over.
        second=births.births(s,marker,2e-9,u,u,source)
        assert not np.array_equal(first.x,second.x)
        np.random.seed(491)
        replay=births.births(s,marker,2e-9,u,u,source)
        for name in first.array_names():
            np.testing.assert_array_equal(getattr(first,name),getattr(replay,name))
        np.random.seed(492)
        different=births.births(s,marker,2e-9,u,u,source)
        assert not np.array_equal(first.x,different.x)
    finally:
        np.random.set_state(saved)


def test_mesh_rejection_is_unchanged_and_failure_restores_rng(monkeypatch):
    births,s,source,marker,u,_=tabulated_birth_fixture(max_proposals=2)
    monkeypatch.setattr(births.kernel.mesh,'locate',lambda x:(np.zeros(len(x),int),np.zeros((len(x),2))))
    np.random.seed(991);before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='mesh proposal budget'):
        births.births(s,marker,2e-9,u,u,source)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    assert not marker.cloud.alive.any()


def test_full_marker_window_accepts_new_births_without_changing_source_method():
    births,s,source,marker,u,_=tabulated_birth_fixture()
    np.random.seed(912)
    cloud=births.births(s,marker,2e-9,u,u,source)
    out,diag=source.advance(marker,2e-9,u,u,births=cloud)
    assert out.cloud.alive.all() and np.isfinite(out.feedback.values).all()
    assert diag['n_steps']>0
    np.testing.assert_array_equal(marker.cloud.i_elm,[0,0])


@pytest.mark.parametrize('mutation',['control','metadata','replace','remove'])
def test_bound_birth_configuration_cannot_change_silently(mutation):
    births,s,source,marker,u,_=tabulated_birth_fixture()
    if mutation=='control':births.position_sampler.cells_per_width=16
    elif mutation=='metadata':births.position_sampler.configuration['n_tor']=3
    elif mutation=='replace':births.position_sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,cells_per_width=16)
    else:births.position_sampler=None
    before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='configur|missing'):
        births.births(s,marker,2e-9,u,u,source)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]


def test_invalid_rng_and_counts_are_rejected_by_public_sample():
    _,table,_,_=prepare()
    for count in (-1,1.5,True):
        with pytest.raises(ValueError,match='count'):table.sample(count)
    class BadRng:
        def random(self,shape):return np.ones(shape)
    with pytest.raises(ValueError,match='uniforms'):table.sample(2,rng=BadRng())


def test_3d_position_mode_not_silently_used_with_H1_coupled_fields():
    from src.models.native_source_birth import NativeProductionBirths
    old,_,_,_,_=birth_fixture()
    with pytest.raises(ValueError,match='currently require H=1'):
        NativeProductionBirths(old.kernel,physical_particles=2e13,n_particle_slots=2,n_cpu=1,rank=0,
            R_geo=2.5,n_plane=1,max_proposals=100,position_sampler=NativeTabulatedBirthSampler(
                geometry='periodic_phi',n_tor=3,phi_cells=16))


def test_tabulated_births_still_use_bound_saved_raw_history_for_velocity():
    from src.models.native_coupling import NativeMarkerSource
    from src.models.native_source_time import NativeSourceTimeSampler
    births,s,source,marker,u,_=tabulated_birth_fixture()
    marker.clock.sim_time_si=.5e-9;marker.clock.residual_si=.2e-9;marker.feedback.time_si=.5e-9
    source=NativeMarkerSource(source.mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=0.,
        time_sampling='jorek_raw_deltas_v1',field_static=False)
    raw=np.zeros_like(u);raw[6,0]=.0004
    sampler=NativeSourceTimeSampler(u,raw,time_prev_si=0.,time_now_si=1e-9,tstep_si=1e-9,static=False)
    previous=u.copy();previous[6]=0
    cloud=births.births(s,marker,1.5e-9,previous,u,source,time_sampler=sampler)
    np.testing.assert_allclose(cloud.V_prev[:,2],-.0008/cloud.x[:,0]/source.mesh.units.t_norm_si,rtol=2e-14)


def test_closed_form_conditional_inverses_follow_linear_curve_inside_cell():
    from src.models.native_birth_positions import PreparedBirthPositions
    # Positive separable density (1+u)*(1+v), constant phi, one volume cell.
    nodes=np.array([[[.125,.125],[.25,.25]],[[.25,.25],[.5,.5]]])
    table=PreparedBirthPositions(np.array([1.,2.]),np.array([0.,1.]),np.array([0.,2*pi]),
        np.ones((1,1,1)),np.ones(1),nodes,{})
    class FixedRng:
        def random(self,shape):return np.array([[.5,.2,.7,.4],[.1,0.,0.,0.]])
    x=table.sample(2,rng=FixedRng())
    expected_u=np.sqrt(1+3*.2)-1;expected_v=np.sqrt(1+3*.7)-1
    np.testing.assert_allclose(x[0],[np.sqrt(1+3*expected_u),expected_v,.4*2*pi],rtol=2e-15)
    np.testing.assert_array_equal(x[1],[1.,0.,0.])


def test_tabulated_birth_mode_two_coupled_events_checkpoint_and_legacy_rejection(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_source_production import NativeProductionShardModel
    from tests.test_native_production_events import production_fixture,event
    model,source,old,state=production_fixture()
    legacy_model=deepcopy(model);legacy_context=deepcopy(state.native_context)
    birth_config=old.birth.configuration
    birth=NativeProductionBirths(old.density,
        **{k:birth_config[k] for k in ('physical_particles','n_particle_slots','n_cpu','rank','R_geo','n_plane','max_proposals')},
        position_sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1))
    shard=NativeProductionShardModel(old.dynamics,old.density,birth,n_flux=old.n_flux,n_table=old.n_table)
    model.shard_configuration=deepcopy(shard.configuration)
    state.native_context.configuration['model']=model.checkpoint_configuration
    np.random.seed(1931)
    first,_=event(state,model,source,shard)
    path=tmp_path/'tabulated-production.pt';save_model_rollout(path,first,model)
    second,_=event(first,model,source,shard);after=deepcopy(np.random.get_state())
    fresh=deepcopy(model)
    restored=load_model_rollout(path,fresh,native_context=state.native_context)
    resumed,_=event(restored,fresh,deepcopy(source),deepcopy(shard))
    assert second.marker.cloud.alive.sum()==2
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(second,name),getattr(resumed,name))
    for name in second.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(second.marker.cloud,name),getattr(resumed.marker.cloud,name))
    for name in ('values','deltas'):
        np.testing.assert_array_equal(getattr(second.marker.feedback,name),getattr(resumed.marker.feedback,name))
    for name in ('positions','velocities','radius','ablation_atoms_s','source_volume_m3'):
        np.testing.assert_array_equal(second.shards[name],resumed.shards[name])
    assert second.marker.clock==resumed.marker.clock
    np.testing.assert_array_equal(after[1],np.random.get_state()[1]);assert after[2:]==np.random.get_state()[2:]
    with pytest.raises(ValueError,match='config'):
        load_model_rollout(path,legacy_model,native_context=legacy_context)
    # Same new table/architecture, but old number-converted acceptance must
    # fail checkpoint signature checking before restoring any tensor or RNG.
    old_scale=deepcopy(model)
    old_scale.shard_configuration['birth'].pop('acceptance')
    old_scale.shard_configuration['birth']['version']='explicit-tabulated-n0-SPI-birth-v1'
    before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='config'):
        load_model_rollout(path,old_scale,native_context=state.native_context)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1]);assert before[2:]==np.random.get_state()[2:]


def test_3d_nonseparable_cell_conditional_sampling_preserves_all_marginals():
    from src.models.native_birth_positions import PreparedBirthPositions
    nodes=np.array([[[.1,.2],[.3,.1]],[[.4,.2],[.9,.6]]])
    table=PreparedBirthPositions(np.array([1.,2.]),np.array([0.,1.]),np.array([0.,2*pi]),
        np.ones((1,1,1)),np.ones(1),nodes,{})
    points=table.sample(80000,rng=np.random.default_rng(1932))
    coordinates=np.column_stack(((points[:,0]**2-1)/3,points[:,1],points[:,2]/(2*pi)))
    for axis in range(3):
        marginal=np.moveaxis(nodes,axis,0).mean(axis=(1,2))
        expected=(marginal[0]+2*marginal[1])/(3*marginal.sum())
        assert coordinates[:,axis].mean()==pytest.approx(expected,abs=.003)
