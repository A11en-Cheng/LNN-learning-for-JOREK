"""H1 projection at the source seam; independent polynomial energy oracle."""
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture
from src.models.native_coupling import NativeSourceMesh


def configuration(mesh, **updates):
    args=dict(filter_perp=1e-4,filter_hyper=1e-8,filter_parallel=.01,F0=-2.,
        psi=mesh.coordinates[1].copy(),field_time_si=0.,field_provenance='fixture_initial_field',
        boundary=np.zeros(mesh.n_nodes,dtype=int),axis_nodes=np.zeros(mesh.n_nodes,dtype=bool),
        fix_axis_nodes=False,apply_dirichlet=True)
    args.update(updates)
    return args


def filtered(mesh, **updates):
    return NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,
        projection='sparse',projection_n0=configuration(mesh,**updates))


def test_source_projection_n0_filter_energy_has_analytic_factors():
    mesh,_,_,_=native_fixture()
    model=filtered(mesh)
    # Trial R on rectangle [2,3]x[0,1], psi=Z, F0=-2.
    # int R^3=65/4; grad=1; cylindrical Laplacian R=1/R;
    # Bgrad R=1/R, BB2=5/R (literal inspected source), full matrix has 2pi.
    coeff=mesh.coordinates[0,:,0,:].reshape(-1)
    expected=65/4+1e-4*2.5+1e-8*np.log(1.5)+.01/5
    assert coeff@model.mass@coeff==pytest.approx(2*np.pi*expected,rel=2e-12)
    rhs=(model.mass@coeff)[None]
    actual=model.project(rhs)
    np.testing.assert_allclose(actual[0,:,0,:],mesh.coordinates[0,:,0,:],atol=2e-10,rtol=0)
    assert model.configuration['projection']=='jorek_inspected_n0_filtered'
    # snapshot retained; no silent per-event magnetic rebuild
    assert model.configuration['projection_n0']['field_time_si']==0.


def test_projection_boundary_if_elseif_priority_and_shared_penalties():
    mesh,_,_,_=native_fixture()
    base=filtered(mesh,apply_dirichlet=False)
    # 3 and 9 take the FIRST arm (degrees1,3), not union with degrees1,2.
    model=filtered(mesh,boundary=np.array([3,9,1,2]))
    difference=(model.mass-base.mass).toarray()
    expected=np.zeros(16)
    for node,degrees in enumerate(((0,2),(0,2),(0,1),(0,2))):
        for degree in degrees:expected[mesh.inverse[degree,node]]+=1e12
    np.testing.assert_allclose(np.diag(difference),expected,rtol=2e-16,atol=1e-10)
    np.testing.assert_array_equal(difference-np.diag(np.diag(difference)),0.)
    assert difference[mesh.inverse[1,0],mesh.inverse[1,0]]==0


def test_projection_configuration_requires_explicit_snapshot_and_flags():
    mesh,_,_,_=native_fixture()
    with pytest.raises(ValueError,match='snapshot|psi'):
        filtered(mesh,psi=None)
    with pytest.raises(ValueError,match='boundary'):
        filtered(mesh,boundary=np.array([99,0,0,0]))
    with pytest.raises(ValueError,match='sparse'):
        NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,
            projection='dense',projection_n0=configuration(mesh))


def test_curved_signed_geometry_hyper_filter_uses_physical_hessian():
    from tests.test_native_gauss import native_coefficients
    mesh,_,_,_=native_fixture()
    r=lambda s,t:2+s+.1*s*s*t
    z=lambda s,t:t+.2*s*t*t
    rc=native_coefficients(r,lambda s,t:1+.2*s*t,lambda s,t:.1*s*s,lambda s,t:.2*s)
    zc=native_coefficients(z,lambda s,t:.2*t*t,lambda s,t:1+.4*s*t,lambda s,t:.4*t)
    size=np.arange(1,17).reshape(4,4)/8.;size[1::2,::2]*=-1
    x=np.stack([rc/size,zc/size])[:,:,None,:]
    reference=NativeSourceMesh(x,size[:,:,None],mesh.vertex,mesh.indices,units=mesh.units)
    a=filtered(reference,filter_perp=.3,filter_hyper=.7,filter_parallel=.2)
    q,w=np.polynomial.legendre.leggauss(4);q=(q+1)/2;w=w/2
    expected=0.
    for i,s in enumerate(q):
        for j,t in enumerate(q):
            R=r(s,t);jac=(1+.2*s*t)*(1+.4*s*t)-.1*s*s*.2*t*t
            expected+=2*np.pi*w[i]*w[j]*jac*R*(R*R+.3+.7/(R*R)+.2/(5*R))
    c=x[0,:,0,:].reshape(-1)
    assert c@a.mass@c==pytest.approx(expected,rel=2e-13)


def test_source_projection_has_full_toroidal_factor_and_preserves_raw_history():
    from src.models.native_coupling import NativeMarkerSource
    from copy import deepcopy
    mesh,source,state,u=native_fixture()
    full=filtered(mesh,filter_perp=0.,filter_hyper=0.,filter_parallel=0.,apply_dirichlet=False)
    configured=NativeMarkerSource(full,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=0.)
    state.feedback.values[0,0]=1e-4
    state.feedback.deltas[1,1]=2e-3
    before=deepcopy(state)
    old,_=source.advance(state,1e-9,u,u)
    new,_=configured.advance(state,1e-9,u,u)
    np.testing.assert_allclose(new.feedback.values,old.feedback.values/(2*np.pi),atol=1e-18,rtol=2e-12)
    np.testing.assert_array_equal(new.feedback.deltas,new.feedback.values-before.feedback.values)
    np.testing.assert_array_equal(state.feedback.values,before.feedback.values)
    np.testing.assert_array_equal(state.cloud.P_imp,before.cloud.P_imp)
    with pytest.raises(ValueError,match='configuration'):
        bad=configuration(mesh);bad['do_zonal']=True
        NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,
            projection='sparse',projection_n0=bad)


def test_two_real_phase_a_filtered_sparse_projection_resource_diagnostics():
    import h5py,json,time,resource
    from pathlib import Path
    from scipy.sparse.linalg import LinearOperator,onenormest
    from scipy.sparse import diags
    from src.models.native_state import NativeUnits
    from src.data.namelist import parse_namelist
    for run,index in [('phaseA_dataset_1d22_ang15_massloss_neon_v400',4110),
                      ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',2703)]:
        folder=Path('data/EHL-2')/run
        params=parse_namelist((folder/'jorek_namelist_spitest').read_text())
        with h5py.File(folder/f'jorek{index:06d}.h5') as f:
            scalar=lambda name:float(np.asarray(f[name][...]).ravel()[0])
            units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
            # Explicit inspected-source default fix_axis_nodes=F (preset_parameters:662).
            # This case tests construction at this snapshot, NOT production's unobserved original init psi.
            settings=dict(filter_perp=params['filter_perp_n0'],filter_hyper=params['filter_hyper_n0'],
                filter_parallel=params['filter_par_n0'],F0=scalar('F0'),psi=f['values'][0],
                field_time_si=scalar('t_now')*units.t_norm_si,field_provenance=f'bounded_snapshot_diagnostic:{index}',
                boundary=f['boundary'][...],axis_nodes=f['axis_node'][...]==b'T',
                fix_axis_nodes=False,apply_dirichlet=True)
            start=time.perf_counter()
            mesh=NativeSourceMesh(f['x'][...],f['size'][...],f['vertex'][...]-1,f['index'][...],
                units=units,projection='sparse',projection_n0=settings,max_elements=5000)
        construction=time.perf_counter()-start
        assert mesh.n_dofs==16419 and mesh.mass.nnz<1000000
        # Manufactured all-axis solution, not physical marker/field ground truth.
        exact=np.sin(np.arange(mesh.n_dofs)*.123)*1e-3
        rhs=(mesh.mass@exact)[None]
        start=time.perf_counter();result=mesh.project(rhs);solve=time.perf_counter()-start
        recovered=np.empty(mesh.n_dofs);recovered[mesh.inverse.ravel()]=result[0,:,0,:].ravel()
        residual=np.linalg.norm(mesh.mass@recovered-rhs[0],np.inf)/np.linalg.norm(rhs,np.inf)
        error=np.max(np.abs(recovered-exact))
        residual_vector=mesh.mass@recovered-rhs[0]
        denominator=abs(mesh.mass)@abs(recovered)+abs(rhs[0])
        backward_error=np.max(abs(residual_vector)/np.maximum(denominator,np.finfo(float).tiny))
        d=mesh._equilibration
        inv=lambda x:d*mesh._factor.solve(d*x)
        op=LinearOperator(mesh.mass.shape,matvec=inv,rmatvec=inv,dtype=float)
        norm=float(np.asarray(abs(mesh.mass).sum(axis=0)).max())
        # Deterministic onenormest uses t=1, no randomized extra directions.
        condition_estimate=norm*onenormest(op,t=1)
        scaled=diags(d)@mesh.mass@diags(d)
        sinv=LinearOperator(mesh.mass.shape,matvec=mesh._factor.solve,rmatvec=mesh._factor.solve,dtype=float)
        scaled_estimate=float(np.asarray(abs(scaled).sum(axis=0)).max())*onenormest(sinv,t=1)
        print(json.dumps(dict(run=run,n_dofs=mesh.n_dofs,nnz=mesh.mass.nnz,
            factor_nnz=mesh._factor.L.nnz+mesh._factor.U.nnz,construction_s=construction,solve_s=solve,
            relative_inf_residual=residual,componentwise_backward_error=backward_error,max_coefficient_error=error,
            raw_one_norm_condition_estimate=condition_estimate,scaled_one_norm_condition_estimate=scaled_estimate,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
        assert residual<1e-12 and backward_error<1e-12 and error<1e-7
        assert np.isfinite([condition_estimate,scaled_estimate]).all()


def test_curved_quadratic_trial_has_nonzero_physical_second_derivative():
    from tests.test_native_gauss import native_coefficients
    mesh,_,_,_=native_fixture()
    r=lambda s,t:2+s+.1*s*t
    rs=lambda s,t:1+.1*t
    rt=lambda s,t:.1*s
    z=lambda s,t:t+.2*s*t*t
    rc=native_coefficients(r,rs,rt,lambda s,t:.1)
    zc=native_coefficients(z,lambda s,t:.2*t*t,lambda s,t:1+.4*s*t,lambda s,t:.4*t)
    x=np.stack([rc,zc])[:,:,None,:]
    reference=NativeSourceMesh(x,mesh.size,mesh.vertex,mesh.indices,units=mesh.units)
    a=filtered(reference,filter_perp=.3,filter_hyper=.7,filter_parallel=.2)
    c=native_coefficients(lambda s,t:r(s,t)**2,lambda s,t:2*r(s,t)*rs(s,t),
        lambda s,t:2*r(s,t)*rt(s,t),lambda s,t:2*rs(s,t)*rt(s,t)+.2*r(s,t)).reshape(-1)
    q,w=np.polynomial.legendre.leggauss(4);q=(q+1)/2;w=w/2
    expected=0.
    for i,s in enumerate(q):
        for j,t in enumerate(q):
            R=r(s,t);jac=rs(s,t)*(1+.4*s*t)-rt(s,t)*.2*t*t
            # grad(R²)=2R, Lap_cyl(R²)=4, Bgrad(R²)=2, BB²=5/R.
            expected+=2*np.pi*w[i]*w[j]*jac*R*(R**4+.3*4*R*R+.7*16+.2*4*R/5)
    assert c@a.mass@c==pytest.approx(expected,rel=3e-13)


def test_marker_projection_is_not_reused_as_independent_density_projector():
    from src.models.native_coupling import SmallShardModel
    mesh,_,_,_=native_fixture()
    with pytest.raises(ValueError,match='density|projection'):
        SmallShardModel(filtered(mesh),width_m=.15,material_density_m3=4.3e28,ablation_model=3)


def test_three_base_events_recompute_feedback_and_resume_with_new_sparse_factor(tmp_path):
    import torch
    from copy import deepcopy
    from tests.test_native_joint_source import joint_fixture
    from src.models.native_coupling import NativeMarkerSource,SmallShardModel,advance_coupled_reference
    from src.models.marker_state import ReferenceRollout,MarkerCloud,save_model_rollout,load_model_rollout
    from src.models.native_state import NativeStateContext
    model,mesh,original,marker,u,d,kw=joint_fixture()
    u=u.detach().clone();u[0]=torch.as_tensor(mesh.coordinates[1])
    def fresh_source():
        m=filtered(mesh,psi=u[0].numpy(),F0=original.F0,field_time_si=1e-9)
        return NativeMarkerSource(m,atomic=original.atomic,atomic_version=original.configuration['atomic'],
            potentials_ev=original.potentials_ev,F0=original.F0,diffusion_ju=0.)
    source=fresh_source()
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    model.source_configuration=deepcopy(source.configuration)
    model.shard_configuration=deepcopy(shard.configuration)
    # Fixed bounded untrained fixture amplitude, no optimizer/update.
    with torch.no_grad():model.field.decoder.weight.mul_(.001);model.field.decoder.bias.mul_(.001)
    marker.cloud=MarkerCloud.empty(10,8)
    shards=dict(positions=np.array([[2.6,.6,0.]]),velocities=np.zeros((1,3)),radius=np.array([1e-5]),ablation_atoms_s=np.array([1e12]))
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,
        'vertex':mesh.vertex+1,'index':mesh.indices},boundary={'policy':'learned_unverified'},
        physics={'feature_parameter_rows':list(kw['feature_parameters'].rows)},
        numerics={'scheme':'implicit Euler'},model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.numpy().copy(),u.numpy().copy(),shards,1e-9,1,(1e-9,)*3,
        fluid_dt_previous_si=1e-9,fluid_deltas=d.detach().numpy().copy(),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=kw['memory'].detach().numpy().copy())
    calls=[];real_source=source.advance;real_shard=shard.update;expected_old=[state.fields_current]
    def source_call(m,target,previous,current,**births):
        calls.append('source');np.testing.assert_array_equal(current,expected_old[0])
        return real_source(m,target,previous,current,**births)
    def shard_call(s,old,time,dt):
        calls.append('shard');np.testing.assert_array_equal(old,expected_old[0])
        return real_shard(s,old,time,dt)
    source.advance=source_call;shard.update=shard_call
    hook=model.register_forward_pre_hook(lambda *args:calls.append('field'))
    path=[]
    for _ in range(3):
        state,_=advance_coupled_reference(state,model=model,source=source,shard_model=shard)
        path.append(state);expected_old[0]=state.fields_current
    hook.remove()
    assert calls==['source','field','shard']*3 and state.plan_cursor==3
    assert all(np.any(s.marker.feedback.values) for s in path)
    checkpoint=tmp_path/'n0.pt';save_model_rollout(checkpoint,path[0],model)
    new_model=deepcopy(model)
    stored=torch.load(checkpoint,weights_only=False)
    inputs=stored["state"].native_context.configuration["model"]["source"]["mesh"]["projection_n0"]["reconstruction_inputs"]
    new_mesh=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,
        projection="sparse",projection_n0=inputs)
    new_source=NativeMarkerSource(new_mesh,atomic=original.atomic,atomic_version=original.configuration["atomic"],
        potentials_ev=original.potentials_ev,F0=original.F0,diffusion_ju=0.)
    new_shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    resumed=load_model_rollout(checkpoint,new_model,native_context=context)
    for _ in range(2):resumed,_=advance_coupled_reference(resumed,model=new_model,source=new_source,shard_model=new_shard)
    for key in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(resumed,key),getattr(state,key))
    np.testing.assert_array_equal(resumed.marker.feedback.values,state.marker.feedback.values)
    np.testing.assert_array_equal(resumed.marker.feedback.deltas,state.marker.feedback.deltas)
    for name in state.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(resumed.marker.cloud,name),getattr(state.marker.cloud,name))
    assert resumed.marker.clock==state.marker.clock
    for name in ('positions','velocities','radius','ablation_atoms_s'):
        np.testing.assert_array_equal(resumed.shards[name],state.shards[name])


def test_axis_replacement_matches_shared_indices_before_assembly():
    mesh,_,_,_=native_fixture()
    indices=mesh.indices.copy();indices[2,1]=indices[2,0]
    shared=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,indices,units=mesh.units)
    axis=np.array([True,True,False,False])
    result=filtered(shared,axis_nodes=axis,fix_axis_nodes=True,apply_dirichlet=False)
    # Both vertices share the degree3 global index: four local entries replace to 1e12.
    k=shared.inverse[2,0]
    assert result.mass[k,k]==pytest.approx(4e12,rel=2e-16)
    assert result.configuration['projection_n0']['fix_axis_nodes'] is True
