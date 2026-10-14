"""Boundary applicability through the complete field-step seam, no fitting."""
import numpy as np
import torch


def model_for(boundary, *, F0=-2.):
    from src.models.native_field_boundary import NativeBoundaryLearningRows
    from src.models.native_field_step import NativeFieldStep, NativeUpdateMapping
    from src.models.stencil_conv import build_fem_stencil_geometry
    from src.models.native_state import NativeUnits
    vertex=np.array([[0],[1],[2],[3]])
    pos=np.array([[2.,0.],[3.,0.],[3.,1.],[2.,1.]])
    indices=np.arange(1,17).reshape(4,4)
    rows=NativeBoundaryLearningRows(vertex=vertex,boundary=np.array(boundary),indices=indices,
        freeboundary=False,RMP_on=False,no_mach1_bc=False,use_simple_bnd_types=False,
        bc_natural_open=True,mach_one_bnd_integral=False,vpar_smoothing=False,grid_to_wall=True,n_wall_blocks=14,
        parameters=dict(gamma=5/3,gamma_sheath_i=32/9,gamma_sheath_e=14/3,
            density_reflection=0.,imp_reflection=0.,min_sheath_angle=1.,F0=F0,
            T_min=1e-8,visco_par_heating=0.),
        parameter_provenance='analytic_fixture_not_production_binding')
    mapping=NativeUpdateMapping(indices,constrained=np.zeros(4,bool),treat_axis=False,
        keep_n0_const=False,boundary_rows=rows)
    return NativeFieldStep(build_fem_stencil_geometry(vertex,pos),mapping,
        units=NativeUnits(1.,8e19,1.,5/3),state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),
        hidden=4,n_layers=1,n_iters=1)


def test_field_step_exposes_exact_applicability_without_hardening_sheath_terms():
    model=model_for([1,1,0,0])
    with torch.no_grad():
        model.decoder.weight.zero_();model.decoder.bias.fill_(1.)
    u=torch.zeros((9,4,1,4),dtype=torch.float64,requires_grad=True)
    out=model(u,torch.zeros_like(u),dt_si=.25,previous_dt_si=.25,scheme='implicit Euler')
    # Bottom target: fixed first four value+s rows; rho/Ti/vpar/rho_imp/Te learn.
    torch.testing.assert_close(out.increment[:4,:2,0,:2],torch.zeros((4,2,2),dtype=u.dtype))
    torch.testing.assert_close(out.increment[4:,:,:, :2],torch.full((5,4,1,2),.25,dtype=u.dtype))
    # Columns36:40 = nodal Mach1 row applicability;40:44 = natural integral rows.
    # Wall side2 uses degree s (index1), no mixed derivative invented.
    expected=torch.tensor([[1.,1.,0.,0.],[1.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]],dtype=u.dtype)
    torch.testing.assert_close(out.boundary_features[:,36:40],expected)
    torch.testing.assert_close(out.boundary_features[:,40:44],expected)
    assert model.checkpoint_configuration['boundary_policy']=='dirichlet_rows_open_learned_v1'
    assert 'learned' in model.checkpoint_configuration['boundary_rows']['weak_terms']


def test_corner_pair_integral_skip_keeps_nodal_mach1_and_field_graph():
    # Two corner codes have no wall side: no natural integral; code19 still
    # gets its nodal Mach1 rows, code20 fixes ALL nine fields (var_rhon=8).
    model=model_for([19,20,0,0])
    u=torch.zeros((9,4,1,4),dtype=torch.float64,requires_grad=True)
    out=model(u,torch.zeros_like(u),dt_si=.25,previous_dt_si=.25,scheme='implicit Euler')
    expected=torch.tensor([[1.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]],dtype=u.dtype)
    torch.testing.assert_close(out.boundary_features[:,36:40],expected)
    assert not out.boundary_features[:,40:44].any()
    assert out.increment[4,0,0,1]==0 and out.increment[7,0,0,1]==0
    out.increment.square().sum().backward()
    assert u.grad is not None and u.grad.abs().sum()>0
    assert model.boundary_encoder.weight.grad.abs().sum()>0


def test_boundary_parameter_route_has_prescribed_response_not_just_diagnostics():
    model=model_for([1,1,0,0])
    # A hand-set adapter reads the ion sheath scalar (column45) and predicts
    # rho. Constant latent features have zero spatial Laplacian on this mesh.
    # This checks routing arithmetic, not the physical response to gamma_i.
    with torch.no_grad():
        model.encoder.weight.zero_();model.encoder.bias.zero_()
        model.boundary_encoder.weight.zero_();model.boundary_encoder.bias.zero_()
        model.boundary_encoder.weight[0,45]=1.
        model.decoder.weight.zero_();model.decoder.bias.zero_()
        model.decoder.weight[16,0]=1.
    u=torch.zeros((9,4,1,4),dtype=torch.float64)
    out=model(u,u,dt_si=.25,previous_dt_si=.25,scheme='implicit Euler')
    torch.testing.assert_close(out.increment[4,0],torch.full((1,4),.25*np.tanh(32/9),dtype=u.dtype),rtol=1e-14,atol=1e-14)


def test_two_source_joint_field_events_read_prediction_and_restore_full_state(tmp_path):
    from copy import deepcopy
    from tests.test_native_joint_source import joint_fixture
    from src.models.native_field_step import NativeFieldStep
    from src.models.native_coupling import NativeMarkerSource,SmallShardModel,advance_coupled_reference
    from src.models.marker_state import ReferenceRollout,save_model_rollout,load_model_rollout,MarkerCloud
    from src.models.native_state import NativeStateContext
    model,mesh,source,marker,u,raw,k=joint_fixture()
    source=NativeMarkerSource(mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=source.diffusion_ju,
        time_sampling='jorek_raw_deltas_v1',field_static=True)
    model.source_configuration=deepcopy(source.configuration)
    original=model.field
    mapping=deepcopy(original.update_mapping)
    mapping.boundary_rows=model_for([1,1,0,0],F0=source.F0).update_mapping.boundary_rows
    model.field=NativeFieldStep(original.geometry,mapping,units=original.units,
        state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),hidden=8,
        n_layers=1,n_iters=2,temporal_core=original.temporal_core,feature_path=original.feature_path)
    # Prescribed exact rate makes rho rise by .1 per event. It is an oracle
    # fixture, not an untrained accuracy or stability gate.
    dt=1e-9
    with torch.no_grad():
        model.field.decoder.weight.zero_();model.field.decoder.bias.zero_()
        model.field.decoder.bias[4*4]=.1*mesh.units.t_norm_si*original.temporal_core.time_scale_ju/dt
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    model.shard_configuration=deepcopy(shard.configuration)
    shards=dict(positions=np.zeros((0,3)),velocities=np.zeros((0,3)),radius=np.zeros(0),ablation_atoms_s=np.zeros(0))
    boundary=model.field.update_mapping.boundary_rows
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,
        'vertex':mesh.vertex+1,'index':mesh.indices},boundary={'policy':boundary.policy,'native':{'boundary':[1,1,0,0]}},
        physics={'feature_parameter_rows':list(k['feature_parameters'].rows)},numerics={'scheme':'implicit Euler'},
        model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.detach().numpy().copy(),u.detach().numpy().copy(),shards,dt,1,(dt,dt),
        fluid_dt_previous_si=dt,fluid_deltas=raw.detach().numpy().copy(),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=k['memory'].detach().numpy().copy())
    first,_=advance_coupled_reference(state,model=model,source=source,shard_model=shard)
    # SCD=1e-12; initial ne=4e19 => q1=.04. New predicted rho=1.1,
    # old rho_imp=.5 and previous neutral feedback => ne=4.8e19.
    np.testing.assert_allclose(first.marker.cloud.P_imp[0,:2],[.96,.04],atol=1e-14)
    np.testing.assert_allclose(first.fields_current[4,0],1.1,atol=1e-14)
    second,_=advance_coupled_reference(first,model=model,source=source,shard_model=shard)
    np.testing.assert_allclose(second.marker.cloud.P_imp[0,:2],[.91392,.08608],atol=2e-13)
    checkpoint=tmp_path/'boundary-event.pt'
    save_model_rollout(checkpoint,first,model)
    fresh=deepcopy(model)
    restored=load_model_rollout(checkpoint,fresh,native_context=context)
    resumed,_=advance_coupled_reference(restored,model=fresh,source=source,shard_model=shard)
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(second,name),getattr(resumed,name))
    for name in MarkerCloud.array_names():
        np.testing.assert_array_equal(getattr(second.marker.cloud,name),getattr(resumed.marker.cloud,name))
    np.testing.assert_array_equal(second.marker.feedback.values,resumed.marker.feedback.values)
    np.testing.assert_array_equal(second.marker.feedback.deltas,resumed.marker.feedback.deltas)
    assert second.marker.clock==resumed.marker.clock
