"""Actual joint-field events with production mechanisms on a small analytic cloud.

The zero joint decoder isolates deterministic source/shard/RNG behavior, without
a frozen-capacity response or a random-network accuracy criterion.
"""
from copy import deepcopy
import numpy as np
import pytest
import torch


def production_fixture():
    from tests.test_native_joint_source import joint_fixture
    from tests.test_native_gauss import native_coefficients
    from src.models.native_source_density import NativeShardDensityKernel
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_source_production import NativeProductionShardModel
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    from src.models.marker_state import ReferenceRollout,MarkerCloud
    from src.models.native_state import NativeStateContext
    from src.models.marker_feedback import EL_CHG,MU_ZERO
    model,mesh,source,marker,u,d,k=joint_fixture()
    with torch.no_grad():model.field.decoder.weight.zero_();model.field.decoder.bias.zero_()
    u=u.detach().numpy().copy();u[:]=0.;u[4,0]=1.25;u[6,0]=1e-3
    u[0,:,0]=native_coefficients(lambda a,b:a,lambda a,b:1.,lambda a,b:0.,lambda a,b:0.)
    u[8,0]=2000*EL_CHG*MU_ZERO*mesh.units.n_norm_m3
    kernel=NativeShardDensityKernel(mesh,ns_radius_min=1.,ns_radius_ratio=1.4,ns_deltaphi=1e3,
        spi_num_vol=True,ns_delta_minor_rad=0.,injection_time_si=0.,toroidal_angles=(0.,))
    birth=NativeProductionBirths(kernel,physical_particles=8e18,n_particle_slots=8,n_cpu=1,rank=0,
        R_geo=1e8,n_plane=1,max_proposals=32)  # explicit large dv bounds this RNG fixture's cost
    dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=4.3e28,F0=-1.,R_geo=2.5,
        magnetic_reduction=False,nonlocal_abl=True,ns_radius_min=1.,ns_radius_ratio=1.4)
    shard=NativeProductionShardModel(dynamics,kernel,birth,n_flux=2,n_table=10)
    shards=dict(positions=np.array([[2.5,.5,0.]]),velocities=np.zeros((1,3)),radius=np.array([.003]),
        ablation_atoms_s=np.array([1e24]),species=np.ones(1),ids=np.array([17]),injector_phi=np.zeros(1),
        positions_coordinate_system='RZphi',velocities_coordinate_system='injector_R_Z_RxZ',time_si=1e-9)
    marker.cloud=MarkerCloud.empty(10,8)
    shards=shard.refresh_diagnostics(shards,u,marker.feedback,analytic_equilibrium(u,1e-9),provenance='analytic initial event')
    model.shard_configuration=deepcopy(shard.configuration)
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,'vertex':mesh.vertex+1,
        'index':model.field.update_mapping.indices.cpu().numpy()},boundary={'policy':'learned_unverified'},
        physics={'feature_parameter_rows':list(k['feature_parameters'].rows)},numerics={'scheme':'implicit Euler'},
        model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.copy(),u.copy(),shards,1e-9,1,(1e-9,)*3,
        fluid_dt_previous_si=1e-9,fluid_deltas=np.zeros_like(u),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=k['memory'].detach().numpy().copy())
    return model,source,shard,state


def analytic_equilibrium(u,time):
    from src.models.native_source_nonlocal import NativeFluxEquilibrium
    return NativeFluxEquilibrium.bind(u,time_si=time,psi_axis=0.,psi_bnd=1.,lcfs_lost=False,
        xcase='none',psi_xpoints=(0.,0.),z_xpoints=(0.,0.),provenance='analytic rectangle psi=R-2')


class AnalyticEquilibriumProvider:
    configuration={'version':'analytic-rectangle-ES-fixture-v1'}

    def __call__(self,fields,time_si):
        return analytic_equilibrium(fields,time_si)


def event(state,model,source,shard):
    from src.models.native_coupling import advance_coupled_reference
    return advance_coupled_reference(state,model=model,source=source,shard_model=shard,
        equilibrium_provider=AnalyticEquilibriumProvider())


def test_two_production_mechanism_events_checkpoint_rng_and_stage_failure(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout
    from src.models.native_coupling import advance_coupled_reference
    model,source,shard,state=production_fixture()
    np.random.seed(1931)
    first,_=event(state,model,source,shard)
    assert first.marker.cloud.alive.sum()==1 and first.plan_cursor==1
    assert first.shards['nonlocal_diagnostic'].time_si==first.fluid_time_si
    path=tmp_path/'production-mechanism.pt';save_model_rollout(path,first,model)
    second,_=event(first,model,source,shard);rng_after=deepcopy(np.random.get_state())
    fresh=deepcopy(model);restored=load_model_rollout(path,fresh,native_context=state.native_context)
    resumed,_=event(restored,fresh,deepcopy(source),deepcopy(shard))
    assert second.marker.cloud.alive.sum()==2 and second.plan_cursor==2
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(second,name),getattr(resumed,name))
    for name in second.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(second.marker.cloud,name),getattr(resumed.marker.cloud,name))
    for name in ('positions','velocities','radius','ablation_atoms_s','source_volume_m3'):
        np.testing.assert_array_equal(second.shards[name],resumed.shards[name])
    np.testing.assert_array_equal(second.marker.feedback.values,resumed.marker.feedback.values)
    np.testing.assert_array_equal(second.marker.feedback.deltas,resumed.marker.feedback.deltas)
    assert second.marker.clock==resumed.marker.clock
    np.testing.assert_array_equal(np.random.get_state()[1],rng_after[1]);assert np.random.get_state()[2:]==rng_after[2:]
    failing=deepcopy(source)
    def bad_atomic(*args):raise RuntimeError('after randomized births')
    failing.atomic=bad_atomic
    before=deepcopy(np.random.get_state())
    with pytest.raises(RuntimeError,match='after randomized births'):
        advance_coupled_reference(first,model=model,source=failing,shard_model=shard,
            equilibrium_provider=AnalyticEquilibriumProvider())
    np.testing.assert_array_equal(np.random.get_state()[1],before[1]);assert np.random.get_state()[2:]==before[2:]
    assert first.marker.cloud.alive.sum()==1 and state.marker.cloud.alive.sum()==0
