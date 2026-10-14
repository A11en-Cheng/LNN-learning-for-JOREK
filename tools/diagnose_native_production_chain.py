#!/usr/bin/env python3
"""Execute nonzero joint nine-field events, not train or certify accuracy.

Standalone analytic fixture; no imports from tests, recorded source or future
fields. Run from repo root: python -m tools.diagnose_native_production_chain.
The deliberately broad source and large proposal dv bound fixture cost; these
are NOT either phaseA run's production parameters.
"""
import argparse
from copy import deepcopy
from dataclasses import fields,is_dataclass
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

from src.models.native_state import NativeUnits,NativeStateContext,configuration_digest
from src.models.marker_state import MarkerCloud,MarkerRuntime,MarkerClock,ProjectionFeedback,ReferenceRollout
from src.models.native_coupling import NativeSourceMesh,NativeMarkerSource,TorchMarkerLoads,CoupledNativeField
from src.models.native_source_nonlocal import NativeFluxEquilibrium
from src.training.base_coupled import TorchRecomputedSourceEvents


class AnalyticEquilibriumProvider:
    configuration=dict(version='chain-analytic-rectangle-ES-v1',production_certified=False)

    def __call__(self,fields,time_si):
        return NativeFluxEquilibrium.bind(fields,time_si=time_si,psi_axis=0.,psi_bnd=1.,lcfs_lost=False,
            xcase='none',psi_xpoints=(0.,0.),z_xpoints=(0.,0.),provenance='explicit analytic fixture psi=R-2')


def prescribed_atomic(te_ev,ne_m3):
    """Fixture-only neutral ionization; never represented as production ADAS."""
    scd=np.zeros((len(ne_m3),10));scd[:,0]=1e-12
    return scd,np.zeros_like(scd),np.zeros((len(ne_m3),11))


def build_fixture(events=3):
    """Build the full production-mechanism event state with fixed nonzero weights."""
    from src.models.marker_feedback import MU_ZERO,ATOMIC_MASS_UNIT,EL_CHG,MarkerFeedbackParameters
    from src.models.native_features import TorchNativeGauss,NativeFeaturePath
    from src.models.native_field_step import NativeFieldStep,NativeUpdateMapping
    from src.models.native_temporal import CfCTimeCore
    from src.models.stencil_conv import build_fem_stencil_geometry
    from src.models.native_source_features import SourceFeatureScales
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    from src.models.native_source_density import NativeShardDensityKernel
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_source_production import NativeProductionShardModel
    x=np.zeros((2,4,1,4))
    x[0,0,0]=[2.,3.,3.,2.];x[1,0,0]=[0.,0.,1.,1.]
    x[0,1,0]=[1/3,-1/3,-1/3,1/3];x[1,2,0]=[1/3,1/3,-1/3,-1/3]
    size=np.ones((4,4,1));vertex=np.arange(4)[:,None];indices=np.arange(1,17).reshape(4,4)
    units=NativeUnits(np.sqrt(MU_ZERO*ATOMIC_MASS_UNIT*8e19),8e19,1.,5/3)
    mesh=NativeSourceMesh(x,size,vertex,indices,units=units)
    source=NativeMarkerSource(mesh,atomic=prescribed_atomic,atomic_version='chain-fixture-SCD_m3_s-not-ADAS',
        potentials_ev=np.full(10,20.),F0=-1.,diffusion_ju=0.,
        time_sampling='jorek_raw_deltas_v1',field_static=False)
    evaluator=TorchNativeGauss(x,size,vertex,units=units)
    # Historical schema says training_support; these are explicit unit-reference
    # fixture scales, not fitted statistics, and no training records are read.
    statistics=dict(version='native-gauss-train-statistics-v1',geometry=evaluator.geometry_digest,
        training_support=[['analytic_fixture_unit_reference_NOT_training',0]],
        scenario_enabled=False,physics_enabled=False,value_center=np.zeros(9),value_scale=np.ones(9),
        gradient_scale=np.ones((9,2)),scenario_scale=np.ones(2),magnetic_scale=np.ones(2))
    path=NativeFeaturePath(evaluator,statistics)
    geometry=build_fem_stencil_geometry(vertex,x[:,0,0].T)
    field=NativeFieldStep(geometry,NativeUpdateMapping(indices,constrained=np.zeros(4,bool),
        treat_axis=False,keep_n0_const=False),units=units,state_scale=np.ones((9,4)),
        increment_scale=np.ones((9,4)),hidden=8,n_layers=1,n_iters=2,feature_path=path,
        temporal_core=CfCTimeCore(8,backbone_units=11,time_scale_ju=2.))
    density=NativeShardDensityKernel(mesh,ns_radius_min=1.,ns_radius_ratio=1.4,ns_deltaphi=1e3,
        spi_num_vol=True,ns_delta_minor_rad=0.,injection_time_si=0.,toroidal_angles=(0.,))
    births=NativeProductionBirths(density,physical_particles=8e18,n_particle_slots=8,n_cpu=1,rank=0,
        R_geo=1e8,n_plane=1,max_proposals=32)
    dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=4.3e28,F0=-1.,R_geo=2.5,
        magnetic_reduction=False,nonlocal_abl=True,ns_radius_min=1.,ns_radius_ratio=1.4)
    shard=NativeProductionShardModel(dynamics,density,births,n_flux=2,n_table=10)
    response=TorchMarkerLoads(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    model=CoupledNativeField(field,response,source_configuration=source.configuration,
        shard_configuration=shard.configuration,source_scales=SourceFeatureScales.physical_reference())
    # Replace EVERY randomly initialized parameter with deterministic prescribed
    # values. Small nonzero decoder weights retain actual memory/source paths.
    with torch.no_grad():
        for ordinal,parameter in enumerate(model.parameters()):
            index=torch.arange(parameter.numel(),dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(.02*torch.sin(index*.17+ordinal*.31))
        model.field.decoder.weight.fill_(1e-4)
        model.field.decoder.bias.copy_(torch.linspace(2e-4,4e-4,36,dtype=torch.float64))
    model.eval()
    u=np.zeros((9,4,1,4));u[0]=x[0];u[0,0]-=2.
    u[4,0]=1.25;u[6,0]=1e-3;u[8,0]=2000*EL_CHG*MU_ZERO*units.n_norm_m3
    raw=np.zeros_like(u);raw[6,0]=.0002
    marker=MarkerRuntime(MarkerCloud.empty(10,8),
        ProjectionFeedback(np.zeros((5,4,1,4)),np.zeros((5,4,1,4)),0.),
        MarkerClock(0.,0.,1e-9,0.,1e-9),{'fluid_control_origin':'external_control'})
    shards=dict(positions=np.array([[2.5,.5,0.]]),velocities=np.array([[.2,0.,0.]]),radius=np.array([.003]),
        ablation_atoms_s=np.array([1e24]),species=np.ones(1),ids=np.array([17]),injector_phi=np.zeros(1),
        positions_coordinate_system='RZphi',velocities_coordinate_system='injector_R_Z_RxZ',time_si=1e-9)
    provider=AnalyticEquilibriumProvider()
    shards=shard.refresh_diagnostics(shards,u,marker.feedback,provider(u,1e-9),provenance='analytic initial event')
    context=NativeStateContext(units,(0,),dict(mesh={'x':x,'size':size,'vertex':vertex+1,'index':indices},
        boundary={'policy':'learned_unverified'},physics={'feature_parameter_rows':[]},
        numerics={'scheme':'implicit Euler'},model=model.checkpoint_configuration))
    runtime=ReferenceRollout(marker,u.copy(),u.copy(),shards,1e-9,1,(1e-9,)*events,
        fluid_dt_previous_si=1e-9,fluid_deltas=raw,time_evol_scheme='implicit Euler',native_context=context,
        field_memory=np.full((4,8),.1))
    return model,source,shard,runtime,provider


def build_teacher_fixture(events=3):
    """Explicit analytic psi-record intervention; NEVER a real-record claim."""
    from types import SimpleNamespace
    from src.models.native_psi_teacher import RecordedPsiFeedback,NativePsiTeacherShardModel
    model,source,base,runtime,analytic=build_fixture(events)
    records=[];time_si=runtime.fluid_time_si
    for index in range(events+1):
        psi=runtime.fields_current[0].copy()
        records.append(SimpleNamespace(psi=psi,equilibrium=analytic(runtime.fields_current,time_si),time_si=time_si,
            geometry_digest=base.dynamics.mesh.configuration['geometry'],units=runtime.native_context.units,
            provenance='analytic-fixture psi record NOT read from data',
            source_sha256=configuration_digest(dict(analytic_fixture_psi=psi,time_si=time_si))))
        if index<events:time_si+=runtime.step_plan_si[index]
    teacher=RecordedPsiFeedback(records,mesh=base.dynamics.mesh)
    shard=NativePsiTeacherShardModel(base,teacher)
    model.shard_configuration=deepcopy(shard.configuration)
    runtime.native_context.configuration['model']=model.checkpoint_configuration
    branch=shard.diagnostic_fields(runtime.fields_current,runtime.fluid_time_si)
    runtime.shards=shard.refresh_diagnostics(runtime.shards,runtime.fields_current,runtime.marker.feedback,
        teacher(branch,runtime.fluid_time_si),provenance='explicit analytic teacher initialization')
    return model,source,shard,runtime,teacher


def _event_row(before,after):
    out=after.runtime;old=before.runtime;s=out.shards;cloud=out.marker.cloud
    return dict(field_time_si=out.fluid_time_si,marker_time_si=out.marker.feedback.time_si,
        field_delta_max_abs=np.max(np.abs(out.fields_current-old.fields_current),axis=(1,2,3)).tolist(),
        field_digest=configuration_digest(out.fields_current),raw_deltas_digest=configuration_digest(out.fluid_deltas),
        feedback_digest=configuration_digest(out.marker.feedback.values),
        live_markers=int(cloud.alive.sum()),new_births=int(np.sum(cloud.i_life-old.marker.cloud.i_life)),
        marker_weight_atoms=cloud.weight[cloud.alive].tolist(),positions_RZphi=s['positions'].tolist(),
        radius_m=s['radius'].tolist(),ablation_atoms_s=s['ablation_atoms_s'].tolist(),
        nonlocal_time_si=s['nonlocal_diagnostic'].time_si,nonlocal_field_digest=s['nonlocal_diagnostic'].field_digest,
        source_volume_time_si=s['source_volume_time_si'],source_volume_m3=s['source_volume_m3'].tolist(),
        diagnostic_field_binding=deepcopy(s.get('diagnostic_field_binding')),
        branch_intervention=deepcopy(s.get('psi_teacher_event')))


def _digest(value):
    """Exact full-state comparison, including diagnostic caches and slot history."""
    def canonical(item):
        if isinstance(item,torch.Tensor):return item.detach().cpu().numpy()
        if is_dataclass(item):return {f.name:canonical(getattr(item,f.name)) for f in fields(item)}
        if isinstance(item,dict):return {key:canonical(v) for key,v in item.items()}
        if isinstance(item,(tuple,list)):return [canonical(v) for v in item]
        return item
    return configuration_digest(canonical(value))


def _rng_state():
    return deepcopy((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def _set_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])


def _verify_restore(directory,builder,events,adapter,states,controls,rng_states,scope):
    from src.data.base_coupled import Support,make_sequence
    from src.training.base_coupled import SequenceSession,save_session,load_session,run_sequence
    sequence=make_sequence(run=scope,task='self_generated',controls=tuple(controls),nodes=(),burnin_steps=0,
        support=(Support(scope,0.,controls[-1].end_si,'complete_diagnostic_history',scope),),
        blocks=(Support(scope,0.,controls[-1].end_si,'diagnostic',scope),),
        state_evidence={'complete_state':'saved'},required_state=('complete_state',),model_reasons=(),
        parameters={'diagnostic_only':True})
    # Save the first event AND its own RNG, not the end-of-window RNG.
    _set_rng(rng_states[1])
    session=SequenceSession(states[1],cursor=1,sequence_fingerprint=sequence.fingerprint,
        adapter_fingerprint=adapter.fingerprint)
    save_session(directory/'session.pt',sequence,adapter,session)
    fresh,source,shard,_,provider=builder(events)
    with torch.no_grad():fresh.field.decoder.bias.add_(.1)
    different=_digest(fresh.state_dict())!=_digest(adapter.model.state_dict())
    other=TorchRecomputedSourceEvents(fresh,source,shard,equilibrium_provider=provider)
    loaded=load_session(directory/'session.pt',sequence,other)
    weights_equal=_digest(fresh.state_dict())==_digest(adapter.model.state_dict())
    resumed=run_sequence(sequence,other,session=loaded)
    direct_digest=_digest(states[-1]);resumed_digest=_digest(resumed.state)
    rng_equal=_digest(_rng_state())==_digest(rng_states[-1])
    verification=dict(fresh_constructor_weights_differ=different,loaded_weights_equal=weights_equal,
        complete_state_bitwise_equal=direct_digest==resumed_digest,python_numpy_torch_cpu_rng_equal=rng_equal,
        direct_state_digest=direct_digest,resumed_state_digest=resumed_digest)
    if not all((different,weights_equal,direct_digest==resumed_digest,rng_equal)):
        raise AssertionError('new-model full event checkpoint recovery differs: '+str(verification))
    return verification


def _verify_bptt(directory,model,states):
    # Observation of the final output only: no targets, teacher reset or loss
    # optimizer. Source/shard are still intentionally NumPy stop-gradient.
    states[-1].values.square().sum().backward()
    tensors={name:getattr(state,attribute) for prefix,state in [('initial',states[0]),('first',states[1])]
        for name,attribute in [(prefix+'_values','values'),(prefix+'_raw_deltas','raw_deltas'),(prefix+'_memory','memory')]}
    gradients={name:None if value.grad is None else value.grad.detach().cpu().numpy().copy()
        for name,value in tensors.items()}
    if any(value is None or not np.isfinite(value).all() or not np.any(value) for value in gradients.values()):
        raise AssertionError('cross-event values/rawD/memory graph was lost')
    result={name:float(np.abs(value).sum()) for name,value in gradients.items()}
    for name,module in [('temporal_parameters',model.temporal_core),('decoder_parameters',model.field.decoder)]:
        result[name]=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        if not np.isfinite(result[name]) or result[name]<=0:raise AssertionError('disconnected '+name)
    np.savez_compressed(directory/'bptt.npz',**gradients)
    return dict(result,loss_input='last_committed_nine_field_values_only_no_labels')


def execute_chain(output_dir,*,events=3,builder=build_fixture,
                  scope='analytic_fixture_production_mechanisms_not_real_data'):
    """Execute complete events; caller may inject an explicit fixture/teacher builder."""
    from src.data.base_coupled import StepControl
    if type(events) is not int or not 2<=events<=3:raise ValueError('two or three complete events required')
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=False)  # never overwrite existing evidence
    random.seed(1931);np.random.seed(1931);torch.manual_seed(1931)
    started=time.perf_counter()
    model,source,shard,runtime,provider=builder(events)
    before_weights=_digest(model.state_dict())
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=provider)
    initial=adapter.initialize(runtime);states=[initial];rows=[];controls=[];rng_states=[_rng_state()]
    for value in (initial.values,initial.raw_deltas,initial.memory):value.requires_grad_()
    for index,dt in enumerate(runtime.step_plan_si[:events]):
        previous=states[-1];start=previous.runtime.fluid_time_si
        control=StepControl(start,dt,start+dt,runtime.marker.provenance['fluid_control_origin'],scope)
        current,_=adapter.advance(previous,control)
        for value in (current.values,current.raw_deltas,current.memory):value.retain_grad()
        row=_event_row(previous,current)
        if not all(np.isfinite(value) and value>0 for value in row['field_delta_max_abs']):
            raise ValueError('actual finite nonzero nine-field predictions required at every event')
        controls.append(control);rows.append(row);states.append(current)
        rng_states.append(_rng_state())
    np.savez_compressed(directory/'trajectory.npz',fields=np.stack([s.runtime.fields_current for s in states]),
        raw_deltas=np.stack([s.runtime.fluid_deltas for s in states]),memory=np.stack([s.runtime.field_memory for s in states]),
        marker_x=np.stack([s.runtime.marker.cloud.x for s in states]),
        marker_P_imp=np.stack([s.runtime.marker.cloud.P_imp for s in states]),
        marker_weight=np.stack([s.runtime.marker.cloud.weight for s in states]),
        marker_i_elm=np.stack([s.runtime.marker.cloud.i_elm for s in states]),
        marker_i_life=np.stack([s.runtime.marker.cloud.i_life for s in states]),
        marker_V_prev=np.stack([s.runtime.marker.cloud.V_prev for s in states]),
        marker_t_birth=np.stack([s.runtime.marker.cloud.t_birth for s in states]),
        feedback=np.stack([s.runtime.marker.feedback.values for s in states]),
        projection_deltas=np.stack([s.runtime.marker.feedback.deltas for s in states]),
        shard_radius=np.stack([s.runtime.shards['radius'] for s in states]),
        shard_positions=np.stack([s.runtime.shards['positions'] for s in states]),
        shard_ablation_atoms_s=np.stack([s.runtime.shards['ablation_atoms_s'] for s in states]))
    checkpoint=_verify_restore(directory,builder,events,adapter,states,controls,rng_states,scope)
    bptt=_verify_bptt(directory,model,states)
    unchanged=before_weights==_digest(model.state_dict())
    if not unchanged:raise AssertionError('diagnostic changed model weights or buffers')
    result=dict(schema='native-nine-field-production-chain-v1',scope=scope,optimizer_updates=0,
        events_completed=len(rows),production_closed_loop_certified=False,zero_decoder=False,
        field_mode=model.mode,gradient_scope=adapter.gradient_scope,events=rows,
        source_configuration=source.configuration,shard_configuration=shard.configuration,
        adapter_configuration_fingerprint=adapter.configuration_fingerprint,checkpoint=checkpoint,bptt=bptt,
        model_weights_unchanged=unchanged,
        feature_scales='explicit_fixture_unit_reference_no_fit',
        elapsed_seconds=time.perf_counter()-started)
    (directory/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--events',type=int,choices=(2,3),default=3)
    parser.add_argument('--psi-feedback',choices=('analytic','teacher-fixture'),default='analytic')
    args=parser.parse_args()
    options={} if args.psi_feedback=='analytic' else dict(builder=build_teacher_fixture,
        scope='analytic_psi_teacher_fixture_not_real_data_not_autonomous')
    print(json.dumps(execute_chain(args.output_dir,events=args.events,**options),indent=2,allow_nan=False))


if __name__=='__main__':main()
