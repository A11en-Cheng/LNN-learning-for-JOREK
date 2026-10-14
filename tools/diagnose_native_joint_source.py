#!/usr/bin/env python3
"""Small analytic fixture diagnostic, not training or real-cloud qualification.

Uses the same publicly tested fixture; no recorded future fields/feedback.
Run from root with PYTHONPATH=.:tests. Output paths must be new (preserve artifacts).
"""
import argparse
from copy import deepcopy,copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from tests.test_native_joint_source import joint_fixture
from src.models.native_coupling import SmallShardModel,advance_coupled_reference
from src.models.marker_state import ReferenceRollout,MarkerCloud,save_model_rollout,load_model_rollout
from src.models.native_state import NativeStateContext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--projection-backend',choices=('dense','sparse'),default='dense')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'joint.pt';metrics=args.output/'diagnostic.json'
    if path.exists() or metrics.exists():raise ValueError('diagnostic output already exists; use a new output directory')
    model,mesh,source,marker,u,d,k=joint_fixture(args.seed)
    with torch.no_grad():model.field.decoder.weight.mul_(.001);model.field.decoder.bias.mul_(.001)
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    if args.projection_backend=='sparse':
        from src.models.native_coupling import NativeSourceMesh,TorchMarkerLoads,CoupledNativeField
        from src.models.native_source_features import SourceFeatureScales
        mesh=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,projection='sparse')
        source.mesh=mesh;source.configuration['mesh']=deepcopy(mesh.configuration)
        shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
        local=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,projection='none')
        response=TorchMarkerLoads(local,parameters=model.response.parameters_physics,temperature_floor=1e-8)
        alternate=CoupledNativeField(model.field,response,source_configuration=source.configuration,
            shard_configuration=shard.configuration,source_scales=SourceFeatureScales.physical_reference())
        alternate.source_encoding.projection.load_state_dict(model.source_encoding.projection.state_dict())
        model=alternate
    model.shard_configuration=deepcopy(shard.configuration)
    shards=dict(positions=np.array([[2.6,.6,0.]]),velocities=np.zeros((1,3)),radius=np.array([1e-5]),ablation_atoms_s=np.array([1e12]))
    marker.cloud=MarkerCloud.empty(10,8)
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,'vertex':mesh.vertex+1,'index':mesh.indices},
        boundary={'policy':'learned_unverified'},physics={'feature_parameter_rows':list(k['feature_parameters'].rows)},
        numerics={'scheme':'implicit Euler'},model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.detach().numpy().copy(),u.detach().numpy().copy(),shards,1e-9,1,(1e-9,)*3,
        fluid_dt_previous_si=1e-9,fluid_deltas=d.detach().numpy().copy(),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=k['memory'].detach().numpy().copy())
    started=time.perf_counter();first,_=advance_coupled_reference(state,model=model,source=source,shard_model=shard)
    second,_=advance_coupled_reference(first,model=model,source=source,shard_model=shard)
    elapsed=time.perf_counter()-started
    save_model_rollout(path,first,model)
    fresh=deepcopy(model);restored=load_model_rollout(path,fresh,native_context=context)
    resumed_source=source;resumed_shard=shard
    if args.projection_backend=='sparse':
        new_mesh=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,projection='sparse')
        resumed_source=copy(source);resumed_source.mesh=new_mesh
        resumed_shard=SmallShardModel(new_mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    resumed,_=advance_coupled_reference(restored,model=fresh,source=resumed_source,shard_model=resumed_shard)
    exact=all(np.array_equal(getattr(second,n),getattr(resumed,n)) for n in ('fields_current','fluid_deltas','field_memory'))
    exact=exact and all(np.array_equal(getattr(second.marker.cloud,n),getattr(resumed.marker.cloud,n)) for n in second.marker.cloud.array_names())
    exact=exact and np.array_equal(second.marker.feedback.values,resumed.marker.feedback.values)
    exact=exact and np.array_equal(second.marker.feedback.deltas,resumed.marker.feedback.deltas)
    exact=exact and second.marker.clock==resumed.marker.clock
    from tools.diagnose_native_real_chain import _array_digest
    exact=exact and _array_digest(getattr(second.marker,'transport_history',None))==_array_digest(getattr(resumed.marker,'transport_history',None))
    exact=exact and all(np.array_equal(second.shards[n],resumed.shards[n]) for n in second.shards)
    if not exact:raise AssertionError('resume not bitwise identical')
    result=dict(scope='untrained_small_analytic_source_joint_learning_mechanism',seed=args.seed,
        mode=model.mode,projection_backend=args.projection_backend,configuration=model.checkpoint_configuration,parameters=sum(p.numel() for p in model.parameters()),
        new_source_encoder_parameters=sum(p.numel() for p in model.source_encoding.parameters()),two_event_seconds=elapsed,
        resume_bitwise_equal=exact,source_path='NumPy_non_differentiable_reference',teacher_policies={'source':'self_generated','field':'continuous_model_state'},
        events=[dict(fluid_time_si=s.fluid_time_si,particle_time_si=s.marker.clock.particle_time_si,
            gather_time_si=s.marker.feedback.gather_time_si,live_markers=int(s.marker.cloud.alive.sum()),
            feedback_norms=np.linalg.norm(s.marker.feedback.values.reshape(5,-1),axis=1).tolist(),
            rawD_field_degree_norms=np.linalg.norm(s.fluid_deltas,axis=(2,3)).tolist()) for s in (first,second)])
    metrics.write_text(json.dumps(result,indent=2)+'\n');print(metrics)


if __name__=='__main__':main()
