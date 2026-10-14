#!/usr/bin/env python3
"""Untrained full-axis field-only CfC BPTT/cost diagnostic; no feedback or optimizer."""
import argparse
from pathlib import Path
import sys
import time
import statistics
import resource
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.diagnose_native_features import (read_segment,atomic_json,build_windows,SplitBlock,WindowConfig,
    NativeScaler,read_native_feature_evaluator,NativeFeaturePath,build_fem_stencil_geometry,
    read_native_update_mapping,NativeFieldStep,parameters_from_current_input)
from src.models.native_temporal import CfCTimeCore
from src.models.native_torch_rollout import initialize_field_state,advance_field,rollout_field_window


def real_fixture(suffix,first,h):
    run=Path('data/EHL-2')/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
    records,_=read_segment(run,range(first,first+4),restart_events=(first,),gamma=5/3,
        gamma_provenance='explicit_untrained_temporal_component',modes=(0,))
    units=records[0].state.units
    windows=build_windows(records,[SplitBlock(run.name,'train',first,first+3)],WindowConfig(1,2,1,0),
        time_plan={(run.name,i):h*units.t_norm_si for i in range(first,first+4)},
        plan_provenance='declared_segment_schedule',scheme='Gears').windows
    ev=read_native_feature_evaluator(run/f'jorek{first:06d}.h5',units=units)
    features=NativeFeaturePath.fit(ev,windows,scenario_enabled=True,physics_enabled=True)
    scales=NativeScaler.fit({'train':list(windows[0].support_records)})
    geom=build_fem_stencil_geometry(ev.vertex.numpy().T,ev.node_positions.numpy())
    mapping=read_native_update_mapping(run/f'jorek{first:06d}.h5',keep_n0_const=False,boundary_policy='learned_unverified')
    model=NativeFieldStep(geom,mapping,units=units,state_scale=scales.value_scale[:,:,0,0],
        increment_scale=scales.history_scale[:,:,0,0],center=scales.value_center[:,:,0,0],
        hidden=12,n_layers=1,n_iters=2,feature_path=features,
        temporal_core=CfCTimeCore(12,backbone_units=24,time_scale_ju=1.))
    return model,windows[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1: parser.error('positive repeats required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(41)
    records=[]
    for suffix,first,h in [('',4111,.05),('_injInner',2704,5.)]:
        start=time.perf_counter();model,w=real_fixture(suffix,first,h);setup=time.perf_counter()-start
        view=w.current_input();p=parameters_from_current_input(view,model.feature_path)
        state=initialize_field_state(model,view.state,previous_dt_si=view.previous_dt_si,scheme=view.scheme)
        for a in (state.values,state.raw_deltas,state.memory): a.requires_grad_()
        metrics={}
        for label in ('single','two_step'):
            def forward():
                if label=='single': return advance_field(model,state,dt_si=view.dt_si,parameters=p)
                return rollout_field_window(model,w,initial_state=state,teacher_mask=(False,False)).states[-1]
            out=forward();(out.values/model.state_scale).square().mean().backward()
            ft=[];bt=[]
            for _ in range(args.repeats):
                model.zero_grad(set_to_none=True)
                for a in (state.values,state.raw_deltas,state.memory): a.grad=None
                start=time.perf_counter();out=forward();ft.append(time.perf_counter()-start)
                start=time.perf_counter();(out.values/model.state_scale).square().mean().backward();bt.append(time.perf_counter()-start)
            metrics[label]=dict(forward_seconds=statistics.median(ft),backward_seconds=statistics.median(bt))
        coverage=lambda a:(a.abs().sum(-1)>0).tolist()
        records.append(dict(run=w.run,support=list(w.support_indices),prediction_index=w.prediction_index,
            scope='field_only_cold_start_K2_no_feedback_no_optimizer',shape=list(state.values.shape),
            memory_shape=list(state.memory.shape),dtype=str(state.values.dtype),device='cpu',threads=1,
            torch_version=torch.__version__,core=model.temporal_core.checkpoint_configuration,
            parameters=sum(p.numel() for p in model.parameters()),
            core_parameters=sum(p.numel() for p in model.temporal_core.parameters()),
            setup_seconds=setup,repeats=args.repeats,cost=metrics,
            values_gradient_coverage=coverage(state.values.grad),rawD_gradient_coverage=coverage(state.raw_deltas.grad),
            initial_memory_gradient_nonzero=bool(state.memory.grad.abs().sum()>0),
            parameter_gradients_finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            caveat='untrained structure, missing feedback history and unverified physical BC; no accuracy/stability/C0 claim'))
    atomic_json(args.output_dir/'diagnostics.json',records)
    for r in records: print(r['run'],r['parameters'],r['cost'],r['process_peak_rss_kib'])


if __name__=='__main__': main()
