#!/usr/bin/env python3
"""CPU float64 G/P component diagnostic. No optimizer, training or future inputs."""
import argparse
from pathlib import Path
import sys
import time
import statistics
import resource
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_window_io import read_segment,atomic_json
from src.data.native_windows import build_windows,SplitBlock,WindowConfig,NativeScaler
from src.data.native_features import read_native_feature_evaluator,parameters_from_current_input
from src.data.native_state import read_native_update_mapping
from src.models.native_features import NativeFeaturePath
from src.models.native_field_step import NativeFieldStep
from src.models.stencil_conv import build_fem_stencil_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1: p.error('repeats must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(7)
    results=[]
    for suffix,first,h in [('',4111,.05),('_injInner',2704,5.)]:
        run=Path('data/EHL-2')/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        start=time.perf_counter()
        records,_=read_segment(run,range(first,first+4),restart_events=(first,),gamma=5/3,
            gamma_provenance='explicit_component_diagnostic_configuration',modes=(0,))
        units=records[0].state.units
        # Each run is a separate component fixture with a declared train block;
        # these statistics are NOT the earlier W benchmark's validation split.
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
            hidden=12,n_layers=1,n_iters=2,feature_path=features)
        setup=time.perf_counter()-start
        view=windows[0].current_input(); params=parameters_from_current_input(view,features)
        u=torch.tensor(view.state.values,requires_grad=True); d=torch.tensor(view.state.raw_deltas,requires_grad=True)
        def step():
            return model(u,d,dt_si=view.dt_si,previous_dt_si=view.previous_dt_si,scheme=view.scheme,feature_parameters=params)
        step().increment.square().mean().backward()  # diagnostic warm-up only
        forwards=[]; backwards=[]
        for _ in range(args.repeats):
            model.zero_grad(set_to_none=True); u.grad=None; d.grad=None
            t=time.perf_counter(); out=step(); forwards.append(time.perf_counter()-t)
            t=time.perf_counter(); out.increment.square().mean().backward(); backwards.append(time.perf_counter()-t)
        groups={}
        for name,param in model.named_parameters():
            group=name.split('.')[0]; groups[group]=groups.get(group,0)+param.numel()
        coverage=lambda a:(a.abs().sum(-1)>0).tolist()
        results.append(dict(run=run.name,scope=[first,first+3],prediction_index=view.state.step_index,
            dtype=str(u.dtype),device='cpu',torch_version=torch.__version__,threads=torch.get_num_threads(),
            coefficient_shape=list(u.shape),gauss_shape=[len(ev.vertex),16,9],gradient_shape=[len(ev.vertex),16,9,2],
            node_feature_shape=[ev.n_nodes,27],parameter_count=sum(p.numel() for p in model.parameters()),parameter_groups=groups,
            parameter_gradient_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
            nonzero_parameter_gradient_count=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum()>0),
            values_degree_gradient_coverage=coverage(u.grad),history_degree_gradient_coverage=coverage(d.grad),
            setup_io_geometry_statistics_model_seconds=setup,repeats=args.repeats,
            forward_median_seconds=statistics.median(forwards),backward_median_seconds=statistics.median(backwards),
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            caveat='prescribed existing fields, component gradients only; no self-generated feedback or prediction claim'))
        features.save_statistics(args.output_dir/(run.name+'-statistics.json'))
    atomic_json(args.output_dir/'diagnostics.json',results)
    for r in results:
        print(r['run'],r['parameter_count'],'parameters; forward',round(r['forward_median_seconds'],6),
              'backward',round(r['backward_median_seconds'],6),'seconds')


if __name__=='__main__': main()
