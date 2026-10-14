"""Paired static sampling and autonomous two-event function-cache benchmark.

A common native first event is executed once, explicitly reported as prefix.
Each fork then executes every particle/shard/field event without future truth.
Fixed untrained weights, no optimizer, no particle cropping or step changes.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import statistics
import time

import numpy as np
import torch

from src.models.native_element_functions import (
    CompiledElementFunctions, NativeElementFunctionSource, build_kernel,
)
from tools.benchmark_native_multi_window import recorded_controls, ROOT
from tools.benchmark_native_lookup import _compare, _rng, _restore
from tools.diagnose_native_real_chain import Stages, _array_digest
from tools.profile_native_coupled_latency import write_json
from tools.native_replay_metrics import relative_metrics


def sampling_probe(source, runtime, kernel, repeats=30):
    """Same exact positions/owners, preparation excluded only from warm query."""
    native = source.source if hasattr(source, 'locator') else source
    ids = np.flatnonzero(runtime.marker.cloud.alive)
    e = runtime.marker.cloud.i_elm[ids].astype(np.int64)-1
    st = runtime.marker.cloud.st[ids]
    fields = runtime.fields_current.copy(); fields.flags.writeable = False
    feedback = runtime.marker.feedback.values.copy(); feedback.flags.writeable = False
    wrapper = NativeElementFunctionSource(source, kernel=kernel)
    tick = time.perf_counter(); cache = wrapper.sampling_cache()
    expected = native.mesh._marker_points(e, st, fields, feedback)
    actual = cache._marker_points(e, st, fields, feedback)
    cold = time.perf_counter()-tick
    precision = dict(values=relative_metrics(expected[0], actual[0]),
        gradients=relative_metrics(expected[1], actual[1]),
        feedback=relative_metrics(expected[2], actual[2]),
        coordinates=relative_metrics(expected[3], actual[3]), channels={})
    for channel in (0,1,4,6,7,8):
        precision['channels'][str(channel)] = dict(
            values=relative_metrics(expected[0][:,channel], actual[0][:,channel]),
            gradients={axis: relative_metrics(expected[1][:,channel,j], actual[1][:,channel,j])
                       for j,axis in enumerate(('R','Z'))})
    rows=[]
    # Singleton fine groups and normal 2048 chunks are distinct workloads.
    for count in (1, min(2048,len(e)), len(e)):
        times={'native':[], 'functions':[]}
        for repeat in range(repeats):
            order=('native','functions') if repeat%2 == 0 else ('functions','native')
            for mode in order:
                call = native.mesh._marker_points if mode == 'native' else cache._marker_points
                tick=time.perf_counter(); call(e[:count],st[:count],fields,feedback)
                times[mode].append(time.perf_counter()-tick)
        row=dict(points=count, repeats=repeats,
            median_seconds={k:statistics.median(v) for k,v in times.items()}, timings=times)
        row['speedup']=row['median_seconds']['native']/row['median_seconds']['functions']
        rows.append(row)
    return dict(scope='initial_live_exact_positions_not_particle_transport', live=len(ids),
        precision=precision, cold_prepare_and_both_oracles_seconds=cold,
        coefficient_prepare_seconds=cache.stats['build_seconds'],
        coefficient_bytes=cache.stats['coefficient_bytes'], peak_cache_workspace_bytes=cache.stats['peak_coefficient_bytes'],
        warm_queries=rows)


def compare_states(reference, candidate, before):
    """Native-unit relative evidence; no all-zero epsilon denominator."""
    report=_compare(reference.runtime.marker,candidate.runtime.marker,before.runtime.marker)
    report['fields']={name: relative_metrics(getattr(reference.runtime,name),getattr(candidate.runtime,name))
        for name in ('fields_current','fields_previous','fluid_deltas','field_memory')}
    report['shards']={name: relative_metrics(reference.runtime.shards[name],candidate.runtime.shards[name])
        for name in ('positions','velocities','radius','ablation_atoms_s')}
    a,b=reference.runtime.marker.transport_history,candidate.runtime.marker.transport_history
    report['transport_history_exact']=_array_digest(a)==_array_digest(b)
    if a is not None and b is not None:
        changed=np.flatnonzero(a.previous_dt_si!=b.previous_dt_si)
        report['transport_history']=dict(
            level_exact=bool(np.array_equal(a.level,b.level)),
            lifetime_exact=bool(np.array_equal(a.lifetime,b.lifetime)),
            configuration_exact=_array_digest(a.configuration)==_array_digest(b.configuration),
            initialization_exact=_array_digest(a.initialization)==_array_digest(b.initialization),
            previous_dt=relative_metrics(a.previous_dt_si,b.previous_dt_si),
            changed_previous_dt_count=len(changed),
            changed_previous_dt=(relative_metrics(a.previous_dt_si[changed],b.previous_dt_si[changed])
                                 if len(changed) else None),
            first_changed_slots=[dict(slot=int(i),reference_dt_si=float(a.previous_dt_si[i]),
                candidate_dt_si=float(b.previous_dt_si[i]),reference_element=int(reference.runtime.marker.cloud.i_elm[i]),
                candidate_element=int(candidate.runtime.marker.cloud.i_elm[i])) for i in changed[:32]])
    report['integer_events_exact']=all(report['cloud_full_slots'][name]['array_equal']
        for name in ('i_elm','i_life','weight','t_birth'))
    return report


def acceptance(report):
    """Explicit diagnostic criterion, not a production accuracy certificate."""
    numbers=[]
    for section in ('fields','shards'):
        numbers.extend(x['relative_l2'] for x in report[section].values())
    for kind in ('values','deltas'):
        numbers.extend(x['relative_metrics']['relative_l2'] for x in report['feedback'][kind].values())
    for name in ('x','st','V_prev','P_imp'):
        numbers.append(report['cloud_full_slots'][name]['relative_metrics']['relative_l2'])
    return dict(threshold_relative_l2=1e-8, scope='full_array_norms_not_every_pointwise_component',
        passed=bool(report['integer_events_exact'] and report['clocks_exact']
            and report['transport_history_exact'] and all(x is not None and x<=1e-8 for x in numbers)),
        max_relative_l2=max((x for x in numbers if x is not None),default=None))


def execute(args,result):
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.native_real_chain_builder import build_real
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required when selected; no silent CPU fallback')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run=ROOT/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else ''))
    anchor=5680 if args.run_index else 4110
    tick=time.perf_counter()
    kernel=CompiledElementFunctions(build_kernel(args.output_dir/'functions.so'))
    result.update(kernel_build_seconds=time.perf_counter()-tick, kernel=kernel.configuration,
        device=str(device), torch_version=torch.__version__, numpy_version=np.__version__)
    stages=Stages(); rows=recorded_controls(run,anchor,getattr(args,'forecast_events',2)+1,stages)
    tick=time.perf_counter()
    model,source,shard,runtime,provider=build_real(run,anchor,rows,stages,args,feedback_mode='autonomous')
    model.to(device).eval()
    result.update(build_seconds=time.perf_counter()-tick, source_configuration=source.configuration,
        controls=rows, initial_live=int(runtime.marker.cloud.alive.sum()), slots=len(runtime.marker.cloud.weight))
    result['sampling']=sampling_probe(source,runtime,kernel)
    write_json(args.output_dir/'sampling.json',result['sampling'])
    print(json.dumps(dict(stage='sampling', warm_queries=[{k:r[k] for k in ('points','median_seconds','speedup')}
        for r in result['sampling']['warm_queries']], coefficient_bytes=result['sampling']['coefficient_bytes'])),flush=True)
    controls=tuple(StepControl(r['start_si'],r['dt_si'],r['end_si'],r['origin'],r['source'],phase=r['phase']) for r in rows)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync=lambda: torch.cuda.synchronize(device) if device.type=='cuda' else None
    original_weights=_array_digest(model.state_dict())
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=provider)
    with torch.no_grad():
        state=adapter.initialize(runtime)
        print(json.dumps(dict(stage='common_native_prefix', status='started')),flush=True)
        sync(); tick=time.perf_counter(); prefix,_=adapter.advance(state,controls[0]); sync()
        result['common_native_prefix_seconds']=time.perf_counter()-tick
        print(json.dumps(dict(stage='common_native_prefix', seconds=result['common_native_prefix_seconds'])),flush=True)
    rng=_rng(); prefix_digest=_array_digest(prefix)
    candidate=NativeElementFunctionSource(source,kernel=kernel)
    result['candidate_configuration']=candidate.configuration
    iterations=[]
    # Only diagnostic fork context changes: explicit source numerical fingerprint.
    for repeat in range(args.repeats):
        outputs={}
        for mode in (('native','functions') if repeat%2==0 else ('functions','native')):
            selected=source if mode=='native' else candidate
            model.source_configuration=deepcopy(selected.configuration)
            fork_runtime=deepcopy(prefix.runtime)
            fork_runtime.native_context.configuration['model']=model.checkpoint_configuration
            adapter=TorchRecomputedSourceEvents(model,selected,shard,equilibrium_provider=provider)
            state=adapter.initialize(fork_runtime); _restore(rng)
            events=[]
            with torch.no_grad():
                for i,control in enumerate(controls[1:],2):
                    print(json.dumps(dict(stage='event',mode=mode,repeat=repeat,event=i,status='started')),flush=True)
                    sync(); tick=time.perf_counter(); state,diag=adapter.advance(state,control); sync()
                    row=dict(event=i,seconds=time.perf_counter()-tick,
                        source_execution=diag.get('execution'),live=int(state.runtime.marker.cloud.alive.sum()),
                        boundary_exits=diag.get('boundary_exits',[]),
                        sampling=None if mode=='native' else candidate.last_stats)
                    events.append(row)
                    print(json.dumps(dict(stage='event',mode=mode,repeat=repeat,event=i,seconds=row['seconds'],live=row['live'])),flush=True)
            outputs[mode]=(state,_array_digest(_rng()))
            iterations.append(dict(mode=mode,repeat=repeat,events=events))
            result['iterations']=iterations
            write_json(args.output_dir/f'progress-{repeat}-{mode}.json',result)
        report=compare_states(outputs['native'][0],outputs['functions'][0],prefix)
        verdict=acceptance(report)
        entry=dict(repeat=repeat,comparison=report,acceptance=verdict,
            rng_exact=outputs['native'][1]==outputs['functions'][1])
        result.setdefault('comparisons',[]).append(entry)
        write_json(args.output_dir/f'comparison-{repeat}.json',entry)
        print(json.dumps(dict(stage='comparison',repeat=repeat,acceptance=verdict,rng_exact=entry['rng_exact'])),flush=True)
        # Stop extra timing repetition on failed conformance; keep evidence.
        if not verdict['passed'] or not entry['rng_exact']:
            break
    result['summary']={mode:dict(event_seconds=[r['seconds'] for it in iterations if it['mode']==mode for r in it['events']],
        median_event_seconds=statistics.median(r['seconds'] for it in iterations if it['mode']==mode for r in it['events']))
        for mode in ('native','functions')}
    result['summary']['paired_total_speedup']=sum(result['summary']['native']['event_seconds'])/sum(result['summary']['functions']['event_seconds'])
    result.update(prefix_unchanged=prefix_digest==_array_digest(prefix),
        weights_unchanged=original_weights==_array_digest(model.state_dict()))
    result['success']=(result['prefix_unchanged'] and result['weights_unchanged'] and
        all(c['acceptance']['passed'] and c['rng_exact'] for c in result['comparisons']))
    if device.type=='cuda':
        result.update(cuda_device=torch.cuda.get_device_name(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index',type=int,choices=(0,1),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--repeats',type=int,choices=(1,2),default=2)
    p.add_argument('--forecast-events',type=int,choices=(1,2),default=2)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--memory-mib',type=int,default=8192)
    p.add_argument('--wall-seconds',type=int,default=900)
    p.add_argument('--birth-max-proposals',type=int,default=1000000)
    p.add_argument('--birth-cells-per-width',type=int,default=64)
    args=p.parse_args()
    if not 0<args.wall_seconds<=900 or min(args.chunk_size,args.threads,args.memory_mib,args.birth_max_proposals,args.birth_cells_per_width)<=0:
        p.error('positive bounded budgets required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1': p.error(key+'=1 required at process startup')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    args.spatial_backend='batched_v1'; args.marker_locator='jorek_nearby_v1'
    args.marker_transport='local_subcycling_boundary_v2'
    args.output_dir.mkdir(parents=True,exist_ok=False)
    paths=[ROOT/'src/models/native_element_functions.py',ROOT/'src/models/native_element_functions.cpp',Path(__file__)]
    result=dict(schema='native-element-function-benchmark-v1',success=False,training=False,
        optimizer_updates=0,backward_calls=0,future_truth_feedback=False,production_qualified=False,
        scope='common_untrained_native_prefix_then_paired_two_event_autonomous_forks',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    def timeout(*_): raise TimeoutError('bounded function-cache benchmark deadline exceeded')
    previous=signal.signal(signal.SIGALRM,timeout); signal.alarm(args.wall_seconds)
    start=time.perf_counter()
    try:
        execute(args,result)
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__,error=str(exc)); traceback.print_exc()
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        result.update(elapsed_seconds=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            hashes_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in result['hashes'].items()))
        write_json(args.output_dir/'result.json',result)
    return 0 if result['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
