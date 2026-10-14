"""Bounded saved-neighbour nearby/current A/B on complete real source windows.

No training. Discrete changes are reported, not waived by float tolerances.
Build, fixed seeded births and comparisons are outside advance-only timers.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np

from tools.benchmark_native_lookup import _json, _rss
from tools.benchmark_native_neighbor_locator import benchmark_source_window, _rng, _restore_rng
from tools.diagnose_native_real_chain import _array_digest


def execute(args,result):
    import torch
    import h5py
    from src.models.native_coupling import source_time_sampler
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    from tools.benchmark_native_cpu_threads import native_thread_counts
    from tools.diagnose_native_real_chain import Stages,_initial_pairing
    from tools.native_real_chain_builder import build_real
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    specs=[w for w in json.loads(args.config.read_text())['windows'] if w['split']=='train']
    if len(specs)!=2:raise ValueError('exactly two declared train anchors required')
    spec=specs[args.run_index];run=Path(spec['run']).resolve();anchor=(4110,5680)[args.run_index]
    expected='phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else '')
    if spec['anchor']!=anchor or run.name!=expected:raise ValueError('run/anchor mismatch')
    stages=Stages();controls=_initial_pairing(run,anchor,stages)
    model,source,shard,runtime,provider=build_real(run,anchor,controls,stages,args,feedback_mode='autonomous')
    field=run/f'jorek{anchor+1:06d}.h5'
    with h5py.File(field) as f:
        neighbours=f['neighbours'][...]
        if not np.array_equal(f['vertex'][...]-1,source.mesh.vertex):
            raise ValueError('saved topology must belong to the actual source mesh')
    nearby=NativeNearbyMarkerSource(source,neighbours=neighbours)
    before=runtime.marker;target=runtime.fluid_time_si
    if len(before.cloud.weight)!=1000000:raise ValueError('full million-slot cloud required')
    def sampler():
        return source_time_sampler(source,before,runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas,completed_dt_si=runtime.fluid_dt_previous_si)
    births=shard.births(runtime.shards,before,target,runtime.fields_previous,runtime.fields_current,
        source,time_sampler=sampler())
    result.update(run=str(run),anchor=anchor,slots=len(before.cloud.weight),
        initial_live=int(before.cloud.alive.sum()),births_count=len(births.weight),target_si=target,
        native_threads=native_thread_counts(),saved_neighbours_source=str(field)+':/neighbours',
        saved_neighbours_digest=_array_digest(neighbours),
        saved_neighbours_counts=dict(positive=int((neighbours>0).sum()),zero=int((neighbours==0).sum()),negative=int((neighbours<0).sum())),
        baseline_configuration=deepcopy(source.configuration),candidate_configuration=deepcopy(nearby.configuration),
        input_marker_digest=_array_digest(before),birth_digest=_array_digest(births),
        field_digest=_array_digest((runtime.fields_previous,runtime.fields_current)),
        initial_provenance=deepcopy(before.provenance))
    other=(runtime.fluid_deltas,runtime.shards,source.configuration,neighbours,
           source.mesh.coordinates,source.mesh.size,source.mesh.vertex,source.mesh.indices)
    immutable=_array_digest(other);rng=_rng()
    def progress(row):print(json.dumps(row,default=_json),flush=True)
    try:
        benchmark_source_window(source,nearby,before,target,runtime.fields_previous,runtime.fields_current,
            births=births,sampler_factory=sampler,repeats=args.repeats,count_pass=True,
            progress=progress,memory_bytes=args.memory_mib*1024**2,record=result)
    finally:
        _restore_rng(rng)
    # Reused harness calls its experimental arm 'neighbor'; label it accurately.
    for row in result['iterations']+result.get('summary',[]):
        if row['mode']=='neighbor':row['mode']='jorek_nearby'
    for key in ('locator_counts','count_pass_comparisons'):
        if 'neighbor' in result.get(key,{}):result[key]['jorek_nearby']=result[key].pop('neighbor')
    result['other_inputs_unchanged']=immutable==_array_digest(other)
    result['success'] &= result['other_inputs_unchanged']
    result['experiment_executed']=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('config/native-autonomous-training-pilot.json'))
    p.add_argument('--run-index',type=int,choices=(0,1),default=0)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--memory-mib',type=int,default=8192)
    p.add_argument('--wall-seconds',type=int,default=600)
    p.add_argument('--birth-max-proposals',type=int,default=1000000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if (not 1<=args.repeats<=3 or not 0<args.chunk_size<=1000000
        or not 0<args.memory_mib<=8192 or not 0<args.wall_seconds<=900
        or not 0<args.birth_max_proposals<=1000000):p.error('invalid bounded configuration')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':p.error(key+'=1 required before process startup')
    args.spatial_backend='batched_v1';args.marker_locator='native'
    repo=Path(__file__).resolve().parents[1]
    paths={k:repo/v for k,v in dict(nearby='src/models/native_nearby_locator.py',
        native='src/models/native_coupling.py',builder='tools/native_real_chain_builder.py',
        comparison='tools/benchmark_native_neighbor_locator.py',tool='tools/benchmark_native_nearby.py').items()}
    paths['jorek_nearby']=Path('/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90')
    paths['jorek_neighbours']=Path('/home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90')
    hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    result=dict(schema='native-jorek-nearby-benchmark-v1',success=False,experiment_executed=False,
        training=False,optimizer_updates=0,production_qualified=False,configuration=vars(args),
        scope='complete_source_window_fixed_fields_rawD_no_NN_or_shard_prediction',
        source_code_sha256=hashes,source_code_paths={k:str(v) for k,v in paths.items()},
        exact_exclusions=['diagnostics.execution.elapsed_seconds','diagnostics.execution.peak_rss_bytes'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def timeout(*_):raise TimeoutError('cooperative wall deadline exceeded')
    with args.output.open('x') as stream:
        old=signal.signal(signal.SIGALRM,timeout);signal.alarm(args.wall_seconds);start=time.perf_counter()
        try:execute(args,result)
        except Exception as exc:
            result.update(success=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,old)
            result['code_unchanged']=all(hashlib.sha256(v.read_bytes()).hexdigest()==hashes[k] for k,v in paths.items())
            result['success'] &= result['code_unchanged']
            result.update(total_seconds=time.perf_counter()-start,peak_rss_bytes=_rss())
            json.dump(result,stream,default=_json,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
