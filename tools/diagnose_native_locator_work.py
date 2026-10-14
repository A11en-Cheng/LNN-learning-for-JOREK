"""Read-only work census for one real, full-cloud source window; no training.

Only locator boundaries and KD-tree queries are wrapped. Geometry methods stay
native: overriding them would disable accepted-trial geometry reuse. Timings
include census overhead and are not an independent speed benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import AbstractContextManager
import hashlib
import json
from pathlib import Path
import random
import resource
import signal
import time
import traceback

import numpy as np


class _TreeProbe:
    def __init__(self, tree, owner):
        self.tree = tree; self.owner = owner

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def query_ball_point(self, *args, **kwargs):
        result = self.tree.query_ball_point(*args, **kwargs)
        if self.owner.depth:
            self.owner.candidates[len(result)] += 1
        return result


class LocatorProbe(AbstractContextManager):
    def __init__(self, mesh):
        self.mesh = mesh; self.depth = 0; self.saved = []
        self.counts = Counter(); self.candidates = Counter(); self.fallback_sizes = Counter()
        self.batch_seconds = 0.; self.fallback_seconds = 0.

    def _install(self, name, value):
        self.saved.append((name, name in self.mesh.__dict__, self.mesh.__dict__.get(name)))
        setattr(self.mesh, name, value)

    def __enter__(self):
        mesh = self.mesh
        scalar = mesh.locate; batch = mesh.locate_batch
        def measured_scalar(positions, **kwargs):
            if not self.depth:
                return scalar(positions, **kwargs)
            tick = time.perf_counter()
            n = len(positions); self.counts['fallback_calls'] += 1
            self.counts['fallback_point_visits'] += n; self.fallback_sizes[n] += 1
            hints = kwargs.get('element_hints')
            if hints is not None:
                e = np.asarray(hints).astype(np.int64)-1
                valid = (e>=0)&(e<len(mesh.gauss))
                self.counts['fallback_invalid_hints'] += int((~valid).sum())
                if valid.any():
                    bounds = mesh._location_bounds[e[valid]]
                    x = np.asarray(positions)[valid,:2]
                    inside = ((x>=bounds[:,0]-1e-9)&(x<=bounds[:,1]+1e-9)).all(axis=1)
                    self.counts['fallback_inside_hint_bbox'] += int(inside.sum())
                    self.counts['fallback_outside_hint_bbox'] += int((~inside).sum())
            try:
                result = scalar(positions, **kwargs)
                found = result[0]>0
                self.counts['fallback_unresolved'] += int((~found).sum())
                if hints is not None:
                    same = found & (result[0]==hints)
                    self.counts['fallback_same_element'] += int(same.sum())
                    self.counts['fallback_changed_element'] += int((found & ~same).sum())
                return result
            finally:
                self.fallback_seconds += time.perf_counter()-tick
        def measured_batch(positions, **kwargs):
            tick = time.perf_counter()
            self.counts['batch_calls'] += 1; self.counts['point_visits'] += len(positions)
            before = self.counts['fallback_point_visits']
            self.depth += 1
            try:
                result = batch(positions, **kwargs)
                self.counts['unresolved_point_visits'] += int((result[0]==0).sum())
                return result
            finally:
                self.depth -= 1; self.batch_seconds += time.perf_counter()-tick
                self.counts['fast_path_point_visits'] += len(positions)-(self.counts['fallback_point_visits']-before)
        self._install('_location_tree', _TreeProbe(mesh._location_tree, self))
        self._install('locate', measured_scalar)
        self._install('locate_batch', measured_batch)
        return self

    def __exit__(self, *exc):
        for name, owned, previous in reversed(self.saved):
            if owned:
                setattr(self.mesh, name, previous)
            else:
                delattr(self.mesh, name)
        self.saved.clear()

    def summary(self):
        query_calls = sum(self.candidates.values())
        return dict(**{k:int(self.counts[k]) for k in ('batch_calls','point_visits',
            'fallback_calls','fallback_point_visits','fast_path_point_visits',
            'fallback_invalid_hints','fallback_inside_hint_bbox','fallback_outside_hint_bbox',
            'fallback_same_element','fallback_changed_element','fallback_unresolved',
            'unresolved_point_visits')},
            fallback_point_fraction=self.counts['fallback_point_visits']/max(1,self.counts['point_visits']),
            batch_seconds=self.batch_seconds, fallback_seconds=self.fallback_seconds,
            batch_excluding_fallback_seconds=self.batch_seconds-self.fallback_seconds,
            query_point_calls=query_calls,
            mean_tree_candidates=sum(k*v for k,v in self.candidates.items())/max(1,query_calls),
            max_tree_candidates=max(self.candidates,default=0),
            candidate_count_histogram=dict(sorted(self.candidates.items())),
            fallback_batch_size_histogram=dict(sorted(self.fallback_sizes.items())),
            count_scope='particle_substep_visits_not_unique_particles; candidates_before_bbox_filter',
            timing_scope='instrumented_boundary_census_not_speedup_benchmark')


def execute(args, result):
    import torch
    from src.models.native_coupling import source_time_sampler
    from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest
    from tools.native_real_chain_builder import build_real
    from tools.benchmark_native_cpu_threads import complete_components
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split']=='train']
    spec = specs[args.run_index]; run = Path(spec['run']); anchor = spec['anchor']
    stages = Stages(); controls = _initial_pairing(run,anchor,stages)
    model,source,shard,runtime,provider = build_real(run,anchor,controls,stages,args,feedback_mode='autonomous')
    before = _array_digest(runtime); mesh = source.mesh; marker = runtime.marker
    target = runtime.fluid_time_si
    sampler = source_time_sampler(source,marker,runtime.fields_current,
        fluid_deltas=runtime.fluid_deltas,completed_dt_si=runtime.fluid_dt_previous_si)
    born = shard.births(runtime.shards,marker,target,runtime.fields_previous,
                        runtime.fields_current,source,time_sampler=sampler)
    result.update(run=str(run),anchor=anchor,initial_binding=before,
        source_configuration=source.configuration,source_fingerprint=_array_digest(source.configuration),
        slots=len(marker.cloud.weight),initial_live=int(marker.cloud.alive.sum()),births=len(born.weight),
        target_si=target,model_parameters=sum(p.numel() for p in model.parameters()))
    del model,provider
    tick = time.perf_counter()
    with LocatorProbe(mesh) as probe:
        out,diag = source.advance(marker,target,runtime.fields_previous,runtime.fields_current,
                                  births=born,time_sampler=sampler)
    source_seconds = time.perf_counter()-tick
    out.cloud.validate()
    result.update(source_seconds=source_seconds,locator=probe.summary(),
        particle_steps=diag['n_steps'],execution=diag['execution'],sampling=diag.get('sampling'),
        final_components=complete_components(out),final_live=int(out.cloud.alive.sum()),
        inputs_unchanged=before==_array_digest(runtime),geometry_methods_not_wrapped=True)
    result['success'] = (result['inputs_unchanged'] and
        probe.counts['point_visits']==diag['execution']['processed_particle_steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--run-index',type=int,choices=(0,1),default=1)
    parser.add_argument('--chunk-size',type=int,default=2048)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=1031)
    parser.add_argument('--birth-max-proposals',type=int,default=1000000)
    parser.add_argument('--wall-seconds',type=int,default=300)
    parser.add_argument('--memory-mib',type=int,default=8192)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not (0<args.chunk_size<=1000000 and 0<args.threads<=16 and 0<args.wall_seconds<=600
            and 0<args.memory_mib<=8192 and 0<args.birth_max_proposals<=1000000):
        parser.error('invalid bounded resource settings')
    args.spatial_backend = 'batched_v1'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result = dict(schema='native-locator-work-census-v1',success=False,settings=vars(args).copy(),
        scope='first_full_source_window_fixed_initial_fields_not_neural_rollout; births_outside_locator_census',
        training=False,optimizer_updates=0,gpu_used=False,
        coupling_sha256=hashlib.sha256(Path('src/models/native_coupling.py').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def deadline(*_):
        raise TimeoutError('source census wall budget exceeded')
    with args.output.open('x') as stream:
        previous = signal.signal(signal.SIGALRM,deadline); signal.alarm(args.wall_seconds)
        tick = time.perf_counter()
        try:
            execute(args,result)
        except Exception as error:
            result['error'] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM,previous)
            result.update(elapsed_seconds=time.perf_counter()-tick,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            json.dump(result,stream,indent=2,default=str,allow_nan=False)
    return 0 if result['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
