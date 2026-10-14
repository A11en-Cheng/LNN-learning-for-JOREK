"""Bounded exact A/B: one full source window with event-local coefficients.

No worker threads, neural advance, shard update, training or production wiring.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import time
import traceback

import numpy as np

from tools.benchmark_native_lookup import _compare, _json, _restore, _rng, _rss
from tools.benchmark_native_cpu_threads import complete_components, stable_diagnostics, native_thread_counts


class _MeasuredMesh:
    def __init__(self, native):
        self.native = native
        self.sample_seconds = 0.
        self.sample_calls = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _marker_points(self, *args):
        tick = time.perf_counter()
        try:
            return self.native._marker_points(*args)
        finally:
            self.sample_seconds += time.perf_counter()-tick
            self.sample_calls += 1


def execute(args, result):
    import torch
    from src.models.native_coupling import source_time_sampler
    from src.models.native_coefficient_cache_experiment import NativeCoefficientCachedSource
    from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest
    from tools.native_real_chain_builder import build_real

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split']=='train']
    if len(specs) != 2:
        raise ValueError('exactly two declared train anchors required')
    run = Path(specs[args.run_index]['run']).resolve()
    anchor = (4110, 5680)[args.run_index]
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else '')
    if specs[args.run_index]['anchor'] != anchor or run.name != expected:
        raise ValueError('anchor/run index mismatch')
    stages = Stages()
    controls = _initial_pairing(run, anchor, stages)
    model, source, shard, runtime, provider = build_real(run, anchor, controls, stages, args,
                                                       feedback_mode='autonomous')
    result['native_threads'] = native_thread_counts()
    before = runtime.marker
    if len(before.cloud.weight) != 1000000:
        raise ValueError('full million-slot cloud required')
    target = runtime.fluid_time_si
    def sampler():
        return source_time_sampler(source, before, runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas, completed_dt_si=runtime.fluid_dt_previous_si)
    births = shard.births(runtime.shards, before, target, runtime.fields_previous,
                         runtime.fields_current, source, time_sampler=sampler())
    rng = _rng()
    initial_components = complete_components(before)
    immutable = _array_digest((runtime.fields_previous, runtime.fields_current,
                              runtime.fluid_deltas, runtime.shards, births))
    result.update(run=str(run), anchor=anchor, slots=len(before.cloud.weight),
        initial_live=int(before.cloud.alive.sum()), births_count=len(births.weight),
        target_si=target, source_configuration=source.configuration,
        source_fingerprint=_array_digest(source.configuration),
        initial_components=initial_components, fields_history_shards_births_digest=immutable,
        initial_provenance=before.provenance, rng_initial_digest=_array_digest(rng))
    del model, provider
    reference = None
    reference_components = None
    reference_diagnostics = None
    reference_rng = None
    result['iterations'] = []
    cached = NativeCoefficientCachedSource(source, max_cache_bytes=args.cache_mib*1024**2)
    result['candidate_configuration'] = cached.configuration

    def advance(mode, repeat, warmup=False):
        nonlocal reference, reference_components, reference_diagnostics, reference_rng
        # Same state/birth clone and fresh sampler, outside the source timer.
        state, born, sample = deepcopy(before), deepcopy(births), sampler()
        _restore(rng)
        if mode == 'baseline':
            mesh = _MeasuredMesh(source.mesh)
            candidate = copy(source)
            candidate.mesh = mesh
        else:
            candidate = cached
        cpu = time.process_time()
        tick = time.perf_counter()
        out, diag = candidate.advance(state, target, runtime.fields_previous,
                                     runtime.fields_current, births=born, time_sampler=sample)
        seconds = time.perf_counter()-tick
        cpu_seconds = time.process_time()-cpu
        components = complete_components(out)
        diagnostic = _array_digest(stable_diagnostics(diag))
        final_rng = _array_digest(_rng())
        if reference is None:
            reference = out
            reference_components, reference_diagnostics, reference_rng = components, diagnostic, final_rng
        sample_stats = (dict(sample_seconds=mesh.sample_seconds, sample_calls=mesh.sample_calls)
                        if mode == 'baseline' else deepcopy(cached.last_stats))
        entry = dict(mode=mode, repeat=repeat, warmup=warmup, seconds=seconds,
            process_cpu_seconds=cpu_seconds, sampling=sample_stats,
            components=components, physical_diagnostic_digest=diagnostic, rng_digest=final_rng,
            output_exact=components==reference_components,
            diagnostics_exact=diagnostic==reference_diagnostics, rng_exact=final_rng==reference_rng,
            particle_steps=diag['n_steps'], live_count=int(out.cloud.alive.sum()), peak_rss_bytes=_rss())
        entry['exact'] = entry['output_exact'] and entry['diagnostics_exact'] and entry['rng_exact']
        result['iterations'].append(entry)
        print(json.dumps({k:entry[k] for k in ('mode','repeat','warmup','seconds','exact','sampling')}), flush=True)
        if not entry['exact'] and 'first_difference' not in result:
            result['first_difference'] = _compare(reference, out, before)
        if _rss() > args.memory_mib*1024**2:
            raise MemoryError('RSS budget exceeded')
    advance('baseline', -1, True)
    advance('cache', -1, True)
    for repeat in range(args.repeats):
        for mode in (('baseline', 'cache') if repeat%2==0 else ('cache', 'baseline')):
            advance(mode, repeat)
    result['summary'] = []
    for mode in ('baseline', 'cache'):
        rows = [row for row in result['iterations'] if row['mode']==mode and not row['warmup']]
        result['summary'].append(dict(mode=mode, seconds=[row['seconds'] for row in rows],
            median_seconds=statistics.median(row['seconds'] for row in rows),
            median_sample_seconds=statistics.median(row['sampling']['sample_seconds'] for row in rows),
            exact=all(row['exact'] for row in rows)))
    result['inputs_unchanged'] = (initial_components==complete_components(before) and
        immutable==_array_digest((runtime.fields_previous, runtime.fields_current,
                                  runtime.fluid_deltas, runtime.shards, births)))
    result['success'] = result['inputs_unchanged'] and all(row['exact'] for row in result['iterations'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    p.add_argument('--run-index', type=int, choices=(0,1), default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--chunk-size', type=int, default=8192)
    p.add_argument('--cache-mib', type=int, default=64)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--wall-seconds', type=int, default=600)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (not 1<=args.repeats<=3 or not 0<args.chunk_size<=1000000 or
        not 0<args.cache_mib<=128 or not 0<args.memory_mib<=8192 or
        not 0<args.wall_seconds<=600 or args.birth_max_proposals<=0):
        p.error('invalid bounded configuration')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            p.error(name+'=1 required before process startup')
    args.spatial_backend = 'batched_v1'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema='native-coefficient-cache-benchmark-v1', success=False,
        scope='first_full_real_source_window_fixed_U_rawD_not_neural_or_shard_chain',
        configuration=vars(args).copy(), training=False, optimizer_updates=0,
        production_qualified=False, numpy_version=np.__version__,
        exact_exclusions=['execution.elapsed_seconds','execution.peak_rss_bytes'],
        native_source_sha256=hashlib.sha256(Path('src/models/native_coupling.py').read_bytes()).hexdigest(),
        cache_module_sha256=hashlib.sha256(Path('src/models/native_coefficient_cache_experiment.py').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def timeout(*_):
        raise TimeoutError('main-thread wall deadline exceeded (native calls may defer interrupt)')
    with args.output.open('x') as stream:
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.wall_seconds)
        tick = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            result['total_seconds'] = time.perf_counter()-tick
            result['peak_rss_bytes'] = _rss()
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
            stream.write('\n')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
