"""Bounded diagnostic only: thread read-only native kernels, not physical steps.

No default backend wiring, neural forecast, backward or optimizer update.
Run with OPENBLAS/OMP/MKL_NUM_THREADS=1 to avoid nested parallelism.
"""
from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import threading
import time
import traceback

import numpy as np

from tools.benchmark_native_lookup import _components, _json, _restore, _rng, _rss


def stable_diagnostics(diagnostic):
    """Exclude only nondeterministic resource counters, not physical diagnostics."""
    stable = deepcopy(diagnostic)
    execution = stable.get('execution', {})
    for key in ('elapsed_seconds', 'peak_rss_bytes'):
        execution.pop(key, None)
    return stable


def complete_components(marker):
    result = _components(marker)
    from tools.diagnose_native_real_chain import _array_digest
    result['cloud_group'] = _array_digest(marker.cloud.group)
    return result


def native_thread_counts():
    """Query loaded OpenBLAS libraries; do not change their runtime settings."""
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if 'openblas' in line and line.split()[-1].startswith('/')})
    evidence = []
    for path in paths:
        library = ctypes.CDLL(path)
        for symbol in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads'):
            if hasattr(library, symbol):
                query = getattr(library, symbol)
                query.restype = ctypes.c_int
                query.argtypes = []
                evidence.append(dict(path=path, symbol=symbol, threads=query()))
                break
        else:
            raise ValueError('cannot verify loaded OpenBLAS thread count: '+path)
    if not evidence or any(item['threads'] != 1 for item in evidence):
        raise ValueError('loaded OpenBLAS libraries must all use one thread')
    return evidence


class ThreadedKernels:
    """Event-owned executor; read-only inputs, ordered gather, serial deposition.

    Only the native batched CPU source is supported. This does not share a
    mutable experimental lookup cache or run independent events concurrently.
    """
    def __init__(self, source, workers, *, kernels=('sample', 'locate', 'atomic')):
        from src.models.native_coupling import NativeSourceMesh
        if (source.spatial_backend != 'batched_v1' or source._spatial is not None or
                type(source.mesh) is not NativeSourceMesh):
            raise ValueError('native batched CPU source required')
        if workers not in (1, 2, 4, 8):
            raise ValueError('workers must be 1/2/4/8')
        if not set(kernels) <= {'sample', 'locate', 'atomic'}:
            raise ValueError('unknown kernel')
        self.source = source
        self.workers = workers
        self.kernels = set(kernels)
        self.pool = None
        self.timings = {}
        self.calls = {}
        self.rows = {}
        self.worker_ids = set()
        self.id_lock = threading.Lock()

    def __enter__(self):
        if self.workers > 1:
            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix='native-benchmark')
            barrier = threading.Barrier(self.workers)
            def ready():
                with self.id_lock:
                    self.worker_ids.add(threading.get_ident())
                barrier.wait(timeout=30)
            futures = [self.pool.submit(ready) for _ in range(self.workers)]
            for future in futures:
                future.result()
        self.adapter = copy(self.source)
        self.adapter.mesh = _ThreadMesh(self)
        self.adapter.atomic = self.atomic
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def dispatch(self, name, length, operation):
        tick = time.perf_counter()
        try:
            if self.pool is None or name not in self.kernels or length < self.workers:
                return operation(slice(0, length))
            edges = np.linspace(0, length, self.workers+1, dtype=int)
            futures = [self.pool.submit(operation, slice(int(a), int(b)))
                       for a, b in zip(edges[:-1], edges[1:])]
            parts = [future.result() for future in futures]
            return tuple(np.concatenate(items, axis=0) for items in zip(*parts))
        finally:
            self.timings[name] = self.timings.get(name, 0.)+time.perf_counter()-tick
            self.calls[name] = self.calls.get(name, 0)+1
            self.rows[name] = self.rows.get(name, 0)+length

    def atomic(self, te, ne):
        return self.dispatch('atomic', len(te), lambda s: self.source.atomic(te[s], ne[s]))


class _ThreadMesh:
    def __init__(self, owner):
        self.owner = owner
        self.native = owner.source.mesh

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _marker_points(self, e, st, fields, feedback):
        return self.owner.dispatch('sample', len(e),
            lambda s: self.native._marker_points(e[s], st[s], fields, feedback))

    def locate_batch(self, positions, *, element_hints=None, local_hints=None):
        return self.owner.dispatch('locate', len(positions), lambda s:
            self.native.locate_batch(positions[s],
                element_hints=None if element_hints is None else element_hints[s],
                local_hints=None if local_hints is None else local_hints[s]))


def table_refresh_benchmark(mesh, fields, workers, repeats):
    """Isolated cache refresh only; no source trajectories or locator changes."""
    from src.models.native_lookup_experiment import ExperimentalLookupMesh
    tick = time.perf_counter()
    lookup = ExperimentalLookupMesh(mesh, grid_points=33)
    build_seconds = time.perf_counter()-tick
    shape = (lookup.n_elements, 33, 33, len(fields))
    reference = lookup._tabulate(fields)
    out = np.empty(shape)
    target = out.reshape(lookup.n_elements, 33**2, len(fields))
    width = max(1, lookup.build_chunk//(33**2))
    chunks = [(a, min(a+width, lookup.n_elements))
              for a in range(0, lookup.n_elements, width)]
    def update(bounds):
        a, b = bounds
        coeff = fields[:, :, 0, :][:, :, lookup.node_ids[a:b]].transpose(2, 3, 1, 0)
        coeff = np.ascontiguousarray(coeff*lookup.local_sizes[a:b, :, :, None])
        np.matmul(lookup.basis_matrix, coeff.reshape(b-a, 16, len(fields)), out=target[a:b])
    entries = []
    for count in workers:
        with ThreadPoolExecutor(count) as pool:
            def refresh():
                if count == 1:
                    for bounds in chunks:
                        update(bounds)
                else:
                    list(pool.map(update, chunks))
            refresh()  # untimed warmup, allocation and same buffer outside timer
            timings = []
            for _ in range(repeats):
                tick = time.perf_counter()
                refresh()
                timings.append(time.perf_counter()-tick)
            entries.append(dict(workers=count, seconds=timings,
                median_seconds=statistics.median(timings), array_equal=bool(np.array_equal(reference, out))))
    return dict(scope='isolated_cached_table_refresh_not_marker_or_chain_speedup',
        grid_points=33, geometry_build_seconds=build_seconds, table_bytes=out.nbytes,
        chunks=len(chunks), entries=entries)


def execute(args, result):
    import torch
    from src.models.native_coupling import source_time_sampler
    from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest
    from tools.native_real_chain_builder import build_real

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split'] == 'train']
    if len(specs) != 2:
        raise ValueError('two declared phaseA train anchors required')
    spec = specs[args.run_index]
    run = Path(spec['run']).resolve()
    anchor = (4110, 5680)[args.run_index]
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected:
        raise ValueError('run/index/anchor mismatch')
    stages = Stages()
    controls = _initial_pairing(run, anchor, stages)
    tick = time.perf_counter()
    model, source, shard, runtime, provider = build_real(run, anchor, controls, stages, args,
                                                        feedback_mode='autonomous')
    result['native_thread_counts'] = native_thread_counts()
    result['torch_threads'] = torch.get_num_threads()
    result['build_seconds'] = time.perf_counter()-tick
    before = runtime.marker
    if len(before.cloud.weight) != 1000000:
        raise ValueError('complete real 1M-slot cloud required')
    target = runtime.fluid_time_si
    def sampler():
        return source_time_sampler(source, before, runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas, completed_dt_si=runtime.fluid_dt_previous_si)
    tick = time.perf_counter()
    births = shard.births(runtime.shards, before, target, runtime.fields_previous,
                         runtime.fields_current, source, time_sampler=sampler())
    result['shared_birth_seconds'] = time.perf_counter()-tick
    post_birth_rng = _rng()
    initial = complete_components(before)
    immutable = _array_digest((runtime.fields_previous, runtime.fields_current,
                              runtime.fluid_deltas, runtime.shards, births))
    result.update(run=str(run), anchor=anchor, initial_live=int(before.cloud.alive.sum()),
        births_count=len(births.weight), slots=len(before.cloud.weight),
        source_configuration=source.configuration, initial_components=initial,
        fields_history_shards_births_digest=immutable, rng_digest=_array_digest(post_birth_rng),
        field_sampling_static=source.field_static, initial_provenance=before.provenance)
    del model, provider

    baseline_components = None
    baseline_rng = None
    baseline_diagnostics = None
    result['iterations'] = []
    def advance(count, repeat, *, warmup=False):
        nonlocal baseline_components, baseline_rng, baseline_diagnostics
        candidate, born, sample = deepcopy(before), deepcopy(births), sampler()
        _restore(post_birth_rng)
        with ThreadedKernels(source, count, kernels=args.kernels) as kernels:
            tick = time.perf_counter()
            cpu = time.process_time()
            out, diagnostic = kernels.adapter.advance(candidate, target,
                runtime.fields_previous, runtime.fields_current, births=born, time_sampler=sample)
            seconds = time.perf_counter()-tick
            cpu_seconds = time.process_time()-cpu
            # Hashing, cloning, thread creation and input checks are NOT in source timing.
            components = complete_components(out)
            rng = _array_digest(_rng())
            diag = _array_digest(stable_diagnostics(diagnostic))
            if baseline_components is None:
                baseline_components, baseline_rng, baseline_diagnostics = components, rng, diag
            entry = dict(workers=count, repeat=repeat, warmup=warmup, seconds=seconds,
                process_cpu_seconds=cpu_seconds, average_process_cores=cpu_seconds/seconds,
                kernel_wall_seconds=kernels.timings, kernel_calls=kernels.calls,
                kernel_rows=kernels.rows, prestarted_worker_count=len(kernels.worker_ids),
                output_components=components, cloud_feedback_clocks_metadata_exact=components==baseline_components,
                rng_exact=rng==baseline_rng, diagnostics_exact=diag==baseline_diagnostics,
                live_count=int(out.cloud.alive.sum()), particle_steps=diagnostic['n_steps'],
                physical_diagnostic_digest=diag,
                peak_rss_bytes=_rss())
            entry['exact'] = all(entry[k] for k in
                ('cloud_feedback_clocks_metadata_exact', 'rng_exact', 'diagnostics_exact'))
            result['iterations'].append(entry)
            print(json.dumps({k: entry[k] for k in ('workers', 'repeat', 'warmup', 'seconds', 'exact',
                                                   'kernel_wall_seconds', 'average_process_cores')}), flush=True)
            if _rss() > args.memory_mib*1024**2:
                raise MemoryError('RSS budget exceeded')
    advance(1, -1, warmup=True)
    for repeat in range(args.repeats):
        # Reverse alternate rounds to reduce fixed-order bias.
        for count in (args.workers if repeat % 2 == 0 else args.workers[::-1]):
            advance(count, repeat)
    result['summary'] = []
    for count in args.workers:
        entries = [e for e in result['iterations'] if not e['warmup'] and e['workers']==count]
        result['summary'].append(dict(workers=count, seconds=[e['seconds'] for e in entries],
            median_seconds=statistics.median(e['seconds'] for e in entries),
            exact=all(e['exact'] for e in entries), kernel_median_seconds={k:
                statistics.median(e['kernel_wall_seconds'][k] for e in entries)
                for k in ('sample', 'locate', 'atomic')}))
    result['inputs_unchanged'] = initial==complete_components(before) and immutable==_array_digest((
        runtime.fields_previous, runtime.fields_current, runtime.fluid_deltas, runtime.shards, births))
    if args.refresh_repeats:
        result['table_refresh'] = table_refresh_benchmark(source.mesh, runtime.fields_current,
                                                           args.workers, args.refresh_repeats)
    result['success'] = (result['inputs_unchanged'] and all(e['exact'] for e in result['iterations'])
        and all(e['array_equal'] for e in result.get('table_refresh', {}).get('entries', [])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    p.add_argument('--run-index', type=int, choices=(0, 1), default=0)
    p.add_argument('--workers', nargs='+', type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    p.add_argument('--kernels', nargs='+', choices=('sample', 'locate', 'atomic'),
                   default=['sample', 'locate', 'atomic'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--refresh-repeats', type=int, default=5)
    p.add_argument('--chunk-size', type=int, default=8192)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--wall-seconds', type=int, default=600)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (1 not in args.workers or len(set(args.workers)) != len(args.workers) or
        not 1 <= args.repeats <= 5 or not 0 <= args.refresh_repeats <= 10 or
        not 0 < args.wall_seconds <= 600 or not 1 <= args.chunk_size <= 1000000 or
        not 0 < args.memory_mib <= 8192 or args.birth_max_proposals <= 0):
        p.error('invalid bounded benchmark configuration')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            p.error(key+'=1 required before process startup')
    args.spatial_backend = 'batched_v1'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema='native-cpu-thread-benchmark-v2', success=False,
        exact_comparison_exclusions=['execution.elapsed_seconds', 'execution.peak_rss_bytes'],
        deadline_policy='SIGALRM interrupts main; running workers drained; not a process hard kill',
        scope='one_fixed_initial_real_source_window_not_full_chain_or_training',
        optimizer_updates=0, training=False, experimental=True, production_qualified=False,
        configuration=vars(args).copy(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
        thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                          'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        numpy_version=np.__version__, native_source_sha256=hashlib.sha256(
            Path('src/models/native_coupling.py').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def timeout(*_):
        raise TimeoutError('external wall deadline exceeded')
    with args.output.open('x') as stream:
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.wall_seconds)
        started = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            result['total_wall_seconds'] = time.perf_counter()-started
            result['peak_rss_bytes'] = _rss()
            json.dump(result, stream, indent=2, default=_json, allow_nan=False)
            stream.write('\n')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
