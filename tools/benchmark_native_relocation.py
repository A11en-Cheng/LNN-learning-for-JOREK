"""Bounded serial prepatch/current A/B on one real Source.advance window."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import random
import signal
import statistics
import threading
import time
import traceback

import numpy as np

from tests.native_relocation_reference import reference_locator
from tools.benchmark_native_neighbor_locator import _compare, _rng, _restore_rng
from tools.benchmark_native_lookup import _json, _rss
from tools.diagnose_native_real_chain import _array_digest


def _profile_counts(profile):
    stats = pstats.Stats(profile).stats
    return {name: dict(primitive_calls=sum(v[0] for k, v in stats.items() if k[2] == name),
                       calls=sum(v[1] for k, v in stats.items() if k[2] == name),
                       total_seconds=sum(v[2] for k, v in stats.items() if k[2] == name),
                       cumulative_seconds=sum(v[3] for k, v in stats.items() if k[2] == name))
            for name in ('locate', 'locate_batch')}


def benchmark_source_window(source, initial, target_si, previous, current, *, births=None,
                            sampler_factory=None, repeats=2, skip_profile=False,
                            progress=None, memory_bytes=None, record=None):
    """Same source and fixed births; scoped class swap outside advance timers."""
    import torch
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('one to three paired repeats required')
    if threading.active_count() != 1 or torch.get_num_threads() != 1:
        raise ValueError('single Python thread and one Torch CPU thread required')
    if not len(initial.cloud.weight) or not np.any(initial.cloud.alive):
        raise ValueError('nonzero live input required')
    if not np.isfinite(target_si) or target_si <= initial.clock.sim_time_si:
        raise ValueError('positive source window required')
    from src.models.native_coupling import NativeSourceMesh
    mesh = source.mesh
    cache_eligible = (type(mesh) is NativeSourceMesh
        and all(name not in mesh.__dict__ for name in
                ('_location_geometries', '_location_geometry', 'locate'))
        and initial.cloud.st.dtype == np.dtype(np.float64)
        and mesh.coordinates.dtype == np.dtype(np.float64)
        and mesh.coordinates.flags.c_contiguous)
    immutable = _array_digest((initial, previous, current, births))
    saved_rng = _rng()
    result = {} if record is None else record
    result.update(success=False, iterations=[], profiles={}, timeline=[],
                  candidate_trial_cache_eligible=bool(cache_eligible),
                  coordinates_c_float64=bool(mesh.coordinates.dtype == np.dtype(np.float64)
                                             and mesh.coordinates.flags.c_contiguous),
                  cache_flag_scope='eligibility_not_observed_cache_hits',
                  rng_initial_digest=_array_digest(saved_rng),
                  timing_scope='Source.advance_only', profile_timing_eligible=False,
                  exact_exclusions=['diagnostics.execution.elapsed_seconds',
                                    'diagnostics.execution.peak_rss_bytes'])
    order = [('baseline', -1, True), ('candidate', -1, True)]
    for repeat in range(repeats):
        order.extend((mode, repeat, False) for mode in
                     (('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')))
    passes = [(m, r, w, False) for m, r, w in order]
    if not skip_profile:
        passes.extend((m, None, False, True) for m in ('baseline', 'candidate'))
    reference = ref_diag = ref_rng = None
    try:
        for mode, repeat, warmup, profiled in passes:
            state = deepcopy(initial)
            sampler = sampler_factory() if sampler_factory is not None else None
            options = {} if sampler is None else {'time_sampler': sampler}
            _restore_rng(saved_rng)
            row = dict(mode=mode, repeat=repeat, warmup=warmup, profiled=profiled)
            if progress:
                progress(dict(row, status='started'))
            profile = cProfile.Profile() if profiled else None
            with reference_locator() if mode == 'baseline' else nullcontext():
                if profile:
                    profile.enable()
                tick = time.perf_counter(); cpu = time.process_time()
                try:
                    out, diag = source.advance(state, target_si, previous, current,
                                               births=births, **options)
                finally:
                    seconds = time.perf_counter() - tick
                    cpu_seconds = time.process_time() - cpu
                    if profile:
                        profile.disable()
            final_rng = _rng()
            if reference is None:
                reference, ref_diag, ref_rng = out, deepcopy(diag), final_rng
            comparison = _compare(reference, out, initial, ref_diag, diag, ref_rng, final_rng)
            comparison['diagnostics_all_exact'] = comparison['diagnostics']['all_exact']
            comparison['success'] &= (comparison['all_arrays_exact'] and
                                      comparison['diagnostics_all_exact'])
            unchanged = (_array_digest((initial, previous, current, births)) == immutable
                         and _array_digest(state) == _array_digest(initial))
            counts = dict(particle_steps=diag['n_steps'],
                          execution=deepcopy(diag.get('execution', {})))
            row.update(comparison=comparison, call_inputs_unchanged=unchanged,
                       rng_digest=_array_digest(final_rng), timeline_counts=counts)
            result['timeline'].append(dict(mode=mode, repeat=repeat, warmup=warmup,
                                           profiled=profiled, **counts))
            if profiled:
                row['locator_counts'] = _profile_counts(profile)
                result['profiles'][mode] = row
            else:
                row.update(seconds=seconds, process_cpu_seconds=cpu_seconds)
                result['iterations'].append(row)
            if progress:
                progress(dict(mode=mode, profiled=profiled, status='completed',
                              exact=comparison['success']))
            if memory_bytes is not None and _rss() > memory_bytes:
                raise MemoryError('process RSS budget exceeded')
    finally:
        _restore_rng(saved_rng)
    result['inputs_unchanged'] = immutable == _array_digest((initial, previous, current, births))
    result['summary'] = [dict(mode=mode, seconds=[r['seconds'] for r in result['iterations']
        if r['mode'] == mode and not r['warmup']], median_seconds=statistics.median(
        r['seconds'] for r in result['iterations'] if r['mode'] == mode and not r['warmup']))
        for mode in ('baseline', 'candidate')]
    rows = result['iterations'] + list(result['profiles'].values())
    result['success'] = result['inputs_unchanged'] and all(
        r['comparison']['success'] and r['call_inputs_unchanged']
        and r['timeline_counts']['particle_steps'] > 0 for r in rows)
    return result


def execute(args, result):
    import torch
    from src.models.native_coupling import source_time_sampler
    from tools.benchmark_native_cpu_threads import native_thread_counts
    from tools.diagnose_native_real_chain import Stages, _initial_pairing
    from tools.native_real_chain_builder import build_real

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split'] == 'train']
    if len(specs) != 2:
        raise ValueError('exactly two declared train anchors required')
    spec = specs[args.run_index]; run = Path(spec['run']).resolve()
    anchor = (4110, 5680)[args.run_index]
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400' + ('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected:
        raise ValueError('anchor/run index mismatch')
    stages = Stages()
    controls = _initial_pairing(run, anchor, stages)
    model, source, shard, runtime, provider = build_real(
        run, anchor, controls, stages, args, feedback_mode='autonomous')
    before = runtime.marker; target = runtime.fluid_time_si
    if len(before.cloud.weight) != 1000000:
        raise ValueError('full million-slot cloud required')
    def sampler():
        return source_time_sampler(source, before, runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas, completed_dt_si=runtime.fluid_dt_previous_si)
    births = shard.births(runtime.shards, before, target, runtime.fields_previous,
                          runtime.fields_current, source, time_sampler=sampler())
    geometry = tuple(getattr(source.mesh, k) for k in ('coordinates', 'size', 'vertex', 'indices'))
    other = (runtime.fluid_deltas, runtime.shards, geometry, source.configuration)
    immutable = _array_digest(other)
    result.update(run=str(run), anchor=anchor, slots=len(before.cloud.weight),
        initial_live=int(before.cloud.alive.sum()), births_count=len(births.weight),
        birth_selection='all_generated_once', birth_cells_per_width=args.birth_cells_per_width,
        birth_max_cells=args.birth_max_cells,
        target_si=target, initial_provenance=deepcopy(before.provenance),
        source_configuration=deepcopy(source.configuration),
        source_fingerprint=_array_digest(source.configuration), native_threads=native_thread_counts(),
        initial_marker_digest=_array_digest(before), births_digest=_array_digest(births),
        fields_digest=_array_digest((runtime.fields_previous, runtime.fields_current)),
        raw_history_shards_geometry_configuration_digest=immutable)
    del model, provider
    benchmark_source_window(source, before, target, runtime.fields_previous, runtime.fields_current,
        births=births, sampler_factory=sampler, repeats=args.repeats, skip_profile=args.skip_profile,
        memory_bytes=args.memory_mib*1024**2, record=result,
        progress=lambda row: print(json.dumps(row, default=_json), flush=True))
    result['other_inputs_unchanged'] = immutable == _array_digest(other)
    result['success'] &= result['other_inputs_unchanged']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--run-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--skip-profile', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--wall-seconds', type=int, default=600)
    parser.add_argument('--memory-mib', type=int, default=8192)
    parser.add_argument('--birth-max-proposals', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not (0 < args.chunk_size <= 1000000 and 0 < args.wall_seconds <= 600
            and 0 < args.memory_mib <= 8192 and 0 < args.birth_max_proposals <= 1000000):
        parser.error('invalid bounded configuration')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            parser.error(key + '=1 required before process startup')
    args.spatial_backend = 'batched_v1'
    args.birth_cells_per_width = 64
    args.birth_max_cells = 16000000
    repo = Path(__file__).resolve().parents[1]
    paths = dict(native_source=repo/'src/models/native_coupling.py',
                 candidate_helper=repo/'src/models/native_location_candidates.py',
                 baseline_helper=repo/'tests/native_relocation_reference.py', tool=Path(__file__).resolve())
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    result = dict(schema='native-relocation-benchmark-v1', success=False, configuration=vars(args).copy(),
        source_code_sha256=hashes, source_code_paths={k: str(v) for k, v in paths.items()},
        baseline_helper_sha256=hashes['baseline_helper'], training=False,
        optimizer_updates=0, production_qualified=False, numpy_version=np.__version__,
        scope='same_source_fixed64_cpw_birth_configuration_actual_all_birthcloud_full_real_cloud_no_NN',
        deadline_policy='cooperative_main_thread_SIGALRM_native_calls_may_defer_interrupt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('cooperative wall deadline exceeded')
    with args.output.open('x') as stream:
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.wall_seconds); tick = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result.update(success=False, error=dict(type=type(exc).__name__, message=str(exc),
                                                   traceback=traceback.format_exc()))
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
            result['code_unchanged'] = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[k]
                                           for k, path in paths.items())
            result['success'] &= result['code_unchanged']
            result.update(total_seconds=time.perf_counter()-tick, peak_rss_bytes=_rss())
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
            stream.write('\n')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
