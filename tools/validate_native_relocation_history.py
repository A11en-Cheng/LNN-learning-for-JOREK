"""Exact two-window relocation/history diagnostic on conditional recorded fields."""
from contextlib import nullcontext
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import threading
import time
import traceback

import numpy as np

from tests.native_relocation_reference import reference_locator
from tools.benchmark_native_neighbor_locator import _compare, _rng, _restore_rng, _tree_equal
from tools.benchmark_native_lookup import _json, _rss
from tools.diagnose_native_real_chain import _array_digest


def recorded_windows(run, anchor, source, runtime, registered, loader):
    """Called only after certified _initial_pairing/build_real poststartup import."""
    fields, raw, recorded_target, dt, recorded = loader(run, anchor+1, source.mesh.units, [])
    logical = runtime.fluid_time_si
    difference = recorded_target - logical
    if not (_tree_equal(fields, runtime.fields_current) and _tree_equal(raw, runtime.fluid_deltas)
            and dt == runtime.fluid_dt_previous_si and np.isfinite(difference)
            and abs(difference) <= 1e-8*dt):
        raise ValueError('builder first window differs from certified recorded fields/rawD/time pairing')
    first_shards = dict(runtime.shards, **{k: recorded[k] for k in
        ('source_volume_m3', 'source_volume_time_si', 'source_volume_provenance')})
    recorded_volume_time = first_shards['source_volume_time_si']
    first_shards['source_volume_time_si'] = logical
    alignment = dict(recorded_target_si=recorded_target, logical_target_si=logical,
        recorded_minus_logical_si=difference, absolute_tolerance_si=1e-8*dt,
        tstep_exact=True, recorded_source_volume_time_si=recorded_volume_time,
        source_volume_recorded_minus_logical_si=recorded_volume_time-logical,
        source_volume_pairing_provenance='registered_poststartup_initial_pairing_same_event_rounding_alignment',
        target_origin='runtime.fluid_time_si', recorded_time_origin=f'jorek{anchor+1:06d}.h5::t_now')
    second_fields, second_raw, target, second_dt, recorded_second = loader(
        run, anchor+2, source.mesh.units, registered)
    second_shards = dict(runtime.shards, **recorded_second)
    for key in ('nonlocal_diagnostic', 'old_equilibrium'):
        second_shards.pop(key, None)
    return [dict(index=anchor+1, previous=runtime.fields_previous, current=fields,
                 raw=raw, target=logical, dt=dt, shards=first_shards),
            dict(index=anchor+2, previous=fields, current=second_fields,
                 raw=second_raw, target=target, dt=second_dt, shards=second_shards)], alignment


def validate_sequence(source, initial, windows, birth_factory, sampler_factory, *, memory_bytes=None):
    """Compact fixture seam: two events, then resume event two from serialized event one.

    Factories receive (source, marker, window); birth_factory additionally gets
    the sampler. Each arm carries its own complete output and RNG into event two.
    Serialization is an in-memory trusted pickle of MarkerRuntime plus RNG.
    """
    import torch
    if len(windows) != 2 or threading.active_count() != 1 or torch.get_num_threads() != 1:
        raise ValueError('exactly two windows and single-thread execution required')
    if not initial.cloud.alive.any():
        raise ValueError('nonzero live input required')
    immutable = _array_digest((initial, windows))
    saved_rng = _rng()
    states = {mode: deepcopy(initial) for mode in ('baseline', 'candidate')}
    rngs = {mode: saved_rng for mode in states}
    snapshots = {}; direct = {}; rows = []; resumes = []; serialization_checks = {}

    def run(mode, marker, rng, window, second):
        before = deepcopy(marker)
        _restore_rng(rng)
        with reference_locator() if mode == 'baseline' else nullcontext():
            if second:
                if not np.isclose(window['target'] - marker.clock.sim_time_si,
                                  window['dt'], rtol=1e-8, atol=0):
                    raise ValueError('recorded t_now gap differs from tstep')
                marker.clock.field_start_si = marker.clock.sim_time_si
                marker.clock.field_end_si = window['target']
            if window['target'] <= marker.clock.sim_time_si or window['dt'] <= 0:
                raise ValueError('positive source window required')
            sampler = sampler_factory(source, marker, window)
            prepared = _array_digest(marker)
            born = birth_factory(source, marker, window, sampler)
            birth_digest = _array_digest(born); birth_rng = _rng()
            out, diag = source.advance(marker, window['target'], window['previous'],
                window['current'], births=born, time_sampler=sampler)
            final_rng = _rng()
        return out, diag, final_rng, dict(births_count=0 if born is None else len(born.weight),
            birth_digest=birth_digest, birth_rng_digest=_array_digest(birth_rng),
            call_inputs_unchanged=prepared == _array_digest(marker)
                and birth_digest == _array_digest(born),
            carried_marker_digest=_array_digest(before),
            carried_feedback_history_digest=_array_digest(before.feedback.deltas),
            carried_clock=vars(before.clock).copy(), particle_steps=diag['n_steps'],
            final_marker_digest=_array_digest(out), final_rng_digest=_array_digest(final_rng))

    def compare(a, b, start):
        comparison = _compare(a[0], b[0], start, a[1], b[1], a[2], b[2])
        comparison['diagnostics_all_exact'] = comparison['diagnostics']['all_exact']
        comparison['births_exact'] = (a[3]['birth_digest'] == b[3]['birth_digest']
            and a[3]['birth_rng_digest'] == b[3]['birth_rng_digest'])
        comparison['success'] &= (comparison['all_arrays_exact']
            and comparison['diagnostics_all_exact'] and comparison['births_exact']
            and a[3]['call_inputs_unchanged'] and b[3]['call_inputs_unchanged']
            and a[3]['particle_steps'] > 0 and b[3]['particle_steps'] > 0)
        return comparison

    try:
        for index, window in enumerate(windows):
            inputs = {mode: states[mode] for mode in states}
            outputs = {mode: run(mode, states[mode], rngs[mode], window, index == 1)
                       for mode in states}
            comparison = compare(outputs['baseline'], outputs['candidate'], inputs['baseline'])
            rows.append(dict(index=window.get('index', index), comparison=comparison,
                             arms={mode: out[3] for mode, out in outputs.items()}))
            for mode, out in outputs.items():
                states[mode], rngs[mode] = out[0], out[2]
                if index == 0:
                    snapshots[mode] = pickle.dumps((out[0], out[2]), protocol=5)
                    recovered, recovered_rng = pickle.loads(snapshots[mode])
                    if (_array_digest(recovered) != _array_digest(out[0])
                            or not _tree_equal(recovered_rng, out[2])):
                        raise ValueError('marker/RNG serialization changed state')
                    serialization_checks[mode] = dict(marker_and_rng_exact=True,
                        serialized_bytes=len(snapshots[mode]),
                        first_feedback_history_nonzero=bool(np.any(out[0].feedback.deltas)))
                    del recovered, recovered_rng
                else:
                    direct[mode] = out
            if memory_bytes is not None and _rss() > memory_bytes:
                raise MemoryError('process RSS budget exceeded')
        for mode in ('baseline', 'candidate'):
            marker, rng = pickle.loads(snapshots.pop(mode))
            resumed = run(mode, marker, rng, windows[1], True)
            resumes.append(dict(mode=mode, comparison=compare(direct[mode], resumed, marker),
                                arm=resumed[3]))
            if memory_bytes is not None and _rss() > memory_bytes:
                raise MemoryError('process RSS budget exceeded')
    finally:
        _restore_rng(saved_rng)
    unchanged = immutable == _array_digest((initial, windows))
    return dict(success=unchanged and all(row['comparison']['success'] for row in rows + resumes),
        windows=rows, serialization_resumes=resumes, inputs_unchanged=unchanged,
        serialization_checks=serialization_checks,
        rng_initial_digest=_array_digest(saved_rng),
        serialization='trusted_in_memory_pickle_protocol5_full_MarkerRuntime_and_RNG',
        timing_scope='validation_only_no_speed_or_profile_comparison',
        exact_exclusions=['diagnostics.execution.elapsed_seconds', 'diagnostics.execution.peak_rss_bytes'])


def execute(args, result):
    import torch
    from src.models.native_source_time import NativeSourceTimeSampler
    from tools.benchmark_native_cpu_threads import native_thread_counts
    from tools.diagnose_native_real_chain import Stages, _initial_pairing
    from tools.native_real_chain_builder import build_real
    from tools.validate_native_birth_windows import load_record

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split'] == 'train']
    if len(specs) != 2:
        raise ValueError('exactly two declared train anchors required')
    spec = specs[args.run_index]; anchor = (4110, 5680)[args.run_index]
    run = Path(spec['run']).resolve()
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400' + ('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected:
        raise ValueError('anchor/run index mismatch')
    stages = Stages()
    model, source, shard, runtime, provider = build_real(run, anchor,
        _initial_pairing(run, anchor, stages), stages, args, feedback_mode='autonomous')
    if len(runtime.marker.cloud.weight) != 1000000:
        raise ValueError('full million-slot cloud required')
    registered = json.loads((repo/'data/restart_events.json').read_text())['runs'][run.name]['restart_events']
    windows, alignment = recorded_windows(run, anchor, source, runtime, registered, load_record)
    def sampler(owner, marker, window):
        return NativeSourceTimeSampler(window['current'], window['raw'],
            time_prev_si=marker.clock.field_start_si, time_now_si=marker.clock.field_end_si,
            tstep_si=window['dt'], static=owner.field_static)
    def births(owner, marker, window, sample):
        return shard.birth.births(window['shards'], marker, window['target'],
            window['previous'], window['current'], owner, time_sampler=sample)
    result.update(run=str(run), anchor=anchor, slots=len(runtime.marker.cloud.weight),
        first_registered_event_pairing=alignment,
        initial_live=int(runtime.marker.cloud.alive.sum()), native_threads=native_thread_counts(),
        source_configuration=deepcopy(source.configuration), field_static=source.field_static,
        birth_configuration=deepcopy(shard.birth.configuration),
        field_context='conditional_diagnostic_recorded_current_fields_and_rawD_not_NN',
        recorded_windows=[dict(index=w['index'], target_si=w['target'], tstep_si=w['dt'],
            fields_digest=_array_digest((w['previous'], w['current'])), rawD_digest=_array_digest(w['raw']),
            shards_digest=_array_digest(w['shards']),
            source_volume_provenance=w['shards']['source_volume_provenance']) for w in windows])
    other = (runtime.shards, source.configuration, shard.birth.configuration,
             tuple(getattr(source.mesh, k) for k in ('coordinates', 'size', 'vertex', 'indices')))
    immutable = _array_digest(other)
    del model, provider
    result.update(validate_sequence(source, runtime.marker, windows, births, sampler,
                                   memory_bytes=args.memory_mib*1024**2))
    result['other_inputs_unchanged'] = immutable == _array_digest(other)
    result['success'] &= result['other_inputs_unchanged']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--run-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            parser.error(key+'=1 required before process startup')
    args.chunk_size = 2048; args.wall_seconds = 600; args.memory_mib = 8192
    args.birth_max_proposals = 1000000; args.birth_cells_per_width = 64
    args.birth_max_cells = 16000000; args.spatial_backend = 'batched_v1'
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), repo/'tests/native_relocation_reference.py',
             repo/'tools/native_real_chain_builder.py', repo/'tools/validate_native_birth_windows.py',
             repo/'tools/benchmark_native_neighbor_locator.py', repo/'src/models/marker_state.py',
             *sorted((repo/'src/models').glob('native*.py'))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(schema='native-relocation-history-v1', success=False, training=False,
        configuration=vars(args).copy(), source_code_sha256=hashes,
        scope='two_conditional_real_source_windows_full_cloud_history_and_serialized_resume')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('cooperative 600s deadline exceeded; native calls may defer interrupt')
    with args.output.open('x') as stream:
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(600); tick = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result.update(success=False, error=dict(type=type(exc).__name__, message=str(exc),
                                                   traceback=traceback.format_exc()))
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
            result['code_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in paths)
            result['success'] &= result['code_unchanged']
            result.update(total_seconds=time.perf_counter()-tick, peak_rss_bytes=_rss())
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
            stream.write('\n')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
