"""Two recorded complete source windows and exact per-arm serialized resumes.

Current production native is the baseline. No reference-locator patch, training,
neural field prediction, shard evolution, or cross-arm birth reuse occurs.
"""
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

from tools.validate_native_relocation_history import recorded_windows
from tools.benchmark_native_neighbor_locator import _compare, _rng, _restore_rng, _tree_equal
from tools.benchmark_native_cpu_threads import stable_diagnostics
from tools.benchmark_native_lookup import _json, _rss
from tools.diagnose_native_real_chain import _array_digest


def _birth_evidence(born, rng):
    return dict(count=0 if born is None else len(born.weight), digest=_array_digest(born),
                rng_digest=_array_digest(rng), group=None if born is None else born.group,
                arrays={} if born is None else {name: dict(shape=list(getattr(born, name).shape),
                    dtype=str(getattr(born, name).dtype), digest=_array_digest(getattr(born, name)))
                    for name in born.array_names()})


def _birth_compare(a, b):
    names = a['arrays'].keys() | b['arrays'].keys()
    return dict(counts=[a['count'], b['count']], count_exact=a['count'] == b['count'],
                group_exact=a['group'] == b['group'],
                arrays_exact={name: _tree_equal(a['arrays'].get(name), b['arrays'].get(name))
                              for name in sorted(names)},
                full_births_exact=a['digest'] == b['digest'],
                post_birth_rng_exact=a['rng_digest'] == b['rng_digest'])


def validate_sequence(source, candidate, initial, windows, birth_factory, sampler_factory,
                      *, memory_bytes=None, progress=None):
    """Each arm carries its own MarkerRuntime and RNG; factories run six times.

    Cross-arm success means mandatory discrete/RNG contracts and finite complete
    float reporting, with no floating tolerance gate or qualification claim.
    Same-arm resumes require complete state, stable diagnostics, births and RNG
    EXACTLY equal to that arm's continuous second window.
    """
    import torch
    if len(windows) != 2 or threading.active_count() != 1 or torch.get_num_threads() != 1:
        raise ValueError('exactly two windows and single-thread execution required')
    if not initial.cloud.alive.any():
        raise ValueError('nonzero live input required')
    immutable = _array_digest((initial, windows))
    saved_rng = _rng()
    owners = dict(current_native=source, nearby=candidate)
    states = {mode: deepcopy(initial) for mode in owners}
    rngs = {mode: deepcopy(saved_rng) for mode in owners}
    snapshots = {}; direct = {}; rows = []; resumes = []; checks = {}
    completed_advances=0

    def budget():
        if memory_bytes is not None and _rss() > memory_bytes:
            raise MemoryError('process RSS budget exceeded')

    def run(owner, marker, rng, window, second, *, resume=False):
        nonlocal completed_advances
        carried = _array_digest(marker)
        carried_history = _array_digest(marker.feedback.deltas)
        _restore_rng(rng)
        if second:
            if not np.isclose(window['target']-marker.clock.sim_time_si, window['dt'], rtol=1e-8, atol=0):
                raise ValueError('recorded t_now gap differs from tstep')
            marker.clock.field_start_si = marker.clock.sim_time_si
            marker.clock.field_end_si = window['target']
        if window['target'] <= marker.clock.sim_time_si or window['dt'] <= 0:
            raise ValueError('positive source window required')
        prepared = _array_digest(marker)
        sample = sampler_factory(owner, marker, window)
        born = birth_factory(owner, marker, window, sample)
        birth = _birth_evidence(born, _rng())
        budget()
        info=dict(stage='complete_source_advance',mode=('nearby' if owner is candidate else 'current_native'),
            window=window.get('index'),resume=resume)
        if progress is not None:progress(dict(info,status='started',completed_advances=completed_advances))
        tick=time.perf_counter()
        out, diag = owner.advance(marker, window['target'], window['previous'],
                                  window['current'], births=born, time_sampler=sample)
        completed_advances+=1
        if progress is not None:progress(dict(info,status='completed',completed_advances=completed_advances,
            seconds=time.perf_counter()-tick))
        rng_out = _rng()
        evidence = dict(births=birth, call_inputs_unchanged=prepared == _array_digest(marker)
                        and birth['digest'] == _array_digest(born),
                        carried_marker_digest=carried, carried_feedback_history_digest=carried_history,
                        particle_steps=diag['n_steps'], final_marker_digest=_array_digest(out),
                        final_rng_digest=_array_digest(rng_out),
                        locator_stats=deepcopy(getattr(owner, 'last_stats', None)),
                        execution=deepcopy(diag['execution']),boundary_exits=deepcopy(diag.get('boundary_exits',[])))
        budget()
        return out, diag, rng_out, evidence

    def compare(a, b, start, *, exact, actual_start=None):
        result = _compare(a[0], b[0], start, a[1], b[1], a[2], b[2],actual_initial=actual_start)
        result['births'] = _birth_compare(a[3]['births'], b[3]['births'])
        result['complete_marker_exact'] = _array_digest(a[0]) == _array_digest(b[0])
        result['diagnostics_all_exact'] = _tree_equal(stable_diagnostics(a[1]), stable_diagnostics(b[1]))
        result['locator_stats_exact'] = _tree_equal(a[3]['locator_stats'], b[3]['locator_stats'])
        result['same_arm_exact_required'] = exact
        result['success'] &= (a[3]['call_inputs_unchanged'] and b[3]['call_inputs_unchanged']
                              and a[3]['particle_steps'] > 0 and b[3]['particle_steps'] > 0)
        if exact:
            result['success'] &= (result['all_arrays_exact'] and result['complete_marker_exact']
                                  and result['diagnostics_all_exact']
                                  and result['locator_stats_exact']
                                  and result['births']['full_births_exact']
                                  and result['births']['post_birth_rng_exact'])
        # Across arms, birth discrepancies are reported, never fixed by sharing
        # future births. Discrete output/RNG differences remain mandatory failures.
        return result

    try:
        budget()
        for index, window in enumerate(windows):
            outputs = {mode: run(owner, states[mode], rngs[mode], window, index == 1)
                       for mode, owner in owners.items()}
            comparison = compare(outputs['current_native'], outputs['nearby'], states['current_native'],
                                 actual_start=states['nearby'],exact=False)
            rows.append(dict(index=window.get('index', index), comparison=comparison,
                             arms={mode: out[3] for mode, out in outputs.items()}))
            for mode, out in outputs.items():
                states[mode], rngs[mode] = out[0], out[2]
                if index == 0:
                    snapshots[mode] = pickle.dumps((out[0], out[2]), protocol=5)
                    restored, restored_rng = pickle.loads(snapshots[mode])
                    if _array_digest(restored) != _array_digest(out[0]) or not _tree_equal(restored_rng, out[2]):
                        raise ValueError('MarkerRuntime/RNG serialization changed state')
                    checks[mode] = dict(marker_and_rng_exact=True, serialized_bytes=len(snapshots[mode]),
                                        first_feedback_history_nonzero=bool(np.any(out[0].feedback.deltas)))
                    del restored, restored_rng
                else:
                    direct[mode] = out
            budget()
        for mode, owner in owners.items():
            marker, rng = pickle.loads(snapshots.pop(mode))
            resumed = run(owner, marker, rng, windows[1], True,resume=True)
            resumes.append(dict(mode=mode, comparison=compare(direct[mode], resumed, marker, exact=True),
                                arm=resumed[3]))
            budget()
    finally:
        _restore_rng(saved_rng)
    unchanged = immutable == _array_digest((initial, windows))
    return dict(success=bool(unchanged and all(row['comparison']['success'] for row in rows+resumes)),
                windows=rows, serialization_resumes=resumes, inputs_unchanged=unchanged,
                completed_source_advances=completed_advances,
                serialization_checks=checks, rng_initial_digest=_array_digest(saved_rng),
                serialization='trusted_in_memory_pickle_protocol5_full_MarkerRuntime_and_RNG',
                baseline='unpatched_current_production_native',
                births_policy='independent_each_arm_each_window_and_resume',
                floating_policy='full_array_relative_reports_no_tolerance_gate_no_waiver',
                production_qualified=False, timing_scope='validation_only_not_benchmark',
                exact_exclusions=['diagnostics.execution.elapsed_seconds', 'diagnostics.execution.peak_rss_bytes'])


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def execute(args, result):
    import h5py
    import torch
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
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
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected:
        raise ValueError('anchor/run index mismatch')
    initial_field = run/f'jorek{anchor+1:06d}.h5'
    input_paths = [args.config.resolve(), repo/'data/restart_events.json',
                   run/'jorek_namelist_spitest', run/f'part{anchor:05d}.h5',
                   run/f'projections{anchor:06d}.h5',
                   *(run/f'jorek{anchor+i:06d}.h5' for i in range(3))]
    input_hashes = {str(p): _file_hash(p) for p in input_paths}
    result['input_file_sha256'] = input_hashes
    stages = Stages()
    model, source, shard, runtime, provider = build_real(run, anchor,
        _initial_pairing(run, anchor, stages), stages, args, feedback_mode='autonomous')
    if len(runtime.marker.cloud.weight) != 1000000:
        raise ValueError('full million-slot cloud required')
    # The builder is explicitly current native; read SAVED initial-field table
    # separately and supply it to the nearby adapter. Never infer adjacency.
    with h5py.File(initial_field, 'r') as handle:
        if 'neighbours' not in handle:
            raise ValueError('actual initial field saved neighbours required')
        neighbours = np.asarray(handle['neighbours'][...])
    candidate = NativeNearbyMarkerSource(source, neighbours=neighbours)
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
        initial_live=int(runtime.marker.cloud.alive.sum()), first_registered_event_pairing=alignment,
        initial_provenance=deepcopy(runtime.marker.provenance), native_threads=native_thread_counts(),
        source_configuration=deepcopy(source.configuration), candidate_configuration=deepcopy(candidate.configuration),
        birth_configuration=deepcopy(shard.birth.configuration), field_static=source.field_static,
        neighbours=dict(origin=str(initial_field)+'::neighbours', digest=_array_digest(neighbours),
                        shape=list(neighbours.shape), negative=int((neighbours<0).sum()), zero=int((neighbours==0).sum())),
        field_context='conditional_recorded_fields_rawD_no_NN_no_shard_evolution',
        recorded_windows=[dict(index=w['index'], target_si=w['target'], tstep_si=w['dt'],
            fields_digest=_array_digest((w['previous'], w['current'])), rawD_digest=_array_digest(w['raw']),
            shards_digest=_array_digest(w['shards']), source_volume_provenance=w['shards']['source_volume_provenance'])
            for w in windows])
    other = (runtime.shards, runtime.fluid_deltas, source.configuration, candidate.configuration,
             shard.birth.configuration, neighbours,
             tuple(getattr(source.mesh, k) for k in ('coordinates', 'size', 'vertex', 'indices')))
    immutable = _array_digest(other)
    del model, provider
    try:
        result.update(validate_sequence(source, candidate, runtime.marker, windows, births, sampler,
                                        memory_bytes=args.memory_mib*1024**2,
                                        progress=lambda row:print(json.dumps(row),flush=True)))
    finally:
        result['other_inputs_unchanged'] = immutable == _array_digest(other)
        result['input_files_unchanged'] = all(_file_hash(p) == input_hashes[str(p)] for p in input_paths)
        result['success'] = bool(result.get('success', False) and result['other_inputs_unchanged']
                                 and result['input_files_unchanged'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--run-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--marker-transport',choices=('native_fixed','local_subcycling_v1','local_subcycling_boundary_v2'),default='native_fixed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            parser.error(key+'=1 required before process startup')
    args.chunk_size = 2048; args.wall_seconds = 600; args.memory_mib = 8192
    args.birth_max_proposals = 1000000; args.birth_cells_per_width = 64
    args.birth_max_cells = 16000000; args.spatial_backend = 'batched_v1'; args.marker_locator = 'native'
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), repo/'tools/validate_native_relocation_history.py',
             repo/'tools/native_real_chain_builder.py', repo/'tools/validate_native_birth_windows.py',
             repo/'src/models/marker_transport.py',
             repo/'tools/benchmark_native_neighbor_locator.py', repo/'tools/benchmark_native_cpu_threads.py',
             repo/'tools/native_replay_metrics.py', repo/'tools/diagnose_native_real_chain.py',
             repo/'tools/benchmark_native_lookup.py',
             *sorted((repo/'src').rglob('*.py'))]
    hashes = {str(p): _file_hash(p) for p in paths}
    result = dict(schema='native-nearby-history-v1', success=False, training=False,
                  configuration=vars(args).copy(), source_code_sha256=hashes,
                  scope='two_conditional_complete_source_windows_full_1m_cloud_and_same_arm_serialized_resume')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('cooperative 600s deadline exceeded; native calls may defer interrupt')
    with args.output.open('x') as stream:
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(600); tick = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result.update(success=False, error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
            if 'input_file_sha256' in result and 'input_files_unchanged' not in result:
                result['input_files_unchanged'] = all(_file_hash(p) == digest
                    for p, digest in result['input_file_sha256'].items())
            result['code_unchanged'] = all(_file_hash(p) == hashes[str(p)] for p in paths)
            result['success'] &= result['code_unchanged'] and result.get('input_files_unchanged', True)
            result.update(total_seconds=time.perf_counter()-tick, peak_rss_bytes=_rss())
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
            stream.write('\n')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
