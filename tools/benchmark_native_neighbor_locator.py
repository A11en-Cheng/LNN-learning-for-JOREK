"""Serial, bounded native/neighbor A/B through complete source windows.

Build and births are outside the source timer. No neural predictions, training,
shard update, geometry microbenchmarks or default-backend integration.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
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

from tools.benchmark_native_cpu_threads import stable_diagnostics
from tools.diagnose_native_real_chain import _array_digest
from tools.native_replay_metrics import relative_metrics
from tools.benchmark_native_lookup import _json, _rss


def _rng():
    import torch
    return dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _tree_equal(a, b):
    """Compare values, not merely hashes; including metadata/RNG containers."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_tree_equal(a[k], b[k]) for k in a))
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_tree_equal(x, y) for x, y in zip(a, b)))
    if hasattr(a, 'detach') or hasattr(b, 'detach'):
        import torch
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and torch.equal(a, b))
    from dataclasses import is_dataclass
    if is_dataclass(a) or is_dataclass(b):
        return type(a) is type(b) and _tree_equal(vars(a),vars(b))
    return type(a) is type(b) and a == b


def _metadata(marker):
    history=getattr(marker,'transport_history',None)
    return dict(group=marker.cloud.group, clock=vars(marker.clock),
                feedback={k:v for k,v in vars(marker.feedback).items()
                          if not isinstance(v, np.ndarray)},
                runtime={k:v for k,v in vars(marker).items()
                         if k not in ('cloud', 'feedback', 'clock','transport_history')},
                transport_history=(None if history is None else
                    dict(configuration=history.configuration,initialization=history.initialization)))


def _arrays(marker):
    arrays = {'cloud.'+k:getattr(marker.cloud, k) for k in marker.cloud.array_names()}
    arrays.update({'cloud.alive':marker.cloud.alive,
                   'feedback.values':marker.feedback.values,
                   'feedback.deltas':marker.feedback.deltas})
    history=getattr(marker,'transport_history',None)
    if history is not None:
        arrays.update({'transport_history.'+k:getattr(history,k)
                       for k in ('previous_dt_si','level','lifetime')})
    return arrays


def _metrics(a, b):
    result = dict(shape=list(a.shape), dtype=str(a.dtype), candidate_dtype=str(b.dtype),
                  shape_exact=a.shape == b.shape, dtype_exact=a.dtype == b.dtype)
    result['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result['array_equal'] = bool(a.shape == b.shape and np.array_equal(a, b))
    if result['shape_exact'] and result['finite']:
        result.update(relative_metrics(a, b))
    return result


def _diagnostics(reference, actual):
    """Physical floats may differ; discrete controls and inventory must agree."""
    errors = {}; controls = {}; inventory = []
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            inventory.append(a.keys() == b.keys())
            for key in a.keys() & b.keys():
                walk(a[key], b[key], path+'.'+str(key))
        elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            inventory.append(len(a) == len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path+'.'+str(i))
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            errors[path] = _metrics(a, b)
            if a.dtype.kind in 'biu':
                controls[path] = errors[path]['array_equal']
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            errors[path] = _metrics(np.asarray(a), np.asarray(b))
            if path.endswith('gather_time_seconds'):
                controls[path] = _tree_equal(a, b)
        else:
            controls[path] = _tree_equal(a, b)
    walk(stable_diagnostics(reference), stable_diagnostics(actual), 'diagnostics')
    return dict(numerical=errors, controls_exact=controls,
                inventory_exact=all(inventory),
                all_exact=_tree_equal(stable_diagnostics(reference), stable_diagnostics(actual)),
                valid=all(inventory) and all(controls.values()) and all(
                    m['finite'] and m['shape_exact'] and m['dtype_exact'] for m in errors.values()))


def _compare(reference, actual, initial, ref_diag, diag, ref_rng, rng, *, actual_initial=None):
    actual_initial=initial if actual_initial is None else actual_initial
    a, b = _arrays(reference), _arrays(actual)
    arrays = {key:_metrics(value, b[key]) for key,value in a.items()}
    mandatory = ('cloud.i_elm', 'cloud.i_life', 'cloud.t_birth', 'cloud.weight', 'cloud.alive')
    if getattr(reference,'transport_history',None) is not None:
        mandatory+=('transport_history.level','transport_history.lifetime')
    exact = {key:arrays[key]['array_equal'] and arrays[key]['dtype_exact'] for key in mandatory}
    exact.update(metadata=_tree_equal(_metadata(reference), _metadata(actual)),
                 rng=_tree_equal(ref_rng, rng))
    physical = _diagnostics(ref_diag, diag)
    valid = all(m['shape_exact'] and m['dtype_exact'] and m['finite'] for m in arrays.values())
    # Relative-to-initial x alone can hide displacement errors. Compare the
    # actual displacement independently, on identical live identity support.
    live = reference.cloud.alive & actual.cloud.alive
    persistent = (live & initial.cloud.alive & actual_initial.cloud.alive
                  & (reference.cloud.i_life == initial.cloud.i_life)
                  & (actual.cloud.i_life == actual_initial.cloud.i_life)
                  & (initial.cloud.i_life == actual_initial.cloud.i_life))
    if (reference.cloud.group != initial.cloud.group or actual.cloud.group != actual_initial.cloud.group
            or initial.cloud.group != actual_initial.cloud.group):
        persistent[:] = False
    increments = {key:_metrics(getattr(reference.cloud, key)[persistent]-getattr(initial.cloud, key)[persistent],
                               getattr(actual.cloud, key)[persistent]-getattr(actual_initial.cloud, key)[persistent])
                  for key in ('x', 'V_prev')}
    # Retain the previous diagnostic under an explicit name: subtracting the
    # SAME origin does not compare per-window changes once two arms diverge.
    accumulated = {key:_metrics(getattr(reference.cloud,key)[persistent]-getattr(initial.cloud,key)[persistent],
                               getattr(actual.cloud,key)[persistent]-getattr(initial.cloud,key)[persistent])
                   for key in ('x','V_prev')}
    live_arrays = {key:_metrics(value[live], b[key][live]) for key,value in a.items()
                   if key.startswith('cloud.')}
    feedback_channels = {
        key:{str(channel):_metrics(getattr(reference.feedback, key)[channel],
                                  getattr(actual.feedback, key)[channel])
             for channel in range(5)} for key in ('values', 'deltas')}
    return dict(arrays=arrays, matched_live_arrays=live_arrays,
                feedback_channels=feedback_channels,
                matched_live_count=int(live.sum()), increments=increments,
                increment_origins='each_arm_own_window_start',
                accumulated_end_difference_over_reference_increment=accumulated,
                increment_support='persistent_initial_live_same_group_slot_lifetime_only',
                persistent_increment_count=int(persistent.sum()),
                required_exact=exact, diagnostics=physical,
                all_arrays_exact=all(m['array_equal'] and m['dtype_exact'] for m in arrays.values()),
                success=bool(valid and all(exact.values()) and physical['valid']))


def benchmark_source_window(source, candidate, initial, target_si, previous, current,
                            *, births=None, sampler_factory=None, repeats=2, count_pass=False,
                            progress=None, memory_bytes=None, record=None):
    """Public fixture/real seam; one warmup per mode, then alternating pairs.

    Success means finite/layout-valid arrays and exact discrete event contracts;
    it is NOT a floating-error tolerance gate or production qualification.
    All numerical comparison work is outside advance-only wall/CPU timers.
    """
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('one to three paired repeats required')
    immutable = _array_digest((initial, previous, current, births))
    call_input_digest = _array_digest((initial, births))
    rng = _rng()
    reference = ref_diag = ref_rng = None
    result = {} if record is None else record
    result.update(success=False, iterations=[], locator_counts={},
                  timing_scope='Source.advance_only_excludes_build_births_clones_sampler_comparison_counts',
                  qualification='finite_arrays_exact_identity_element_lost_clock_gather_RNG_not_float_tolerance',
                  production_qualified=False, rng_initial_digest=_array_digest(rng))
    order = [('baseline', -1, True), ('neighbor', -1, True)]
    for repeat in range(repeats):
        order.extend((mode, repeat, False) for mode in
                     (('baseline', 'neighbor') if repeat % 2 == 0 else ('neighbor', 'baseline')))
    for mode, repeat, warmup in order:
        state, born = deepcopy(initial), deepcopy(births)
        sampler = sampler_factory() if sampler_factory is not None else None
        options = {} if sampler is None else {'time_sampler':sampler}
        _restore_rng(rng)
        if progress is not None:
            progress(dict(phase='timing', mode=mode, repeat=repeat, warmup=warmup, status='started'))
        cpu = time.process_time(); tick = time.perf_counter()
        out, diag = (source if mode == 'baseline' else candidate).advance(
            state, target_si, previous, current, births=born, **options)
        seconds = time.perf_counter()-tick; cpu_seconds = time.process_time()-cpu
        final_rng = _rng()
        if reference is None:
            reference, ref_diag, ref_rng = out, deepcopy(diag), final_rng
        comparison = _compare(reference, out, initial, ref_diag, diag, ref_rng, final_rng)
        call_inputs_unchanged = call_input_digest == _array_digest((state, born))
        if mode == 'baseline':
            comparison['success'] &= comparison['all_arrays_exact'] and comparison['diagnostics']['all_exact']
        entry = dict(mode=mode, repeat=repeat, warmup=warmup, seconds=seconds,
                     process_cpu_seconds=cpu_seconds, comparison=comparison,
                     call_inputs_unchanged=call_inputs_unchanged,
                     rng_digest=_array_digest(final_rng), particle_steps=diag['n_steps'])
        result['iterations'].append(entry)
        if progress is not None:
            progress(dict(phase='timing', mode=mode, repeat=repeat, warmup=warmup, status='completed',
                          seconds=seconds, contract_valid=comparison['success'],
                          all_arrays_exact=comparison['all_arrays_exact']))
        if memory_bytes is not None and _rss() > memory_bytes:
            raise MemoryError('process RSS budget exceeded')
    result['summary'] = []
    for mode in ('baseline', 'neighbor'):
        rows = [r for r in result['iterations'] if r['mode'] == mode and not r['warmup']]
        result['summary'].append(dict(mode=mode, seconds=[r['seconds'] for r in rows],
            median_seconds=statistics.median(r['seconds'] for r in rows),
            contract_valid=all(r['comparison']['success'] for r in rows)))
    result['locator_count_pass_timing_eligible'] = False
    result['count_pass_contract_valid'] = True
    if count_pass:
        for mode, runner in (('baseline', source), ('neighbor', candidate)):
            state, born = deepcopy(initial), deepcopy(births)
            sampler = sampler_factory() if sampler_factory is not None else None
            options = {} if sampler is None else {'time_sampler':sampler}
            _restore_rng(rng)
            if progress is not None:
                progress(dict(phase='counts_not_timing', mode=mode, status='started'))
            out, diag = runner.advance(state, target_si, previous, current, births=born, **options)
            final_rng = _rng()
            comparison = _compare(reference, out, initial, ref_diag, diag, ref_rng, final_rng)
            if mode == 'baseline':
                comparison['success'] &= comparison['all_arrays_exact'] and comparison['diagnostics']['all_exact']
            # No method overrides, even for diagnostics. The public native
            # window exposes processed chunks/points, but not scalar fallbacks.
            if mode == 'baseline' or candidate is source:
                stats = dict(batch_calls=diag['execution']['chunks'],
                    point_visits=diag['execution']['processed_particle_steps'],
                    neighbor_accepted=None, global_fallback_points=None,
                    edge_ambiguous=None, topology_bytes=None,
                    scope='public_window_diagnostics_no_locator_instrumentation')
            else:
                stats = deepcopy(candidate.last_stats)
            result['locator_counts'][mode] = stats
            result.setdefault('count_pass_comparisons', {})[mode] = comparison
            result['count_pass_contract_valid'] &= (comparison['success'] and
                call_input_digest == _array_digest((state, born)))
            if progress is not None:
                progress(dict(phase='counts_not_timing', mode=mode, status='completed',
                              contract_valid=comparison['success'], locator_counts=stats))
            if memory_bytes is not None and _rss() > memory_bytes:
                raise MemoryError('process RSS budget exceeded')
    result['inputs_unchanged'] = immutable == _array_digest((initial, previous, current, births))
    result['success'] = result['inputs_unchanged'] and result['count_pass_contract_valid'] and all(
        r['comparison']['success'] and r['call_inputs_unchanged'] for r in result['iterations'])
    return result


def execute(args, result):
    """Prepare a real million-slot window once; never advance the built model."""
    import torch
    from src.models.native_coupling import source_time_sampler
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tools.benchmark_native_cpu_threads import native_thread_counts
    from tools.diagnose_native_real_chain import Stages, _initial_pairing
    from tools.native_real_chain_builder import build_real

    def progress(row):
        print(json.dumps(row, default=_json), flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    specs = [w for w in json.loads(args.config.read_text())['windows'] if w['split'] == 'train']
    if len(specs) != 2:
        raise ValueError('exactly two declared train anchors required')
    spec = specs[args.run_index]; run = Path(spec['run']).resolve()
    anchor = (4110, 5680)[args.run_index]
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected:
        raise ValueError('anchor/run index mismatch')
    stages = Stages()
    progress(dict(phase='build_not_timing', status='started', run=str(run), anchor=anchor))
    controls = _initial_pairing(run, anchor, stages)
    model, source, shard, runtime, provider = build_real(
        run, anchor, controls, stages, args, feedback_mode='autonomous')
    result['native_threads'] = native_thread_counts()
    before = runtime.marker; target = runtime.fluid_time_si
    if len(before.cloud.weight) != 1000000:
        raise ValueError('full million-slot cloud required')
    def sampler():
        return source_time_sampler(source, before, runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas, completed_dt_si=runtime.fluid_dt_previous_si)
    progress(dict(phase='births_once_not_timing', status='started'))
    births = shard.births(runtime.shards, before, target, runtime.fields_previous,
                         runtime.fields_current, source, time_sampler=sampler())
    # Construct before any source call; no locator/geometry methods are patched.
    candidate = NativeNeighborMarkerSource(source, max_hops=args.max_hops)
    geometry = tuple(getattr(source.mesh, k) for k in ('coordinates','size','vertex','indices'))
    immutable = _array_digest((runtime.fluid_deltas, runtime.shards, geometry, source.configuration))
    result.update(run=str(run), anchor=anchor, slots=len(before.cloud.weight),
        initial_live=int(before.cloud.alive.sum()), births_count=len(births.weight), target_si=target,
        initial_provenance=deepcopy(before.provenance), source_configuration=source.configuration,
        source_fingerprint=_array_digest(source.configuration),
        candidate_configuration=candidate.configuration,
        candidate_fingerprint=_array_digest(candidate.configuration),
        raw_history_shards_geometry_configuration_digest=immutable,
        initial_marker_digest=_array_digest(before), fields_digest=_array_digest(
            (runtime.fields_previous, runtime.fields_current)), births_digest=_array_digest(births))
    del model, provider
    progress(dict(phase='setup_complete_not_timing', slots=result['slots'],
                  initial_live=result['initial_live'], births_count=result['births_count']))
    benchmark_source_window(source, candidate, before, target, runtime.fields_previous,
        runtime.fields_current, births=births, sampler_factory=sampler, repeats=args.repeats,
        count_pass=True, progress=progress, memory_bytes=args.memory_mib*1024**2, record=result)
    result['other_inputs_unchanged'] = immutable == _array_digest(
        (runtime.fluid_deltas, runtime.shards, geometry, source.configuration))
    result['success'] &= result['other_inputs_unchanged']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--run-index', type=int, choices=(0,1), default=0)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--max-hops', type=int, default=4)
    parser.add_argument('--memory-mib', type=int, default=8192)
    parser.add_argument('--wall-seconds', type=int, default=600)
    parser.add_argument('--birth-max-proposals', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 1 <= args.repeats <= 3 or not 0 < args.chunk_size <= 1000000 or
            not 1 <= args.max_hops <= 8 or not 0 < args.memory_mib <= 8192 or
            not 0 < args.wall_seconds <= 900 or not 0 < args.birth_max_proposals <= 1000000):
        parser.error('invalid bounded configuration')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            parser.error(key+'=1 required before process startup')
    args.spatial_backend = 'batched_v1'
    repo = Path(__file__).resolve().parents[1]
    code = dict(native_source=repo/'src/models/native_coupling.py',
                neighbor_module=repo/'src/models/native_neighbor_locator_experiment.py',
                tool=Path(__file__).resolve())
    hashes = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in code.items()}
    result = dict(schema='native-neighbor-locator-benchmark-v1', success=False,
        scope='first_full_real_source_window_fixed_initial_U_rawD_not_learned_weights_or_NN_chain',
        optimization_target='unresolved_native_batch_scalar_global_fallback_only_old_cell_fast_path_unchanged',
        configuration=vars(args).copy(), seed=args.seed, training=False,
        optimizer_instantiated=False, optimizer_updates=0, production_qualified=False,
        numpy_version=np.__version__, source_code_sha256=hashes,
        source_code_paths={key:str(path) for key,path in code.items()},
        exact_exclusions=['diagnostics.execution.elapsed_seconds','diagnostics.execution.peak_rss_bytes'],
        numeric_error_policy='report_all_arrays_no_tolerance_or_bitwise_GPU_claim',
        deadline_policy='cooperative_main_thread_SIGALRM_native_calls_may_defer_interrupt',
        memory_scope='cumulative_process_RSS_includes_full_native_reference_and_numerical_comparison_scratch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('cooperative wall deadline exceeded; native calls may defer interrupt')
    with args.output.open('x') as stream:
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.wall_seconds); tick = time.perf_counter()
        try:
            execute(args, result)
        except Exception as exc:
            result['success'] = False
            result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous)
            result['code_unchanged'] = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[key]
                                           for key,path in code.items())
            result['success'] &= result['code_unchanged']
            result.update(total_seconds=time.perf_counter()-tick, peak_rss_bytes=_rss())
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
            stream.write('\n')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
