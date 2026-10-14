"""Paired exact ADAS-provider change on the cached complete coupling chain.

Everything except the explicit atomic provider stays fixed. One common cached
NumPy prefix, identical state/RNG forks, complete ordinary events, no training.
"""
import argparse
from copy import copy, deepcopy
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

from src.models.native_compiled_atomic import CompiledAtomicInterpolation, CompiledNeonAtomicRates, build_kernel as build_atomic
from src.models.native_element_functions import CompiledElementFunctions, NativeElementFunctionSource, build_kernel as build_fields
from src.models.native_nearby_locator import NativeNearbyMarkerSource
from tools.benchmark_native_element_functions import acceptance, compare_states
from tools.benchmark_native_multi_window import ROOT, recorded_controls
from tools.diagnose_native_real_chain import Stages, _array_digest
from tools.profile_native_coupled_backward import Timing, component_timing, exclusive_stage_summary
from tools.profile_native_coupled_latency import component_digests, rng_state, write_json
from tools.train_native_autonomous_pilot import _restore_rng
from tools.native_replay_metrics import relative_metrics


class AtomicProbe:
    """Diagnostic-only timer, no input changes or identity fingerprint changes."""
    def __init__(self, provider):
        self.provider = provider
        self.version = provider.version
        self.reset()

    def reset(self):
        self.calls = self.points = 0
        self.seconds = 0.

    def __call__(self, te, ne):
        tick = time.perf_counter()
        try:
            return self.provider(te, ne)
        finally:
            self.calls += 1; self.points += len(te)
            self.seconds += time.perf_counter()-tick

    def summary(self):
        return dict(calls=self.calls, points=self.points, seconds=self.seconds,
            scope='atomic_provider_inclusive_inside_source_not_extra_time')


def bind_source(source, provider, neighbours):
    """Fresh explicit source fingerprint; never mutate the reference source."""
    nearby = isinstance(source, NativeNearbyMarkerSource)
    native = source.source if nearby else source
    candidate = copy(native)
    candidate.atomic = provider
    candidate.configuration = deepcopy(native.configuration)
    candidate.configuration['atomic'] = provider.version
    if nearby:
        if neighbours is None:
            raise ValueError('original saved neighbours required for atomic benchmark')
        candidate = NativeNearbyMarkerSource(candidate, neighbours=neighbours)
        expected = deepcopy(source.configuration); expected['atomic'] = provider.version
        if _array_digest(expected) != _array_digest(candidate.configuration):
            raise ValueError('atomic benchmark changed non-atomic source configuration')
    return candidate


def microbenchmark(reference, candidate, repeats=30):
    rng = np.random.default_rng(181)
    rows = []
    for count in (1, 2048, 52604):
        te = 10.**rng.uniform(-1, 5, count)
        ne = 10.**rng.uniform(15, 22, count)
        expected, actual = reference(te, ne), candidate(te, ne)
        metrics = {k:relative_metrics(a, b) for k,a,b in zip(('scd','acd','radiation'), expected, actual)}
        exact = all(np.array_equal(a, b) for a,b in zip(expected, actual))
        times = {'numpy': [], 'compiled': []}
        for repeat in range(repeats):
            order = (('numpy',reference),('compiled',candidate)) if repeat%2 == 0 else (
                ('compiled',candidate),('numpy',reference))
            for label, call in order:
                tick = time.perf_counter(); call(te, ne)
                times[label].append(time.perf_counter()-tick)
        medians = {k:statistics.median(v) for k,v in times.items()}
        row = dict(points=count, repeats=repeats, metrics=metrics, exact=exact,
            times=times, median_seconds=medians, speedup=medians['numpy']/medians['compiled'])
        rows.append(row)
        print(json.dumps(dict(stage='atomic_micro', points=count, exact=exact, median_seconds=medians,
            speedup=row['speedup'])), flush=True)
    return rows


def execute(args, result):
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.native_real_chain_builder import build_real
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable; no CPU fallback')
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400' +
        ('_injInner' if args.run_index else ''))
    anchor = 5680 if args.run_index else 4110
    stages = Stages(); rows = recorded_controls(run, anchor, args.forecast_events+1, stages)
    field_kernel = CompiledElementFunctions(build_fields(args.output_dir/'fields.so'))
    atomic_kernel = CompiledAtomicInterpolation(build_atomic(args.output_dir/'atomic.so'))
    model, original, shard, runtime, provider = build_real(run, anchor, rows, stages, args,
        feedback_mode='autonomous')
    model.to(device).eval()
    native = original.source if isinstance(original, NativeNearbyMarkerSource) else original
    reference = native.atomic
    compiled = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    result.update(kernels=dict(fields=field_kernel.configuration, atomic=atomic_kernel.configuration),
        compiled_configuration=compiled.configuration,
        source_configuration=original.configuration, initial_live=int(runtime.marker.cloud.alive.sum()),
        slots=len(runtime.marker.cloud.weight), controls=rows)
    result['micro'] = microbenchmark(reference, compiled)
    write_json(args.output_dir/'micro.json', result['micro'])
    if not all(row['exact'] for row in result['micro']):
        raise ValueError('complete ADAS micro oracle not bitwise equal; no real advance')
    neighbours = runtime.native_context.configuration['mesh'].get('neighbours')
    probes = {mode:AtomicProbe(rates) for mode,rates in (('numpy',reference),('compiled',compiled))}
    sources = {mode:NativeElementFunctionSource(bind_source(original, probe, neighbours), kernel=field_kernel)
        for mode,probe in probes.items()}
    result['branch_configurations'] = {k:v.configuration for k,v in sources.items()}
    model.source_configuration = deepcopy(sources['numpy'].configuration)
    runtime.native_context.configuration['model'] = model.checkpoint_configuration
    controls = tuple(StepControl(r['start_si'], r['dt_si'], r['end_si'], r['origin'], r['source'],
        phase=r['phase']) for r in rows)
    adapter = TorchRecomputedSourceEvents(model, sources['numpy'], shard, equilibrium_provider=provider)
    weights_digest = _array_digest(model.state_dict())
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    with torch.no_grad():
        state = adapter.initialize(runtime)
        print(json.dumps(dict(stage='common_cached_numpy_prefix', status='started')), flush=True)
        synchronize(); tick = time.perf_counter(); prefix, _ = adapter.advance(state, controls[0]); synchronize()
        result['common_prefix_seconds'] = time.perf_counter()-tick
        print(json.dumps(dict(stage='common_cached_numpy_prefix', seconds=result['common_prefix_seconds'])), flush=True)
    prefix_digest = _array_digest(prefix); rng = rng_state()
    for repeat in range(args.repeats):
        outputs = {}
        for mode in ('numpy','compiled') if repeat%2 == 0 else ('compiled','numpy'):
            selected = sources[mode]
            model.source_configuration = deepcopy(selected.configuration)
            fork = deepcopy(prefix.runtime)
            fork.native_context.configuration['model'] = model.checkpoint_configuration
            adapter = TorchRecomputedSourceEvents(model, selected, shard, equilibrium_provider=provider)
            state = adapter.initialize(fork); _restore_rng(rng)
            events = []
            with torch.no_grad():
                for i,control in enumerate(controls[1:], 2):
                    probes[mode].reset(); timing = Timing(device, args.memory_mib)
                    with component_timing(adapter, timing, fine=True), timing.stage('forward_event_1'):
                        state, diag = adapter.advance(state, control)
                    row = dict(event=i, seconds=timing.rows[-1]['seconds'],
                        source_execution=diag.get('execution'), boundary_exits=diag.get('boundary_exits', []),
                        live=int(state.runtime.marker.cloud.alive.sum()), sampling=selected.last_stats,
                        atomic=probes[mode].summary(), exclusive_timing=exclusive_stage_summary(timing.rows))
                    events.append(row)
                    print(json.dumps(dict(stage='event', repeat=repeat, mode=mode, event=i,
                        seconds=row['seconds'], atomic=row['atomic'])), flush=True)
            outputs[mode] = (state, component_digests(state))
            result.setdefault('iterations', []).append(dict(mode=mode, repeat=repeat, events=events))
            write_json(args.output_dir/f'progress-{repeat}-{mode}.json', result)
        comparison = compare_states(outputs['numpy'][0], outputs['compiled'][0], prefix)
        strict = acceptance(comparison)
        exact = outputs['numpy'][1] == outputs['compiled'][1]
        entry = dict(repeat=repeat, comparison=comparison, acceptance=strict, all_components_and_rng_exact=exact)
        result.setdefault('comparisons', []).append(entry)
        write_json(args.output_dir/f'comparison-{repeat}.json', entry)
        print(json.dumps(dict(stage='comparison', repeat=repeat, acceptance=strict, exact=exact)), flush=True)
        del outputs, state, fork
        if not strict['passed'] or not exact:
            break
    result['summary'] = {}
    for mode in ('numpy','compiled'):
        entries = [e for it in result['iterations'] if it['mode'] == mode for e in it['events']]
        result['summary'][mode] = dict(event_seconds=[e['seconds'] for e in entries],
            mean_event_seconds=statistics.mean(e['seconds'] for e in entries),
            mean_atomic_seconds=statistics.mean(e['atomic']['seconds'] for e in entries),
            mean_source_seconds=statistics.mean(e['exclusive_timing']['categories_seconds']['marker_transport'] for e in entries))
    a, b = result['summary']['numpy'], result['summary']['compiled']
    result['summary'].update(event_latency_reduction_fraction=1-b['mean_event_seconds']/a['mean_event_seconds'],
        paired_event_speedup=a['mean_event_seconds']/b['mean_event_seconds'])
    result.update(weights_unchanged=weights_digest == _array_digest(model.state_dict()),
        prefix_unchanged=prefix_digest == _array_digest(prefix))
    result['success'] = (result['weights_unchanged'] and result['prefix_unchanged'] and
        all(e['acceptance']['passed'] and e['all_components_and_rng_exact'] for e in result['comparisons']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index', type=int, choices=(0,1), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--forecast-events', type=int, choices=(1,2), default=2)
    p.add_argument('--repeats', type=int, choices=(1,2), default=2)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--wall-seconds', type=int, default=900)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--birth-cells-per-width', type=int, default=64)
    args = p.parse_args()
    if not 0 < args.wall_seconds <= 900 or not 0 < args.memory_mib <= 8192 or min(
        args.threads, args.chunk_size, args.birth_max_proposals, args.birth_cells_per_width) <= 0:
        p.error('positive bounded resources required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key) != '1': p.error(key+'=1 required at process startup')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    args.spatial_backend = 'batched_v1'; args.marker_locator = 'jorek_nearby_v1'
    args.marker_transport = 'local_subcycling_boundary_v2'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'src/models').glob('*.cpp')) + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(schema='native-compiled-atomic-benchmark-v1', success=False, training=False,
        optimizer_updates=0, backward_calls=0, future_truth_feedback=False, production_qualified=False,
        weights='fixed_prescribed_nonzero_untrained', scope='paired_provider_only_change_with_field_cache',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, hashes=hashes)
    def timeout(*_): raise TimeoutError('compiled atomic benchmark deadline exceeded')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds); tick = time.perf_counter()
    try:
        execute(args, result)
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__, error=str(exc)); traceback.print_exc()
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        result.update(elapsed_seconds=time.perf_counter()-tick,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            hashes_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items()))
        if torch.cuda.is_initialized():
            result.update(cuda_device=torch.cuda.get_device_name(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(args.output_dir/'result.json', result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
