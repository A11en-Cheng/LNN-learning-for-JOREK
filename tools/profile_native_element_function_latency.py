"""Read-only residual latency census after exact element-function caching.

A native common prefix precedes two identical cached ordinary-event replays:
boundary timers only, then cProfile. Never train or replace predicted fields.
Profiler timings have overhead; inclusive callees must not be added together.
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
import time

import numpy as np
import torch

from src.models.native_element_functions import CompiledElementFunctions, NativeElementFunctionSource, build_kernel
from tools.benchmark_native_multi_window import ROOT, recorded_controls
from tools.benchmark_native_lookup import _rng, _restore
from tools.diagnose_native_real_chain import Stages, _array_digest
from tools.profile_native_coupled_backward import Timing, component_timing, exclusive_stage_summary
from tools.profile_native_coupled_latency import component_digests, timed_call, write_json


def execute(args, result):
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.native_real_chain_builder import build_real

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but unavailable; no silent fallback')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run = ROOT/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400' +
        ('_injInner' if args.run_index else ''))
    anchor = 5680 if args.run_index else 4110
    stages = Stages()
    rows = recorded_controls(run, anchor, 2, stages)
    kernel = CompiledElementFunctions(build_kernel(args.output_dir/'functions.so'))
    model, source, shard, runtime, provider = build_real(run, anchor, rows, stages, args,
        feedback_mode='autonomous')
    model.to(device).eval()
    controls = tuple(StepControl(r['start_si'], r['dt_si'], r['end_si'], r['origin'],
        r['source'], phase=r['phase']) for r in rows)
    result.update(controls=rows, kernel=kernel.configuration,
        initial_live=int(runtime.marker.cloud.alive.sum()), slots=len(runtime.marker.cloud.weight))
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=provider)
    synchronize = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    before_model = _array_digest(model.state_dict())
    with torch.no_grad():
        state = adapter.initialize(runtime)
        print(json.dumps(dict(stage='native_common_prefix', status='started')), flush=True)
        synchronize(); tick = time.perf_counter()
        prefix, _ = adapter.advance(state, controls[0])
        synchronize()
        result['native_common_prefix_seconds'] = time.perf_counter()-tick
        print(json.dumps(dict(stage='native_common_prefix', seconds=result['native_common_prefix_seconds'])), flush=True)
    rng = _rng(); before_prefix = _array_digest(prefix)
    candidate = NativeElementFunctionSource(source, kernel=kernel)
    result['source_configuration'] = candidate.configuration
    fingerprints = []
    for profiled in (False, True):
        mode = 'profiled' if profiled else 'boundary_timed'
        model.source_configuration = deepcopy(candidate.configuration)
        fork = deepcopy(prefix.runtime)
        fork.native_context.configuration['model'] = model.checkpoint_configuration
        adapter = TorchRecomputedSourceEvents(model, candidate, shard, equilibrium_provider=provider)
        state = adapter.initialize(fork)
        _restore(rng)
        timing = Timing(device, args.memory_mib)
        def operation():
            with timing.stage('forward_event_1'):
                return adapter.advance(state, controls[1])
        with torch.no_grad(), component_timing(adapter, timing, fine=True):
            (out, diagnostic), elapsed = timed_call(mode, operation, args.output_dir,
                profile=profiled, device=device)
        row = dict(mode=mode, seconds=elapsed, sampling=candidate.last_stats,
            execution=diagnostic.get('execution'), boundary_exits=diagnostic.get('boundary_exits', []),
            live=int(out.runtime.marker.cloud.alive.sum()),
            stages=timing.rows, exclusive_timing=exclusive_stage_summary(timing.rows),
            components=component_digests(out))
        fingerprints.append(row['components'])
        result.setdefault('events', []).append(row)
        write_json(args.output_dir/(mode+'.json'), row)
        print(json.dumps(dict(stage='ordinary_event', mode=mode, seconds=elapsed,
            categories=row['exclusive_timing']['categories_seconds'])), flush=True)
        del out, fork, state
    result.update(profile_transparency_exact=fingerprints[0] == fingerprints[1],
        prefix_unchanged=before_prefix == _array_digest(prefix),
        weights_unchanged=before_model == _array_digest(model.state_dict()))
    result['success'] = all(result[k] for k in
        ('profile_transparency_exact', 'prefix_unchanged', 'weights_unchanged'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index', type=int, choices=(0, 1), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--wall-seconds', type=int, default=900)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--birth-cells-per-width', type=int, default=64)
    args = p.parse_args()
    if (not 0 < args.wall_seconds <= 900 or not 0 < args.memory_mib <= 8192 or
        not 0 < args.threads <= 16 or min(args.chunk_size, args.birth_max_proposals, args.birth_cells_per_width) <= 0):
        p.error('positive bounded diagnostic resources required')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            p.error(key+'=1 required at process startup')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    args.spatial_backend = 'batched_v1'; args.marker_locator = 'jorek_nearby_v1'
    args.marker_transport = 'local_subcycling_boundary_v2'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT/'src').rglob('*.py')) + [ROOT/'src/models/native_element_functions.cpp', Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = dict(schema='native-element-function-residual-profile-v1', success=False,
        training=False, optimizer_updates=0, backward_calls=0, future_truth_feedback=False,
        weights='fixed_prescribed_nonzero_untrained', production_qualified=False,
        scope='one_native_prefix_then_identical_cached_ordinary_event_replays',
        profile_note='cProfile timings have overhead; inclusive parents overlap children',
        settings={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, hashes=hashes)
    def timeout(*_):
        raise TimeoutError('bounded residual profiling deadline exceeded')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds)
    tick = time.perf_counter()
    try:
        execute(args, result)
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__, error=str(exc)); traceback.print_exc()
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        result.update(elapsed_seconds=time.perf_counter()-tick,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            hashes_unchanged=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
                for path,digest in hashes.items()))
        if torch.cuda.is_initialized():
            result.update(cuda_device=torch.cuda.get_device_name(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(args.output_dir/'result.json', result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
