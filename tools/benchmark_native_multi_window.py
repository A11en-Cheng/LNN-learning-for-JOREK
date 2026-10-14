"""Bounded, contiguous complete-coupling forward timings; never training.

Only scalar timing controls are read from future HDF records. Initial fields
are read once by the autonomous builder; every later field is predicted.
Each process retains the full cloud and uses fixed prescribed untrained weights.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import time

import h5py
import numpy as np
import torch

from tools.diagnose_native_real_chain import Stages, _initial_pairing, _scalar, _array_digest
from tools.profile_native_coupled_latency import write_json
from tools.profile_native_coupled_backward import Timing, component_timing, exclusive_stage_summary

ROOT = Path(__file__).resolve().parents[1]


def recorded_controls(run, anchor, count, stages):
    """Preserve existing registered initialization, extend scalar-only controls."""
    if type(count) is not int or not 2 <= count <= 16:
        raise ValueError('between 2 and 16 contiguous complete events required')
    rows = _initial_pairing(run, anchor, stages)
    registered = json.loads((ROOT/'data/restart_events.json').read_text())['runs'][run.name]['restart_events']
    with h5py.File(run/f'jorek{anchor+1:06d}.h5', 'r') as f:
        tn = _scalar(f, 't_norm')
    for offset in range(4, count+2):
        index = anchor+offset
        if index in registered:
            raise ValueError('multi-window timing cannot cross a registered restart')
        path = run/f'jorek{index:06d}.h5'
        with h5py.File(path, 'r') as f:
            if _scalar(f, 'index_now') != index or _scalar(f, 't_norm') != tn:
                raise ValueError('HDF event index or time unit mismatch')
            end = _scalar(f, 't_now')*tn
            dt = _scalar(f, 'tstep')*tn
        start = rows[-1]['end_si']
        if not np.isfinite((start, end, dt)).all() or dt <= 0 or abs(start+dt-end) > max(1e-14, tn*1e-7):
            raise ValueError('recorded HDF dt does not reach contiguous event endpoint')
        rows.append(dict(start_si=start, solve_start_si=start, end_si=end, dt_si=dt,
            phase='ordinary', event_index=index, origin='recorded_schedule_replay', source=str(path)))
    return rows


def slice_summary(rows, root_id):
    """Reuse validated direct-child accounting, select exactly one event tree."""
    indexed = {r['id']: r for r in rows}
    selected = []
    for row in rows:
        ancestor = row
        while ancestor['parent_id'] is not None:
            ancestor = indexed[ancestor['parent_id']]
        if ancestor['id'] == root_id:
            selected.append(dict(row, stage='forward_event_1' if row['id'] == root_id else row['stage']))
    result = exclusive_stage_summary(selected)
    result['scope'] = 'one_complete_forward_event_excludes_setup_and_post_event_evidence'
    return result


def distribution(values):
    if not values:
        return dict(count=0, mean_seconds=None, median_seconds=None, min_seconds=None, max_seconds=None)
    values = np.asarray(values)
    return dict(count=len(values), mean_seconds=float(values.mean()), median_seconds=float(np.median(values)),
        min_seconds=float(values.min()), max_seconds=float(values.max()))


def measure_windows(adapter, runtime, controls, output_dir, *, device='cpu', memory_mib=8192):
    """Actual public events, no labels/optimizer/backward, transparent timers."""
    from src.models.marker_state import particle_window_schedule
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    timing = Timing(device, memory_mib)
    before_model = _array_digest(adapter.model.state_dict())
    before_runtime = _array_digest(runtime)
    result = dict(success=False, events=[], planned_events=len(controls), completed_events=0,
        training=False, optimizer_instantiated=False, optimizer_updates=0, backward_calls=0,
        future_truth_feedback=False, timing_scope='complete_autonomous_coupled_forward_no_grad',
        gradient_scope=adapter.gradient_scope, initial_model_digest=before_model)
    try:
        with torch.no_grad(), component_timing(adapter, timing, fine=True):
            with timing.stage('initialize'):
                state = adapter.initialize(runtime)
            result.update(initial_slots=len(state.runtime.marker.cloud.weight),
                initial_live=int(state.runtime.marker.cloud.alive.sum()))
            for i, control in enumerate(controls, 1):
                source_start = state.runtime.marker.clock.particle_time_si
                source_target = state.runtime.fluid_time_si
                _, source_steps, gather = particle_window_schedule(state.runtime.marker.clock, source_target)
                row_start = len(timing.rows)
                with timing.stage(f'complete_slice_{i}'):
                    state, diag = adapter.advance(state, control)
                root = timing.rows[-1]
                summary = slice_summary(timing.rows[row_start:], root['id'])
                cloud = state.runtime.marker.cloud
                row = dict(slice=i, control=asdict(control), seconds=root['seconds'],
                    fluid_time_si=state.runtime.fluid_time_si, fluid_index=state.runtime.fluid_index,
                    source_window_start_si=source_start, source_window_end_si=source_target,
                    source_base_steps=source_steps, source_gather_seconds=gather,
                    slots=len(cloud.weight), live_markers=int(cloud.alive.sum()),
                    source_execution=diag.get('execution'),
                    source_boundary_exits=diag.get('boundary_exits', []), exclusive_timing=summary)
                result['events'].append(row)
                result['completed_events'] = i
                write_json(output_dir/f'slice-{i}.json', row)
                print(json.dumps(dict(stage='slice_result', slice=i, seconds=row['seconds'],
                    live=row['live_markers'], source_base_steps=source_steps)), flush=True)
            result.update(success=True, final_state_digest=_array_digest(state),
                first_slice_seconds=result['events'][0]['seconds'] if controls else None)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(model_unchanged=before_model == _array_digest(adapter.model.state_dict()),
            initial_runtime_unchanged=before_runtime == _array_digest(runtime), stages=timing.rows,
            subsequent_slice_distribution=distribution([r['seconds'] for r in result['events'][1:]]),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.device(device).type == 'cuda':
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(output_dir/'result.json', result)
    return result


def dependency_hashes():
    paths = sorted((ROOT/'src').rglob('*.py')) + [ROOT/'tools'/name for name in
        ('benchmark_native_multi_window.py', 'native_real_chain_builder.py',
         'profile_native_coupled_backward.py', 'profile_native_coupled_latency.py', 'diagnose_native_real_chain.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index', type=int, choices=(0, 1), required=True)
    p.add_argument('--windows', type=int, default=6, choices=range(2, 17))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--wall-seconds', type=float, default=900.)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--spatial-backend', choices=('batched_v1',), default='batched_v1')
    p.add_argument('--marker-locator', choices=('native', 'jorek_nearby_v1'), default='jorek_nearby_v1')
    p.add_argument('--marker-transport', choices=('native_fixed', 'local_subcycling_v1', 'local_subcycling_boundary_v2'),
        default='local_subcycling_boundary_v2')
    p.add_argument('--birth-cells-per-width', type=int, default=64)
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()
    if not 0 < args.wall_seconds <= 900 or min(args.memory_mib, args.threads, args.chunk_size,
            args.birth_max_proposals, args.birth_cells_per_width) <= 0:
        p.error('positive budgets and wall-seconds <=900 required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(success=False, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        training=False, optimizer_updates=0, future_truth_feedback=False,
        weights='builder_prescribed_nonzero_untrained_fixed_parameters',
        control_origin='recorded_schedule_replay', dependency_hashes=dependency_hashes(),
        environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    start = time.perf_counter()
    def deadline(*_):
        raise TimeoutError('whole multi-window benchmark deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        if args.deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(args.deterministic)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback for real benchmark')
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        result.update(device=torch.cuda.get_device_name(), torch_version=torch.__version__, numpy_version=np.__version__,
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), cpu_count=os.cpu_count())
        name = 'phaseA_dataset_1d22_ang15_massloss_neon_v400' + ('_injInner' if args.run_index else '')
        anchor = 5680 if args.run_index else 4110
        run = ROOT/'data/EHL-2'/name
        stages = Stages()
        setup = Timing('cuda', args.memory_mib)
        with setup.stage('scalar_control_read'):
            rows = recorded_controls(run, anchor, args.windows, stages)
        result['controls'] = rows
        from tools.native_real_chain_builder import build_real
        with setup.stage('build'):
            model, source, shard, runtime, provider = build_real(run, anchor, rows, stages, args, feedback_mode='autonomous')
        result.update(source_configuration=source.configuration, shard_configuration=shard.configuration,
            equilibrium_configuration=provider.configuration, parameter_count=sum(v.numel() for v in model.parameters()))
        with setup.stage('cuda_transfer'):
            model.to('cuda'); model.eval()
        from src.training.base_coupled import TorchRecomputedSourceEvents
        from src.data.base_coupled import StepControl
        adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=provider)
        controls = tuple(StepControl(r['start_si'], r['dt_si'], r['end_si'], r['origin'], r['source'], phase=r['phase']) for r in rows)
        result['setup_stages'] = setup.rows
        result['chain'] = measure_windows(adapter, runtime, controls, args.output_dir/'chain', device='cuda', memory_mib=args.memory_mib)
        result['success'] = result['chain']['success']
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(elapsed_seconds=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            dependencies_unchanged=result['dependency_hashes'] == dependency_hashes())
        write_json(args.output_dir/'result.json', result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
