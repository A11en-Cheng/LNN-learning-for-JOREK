"""Bounded full-live-cloud spatial comparison; no source advance or training.

Run as python -m tools.benchmark_native_cuda_spatial. Only the poststartup
initial field file and imported particle file are read, never future labels.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np
import torch

from tools.native_replay_metrics import relative_metrics

FIELDS = ('psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e')


def benchmark_spatial(mesh, candidate, fields, positions, element_hints, local_hints,
                      *, chunk_size=2048, repeats=3, synchronize=lambda: None):
    """Public points/locate seam; all input rows visited, transfers timed."""
    n = len(positions)
    if n == 0 or chunk_size < 1 or repeats < 1:
        raise ValueError('nonempty cloud and positive chunk/repeats required')
    if (fields.shape[0] != 9 or positions.shape != (n, 3)
        or local_hints.shape != (n, 2) or element_hints.shape != (n,)
        or (element_hints <= 0).any()):
        raise ValueError('complete nine fields and located live cloud required')
    def batched(call):
        rows = [call(slice(i, min(i+chunk_size, n))) for i in range(0, n, chunk_size)]
        return tuple(np.concatenate(a, axis=0) for a in zip(*rows))
    def timed(call):
        durations = []
        for _ in range(repeats):
            synchronize(); start = time.perf_counter()
            out = call()
            synchronize(); durations.append(time.perf_counter()-start)
        return out, dict(seconds=durations, median_seconds=float(np.median(durations)))
    def metrics(a, b):
        return dict(values_exact=np.array_equal(a[0], b[0]), gradients_exact=np.array_equal(a[1], b[1]),
            per_field={name: dict(values=relative_metrics(a[0][:, f], b[0][:, f]),
                dR=relative_metrics(a[1][:, f, 0], b[1][:, f, 0]),
                dZ=relative_metrics(a[1][:, f, 1], b[1][:, f, 1])) for f, name in enumerate(FIELDS)},
            geometry=[relative_metrics(a[k], b[k]) for k in (2, 3, 4)])
    cpu, cpu_time = timed(lambda: batched(lambda s: mesh.points(element_hints[s]-1, local_hints[s], fields)))
    resident = hasattr(candidate, 'points_frozen')
    scope = candidate.event_scope({'benchmark': 'static_initial_fields'}) if resident else nullcontext()
    with scope:
        got, gpu_time = timed(lambda: batched(lambda s: candidate.points(element_hints[s]-1, local_hints[s], fields)))
        result = dict(point_count=n, chunk_size=chunk_size, repeats=repeats,
            points=dict(**metrics(cpu, got), cpu=cpu_time, candidate_inclusive=gpu_time),
            transfer_scope='points: fresh field H2D per chunk, particle H2D and all five output D2H; concatenation included',
            hot_resident=dict(status='not_available_no_residency_API'), locate=[])
        if resident:
            synchronize(); start = time.perf_counter()
            candidate.freeze_fields('initial', fields)
            synchronize(); upload = time.perf_counter()-start
            hot, timing = timed(lambda: batched(lambda s: candidate.points_frozen('initial', element_hints[s]-1, local_hints[s])))
            result['hot_resident'] = dict(status='measured', upload_seconds=upload, timing=timing,
                transfer_scope='field resident; particle H2D and all outputs D2H remain', **metrics(cpu, hot))
        moved = positions.copy()
        moved[:, 0] += .002*np.cos(np.arange(n))
        moved[:, 1] += .002*np.sin(np.arange(n))
        invalid = element_hints.copy(); invalid[::97] = 0
        for name, x, hints, forced in [('actual', positions, element_hints, 0),
                ('moved', moved, element_hints, 0), ('moved_forced_fallback', moved, invalid, len(invalid[::97]))]:
            def locations(target):
                return batched(lambda s: target.locate_batch(x[s], element_hints=hints[s], local_hints=local_hints[s]))
            a, ta = timed(lambda: locations(mesh)); b, tb = timed(lambda: locations(candidate))
            matched = (a[0] == b[0]) & (a[0] > 0)
            result['locate'].append(dict(case=name, point_count=n, forced_invalid_hints=forced,
                forced_fallback_note='known invalid hints only, not total Newton/fallback count',
                displacement_m=0. if name == 'actual' else .002,
                element_ids_exact=np.array_equal(a[0], b[0]), element_id_mismatches=int(np.count_nonzero(a[0] != b[0])),
                cpu_unresolved=int(np.count_nonzero(a[0] == 0)), candidate_unresolved=int(np.count_nonzero(b[0] == 0)),
                matched_located_count=int(matched.sum()), local_coordinate_max_abs_difference=(
                    float(np.max(np.abs(a[1][matched]-b[1][matched]))) if matched.any() else None),
                local_coordinates=relative_metrics(a[1][matched], b[1][matched]), cpu=ta, candidate_inclusive=tb))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--anchor', type=int, choices=(4110, 5680), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--wall-seconds', type=float, default=880.)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261009)
    args = p.parse_args()
    if not 0 < args.wall_seconds <= 900 or min(args.chunk_size, args.repeats, args.threads) <= 0:
        p.error('positive settings, wall <=900 required')
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.anchor == 5680 else '')
    if args.run.name != expected:
        p.error('run/anchor pairing mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(success=False, stage='initialization', training=False, future_truth_read=False,
        settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    def deadline(*_):
        raise TimeoutError('bounded spatial probe deadline')
    signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    start = time.perf_counter()
    try:
        import h5py
        from src.data.marker_checkpoint import read_marker_snapshot
        from src.models.native_state import NativeUnits, configuration_digest
        from src.models.native_coupling import NativeSourceMesh
        from src.models.native_cuda_spatial import NativeCudaSpatial
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        torch.cuda.reset_peak_memory_stats()
        result['stage'] = 'read_initial_files'
        field_path = args.run/f'jorek{args.anchor+1:06d}.h5'
        cloud_path = args.run/f'part{args.anchor:05d}.h5'
        with h5py.File(field_path, 'r') as f:
            scalar = lambda k: float(np.asarray(f[k][...]).ravel()[0])
            if any(scalar(k) != v for k,v in [('jorek_model',502),('n_order',3),('n_tor',1),('n_plane',1)]):
                raise ValueError('only native cubic model502 n0 initial fields supported')
            units = NativeUnits(scalar('t_norm'), scalar('central_density')*1e20, scalar('central_mass'), 5/3)
            units.validate()
            fields = f['values'][...]
            mesh = NativeSourceMesh(f['x'][...], f['size'][...], f['vertex'][...]-1, f['index'][...],
                units=units, projection='none', max_elements=5000)
            result['field_time_si'] = scalar('t_now')*units.t_norm_si
        cloud, particle_time, metadata = read_marker_snapshot(cloud_path)
        live = np.flatnonzero(cloud.alive)
        result.update(marker_slots=len(cloud.weight), live_markers=len(live), particle_time_si=particle_time,
            input_files=[str(field_path), str(cloud_path)], cloud_metadata=metadata,
            geometry=mesh.configuration, field_input_digest=configuration_digest(fields),
            cloud_input_digest=configuration_digest(dict(x=cloud.x, st=cloud.st, i_elm=cloud.i_elm, alive=cloud.alive)),
            input_scope='all initial live markers; full slot arrays loaded; initial field sampled without any time advancement')
        result['stage'] = 'cuda_geometry_build'
        candidate = NativeCudaSpatial(mesh, device='cuda')
        torch.cuda.synchronize()
        result.update(build_seconds=time.perf_counter()-start, spatial_configuration=candidate.configuration)
        print(json.dumps(dict(stage='spatial_probe', live_markers=len(live), marker_slots=len(cloud.weight))), flush=True)
        result['stage'] = 'full_live_points_and_locations'
        result['comparison'] = benchmark_spatial(mesh, candidate, fields, cloud.x[live], cloud.i_elm[live], cloud.st[live],
            chunk_size=args.chunk_size, repeats=args.repeats, synchronize=torch.cuda.synchronize)
        result.update(success=True, stage='completed', cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name())
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(total_seconds=time.perf_counter()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with (args.output_dir/'result.json').open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
