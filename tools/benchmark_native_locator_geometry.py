"""CPU-only frozen/current locator diagnostic; not a full trajectory benchmark.

Timing and cProfile runs are separate. No geometry methods are wrapped: this
matters for production guards which disable reuse for overridden geometry.
The small curved mesh is explicitly a test fixture, not a real-cloud sample.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import signal
import sys
import time

import numpy as np

REFERENCE = Path('output/native-cpu-locator-2026-10-09/native_coupling_before.py')
REFERENCE_SHA = '1471f8567f1abcc0ccce115a996dda50fee549317cc2c76bbc774bc9c2d45c40'


def load_reference_mesh(path: Path):
    """Load trusted local Python under a separate package alias, never canonical."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    alias = 'src.models._locator_reference'
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(alias)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = previous
        raise
    return module.NativeSourceMesh, dict(path=str(path), sha256=digest, alias=alias)


def _digest(array):
    a = np.asarray(array)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def _comparison(a, b):
    return dict(element_ids_exact=bool(np.array_equal(a[0], b[0])),
                local_coordinates_exact=bool(np.array_equal(a[1], b[1])),
                local_max_abs=float(np.max(np.abs(a[1]-b[1]), initial=0)),
                unresolved_reference=int(np.count_nonzero(a[0] == 0)),
                unresolved_candidate=int(np.count_nonzero(b[0] == 0)))


def _profile_rows(profile):
    rows = []
    for (filename, line, name), (cc, nc, own, cumulative, _) in pstats.Stats(profile).stats.items():
        if name in {'_location_geometry', '_location_geometries', '_cubic_basis',
                    'locate', 'locate_batch'}:
            rows.append(dict(file=filename, line=line, function=name,
                             primitive_calls=cc, total_calls=nc,
                             self_seconds=own, cumulative_seconds=cumulative))
    return rows


def benchmark(args, result):
    from src.models import native_coupling as production
    from tests.test_native_locator_latency import locator_mesh

    reference_type, evidence = load_reference_mesh(args.reference_file)
    if evidence['sha256'] != args.reference_sha256:
        raise ValueError('Frozen reference SHA256 mismatch')
    result['reference'] = evidence
    result['candidate'] = dict(path=production.__file__, sha256=hashlib.sha256(
        Path(production.__file__).read_bytes()).hexdigest())
    meshes = dict(reference=locator_mesh(reference_type), candidate=locator_mesh())
    rng = np.random.default_rng(20261009)
    st = np.vstack((rng.uniform(0, 1, (64, 2)),
                    [[0, 0], [1, 1], [0, .5], [1, .5], [.5, 0], [.5, 1],
                     [1e-11, .5], [1-1e-11, .5]]))
    e = np.arange(len(st)) % 2
    rz = meshes['reference'].points(e, st, meshes['reference'].coordinates[:1])[2]
    positions = np.vstack((np.column_stack((rz, np.zeros(len(st)))),
                           [[9, 9, 0], [2-5e-10, .5, 0]]))
    hints = np.r_[e+1, 1, 1]
    local = np.vstack((st, [[.5, .5], [.5, .5]]))
    cases = dict(no_hints={}, correct_hints=dict(element_hints=hints, local_hints=local),
                 wrong_cell_hints=dict(element_hints=3-hints, local_hints=local),
                 invalid_hints=dict(element_hints=np.zeros_like(hints), local_hints=local))
    arrays = dict(input_positions=positions, input_element_hints=hints, input_local_hints=local)
    result.update(scope='curved_two_cell_fixture_not_real_trajectory',
                  fixture='tests.test_native_locator_latency.locator_mesh',
                  points=len(positions), input_digests={k: _digest(v) for k, v in arrays.items()},
                  mesh_configuration=meshes['candidate'].configuration, cases=[])
    methods = ['locate', 'locate_batch'] if args.method == 'both' else [args.method]
    for method in methods:
        for case, kwargs in cases.items():
            record = dict(method=method, case=case, timing_seconds={'reference': [], 'candidate': []},
                          order=[], profiles={}, repeat_exact=True)
            result['cases'].append(record)
            outputs = {}
            for repeat in range(args.repeats):
                order = ['reference', 'candidate'] if repeat % 2 == 0 else ['candidate', 'reference']
                record['order'].append(order)
                for side in order:
                    started = time.perf_counter()
                    out = getattr(meshes[side], method)(positions, **kwargs)
                    record['timing_seconds'][side].append(time.perf_counter()-started)
                    if side in outputs:
                        record['repeat_exact'] &= all(np.array_equal(x, y) for x, y in zip(outputs[side], out))
                    outputs[side] = out
            record['comparison'] = _comparison(outputs['reference'], outputs['candidate'])
            for side in ('reference', 'candidate'):
                key = f'{method}_{case}_{side}'
                arrays[key+'_elements'], arrays[key+'_local'] = outputs[side]
                profile = cProfile.Profile()
                profile_path = args.output.with_name(args.output.stem+'.'+key+'.prof')
                # Exclusive reservation; retain even a partial profile on bounded failure.
                with profile_path.open('xb'):
                    pass
                try:
                    profile.enable()
                    out = getattr(meshes[side], method)(positions, **kwargs)
                finally:
                    profile.disable()
                    profile.dump_stats(str(profile_path))
                record['profiles'][side] = dict(path=str(profile_path), work=_profile_rows(profile),
                    matches_unprofiled=all(np.array_equal(x, y) for x, y in zip(outputs[side], out)))
    array_path = args.output.with_suffix('.npz')
    with array_path.open('xb') as stream:
        np.savez(stream, **arrays)
    result['arrays_path'] = str(array_path)
    result['all_exact'] = all(c['repeat_exact'] and c['comparison']['element_ids_exact']
        and c['comparison']['local_coordinates_exact'] and
        all(p['matches_unprofiled'] for p in c['profiles'].values()) for c in result['cases'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Exclusive JSON; NPZ/profiles beside it')
    parser.add_argument('--reference-file', type=Path, default=REFERENCE, help='Trusted local executable Python')
    parser.add_argument('--reference-sha256', default=REFERENCE_SHA)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--method', choices=['locate', 'locate_batch', 'both'], default='both')
    parser.add_argument('--wall-seconds', type=int, default=300)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.wall_seconds <= 900:
        parser.error('Require repeats >= 1 and wall-seconds in [1,900]')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(success=False, settings=vars(args).copy(), timing_scope='unprofiled_locator_only',
                  profiling_scope='separate_extra_call_not_speedup', gpu_used=False)
    def timeout(*_):
        raise TimeoutError('Locator diagnostic wall budget exceeded')
    with args.output.open('x') as stream:
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.wall_seconds)
        try:
            benchmark(args, result)
            result['success'] = True
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            json.dump(result, stream, indent=2, default=str, allow_nan=False)
    return 0 if result['success'] and result['all_exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
