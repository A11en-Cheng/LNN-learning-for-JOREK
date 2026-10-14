"""Offline full-event comparison; no physics execution, GPU, or tolerance gate.

NPZ keys: values/raw_deltas/memory, cloud.<array>, cloud.group (Unicode
scalar), feedback.values/deltas, shards.<array>. Optional fields_previous and
native_density_source are compared when captured. Event JSON supplies event,
time_si, explicit clocks and components.rng digests. Never load pickle arrays.
Native units, all slots retained; cloud numerical metrics use matched live IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.native_replay_metrics import compare_cloud, relative_metrics

_BINDING_KEYS = ('initial_binding', 'initial_model_digest', 'initial_rng_digest', 'model_fingerprint',
                 'initial_components', 'initial_model_components', 'source_configuration',
                 'initial_spatial_mesh_binding')


def compare_timings(reference_dir, candidate_dir, *, allow_spatial_backend_change=False):
    """Optional independent unprofiled timing artifacts; never use capture times.

    Accept run roots or their chain directories. One observation per backend is
    an observed ratio, not a statistical speedup claim. Hardware/exclusivity and
    numerical acceptability remain external evidence, not inferred from times.
    """
    result = dict(eligible=False, speedup_claim=False, events=[], total=None,
        scope='single_observed_event_wall_ratio_excludes_build_init_capture_disk_io',
        limitations=['not_repeated_statistical_speedup', 'hardware_and_exclusivity_not_verified',
                     'numeric_acceptability_not_inferred_from_timing'])
    try:
        summaries, settings, provenance = [], [], []
        for root in map(Path, (reference_dir, candidate_dir)):
            chain = root/'chain' if (root/'chain').is_dir() else root
            paths = [chain/'result.json', chain.parent/'result.json']
            records = [json.loads(p.read_text()) for p in paths]
            for record in records:
                json.dumps(record, allow_nan=False)
            summary, outer = records
            if summary.get('success') is not True or outer.get('success') is not True:
                raise ValueError('timing requires completed successful runs')
            if summary.get('profiled') is not False or summary.get('capture_arrays') is not False:
                raise ValueError('capture/profile timing cannot be an uninstrumented baseline')
            required = ('deterministic', 'threads', 'chunk_size', 'birth_max_proposals')
            setup = outer.get('settings', {})
            if any(k not in setup for k in required):
                raise ValueError('timing execution settings missing')
            settings.append({k: setup[k] for k in required})
            summaries.append(summary)
            provenance.append([dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
        result['provenance'] = dict(reference=provenance[0], candidate=provenance[1])
        result['settings'] = dict(reference=settings[0], candidate=settings[1])
        if settings[0] != settings[1]:
            raise ValueError('timing execution settings differ')
        bindings = [{k: s.get(k) for k in _BINDING_KEYS} for s in summaries]
        result['initial_bindings'] = dict(reference=bindings[0], candidate=bindings[1])
        result['physical_binding'] = _physical_binding(*bindings, allow_spatial_backend_change=allow_spatial_backend_change)
        if not result['physical_binding']['verified']:
            raise ValueError('timing physical/model/source binding is not verified')
        events = [s.get('events', []) for s in summaries]
        if not events[0] or len(events[0]) != len(events[1]):
            raise ValueError('matched nonempty timing events required')
        rows = []
        for a, b in zip(*events):
            if any(k not in a or k not in b or a[k] != b[k] for k in ('event', 'time_si', 'phase')):
                raise ValueError('timing event endpoints differ')
            x, y = a['seconds'], b['seconds']
            if not np.isfinite([x, y]).all() or min(x, y) <= 0:
                raise ValueError('finite positive event durations required')
            rows.append(dict(event=a['event'], time_si=a['time_si'], reference_seconds=x,
                             candidate_seconds=y, reference_over_candidate=x/y))
        x = sum(row['reference_seconds'] for row in rows)
        y = sum(row['candidate_seconds'] for row in rows)
        result.update(eligible=True, events=rows,
                      total=dict(reference_seconds=x, candidate_seconds=y, reference_over_candidate=x/y))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result['reason'] = str(exc)
    return result


def _metric(reference, candidate):
    result = relative_metrics(reference, candidate)
    result['exact'] = bool(np.array_equal(reference, candidate))
    result['mismatch_count'] = int(np.count_nonzero(np.asarray(reference) != candidate))
    return result


def compare_rollouts(reference_dir, candidate_dir, *, output, allow_spatial_backend_change=False,
                     reference_timing=None, candidate_timing=None):
    """Read paired profiler directories one event at a time; preserve failures.

    Success means a complete comparison was produced, NOT numerical equivalence
    or performance qualification. Output is exclusive-create, never overwritten.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (reference_timing is None) != (candidate_timing is None):
        raise ValueError('both independent timing directories required')
    paths = [Path(reference_dir), Path(candidate_dir)]
    result = dict(schema='native-spatial-rollouts-v1', success=False, events=[],
        reference=str(paths[0]), candidate=str(paths[1]),
        scope='offline_numeric_comparison_not_a_speed_or_tolerance_gate',
        timing='capture runs are not uninstrumented timing baselines')
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    stage = 'capture_inventory'
    try:
        inventories = [{int(p.name.split('-')[1]): p for p in root.glob('event-*-arrays.npz')} for root in paths]
        if not inventories[0] or inventories[0].keys() != inventories[1].keys():
            raise ValueError('nonempty matched full-event capture sets required')
        bindings = []
        for root in paths:
            summary = root/'result.json'
            if summary.exists():
                data = json.loads(summary.read_text())
                bindings.append({k: data.get(k) for k in _BINDING_KEYS})
            else:
                bindings.append(None)
        result['initial_bindings'] = dict(reference=bindings[0], candidate=bindings[1],
            comparison=_exact_tree(*bindings) if all(v is not None for v in bindings) else None)
        result['physical_binding'] = _physical_binding(*bindings, allow_spatial_backend_change=allow_spatial_backend_change)
        for event in sorted(inventories[0]):
            stage = f'event-{event}'
            npzs = [inventory[event] for inventory in inventories]
            metas = [root/f'event-{event}.json' for root in paths]
            metadata = [json.loads(p.read_text()) for p in metas]
            if any(m.get('event') != event for m in metadata):
                raise ValueError('event filename and metadata disagree')
            with np.load(npzs[0], allow_pickle=False) as a, np.load(npzs[1], allow_pickle=False) as b:
                comparison = compare_event(a, b, reference_metadata=metadata[0], candidate_metadata=metadata[1])
            comparison['provenance'] = {side: dict(capture=str(npz), capture_sha256=sha(npz),
                metadata=str(meta), metadata_sha256=sha(meta))
                for side, npz, meta in zip(('reference', 'candidate'), npzs, metas)}
            result['events'].append(comparison)
        if reference_timing is not None:
            stage = 'optional_unprofiled_timing'
            timing = compare_timings(reference_timing, candidate_timing,
                                     allow_spatial_backend_change=allow_spatial_backend_change)
            result['unprofiled_timing'] = timing
            if timing['eligible']:
                same = all(_physical_binding(binding, timing['initial_bindings'][side],
                    allow_spatial_backend_change=False)['verified']
                    for side, binding in zip(('reference', 'candidate'), bindings))
                endpoints = [(e['event'], e['time_si']) for e in result['events']]
                same = same and endpoints == [(e['event'], e['time_si']) for e in timing['events']]
                timing['capture_binding_verified'] = same
                if not same:
                    timing.update(eligible=False, events=[], total=None,
                                  reason='independent timing does not match numerical capture binding/endpoints')
        result['success'] = True
    except Exception as exc:
        result.update(failed_stage=stage, error_type=type(exc).__name__, error=str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-timing', type=Path, help='separate no-profile/no-capture run root or chain directory')
    parser.add_argument('--candidate-timing', type=Path, help='paired separate no-profile/no-capture timing run')
    parser.add_argument('--allow-spatial-backend-change', action='store_true',
                        help='report-only intentional spatial backend comparison; never a checkpoint restore')
    args = parser.parse_args()
    if (args.reference_timing is None) != (args.candidate_timing is None):
        parser.error('both timing paths must be supplied together')
    result = compare_rollouts(args.reference, args.candidate, output=args.output,
                             allow_spatial_backend_change=args.allow_spatial_backend_change,
                             reference_timing=args.reference_timing, candidate_timing=args.candidate_timing)
    print(json.dumps(dict(success=result['success'], events=len(result['events']),
        output=str(args.output), error=result.get('error')), allow_nan=False))
    return 0 if result['success'] else 1


def _channels(reference, candidate, count, label):
    if reference.ndim != 4 or reference.shape[:3] != (count, 4, 1):
        raise ValueError(f'{label} requires shape ({count},4,1,N)')
    channels = {}
    for channel in range(count):
        channels[str(channel)] = dict(all=_metric(reference[channel], candidate[channel]),
            per_degree={str(d): _metric(reference[channel, d], candidate[channel, d]) for d in range(4)})
    return dict(all=_metric(reference, candidate), **{label: channels})


def _exact_tree(reference, candidate):
    def flatten(value, prefix=''):
        if isinstance(value, dict):
            return {prefix: ('dict', len(value)),
                    **{k: v for key, item in value.items() for k, v in flatten(item, prefix+'/'+str(key)).items()}}
        if isinstance(value, (tuple, list)):
            return {prefix: (type(value).__name__, len(value)),
                    **{k: v for index, item in enumerate(value) for k, v in flatten(item, prefix+'/'+str(index)).items()}}
        return {prefix: (type(value).__name__, value)}
    a, b = flatten(reference), flatten(candidate)
    missing = sorted(a.keys()-b.keys()); extra = sorted(b.keys()-a.keys())
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    return dict(equal=not (missing or extra or changed), mismatch_count=len(changed),
                mismatch_paths=changed, reference_only_paths=missing, candidate_only_paths=extra)


def _physical_binding(reference, candidate, *, allow_spatial_backend_change):
    """Report-only whitelist; does not restore or alter any checkpoint."""
    if any(not isinstance(v, dict) or any(item is None for item in v.values()) for v in (reference, candidate)):
        return dict(verified=False, reason='missing_initial_binding_evidence', allowed_source_changes=[])
    a, b = dict(reference), dict(candidate)
    sa, sb = dict(a.pop('source_configuration')), dict(b.pop('source_configuration'))
    ba, bb = sa.pop('spatial_backend', 'scalar'), sb.pop('spatial_backend', 'scalar')
    spatial_valid = all(_spatial_configuration_valid(s, backend,
                        binding['initial_spatial_mesh_binding'])
                        for s, backend, binding in ((sa, ba, reference), (sb, bb, candidate)))
    spatial_changes = []
    if allow_spatial_backend_change:
        ca, cb = sa.get('spatial_configuration'), sb.get('spatial_configuration')
        if ca != cb:
            if spatial_valid:
                sa.pop('spatial_configuration', None); sb.pop('spatial_configuration', None)
                spatial_changes.append('spatial_configuration')
        if spatial_valid and (ba != bb or spatial_changes):
            # Ignore the source-containing fingerprint only for an actual
            # whitelisted migration; all model tensor hashes remain strict.
            a.pop('model_fingerprint'); b.pop('model_fingerprint')
    physical = _exact_tree(a, b)
    source = _exact_tree(sa, sb)
    known = all(v in ('scalar', 'batched_v1', 'cuda_fp64_v1') for v in (ba, bb))
    complete = all(isinstance(v.get('initial_components'), dict) and
                   all(v['initial_components'].get(k) for k in ('cloud', 'clocks', 'shards'))
                   and isinstance(v.get('initial_model_components'), dict) and bool(v['initial_model_components'])
                   for v in (reference, candidate))
    permitted = allow_spatial_backend_change or ba == bb
    verified = physical['equal'] and source['equal'] and known and complete and permitted and spatial_valid
    fingerprint_equal = reference['model_fingerprint'] == candidate['model_fingerprint']
    return dict(verified=verified,
        model_fingerprint_status=('equal' if fingerprint_equal else
            'unequal_expected' if verified and (ba != bb or spatial_changes) else 'unequal_unverified'),
        physical=physical, source_configuration_excluding_spatial_backend=source,
        reference_backend=ba, candidate_backend=bb, known_backends=known,
        full_initial_components_present=complete,
        allow_spatial_backend_change=allow_spatial_backend_change,
        spatial_configuration_valid=spatial_valid,
        allowed_source_changes=(['spatial_backend'] if ba != bb and known and permitted else [])+spatial_changes,
        scope='report_only_initial_digest_evidence_no_checkpoint_guard_changes')


def _spatial_configuration_valid(source, backend, actual_mesh_binding):
    config = source.get('spatial_configuration')
    if config is None:
        return backend != 'cuda_fp64_v1'
    if not isinstance(config, dict) or backend != 'cuda_fp64_v1':
        return False
    expected = dict(version='native-cuda-spatial-fp64-v1', backend='cuda_fp64_v1', test_only=False,
        precision='float64', field_transfer='event_owned_named_snapshots_v1',
        locator='hinted_newton_cpu_scalar_fallback', max_iterations=25,
        damping=[1., .5, .25, .125, .0625, .03125], bounds_tolerance=1e-9,
        residual_tolerance=1e-10, roundoff_factor=256., edge_guard=1e-8)
    # Only this explicit implementation's configuration is eligible. No generic
    # subtree deletion that could hide a changed geometry or solver policy.
    return (set(config) == set(expected) | {'device', 'geometry', 'mesh_binding'}
        and all(config[k] == v for k, v in expected.items())
        and config['geometry'] == source.get('mesh', {}).get('geometry')
        and isinstance(config['mesh_binding'], str) and bool(config['mesh_binding'])
        and config['mesh_binding'] == actual_mesh_binding
        and (config['device'] == 'cuda' or
             (isinstance(config['device'], str) and config['device'].startswith('cuda:') and config['device'][5:].isdigit())))


def compare_event(reference, candidate, *, reference_metadata, candidate_metadata):
    """Compare two complete captures (array mappings), rejecting partial records.

    Event number/time must identify the same endpoint. Zero/zero relative norms
    are zero; nonzero/zero is null with reasons, never epsilon regularized.
    ``all_arrays_exact`` includes inactive storage, unlike live cloud metrics.
    """
    for meta in (reference_metadata, candidate_metadata):
        try:
            json.dumps(meta, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError('finite JSON metadata required') from exc
    a = {k: np.asarray(reference[k]) for k in reference}
    b = {k: np.asarray(candidate[k]) for k in candidate}
    cloud_keys = ('x', 'st', 'V_prev', 'P_imp', 'weight', 'i_elm', 'i_life', 't_birth', 'group')
    required = {'values', 'raw_deltas', 'memory', 'feedback.values', 'feedback.deltas'} | {'cloud.'+k for k in cloud_keys}
    for name, arrays in (('reference', a), ('candidate', b)):
        missing = required-arrays.keys()
        if missing or not any(k.startswith('shards.') for k in arrays):
            raise ValueError(f'{name} incomplete full-event capture: missing {sorted(missing)} or shard arrays')
        for key, value in arrays.items():
            if key == 'cloud.group':
                if value.shape != () or value.dtype.kind != 'U':
                    raise ValueError('cloud.group requires a Unicode scalar')
            elif value.dtype.kind not in 'biuf':
                raise ValueError(f'{name} {key}: nonnumeric array')
            elif not np.isfinite(value).all():
                count = int(np.count_nonzero(~np.isfinite(value)))
                raise ValueError(f'{name} {key}: nonfinite_count={count}; no metrics hidden')
    if a.keys() != b.keys():
        raise ValueError('capture array keys differ; incomplete paired evidence')
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError(f'{key}: shape mismatch')
    for key in ('event', 'time_si'):
        if key not in reference_metadata or key not in candidate_metadata or reference_metadata[key] != candidate_metadata[key]:
            raise ValueError(f'unmatched event endpoint: {key}')
    for meta in (reference_metadata, candidate_metadata):
        if 'clocks' not in meta or not meta.get('components', {}).get('rng'):
            raise ValueError('explicit clocks and RNG digests required')
    cloud = lambda arrays: {k: arrays['cloud.'+k].item() if k == 'group' else arrays['cloud.'+k] for k in cloud_keys}
    fields = {k: _channels(a[k], b[k], 9, 'per_field') for k in ('values', 'raw_deltas')}
    if 'fields_previous' in a:
        fields['fields_previous'] = _channels(a['fields_previous'], b['fields_previous'], 9, 'per_field')
    result = dict(schema='native-spatial-rollout-comparison-v1', event=reference_metadata['event'],
        time_si=reference_metadata['time_si'], units='native_no_conversion',
        all_arrays_exact=all(np.array_equal(a[k], b[k]) for k in a),
        fields=fields, memory=_metric(a['memory'], b['memory']), cloud=compare_cloud(cloud(a), cloud(b)),
        projection={k: _channels(a['feedback.'+k], b['feedback.'+k], 5, 'per_channel') for k in ('values', 'deltas')},
        shards={k[7:]: _metric(a[k], b[k]) for k in a if k.startswith('shards.')},
        clocks=_exact_tree(reference_metadata['clocks'], candidate_metadata['clocks']),
        rng=_exact_tree(reference_metadata['components']['rng'], candidate_metadata['components']['rng']))
    if 'native_density_source' in a:
        result['native_density_source'] = _channels(a['native_density_source'], b['native_density_source'], 2, 'per_channel')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
