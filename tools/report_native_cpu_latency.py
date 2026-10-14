"""Compare matched unprofiled, no-training full-cloud latency probes."""
import argparse
import json
from pathlib import Path


def compare(before, after, before_chain, after_chain):
    for run, chain in ((before, before_chain), (after, after_chain)):
        if not run['success'] or not chain['success'] or chain['profiled']:
            raise ValueError('successful unprofiled probes required for speedup')
        if run['training'] or run['future_truth_feedback'] or chain['optimizer_updates']:
            raise ValueError('no training or truth feedback permitted')
        if not chain['model_unchanged'] or len(chain['events']) != 2:
            raise ValueError('two complete events with frozen weights required')
    keys = ('run', 'anchor', 'seed', 'chunk_size', 'threads', 'spatial_backend',
            'birth_max_proposals', 'memory_mib')
    for key in keys:
        if before['settings'][key] != after['settings'][key]:
            raise ValueError('probe setting mismatch: ' + key)
    for key in ('deterministic', 'capture_arrays'):
        if before['settings'].get(key, False) != after['settings'].get(key, False):
            raise ValueError('probe setting mismatch: ' + key)
    for key in ('source_fingerprint', 'model_fingerprint', 'controls', 'device',
                'torch_version', 'numpy_version', 'environment'):
        if before[key] != after[key]:
            raise ValueError('probe context mismatch: ' + key)
    if before_chain['initial_components'] != after_chain['initial_components']:
        raise ValueError('initial physical state or RNG mismatch')
    events = []
    for old, new in zip(before_chain['events'], after_chain['events']):
        for key in ('event', 'phase', 'time_si', 'marker_slots', 'live_markers'):
            if old[key] != new[key]:
                raise ValueError('event identity/clock/cloud mismatch: ' + key)
        changed = [key for key in old['components']
                   if old['components'][key] != new['components'].get(key)]
        if set(old['components']) != set(new['components']):
            raise ValueError('component schema mismatch')
        events.append(dict(event=old['event'], before_seconds=old['seconds'],
            after_seconds=new['seconds'], speedup=old['seconds']/new['seconds'],
            bitwise_components_equal=not changed, changed_component_groups=changed))
    old_total = sum(e['before_seconds'] for e in events)
    new_total = sum(e['after_seconds'] for e in events)
    return dict(scope='single_paired_measurement_not_statistical_speedup_guarantee',
        events=events, two_event_seconds_before=old_total, two_event_seconds_after=new_total,
        event_speedup=old_total/new_total,
        event_latency_reduction_fraction=1-new_total/old_total,
        complete_probe_seconds_before=before['elapsed_seconds'],
        complete_probe_seconds_after=after['elapsed_seconds'],
        all_event_components_bitwise_equal=all(e['bitwise_components_equal'] for e in events),
        final_state_digest_equal=before_chain['final_state_digest']==after_chain['final_state_digest'],
        final_rng_digest_equal=before_chain['final_rng_digest']==after_chain['final_rng_digest'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda root, path: json.loads((root/path).read_text())
    result = compare(read(args.before, 'result.json'), read(args.after, 'result.json'),
                     read(args.before, 'chain/result.json'), read(args.after, 'chain/result.json'))
    result.update(before=str(args.before), after=str(args.after))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result['all_event_components_bitwise_equal'] and result['final_rng_digest_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
