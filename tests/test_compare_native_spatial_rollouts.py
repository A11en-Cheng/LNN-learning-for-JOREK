"""Known-value examples through the full-event capture comparison interface."""
import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest


def capture():
    a = dict(values=np.ones((9, 4, 1, 2)), raw_deltas=np.zeros((9, 4, 1, 2)),
             memory=np.zeros((2, 3)))
    a.update({'cloud.'+k: np.zeros(shape) for k, shape in {
        'x': (3, 3), 'st': (3, 2), 'V_prev': (3, 3), 'P_imp': (3, 11),
        'weight': (3,), 't_birth': (3,)}.items()})
    a.update({'cloud.group': np.array('neon'), 'cloud.i_elm': np.array([1, 1, 0]),
              'cloud.i_life': np.array([1, 2, 0]),
              'feedback.values': np.zeros((5, 4, 1, 2)),
              'feedback.deltas': np.zeros((5, 4, 1, 2)),
              'shards.radius': np.array([1., 2.]), 'shards.ids': np.array([0, 1])})
    a['cloud.P_imp'][:, 0] = 1
    a['cloud.weight'][:] = 1
    return a


def metadata():
    return dict(event=1, time_si=0.25, clocks={'particle_time_si': .25},
                components={'clocks': 'clockhash', 'rng': {'numpy': 'rnghash'}})


def test_public_event_metrics_keep_field_degree_and_zero_reference_channels_separate():
    from tools.compare_native_spatial_rollouts import compare_event
    ref = capture(); got = deepcopy(ref)
    got['values'][3, 2, 0] = [2., 2.]
    got['raw_deltas'][0, 0, 0, 0] = 2.
    result = compare_event(ref, got, reference_metadata=metadata(), candidate_metadata=metadata())
    assert result['fields']['values']['per_field']['3']['per_degree']['2']['relative_l2'] == 1.
    assert result['fields']['raw_deltas']['all']['relative_l2'] is None
    assert result['fields']['raw_deltas']['all']['zero_reference_mismatch_count'] == 1
    assert result['cloud']['counts']['matched_live'] == 2
    assert result['projection']['values']['per_channel']['4']['all']['relative_l2'] == 0.
    assert result['rng']['equal'] and result['clocks']['equal']
    assert not result['all_arrays_exact']
    json.dumps(result, allow_nan=False)


def test_public_event_reports_reused_ids_circular_phi_and_exact_clock_rng_changes():
    from tools.compare_native_spatial_rollouts import compare_event
    ref = capture(); got = deepcopy(ref)
    ref['cloud.x'][0, 2] = 2*np.pi-.25
    got['cloud.x'][0, 2] = .25
    got['cloud.i_life'][1] = 3
    got['cloud.x'][1, 0] = 1000.  # Not a matching identity.
    got['cloud.x'][2, 0] = 1000.  # Inactive, not numerical evidence.
    rm = metadata(); cm = deepcopy(rm)
    cm['clocks']['particle_time_si'] = .5
    cm['components']['rng']['numpy'] = 'different'
    r = compare_event(ref, got, reference_metadata=rm, candidate_metadata=cm)
    assert r['cloud']['counts']['matched_live'] == 1
    assert r['cloud']['numeric']['x']['phi']['max_abs_difference'] == .5
    assert r['cloud']['numeric']['x']['R']['max_abs_difference'] == 0.
    assert r['cloud']['exact']['i_life']['mismatch_count'] == 1
    assert r['clocks']['mismatch_count'] == r['rng']['mismatch_count'] == 1
    assert r['cloud']['identity']['candidate_only_live_count'] == 1


def test_file_rollout_compares_all_matching_events_and_writes_finite_evidence(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_rollouts
    paths = [tmp_path/'reference', tmp_path/'candidate']
    for path in paths:
        path.mkdir()
        summary = dict(initial_binding='geometry_units_fields_controls', initial_model_digest='exact_state_dict',
            initial_spatial_mesh_binding='meshhash',
            initial_model_components={'weight': 'same-weight', 'term_lambda': 'same-buffer'},
            initial_rng_digest='same_rng', model_fingerprint='model',
            initial_components={'cloud': {'x': 'all_slots'}, 'clocks': 'same', 'shards': {'radius': 'same'}},
            source_configuration={'geometry': 'same', 'spatial_backend': 'scalar' if path == paths[0] else 'batched_v1'})
        (path/'result.json').write_text(json.dumps(summary))
        for event in (1, 2):
            meta = metadata(); meta['event'] = event; meta['time_si'] = event*.25
            (path/f'event-{event}.json').write_text(json.dumps(meta))
            np.savez(path/f'event-{event}-arrays.npz', **capture())
    output = tmp_path/'comparison.json'
    r = compare_rollouts(*paths, output=output, allow_spatial_backend_change=True)
    assert r['success'] and len(r['events']) == 2
    assert r['physical_binding']['verified']
    assert r['physical_binding']['allowed_source_changes'] == ['spatial_backend']
    assert all(e['all_arrays_exact'] for e in r['events'])
    assert json.loads(output.read_text())['success']
    with pytest.raises(FileExistsError):
        compare_rollouts(*paths, output=output)
    cli_output = tmp_path/'cli.json'
    command = [sys.executable, '-m', 'tools.compare_native_spatial_rollouts',
               '--reference', str(paths[0]), '--candidate', str(paths[1]), '--output', str(cli_output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(cli_output.read_text())['success']
    assert not json.loads(cli_output.read_text())['physical_binding']['verified']
    summary['source_configuration']['geometry'] = 'different'
    (paths[1]/'result.json').write_text(json.dumps(summary))
    changed = compare_rollouts(*paths, output=tmp_path/'wrong-geometry.json')
    assert not changed['physical_binding']['verified']
    summary['source_configuration']['geometry'] = 'same'
    summary['source_configuration']['unapproved_empty_option'] = {}
    (paths[1]/'result.json').write_text(json.dumps(summary))
    changed = compare_rollouts(*paths, output=tmp_path/'empty-option.json', allow_spatial_backend_change=True)
    assert not changed['physical_binding']['verified']


def test_intentional_backend_change_keeps_weights_and_physical_state_binding_strict(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_rollouts
    roots = [tmp_path/'ref', tmp_path/'got']
    for i, root in enumerate(roots):
        root.mkdir()
        np.savez(root/'event-1-arrays.npz', **capture())
        (root/'event-1.json').write_text(json.dumps(metadata()))
        summary = dict(initial_binding='physical', initial_model_digest='weights', initial_rng_digest='rng',
            initial_spatial_mesh_binding='meshhash',
            initial_model_components={'weight': 'same-weight', 'term_lambda': 'same-buffer'},
            model_fingerprint=f'backend_specific_{i}', initial_components={'cloud': 'same', 'clocks': 'same', 'shards': 'same'},
            source_configuration={'spatial_backend': ('scalar', 'batched_v1')[i]})
        (root/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'allowed.json', allow_spatial_backend_change=True)
    assert r['physical_binding']['verified']
    assert r['physical_binding']['model_fingerprint_status'] == 'unequal_expected'
    summary['initial_model_components']['term_lambda'] = 'changed-buffer'
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'buffer.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']
    assert r['physical_binding']['model_fingerprint_status'] == 'unequal_unverified'
    summary['initial_model_components']['term_lambda'] = 'same-buffer'
    summary['source_configuration']['spatial_backend'] = 'scalar'
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'unexpected-fingerprint.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']
    summary['source_configuration']['spatial_backend'] = 'batched_v1'
    summary['initial_model_digest'] = 'different_weights'
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'rejected.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']


@pytest.mark.parametrize('key,index', [('values', (0, 0, 0, 0)), ('cloud.x', (2, 0))])
def test_nonfinite_even_in_inactive_slots_is_explicitly_rejected_with_count(key, index):
    from tools.compare_native_spatial_rollouts import compare_event
    ref = capture(); got = deepcopy(ref); got[key][index] = np.nan
    with pytest.raises(ValueError, match='nonfinite_count=1'):
        compare_event(ref, got, reference_metadata=metadata(), candidate_metadata=metadata())


def test_nonfinite_clock_rejected_and_partial_capture_never_qualified():
    from tools.compare_native_spatial_rollouts import compare_event
    meta = metadata(); meta['clocks']['particle_time_si'] = float('nan')
    with pytest.raises(ValueError, match='finite JSON metadata'):
        compare_event(capture(), capture(), reference_metadata=meta, candidate_metadata=metadata())
    partial = capture(); del partial['cloud.weight']
    with pytest.raises(ValueError, match='incomplete full-event'):
        compare_event(partial, capture(), reference_metadata=metadata(), candidate_metadata=metadata())


def test_failed_capture_comparison_preserves_a_finite_failure_file(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_rollouts
    roots = [tmp_path/'a', tmp_path/'b']
    for i, root in enumerate(roots):
        root.mkdir(); a = capture()
        if i: a['cloud.weight'][0] = np.inf
        np.savez(root/'event-1-arrays.npz', **a)
        (root/'event-1.json').write_text(json.dumps(metadata()))
    output = tmp_path/'failure.json'
    result = compare_rollouts(*roots, output=output)
    assert not result['success'] and result['failed_stage'] == 'event-1'
    assert 'nonfinite_count=1' in result['error']
    assert json.loads(output.read_text()) == result


def test_spatial_configuration_whitelist_binds_geometry_and_rejects_formula_changes(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_rollouts
    roots = [tmp_path/'ref', tmp_path/'cuda']
    for i, root in enumerate(roots):
        root.mkdir(); np.savez(root/'event-1-arrays.npz', **capture())
        (root/'event-1.json').write_text(json.dumps(metadata()))
        summary = dict(initial_binding='physical', initial_model_digest='weights', initial_rng_digest='rng',
            initial_spatial_mesh_binding='meshhash',
            initial_model_components={'weight': 'same-weight', 'term_lambda': 'same-buffer'},
            model_fingerprint=f'backend_{i}', initial_components={'cloud': 'same', 'clocks': 'same', 'shards': 'same'},
            source_configuration={'mesh': {'geometry': 'real-mesh'}, 'spatial_backend': ('batched_v1', 'cuda_fp64_v1')[i]})
        if i:
            summary['source_configuration']['spatial_configuration'] = dict(
                version='native-cuda-spatial-fp64-v1', backend='cuda_fp64_v1', device='cuda:0', test_only=False,
                geometry='real-mesh', mesh_binding='meshhash', precision='float64',
                field_transfer='event_owned_named_snapshots_v1', locator='hinted_newton_cpu_scalar_fallback',
                max_iterations=25, damping=[1., .5, .25, .125, .0625, .03125],
                bounds_tolerance=1e-9, residual_tolerance=1e-10, roundoff_factor=256., edge_guard=1e-8)
        (root/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'valid.json', allow_spatial_backend_change=True)
    assert r['physical_binding']['verified']
    summary['source_configuration']['spatial_configuration']['mesh_binding'] = 'not-the-live-mesh'
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'mesh-binding.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']
    summary['source_configuration']['spatial_configuration']['mesh_binding'] = 'meshhash'
    summary['source_configuration']['spatial_configuration']['geometry'] = 'wrong'
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'wrong.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']
    summary['source_configuration']['spatial_configuration']['geometry'] = 'real-mesh'
    summary['source_configuration']['spatial_configuration']['residual_tolerance'] = 1e-5
    (roots[1]/'result.json').write_text(json.dumps(summary))
    r = compare_rollouts(*roots, output=tmp_path/'formula.json', allow_spatial_backend_change=True)
    assert not r['physical_binding']['verified']


def timing_run(root, seconds):
    chain = root/'chain'; chain.mkdir(parents=True)
    summary = dict(success=True, profiled=False, capture_arrays=False,
        initial_binding='physical', initial_model_digest='weights', initial_model_components={'weight': 'same'},
        initial_spatial_mesh_binding='mesh', initial_rng_digest='rng', model_fingerprint='model',
        initial_components={'cloud': 'same', 'clocks': 'same', 'shards': 'same'},
        source_configuration={'spatial_backend': 'batched_v1'},
        events=[dict(event=i+1, time_si=(i+1)*.25, phase='base', seconds=t) for i, t in enumerate(seconds)])
    (chain/'result.json').write_text(json.dumps(summary))
    (root/'result.json').write_text(json.dumps(dict(success=True, settings=dict(
        deterministic=True, threads=2, chunk_size=2048, birth_max_proposals=1000000))))
    return summary


def test_optional_unprofiled_timing_compares_observed_event_ratios_not_capture_timings(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_timings
    a, b = tmp_path/'reference', tmp_path/'candidate'
    timing_run(a, [8., 4.]); timing_run(b, [4., 2.])
    result = compare_timings(a, b)
    assert result['eligible']
    assert result['total']['reference_seconds'] == 12.
    assert result['total']['candidate_seconds'] == 6.
    assert result['total']['reference_over_candidate'] == 2.
    assert result['events'][1]['reference_over_candidate'] == 2.
    assert result['speedup_claim'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['capture', 'profile', 'threads', 'duration', 'weights'])
def test_timing_rejects_contaminated_or_unmatched_evidence_without_ratios(tmp_path, change):
    from tools.compare_native_spatial_rollouts import compare_timings
    a, b = tmp_path/'a', tmp_path/'b'
    timing_run(a, [8., 4.]); summary = timing_run(b, [4., 2.])
    if change == 'capture': summary['capture_arrays'] = True
    if change == 'profile': summary['profiled'] = True
    if change == 'duration': summary['events'][1]['seconds'] = 0.
    if change == 'weights': summary['initial_model_digest'] = 'different'
    if change == 'threads':
        outer = json.loads((b/'result.json').read_text()); outer['settings']['threads'] = 8
        (b/'result.json').write_text(json.dumps(outer))
    (b/'chain'/'result.json').write_text(json.dumps(summary))
    result = compare_timings(a, b)
    assert not result['eligible'] and result['total'] is None and result['events'] == []
    json.dumps(result, allow_nan=False)


def test_rollout_report_optional_timing_is_separate_and_bound_to_captures(tmp_path):
    from tools.compare_native_spatial_rollouts import compare_rollouts
    timing_roots = [tmp_path/'time-a', tmp_path/'time-b']
    capture_roots = [tmp_path/'capture-a', tmp_path/'capture-b']
    for i in range(2):
        summary = timing_run(timing_roots[i], [8./(i+1), 4./(i+1)])
        root = capture_roots[i]; root.mkdir()
        summary['capture_arrays'] = True
        (root/'result.json').write_text(json.dumps(summary))
        for event in (1, 2):
            meta = metadata(); meta.update(event=event, time_si=event*.25)
            (root/f'event-{event}.json').write_text(json.dumps(meta))
            np.savez(root/f'event-{event}-arrays.npz', **capture())
    report = compare_rollouts(*capture_roots, output=tmp_path/'combined.json',
        reference_timing=timing_roots[0], candidate_timing=timing_roots[1])
    assert report['success'] and report['unprofiled_timing']['eligible']
    assert report['unprofiled_timing']['total']['reference_over_candidate'] == 2.
