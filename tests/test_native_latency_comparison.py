from copy import deepcopy
import pytest
from tools.report_native_cpu_latency import compare


def fixture():
    run = dict(success=True, training=False, future_truth_feedback=False,
        settings=dict.fromkeys(('run','anchor','seed','chunk_size','threads','spatial_backend',
                                'birth_max_proposals','memory_mib'), 'same'),
        elapsed_seconds=15.)
    run.update(dict.fromkeys(('source_fingerprint','model_fingerprint','controls','device',
                             'torch_version','numpy_version','environment'), 'same'))
    chain = dict(success=True, profiled=False, optimizer_updates=0, model_unchanged=True,
        initial_components={'cloud':'a'}, final_state_digest='state', final_rng_digest='rng',
        events=[dict(event=i, phase='ordinary', time_si=float(i), marker_slots=100,
                     live_markers=10, components={'cloud':'b','rng':'r'}, seconds=5.) for i in (1,2)])
    return run, chain


def test_matched_timing_and_exact_state_are_separate():
    run, before = fixture(); after = deepcopy(before)
    for e in after['events']:e['seconds'] = 2.5
    result = compare(run, run, before, after)
    assert result['event_speedup'] == 2.
    assert result['all_event_components_bitwise_equal']
    after['events'][1]['components']['cloud'] = 'changed'
    result = compare(run, run, before, after)
    assert not result['all_event_components_bitwise_equal']
    assert result['events'][1]['changed_component_groups'] == ['cloud']


@pytest.mark.parametrize('mutation', ['profile', 'clock', 'initial', 'seed', 'deterministic', 'capture_arrays'])
def test_reject_incomparable_benchmarks(mutation):
    run, before = fixture(); other = deepcopy(run); after = deepcopy(before)
    if mutation == 'profile':after['profiled'] = True
    if mutation == 'clock':after['events'][0]['time_si'] += 1.
    if mutation == 'initial':after['initial_components']['cloud'] = 'changed'
    if mutation == 'seed':other['settings']['seed'] = 9
    if mutation in ('deterministic', 'capture_arrays'):other['settings'][mutation] = True
    with pytest.raises(ValueError):compare(run, other, before, after)
