"""Benchmark qualification through complete public Source.advance windows."""
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_locator_events import event_fixture, assert_runtime_exact


def test_public_window_records_all_arrays_and_alternating_timing_without_mutation():
    from tools.benchmark_native_neighbor_locator import benchmark_source_window

    source, state, fields = event_fixture(backend='batched_v1')
    before = deepcopy(state)
    result = benchmark_source_window(source, source, state, 1e-9, fields, fields,
                                     repeats=2, count_pass=False)
    assert result['success']
    assert [(row['mode'], row['warmup']) for row in result['iterations']] == [
        ('baseline', True), ('neighbor', True), ('baseline', False),
        ('neighbor', False), ('neighbor', False), ('baseline', False)]
    comparison = result['iterations'][-1]['comparison']
    assert set(comparison['arrays']) == {
        'cloud.'+name for name in state.cloud.array_names()
    } | {'cloud.alive', 'feedback.values', 'feedback.deltas'}
    assert comparison['arrays']['cloud.P_imp']['relative_l2'] == 0.
    assert comparison['arrays']['feedback.deltas']['max_abs_difference'] == 0.
    assert result['inputs_unchanged']
    assert_runtime_exact(state, before)


class ReturnedPerturbation:
    """A public source adapter with a prescribed output-contract violation."""
    def __init__(self, source, change):
        self.source, self.change = source, change

    def __getattr__(self, name):
        return getattr(self.source, name)

    def advance(self, *args, **kwargs):
        out, diag = self.source.advance(*args, **kwargs)
        self.change(out, diag)
        return out, diag


def test_counts_pass_is_separate_from_public_window_timing_and_outputs():
    from tools.benchmark_native_neighbor_locator import benchmark_source_window

    source, state, fields = event_fixture(backend='batched_v1', scenario='crossing')
    result = benchmark_source_window(source, source, state, 1e-9, fields, fields,
                                     repeats=1, count_pass=True)
    assert result['success']
    assert len(result['iterations']) == 4  # warmup pair + measured pair only
    assert result['locator_counts']['baseline']['batch_calls'] > 0
    assert result['locator_counts']['baseline']['global_fallback_points'] is None
    assert result['count_pass_contract_valid']
    assert result['locator_count_pass_timing_eligible'] is False


def test_public_window_reports_float_error_without_requiring_bitwise_values():
    from tools.benchmark_native_neighbor_locator import benchmark_source_window

    source, state, fields = event_fixture(backend='batched_v1')
    def change(out, diag):
        out.cloud.st[0, 0] += 1e-6
    result = benchmark_source_window(source, ReturnedPerturbation(source, change),
                                     state, 1e-9, fields, fields, repeats=1)
    assert result['success']
    comparison = result['iterations'][-1]['comparison']
    assert not comparison['all_arrays_exact']
    metric = comparison['arrays']['cloud.st']
    assert metric['max_abs_difference'] == pytest.approx(1e-6)
    assert metric['relative_l2'] > 0
    assert metric['pointwise_relative']['max'] > 0
    assert metric['count'] == state.cloud.st.size  # All slots, not just live.
    assert comparison['increments']['x']['count'] == 3  # One persistent live identity.


@pytest.mark.parametrize('change_kind',['float_interval','level','lifetime','configuration'])
def test_transport_arrays_separate_from_scalar_metadata_but_discrete_history_stays_strict(change_kind):
    from src.models.native_coupling import NativeMarkerSource
    from src.models.marker_transport import LocalSubcycling,initialize_transport_history
    from tools.benchmark_native_neighbor_locator import benchmark_source_window
    old,state,fields=event_fixture(backend='batched_v1')
    policy=LocalSubcycling()
    source=NativeMarkerSource(old.mesh,atomic=old.atomic,atomic_version=old.configuration['atomic'],
        potentials_ev=old.potentials_ev,F0=old.F0,diffusion_ju=old.diffusion_ju,
        spatial_backend='batched_v1',transport_policy=policy)
    state=initialize_transport_history(state,policy,previous_dt_si=state.clock.particle_dt_si)
    def change(out,diag):
        h=out.transport_history
        if change_kind=='float_interval':h.previous_dt_si[0]=np.nextafter(h.previous_dt_si[0],np.inf)
        elif change_kind=='level':h.level[0]=policy.min_level
        elif change_kind=='lifetime':h.lifetime[0]+=1
        else:h.configuration['version']='changed'
    result=benchmark_source_window(source,ReturnedPerturbation(source,change),state,1e-9,
        fields,fields,repeats=1)
    comparison=result['iterations'][-1]['comparison']
    assert result['success']==(change_kind=='float_interval')
    if change_kind=='float_interval':
        assert comparison['required_exact']['metadata']
        assert not comparison['all_arrays_exact']
        assert comparison['arrays']['transport_history.previous_dt_si']['relative_l2']>0
        assert comparison['required_exact']['transport_history.level']
        assert comparison['required_exact']['transport_history.lifetime']


@pytest.mark.parametrize('violation', ['element', 'life', 'group', 'clock', 'gather', 'rng'])
def test_public_window_rejects_discrete_event_contract_changes(violation):
    from tools.benchmark_native_neighbor_locator import benchmark_source_window
    import random

    source, state, fields = event_fixture(backend='batched_v1')
    def change(out, diag):
        if violation == 'element': out.cloud.i_elm[0] = 0
        elif violation == 'life': out.cloud.i_life[0] += 1
        elif violation == 'group': out.cloud.group = 'different'
        elif violation == 'clock': out.clock.window_index += 1
        elif violation == 'gather': out.feedback.gather_time_si += 1e-12
        else: random.random()
    result = benchmark_source_window(source, ReturnedPerturbation(source, change),
                                     state, 1e-9, fields, fields, repeats=1)
    assert not result['success']
    assert not all(result['iterations'][-1]['comparison']['required_exact'].values())


def test_public_window_rejects_mutation_of_private_call_input_even_if_original_is_unchanged():
    from tools.benchmark_native_neighbor_locator import benchmark_source_window

    source, state, fields = event_fixture(backend='batched_v1')
    class MutatingSource:
        def advance(self, marker, *args, **kwargs):
            out = source.advance(marker, *args, **kwargs)
            marker.cloud.x[0, 0] += .1
            return out
    result = benchmark_source_window(source, MutatingSource(), state, 1e-9,
                                     fields, fields, repeats=1)
    assert not result['success']
    assert not result['iterations'][-1]['call_inputs_unchanged']


@pytest.mark.parametrize('scenario', ['crossing', 'lost'])
def test_real_neighbor_adapter_qualifies_through_complete_public_source_window(scenario):
    from tools.benchmark_native_neighbor_locator import benchmark_source_window
    from src.models.native_neighbor_locator_experiment import NativeNeighborMarkerSource
    from tests.test_native_neighbor_locator import shared_event

    source, state, fields = shared_event(scenario)
    candidate = NativeNeighborMarkerSource(source, max_hops=4)
    result = benchmark_source_window(source, candidate, state, 1e-9, fields, fields,
                                     repeats=1, count_pass=True)
    assert result['success']
    assert result['locator_counts']['neighbor']['neighbor_accepted'] == (scenario == 'crossing')
    assert result['locator_counts']['neighbor']['global_fallback_points'] == (scenario == 'lost')
    assert result['locator_counts']['neighbor']['topology_bytes'] > 0
    assert result['count_pass_comparisons']['neighbor']['required_exact']['cloud.i_elm']


def test_cli_help_exposes_safe_defaults_without_building_real_data(capsys):
    from tools.benchmark_native_neighbor_locator import main
    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert '--max-hops' in text and '--repeats' in text


def test_public_window_reports_each_feedback_channel_without_zero_denominator_epsilon():
    from tools.benchmark_native_neighbor_locator import benchmark_source_window

    source, state, fields = event_fixture(backend='batched_v1')
    def change(out, diag):
        # The prescribed atomic fixture has exactly zero radiation channel.
        out.feedback.values[1, 0, 0, 0] = 1e-6
        out.feedback.deltas[1, 0, 0, 0] = 2e-6
    result = benchmark_source_window(source, ReturnedPerturbation(source, change),
                                     state, 1e-9, fields, fields, repeats=1)
    assert result['success']
    channels = result['iterations'][-1]['comparison']['feedback_channels']
    assert set(channels) == {'values', 'deltas'}
    assert set(channels['values']) == set(map(str, range(5)))
    assert set(channels['deltas']) == set(map(str, range(5)))
    for key, expected in [('values', 1e-6), ('deltas', 2e-6)]:
        metric = channels[key]['1']
        assert metric['relative_l2'] is None
        assert metric['undefined']['relative_l2'] == 'zero_reference_norm'
        assert metric['zero_reference_mismatch_count'] == 1
        assert metric['max_abs_difference'] == expected
