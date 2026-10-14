"""Known-value oracles at the public cloud/replay comparison seam."""
import json

import numpy as np
import pytest


def test_relative_metrics_separate_norm_pointwise_and_zero_reference_errors():
    from tools.native_replay_metrics import relative_metrics

    result = relative_metrics([3., 4., 0.], [6., 8., 12.])
    assert result['reference_l2'] == 5.
    assert result['difference_l2'] == 13.
    assert result['relative_l2'] == 2.6
    assert result['max_abs_difference'] == 12.
    assert result['max_abs_difference_over_reference_max_abs_nonpointwise'] == 3.
    assert result['pointwise_relative']['max'] == 1.
    assert result['pointwise_relative']['count'] == 2
    assert result['zero_reference_count'] == 1
    assert result['zero_reference_mismatch_count'] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('scale', [1e-200, 1., 1e200])
def test_relative_metrics_are_scale_invariant_with_explicit_undefined_zeros(scale):
    from tools.native_replay_metrics import relative_metrics

    result = relative_metrics(np.array([3., 4.])*scale, np.array([6., 8.])*scale)
    assert result['relative_l2'] == pytest.approx(1.)
    assert result['max_abs_difference_over_reference_max_abs_nonpointwise'] == pytest.approx(1.)
    zero = relative_metrics([0., 0.], [0., scale])
    assert zero['relative_l2'] is None
    assert zero['undefined']['relative_l2'] == 'zero_reference_norm'
    assert zero['pointwise_relative']['max'] is None
    assert zero['zero_reference_mismatch_count'] == 1
    json.dumps(zero, allow_nan=False)


def test_zero_to_zero_has_zero_relative_l2_but_empty_has_no_observations():
    from tools.native_replay_metrics import relative_metrics

    zero = relative_metrics([0., 0.], [0., 0.])
    assert zero['relative_l2'] == 0.
    assert zero['max_abs_difference_over_reference_max_abs_nonpointwise'] == 0.
    assert zero['pointwise_relative']['count'] == 0
    assert zero['pointwise_relative']['max'] is None
    empty = relative_metrics([], [])
    assert empty['relative_l2'] is None
    assert empty['undefined']['relative_l2'] == 'empty_comparison'


def cloud(live=(True, True, False)):
    from src.models.marker_state import MarkerCloud

    result = MarkerCloud.empty(capacity=len(live))
    result.i_elm[:] = np.asarray(live, dtype=int)
    result.i_life[:] = np.arange(len(live))+1
    result.weight[:] = [2., 3., 999.][:len(live)]
    return result


def test_cloud_components_phi_and_probabilities_exclude_inactive_numeric_values():
    from tools.native_replay_metrics import compare_cloud

    ref, got = cloud(), cloud()
    ref.x[:2] = [[3., 4., 2*np.pi-.25], [2., 3., .25]]
    got.x[:2] = [[6., 8., .25], [2., 3., .25]]
    got.x[2] = np.nan  # Inactive storage is excluded, not imputed.
    got.st[0] = [.2, .3]
    got.V_prev[0] = [1., 2., 3.]
    got.P_imp[0, :2] = [.75, .25]
    result = compare_cloud(ref, got)
    assert result['counts']['matched_live'] == 2
    assert result['counts']['reference_inactive_excluded'] == 1
    assert result['numeric']['x']['R']['max_abs_difference'] == 3.
    assert result['numeric']['x']['Z']['relative_l2'] == .8
    assert result['numeric']['x']['phi']['max_abs_difference'] == pytest.approx(.5)
    assert result['numeric']['st']['s']['zero_reference_mismatch_count'] == 1
    assert result['numeric']['V_prev']['phi']['max_abs_difference'] == 3.
    assert len(result['numeric']['P_imp']) == 11
    assert result['numeric']['P_imp']['1']['zero_reference_mismatch_count'] == 1
    assert result['increments'] is None
    json.dumps(result, allow_nan=False)


def test_cloud_missing_and_reused_identities_do_not_enter_numeric_comparisons():
    from tools.native_replay_metrics import compare_cloud

    ref, got = cloud((True, True, True)), cloud((True, True))
    got.i_life[1] = 3
    got.i_elm[1] = 7
    got.x[1] = 1e100
    got.t_birth[0] = .25
    result = compare_cloud(ref, got)
    assert result['counts']['matched_live'] == 1
    assert result['numeric']['x']['R']['count'] == 1
    assert result['numeric']['x']['R']['relative_l2'] == 0.
    assert result['identity']['reference_only_live_ids'] == [['001', 1, 2], ['001', 2, 3]]
    assert result['identity']['candidate_only_live_ids'] == [['001', 1, 3]]
    assert result['identity']['reused_live_slots'] == [1]
    assert result['exact']['group']['equal'] is True
    assert result['exact']['i_life']['mismatch_count'] == 1
    assert result['exact']['i_elm']['mismatch_count'] == 1
    assert result['exact']['alive']['reference_only_slots'] == 1
    assert result['exact']['t_birth']['mismatch_count'] == 1
    assert result['exact']['t_birth']['equal'] is False
    got.group = '002'
    different_group = compare_cloud(ref, got)
    assert different_group['counts']['matched_live'] == 0
    assert different_group['exact']['group']['equal'] is False
    assert different_group['numeric']['weight']['relative_l2'] is None
    json.dumps(different_group, allow_nan=False)


def test_increments_use_only_persisting_identity_and_report_endpoint_births_losses():
    from tools.native_replay_metrics import compare_cloud

    before = cloud()
    ref, got = cloud((True, True, True)), cloud((True, True, True))
    ref.i_life[:] = got.i_life[:] = [1, 3, 4]
    before.x[0] = [100., 0., 2*np.pi-.125]
    ref.x[0] = [103., 0., .125]
    got.x[0] = [106., 0., .375]
    before.V_prev[0] = [1., 2., 3.]
    ref.V_prev[0] = [2., 4., 6.]
    got.V_prev[0] = [3., 6., 9.]
    ref.P_imp[0, :2] = [.75, .25]
    got.P_imp[0, :2] = [.5, .5]
    ref.weight[0], got.weight[0] = 3., 4.
    got.x[1:] = 1e100  # New identities match each other but have no before increment.
    result = compare_cloud(ref, got, before=before)
    assert result['counts']['matched_live'] == 3
    assert result['increments']['count'] == 1
    numeric = result['increments']['numeric']
    assert numeric['x']['R']['reference_l2'] == 3.
    assert numeric['x']['R']['relative_l2'] == 1.
    assert numeric['x']['phi']['reference_l2'] == pytest.approx(.25)
    assert numeric['x']['phi']['relative_l2'] == pytest.approx(1.)
    assert numeric['V_prev']['phi']['relative_l2'] == 1.
    assert numeric['P_imp']['0']['relative_l2'] == 1.
    assert numeric['weight']['relative_l2'] == 1.
    for side in ('reference', 'candidate'):
        changes = result['identity']['from_before'][side]
        assert changes['birth_ids'] == [['001', 1, 3], ['001', 2, 4]]
        assert changes['loss_ids'] == [['001', 1, 2]]
        assert changes['birth_count'] == 2 and changes['loss_count'] == 1
    assert before.i_life.tolist() == [1, 2, 3]
    assert before.x[0, 0] == 100.
    json.dumps(result, allow_nan=False)


def test_probability_normalization_reports_raw_weighted_charge_and_invalid_rows():
    from tools.native_replay_metrics import compare_cloud

    ref, got = cloud(), cloud()
    ref.P_imp[0, :2] = [.5, .5]
    ref.P_imp[1, :2] = [0., 1.]
    got.P_imp[0, :2] = [-.25, 1.5]  # Sum 1.25: report, do not repair.
    got.P_imp[1, :2] = [0., 1.]
    ref.P_imp[2] = got.P_imp[2] = np.nan
    result = compare_cloud(ref, got)
    a, b = result['normalization']['reference'], result['normalization']['candidate']
    assert a['total_weight'] == b['total_weight'] == 5.
    assert a['weight_times_probability_by_charge']['0'] == 1.
    assert a['weight_times_probability_by_charge']['1'] == 4.
    assert a['probability_row_sum_max_abs_deviation_from_one'] == 0.
    assert b['weight_times_probability_by_charge']['0'] == -.5
    assert b['weight_times_probability_by_charge']['1'] == 6.
    assert b['probability_row_sum_max_abs_deviation_from_one'] == .25
    assert b['negative_probability_count'] == 1
    assert b['weighted_probability_sum'] == 5.5
    assert b['weighted_normalization_signed_error'] == pytest.approx(.1)
    ref.weight[:] = 0.
    zero_weight = compare_cloud(ref, ref)['normalization']['reference']
    assert zero_weight['weighted_normalization_signed_error'] is None
    assert zero_weight['undefined']['weighted_normalization_signed_error'] == 'zero_total_weight'
    json.dumps(result, allow_nan=False)
