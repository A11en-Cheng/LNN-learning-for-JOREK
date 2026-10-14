"""Full event/checkpoint seam diagnostics, without training or accuracy claims."""
import json
import numpy as np
import pytest


def test_three_events_emit_nonzero_bounded_nine_field_predictions_and_persistent_source(tmp_path):
    from tools.diagnose_native_production_chain import execute_chain
    result=execute_chain(tmp_path/'chain',events=3)
    assert result['scope']=='analytic_fixture_production_mechanisms_not_real_data'
    assert result['optimizer_updates']==0 and result['events_completed']==3
    assert result['production_closed_loop_certified'] is False
    assert result['zero_decoder'] is False
    assert [row['live_markers'] for row in result['events']]==[1,2,3]
    for row in result['events']:
        assert len(row['field_delta_max_abs'])==9
        assert all(0.<value<1e-4 for value in row['field_delta_max_abs'])
        assert row['nonlocal_time_si']==row['field_time_si']
        assert row['source_volume_time_si']==row['field_time_si']
        assert row['marker_time_si']<row['field_time_si']
        assert row['radius_m'][0]<.003
        assert row['ablation_atoms_s'][0]>0
        assert row['new_births']==1
    assert result['events'][-1]['radius_m'][0]<result['events'][0]['radius_m'][0]
    assert result['events'][-1]['positions_RZphi'][0][0]>2.5
    disk=json.loads((tmp_path/'chain'/'diagnostic.json').read_text())
    assert disk==result
    states=np.load(tmp_path/'chain'/'trajectory.npz')
    assert states['fields'].shape==(4,9,4,1,4)
    for index,row in enumerate(result['events'],1):
        expected=np.max(np.abs(states['fields'][index]-states['fields'][index-1]),axis=(1,2,3))
        np.testing.assert_array_equal(row['field_delta_max_abs'],expected)


def test_newly_constructed_model_resumes_complete_state_and_all_cpu_rng(tmp_path):
    from tools.diagnose_native_production_chain import execute_chain
    result=execute_chain(tmp_path/'restore',events=3)
    verification=result['checkpoint']
    assert verification['fresh_constructor_weights_differ'] is True
    assert verification['loaded_weights_equal'] is True
    assert verification['complete_state_bitwise_equal'] is True
    assert verification['python_numpy_torch_cpu_rng_equal'] is True
    assert verification['direct_state_digest']==verification['resumed_state_digest']
    assert (tmp_path/'restore'/'session.pt').is_file()


def test_nonzero_nine_field_prediction_keeps_cross_event_bptt_without_parameter_updates(tmp_path):
    from tools.diagnose_native_production_chain import execute_chain
    result=execute_chain(tmp_path/'bptt',events=2)
    assert result['model_weights_unchanged'] is True
    gradient=result['bptt']
    assert gradient['loss_input']=='last_committed_nine_field_values_only_no_labels'
    for name in ('initial_values','initial_raw_deltas','initial_memory','first_values','first_raw_deltas',
                 'first_memory','temporal_parameters','decoder_parameters'):
        assert np.isfinite(gradient[name]) and gradient[name]>0
    artifact=np.load(tmp_path/'bptt'/'bptt.npz')
    for name in ('initial_values','initial_raw_deltas','initial_memory','first_values','first_raw_deltas','first_memory'):
        assert np.isfinite(artifact[name]).all() and np.abs(artifact[name]).sum()>0


def test_psi_teacher_intervention_is_explicit_and_never_resets_committed_nine_fields(tmp_path):
    from tools.diagnose_native_production_chain import execute_chain,build_teacher_fixture
    result=execute_chain(tmp_path/'teacher',events=3,builder=build_teacher_fixture,
        scope='analytic_psi_teacher_fixture_not_real_data_not_autonomous')
    assert result['production_closed_loop_certified'] is False
    assert result['checkpoint']['complete_state_bitwise_equal'] is True
    for row in result['events']:
        intervention=row['branch_intervention']
        assert intervention['mode']=='recorded_psi_diagnostic_intervention'
        assert intervention['autonomous'] is False
        assert intervention['record_time_si']==row['field_time_si']
        binding=row['diagnostic_field_binding']
        assert binding['predicted_field_digest']==row['field_digest']
        assert binding['diagnostic_field_digest']==row['nonlocal_field_digest']
        assert binding['predicted_field_digest']!=binding['diagnostic_field_digest']
        assert all(value>0 for value in row['field_delta_max_abs'])
    states=np.load(tmp_path/'teacher'/'trajectory.npz')
    # Analytic teacher psi is the initial psi, but every committed psi changes.
    assert not np.array_equal(states['fields'][-1,0],states['fields'][0,0])


def test_zero_decoder_cannot_be_reported_as_nonzero_nine_field_execution(tmp_path):
    import torch
    from tools.diagnose_native_production_chain import execute_chain,build_fixture
    def zero_builder(events):
        built=build_fixture(events)
        with torch.no_grad():
            built[0].field.decoder.weight.zero_();built[0].field.decoder.bias.zero_()
        return built
    with pytest.raises(ValueError,match='nonzero nine-field'):
        execute_chain(tmp_path/'zero',events=2,builder=zero_builder)
    assert not (tmp_path/'zero'/'diagnostic.json').exists()
