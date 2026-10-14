"""Actual event entry fail-closed tests; no fixture or shortened real timestep."""
from pathlib import Path


def test_ordinary_real_anchor_cannot_manufacture_missing_continuous_history(tmp_path):
    from tools.diagnose_native_real_chain import attempt_real_chain
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner')
    result=attempt_real_chain(run,2710,tmp_path/'attempt',wall_seconds=30,memory_mib=8192)
    assert result['full_coupled_events']==0 and result['success'] is False
    assert result['failure_stage']=='initial_event_pairing'
    assert 'continuous' in result['failure_reason'] and 'history' in result['failure_reason']
    assert result['optimizer_updates']==0 and result['cloud_cropped'] is False
    assert (tmp_path/'attempt'/'diagnostic.json').is_file()
    pairing=next(row for row in result['stages'] if row['stage']=='initial_event_pairing' and row['status']=='failed')
    assert pairing['marker_slots']==1000000


def test_registered_poststartup_entry_counts_only_ordinary_cloud_events(tmp_path):
    from tools.diagnose_native_real_chain import attempt_real_chain
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')
    result=attempt_real_chain(run,4110,tmp_path/'poststartup',wall_seconds=3,memory_mib=8192)
    pairing=next(row for row in result['stages'] if row['stage']=='initial_event_pairing' and row['status']=='completed')
    assert pairing['initialization']=='recorded_poststartup'
    assert pairing['initial_field_index']==4111
    assert [row['event_index'] for row in pairing['controls']]==[4112,4113]
    assert all(row['phase']=='ordinary' for row in pairing['controls'])
    assert pairing['field_start_si']==pairing['marker_time_si']
    assert pairing['field_end_si']==pairing['field_time_si']
    assert pairing['projection_max_abs']==0
    assert result['startup_fluid_events']==0
    assert result['ordinary_source_field_events']==result['full_coupled_events']
    assert result['target_ordinary_source_field_events']==2


def test_real_event_entry_builds_full_cloud_and_declares_actual_physics_inputs(tmp_path):
    from tools.diagnose_native_real_chain import attempt_real_chain
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')
    result=attempt_real_chain(run,4110,tmp_path/'real',wall_seconds=45,memory_mib=8192)
    loaded=next(row for row in result['stages'] if row['stage']=='all_slot_cloud_read' and row['status']=='completed')
    assert loaded['marker_slots']==1000000 and loaded['live_markers']==11168
    ready=next(row for row in result['stages'] if row['stage']=='poststartup_ready' and row['status']=='completed')
    assert ready['startup_pending'] is False
    assert ready['diffusion_ju']==5e-6 and ready['ns_radius_min']==.05
    assert ready['projection']=='jorek_inspected_n0_filtered'
    assert ready['field_shape']==[9,4,1,4125]
    assert ready['source_field_static'] is True
    assert ready['actual_first_particle_steps']==40
    assert ready['model_weights_unchanged'] is True
    assert result['qualification_granted'] is False


def test_complete_event_runner_restores_second_event_without_retaining_cloud_trajectory(tmp_path):
    from tools.diagnose_native_real_chain import execute_coupled_events,Stages
    from tools.diagnose_native_production_chain import build_teacher_fixture
    from src.data.base_coupled import StepControl
    bundle=build_teacher_fixture(2)
    controls=(StepControl(1e-9,1e-9,2e-9,'external_control','explicit analytic fixture'),
              StepControl(2e-9,1e-9,3e-9,'external_control','explicit analytic fixture'))
    result=execute_coupled_events(bundle,controls,Stages(),tmp_path,
        rebuild=lambda:build_teacher_fixture(2),scope='analytic_fixture_NOT_real_acceptance',verify_bptt=True)
    assert result['ordinary_source_field_events']==2
    assert result['startup_fluid_events']==0
    assert result['checkpoint']['complete_state_bitwise_equal'] is True
    assert result['checkpoint']['full_rng_equal'] is True
    assert result['checkpoint']['fresh_constructor_weights_differ'] is True
    assert result['checkpoint']['loaded_weights_equal'] is True
    assert result['retains_full_cloud_trajectory'] is False
    assert result['model_weights_unchanged'] is True
    assert result['bptt']['status']=='verified'
    assert all(value>0 for value in result['bptt']['cross_event_gradient_l1'].values())
    assert (tmp_path/'first-event.pt').is_file()
    assert (tmp_path/'events.json').is_file()


def test_event_comparison_observer_receives_both_branches_and_history(tmp_path):
    from tools.diagnose_native_real_chain import execute_coupled_events,Stages
    from tools.diagnose_native_production_chain import build_teacher_fixture
    from src.data.base_coupled import StepControl
    from tools.capture_native_replay_pair import attach_source_evidence
    from src.training.base_coupled import detach_state
    import torch
    import json
    seen=[]
    def observe(phase,state):
        seen.append((phase,state.runtime.fluid_index,len(state.runtime.marker.cloud.weight)))
        torch.save(detach_state(state),tmp_path/f'{phase}.pt')
    controls=(StepControl(1e-9,1e-9,2e-9,'external_control','explicit analytic fixture'),
              StepControl(2e-9,1e-9,3e-9,'external_control','explicit analytic fixture'))
    def build(prefix):
        bundle=build_teacher_fixture(2)
        attach_source_evidence(bundle[1],tmp_path,prefix)
        return bundle
    execute_coupled_events(build('continuous'),controls,Stages(),tmp_path,
        rebuild=lambda:build('restored'),scope='analytic_fixture_NOT_real_acceptance',
        replay_observer=observe)
    assert [r[0] for r in seen]==['before_second','continuous_second','restored_second']
    assert seen[1][1:]==seen[2][1:]
    assert seen[1][1]==seen[0][1]+1
    inputs=[json.loads((tmp_path/f'{p}-source-input.json').read_text()) for p in ('continuous','restored')]
    assert inputs[0]==inputs[1]
    for prefix in ('continuous','restored'):
        assert (tmp_path/f'{prefix}-projection.npz').is_file()
        assert (tmp_path/f'{prefix}-sample-0.npz').is_file()
        assert (tmp_path/f'{prefix}-births.pt').is_file()
    from tools.report_native_replay_pair import report_directory
    report=report_directory(tmp_path)
    assert all(report['source_input_exact'].values())
    assert report['fields']['T_e']['raw_deltas']['relative_l2']==0.
    assert report['projection']['rad_sink']['values']['relative_l2']==0.
    json.dumps(report,allow_nan=False)
