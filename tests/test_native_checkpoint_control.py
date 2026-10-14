"""Recorded imported dt is history, never an implicit next-step controller."""
import pytest
from tests.test_marker_state import native_fixture
from src.data.marker_checkpoint import initialize_native_checkpoint,reference_rollout_from_checkpoint


def test_checkpoint_does_not_require_truncated_namelist_arrays(tmp_path):
    kw=native_fixture(tmp_path)
    kw['namelist_path'].write_text('tstep_particles=1d-9\n')
    cp=initialize_native_checkpoint(**kw)
    assert cp.provenance['tstep_h5']==.5
    with pytest.raises(ValueError,match='control'):reference_rollout_from_checkpoint(cp)
    state=reference_rollout_from_checkpoint(cp,control_plan_si=(cp.t_norm*.2,cp.t_norm*.7))
    assert state.step_plan_si==(cp.t_norm*.2,cp.t_norm*.7)


def test_replay_schedule_requires_explicit_origin_and_preserves_import_history(tmp_path):
    cp=initialize_native_checkpoint(**native_fixture(tmp_path))
    with pytest.raises(ValueError,match='origin'):
        reference_rollout_from_checkpoint(cp,control_plan_si=(1e-9,),control_origin='future_labels')
    state=reference_rollout_from_checkpoint(cp,control_plan_si=(1e-9,2e-9),control_origin='recorded_schedule_replay')
    assert state.marker.provenance['fluid_control_origin']=='recorded_schedule_replay'
    assert state.marker.provenance['tstep_h5']==.5
