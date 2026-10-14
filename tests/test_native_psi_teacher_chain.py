"""Explicit psi intervention through the full event, never a state reset."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tests.test_native_production_events import production_fixture,analytic_equilibrium


def teacher_fixture(times=(1e-9,2e-9,3e-9,4e-9)):
    from src.models.native_psi_teacher import RecordedPsiFeedback,NativePsiTeacherShardModel
    from src.models.native_state import configuration_digest
    model,source,base,state=production_fixture()
    # Real-record reader has this Interface; here records are explicit analytic fixtures.
    records=[SimpleNamespace(psi=state.fields_current[0].copy(),
        equilibrium=analytic_equilibrium(state.fields_current,t),time_si=t,
        geometry_digest=base.dynamics.mesh.configuration['geometry'],units=base.dynamics.mesh.units,
        provenance='analytic psi teacher fixture',source_sha256=configuration_digest({'fixture_time':t})) for t in times]
    feedback=RecordedPsiFeedback(records,mesh=base.dynamics.mesh)
    shard=NativePsiTeacherShardModel(base,feedback)
    model.shard_configuration=deepcopy(shard.configuration)
    state.native_context.configuration['model']=model.checkpoint_configuration
    state.shards=shard.refresh_diagnostics(state.shards,state.fields_current,state.marker.feedback,
        feedback(state.fields_current,state.fluid_time_si),provenance='explicit teacher initialization')
    return model,source,shard,state,feedback


def test_complete_event_psi_teacher_does_not_replace_nine_field_prediction():
    from src.models.native_coupling import advance_coupled_reference
    from src.models.native_state import configuration_digest
    model,source,shard,state,teacher=teacher_fixture()
    # Explicit finite nonzero update to every field, not a trained accuracy claim.
    with torch.no_grad():model.field.decoder.bias.fill_(1e-5)
    original=state.fields_current.copy()
    np.random.seed(1931)
    out,_=advance_coupled_reference(state,model=model,source=source,shard_model=shard,
        equilibrium_provider=teacher)
    assert np.all(np.max(np.abs(out.fields_current-original),axis=(1,2,3))>0)
    assert not np.array_equal(out.fields_current[0],teacher.record(out.fluid_time_si).psi)
    branch=shard.diagnostic_fields(out.fields_current,out.fluid_time_si)
    np.testing.assert_array_equal(branch[0],teacher.record(out.fluid_time_si).psi)
    np.testing.assert_array_equal(branch[1:],out.fields_current[1:])
    assert out.shards['nonlocal_diagnostic'].field_digest==configuration_digest(branch)
    assert out.shards['psi_teacher_event']['mode']=='recorded_psi_diagnostic_intervention'
    np.testing.assert_array_equal(state.fields_current,original)


def test_missing_same_time_psi_rolls_back_event_without_using_future_record():
    from src.models.native_coupling import advance_coupled_reference
    model,source,shard,state,teacher=teacher_fixture(times=(1e-9,3e-9))
    np.random.seed(1931);before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='same-time'):
        advance_coupled_reference(state,model=model,source=source,shard_model=shard,
            equilibrium_provider=teacher)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    assert state.plan_cursor==0 and state.marker.cloud.alive.sum()==0


def test_psi_teacher_two_events_restore_without_replacing_network_state(tmp_path):
    from src.models.native_coupling import advance_coupled_reference
    from src.models.marker_state import save_model_rollout,load_model_rollout
    model,source,shard,state,teacher=teacher_fixture()
    with torch.no_grad():model.field.decoder.bias.fill_(1e-5)
    advance=lambda state,model:advance_coupled_reference(state,model=model,source=source,
        shard_model=shard,equilibrium_provider=teacher)[0]
    np.random.seed(1931)
    first=advance(state,model)
    path=tmp_path/'psi-teacher.pt';save_model_rollout(path,first,model)
    expected=advance(first,model)
    fresh=deepcopy(model);restored=load_model_rollout(path,fresh,native_context=state.native_context)
    actual=advance(restored,fresh)
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    for name in actual.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(actual.marker.cloud,name),getattr(expected.marker.cloud,name))
    assert actual.shards['psi_teacher_event']==expected.shards['psi_teacher_event']
    assert actual.shards['diagnostic_field_binding']==expected.shards['diagnostic_field_binding']


def test_psi_teacher_record_units_and_geometry_cannot_be_silently_rebound():
    from src.models.native_psi_teacher import RecordedPsiFeedback
    _,_,shard,_,teacher=teacher_fixture()
    record=teacher.record(1e-9);record.geometry_digest='other mesh'
    with pytest.raises(ValueError,match='geometry'):
        RecordedPsiFeedback([record],mesh=shard.dynamics.mesh)


def test_torch_teacher_branch_retains_network_bptt():
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tests.test_native_production_torch_integration import control
    model,source,shard,runtime,teacher=teacher_fixture()
    runtime.marker.provenance['fluid_control_origin']='external_control'
    with torch.no_grad():model.field.decoder.weight.fill_(1e-4)
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=teacher)
    initial=adapter.initialize(runtime);initial.values.requires_grad_();initial.memory.requires_grad_()
    np.random.seed(1931)
    first,_=adapter.advance(initial,control());first.values.retain_grad()
    second,_=adapter.advance(first,control(1))
    second.values.square().sum().backward()
    assert first.values.grad is not None and first.values.grad.abs().sum()>0
    assert initial.memory.grad is not None and initial.memory.grad.abs().sum()>0
    assert not np.array_equal(second.runtime.fields_current[0],teacher.record(3e-9).psi)


def test_psi_teacher_provider_must_match_the_shard_intervention_configuration():
    from src.models.native_coupling import advance_coupled_reference
    from src.models.native_psi_teacher import RecordedPsiFeedback
    model,source,shard,state,teacher=teacher_fixture()
    records=[teacher.record(t) for t in (1e-9,2e-9,3e-9,4e-9)]
    records[-1].source_sha256='different_record'
    other=RecordedPsiFeedback(records,mesh=shard.dynamics.mesh)
    with pytest.raises(ValueError,match='teacher.*configuration'):
        advance_coupled_reference(state,model=model,source=source,shard_model=shard,
            equilibrium_provider=other)


@pytest.mark.parametrize('suffix,index',[('',4110),('_injInner',5680)])
def test_real_record_psi_routes_to_diagnostic_branch_not_other_predicted_fields(suffix,index):
    from pathlib import Path
    import h5py
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.models.native_psi_teacher import RecordedPsiFeedback
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_state import configuration_digest
    path=Path('data/EHL-2')/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)/f'jorek{index:06d}.h5'
    if not path.exists():pytest.skip('local phaseA record unavailable')
    record=read_native_psi_teacher(path)
    mesh=NativeSourceMesh(record.coordinates,record.size,record.vertex,record.indices,
        units=record.units,projection='none',max_elements=5000)
    teacher=RecordedPsiFeedback([record],mesh=mesh)
    with h5py.File(path,'r') as f:predicted=f['values'][...]
    predicted[0,0]+=1e-3;predicted[8]*=1.01
    before=predicted.copy();branch=teacher.diagnostic_fields(predicted,record.time_si)
    np.testing.assert_array_equal(predicted,before)
    np.testing.assert_array_equal(branch[0],record.psi)
    np.testing.assert_array_equal(branch[1:],predicted[1:])
    eq=teacher(branch,record.time_si)
    assert eq.field_digest==configuration_digest(branch)
    assert eq.field_digest!=configuration_digest(predicted)
    assert eq.provenance.startswith(teacher.mode)
