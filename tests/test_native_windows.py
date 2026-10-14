import numpy as np
import pytest
from src.models.native_state import NativeFieldState, NativeUnits


def sequence(n=12):
    from src.data.native_windows import FieldRecord
    units=NativeUnits(1.,1e20,1.,5/3)
    times=np.cumsum([0.]+[.5 if i%2 else 1. for i in range(1,n)])
    return [FieldRecord('run',NativeFieldState(np.full((9,4,1,2),float(i)),
        np.full((9,4,1,2),100.+i),units,float(t),(0,),i),
        previous_dt_si=.5 if i%2 else 1.,compatibility='mesh-v1',provenance=f'frame:{i}')
        for i,t in enumerate(times)]


def test_split_before_window_and_causal_input():
    from src.data.native_windows import SplitBlock, WindowConfig, build_windows
    records=sequence()
    result=build_windows(records,[SplitBlock('run','train',0,5),SplitBlock('run','test',7,11)],
        WindowConfig(warmup=1,horizon=2,stride=1,isolation=1),
        time_plan={('run',i):(.5 if (i+1)%2 else 1.) for i in range(11)},
        plan_provenance='declared_run_control',scheme='Gears')
    assert [w.prediction_index for w in result.windows]==[1,2,3,8,9]
    first=result.windows[0]
    assert first.support_indices==(0,1,2,3)
    view=first.current_input()
    assert view.state.values[8,3,0,1]==1
    assert view.state.raw_deltas[8,3,0,1]==101
    assert view.dt_si==1. and view.previous_dt_si==.5
    assert not hasattr(view,'targets')
    first.targets[0].values.fill(999.)
    np.testing.assert_array_equal(first.current_input().state.values,view.state.values)
    assert first.loss_mask.tolist()==[False,True,True]
    assert first.qualification=='field_supervised'
    assert first.initialization['coupled_gap']


def test_scaling_train_only_and_all_degree_roundtrip(tmp_path):
    from src.data.native_windows import NativeScaler
    train=sequence(3)
    scaler=NativeScaler.fit({'train':train,'validation':sequence(5)})
    changed=sequence(5)
    changed[0].state.values.fill(1e15)
    other=NativeScaler.fit({'train':train,'validation':changed})
    np.testing.assert_array_equal(scaler.value_scale,other.value_scale)
    assert scaler.value_center.shape==(9,4,1,1)
    state=train[2].state
    u,d=scaler.encode(state)
    ru,rd=scaler.decode(u,d)
    np.testing.assert_allclose(ru,state.values,rtol=0,atol=1e-14)
    np.testing.assert_allclose(rd,state.raw_deltas,rtol=3e-16,atol=1e-14)
    scaler.save(tmp_path/'scale.json')
    loaded=NativeScaler.load(tmp_path/'scale.json')
    np.testing.assert_array_equal(loaded.value_scale,scaler.value_scale)
    with pytest.raises(ValueError,match='training'): NativeScaler.fit({'test':train})


def test_parameter_routes_fail_unknown_and_preserve_defaults():
    from src.data.native_parameters import route_parameters, parameter_variation
    payload={'names':['eta','n_spi','time_evol_zeta'], 'vector':np.array([.01,3.,.5]),
             'provenance':{'eta':'h5','n_spi':'default','time_evol_zeta':'derived'}}
    rows=route_parameters(payload,source='checkpoint:1',namelist_source='segment:cfg')
    assert rows[0]['group']=='physical_coefficient'
    assert rows[1]['status']=='unconfirmed_default'
    assert rows[1]['available_at']=='unavailable_until_confirmed'
    assert rows[2]['consumer']=='time_contract_recompute_from_dt_pair'
    assert rows[2]['status']=='audit_only'
    with pytest.raises(ValueError,match='provenance'):
        route_parameters({**payload,'provenance':{'eta':'unknown'}},source='x',namelist_source='y')
    with pytest.raises(ValueError,match='registered'):
        route_parameters({'names':['run_id'],'vector':[1.],'provenance':{'run_id':'h5'}},source='x',namelist_source='y')
    audit=parameter_variation({'train':[rows,rows]})
    assert 'eta' in audit['train']['constant_unidentifiable']


def test_real_two_phase_a_fragments_and_full_history():
    import h5py
    from pathlib import Path
    from src.data.native_window_io import read_segment
    from src.data.native_windows import build_windows,SplitBlock,WindowConfig
    root=Path('data/EHL-2')
    for suffix,ids,events,h in [('',range(4111,4115),(4111,),.05),('_injInner',range(2704,2708),(2704,),5.)]:
        run=root/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        records,audit=read_segment(run,ids,restart_events=events,gamma=5/3,
            gamma_provenance='explicit_test_configuration',modes=(0,))
        units=records[0].state.units
        result=build_windows(records,[SplitBlock(run.name,'train',ids.start,ids.stop-1)],
            WindowConfig(1,2,1,0),time_plan={(run.name,i):h*units.t_norm_si for i in ids},
            plan_provenance='declared_segment_schedule',scheme='Gears')
        assert len(result.windows)==1
        assert result.windows[0].qualification=='field_supervised'
        with h5py.File(run/f'jorek{ids.start:06d}.h5') as f:
            assert records[0].state.values.tobytes()==f['values'][()].tobytes()
            assert records[0].state.raw_deltas.tobytes()==f['deltas'][()].tobytes()
        assert all(not r['projection_history_present'] for r in audit)
        assert result.windows[0].current_input().state.values.shape==(9,4,1,4125)


def make_windows(records,**kwargs):
    from src.data.native_windows import build_windows,SplitBlock,WindowConfig
    config=kwargs.pop('config',WindowConfig(1,2,1,0))
    blocks=kwargs.pop('blocks',[SplitBlock('run','train',0,len(records)-1)])
    return build_windows(records,blocks,config,
        time_plan={('run',i):(.5 if (i+1)%2 else 1.) for i in range(len(records)-1)},
        plan_provenance='declared_run_control',scheme='Gears',**kwargs)


def test_restart_and_support_isolation():
    from src.data.native_windows import SplitBlock,WindowConfig
    records=sequence(8); records[3].restart=True
    result=make_windows(records)
    assert [w.prediction_index for w in result.windows]==[4,5]
    assert result.exclusions=={'restart_crossing':3}
    with pytest.raises(ValueError,match='isolation'):
        make_windows(records,blocks=[SplitBlock('run','train',0,4),SplitBlock('run','test',4,7)])
    result=make_windows(sequence(12),config=WindowConfig(1,2,1,1),
        blocks=[SplitBlock('run','train',0,5),SplitBlock('run','test',7,11)])
    train={i for w in result.windows if w.split=='train' for i in w.support_indices}
    test={i for w in result.windows if w.split=='test' for i in w.support_indices}
    assert not train.intersection(test)


def test_malformed_state_time_and_incomplete_initialization():
    from src.data.native_windows import WindowConfig
    records=sequence(5); original=records[1].state.values.copy()
    records[2].state.raw_deltas=None
    with pytest.raises(ValueError,match='history'): make_windows(records)
    np.testing.assert_array_equal(records[1].state.values,original)
    records=sequence(5); records[2].state.time_si+=.01
    with pytest.raises(ValueError,match='time pairing'): make_windows(records)
    with pytest.raises(ValueError,match='replay'): WindowConfig(1,2,1,0,replay_steps=1)
    with pytest.raises(ValueError,match='feedback'): make_windows(sequence(5),config=WindowConfig(1,2,1,0,source_teacher_policy='recorded_full_history'))


def test_same_numbered_feedback_and_separate_gather_time():
    from src.models.native_state import NativeFeedbackState
    from src.data.native_windows import WindowConfig
    records=sequence(5)
    for r in records:
        t=r.state.time_si
        r.feedback=NativeFeedbackState(np.full((5,4,1,2),float(r.state.step_index)),
            np.full((5,4,1,2),.125),r.state.units,t,(0,),f'saved_history:{r.state.step_index}',.03)
        r.source_interval_si=(t-.2,t)
    result=make_windows(records,config=WindowConfig(1,2,1,0,source_teacher_policy='recorded_full_history'))
    inp=result.windows[0].current_input()
    assert inp.feedback.values[4,3,0,1]==1.  # S_1, not future S_2
    assert inp.feedback.raw_deltas[4,3,0,1]==.125
    assert inp.feedback.gather_time_si==.03 and inp.dt_si==1.
    assert inp.source_interval_si==(.3,.5)
    records[1].feedback.time_si=1.5
    with pytest.raises(ValueError,match='pairing'): make_windows(records,config=WindowConfig(1,2,1,0,source_teacher_policy='recorded_full_history'))


def test_parameter_audit_does_not_enter_input_and_physical_changes_split():
    records=sequence(6)
    rows=({'name':'eta','group':'physical_coefficient','unit':'native','value':1.,
        'status':'unconfirmed_default','available_at':'unavailable_until_confirmed','provenance':{'kind':'default'}},)
    for r in records: r.parameters=rows
    result=make_windows(records)
    assert result.windows[0].current_input().parameters==()
    records[2].parameters=({**rows[0],'value':2.},)
    assert make_windows(records).exclusions['parameter_change']>0


def test_audit_command_writes_reproducible_manifest_and_train_statistics(tmp_path):
    from src.data.native_window_io import audit_spec
    import json
    spec=json.loads(__import__('pathlib').Path('config/native-window-phaseA-audit.json').read_text())
    report=audit_spec(spec,tmp_path)
    assert report['window_counts']=={'train':2,'validation':2,'test':1}
    assert report['coupled_rollout_windows']==0
    assert (tmp_path/'manifest.json').exists() and (tmp_path/'normalizer.json').exists()
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['schema']=='native-causal-window-v2'
    for w in manifest['windows']:
        assert len(w['support'])==4
        assert len(w['support_time_plan_si'])==3
        assert len(w['valid_target_mask'])==2
    with pytest.raises(ValueError,match='unrecognized'):
        audit_spec({**spec,'old_manifest':'unsafe'},tmp_path)


def test_scaler_binds_units_and_full_harmonics():
    from dataclasses import replace
    from src.data.native_windows import NativeScaler
    rs=sequence(3)
    for r in rs:
        r.state=replace(r.state,values=np.repeat(r.state.values,3,axis=2),
            raw_deltas=np.repeat(r.state.raw_deltas,3,axis=2),modes=(0,1,1))
        r.state.values[:,1,1,:]+=4.
        r.state.raw_deltas[:,3,2,:]+=.25
    scaler=NativeScaler.fit({'train':rs})
    u,d=scaler.encode(rs[1].state)
    a,b=scaler.decode(u,d)
    np.testing.assert_allclose(a,rs[1].state.values,rtol=3e-16,atol=1e-14)
    np.testing.assert_allclose(b,rs[1].state.raw_deltas,rtol=3e-16,atol=1e-14)
    other=replace(rs[1].state,units=NativeUnits(2.,1e20,1.,5/3))
    with pytest.raises(ValueError,match='units'): scaler.encode(other)


def test_unknown_parameter_provenance_and_incompatible_state_fail_explicitly():
    rs=sequence(5)
    rs[1].parameters=({'name':'eta','group':'physical_coefficient','value':1.,'status':'declared',
        'unit':'native','available_at':'initialization','provenance':{'kind':'unknown'}},)
    with pytest.raises(ValueError,match='provenance'): make_windows(rs)
    rs=sequence(5); rs[2].compatibility='other_mesh'
    assert make_windows(rs).exclusions=={'incompatible_state':2}


def test_pellet_density_retains_native_scale_not_si_label():
    from src.data.native_parameters import route_parameters
    row=route_parameters({'names':['pellet_density'],'vector':[4.3e8],
        'provenance':{'pellet_density':'namelist'}},source='h5',namelist_source='production')[0]
    assert row['unit']=='10^20 m^-3'
    assert row['value']==4.3e8


def test_parameter_consumers_are_named_and_legacy_thermal_unit_is_not_trusted():
    from src.data.native_parameters import route_parameters
    payload={'names':['eta','ZK_par_max','n_spi'], 'vector':[.01,1e4,846.],
        'provenance':{'eta':'h5','ZK_par_max':'namelist','n_spi':'namelist'}}
    rows=route_parameters(payload,source='field',namelist_source='production')
    assert rows[0]['consumer']=='magnetic_diffusion_coefficient_pending'
    assert rows[1]['unit'] is None and rows[1]['status']=='unresolved_unit'
    assert rows[2]['consumer']=='pellet_count_condition_encoder_pending'


def test_teacher_feedback_requires_same_native_units_and_axes():
    from src.models.native_state import NativeFeedbackState
    from src.data.native_windows import WindowConfig
    rs=sequence(5)
    for r in rs:
        t=r.state.time_si
        r.feedback=NativeFeedbackState(np.ones((5,4,1,3)),np.ones((5,4,1,3)),
            NativeUnits(2.,1e20,1.,5/3),t,(0,),'saved-history',.1)
        r.source_interval_si=(t-.1,t)
    with pytest.raises(ValueError,match='feedback.*units/axes'):
        make_windows(rs,config=WindowConfig(1,2,1,0,source_teacher_policy='recorded_full_history'))


def test_window_preserves_each_axis_and_uses_declared_plan_only():
    from src.data.native_windows import build_windows,SplitBlock,WindowConfig
    rs=sequence(5)
    exact=np.arange(72,dtype=np.float64).reshape(9,4,1,2)-35.
    rs[1].state.values=exact.copy(); rs[1].state.raw_deltas=exact.copy()/8.
    w=make_windows(rs).windows[0]
    assert w.current_input().state.values.tobytes()==exact.tobytes()
    assert w.current_input().state.raw_deltas.tobytes()==(exact/8.).tobytes()
    kwargs=dict(records=rs,blocks=[SplitBlock('run','train',0,4)],config=WindowConfig(1,2,1,0),scheme='Gears')
    with pytest.raises(ValueError,match='declared time plan'):
        build_windows(**kwargs,time_plan={},plan_provenance='declared_run_control')
    with pytest.raises(ValueError,match='inference-available'):
        build_windows(**kwargs,time_plan={},plan_provenance='future_target_differences')
    with pytest.raises(ValueError,match='coupled initialization'):
        WindowConfig(1,2,1,0,required_qualification='coupled_rollout')


def test_collinear_parameters_are_marked_unidentifiable():
    from src.data.native_parameters import parameter_variation
    rows=[[{'name':'a','value':1.},{'name':'b','value':3.}],
          [{'name':'a','value':2.},{'name':'b','value':5.}],
          [{'name':'a','value':3.},{'name':'b','value':7.}]]
    audit=parameter_variation({'train':rows})['train']
    assert audit['affine_collinear']==[['a','b',1.]]
