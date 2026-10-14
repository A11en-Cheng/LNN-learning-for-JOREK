"""Conditional dual-source windows do not certify a self-generated marker loop."""
import numpy as np
from tests.test_native_windows import sequence


def test_given_legal_source_window_preserves_both_sources_and_rejects_missing_history():
    import pytest
    from src.data.native_windows import FieldRecord,SplitBlock,WindowConfig,build_windows,WindowSources
    from src.models.native_state import NativeFeedbackState
    records=sequence(4)
    for r in records:r.state.time_si+=1.
    for r in records[:-1]:
        p=np.full((5,4,1,2),2.);d=np.full_like(p,.1)
        feedback=NativeFeedbackState(p,d,r.state.units,r.state.time_si,(0,),
            history_provenance='explicit_saved_projection_history',gather_time_si=.25)
        r.sources=WindowSources(feedback,np.full((2,4,1,2),.3),
            'model502_mass_density_per_JU',{'status':'saved','source':'fixture_density',
                'support_indices':[r.state.step_index]},r.compatibility,
            (r.state.time_si-.5,r.state.time_si),{'status':'saved','source':'fixture_full_projection_history','values_source':'fixture_full_projection_values',
                'support_indices':[r.state.step_index]})
    kw=dict(time_plan={('run',i):(.5 if (i+1)%2 else 1.) for i in range(3)},
        plan_provenance='declared_run_control',scheme='Gears')
    config=WindowConfig(0,2,1,0,source_teacher_policy='recorded_full_history',required_qualification='conditional_source')
    result=build_windows(records,[SplitBlock('run','train',0,3)],config,**kw)
    assert result.windows[0].qualification=='conditional_source'
    view=result.windows[0].current_input()
    np.testing.assert_array_equal(view.sources.shard_density_rate,records[0].sources.shard_density_rate)
    result.windows[0].targets[0].values.fill(1000)
    assert view.state.values[0,0,0,0]==0
    assert not result.windows[0].eligibility['self_generated_source']['eligible']
    records[0].sources.density_provenance['support_indices']=[-1,0]
    outside=build_windows(records,[SplitBlock('run','train',0,3)],config,**kw)
    assert outside.exclusions['source_history_outside_window_support']==1
    assert outside.rejections[0]['support_indices']==[0,1,2]
    records[0].sources.density_provenance['support_indices']=[0]
    records[0].sources.projection_history_provenance['status']='verified_reconstructed'
    with pytest.raises(ValueError,match='certification'):
        build_windows(records,[SplitBlock('run','train',0,3)],config,**kw)
    records[0].sources.projection_history_provenance['status']='saved'
    records[0].sources.feedback.raw_deltas=None
    with pytest.raises(ValueError,match='history'):build_windows(records,[SplitBlock('run','train',0,3)],config,**kw)


def test_self_generated_window_requires_validated_full_runtime_not_a_marker_path():
    from copy import deepcopy
    from src.data.native_windows import RecoveredCoupledAnchor,FieldRecord,SplitBlock,WindowConfig,build_windows
    from src.models.native_state import NativeFieldState
    from src.models.native_coupling import SmallShardModel
    from src.models.marker_state import ReferenceRollout,MarkerCloud
    from tests.test_native_joint_source import joint_fixture
    from tests.test_native_context_guard import coupled_context
    model,mesh,source,marker,u,raw,kw=joint_fixture()
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    model.shard_configuration=deepcopy(shard.configuration)
    context=coupled_context(model,mesh,kw['feature_parameters'])
    context.configuration['numerics']['scheme']='Gears'
    marker.cloud=MarkerCloud.empty(10,8)
    marker.provenance['fluid_control_origin']='external_control'
    marker.provenance['confirmed_empty']=True
    state=ReferenceRollout(marker,u.detach().numpy().copy(),u.detach().numpy().copy(),
        dict(positions=np.array([[2.6,.6,0.]]),velocities=np.zeros((1,3)),radius=np.array([1e-5]),
             ablation_atoms_s=np.array([1e12])),1e-9,1,(1e-9,1e-9),
        fluid_dt_previous_si=1e-9,fluid_deltas=raw.detach().numpy().copy(),time_evol_scheme='Gears',
        native_context=context,field_memory=kw['memory'].detach().numpy().copy())
    statuses={k:'saved' for k in ('field','fluid_history','previous_field','marker','feedback_history','clock','shard','memory','control','configuration')}
    anchor=RecoveredCoupledAnchor(state,statuses,'explicit_fixture_full_runtime',(1,))
    records=[FieldRecord('fixture',NativeFieldState(state.fields_current.copy(),state.fluid_deltas.copy(),
        mesh.units,i*1e-9,(0,),i),1e-9,'mesh-fixture',f'fixture:{i}') for i in range(1,4)]
    records[0].coupled_anchor=anchor
    config=WindowConfig(0,2,1,0,field_teacher_policy='autoregressive',
        source_teacher_policy='self_generated_reference',required_qualification='self_generated_source')
    kwargs=dict(time_plan={('fixture',i):1e-9 for i in (1,2)},plan_provenance='declared_run_control',scheme='Gears')
    collection=build_windows(records,[SplitBlock('fixture','train',1,3)],config,**kwargs)
    assert len(collection.windows)==1
    assert collection.windows[0].eligibility['self_generated_source']['eligible']
    from src.models.native_coupling import advance_coupled_reference
    initial=collection.windows[0].initial_coupled_state()
    first,_=advance_coupled_reference(initial,model=model,source=source,shard_model=shard)
    second,_=advance_coupled_reference(first,model=model,source=source,shard_model=shard)
    assert second.fluid_index==3 and not np.array_equal(first.fields_current,initial.fields_current)
    loaded=collection.windows[0].initial_coupled_state()
    loaded.fields_current.fill(99)
    assert not np.array_equal(loaded.fields_current,state.fields_current)
    anchor.state_status['clock']='unavailable'
    rejected=build_windows(records,[SplitBlock('fixture','train',1,3)],config,**kwargs)
    assert not rejected.windows and any('clock' in key for key in rejected.exclusions)
