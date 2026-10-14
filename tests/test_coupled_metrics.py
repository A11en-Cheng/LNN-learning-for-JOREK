"""Independent arithmetic oracles: no fitting or implementation-derived truth."""
import numpy as np
import pytest
import torch


def test_pearson_is_not_r_squared_and_undefined_cases_remain_visible():
    from src.training.coupled_metrics import weighted_errors
    out=weighted_errors([2.,4.,6.],[1.,2.,3.])
    assert out['pearson_r']==pytest.approx(1.)
    assert out['relative_l2']==pytest.approx(1.)
    assert out['mae']==pytest.approx(2.)
    assert out['rmse']==pytest.approx(np.sqrt(14/3))
    constant=weighted_errors([2.,2.],[0.,0.])
    assert constant['pearson_r'] is None and constant['relative_l2'] is None
    assert constant['rmse']==2. and constant['relative_status']=='zero_reference'
    assert weighted_errors([],[])['status']=='empty'
    tiny=weighted_errors([2e-20],[1e-20],near_zero_rms=1e-19)
    assert tiny['relative_l2'] is None and tiny['relative_status']=='near_zero_reference'
    assert weighted_errors([.1,.1,.1],[.2,.2,.2],weights=[1.,2.,3.])['pearson_r'] is None


def test_relative_error_does_not_turn_tiny_nonzero_reference_into_zero_by_squaring():
    from src.training.coupled_metrics import weighted_errors
    result=weighted_errors([2e-200,4e-200,6e-200],[1e-200,2e-200,3e-200])
    assert result['relative_l2']==pytest.approx(1.)
    assert result['pearson_r']==pytest.approx(1.)
    assert result['relative_status']=='defined'


def test_nine_field_loss_has_independent_value_gradient_and_degree_oracle():
    from src.training.coupled_metrics import native_supervision
    from tests.test_native_gauss import rectangle_geometry,native_coefficients
    from src.models.native_features import TorchNativeGauss
    from tests.test_native_coupling import native_fixture
    units=native_fixture()[0].units
    g=rectangle_geometry()
    ev=TorchNativeGauss(g.coordinates,g.size[:,:,None],np.arange(4)[:,None],units=units)
    # First field is R=2+s: gradient exactly [1,0]. No error in other 8 fields.
    p=torch.zeros((9,4,1,4),dtype=torch.float64)
    p[0,:,0]=torch.tensor(native_coefficients(lambda s,t:2+s,lambda s,t:1.,lambda s,t:0.,lambda s,t:0.))
    p.requires_grad_()
    truth=torch.zeros_like(p)
    pred=dict(values=p,raw_deltas=truth.clone(),units=units,time_si=1.)
    target=dict(values=truth,raw_deltas=truth.clone(),units=units,time_si=1.)
    scales=dict(coefficients=torch.ones((9,4,1,1)),raw_history=torch.ones((9,4,1,1)),
        gauss_values=torch.ones(9),compatible_gradients=torch.ones((9,2)),provenance='fixture_physical_native_units')
    weights=dict(coefficients=0.,raw_history=0.,gauss_values=1.,compatible_gradients=2.)
    loss=native_supervision(pred,target,evaluator=ev,scales=scales,weights=weights)
    # Axisymmetric volume R dR dZ: integral R^3 / integral R = 6.5.
    # Gradient MSE averages [1,0] -> .5. Then average nine fields.
    assert loss.report()['unweighted']['gauss_values'][0]==pytest.approx(6.5)
    assert loss.report()['unweighted']['compatible_gradients'][0]==pytest.approx([1.,0.])
    assert float(loss.total.detach())==pytest.approx(7.5/9)
    assert sum(loss.report()['per_field_total_contribution'])==pytest.approx(7.5/9)
    loss.total.backward()
    assert p.grad is not None and p.grad[0].abs().sum()>0
    torch.testing.assert_close(p.detach()[0,0,0],torch.tensor([2.,3.,3.,2.],dtype=p.dtype))


def test_full_report_preserves_channels_shard_id_mismatches_and_group_keys():
    from src.training.coupled_metrics import coupled_node_metrics,MetricGroups
    from tests.test_native_joint_source import joint_fixture
    model,mesh,_,_,u,d,_=joint_fixture()
    projection=np.ones((5,4,1,4))
    shard={'ids':['a','b'],'positions_m':np.array([[0.,0.,0.],[1.,0.,0.]]),'alive':[True,True],
           'initial_positions_m':np.zeros((2,3))}
    predicted=dict(field=dict(values=u,raw_deltas=d,units=mesh.units,time_si=1.),
        marker_projection=projection,marker_time_si=1.,shard_density=projection[:2],shard_density_time_si=1.,shard_positions=shard,shard_time_si=1.)
    from copy import deepcopy
    target=deepcopy(predicted);target['field']['values']=u.detach().clone()
    target['shard_positions']={'ids':['a','c'],'positions_m':np.array([[1.,0.,0.],[2.,0.,0.]]),
        'alive':[False,True],'initial_positions_m':np.zeros((2,3))}
    r=coupled_node_metrics(predicted,target,evaluator=model.feature_path.evaluator,
        normalization_source='native_units_no_metric_scaling',region_name='all_elements')
    assert len(r['marker_projection'])==5 and len(r['fields']['coefficients'])==9
    assert len(r['fields']['coefficients']['T_e'])==4
    assert r['shard_positions']['unmatched_prediction']==['b']
    assert r['shard_positions']['unmatched_target']==['c']
    assert r['shard_positions']['liveness_mismatch']==['a']
    assert r['shard_positions']['matched_position_rmse_m']==pytest.approx(1.)
    groups=MetricGroups()
    key=dict(run='fixture',task='self_generated',burnin_time_si=1.,burnin_record_count=2,
        prediction_time_si=2.,dt_si=.5,dt_ratio=2.,teacher_policy='none')
    groups.add(key,r);groups.add(key,r)
    out=groups.report()
    assert out[0]['node_count']==2 and out[0]['group']==key
    assert out[0]['time_aggregation']=='equal_record_nodes'


def test_shard_trajectory_correlation_uses_fixed_id_actual_times_not_pooled_windows():
    from src.training.coupled_metrics import MetricGroups,shard_position_metrics
    groups=MetricGroups()
    for t,x in [(1.,0.),(2.,1.),(4.,2.)]:
        p=dict(ids=['stable-shard'],positions_m=[[2*x,0.,0.]],alive=[True])
        truth=dict(ids=['stable-shard'],positions_m=[[x,0.,0.]],alive=[True])
        metrics=dict(time_si=t,shard_positions=shard_position_metrics(p,truth))
        group=dict(run='run',task='self_generated',burnin_time_si=0.,burnin_record_count=0,
            prediction_time_si=t,dt_si=.5,dt_ratio=1.,teacher_policy='none')
        groups.add(group,metrics,trajectory_id='one_continuous_rollout')
    report=groups.trajectory_report()[0]
    track=report['shards']['stable-shard']
    assert track['times_si']==[1.,2.,4.]
    assert track['coordinates']['x_m']['pearson_r']==pytest.approx(1.)
    assert track['coordinates']['x_m']['relative_l2']==pytest.approx(1.)
    assert track['coordinates']['y_m']['pearson_r'] is None


def test_actual_metric_entry_rejects_shard_clock_mismatch_and_marks_missing_fields():
    from src.training.coupled_metrics import coupled_node_metrics
    from tests.test_native_joint_source import joint_fixture
    model,mesh,_,_,_,_,_=joint_fixture()
    shard=dict(ids=['a'],positions_m=[[1.,0.,0.]],alive=[True])
    kw=dict(evaluator=model.feature_path.evaluator,normalization_source='explicit',region_name='all')
    with pytest.raises(ValueError,match='shard.*time'):
        coupled_node_metrics(dict(shard_positions=shard,shard_time_si=1.),
            dict(shard_positions=shard,shard_time_si=2.),**kw)
    out=coupled_node_metrics({}, {},**kw)
    assert out['fields']['status']=='missing'
