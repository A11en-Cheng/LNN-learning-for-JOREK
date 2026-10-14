"""Budgeted field-only training mechanism tests, independent of performance claims."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from tests.test_native_temporal import gp_temporal_fixture


def test_supervision_has_independent_constant_field_and_raw_history_oracle():
    from src.training.native_curriculum import field_supervision,LossWeights
    from src.models.native_torch_rollout import initialize_field_state
    model,p,w=gp_temporal_fixture()
    with torch.no_grad():
        model.state_scale.fill_(2.);model.increment_scale.fill_(3.)
        model.feature_path.value_scale.fill_(2.);model.feature_path.gradient_scale.fill_(1.)
    target=replace(w.targets[0],values=np.zeros((9,4,1,4)),raw_deltas=np.zeros((9,4,1,4)))
    state=initialize_field_state(model,target,previous_dt_si=.5,scheme='Gears')
    state.values[:,0]=2.;state.values.requires_grad_()
    state.raw_deltas.fill_(3.);state.raw_deltas.requires_grad_()
    loss=field_supervision(model,state,target,LossWeights(1.,.25,.25,.1))
    torch.testing.assert_close(loss.total,torch.tensor(.75,dtype=torch.float64),rtol=0,atol=2e-15)
    assert loss.metrics()['unweighted']['coefficients']==pytest.approx(.25)
    assert loss.metrics()['unweighted']['raw_history']==pytest.approx(1.)
    assert loss.metrics()['unweighted']['gauss_values']==pytest.approx(1.)
    loss.total.backward()
    assert state.values.grad.abs().sum()>0 and state.raw_deltas.grad.abs().sum()>0
    bad=deepcopy(target);object.__setattr__(bad,"raw_deltas",None)
    with pytest.raises(ValueError): field_supervision(model,state,bad,LossWeights(1.,1.,1.,1.))


def small_banks():
    from tests.test_native_windows import sequence
    from src.data.native_windows import build_windows,SplitBlock,WindowConfig
    from tests.test_native_features import parameters,UNITS
    rs=sequence(9)
    for r in rs:
        r.state=replace(r.state,units=UNITS,values=np.arange(144,dtype=float).reshape(9,4,1,4)/144+r.state.step_index/20,
            raw_deltas=np.full((9,4,1,4),.125))
        r.parameters=parameters()
    return {k:build_windows(rs,[SplitBlock('run','train',0,8)],WindowConfig(1,k,1,0),
        time_plan={('run',i):(.5 if (i+1)%2 else 1.) for i in range(9)},
        plan_provenance='declared_run_control',scheme='Gears').windows for k in (1,2,4)}


def trainer_fixture(seed=19):
    from src.training.native_curriculum import NativeCurriculumTrainer,LossWeights,CurriculumStage
    model,_,_=gp_temporal_fixture()
    return NativeCurriculumTrainer(model,small_banks(),weights=LossWeights(1.,.25,.25,.1),
        stages=(CurriculumStage('smoke',1,1,1.),CurriculumStage('mixed',1,2,.5),CurriculumStage('AR',1,4,0.)),
        seed=seed,learning_rate=.001,lr_decay=.99,max_grad_norm=10.,
        experiment_configuration={'scope':'field_only_no_feedback_consumer','dataset':'small_declared_fixture'})


def test_optimizer_updates_and_course_keeps_source_policy_none():
    torch.manual_seed(11)
    trainer=trainer_fixture()
    before=deepcopy(trainer.model.state_dict())
    log=trainer.advance()
    assert log['update']==1 and log['stage']=='smoke' and log['teacher_mask']==[True]
    assert log['scope']=='field_only_no_feedback_consumer' and log['source_policy']=='none'
    assert log['gradient_norm']>0 and log['parameter_change_l2']>0
    assert any(not torch.equal(v,trainer.model.state_dict()[n]) for n,v in before.items())
    trainer.advance();last=trainer.advance()
    assert last['teacher_mask']==[False]*4 and trainer.update_count==3
    assert trainer.last_state.memory is not None and trainer.last_state.values.grad_fn is None
    with pytest.raises(ValueError,match='budget'): trainer.advance()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def test_training_resume_repeats_next_window_mask_weights_optimizer_and_loss(tmp_path):
    torch.manual_seed(31)
    continuous=trainer_fixture()
    continuous.advance()
    path=tmp_path/'training.pt';continuous.save(path)
    expected=continuous.advance()
    weights=deepcopy(continuous.model.state_dict());optimizer=deepcopy(continuous.optimizer.state_dict())
    torch.manual_seed(999)
    restored=trainer_fixture();restored.load(path)
    actual=restored.advance()
    assert actual==expected
    assert equal_tree(weights,restored.model.state_dict())
    assert equal_tree(optimizer,restored.optimizer.state_dict())
    assert equal_tree(continuous.scheduler.state_dict(),restored.scheduler.state_dict())
    assert torch.equal(continuous.last_state.memory,restored.last_state.memory)
    bad=torch.load(path,weights_only=False)
    bad['last_state'].memory=None
    invalid=tmp_path/'bad.pt';torch.save(bad,invalid)
    old=deepcopy(restored.model.state_dict());oldopt=deepcopy(restored.optimizer.state_dict())
    oldrng=restored.generator.get_state().clone();count=restored.update_count
    with pytest.raises(ValueError):restored.load(invalid)
    assert equal_tree(old,restored.model.state_dict()) and equal_tree(oldopt,restored.optimizer.state_dict())
    assert torch.equal(oldrng,restored.generator.get_state()) and restored.update_count==count
    mismatch=trainer_fixture(seed=20)
    with pytest.raises(ValueError,match='configuration'):mismatch.load(path)


def test_explicit_real_training_bundle_statistics_are_train_only(tmp_path):
    import json
    from src.data.native_training import prepare_training_data
    cfg=json.load(open('config/native-field-curriculum-phaseA-small.json'))
    # A tiny audit instance exercises the same real file Adapter without a full experiment run.
    spec=deepcopy(cfg['data'])
    spec['segments']=[dict(spec['segments'][0],last=4117),dict(spec['segments'][3],last=2710)]
    spec['segments'][0]['plan_jorek']=[dict(first=4111,last=4117,dt=.05)]
    spec['segments'][1]['plan_jorek']=[dict(first=2704,last=2710,dt=5.)]
    spec['blocks']=[dict(spec['blocks'][0],last=4117),dict(spec['blocks'][3],last=2710)]
    bundle=prepare_training_data(spec,tmp_path)
    assert set(bundle.banks)=={1,2,4}
    assert bundle.feature_statistics['training_support']
    assert all(r[0]==spec['training_run'] for r in bundle.feature_statistics['training_support'])
    assert all(w.current.feedback is None for ws in bundle.banks.values() for w in ws)
    assert json.load(open(tmp_path/'manifest.json'))['scope']=='field_only_no_feedback_consumer'
    assert bundle.coverage['coupled_rollout_windows']==0


def test_corrupt_optimizer_and_statistics_cannot_partially_restore(tmp_path):
    trainer=trainer_fixture();trainer.advance()
    p=tmp_path/'state.pt';trainer.save(p);original=torch.load(p,weights_only=False)
    before=deepcopy(trainer.model.state_dict());opt=deepcopy(trainer.optimizer.state_dict())
    for kind in ('optimizer_shape','optimizer_policy','statistics'):
        bad=deepcopy(original)
        if kind=='optimizer_shape':bad['optimizer']['state'][0]['exp_avg']=torch.zeros(1)
        elif kind=='optimizer_policy':bad['optimizer']['param_groups'][0]['betas']=(.1,.2)
        else:bad['model']['state_scale'].mul_(2.)
        q=tmp_path/(kind+'.pt');torch.save(bad,q)
        with pytest.raises(ValueError):trainer.load(q)
        assert equal_tree(before,trainer.model.state_dict()) and equal_tree(opt,trainer.optimizer.state_dict())


def test_evaluation_does_not_advance_sampler_and_distinguishes_dt_zero_limit():
    from src.training.native_evaluation import evaluate_window,dt_limit_diagnostic
    trainer=trainer_fixture();w=trainer.window_banks[4][0]
    rng=trainer.generator.get_state().clone()
    metrics=evaluate_window(trainer.model,w,trainer.weights,teacher_forced=False)
    assert len(metrics['steps'])==4 and metrics['steps'][1]['elapsed_ju']>metrics['steps'][0]['elapsed_ju']
    assert torch.equal(rng,trainer.generator.get_state())
    probe=dt_limit_diagnostic(trainer.model,w,[1.,1e-3,1e-6,0.])
    assert probe['rows'][-1]['increment_scaled_rms']==0.
    assert probe['rows'][-1]['memory_change_rms']==0.
    assert probe['rows'][-2]['increment_scaled_rms']>0 and probe['rows'][-2]['memory_change_rms']>0
    assert metrics['scope']=='field_only_no_feedback_consumer'


def test_budget_train_split_and_global_rng_restore_are_explicit(tmp_path):
    import random
    from src.training.native_curriculum import NativeCurriculumTrainer,CurriculumStage
    trainer=trainer_fixture();trainer.advance()
    p=tmp_path/'rng.pt';trainer.save(p)
    expected=(random.random(),np.random.rand(),torch.rand(3))
    trainer.load(p)
    actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
    kwargs=dict(weights=trainer.weights,stages=trainer.stages,seed=19,learning_rate=.001,
        lr_decay=.99,max_grad_norm=10.,experiment_configuration={'scope':'field_only_no_feedback_consumer'})
    with pytest.raises(ValueError,match='budget'):
        NativeCurriculumTrainer(trainer.model,trainer.window_banks,
            **dict(kwargs,stages=(CurriculumStage('too_many',201,1,1.),)))
    invalid=deepcopy(trainer.window_banks);invalid[1][0].split='test'
    with pytest.raises(ValueError,match='train-only'):
        NativeCurriculumTrainer(trainer.model,invalid,**kwargs)


def test_subdivision_probe_preserves_model_and_sampler_measures_actual_paths():
    from src.training.native_evaluation import subdivision_diagnostic
    trainer=trainer_fixture();model=trainer.model
    before={k:v.clone() for k,v in model.state_dict().items()}
    rng=trainer.generator.get_state().clone()
    result=subdivision_diagnostic(model,trainer.window_banks[4][0])
    assert [r['substeps'] for r in result['rows']]==[1,2,4,8]
    times=[r['elapsed_si'] for r in result['rows']]
    assert max(times)-min(times)<1e-12
    assert result['rows'][0]['field_difference_to_coarse_scaled_rms']==0
    assert result['rows'][-1]['field_difference_to_coarse_scaled_rms']>0
    assert result['rows'][-1]['memory_change_rms']>0
    assert torch.equal(rng,trainer.generator.get_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
