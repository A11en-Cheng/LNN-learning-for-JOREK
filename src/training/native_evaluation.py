"""Field-only recorded-endpoint metrics and independent exact-zero/positive-dt probe."""
import math
import torch
from .native_curriculum import field_supervision,SCOPE
from src.models.native_torch_rollout import rollout_field_window
from src.data.native_features import parameters_from_current_input


def dt_bucket(h):
    if h<.1:return 'JU<0.1'
    if h<1:return '0.1<=JU<1'
    return 'JU>=1'


def ratio_bucket(r):
    if r<.5:return 'ratio<0.5'
    if r>2:return 'ratio>2'
    return '0.5<=ratio<=2'


def evaluate_window(model,window,weights,*,teacher_forced):
    training=model.training;model.eval()
    try:
        with torch.no_grad():
            result=rollout_field_window(model,window,teacher_mask=(teacher_forced,)*window.config.horizon)
            steps=[]
            prev=window.current.previous_dt_si;tn=model.units.t_norm_si
            for i,(s,t,h) in enumerate(zip(result.states,window.targets,window.time_plan_si)):
                loss=field_supervision(model,s,t,weights)
                steps.append(dict(offset=i+1,target_index=t.step_index,time_si=t.time_si,
                    elapsed_si=t.time_si-window.current.state.time_si,
                    elapsed_ju=(t.time_si-window.current.state.time_si)/tn,
                    dt_ju=h/tn,dt_ratio=h/prev,dt_bucket=dt_bucket(h/tn),ratio_bucket=ratio_bucket(h/prev),
                    **loss.metrics()))
                prev=h
            return dict(scope=SCOPE,split=window.split,run=window.run,prediction_index=window.prediction_index,
                horizon=window.config.horizon,policy='TF' if teacher_forced else 'AR',
                source_policy='none',steps=steps,mean_loss=sum(s['total'] for s in steps)/len(steps))
    finally:model.train(training)


def evaluate_banks(model,banks,weights,*,training_run,max_windows,include_test=False):
    if type(max_windows)!=int or max_windows<1:raise ValueError('explicit positive evaluation window cap required')
    results=[]
    for k,windows in sorted(banks.items()):
        groups={}
        for w in windows:
            if w.split=='test' and not include_test:continue
            tag=w.split if w.split!='validation' else ('same_run_validation' if w.run==training_run else 'cross_run_validation')
            groups.setdefault(tag,[]).append(w)
        for tag,ws in sorted(groups.items()):
            selected=sorted(ws,key=lambda w:(w.run,w.prediction_index))[:max_windows]
            for w in selected:
                for teacher in (True,False):
                    r=evaluate_window(model,w,weights,teacher_forced=teacher);r['evaluation_group']=tag;results.append(r)
    return results


def dt_limit_diagnostic(model,window,factors):
    if (not factors or any(not math.isfinite(f) or f<0 for f in factors)
        or factors[-1]!=0):raise ValueError('positive dt factors followed by exact zero required')
    training=model.training;model.eval()
    try:
        with torch.no_grad():
            # One fixed legal warm state; subsequent probe calls do not advance it.
            result=rollout_field_window(model,window,teacher_mask=(False,)*window.config.horizon)
            state=result.inputs[0];params=parameters_from_current_input(window.current_input(),model.feature_path)
            rows=[];base=window.time_plan_si[0]
            for f in factors:
                out=model(state.values,state.raw_deltas,dt_si=base*f,previous_dt_si=state.previous_dt_si,
                    scheme=state.scheme,feature_parameters=params,memory=state.memory)
                rows.append(dict(factor=f,dt_si=base*f,dt_ju=base*f/model.units.t_norm_si,
                    increment_scaled_rms=float((out.increment/model.increment_scale).square().mean().sqrt()),
                    increment_native_field_degree_rms=out.increment.square().mean((2,3)).sqrt().tolist(),
                    memory_change_rms=float((out.memory-state.memory).square().mean().sqrt()),
                    raw_history_change_scaled_rms=float(((out.fluid_deltas-state.raw_deltas)/model.increment_scale).square().mean().sqrt()),
                    field_local_rate_scaled_per_ju=None if f==0 else float((out.increment/model.increment_scale).square().mean().sqrt())/(base*f/model.units.t_norm_si),
                    memory_local_rate_per_ju=None if f==0 else float((out.memory-state.memory).square().mean().sqrt())/(base*f/model.units.t_norm_si)))
            return dict(scope=SCOPE,fixed_state_index=state.step_index,previous_dt_si=state.previous_dt_si,
                rows=rows,caveat='exact no-op differs from positive-dt limit; no semigroup/stability/physical response claim')
    finally:model.train(training)


def subdivision_diagnostic(model,window,subdivisions=(1,2,4,8)):
    """Same elapsed time, actual predicted U/rawD/h and per-substep previous dt.

    This measures sensitivity, not equality to an exact flow or unseen truth.
    """
    from src.models.native_torch_rollout import advance_field
    if (not subdivisions or subdivisions[0]!=1 or
        any(type(n)!=int or n<1 for n in subdivisions)):
        raise ValueError('positive integer subdivisions starting at one required')
    training=model.training;model.eval()
    try:
        with torch.no_grad():
            start=rollout_field_window(model,window,teacher_mask=(False,)*window.config.horizon).inputs[0]
            params=parameters_from_current_input(window.current_input(),model.feature_path)
            interval=window.time_plan_si[0];rows=[];coarse=None
            for n in subdivisions:
                state=start
                for _ in range(n):state=advance_field(model,state,dt_si=interval/n,parameters=params)
                if coarse is None:coarse=state
                rows.append(dict(substeps=n,elapsed_si=state.time_si-start.time_si,
                    dt_ju=interval/n/model.units.t_norm_si,
                    field_change_scaled_rms=float(((state.values-start.values)/model.state_scale).square().mean().sqrt()),
                    memory_change_rms=float((state.memory-start.memory).square().mean().sqrt()),
                    field_difference_to_coarse_scaled_rms=float(((state.values-coarse.values)/model.state_scale).square().mean().sqrt()),
                    memory_difference_to_coarse_rms=float((state.memory-coarse.memory).square().mean().sqrt()),
                    last_raw_history_scaled_rms=float((state.raw_deltas/model.increment_scale).square().mean().sqrt())))
            return dict(scope=SCOPE,rows=rows,fixed_initial_index=start.step_index,
                caveat='rawD is last substep increment; no strict semigroup, exact integration or stability claim')
    finally:model.train(training)
