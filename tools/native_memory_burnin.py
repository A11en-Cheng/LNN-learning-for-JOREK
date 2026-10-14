"""Memory cold-start comparisons on weight-bound, fully causal physical caches.

Not a comparison of independent JOREK physical initializations: every cache has
the complete prefix support. Reset only learned memory, never physical history.
"""
from dataclasses import dataclass
import torch
from src.training.base_coupled import detach_state
from tools.train_native_autonomous_pilot import _rng,_restore_rng


@dataclass
class CausalPrefix:
    fingerprint: str
    states: tuple
    rng_states: tuple
    controls: tuple
    training_mode: bool


def make_prefix(adapter,runtime,controls,forecast_cursor):
    if not 0<=forecast_cursor<len(controls):raise ValueError('forecast cursor outside controls')
    state=adapter.initialize(runtime);states=[detach_state(state)];rngs=[_rng()]
    with torch.no_grad():
        for control in controls[:forecast_cursor]:
            state,_=adapter.advance(state,control)
            states.append(detach_state(state));rngs.append(_rng())
    return CausalPrefix(adapter.fingerprint,tuple(states),tuple(rngs),tuple(controls),adapter.model.training)


def forecast(adapter,prefix,*,burnin,horizon,grad=False,reset_memory=True):
    """Same forecast clock/horizon, varying *explicit memory reset* location."""
    cursor=len(prefix.states)-1
    if (type(burnin) is not int or not 0<=burnin<=cursor or horizon<1
        or cursor+horizon>len(prefix.controls)):
        raise ValueError('invalid burn-in or forecast support')
    if adapter.fingerprint!=prefix.fingerprint or adapter.model.training!=prefix.training_mode:
        raise ValueError('causal prefix stale after weights/configuration changed')
    start=cursor-burnin
    reuse=reset_memory and burnin==cursor and not bool(torch.count_nonzero(prefix.states[0].memory))
    effective_start=cursor if reuse else start
    state=detach_state(prefix.states[effective_start])
    _restore_rng(prefix.rng_states[effective_start])
    if reset_memory and not reuse:
        state.memory=torch.zeros_like(state.memory)
        state.runtime.field_memory=state.memory.detach().cpu().numpy().copy()
    adapter.validate(state)
    with torch.no_grad():
        for control in prefix.controls[effective_start:cursor]:state,_=adapter.advance(state,control)
    from tools.diagnose_native_real_chain import _array_digest
    baseline=prefix.states[cursor]
    delta=state.values.detach()-baseline.values
    reference=baseline.values.norm()
    start_proof=dict(memory_reset=reset_memory,warmup_reused_from_current_weight_prefix=reuse,
        forecast_initial_state_digest=_array_digest(state.runtime),
        forecast_rng_digest=_array_digest(_rng()),
        forecast_field_delta_from_continuous=float(delta.norm().cpu()),
        forecast_field_relative_l2_from_continuous=None if reference==0 else float((delta.norm()/reference).cpu()),
        forecast_memory_delta_from_continuous=float((state.memory-baseline.memory).norm().cpu()))
    predictions=[]
    with torch.set_grad_enabled(grad):
        for control in prefix.controls[cursor:cursor+horizon]:
            state,_=adapter.advance(state,control);predictions.append(adapter.prediction(state))
    return predictions,dict(experiment='memory_cold_start_on_causal_prefix_v1',
        physical_initialization='full_model_generated_cache_not_truth_replacement',
        burnin_steps=burnin,forecast_steps=horizon,
        forecast_start_si=prefix.controls[cursor].start_si,
        forecast_end_si=prefix.controls[cursor+horizon-1].end_si,
        burnin_time_si=sum(c.dt_si for c in prefix.controls[start:cursor]),
        full_prefix_support_steps=cursor,cache_fingerprint=prefix.fingerprint,**start_proof)
