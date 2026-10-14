"""Field-only Torch iteration over existing causal windows, NOT a physical event scheduler."""
from dataclasses import dataclass,replace
import math
import torch
from .native_state import configuration_digest
from .marker_state import fluid_time_coefficients


@dataclass
class TorchFieldState:
    values: torch.Tensor
    raw_deltas: torch.Tensor
    memory: torch.Tensor
    units: object
    time_si: float
    previous_dt_si: float
    step_index: int
    scheme: str
    model_configuration: str
    version: str='native-torch-field-v1'

    def validate_for_model(self,model):
        if (self.version!='native-torch-field-v1' or model.temporal_core is None
            or self.model_configuration!=configuration_digest(model.checkpoint_configuration)
            or self.units!=model.units or type(self.step_index)!=int or self.step_index<0
            or not math.isfinite(self.time_si)):
            raise ValueError('Torch field state/configuration mismatch')
        fluid_time_coefficients(self.scheme,self.previous_dt_si,self.previous_dt_si)
        for a in (self.values,self.raw_deltas):
            if (not isinstance(a,torch.Tensor) or a.shape!=(9,4,1,model.geometry.n_nodes)
                or a.dtype!=model.encoder.weight.dtype or a.device!=model.encoder.weight.device
                or not torch.isfinite(a).all()):
                raise ValueError('invalid complete Torch field/history')
        model.temporal_core.validate_memory(self.memory,self.values.permute(3,0,1,2))

    def truncated(self):
        """Only explicit TBPTT points call this; all physical values/clocks survive."""
        return replace(self,values=self.values.detach(),raw_deltas=self.raw_deltas.detach(),memory=self.memory.detach())


def initialize_field_state(model,native_state,*,previous_dt_si,scheme):
    """Explicit cold start at a field anchor; no future labels or marker assumptions."""
    native_state.validate()
    if model.temporal_core is None: raise ValueError('explicit temporal core required')
    like=model.encoder.weight
    state=TorchFieldState(torch.as_tensor(native_state.values,dtype=like.dtype,device=like.device).clone(),
        torch.as_tensor(native_state.raw_deltas,dtype=like.dtype,device=like.device).clone(),
        like.new_zeros((model.geometry.n_nodes,model.temporal_core.hidden)),native_state.units,
        native_state.time_si,previous_dt_si,native_state.step_index,scheme,
        configuration_digest(model.checkpoint_configuration))
    state.validate_for_model(model)
    return state


def advance_field(model,state,*,dt_si,parameters=None):
    """One field-only candidate, no feedback conversion or marker/shard events."""
    state.validate_for_model(model)
    result=model(state.values,state.raw_deltas,dt_si=dt_si,previous_dt_si=state.previous_dt_si,
        scheme=state.scheme,feature_parameters=parameters,memory=state.memory)
    if dt_si==0: return state
    candidate=replace(state,values=result.values,raw_deltas=result.fluid_deltas,memory=result.memory,
        time_si=state.time_si+dt_si,previous_dt_si=dt_si,step_index=state.step_index+1)
    candidate.validate_for_model(model)
    return candidate


@dataclass
class TorchWindowResult:
    states: tuple
    inputs: tuple
    warmup_states: tuple
    loss_mask: tuple
    scope: str='field_only_no_feedback_consumer'


def rollout_field_window(model,window,*,initial_state=None,teacher_mask=None,detach_before=()):
    """Warm-up uses only past paired U/D. Teachers preserve memory, replace U/D together.

    An explicit initial state is already at prediction start (chunk/resume); no
    warm-up is replayed. Source teachers/coupled qualification are unsupported.
    No optimizer, curriculum RNG or implicit truncation is performed.
    """
    if window.qualification!='field_supervised' or window.config.source_teacher_policy!='none':
        raise ValueError('Torch loop supports field-only windows without a feedback consumer')
    k=window.config.horizon
    mask=tuple(window.config.field_teacher_policy=='paired_U_rawD' for _ in range(k)) if teacher_mask is None else tuple(teacher_mask)
    if len(mask)!=k or any(type(b)!=bool for b in mask):
        raise ValueError('explicit boolean field teacher mask required')
    cuts=tuple(detach_before)
    if len(set(cuts))!=len(cuts) or any(type(i)!=int or not 0<=i<k for i in cuts):
        raise ValueError('invalid explicit truncation points')
    if (len(window.targets)!=k or len(window.time_plan_si)!=k or len(window.warmup)!=window.config.warmup
        or window.plan_provenance not in ('declared_run_control','declared_segment_schedule','recorded_schedule_replay')):
        raise ValueError('incomplete causal window/time plan')
    # Validate all intervals before computing candidates; no external state is mutated.
    for dt in window.support_time_plan_si:
        if not math.isfinite(dt) or dt<=0: raise ValueError('window intervals must be positive physical steps')
    def params(record):
        if model.feature_path is None: return None
        from .native_features import select_feature_parameters
        return select_feature_parameters(record.parameters,scenario_enabled=model.feature_path.scenario_enabled,
            physics_enabled=model.feature_path.physics_enabled)
    def teacher(state,native):
        native.validate()
        if (native.units!=state.units or native.step_index!=state.step_index
            or abs(native.time_si-state.time_si)>max(1e-14,state.units.t_norm_si*1e-7)):
            raise ValueError('teacher pair time/units mismatch')
        return replace(state,values=torch.as_tensor(native.values,dtype=state.values.dtype,device=state.values.device).clone(),
            raw_deltas=torch.as_tensor(native.raw_deltas,dtype=state.raw_deltas.dtype,device=state.raw_deltas.device).clone())
    warm=[]
    if initial_state is None:
        anchor=window.warmup[0] if window.warmup else window.current
        state=initialize_field_state(model,anchor.state,previous_dt_si=anchor.previous_dt_si,scheme=window.scheme)
        for i,record in enumerate(window.warmup):
            state=teacher(state,record.state)
            state=advance_field(model,state,dt_si=window.support_time_plan_si[i],parameters=params(record))
            warm.append(state)
        state=teacher(state,window.current.state)
    else:
        initial_state.validate_for_model(model)
        state=initial_state
    if (state.step_index!=window.prediction_index or state.scheme!=window.scheme
        or abs(state.time_si-window.current.state.time_si)>max(1e-14,state.units.t_norm_si*1e-7)
        or not math.isclose(state.previous_dt_si,window.current.previous_dt_si,rel_tol=1e-9,abs_tol=1e-14)):
        raise ValueError('chunk state must match prediction start clock/history')
    outputs=[]; inputs=[]
    p=params(window.current)  # window contract fixes parameters; no future parameter reads
    for i,dt in enumerate(window.time_plan_si):
        if mask[i]: state=teacher(state,window.current.state if i==0 else window.targets[i-1])
        if i in cuts: state=state.truncated()
        inputs.append(state)
        state=advance_field(model,state,dt_si=dt,parameters=p)
        outputs.append(state)
    return TorchWindowResult(tuple(outputs),tuple(inputs),tuple(warm),tuple(window.loss_mask.tolist()))
