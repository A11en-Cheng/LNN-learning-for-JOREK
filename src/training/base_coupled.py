"""Complete base-event iteration. No optimizer and no automatic truth resets."""
from dataclasses import dataclass,fields,is_dataclass,replace,field
from copy import deepcopy
import math
import random
import os
import tempfile
import torch
import numpy as np
from src.data.base_coupled import TIME_TOL_SI
from src.models.native_state import configuration_digest


def _equilibrium_configuration(shard_model,provider):
    """Validate the external ES port before any physical event or RNG draw."""
    from src.models.native_source_production import NativeProductionShardModel
    if not isinstance(shard_model,NativeProductionShardModel):
        if provider is not None:raise ValueError('equilibrium provider requires production shard diagnostics')
        return {}
    if (not callable(provider) or not isinstance(getattr(provider,'configuration',None),dict)
        or not provider.configuration):
        raise ValueError('explicit configured equilibrium provider required for production event')
    return dict(equilibrium_provider=provider.configuration)


def detach_state(value):
    """Detach all Tensor graph edges, keeping complete physical state values."""
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if is_dataclass(value):return replace(value,**{f.name:detach_state(getattr(value,f.name)) for f in fields(value)})
    if isinstance(value,dict):return {k:detach_state(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(detach_state(v) for v in value)
    if isinstance(value,list):return [detach_state(v) for v in value]
    return deepcopy(value)


@dataclass
class SequenceSession:
    state: object
    cursor: int=0
    supervision: tuple=()
    interventions: tuple=()
    detach_before: tuple=()
    noise_state: object=None
    teacher_masks: dict=field(default_factory=dict)
    teacher_bundles: dict=field(default_factory=dict)
    sequence_fingerprint: str=''
    adapter_fingerprint: str=''


class ReferenceEvents:
    """Adapter to existing NumPy complete-event and full-state checkpoints."""
    gradient_scope='numpy_reference_nondifferentiable'

    def __init__(self,model,source,shard_model,*,equilibrium_provider=None):
        _equilibrium_configuration(shard_model,equilibrium_provider)
        self.model=model;self.source=source;self.shard_model=shard_model
        self.equilibrium_provider=equilibrium_provider

    @property
    def configuration_fingerprint(self):
        return configuration_digest(dict(model=self.model.checkpoint_configuration,
            source=self.source.configuration,shard=self.shard_model.configuration,gradient_scope=self.gradient_scope,
            **_equilibrium_configuration(self.shard_model,self.equilibrium_provider)))

    @property
    def fingerprint(self):
        return configuration_digest(dict(model=self.model.checkpoint_configuration,
            weights={k:v.detach().cpu().numpy() for k,v in self.model.state_dict().items()},
            source=self.source.configuration,shard=self.shard_model.configuration,gradient_scope=self.gradient_scope,
            **_equilibrium_configuration(self.shard_model,self.equilibrium_provider)))

    def validate(self,state):
        state.validate_fluid_history()
        self.model.reference_solver(state.native_context)  # existing public context validation
        if (self.model.source_configuration!=self.source.configuration
            or self.model.shard_configuration!=self.shard_model.configuration):
            raise ValueError('executing source/shard/model configuration mismatch')

    def validate_control(self,state,control):
        self.validate(state)
        if (abs(state.fluid_time_si-control.start_si)>TIME_TOL_SI
            or state.plan_cursor>=len(state.step_plan_si)
            or not math.isclose(state.step_plan_si[state.plan_cursor],control.dt_si,rel_tol=1e-12,abs_tol=0)
            or state.marker.provenance.get('fluid_control_origin')!=control.origin
            or state.marker.clock.startup_pending!=(control.phase=='registered_restart')):
            raise ValueError('explicit control does not match complete runtime/phase/origin')

    def advance(self,state,control):
        from src.models.native_coupling import advance_coupled_reference
        self.validate_control(state,control)
        candidate,diagnostics=advance_coupled_reference(state,model=self.model,source=self.source,
            shard_model=self.shard_model,equilibrium_provider=self.equilibrium_provider)
        if abs(candidate.fluid_time_si-control.end_si)>TIME_TOL_SI:
            raise ValueError('committed event endpoint differs from declared control')
        return candidate,diagnostics

    def prediction(self,state):
        out=dict(field=dict(values=state.fields_current.copy(),raw_deltas=state.fluid_deltas.copy(),
                   time_si=state.fluid_time_si,units=state.native_context.units),
            marker_projection=state.marker.feedback.values.copy(),
            marker_time_si=state.marker.feedback.time_si)
        s=state.shards
        if 'psi_teacher_event' in s:
            out['psi_feedback_intervention']=deepcopy(s['psi_teacher_event'])
            out['autonomous_prediction']=False
        if 'diagnostic_field_binding' in s:
            out['diagnostic_field_binding']=deepcopy(s['diagnostic_field_binding'])
        if 'ids' in s and s.get('positions_coordinate_system') in ('RZphi','cartesian_m'):
            pos=np.asarray(s['positions'])
            if s['positions_coordinate_system']=='RZphi':
                pos=np.column_stack([pos[:,0]*np.cos(pos[:,2]),pos[:,0]*np.sin(pos[:,2]),pos[:,1]])
            out['shard_time_si']=state.fluid_time_si
            out['shard_positions']=dict(ids=list(s['ids']),positions_m=pos.copy(),alive=(np.asarray(s['radius'])>0).tolist())
            if 'initial_positions_m' in s:out['shard_positions']['initial_positions_m']=np.array(s['initial_positions_m'],copy=True)
        else:out['shard_positions_missing']='fixed_identity_or_coordinate_evidence_not_supplied'
        density=state.shards.get('native_density_source')
        if density is not None:
            out['shard_density']=density.values.detach().cpu().numpy().copy()
            out['shard_density_time_si']=state.marker.feedback.time_si
        return out


@dataclass
class TorchCoupledState:
    """Torch owns field graph; runtime owns detached physical source/shard state."""
    runtime: object
    values: torch.Tensor
    raw_deltas: torch.Tensor
    memory: torch.Tensor


class TorchRecomputedSourceEvents:
    """Existing complete-event driver with Torch field and NumPy source/shard.

    Source and shard consume detached current predictions. This deliberately
    does NOT differentiate their dependence on prior field predictions.
    """
    gradient_scope='torch_field_bptt_stop_gradient_numpy_source_shard'

    def __init__(self,model,source,shard_model,*,observation_noise=None,equilibrium_provider=None):
        self.observation_noise=observation_noise
        self.reference=ReferenceEvents(model,source,shard_model,equilibrium_provider=equilibrium_provider)
        self.model=model;self.source=source;self.shard_model=shard_model
        self.equilibrium_provider=equilibrium_provider

    @property
    def configuration_fingerprint(self):
        return configuration_digest(dict(reference=self.reference.configuration_fingerprint,gradient_scope=self.gradient_scope,
            observation_noise=None if self.observation_noise is None else self.observation_noise.configuration))

    @property
    def fingerprint(self):
        return configuration_digest(dict(reference=self.reference.fingerprint,gradient_scope=self.gradient_scope,
            observation_noise=None if self.observation_noise is None else self.observation_noise.configuration))

    def initialize(self,runtime):
        runtime=deepcopy(runtime);self.reference.validate(runtime)
        like=self.model.encoder.weight
        tensor=lambda x:torch.as_tensor(x,dtype=like.dtype,device=like.device).clone()
        if runtime.field_memory is None:raise ValueError('explicit field memory required')
        return TorchCoupledState(runtime,tensor(runtime.fields_current),tensor(runtime.fluid_deltas),tensor(runtime.field_memory))

    def restore_state(self,state):
        like=self.model.encoder.weight
        return replace(state,values=state.values.to(like),raw_deltas=state.raw_deltas.to(like),memory=state.memory.to(like))

    def validate(self,state):
        if not isinstance(state,TorchCoupledState):raise ValueError('complete Torch coupled state required')
        self.reference.validate(state.runtime)
        for tensor,array in [(state.values,state.runtime.fields_current),
                             (state.raw_deltas,state.runtime.fluid_deltas),(state.memory,state.runtime.field_memory)]:
            if not np.array_equal(tensor.detach().cpu().numpy(),array):
                raise ValueError('Torch graph and committed NumPy mirror disagree')

    def validate_control(self,state,control):
        self.validate(state);self.reference.validate_control(state.runtime,control)

    def advance(self,state,control):
        # The physical input is immutable; failed noisy events must not consume RNG.
        weights={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        try:
            result=self._advance(state,control)
            if any(not torch.equal(v,self.model.state_dict()[k]) for k,v in weights.items()):
                raise ValueError('event mutated model parameters/buffers')
            return result
        except Exception:
            self.model.load_state_dict(weights)
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            raise

    def _advance(self,state,control):
        from src.models.marker_state import reference_fluid_event,reference_particle_then_fluid,FluidSolveResult
        from src.models.native_coupling import advance_source_event,bind_coupled_equilibrium,refresh_coupled_diagnostics
        self.validate_control(state,control)
        candidate=deepcopy(state.runtime)
        bind_coupled_equilibrium(candidate,self.shard_model,self.equilibrium_provider)
        candidate.shards['native_density_source']=self.shard_model.density_source(candidate.shards)
        public_solver=self.model.reference_solver(candidate.native_context)
        saved=[]
        def solve(old,shards,feedback,dt,previous_dt,*,fluid_deltas,time_coefficients,field_event,field_memory):
            # Incoming driver arrays are validated mirrors. The actual graph
            # must enter from state tensors, not be rebuilt from those arrays.
            inputs=self.model.coupled_inputs(shards,feedback,self.model.encoder.weight.device)
            observation=(None if self.observation_noise is None else
                self.observation_noise.observe(state.values,state.raw_deltas,model=self.model))
            options={} if observation is None else {"observation_values":observation.values}
            out=self.model(state.values,state.raw_deltas,dt_si=dt,previous_dt_si=previous_dt,
                scheme=candidate.time_evol_scheme,time_coefficients=time_coefficients,
                feature_parameters=public_solver.feature_parameters,memory=state.memory,
                field_event=field_event,**inputs,**options)
            saved.append(out)
            array=lambda x:x.detach().cpu().numpy().copy()
            return FluidSolveResult(array(out.values),array(out.fluid_deltas),array(out.memory))
        solve.supports_field_event=True;solve.supports_field_memory=True
        if candidate.marker.clock.startup_pending:
            out=reference_fluid_event(candidate,solve_fluid=solve,update_shards=self.shard_model.update)
            diag={'event':'registered_restart_zero_projection_startup'}
        else:
            def particle(marker,target,previous,current,shards):
                return advance_source_event(candidate,self.source,self.shard_model,marker,target,previous,current,shards)
            out,diag=reference_particle_then_fluid(candidate,particle_step=particle,
                solve_fluid=solve,update_shards=self.shard_model.update)
        refresh_coupled_diagnostics(out,self.shard_model,self.equilibrium_provider)
        if abs(out.fluid_time_si-control.end_si)>TIME_TOL_SI:raise ValueError('event endpoint/control mismatch')
        result=TorchCoupledState(out,saved[0].values,saved[0].fluid_deltas,saved[0].memory)
        self.validate(result)
        return result,dict(diag,gradient_scope=self.gradient_scope)

    def prediction(self,state):
        out=self.reference.prediction(state.runtime)
        out['field']['values']=state.values.clone()
        out['field']['raw_deltas']=state.raw_deltas.clone()
        return out


def run_sequence(sequence,adapter,initial_state=None,*,session=None,stop_cursor=None,
                 detach_before=(),teacher_masks=None,teacher_bundles=None,loss_fn=None):
    """Advance every control, observe only real nodes; resume skips burn-in.

    Only coherent full physical bundle interventions are supported here. Given
    source-only teachers require a separately named conditional-source Adapter.
    """
    if not sequence.qualifications['combined']['eligible']:
        raise ValueError('ineligible sequence: '+str(sequence.qualifications['combined']['reasons']))
    if sequence.task=='conditional_source':raise ValueError('conditional-source execution requires an explicit conditional Adapter')
    n=len(sequence.controls)
    cuts=tuple(detach_before)
    if len(set(cuts))!=len(cuts) or any(type(i)!=int or not 0<=i<n for i in cuts):
        raise ValueError('invalid TBPTT cut cursors')
    if session is None:
        if initial_state is None:raise ValueError('complete initial state required; burn-in cannot manufacture it')
        out=SequenceSession(detach_state(initial_state),detach_before=cuts,
            teacher_masks=deepcopy(teacher_masks or {}),teacher_bundles=detach_state(teacher_bundles or {}))
    else:
        if initial_state is not None:raise ValueError('choose continuation or independent initialization')
        if cuts and cuts!=session.detach_before:raise ValueError('resume truncation policy mismatch')
        if teacher_masks is not None and teacher_masks!=session.teacher_masks:
            raise ValueError('resume teacher mask policy mismatch')
        if teacher_bundles is not None:raise ValueError('resume uses checkpoint-bound teacher bundles')
        out=replace(session)
    if out.sequence_fingerprint and (out.sequence_fingerprint!=sequence.fingerprint or out.adapter_fingerprint!=adapter.fingerprint):
        raise ValueError('continuation sequence/model/parameters fingerprint mismatch')
    if not out.sequence_fingerprint and out.cursor!=0:raise ValueError('unbound continuation state')
    out=replace(out,sequence_fingerprint=sequence.fingerprint,adapter_fingerprint=adapter.fingerprint)
    stop=n if stop_cursor is None else stop_cursor
    if type(stop)!=int or not 0<=out.cursor<=stop<=n:raise ValueError('invalid sequence cursor')
    adapter.validate(out.state)
    masks=out.teacher_masks;bundles=out.teacher_bundles
    for cursor,mask in masks.items():
        if (type(cursor)!=int or not 1<=cursor<=n or set(mask)!={'field','source','shard'}
            or any(type(v)!=bool for v in mask.values()) or len(set(mask.values()))!=1):
            raise ValueError('teachers must replace a coherent complete field/source/shard bundle')
        if any(mask.values()) and cursor not in bundles:raise ValueError('complete certified teacher bundle required')
        if any(mask.values()):
            support=bundles[cursor].get('support',())
            if not support or any(s.run!=sequence.run or s.last_si>sequence.controls[cursor-1].end_si+TIME_TOL_SI
                or not any(s.run==whole.run and whole.first_si<=s.first_si+TIME_TOL_SI
                    and s.last_si<=whole.last_si+TIME_TOL_SI for whole in sequence.support) for s in support):
                raise ValueError('teacher cache full support must already be isolated and causal in sequence')
    while out.cursor<stop:
        i=out.cursor
        state=detach_state(out.state) if i in out.detach_before else out.state
        before_runtime=getattr(state,'runtime',state)
        previous_dt=sequence.controls[i].dt_si if before_runtime.marker.clock.startup_pending else before_runtime.fluid_dt_previous_si
        state,diag=adapter.advance(state,sequence.controls[i])
        cursor=i+1;observations=out.supervision;interventions=out.interventions
        mask=sequence.mask_at(cursor);node=sequence.node_at(cursor)
        if any(mask.values()):
            prediction=adapter.prediction(state)
            # Object event times may differ from fluid time; labels have to bind
            # to their owner, never to a convenient filename.
            if mask['marker_projection']:
                expected=node.object_times_si.get('marker_projection',node.time_si)
                if abs(prediction['marker_time_si']-expected)>TIME_TOL_SI:
                    raise ValueError('marker projection supervision event time mismatch')
            loss=None if loss_fn is None else loss_fn(prediction,node,mask)
            group=dict(run=sequence.run,task=sequence.task,
                burnin_time_si=sum(c.dt_si for c in sequence.controls[:sequence.burnin_steps]),
                burnin_record_count=sequence.manifest()['burnin_record_count'],
                prediction_time_si=sum(c.dt_si for c in sequence.controls[sequence.burnin_steps:cursor]),
                dt_si=sequence.controls[i].dt_si,dt_ratio=sequence.controls[i].dt_si/previous_dt,
                teacher_policy='coherent_bundle' if any(any(m.values()) for m in masks.values()) else 'none')
            observations=(*observations,dict(cursor=cursor,time_si=node.time_si,mask=mask,loss=loss,diagnostics=diag,group=group))
        if any(masks.get(cursor,{}).values()):
            bundle=bundles[cursor]
            if (not isinstance(bundle,dict) or bundle.get('status') not in ('saved','causal_replayed','verified_reconstructed')
                or not bundle.get('provenance') or bundle.get('memory_policy')!='same_trajectory_cache'
                or bundle.get('adapter_fingerprint')!=adapter.fingerprint
                or not bundle.get('support_verified')):
                raise ValueError('teacher bundle requires certified same-trajectory memory and full split support')
            replacement=detach_state(bundle['state']);adapter.validate(replacement)
            runtime=getattr(replacement,'runtime',replacement)
            if abs(runtime.fluid_time_si-sequence.controls[cursor-1].end_si)>TIME_TOL_SI:
                raise ValueError('teacher bundle must match the supervised event time')
            # Next control checks all clocks/plan/origin before any further update.
            if cursor<n:adapter.validate_control(replacement,sequence.controls[cursor])
            state=replacement;interventions=(*interventions,dict(cursor=cursor,provenance=bundle['provenance']))
        out=replace(out,state=state,cursor=cursor,supervision=observations,interventions=interventions)
    return out


def save_session(path,sequence,adapter,session):
    """Trusted local checkpoint, no optimizer. All model/plan changes invalidate it."""
    payload=dict(schema='base-coupled-session-v1',sequence=sequence.fingerprint,
        adapter=adapter.fingerprint,gradient_scope=adapter.gradient_scope,
        adapter_configuration=adapter.configuration_fingerprint,
        model_state={k:v.detach().cpu().clone() for k,v in adapter.model.state_dict().items()},
        manifest=sequence.manifest(),session=detach_state(session),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()))
    directory=os.path.dirname(os.path.abspath(path))
    fd,temporary=tempfile.mkstemp(prefix='.base-session-',dir=directory);os.close(fd)
    try:
        with open(temporary,'wb') as stream:torch.save(payload,stream)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_session(path,sequence,adapter):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload.get('schema')!='base-coupled-session-v1' or payload['sequence']!=sequence.fingerprint
        or payload['adapter_configuration']!=adapter.configuration_fingerprint or payload['gradient_scope']!=adapter.gradient_scope):
        raise ValueError('checkpoint sequence/model/source/control fingerprint mismatch')
    before={k:v.detach().clone() for k,v in adapter.model.state_dict().items()}
    rng_before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        adapter.model.load_state_dict(payload['model_state'],strict=True)
        if payload['adapter']!=adapter.fingerprint:raise ValueError('checkpoint weights/Adapter fingerprint mismatch')
        session=payload['session']
        if hasattr(adapter,'restore_state'):session=replace(session,state=adapter.restore_state(session.state))
        adapter.validate(session.state)
        if not 0<=session.cursor<=len(sequence.controls):raise ValueError('checkpoint cursor outside sequence')
        if session.cursor<len(sequence.controls):adapter.validate_control(session.state,sequence.controls[session.cursor])
        random.setstate(payload['rng']['python']);np.random.set_state(payload['rng']['numpy']);torch.set_rng_state(payload['rng']['torch'])
    except Exception:
        adapter.model.load_state_dict(before)
        random.setstate(rng_before[0]);np.random.set_state(rng_before[1]);torch.set_rng_state(rng_before[2])
        raise
    return session
