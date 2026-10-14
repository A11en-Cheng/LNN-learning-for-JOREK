"""Causal full-native windows. Dataset labels never enter CurrentFieldInput.

Run/block assignment precedes window enumeration. This module prepares data;
no rollout scheduler, solver, gradient evaluator or training loop lives here.
"""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from collections import Counter
import numpy as np
from src.models.native_state import NativeFieldState, NativeFeedbackState

SCHEMA='native-causal-window-v2'


@dataclass
class WindowSources:
    """Given legal sources, not a certificate of a recoverable marker cloud."""
    feedback: NativeFeedbackState
    shard_density_rate: np.ndarray
    density_unit: str
    density_provenance: dict
    compatibility: str
    source_interval_si: tuple
    projection_history_provenance: dict
    schema_version: str='native-window-dual-source-v1'

    def validate(self,record):
        if self.schema_version!='native-window-dual-source-v1' or not isinstance(self.feedback,NativeFeedbackState):
            raise ValueError('complete versioned dual-source state required')
        self.feedback.validate()
        f=self.feedback;s=record.state
        if (f.units!=s.units or f.modes!=s.modes or f.values.dtype!=s.values.dtype or f.values.shape[1:]!=s.values.shape[1:]
            or self.compatibility!=record.compatibility):
            raise ValueError('source/field native context or units mismatch')
        history=self.projection_history_provenance
        if (not isinstance(history,dict) or history.get('status')!='saved' or not history.get('source')
            or not history.get('support_indices') or not history.get('values_source')):
            raise ValueError('saved projection-history evidence required; reconstruction certification is not implemented here')
        provenance=self.density_provenance
        if (not isinstance(provenance,dict) or provenance.get('status') not in ('saved','verified_reconstructed','causal_replayed')
            or not provenance.get('source') or not provenance.get('support_indices')):
            raise ValueError('explicit density source provenance and raw support required')
        if (self.density_unit!='model502_mass_density_per_JU'
            or not isinstance(self.shard_density_rate,np.ndarray)
            or self.shard_density_rate.shape!=(2,*s.values.shape[1:])
            or self.shard_density_rate.dtype!=s.values.dtype
            or not np.isfinite(self.shard_density_rate).all()):
            raise ValueError('independent full-native shard density rate/units/provenance required')
        tol=max(1e-14,s.units.t_norm_si*1e-7)
        if abs(f.time_si-s.time_si)>tol or f.gather_time_si is None:
            raise ValueError('source event requires current field time and actual gather')
        interval=np.asarray(self.source_interval_si)
        if (interval.shape!=(2,) or not np.isfinite(interval).all() or interval[0]<0
            or interval[0]>=interval[1] or abs(interval[1]-f.time_si)>tol):
            raise ValueError('invalid source sampling interval; it is not gather or next dt')

    def manifest(self):
        return dict(schema=self.schema_version,shape=list(self.shard_density_rate.shape),
            density_unit=self.density_unit,density_provenance=self.density_provenance,
            compatibility=self.compatibility,source_interval_si=list(self.source_interval_si),
            source_time_si=self.feedback.time_si,gather_time_si=self.feedback.gather_time_si,
            feedback_values_provenance=self.projection_history_provenance['values_source'],
            feedback_history_provenance=self.feedback.history_provenance,
            projection_history_evidence=deepcopy(self.projection_history_provenance),
            teacher_policy='given_source_condition_not_self_generated_cloud')


@dataclass
class RecoveredCoupledAnchor:
    """A restored complete ReferenceRollout, not an opaque snapshot path.

    This certifies the registered small reference subset, not production BC or
    source accuracy. Providers must still match the saved configurations at run.
    """
    runtime: object
    state_status: dict
    provenance: str
    support_indices: tuple

    def validate(self,record,plan,scheme):
        from src.models.marker_state import ReferenceRollout
        from src.models.native_state import configuration_digest
        required=('field','fluid_history','previous_field','marker','feedback_history','clock',
                  'shard','memory','control','configuration')
        for name in required:
            if self.state_status.get(name) not in ('saved','verified_reconstructed','causal_replayed'):
                raise ValueError('unavailable coupled anchor '+name)
        if not self.provenance or not self.support_indices or not isinstance(self.runtime,ReferenceRollout):
            raise ValueError('complete restored coupled anchor/provenance/support required')
        state=self.runtime
        state.validate_fluid_history()
        if state.native_context is None:raise ValueError('missing native coupled anchor context')
        if state.marker.clock.startup_pending:
            raise ValueError('registered startup is a discrete event, not an ordinary window anchor')
        current=state.native_field_state()
        if (current.units!=record.state.units or current.modes!=record.state.modes
            or current.time_si!=record.state.time_si or current.step_index!=record.state.step_index
            or not np.array_equal(current.values,record.state.values)
            or not np.array_equal(current.raw_deltas,record.state.raw_deltas)):
            raise ValueError('coupled anchor does not match complete current field/history')
        if state.time_evol_scheme!=scheme or state.fluid_dt_previous_si!=record.previous_dt_si:
            raise ValueError('coupled anchor previous dt/format mismatch')
        if state.marker.provenance.get('fluid_control_origin') not in ('external_control','recorded_schedule_replay'):
            raise ValueError('coupled anchor lacks explicit next-step control origin')
        remaining=state.step_plan_si[state.plan_cursor:state.plan_cursor+len(plan)]
        if len(remaining)!=len(plan) or not np.allclose(remaining,plan,rtol=1e-12,atol=0):
            raise ValueError('coupled anchor control differs from window')
        model=state.native_context.configuration['model']
        source=model.get('source',{});shards=model.get('shards',{})
        if (model.get('coupled_version')!='native-source-field-v3_bound_events'
            or model.get('coupling_mode')!='joint_learning'
            or source.get('version')!='native-small-marker-source-v1'
            or shards.get('version')!='small-native-Ne-shard-v1'):
            raise ValueError('unsupported coupled anchor source/model configuration')
        mesh=state.native_context.configuration['mesh']
        if not all(k in mesh for k in ('x','size','vertex','index')):
            raise ValueError('coupled anchor requires full geometry and shared indices')
        digest=configuration_digest(dict(x=np.asarray(mesh['x']),size=np.asarray(mesh['size']),
            vertex=np.asarray(mesh['vertex'])-1,indices=np.asarray(mesh['index'])))
        if source.get('mesh',{}).get('geometry')!=digest or shards.get('mesh',{}).get('geometry')!=digest:
            raise ValueError('coupled anchor source geometry mismatch')
        source_mesh=source['mesh']
        if source_mesh.get('backend') not in ('dense','sparse') or source_mesh.get('units')!=vars(current.units):
            raise ValueError('coupled anchor requires a supported projection solver and matching units')
        if shards.get('ablation_model')!=3 or not {'no_rocket','no_nonlocal'}.issubset(shards.get('approximations',())):
            raise ValueError('unsupported shard configuration at coupled anchor')
        feedback=state.marker.feedback
        if feedback.gather_time_si is None:
            empty=(not state.marker.cloud.alive.any() and not feedback.values.any() and not feedback.deltas.any())
            if not empty or state.marker.provenance.get('confirmed_empty') is not True:
                raise ValueError('missing actual feedback gather; only verified empty startup is exempt')
        if state.marker.cloud.P_imp.shape[1]!=11 or state.marker.cloud.alive.sum()>10000:
            raise ValueError('unsupported source species or live-marker reference resource limit')
        if not isinstance(state.shards,dict) or any(k not in state.shards for k in
                ('positions','velocities','radius','ablation_atoms_s')):
            raise ValueError('missing complete independent shard state')
        positions=np.asarray(state.shards['positions']);n=len(positions)
        if (positions.shape!=(n,3) or np.shape(state.shards['velocities'])!=(n,3)
            or any(np.shape(state.shards[k])!=(n,) for k in ('radius','ablation_atoms_s'))
            or any(not np.isfinite(state.shards[k]).all() for k in ('positions','velocities','radius','ablation_atoms_s'))):
            raise ValueError('invalid complete shard arrays')
        births=int(np.count_nonzero(np.asarray(state.shards['radius'])>0))*len(plan)
        live=int(state.marker.cloud.alive.sum())
        if len(state.marker.cloud.weight)-live<births or live+births>10000:
            raise ValueError('insufficient marker slot/live resource budget for reference horizon')

    def manifest(self):
        return dict(provenance=self.provenance,state_status=dict(self.state_status),
            native_context_fingerprint=self.runtime.native_context.fingerprint(),
            fluid_control_origin=self.runtime.marker.provenance.get('fluid_control_origin'),
            support_indices=list(self.support_indices),boundary='ordinary_continuation',
            scope='registered_small_reference_not_production_configuration')


@dataclass
class FieldRecord:
    run: str
    state: NativeFieldState
    previous_dt_si: float
    compatibility: str
    provenance: str
    restart: bool=False
    feedback: NativeFeedbackState | None=None
    source_interval_si: tuple | None=None
    parameters: tuple=()
    # Only a verified full checkpoint can qualify; marker replay is unsupported.
    initialization: dict=field(default_factory=dict)
    sources: WindowSources | None=None
    coupled_anchor: RecoveredCoupledAnchor | None=None


@dataclass(frozen=True)
class SplitBlock:
    run: str
    split: str
    first: int
    last: int


@dataclass(frozen=True)
class WindowConfig:
    warmup: int
    horizon: int
    stride: int
    isolation: int
    replay_steps: int=0
    field_teacher_policy: str='paired_U_rawD'
    source_teacher_policy: str='none'
    truncation_steps: int | None=None
    required_qualification: str='field_supervised'

    def __post_init__(self):
        if any(type(x)!=int for x in (self.warmup,self.horizon,self.stride,self.isolation,self.replay_steps)):
            raise ValueError('window lengths must be integers')
        if min(self.warmup,self.isolation,self.replay_steps)<0 or min(self.horizon,self.stride)<1:
            raise ValueError('invalid window lengths')
        if self.replay_steps: raise ValueError('causal marker replay is not implemented')
        if self.required_qualification not in ('field_supervised','conditional_source','self_generated_source'):
            raise ValueError('coupled initialization qualification is unsupported; no verified replay')
        if self.required_qualification=='conditional_source' and self.source_teacher_policy!='recorded_full_history':
            raise ValueError('conditional source windows require explicit source teacher policy')
        if self.required_qualification=='self_generated_source' and (self.warmup or self.field_teacher_policy!='autoregressive' or self.source_teacher_policy!='self_generated_reference'):
            raise ValueError('self-generated reference requires restored prediction anchor, no implicit warm-up or field teachers')
        if self.field_teacher_policy not in ('paired_U_rawD','autoregressive') or self.source_teacher_policy not in ('none','recorded_full_history','self_generated_reference'):
            raise ValueError('unsupported teacher policy')
        if self.truncation_steps is not None and (type(self.truncation_steps)!=int or self.truncation_steps<1):
            raise ValueError('invalid truncation plan')


@dataclass(frozen=True)
class CurrentFieldInput:
    state: NativeFieldState
    dt_si: float
    previous_dt_si: float
    scheme: str
    endpoint_si: float
    plan_provenance: str
    feedback: NativeFeedbackState | None
    source_interval_si: tuple | None
    parameters: tuple
    event_phase: str='before_fluid_solve'
    sources: WindowSources | None=None


@dataclass
class NativeWindow:
    split: str
    run: str
    prediction_index: int
    support_indices: tuple
    warmup: tuple
    current: FieldRecord
    targets: tuple
    time_plan_si: tuple
    plan_provenance: str
    scheme: str
    config: WindowConfig
    qualification: str
    initialization: dict
    support_records: tuple
    support_time_plan_si: tuple
    eligibility: dict=field(default_factory=dict)
    coupled_anchor: RecoveredCoupledAnchor | None=None

    def initial_coupled_state(self):
        if not self.eligibility.get('self_generated_source',{}).get('eligible') or self.coupled_anchor is None:
            raise ValueError('window has no recovered self-generated source anchor')
        self.coupled_anchor.validate(self.current,self.time_plan_si,self.scheme)
        return deepcopy(self.coupled_anchor.runtime)

    @property
    def loss_mask(self):
        return np.array([False]*self.config.warmup+[True]*self.config.horizon)

    def current_input(self):
        r=self.current
        return CurrentFieldInput(deepcopy(r.state),self.time_plan_si[0],r.previous_dt_si,
            self.scheme,r.state.time_si+self.time_plan_si[0],self.plan_provenance,
            deepcopy(r.feedback if r.sources is None else r.sources.feedback) if self.config.source_teacher_policy!='none' else None,
            r.source_interval_si if r.sources is None else r.sources.source_interval_si,deepcopy(tuple(p for p in r.parameters if p['status']=='declared')),
            sources=deepcopy(r.sources) if self.config.source_teacher_policy!='none' else None)

    def manifest(self):
        return dict(schema=SCHEMA,run=self.run,split=self.split,prediction_index=self.prediction_index,
            support_indices=list(self.support_indices),config=asdict(self.config),
            qualification=self.qualification,eligibility=self.eligibility,initialization=self.initialization,
            support_time_plan_si=list(self.support_time_plan_si),
            time_plan_si=list(self.time_plan_si),plan_provenance=self.plan_provenance,scheme=self.scheme,
            support=[dict(index=r.state.step_index,time_si=r.state.time_si,previous_dt_si=r.previous_dt_si,
                provenance=r.provenance,restart=r.restart,compatibility=r.compatibility,
                units=asdict(r.state.units),modes=list(r.state.modes),
                state_schema=r.state.schema_version,shape=list(r.state.values.shape),
                source_interval_si=r.source_interval_si,sources=None if r.sources is None else r.sources.manifest(),
                feedback_history_provenance=None if r.feedback is None else r.feedback.history_provenance,
                gather_time_si=None if r.feedback is None else r.feedback.gather_time_si,parameters=list(r.parameters))
                for r in self.support_records],
            current_provenance=self.current.provenance,loss_mask=self.loss_mask.tolist(),
            warmup_mask=[True]*self.config.warmup+[False]*self.config.horizon,
            valid_target_mask=[True]*self.config.horizon,
            field_teacher_mask=[self.config.field_teacher_policy=='paired_U_rawD']*self.config.horizon,
            source_teacher_mask=[self.config.source_teacher_policy=='recorded_full_history']*self.config.horizon,
            future_labels_only=True,gradient_contract='evaluate actual full U with native B/G in physical units; then scale',
            teacher_replacement='U and rawD together; recompute gradients; no coupled partial replacement')


@dataclass
class WindowCollection:
    windows: list
    exclusions: dict
    rejections: list=field(default_factory=list)

    def manifest(self):
        return dict(schema=SCHEMA,windows=[w.manifest() for w in self.windows],exclusions=self.exclusions,rejections=self.rejections)


def build_windows(records, blocks, config, *, time_plan, plan_provenance, scheme):
    """Explicit external control or declared recorded replay, keyed by current step.

    Recorded replay is not an inference-known autonomous control plan.
    Plan is never silently inferred from labels. Future endpoint times only validate it.
    Incompatible/restart windows are excluded with counts; malformed required
    state/time/plan fails. All work is private; input records remain untouched.
    """
    from src.models.marker_state import fluid_time_coefficients
    if not plan_provenance or plan_provenance not in ('declared_run_control','declared_segment_schedule','recorded_schedule_replay'):
        raise ValueError('time plan must have declared inference-available provenance')
    fluid_time_coefficients(scheme,1.,1.)
    for b in blocks:
        if b.split not in ('train','validation','test') or b.first>b.last:
            raise ValueError('invalid split block')
    for i,a in enumerate(blocks):
        for b in blocks[i+1:]:
            if a.run==b.run:
                distance=max(a.first,b.first)-min(a.last,b.last)-1
                if distance<0 or (a.split!=b.split and distance<config.isolation):
                    raise ValueError('split full support overlaps or isolation is insufficient')
    lookup={}
    for r in records:
        r.state.validate()
        key=(r.run,r.state.step_index)
        if key in lookup: raise ValueError('duplicate raw record identity')
        if not r.compatibility or not r.provenance or not np.isfinite(r.previous_dt_si) or r.previous_dt_si<=0:
            raise ValueError('missing compatibility/provenance/previous dt')
        for p in r.parameters:
            if p.get('provenance',{}).get('kind') not in ('h5','namelist','default','derived'):
                raise ValueError('unknown parameter provenance')
            if p.get('status')=='declared' and (not p.get('unit') or p.get('available_at')!='initialization'):
                raise ValueError('declared parameter missing units/availability')
        if r.sources is not None:r.sources.validate(r)
        lookup[key]=r
    out=[]; excluded=Counter();rejections=[]
    def reject(reason):
        excluded[reason]+=1
        rejections.append(dict(run=b.run,split=b.split,prediction_index=n,
            support_indices=list(ids),reason=reason))
    for b in blocks:
        for n in range(b.first+config.warmup,b.last-config.horizon+1,config.stride):
            ids=tuple(range(n-config.warmup,n+config.horizon+1))
            if any((b.run,i) not in lookup for i in ids):
                reject('missing_record'); continue
            rs=[lookup[b.run,i] for i in ids]
            if any(r.restart for r in rs[1:]):
                reject('restart_crossing'); continue
            if len({r.compatibility for r in rs})!=1 or any(r.state.units!=rs[0].state.units or r.state.modes!=rs[0].state.modes or r.state.values.shape!=rs[0].state.values.shape for r in rs):
                reject('incompatible_state'); continue
            from src.models.native_state import configuration_digest
            fixed=[configuration_digest([(p['name'],p['value']) for p in r.parameters if p['group']!='numeric_time']) for r in rs]
            if len(set(fixed))!=1:
                reject('parameter_change'); continue
            plan=[]
            for a,z in zip(rs,rs[1:]):
                h=time_plan.get((b.run,a.state.step_index))
                if h is None or not np.isfinite(h) or h<=0:
                    raise ValueError('missing positive declared time plan')
                if not np.isclose(z.state.time_si-a.state.time_si,h,rtol=1e-9,atol=max(1e-14,a.state.units.t_norm_si*1e-7)):
                    raise ValueError('wrong time pairing or declared plan')
                if not np.isclose(z.previous_dt_si,h,rtol=1e-9,atol=1e-14):
                    raise ValueError('saved previous dt disagrees with declared plan')
                plan.append(float(h))
            if config.source_teacher_policy=='recorded_full_history':
                for r in rs[:-1]:
                    feedback=r.feedback if r.sources is None else r.sources.feedback
                    if feedback is None: raise ValueError('missing full feedback and raw projection history')
                    feedback.validate()
                    interval=r.source_interval_si if r.sources is None else r.sources.source_interval_si
                    if (feedback.units!=r.state.units or feedback.modes!=r.state.modes
                            or feedback.values.shape[1:]!=r.state.values.shape[1:]):
                        raise ValueError('feedback and field units/axes mismatch')
                    if abs(feedback.time_si-r.state.time_si)>1e-14:
                        raise ValueError('wrong S_n / U_n time pairing')
                    if interval is None or interval[1]!=feedback.time_si or interval[0]>=interval[1]:
                        raise ValueError('missing positive source sampling interval (not gather time)')
            if any(not (set(r.sources.density_provenance['support_indices']) |
                        set(r.sources.projection_history_provenance['support_indices'])).issubset(ids)
                   for r in rs[:-1] if r.sources is not None):
                reject('source_history_outside_window_support');continue
            anchor=rs[0].coupled_anchor
            anchor_reasons=[]
            if anchor is None:
                anchor_reasons=['missing_restored_complete_coupled_anchor']
            elif config.warmup:
                anchor_reasons=['causal_coupled_warmup_replay_not_implemented']
            elif not set(anchor.support_indices).issubset(ids):
                anchor_reasons=['coupled_anchor_history_outside_window_support']
            else:
                try:anchor.validate(rs[0],plan,scheme)
                except ValueError as error:anchor_reasons=[str(error)]
            source_missing=[r.state.step_index for r in rs[:-1] if r.sources is None]
            eligibility={'field_supervised':{'eligible':True,'reasons':[]},
                'conditional_source':{'eligible':not source_missing,
                    'reasons':[] if not source_missing else ['missing_dual_source_records:'+str(source_missing)]},
                'self_generated_source':{'eligible':not anchor_reasons,'reasons':anchor_reasons,
                    'scope':'registered_small_reference_not_production_configuration'}}
            if config.required_qualification=='conditional_source' and source_missing:
                reject('missing_dual_source');continue
            if config.required_qualification=='self_generated_source' and anchor_reasons:
                reject('self_generated_source_rejected:'+anchor_reasons[0]);continue
            init=deepcopy(rs[0].initialization)
            if anchor is not None:init['recovered_anchor']=anchor.manifest()
            # An opaque marker path alone is not recoverability evidence.
            if anchor_reasons:init['coupled_gap']='; '.join(anchor_reasons)
            else:init.pop('coupled_gap',None)
            init.update(field_anchor=rs[0].provenance,anchor_index=ids[0],replay_steps=0,
                        ordinary_window_restart=False)
            out.append(NativeWindow(b.split,b.run,n,ids,tuple(deepcopy(rs[:config.warmup])),
                deepcopy(rs[config.warmup]),tuple(deepcopy(r.state) for r in rs[config.warmup+1:]),
                tuple(plan[config.warmup:]),plan_provenance,scheme,config,config.required_qualification,init,tuple(deepcopy(rs)),tuple(plan),eligibility,deepcopy(anchor) if not anchor_reasons else None))
    return WindowCollection(out,dict(excluded),rejections)


@dataclass
class NativeScaler:
    """Affine per-field/degree/harmonic; history has no centering or dt factor.

    RMS statistics over unique training records and nodes only. Positive unity
    scale for identically zero groups is an invertible scaling convention, not
    missing physical state. Not nonlinear positive-scalar normalization.
    """
    value_center: np.ndarray
    value_scale: np.ndarray
    history_scale: np.ndarray
    training_support: tuple
    unit_digest: str
    modes: tuple
    version: str='native-affine-train-v1'

    @classmethod
    def fit(cls, split_records):
        train=split_records.get('train',())
        if not train: raise ValueError('training records required for statistics')
        unique={(r.run,r.state.step_index):r for r in train}
        rs=list(unique.values())
        for r in rs: r.state.validate()
        first=rs[0].state
        if any(r.state.units!=first.units or r.state.modes!=first.modes or r.state.values.shape!=first.values.shape for r in rs):
            raise ValueError('incompatible training scaling groups')
        u=np.stack([r.state.values for r in rs]); d=np.stack([r.state.raw_deltas for r in rs])
        center=u.mean(axis=(0,4))[...,None]
        scale=np.sqrt(np.mean((u-center)**2,axis=(0,4)))[...,None]
        ds=np.sqrt(np.mean(d*d,axis=(0,4)))[...,None]
        from src.models.native_state import configuration_digest
        return cls(center,np.where(scale>0,scale,1.),np.where(ds>0,ds,1.),tuple(sorted(unique)),
            configuration_digest(first.units),first.modes)

    def _validate(self):
        if self.version!='native-affine-train-v1' or not self.training_support:
            raise ValueError('unsupported normalizer version/support')
        if self.value_center.shape[:2]!=(9,4) or self.value_center.shape[-1]!=1 or self.value_center.ndim!=4:
            raise ValueError('invalid scaling layout')
        for a in (self.value_scale,self.history_scale):
            if a.shape!=self.value_center.shape or not np.isfinite(a).all() or (a<=0).any():
                raise ValueError('invalid group scale')
        if not np.isfinite(self.value_center).all(): raise ValueError('invalid center')

    def encode(self,state):
        self._validate(); state.validate()
        from src.models.native_state import configuration_digest
        if configuration_digest(state.units)!=self.unit_digest or state.modes!=self.modes:
            raise ValueError('normalizer units/modes mismatch')
        if state.values.shape[:3]!=self.value_center.shape[:3]: raise ValueError('scaling axes mismatch')
        return (state.values-self.value_center)/self.value_scale,state.raw_deltas/self.history_scale

    def decode(self,u,d):
        self._validate()
        if u.shape!=d.shape or u.shape[:3]!=self.value_center.shape[:3]: raise ValueError('scaling axes mismatch')
        return u*self.value_scale+self.value_center,d*self.history_scale

    def save(self,path):
        import json
        self._validate()
        payload=dict(version=self.version,training_support=self.training_support,unit_digest=self.unit_digest,modes=self.modes,
            value_center=self.value_center.tolist(),value_scale=self.value_scale.tolist(),history_scale=self.history_scale.tolist())
        from .native_window_io import atomic_json
        atomic_json(path,payload)

    @classmethod
    def load(cls,path):
        import json
        p=json.loads(__import__('pathlib').Path(path).read_text())
        obj=cls(*(np.asarray(p[k]) for k in ('value_center','value_scale','history_scale')),
            tuple(tuple(x) for x in p['training_support']),p['unit_digest'],tuple(p['modes']),p['version'])
        obj._validate(); return obj
