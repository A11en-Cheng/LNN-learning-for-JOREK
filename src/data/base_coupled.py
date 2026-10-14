"""Causal base controls and sparse real-node labels; no physical scheduler."""
from dataclasses import dataclass,field,asdict
import math
import hashlib
from pathlib import Path
from copy import deepcopy
from src.models.native_state import configuration_digest

TIME_TOL_SI=1e-14
VALID_EVIDENCE={'saved','verified_reconstructed','causal_replayed'}


@dataclass(frozen=True)
class StepControl:
    start_si: float
    dt_si: float
    end_si: float
    origin: str
    provenance: str
    phase: str='ordinary'

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (self.start_si,self.dt_si,self.end_si))
            or self.dt_si<=0 or self.start_si<0 or not self.provenance
            or self.origin not in ('external_control','recorded_schedule_replay')
            or self.phase not in ('ordinary','registered_restart')):
            raise ValueError('explicit positive base control/origin/phase required')
        if self.phase=='ordinary' and (self.end_si<=self.start_si or abs(self.end_si-self.start_si-self.dt_si)>TIME_TOL_SI):
            raise ValueError('ordinary base interval must match physical dt')


@dataclass(frozen=True)
class Support:
    run: str
    first_si: float
    last_si: float
    role: str
    provenance: str

    def __post_init__(self):
        if (not self.run or not self.role or not self.provenance
            or not all(math.isfinite(x) for x in (self.first_si,self.last_si))
            or not 0<=self.first_si<=self.last_si):
            raise ValueError('invalid complete physical-time support')


@dataclass
class RecordedNode:
    time_si: float
    targets: dict
    provenance: str
    # Each object owns its time. Missing objects are absent, never fabricated.
    object_times_si: dict=field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.time_si) or self.time_si<0 or not self.provenance or not self.targets:
            raise ValueError('real recorded node/targets required')
        if set(self.targets)-{'field','marker_projection','shard_density','shard_positions','coupled_bundle'}:
            raise ValueError('unknown supervision object')
        if set(self.object_times_si)-set(self.targets) or any(not math.isfinite(t) or t<0 or t>self.time_si+TIME_TOL_SI
                for t in self.object_times_si.values()):
            raise ValueError('supervision objects require explicit causal event times')


@dataclass
class BaseSequence:
    run: str
    task: str
    controls: tuple
    nodes: tuple
    burnin_steps: int
    support: tuple
    split: str | None
    qualifications: dict
    parameters: dict

    @property
    def fingerprint(self):
        # Label contents deliberately excluded: labels are not state/control.
        return configuration_digest(self.manifest())

    def node_at(self,cursor):
        t=self.controls[cursor-1].end_si
        return next((n for n in self.nodes if abs(n.time_si-t)<=TIME_TOL_SI),None)

    def mask_at(self,cursor):
        node=self.node_at(cursor)
        return {k:bool(cursor>self.burnin_steps and node is not None and k in node.targets)
                for k in ('field','marker_projection','shard_density','shard_positions')}

    def manifest(self):
        return dict(schema='native-base-coupled-sequence-v1',run=self.run,task=self.task,
            controls=[asdict(c) for c in self.controls],burnin_steps=self.burnin_steps,
            burnin_time_si=sum(c.dt_si for c in self.controls[:self.burnin_steps]),
            burnin_record_count=sum(n.time_si<=self.controls[self.burnin_steps-1].end_si+TIME_TOL_SI for n in self.nodes) if self.burnin_steps else 0,
            prediction_time_si=sum(c.dt_si for c in self.controls[self.burnin_steps:]),
            nodes=[dict(time_si=n.time_si,objects=sorted(n.targets),provenance=n.provenance,
                object_times_si=n.object_times_si) for n in self.nodes],
            supervision_masks=[self.mask_at(i) for i in range(1,len(self.controls)+1)],
            support=[asdict(s) for s in self.support],split=self.split,
            qualifications=deepcopy(self.qualifications),parameters=deepcopy(self.parameters))


def make_sequence(*,run,task,controls,nodes,burnin_steps,support,blocks,
                  state_evidence,required_state,model_reasons,parameters):
    """Split blocks exist before this call; caches retain their full raw support.

    model_reasons comes from the current executing Adapter's validator, not a
    frozen list of production deficiencies. This function never runs a model.
    """
    controls=tuple(controls);nodes=tuple(deepcopy(nodes));support=tuple(support)
    if (not controls or task not in ('self_generated','conditional_source','autonomous')
        or type(burnin_steps)!=int or not 0<=burnin_steps<len(controls)
        or not support or not required_state):
        raise ValueError('explicit sequence/task/initialization evidence required')
    for before,after in zip(controls,controls[1:]):
        if abs(before.end_si-after.start_si)>TIME_TOL_SI:
            raise ValueError('base control sequence has a physical-time gap')
    if any(c.phase=='registered_restart' for c in controls[1:]):
        raise ValueError('internal restart requires a new explicitly initialized sequence')
    times=[n.time_si for n in nodes]
    if len(set(times))!=len(times):raise ValueError('duplicate supervision times')
    if any(not any(abs(t-c.end_si)<=TIME_TOL_SI for c in controls) for t in times):
        raise ValueError('recorded node is not an actual base endpoint; no pseudo-label interpolation')
    endpoints=[t for c in controls for t in (c.start_si,c.end_si)]
    total_support=(*support,Support(run,min(endpoints),max(endpoints),'prediction_and_burnin','explicit_controls'))
    candidates=[b for b in blocks if b.run==run and all(s.run==b.run and b.first_si<=s.first_si+TIME_TOL_SI
                and s.last_si<=b.last_si+TIME_TOL_SI for s in total_support)]
    # Overlapping split declarations are malformed, not a reason to silently pick one.
    for i,a in enumerate(blocks):
        for b in blocks[i+1:]:
            if a.run==b.run and a.role!=b.role and max(a.first_si,b.first_si)<=min(a.last_si,b.last_si):
                raise ValueError('overlapping split blocks')
    if len(candidates)>1:raise ValueError('ambiguous split assignment')
    data_reasons=[f'{key}:{state_evidence.get(key,"missing")}' for key in required_state
                  if state_evidence.get(key) not in VALID_EVIDENCE]
    control_reasons=(['recorded_control_not_autonomous'] if task=='autonomous'
                     and any(c.origin!='external_control' for c in controls) else [])
    q={name:dict(eligible=not reasons,reasons=list(reasons)) for name,reasons in
       [('data_state',data_reasons),('model_support',model_reasons),('control_source',control_reasons),
        ('split_isolation',[] if candidates else ['complete_support_outside_split'])]}
    q['combined']=dict(eligible=all(v['eligible'] for v in q.values()),
        reasons=[f'{k}:{r}' for k,v in q.items() for r in v['reasons']])
    return BaseSequence(run,task,controls,nodes,burnin_steps,total_support,
                        candidates[0].role if candidates else None,q,deepcopy(parameters))


def read_recorded_node(field_path,*,gamma,projection_path=None,shard_identity=None):
    """Read labels only, never manufacture a recoverable runtime from a field.

    projection_path is explicit: its event can precede the fluid label event.
    Missing raw projection history/identity stays missing, not zero.
    """
    import h5py
    import numpy as np
    from src.models.native_state import NativeUnits
    field_path=Path(field_path)
    def source(path):
        return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    with h5py.File(field_path,'r') as f:
        scalar=lambda k:float(np.ravel(f[k][()])[0])
        units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),gamma)
        units.validate()
        time_si=scalar('t_now')*units.t_norm_si
        values=np.asarray(f['values'][()]);raw=np.asarray(f['deltas'][()])
        if values.shape[:3]!=(9,4,1) or raw.shape!=values.shape or not np.isfinite(values).all() or not np.isfinite(raw).all():
            raise ValueError('recorded label requires complete finite field and saved rawD')
        targets={'field':dict(values=values,raw_deltas=raw,units=units,time_si=time_si)}
        metadata=dict(field_index=int(scalar('index_now')),field_tstep_ju=scalar('tstep'),
            field_source=source(field_path),missing=[],shard_arrays_saved=[k for k in f if k.startswith('spi_')])
        if shard_identity is None:
            metadata['missing'].append('fixed_shard_identity_not_supplied')
        else:
            if not shard_identity.get('provenance') or 'ids' not in shard_identity or 'alive' not in shard_identity:
                raise ValueError('explicit fixed shard identity/liveness evidence required')
            r=np.asarray(f['spi_R_arr'][()]);z=np.asarray(f['spi_Z_arr'][()]);phi=np.asarray(f['spi_phi_arr'][()])
            pos=np.column_stack([r*np.cos(phi),r*np.sin(phi),z])
            if len(shard_identity['ids'])!=len(r) or len(set(shard_identity['ids']))!=len(r):
                raise ValueError('fixed shard identities do not cover recorded slots')
            targets['shard_positions']=dict(ids=list(shard_identity['ids']),positions_m=pos,
                alive=list(shard_identity['alive']),identity_provenance=shard_identity['provenance'],
                coordinate_transform='(R*cos(phi),R*sin(phi),Z)')
    object_times={'field':time_si}
    if 'shard_positions' in targets:object_times['shard_positions']=time_si
    if projection_path is not None:
        with h5py.File(projection_path,'r') as f:
            p=np.asarray(f['values'][()])
            if p.shape!=(5,*values.shape[1:]) or not np.isfinite(p).all():raise ValueError('full native projection label required')
            targets['marker_projection']=p
            object_times['marker_projection']=float(np.ravel(f['t_now'][()])[0])
            metadata['projection_rawD_status']='saved' if 'deltas' in f else 'unavailable'
            metadata['projection_source']=source(projection_path)
    metadata['missing'].append('independent_shard_density_label_not_exported')
    return RecordedNode(time_si,targets,str(field_path),object_times),metadata


def burnin_comparison_candidates(controls,nodes,*,prediction_cursor,forecast_steps,start_cursors):
    """Design candidates only. Never choose an optimal length or create initial states."""
    if (type(prediction_cursor)!=int or type(forecast_steps)!=int or forecast_steps<1
        or not 0<=prediction_cursor<len(controls) or prediction_cursor+forecast_steps>len(controls)
        or not start_cursors or len(set(start_cursors))!=len(start_cursors)
        or any(type(i)!=int or not 0<=i<=prediction_cursor for i in start_cursors)):
        raise ValueError('explicit burn-in candidates within one fixed forecast required')
    prediction_start=controls[prediction_cursor].start_si
    horizon=sum(c.dt_si for c in controls[prediction_cursor:prediction_cursor+forecast_steps])
    rows=[]
    for start in start_cursors:
        time=controls[start].start_si
        rows.append(dict(start_cursor=start,prediction_cursor=prediction_cursor,
            prediction_start_si=prediction_start,prediction_time_si=horizon,
            burnin_steps=prediction_cursor-start,burnin_time_si=sum(c.dt_si for c in controls[start:prediction_cursor]),
            burnin_record_count=sum(time+TIME_TOL_SI<n.time_si<=prediction_start+TIME_TOL_SI for n in nodes),
            initialization_requirement='certified_state_or_causal_cache_at_burnin_start',
            selection_status='candidate_only_no_training_evidence'))
    return rows
