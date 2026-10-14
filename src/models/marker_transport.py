"""Explicit local dyadic transport approximation, not JOREK's fixed particle_dt.

The density-gradient indicator is a causal screening heuristic, not an error
estimator or stability proof. Refinement is sticky for a particle lifetime:
near a density extremum the indicator can vanish despite stiff curvature.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class LocalSubcycling:
    trigger_indicator: float = .1
    max_step_indicator: float = 1.
    min_level: int = 6
    max_level: int = 10
    max_refined_particle_steps: int = 1000000
    boundary_exit: bool = False

    def __post_init__(self):
        if (not np.isfinite([self.trigger_indicator,self.max_step_indicator]).all()
            or not 0 < self.trigger_indicator <= self.max_step_indicator
            or type(self.min_level) is not int or type(self.max_level) is not int
            or not 1 <= self.min_level <= self.max_level <= 16
            or type(self.max_refined_particle_steps) is not int or self.max_refined_particle_steps <= 0
            or type(self.boundary_exit) is not bool):
            raise ValueError('invalid explicit local subcycling configuration')

    @property
    def configuration(self):
        fields=asdict(self)
        if not self.boundary_exit: fields.pop('boundary_exit')  # preserve v1 checkpoint fingerprint
        return dict(version=('local-direction-subcycling-v2_boundary_exit' if self.boundary_exit
                             else 'local-direction-subcycling-v1'),**fields,
            selector='dt_D_gradni_squared_over_floored_ni_squared',
            history='per_lifetime_sticky_level_previous_accepted_dt',
            approximation=('variable_direction_fractional_exterior_residence_not_native_fixed_step'
                           if self.boundary_exit else 'variable_interval_direction_extrapolation_not_native_fixed_step'))


@dataclass
class TransportHistory:
    previous_dt_si: np.ndarray
    level: np.ndarray
    lifetime: np.ndarray
    configuration: dict
    initialization: dict

    def validate(self, cloud, policy=None):
        n=len(cloud.weight)
        if not isinstance(self.configuration,dict):raise ValueError('invalid transport history configuration')
        try:
            registered=LocalSubcycling(**{name:(self.configuration.get(name,False) if name=='boundary_exit'
                                          else self.configuration[name]) for name in LocalSubcycling.__dataclass_fields__})
        except (KeyError,TypeError,ValueError) as error:
            raise ValueError('invalid transport history configuration') from error
        if self.configuration != registered.configuration:
            raise ValueError('unsupported transport history configuration')
        if (not isinstance(self.initialization,dict)
            or set(self.initialization)!={'kind','previous_dt_si'}
            or self.initialization['kind']!='explicit_native_fixed_interval_import'
            or not np.isscalar(self.initialization['previous_dt_si'])
            or not np.isfinite(self.initialization['previous_dt_si'])
            or self.initialization['previous_dt_si']<=0):
            raise ValueError('missing/invalid transport history initialization schema')
        if policy is not None and self.configuration != policy.configuration:
            raise ValueError('transport history configuration mismatch')
        for name in ('previous_dt_si','level','lifetime'):
            if not isinstance(getattr(self,name),np.ndarray) or getattr(self,name).shape != (n,):
                raise ValueError('invalid transport history slot shape')
        if (not np.isfinite(self.previous_dt_si).all() or (self.previous_dt_si<=0).any()
            or self.previous_dt_si.dtype != np.float64
            or self.level.dtype.kind not in 'iu' or self.lifetime.dtype.kind not in 'iu'
            or self.lifetime.dtype != cloud.i_life.dtype
            or (self.level<0).any() or (self.level>self.configuration['max_level']).any()
            or ((self.level>0)&(self.level<self.configuration['min_level'])).any()
            or not np.array_equal(self.lifetime,cloud.i_life)):
            raise ValueError('invalid transport history interval/level/identity')


def validate_transport_runtime(marker, *, expected_configuration=None):
    history=getattr(marker,'transport_history',None)
    if history is None:
        if expected_configuration is not None:
            raise ValueError('missing configured marker transport history')
        return
    if not isinstance(history,TransportHistory):raise ValueError('invalid marker transport history type')
    history.validate(marker.cloud)
    if expected_configuration is not None and history.configuration != expected_configuration:
        raise ValueError('marker transport history configuration mismatch')


def initialize_transport_history(state, policy, *, previous_dt_si):
    """Explicit native import assumption; never invoked implicitly on restore.

    Native v stores direction but not its sampling interval. Caller attests the
    fixed native interval. Newborn v is the existing birth velocity convention.
    """
    if not isinstance(policy,LocalSubcycling):raise ValueError('explicit local subcycling policy required')
    if getattr(state,'transport_history',None) is not None:
        raise ValueError('transport history already initialized; do not reset continuation')
    if (not np.isfinite(previous_dt_si) or previous_dt_si<=0
        or previous_dt_si != state.clock.particle_dt_si):
        raise ValueError('native transport history requires the declared fixed particle interval')
    state.cloud.validate()
    out=deepcopy(state);n=len(out.cloud.weight)
    out.transport_history=TransportHistory(np.full(n,float(previous_dt_si),dtype=np.float64),np.zeros(n,np.uint8),
        out.cloud.i_life.copy(),policy.configuration,
        dict(kind='explicit_native_fixed_interval_import',previous_dt_si=float(previous_dt_si)))
    return out


def push_variable_direction(x, previous, velocity, dt, previous_dt):
    from .marker_state import push_velocity
    previous_dt=np.asarray(previous_dt)
    if previous_dt.shape!=(len(x),) or not np.isfinite(previous_dt).all() or (previous_dt<=0).any():
        raise ValueError('positive per-marker velocity history intervals required')
    if np.all(previous_dt==dt):return push_velocity(x,previous,velocity,dt)
    speed=np.linalg.norm(velocity,axis=1);old_speed=np.linalg.norm(previous,axis=1)
    if (speed==0).any() or (old_speed==0).any() or (x[:,0]<=0).any():
        raise ValueError('JOREK velocity pusher requires nonzero current/previous speed and R>0')
    ratio=dt/previous_dt
    delta=((1+ratio[:,None]/2)*velocity/speed[:,None]
           -ratio[:,None]/2*previous/old_speed[:,None])*speed[:,None]*dt
    result=x.copy();result[:,:2]+=delta[:,:2];result[:,2]+=delta[:,2]/x[:,0]
    equal=previous_dt==dt
    if equal.any():result[equal]=push_velocity(x[equal],previous[equal],velocity[equal],dt)
    return result


def transport_batches(cloud, ids, *, time_si, base_dt, feedback, sample, history, policy, boundary_exit=None):
    """Yield accepted groups; caller updates physics before resuming the scheduler.

    Integer dyadic ticks avoid cumulative SI clock drift. Each particle reaches
    the original base endpoint or is lost; no rejected charge/deposit trials.
    """
    ticks=np.zeros(len(ids),np.int32);end=1<<policy.max_level
    while True:
        active=np.flatnonzero((ticks<end)&cloud.alive[ids])
        if not len(active):break
        tick=int(ticks[active].min());positions=active[ticks[active]==tick]
        batch=ids[positions];a=sample(time_si+base_dt*(tick/end),cloud,batch,feedback)
        rate=np.asarray(a.get('transport_rate_si'))
        if rate.shape!=(len(batch),) or not np.isfinite(rate).all() or (rate<0).any():
            raise ValueError('finite causal transport_rate_si required for local subcycling')
        score=base_dt*rate
        requested=np.zeros(len(batch),np.int32)
        flagged=score>policy.trigger_indicator
        requested[flagged]=np.maximum(policy.min_level,
            np.ceil(np.log2(np.maximum(score[flagged]/policy.max_step_indicator,1.))).astype(np.int32))
        required=np.maximum(history.level[batch],requested)
        levels=np.minimum(required,policy.max_level)
        plans=[]
        for level in np.unique(levels):
            selected=np.flatnonzero(levels==level);subset=batch[selected]
            h=base_dt/(1<<int(level))
            exits=({} if boundary_exit is None else boundary_exit(cloud,subset,a['velocity'][selected],h,
                history.previous_dt_si[subset]))
            ordinary=np.array([i for i in range(len(selected)) if i not in exits],dtype=np.int64)
            if (required[selected[ordinary]]>policy.max_level).any():
                raise ValueError('local transport refinement limit exceeded; candidate not committed')
            if len(ordinary):
                ix=selected[ordinary]
                plans.append((batch[ix],h,{name:np.asarray(value)[ix] for name,value in a.items()},positions[ix],int(level)))
            for i,event in exits.items():
                ix=selected[i:i+1];values={name:np.asarray(value)[ix] for name,value in a.items()}
                values['_boundary_exit']=dict(event=event,time_si=time_si+base_dt*(tick/end)+event.dt_si,
                    proposed_dt_si=h,required_level=int(required[ix[0]]),indicator=float(score[ix[0]]))
                plans.append((batch[ix],event.dt_si,values,positions[ix],int(level)))
        history.level[batch]=levels
        for subset,h,values,ix,level in plans:
            yield subset,h,values
            ticks[ix]+=end>>level
