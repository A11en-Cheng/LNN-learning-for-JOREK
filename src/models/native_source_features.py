"""Explicit dual-source conditioning for joint native learning (not forcing).

Full native coefficient identities survive alongside pooled Gauss observations.
Only the field decoder/time adapter owns the learned response integration.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from .native_state import configuration_digest

SOURCE_SCHEMA=('marker_c1','marker_c2','marker_c3','marker_c4','marker_c5',
               'raw_projection_D1','raw_projection_D2','raw_projection_D3','raw_projection_D4','raw_projection_D5',
               'shard_background_mass_rate','shard_impurity_mass_rate')
CLOSURE_SCHEMA=('mean_charge','mean_square_charge','alpha_e','electron_density','zeff',
                'electron_pressure','electron_capacity','charge_gradient_R','charge_gradient_Z',
                'pressure_gradient_R','pressure_gradient_Z','scaled_charge_history','interpreted_energy_rate','projection_history_pressure_increment')


@dataclass(frozen=True)
class FieldEventContext:
    """Caller-owned current state clock; never derived from source timestamps.

    ReferenceRollout supplies this from its committed clocks. Conditional Torch
    callers must supply a provenance-bound current state, not a future label.
    """
    field_time_si: float
    solve_time_si: float
    phase: str
    provenance: str
    gather_time_si: float | None = None

    def validate_source(self, source, *, dt_si, units):
        if (not isinstance(source,SourceEventTime) or not self.provenance
                or not np.isfinite([self.field_time_si,self.solve_time_si,dt_si]).all()):
            raise ValueError('explicit field/source event context required')
        tol=max(1e-14,units.t_norm_si*1e-7)
        if abs(source.event_time_si-self.solve_time_si)>tol:
            raise ValueError('source event is stale/future relative to committed field event')
        if self.phase in ('ordinary','zero_time_probe'):
            if (self.gather_time_si is None or not np.isfinite(self.gather_time_si) or self.gather_time_si<=0
                    or source.gather_time_si is None or not np.isfinite(source.gather_time_si)
                    or abs(source.gather_time_si-self.gather_time_si)>tol):
                raise ValueError('source actual gather disagrees with field event context')
            if abs(self.field_time_si-self.solve_time_si)>tol or source.phase!='ordinary':
                raise ValueError('ordinary field/source event phase mismatch')
            if (self.phase=='ordinary' and dt_si<=0) or (self.phase=='zero_time_probe' and dt_si!=0):
                raise ValueError('zero probe and ordinary positive event are distinct')
        elif self.phase=='registered_restart':
            if (self.gather_time_si not in (None,0.) or dt_si<=0 or self.solve_time_si>self.field_time_si+tol
                    or source.phase!='registered_restart_zero_projection'):
                raise ValueError('registered restart field/source event mismatch')
        else:
            raise ValueError('unsupported field event phase; model resume is not restart')


@dataclass(frozen=True)
class SourceEventTime:
    event_time_si: float
    gather_time_si: float | None
    phase: str='ordinary'
    units: object=None  # explicit common native normalization, never inferred from numeric payload

    def validate(self,p,d):
        if not np.isfinite(self.event_time_si):raise ValueError('explicit finite source event time required')
        if self.phase=='ordinary':
            if self.gather_time_si is None or not np.isfinite(self.gather_time_si) or self.gather_time_si<=0:
                raise ValueError('ordinary source event requires positive actual gather time')
        elif self.phase=='registered_restart_zero_projection':
            if self.gather_time_si not in (None,0.) or p.any() or d.any():
                raise ValueError('registered restart requires zero projection/history and no gather')
        else:raise ValueError('unsupported source event phase')


@dataclass(frozen=True)
class SourceFeatureScales:
    coefficients: object            # [12,4], respective native coefficient units
    gauss_values: object            # [12], respective native value units
    gauss_gradients: object         # [12,2], respective native value units / m
    closure: object                 # [14], fixed CLOSURE_SCHEMA units
    time_ju: object                 # [2], gather/event time scales, in JU
    provenance: str
    version: str='native-source-linear-scales-v1'
    time_origin_si: float=0.
    time_origin_provenance: str=''

    @classmethod
    def physical_reference(cls):
        """Explicit unit references, NOT fitted statistics or defaults on missing data."""
        return cls(np.ones((12,4)),np.ones(12),np.ones((12,2)),np.ones(14),np.ones(2),
                   'explicit_physical_reference_native_JU_m_v1')

    @classmethod
    def anchored_physical_reference(cls,*,units,origin_si,gather_scale_si,event_scale_si,origin_provenance):
        units.validate()
        if (not np.isfinite([origin_si,gather_scale_si,event_scale_si]).all()
            or min(gather_scale_si,event_scale_si)<=0 or not origin_provenance):
            raise ValueError('explicit persistent physical clock origin/scales required')
        base=cls.physical_reference()
        from dataclasses import replace
        result=replace(base,time_ju=np.array([gather_scale_si,event_scale_si])/units.t_norm_si,
            version='native-source-linear-scales-v2_anchored_time',
            time_origin_si=float(origin_si),time_origin_provenance=origin_provenance)
        result.validate();return result

    def validate(self):
        if not np.isfinite(self.time_origin_si):raise ValueError('finite persistent time origin required')
        if self.version=='native-source-linear-scales-v1' and (self.time_origin_si!=0 or self.time_origin_provenance):
            raise ValueError('legacy time scaling cannot change origin')
        if self.version=='native-source-linear-scales-v2_anchored_time' and not self.time_origin_provenance:
            raise ValueError('persistent time origin provenance required')
        if self.version not in ('native-source-linear-scales-v1','native-source-linear-scales-v2_anchored_time') or not self.provenance.startswith('explicit_physical_reference_'):
            raise ValueError('supported explicit physical source scaling provenance/version required; no implicit statistics')
        for name,shape in (('coefficients',(12,4)),('gauss_values',(12,)),('gauss_gradients',(12,2)),
                           ('closure',(14,)),('time_ju',(2,))):
            a=np.asarray(getattr(self,name))
            if a.shape!=shape or not np.isfinite(a).all() or (a<=0).any():raise ValueError('invalid source scale '+name)


class JointSourceEncoding(nn.Module):
    """48 full coefficients + 50 pooled Gauss/closure + 3 physical clock features.

    Pooling is lossy positive R|J|-weighted partition; it is feature aggregation,
    not weak assembly, capacity inversion, or replacement of native state.
    """
    def __init__(self,consumer,evaluator,*,scales,hidden):
        super().__init__();scales.validate()
        # Consumer is owned/registered by CoupledNativeField.response, once.
        object.__setattr__(self,"consumer",consumer)
        # Do not register evaluator twice; store only the immutable pool geometry.
        for name in ('vertex','pool_weights','pool_norm'):
            self.register_buffer(name,getattr(evaluator,name).clone())
        for name in ('coefficients','gauss_values','gauss_gradients','closure','time_ju'):
            self.register_buffer('scale_'+name,torch.as_tensor(np.array(getattr(scales,name),copy=True),dtype=torch.float64))
        self.scaling_provenance=scales.provenance;self.scaling_version=scales.version
        self.time_origin_si=float(scales.time_origin_si);self.time_origin_provenance=scales.time_origin_provenance
        self.projection=nn.Linear(101,hidden,dtype=torch.float64)

    @property
    def checkpoint_configuration(self):
        config=dict(version='native-joint-source-encoding-v1',source_schema=SOURCE_SCHEMA,closure_schema=CLOSURE_SCHEMA,
            units=asdict(self.consumer.units),consumer=self.consumer.checkpoint_configuration,
            scaling_version=self.scaling_version,scaling_provenance=self.scaling_provenance,
            scales={k:getattr(self,'scale_'+k).cpu().tolist() for k in ('coefficients','gauss_values','gauss_gradients','closure','time_ju')},
            pool=configuration_digest({k:getattr(self,k).cpu().numpy() for k in ('vertex','pool_weights','pool_norm')}),
            dimensions={'native':48,'gauss_pooled':50,'clock':3,'encoded':self.projection.out_features},
            response_semantics='all_response_learned_no_physical_increment',
            field_teacher_policy='external_paired_U_rawD_preserve_memory',source_teacher_policy='external_explicit_or_self_generated',
            approximation='positive_R_absJ_partition_pool_plus_full_coefficients')
        if self.scaling_version=='native-source-linear-scales-v2_anchored_time':
            config.update(version='native-joint-source-encoding-v2_anchored_time',
                time_origin_si=self.time_origin_si,time_origin_provenance=self.time_origin_provenance)
        return config

    def forward(self,u,p,d,shard,*,source_time,time,dt_si):
        from .native_coupling import NativeDensitySource
        if not isinstance(shard,NativeDensitySource) or shard.units!='model502_mass_density_per_JU' or not shard.provenance:
            raise ValueError('explicit independent shard mass source with units/provenance required')
        # Validate before cat/clock checks so None never becomes a zero source.
        pv,pg=self.consumer.evaluate(p,5);dv,dg=self.consumer.evaluate(d,5)
        sv,sg=self.consumer.evaluate(shard.values,2)
        if not isinstance(source_time,SourceEventTime):raise ValueError('explicit source time/gather metadata required')
        if source_time.units!=self.consumer.units:raise ValueError('source normalization units mismatch or missing')
        source_time.validate(p,d)
        c=self.consumer(u,p,d,time=time,dt_si=dt_si)
        native=torch.cat([p,d,shard.values],0)
        identity=(native/self.scale_coefficients[:,:,None,None]).permute(3,0,1,2).reshape(-1,48)
        values=torch.cat([pv,dv,sv],-1)/self.scale_gauss_values
        gradients=(torch.cat([pg,dg,sg],-2)/self.scale_gauss_gradients).flatten(-2)
        scalar=torch.stack([c.mean_charge,c.mean_square_charge,c.alpha_e,c.ne,c.zeff,c.pressure,c.electron_capacity,
            c.mean_charge_gradient[:,:,0],c.mean_charge_gradient[:,:,1],c.pressure_gradient[:,:,0],
            c.pressure_gradient[:,:,1],c.charge_history,c.energy_rate,c.projection_history_value],-1)/self.scale_closure
        gauss=torch.cat([values,gradients,scalar],-1)
        local=torch.einsum('eqv,eqf->evf',self.pool_weights,gauss)
        pooled=local.new_zeros(len(self.pool_norm),50).index_add(0,self.vertex.flatten(),local.flatten(0,1))/self.pool_norm[:,None]
        clock=u.new_tensor([(source_time.gather_time_si or 0.)/self.consumer.units.t_norm_si,
            (source_time.event_time_si-self.time_origin_si)/self.consumer.units.t_norm_si])/self.scale_time_ju
        clock=torch.cat([clock,u.new_tensor([float(source_time.phase=='registered_restart_zero_projection')])])
        features=torch.cat([identity,pooled,clock.expand(len(identity),3)],-1)
        if not torch.isfinite(features).all():raise ValueError('nonfinite normalized source features')
        return features,torch.tanh(self.projection(features))
