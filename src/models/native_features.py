"""H=1 compatible full-cubic observations and explicit grouped conditions.

Static B/G are compiled from the verified native reference. All dynamic field
operations are Torch contractions; pooled observations do not replace U/rawD.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from .native_gauss import NativeElementGeometry,GaussRule,ToroidalSampling,evaluate_native_element
from .native_state import configuration_digest

BASIS_VERSION='jorek-cubic-native-gauss4-h1-v1'


@dataclass
class TorchGaussFields:
    values: torch.Tensor          # [E,16,9], native field units
    gradients: torch.Tensor       # [E,16,9,2], native field units / m, R then Z


class TorchNativeGauss(nn.Module):
    def __init__(self,coordinates,size,vertex,*,units):
        super().__init__(); units.validate(); self.units=units
        x=np.asarray(coordinates); sz=np.asarray(size); v=np.asarray(vertex)
        if (x.ndim!=4 or x.shape[:3]!=(2,4,1) or not np.isfinite(x).all()
            or v.ndim!=2 or v.shape[0]!=4 or v.shape[1]<1 or v.dtype.kind not in 'iu'
            or (v<0).any() or (v>=x.shape[-1]).any()
            or sz.shape!=(4,4,v.shape[1]) or not np.isfinite(sz).all()
            or any(len(set(ids))!=4 for ids in v.T)):
            raise ValueError('requires full axisymmetric cubic geometry and zero-based native vertex order')
        self.n_nodes=x.shape[-1]
        self.geometry_digest=configuration_digest(dict(x=x,size=sz,vertex=v,units=units,basis=BASIS_VERSION))
        probe=np.eye(16).reshape(16,4,1,4)
        basis=[]; gradient=[]; coords=[]; volumes=[]; pools=[]
        for e,ids in enumerate(v.T):
            g=evaluate_native_element(NativeElementGeometry(x[:,:,:,ids],sz[:,:,e],3),probe,
                rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1))
            basis.append(g.values.reshape(16,4,4).transpose(0,2,1))
            gradient.append(np.stack([g.d_x,g.d_y],-1).reshape(16,4,4,2).transpose(0,2,1,3))
            coords.append(g.coordinates)
            volume=g.weights*np.abs(g.jacobian)*g.coordinates[:,0]
            volumes.append(volume)
            # Positive partition of unity for FEATURE POOLING only. This is
            # not a surrogate basis for U or a weak load / inverse FEM solve.
            s,t=g.s,g.t
            hat=np.stack([(1-s)*(1-t),s*(1-t),s*t,(1-s)*t],-1)
            pools.append(volume[:,None]*hat)
        arrays=dict(node_positions=x[:,0,0,:].T,vertex=v.T.astype(np.int64),basis=np.array(basis),gradient_basis=np.array(gradient),
            coordinates=np.array(coords),volume=np.array(volumes),pool_weights=np.array(pools))
        norm=np.zeros(self.n_nodes)
        np.add.at(norm,v.T.ravel(),arrays['pool_weights'].sum(1).ravel())
        if not np.isfinite(norm).all() or (norm<=0).any(): raise ValueError('uncovered node or invalid feature aggregation weights')
        arrays['pool_norm']=norm
        for name,a in arrays.items(): self.register_buffer(name,torch.from_numpy(a.copy()))

    @property
    def checkpoint_configuration(self):
        return dict(basis=BASIS_VERSION,geometry_digest=self.geometry_digest,units=asdict(self.units),
            modes=[0],aggregation='positive_R_absJ_gauss_bilinear_partition_feature_pool_v1')

    def forward(self,values,*,units):
        if units!=self.units: raise ValueError('native feature units mismatch')
        if (not isinstance(values,torch.Tensor) or values.shape!=(9,4,1,self.n_nodes)
            or values.dtype!=self.basis.dtype or values.device!=self.basis.device
            or not torch.isfinite(values).all()):
            raise ValueError('requires finite full Torch [9,4,1,N] matching geometry dtype/device')
        a=values[:,:,0,self.vertex].permute(2,3,1,0)
        val=torch.einsum('eqvd,evdf->eqf',self.basis,a)
        grad=torch.einsum('eqvdk,evdf->eqfk',self.gradient_basis,a)
        if not torch.isfinite(val).all() or not torch.isfinite(grad).all(): raise ValueError('nonfinite Gauss features')
        return TorchGaussFields(val,grad)

    def aggregate(self,features):
        if (features.ndim!=3 or features.shape[:2]!=self.volume.shape
            or features.dtype!=self.basis.dtype or features.device!=self.basis.device
            or not torch.isfinite(features).all()): raise ValueError('invalid Gauss features for pooling')
        local=torch.einsum('eqv,eqf->evf',self.pool_weights,features)
        pooled=features.new_zeros(self.n_nodes,features.shape[-1]).index_add(0,self.vertex.flatten(),local.reshape(-1,features.shape[-1]))
        return pooled/self.pool_norm[:,None]


PARAMETER_SCHEMA=(('n_spi','scenario','1'),('pellet_density','scenario','10^20 m^-3'),
                  ('eta','physical_coefficient','m^2 / JOREK time'))
PARAMETER_VERSION='native-feature-parameters-v1'


@dataclass(frozen=True)
class FeatureParameters:
    rows: tuple
    excluded_names: tuple
    scenario_enabled: bool
    physics_enabled: bool
    version: str=PARAMETER_VERSION


def select_feature_parameters(rows,*,scenario_enabled,physics_enabled):
    """Explicit Adapter from the wider audit inventory to the fixed schema.

    Known nonconsumed inventory entries are explicitly excluded; unknown names
    fail. Metadata validates eligibility and is never encoded numerically.
    """
    if type(scenario_enabled) is not bool or type(physics_enabled) is not bool:
        raise ValueError('parameter group flags must be explicit booleans')
    from src.data.native_run import _GROUP_BY_PARAMETER
    selected=[]; excluded=[]; seen=set()
    wanted={
        n for n,g,_ in PARAMETER_SCHEMA if (scenario_enabled if g=='scenario' else physics_enabled)}
    for p in rows:
        n=p['name']
        if n.split('[')[0] not in _GROUP_BY_PARAMETER: raise ValueError('unknown parameter name')
        if n not in wanted: excluded.append(n); continue
        if n in seen: raise ValueError('duplicate required parameter')
        seen.add(n); selected.append(p)
    if seen!=wanted: raise ValueError('missing required feature parameters')
    return FeatureParameters(tuple(selected),tuple(excluded),scenario_enabled,physics_enabled)


def _parameter_values(parameters,like):
    if not isinstance(parameters,FeatureParameters) or parameters.version!=PARAMETER_VERSION:
        raise ValueError('fixed feature parameter schema required')
    mapping={p['name']:p for p in parameters.rows}
    expected=[(n,g,u) for n,g,u in PARAMETER_SCHEMA if (parameters.scenario_enabled if g=='scenario' else parameters.physics_enabled)]
    if len(mapping)!=len(parameters.rows) or set(mapping)!={n for n,_,_ in expected}:
        raise ValueError('unknown/missing/duplicate feature parameter')
    from src.data.native_parameters import CONSUMER_BY_BASE
    values={}
    for n,g,unit in expected:
        p=mapping[n]
        if (p.get('group')!=g or p.get('unit')!=unit or p.get('status')!='declared'
            or p.get('consumer')!=CONSUMER_BY_BASE[n]
            or p.get('available_at')!='initialization'
            or p.get('provenance',{}).get('kind') not in ('h5','namelist')
            or not p.get('provenance',{}).get('source')):
            raise ValueError('unconfirmed feature parameter source/unit/group')
        a=torch.as_tensor(p['value'],dtype=like.dtype,device=like.device)
        if a.ndim!=0 or not torch.isfinite(a) or a<0 or (n!='eta' and a<=0):
            raise ValueError('invalid feature parameter value/range')
        if n=='n_spi' and a!=a.round(): raise ValueError('n_spi must be a positive integer count')
        values[n]=a
    return values


@dataclass
class NativeFeatures:
    gauss: TorchGaussFields
    scaled_values: torch.Tensor
    scaled_gradients: torch.Tensor
    node_observations: torch.Tensor
    scenario: torch.Tensor | None
    magnetic: torch.Tensor | None
    raw_fluid_deltas: torch.Tensor


class NativeFeaturePath(nn.Module):
    """Single compatible feature Interface. No feedback or time substitution.

    Scenario uses a separate 2->4 encoder. Physical eta enters ONLY the
    declared eta*grad(psi) observation, not scenario FiLM or arbitrary concat.
    This observation is a learning feature, NOT a JOREK resistive RHS/flux.
    """
    def __init__(self,evaluator,statistics):
        super().__init__(); self.evaluator=evaluator
        if statistics.get('version')!='native-gauss-train-statistics-v1' or statistics.get('geometry')!=evaluator.geometry_digest:
            raise ValueError('feature statistics version/geometry/units mismatch')
        if type(statistics.get('scenario_enabled')) is not bool or type(statistics.get('physics_enabled')) is not bool:
            raise ValueError('parameter group flags must be explicit booleans')
        self.scenario_enabled=statistics['scenario_enabled']; self.physics_enabled=statistics['physics_enabled']
        self.training_support=statistics['training_support']
        if not self.training_support: raise ValueError('training support required')
        shapes={'value_center':(9,),'value_scale':(9,),'gradient_scale':(9,2),
                'scenario_scale':(2,),'magnetic_scale':(2,)}
        for n,shape in shapes.items():
            a=torch.as_tensor(statistics[n],dtype=evaluator.basis.dtype,device=evaluator.basis.device)
            if a.shape!=shape or not torch.isfinite(a).all() or (n!='value_center' and (a<=0).any()):
                raise ValueError('invalid feature statistics')
            self.register_buffer(n,a.clone())
        self.scenario_encoder=nn.Linear(2,4,dtype=evaluator.basis.dtype,device=evaluator.basis.device) if self.scenario_enabled else None

    @classmethod
    def fit(cls,evaluator,windows,*,scenario_enabled,physics_enabled):
        unique={}
        for w in windows:
            if w.split=='train':
                for r in w.support_records: unique[r.run,r.state.step_index]=r
        if not unique: raise ValueError('training windows required for feature statistics')
        total=evaluator.volume.sum()*len(unique)
        sums=torch.zeros(9,dtype=evaluator.basis.dtype,device=evaluator.basis.device)
        sumsq=torch.zeros_like(sums); g2=torch.zeros(9,2,dtype=sums.dtype,device=sums.device)
        p2=torch.zeros(2,dtype=sums.dtype,device=sums.device); m2=torch.zeros_like(p2)
        with torch.no_grad():
            for r in unique.values():
                r.state.validate()
                a=torch.as_tensor(r.state.values,dtype=sums.dtype,device=sums.device)
                g=evaluator(a,units=r.state.units)
                weight=evaluator.volume
                sums+=(g.values*weight[...,None]).sum((0,1))
                sumsq+=(g.values.square()*weight[...,None]).sum((0,1))
                g2+=(g.gradients.square()*weight[...,None,None]).sum((0,1))
                pv=_parameter_values(select_feature_parameters(r.parameters,scenario_enabled=scenario_enabled,physics_enabled=physics_enabled),a)
                if scenario_enabled: p2+=torch.stack([pv['n_spi'],pv['pellet_density']]).square()
                if physics_enabled: m2+=((pv['eta']*g.gradients[:,:,0,:]).square()*weight[...,None]).sum((0,1))
            center=sums/total
            scale=(sumsq/total-center.square()).clamp_min(0).sqrt()
            gs=(g2/total).sqrt()
            ps=(p2/len(unique)).sqrt() if scenario_enabled else torch.ones_like(p2)
            ms=(m2/total).sqrt() if physics_enabled else torch.ones_like(m2)
        positive=lambda a:torch.where(a>0,a,torch.ones_like(a)).tolist()
        stats=dict(version='native-gauss-train-statistics-v1',geometry=evaluator.geometry_digest,
            training_support=[list(k) for k in sorted(unique)],scenario_enabled=scenario_enabled,physics_enabled=physics_enabled,
            value_center=center.tolist(),value_scale=positive(scale),gradient_scale=positive(gs),
            scenario_scale=positive(ps),magnetic_scale=positive(ms))
        return cls(evaluator,stats)

    @property
    def checkpoint_configuration(self):
        return dict(version='native-compatible-features-v1',evaluation=self.evaluator.checkpoint_configuration,
            statistics=self.statistics(),parameter_schema=PARAMETER_SCHEMA,parameter_version=PARAMETER_VERSION,
            disabled_groups=[g for g,on in [('scenario',self.scenario_enabled),('physical_eta',self.physics_enabled)] if not on],
            physical_feature='eta_times_native_psi_RZ_gradient_not_FEM_flux')

    def statistics(self):
        return dict(version='native-gauss-train-statistics-v1',geometry=self.evaluator.geometry_digest,
            training_support=self.training_support,scenario_enabled=self.scenario_enabled,physics_enabled=self.physics_enabled,
            **{n:getattr(self,n).cpu().tolist() for n in ('value_center','value_scale','gradient_scale','scenario_scale','magnetic_scale')})

    def save_statistics(self,path):
        from src.data.native_window_io import atomic_json
        atomic_json(path,self.statistics())

    @classmethod
    def from_statistics(cls,evaluator,path):
        import json
        from pathlib import Path
        return cls(evaluator,json.loads(Path(path).read_text()))

    def restore_values(self,a): return a*self.value_scale+self.value_center
    def restore_gradients(self,a): return a*self.gradient_scale

    def forward(self,values,raw_fluid_deltas,*,units,parameters):
        if not isinstance(values,torch.Tensor): raise ValueError('full Torch field required')
        if (not isinstance(raw_fluid_deltas,torch.Tensor) or raw_fluid_deltas.shape!=values.shape
            or raw_fluid_deltas.dtype!=values.dtype or raw_fluid_deltas.device!=values.device
            or not torch.isfinite(raw_fluid_deltas).all()): raise ValueError('full raw fluid history required')
        g=self.evaluator(values,units=units)
        if not isinstance(parameters,FeatureParameters): raise ValueError('fixed feature parameter schema required')
        if parameters.scenario_enabled!=self.scenario_enabled or parameters.physics_enabled!=self.physics_enabled:
            raise ValueError('feature parameter group configuration mismatch')
        p=_parameter_values(parameters,values)
        v=(g.values-self.value_center)/self.value_scale
        grad=g.gradients/self.gradient_scale
        observations=self.evaluator.aggregate(torch.cat([v,grad.flatten(-2)],-1))
        scenario=None
        if self.scenario_enabled:
            scaled=torch.stack([p['n_spi'],p['pellet_density']])/self.scenario_scale
            if not torch.isfinite(scaled).all(): raise ValueError('nonfinite scaled scenario parameters')
            scenario=torch.tanh(self.scenario_encoder(scaled))
        magnetic=self.evaluator.aggregate(p['eta']*g.gradients[:,:,0,:]/self.magnetic_scale) if self.physics_enabled else None
        return NativeFeatures(g,v,grad,observations,scenario,magnetic,raw_fluid_deltas)
