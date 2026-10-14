"""Native observation-only perturbations, never an invented physical history."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from src.models.native_state import configuration_digest


@dataclass
class NoisyNativeObservation:
    values: torch.Tensor
    raw_deltas: torch.Tensor
    gauss_values: torch.Tensor
    gradients: torch.Tensor
    closure: object
    provenance: dict


def observe_native_noise(values,raw_deltas,*,evaluator,units,basis,coefficients,
                         constrained,indices,configuration,support_validator,closure):
    """Caller supplies correlated draws (and persists RNG), never noise on labels.

    The result is an observation, NOT a committed physical state. Existing
    model event Adapter does not silently substitute it for physical U/rawD.
    """
    c=configuration
    if (not c.get('scale_provenance') or not c.get('correlation_provenance') or not c.get('temporal_policy')
        or not math.isfinite(c['amplitude']) or c['amplitude']<0):
        raise ValueError('explicit noise units/scales/correlation/amplitude provenance required')
    like=values
    tensor=lambda a:torch.as_tensor(a,dtype=like.dtype,device=like.device)
    b=tensor(basis);draw=tensor(coefficients);scale=tensor(c['native_scales'])
    mask=torch.as_tensor(constrained,dtype=torch.bool,device=like.device)
    if (values.ndim!=4 or values.shape[:3]!=(9,4,1) or raw_deltas.shape!=values.shape
        or b.ndim!=5 or b.shape[1:]!=values.shape or draw.shape!=(len(b),)
        or scale.shape!=(9,4,1,1) or mask.shape not in (values.shape[1:],values.shape)
        or any(not torch.isfinite(a).all() for a in (values,raw_deltas,b,draw,scale)) or (scale<=0).any()
        or not callable(support_validator) or not callable(closure)):
        raise ValueError('complete native noise support, constraints, validator and closure required')
    ids=np.asarray(indices)
    if ids.shape!=(4,values.shape[-1]) or ids.dtype.kind not in 'iu':raise ValueError('shared DOF indices required for noise')
    _,inverse=np.unique(ids.ravel(),return_inverse=True)
    index=torch.as_tensor(inverse,dtype=torch.long,device=like.device)
    n=int(index.max())+1
    delta=(torch.einsum('r,rfdhn->fdhn',draw,b)*scale*c['amplitude']).reshape(9,-1)
    # One draw per unique physical DOF. Any constrained alias constrains all.
    shared=like.new_zeros((9,n)).index_add(1,index,delta)
    counts=like.new_zeros(n).index_add(0,index,like.new_ones(len(index)))
    if mask.shape==values.shape[1:]:mask=mask.unsqueeze(0).expand_as(values)
    blocked=like.new_zeros((9,n)).index_add(1,index,mask.reshape(9,-1).to(like.dtype))>0
    shared=shared/counts[None];shared=shared.masked_fill(blocked,0.)
    delta=shared[:,index].reshape_as(values)
    observed=values+delta
    if not bool(support_validator(observed)):raise ValueError('noise leaves declared physical support domain')
    evaluated=evaluator(observed,units=units)
    return NoisyNativeObservation(observed,raw_deltas.clone(),evaluated.values,evaluated.gradients,closure(evaluated),
        dict(mode='observation_only_not_committed_state',amplitude=c['amplitude'],
            scale_provenance=c['scale_provenance'],correlation_provenance=c['correlation_provenance'],
            temporal_policy=c['temporal_policy'],basis_digest=configuration_digest(b.detach().cpu().numpy()),
            native_scales=scale.detach().cpu().tolist(),constraints_digest=configuration_digest(mask.cpu().numpy()),
            shared_indices_digest=configuration_digest(ids),
            draws=draw.detach().cpu().tolist(),rawD_noise=False,teacher_replacement=False))


class NativeObservationNoise:
    """Static correlated basis; global CPU Torch RNG is checkpointed by session.

    Observations are consumed only by the network. Source/shard use committed U.
    Closure evaluation is owned by the actual joint-source field consumer.
    """
    def __init__(self,*,basis,constrained,indices,configuration,support_validator):
        from copy import deepcopy
        self.basis=np.array(basis,copy=True)
        self.constrained=np.array(constrained,copy=True)
        self.indices=np.array(indices,copy=True)
        self.spec=deepcopy(configuration)
        if not self.spec.get('support_provenance') or not callable(support_validator):
            raise ValueError('explicit noise support validator provenance required')
        self.support_validator=support_validator

    @property
    def configuration(self):
        return dict(version='native-network-observation-noise-v1',
            spec={k:np.asarray(v).tolist() if k=='native_scales' else v for k,v in self.spec.items()},
            basis=configuration_digest(self.basis),constraints=configuration_digest(self.constrained),
            indices=configuration_digest(self.indices),rng='session_global_CPU_torch',
            physical_source_and_shard='clean_committed_state',rawD_noise=False)

    def observe(self,values,raw_deltas,*,model):
        if model.feature_path is None:raise ValueError('noise requires compatible native evaluator')
        draws=torch.randn(len(self.basis),dtype=values.dtype,device='cpu')
        return observe_native_noise(values,raw_deltas,evaluator=model.feature_path.evaluator,units=model.units,
            basis=self.basis,coefficients=draws,constrained=self.constrained,indices=self.indices,
            configuration=self.spec,support_validator=self.support_validator,
            closure=lambda evaluated:dict(policy='recomputed_in_actual_joint_source_consumer'))
