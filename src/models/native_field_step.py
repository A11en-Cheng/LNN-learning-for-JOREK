"""Joint full-coefficient learning step, not a JOREK matrix solver.

H=1 only. Node P1/lumped stencil in latent space is a registered spatial
learning approximation; it is not the native cubic FEM operator. No marker
forcing / capacity / closure consumer is implemented here.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .native_state import NativeUnits, configuration_digest
from .marker_state import fluid_time_coefficients
from .implicit_stencil import ImplicitStencilLayer, build_term_operators


@dataclass
class TorchFieldStepResult:
    values: torch.Tensor
    fluid_deltas: torch.Tensor  # raw post-update history, not values-old_values
    increment: torch.Tensor    # applied native coefficient increment this step
    memory: torch.Tensor | None = None  # field temporal memory, never rawD/marker state
    source_features: torch.Tensor | None = None  # conditioning diagnostic, never a physical increment
    boundary_features: torch.Tensor | None = None  # applicability/parameters, not physical loads


class NativeUpdateMapping(nn.Module):
    """Supported update_values rows: unconstrained, treat_axis=False, H=1.

    Hanging-node interpolation and active axis transformations are refused.
    This is an update mapping, not model502 physical boundary assembly.
    """
    def __init__(self, indices, *, constrained, treat_axis, keep_n0_const, jeccd=False, boundary_rows=None):
        super().__init__()
        a=np.asarray(indices)
        if a.ndim!=2 or a.shape[0]!=4 or a.dtype.kind not in 'iu' or (a<=0).any():
            raise ValueError('requires positive native degree index [4,N]')
        c=np.asarray(constrained)
        if c.dtype!=bool or c.shape!=(a.shape[1],) or c.any():
            raise ValueError('constrained hanging-node update is unsupported')
        if treat_axis is not False:
            raise ValueError('active axis transformation is unsupported')
        if jeccd is not False:
            raise ValueError('JECCD update branch is unsupported')
        if not isinstance(keep_n0_const,bool):
            raise ValueError('explicit keep_n0_const boolean required')
        self.boundary_rows=boundary_rows
        self.keep_n0_const=keep_n0_const
        self.register_buffer('indices',torch.from_numpy(a.astype(np.int64).copy()))
        _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
        self.register_buffer('inverse',torch.from_numpy(inverse.reshape(-1).astype(np.int64)))
        self.register_buffer('counts',torch.from_numpy(counts.astype(np.float64)))

    def forward(self, values, raw_history, proposed_increment):
        if self.keep_n0_const:
            # update_values starts harmonic loop at 2: H=1 performs no writes.
            return TorchFieldStepResult(values.clone(),raw_history.clone(),torch.zeros_like(values))
        # Decoder proposes nodal increments; average in shared native index
        # space, then broadcast one increment per actual solver dof. The mean
        # is a learning projection choice, not a reconstruction of JOREK RHS.
        flat=proposed_increment.reshape(9,-1).T
        shared=flat.new_zeros((len(self.counts),9)).index_add(0,self.inverse,flat)
        shared=shared/self.counts[:,None]
        delta=shared.index_select(0,self.inverse).T.reshape_as(proposed_increment)
        if self.boundary_rows is not None:delta=self.boundary_rows(delta)
        updated=values+delta
        if not torch.isfinite(updated).all():
            raise ValueError('nonfinite native updated values')
        return TorchFieldStepResult(updated,delta,delta)


class NativeFieldStep(nn.Module):
    """All 36 coefficients jointly encoded; rate adapter returns a STEP increment.

    U input: (U-center)/state_scale; raw D input: D/increment_scale.
    Decode: increment_scale * dimensionless_rate * (dt_JU/reference_time_JU).
    This outer learned rate is never an integrated weak load or marker forcing.
    """
    def __init__(self, geometry, update_mapping, *, units: NativeUnits,
                 state_scale, increment_scale, center=None, hidden=16,
                 n_layers=2, n_iters=4, stencil_strength=.01, feature_path=None, temporal_core=None):
        super().__init__()
        units.validate()
        if hidden<1 or n_layers<1 or n_iters<1 or not np.isfinite(stencil_strength) or stencil_strength<0:
            raise ValueError('invalid native learning architecture/stencil strength')
        self.units=units
        self.geometry=geometry
        self.stencil_strength=float(stencil_strength)
        self._geometry_fingerprint=configuration_digest({'pos':geometry.pos,'vertex':geometry.vertex,
            'cluster':geometry.cluster,'mass':geometry.mass,'kernel':geometry.element_kernel})
        self.update_mapping=update_mapping
        if update_mapping.indices.shape!=(4,geometry.n_nodes):
            raise ValueError('native update mapping/geometry node mismatch')
        for name,value in [('state_scale',state_scale),('increment_scale',increment_scale),
                           ('center',np.zeros((9,4)) if center is None else center)]:
            a=torch.as_tensor(value,dtype=torch.float64)
            if a.shape!=(9,4) or not torch.isfinite(a).all() or (name!='center' and (a<=0).any()):
                raise ValueError('scaling requires finite [9,4], with strictly positive scales')
            self.register_buffer(name,a.clone().reshape(9,4,1,1))
        self.encoder=nn.Linear(77,hidden,dtype=torch.float64)
        self.decoder=nn.Linear(hidden,36,dtype=torch.float64)
        rows=update_mapping.boundary_rows
        self.boundary_encoder=(nn.Linear(rows.feature_width,hidden,dtype=torch.float64)
            if hasattr(rows,'feature_width') else None)
        self.temporal_core=temporal_core
        from .native_temporal import TimeConsistentResidualAdapter
        self.time_adapter=TimeConsistentResidualAdapter(
            1. if temporal_core is None else temporal_core.time_scale_ju)
        if temporal_core is not None and temporal_core.hidden!=hidden:
            raise ValueError("temporal core/field hidden width mismatch")
        self.feature_path=feature_path
        if feature_path is not None:
            ev=feature_path.evaluator
            if (ev.units!=units or ev.n_nodes!=geometry.n_nodes
                or not torch.equal(ev.vertex.T,torch.as_tensor(geometry.vertex,dtype=torch.int64))
                or not torch.equal(ev.node_positions,torch.as_tensor(geometry.pos,dtype=torch.float64))):
                raise ValueError('native feature path/stencil geometry or units mismatch')
            self.feature_encoder=nn.Linear(27,hidden,dtype=torch.float64)
            self.scenario_film=nn.Linear(4,2*hidden,dtype=torch.float64) if feature_path.scenario_enabled else None
            self.magnetic_encoder=nn.Linear(2,hidden,bias=False,dtype=torch.float64) if feature_path.physics_enabled else None
        # Reuse implicit stencil matvec/iteration, never its NumPy direct solve.
        operators=build_term_operators(geometry,terms=('eta',),metric_weight='none')
        self.stencil_layers=nn.ModuleList()
        for _ in range(n_layers):
            layer=ImplicitStencilLayer(geometry,operators=operators,n_channels=hidden,
                channel_terms={},alphas={'eta':stencil_strength},explicit_terms=(),
                implicit_terms=('eta',),solver='richardson',n_iters=n_iters)
            layer.channel_mask.fill_(1.)  # all latent channels; not physical eta
            self.stencil_layers.append(layer)

    @property
    def checkpoint_configuration(self):
        """Immutable meaning/scaling of this step, excluding learned weights.

        This control/storage path is outside forward's differentiable graph.
        NativeStateContext.model must bind this exact declared configuration.
        """
        return {'native_field_step':'v3_rate_memory_observation',
            'observation_policy':'clean_identity_rawD_network_observation_values_v1',
            'output_semantics':self.time_adapter.checkpoint_configuration,'units':vars(self.units),'modes':[0],
            'state_scale':self.state_scale.cpu().tolist(),'increment_scale':self.increment_scale.cpu().tolist(),
            'center':self.center.cpu().tolist(),'geometry':self._geometry_fingerprint,
            'indices':self.update_mapping.indices.cpu().tolist(),
            'keep_n0_const':self.update_mapping.keep_n0_const,
            'update_branch':'standard_nonJECCD',
            'hidden':self.encoder.out_features,'n_layers':len(self.stencil_layers),
            'layers':[{name:getattr(layer,name) for name in ('solver','n_iters','preconditioner',
                'max_alpha_scale_delta','max_relax','implicit_terms','explicit_terms','term_names')}
                for layer in self.stencil_layers],
            'spatial_coefficients':[layer.alpha0.cpu().tolist() for layer in self.stencil_layers],
            'stencil_strength':self.stencil_strength,'spatial_approximation':'node_P1_latent',
            'boundary_policy':('learned_unverified' if self.update_mapping.boundary_rows is None else
                getattr(self.update_mapping.boundary_rows,'policy','dirichlet_subset_open_unverified')),
            'boundary_rows':None if self.update_mapping.boundary_rows is None else self.update_mapping.boundary_rows.configuration,
            'feedback_policy':'zero_only',
            'features':None if self.feature_path is None else self.feature_path.checkpoint_configuration,
            'spatial_residual':self.feature_path is not None or self.temporal_core is not None,
            'temporal_core':None if self.temporal_core is None else self.temporal_core.checkpoint_configuration}

    def reference_solver(self, context):
        """Inference Adapter for the existing ReferenceRollout solve_fluid Seam.

        Core forward stays differentiable. The NumPy driver commits detached
        results here; this Adapter does not promise rollout backpropagation.
        """
        validate_reference_context(self,context)
        return NativeReferenceSolver(self,context)

    def forward(self, values, raw_fluid_deltas, *, dt_si, previous_dt_si, scheme, time_coefficients=None, feature_parameters=None, memory=None, source_encoding=None, observation_values=None):
        if any(layer.solver!='richardson' for layer in self.stencil_layers):
            raise ValueError('native learning path supports Torch Richardson only')
        for name,a in [('values',values),('raw fluid history',raw_fluid_deltas)]:
            if (not isinstance(a,torch.Tensor) or a.shape!=(9,4,1,self.geometry.n_nodes)
                    or a.dtype!=self.encoder.weight.dtype or a.device!=self.encoder.weight.device
                    or not torch.isfinite(a).all()):
                raise ValueError(f'{name} requires finite native [9,4,1,N] matching model dtype/device')
        observed=values if observation_values is None else observation_values
        if (not isinstance(observed,torch.Tensor) or observed.shape!=values.shape
            or observed.dtype!=values.dtype or observed.device!=values.device or not torch.isfinite(observed).all()):
            raise ValueError('observation requires full native coefficients matching current state')
        if self.temporal_core is not None:
            self.temporal_core.validate_memory(memory,values.permute(3,0,1,2))
        elif memory is not None:
            raise ValueError('memory provided to registered memoryless candidate')
        # Structural full-step no-op: even arbitrary decoder bias cannot update U/D/h.
        # Validate the supported format/previous time without a zero physical event.
        if not np.isfinite(dt_si) or dt_si<0:
            raise ValueError('invalid field interval')
        if dt_si==0:
            fluid_time_coefficients(scheme,previous_dt_si,previous_dt_si)
            if time_coefficients is not None:
                raise ValueError('zero-time no-op has no physical solve coefficients')
            return TorchFieldStepResult(values,raw_fluid_deltas,torch.zeros_like(values),memory)
        coeff=fluid_time_coefficients(scheme,dt_si,previous_dt_si)
        if time_coefficients is not None and time_coefficients != coeff:
            raise ValueError('time coefficients disagree with explicit steps/scheme')
        h,k=dt_si/self.units.t_norm_si,previous_dt_si/self.units.t_norm_si
        # Tensor permutations remain in the graph; no storage adapter/detach.
        u=((observed-self.center)/self.state_scale).permute(3,0,1,2).reshape(-1,36)
        d=(raw_fluid_deltas/self.increment_scale).permute(3,0,1,2).reshape(-1,36)
        time=values.new_tensor([h,k,coeff.delta_scale,coeff.theta,coeff.zeta]).expand(len(u),5)
        if not torch.isfinite(u).all() or not torch.isfinite(d).all() or not torch.isfinite(time).all():
            raise ValueError('nonfinite native normalization/time features')
        hidden=torch.tanh(self.encoder(torch.cat([u,d,time],dim=1)))
        boundary_features=None
        if self.boundary_encoder is not None:
            boundary_features=self.update_mapping.boundary_rows.learning_features(observed)
            hidden=hidden+torch.tanh(self.boundary_encoder(boundary_features))
        if self.feature_path is not None:
            features=self.feature_path(observed,raw_fluid_deltas,units=self.units,parameters=feature_parameters)
            hidden=hidden+torch.tanh(self.feature_encoder(features.node_observations))
            if features.magnetic is not None:
                hidden=hidden+torch.tanh(self.magnetic_encoder(features.magnetic))
            if features.scenario is not None:
                gamma,beta=self.scenario_film(features.scenario).chunk(2)
                hidden=hidden*(1+.1*torch.tanh(gamma))+.1*torch.tanh(beta)
        elif feature_parameters is not None:
            raise ValueError('feature parameters supplied without enabled feature path')
        if source_encoding is not None:
            if (not isinstance(source_encoding,torch.Tensor) or source_encoding.shape!=hidden.shape
                or source_encoding.dtype!=hidden.dtype or source_encoding.device!=hidden.device
                or not torch.isfinite(source_encoding).all()):
                raise ValueError('source encoding must match native latent shape/dtype/device')
            hidden=hidden+source_encoding  # latent conditioning, not weak load or field forcing
        for layer in self.stencil_layers:
            spatial=layer(hidden,dt=coeff.theta*h)
            hidden=spatial if self.feature_path is None and self.temporal_core is None else hidden+torch.tanh(spatial-hidden)
        next_memory=None
        if self.temporal_core is not None:
            _,candidate_memory=self.temporal_core(hidden,memory,dt_ju=h)
            next_memory=self.time_adapter.memory_update(memory,candidate_memory,dt_ju=h)
            hidden=next_memory
        rate=self.decoder(hidden).reshape(-1,9,4,1).permute(1,2,3,0)
        delta=self.time_adapter.integrate_rate(rate,self.increment_scale,dt_ju=h)
        if not torch.isfinite(delta).all():
            raise ValueError('nonfinite native candidate increment')
        result=self.update_mapping(values,raw_fluid_deltas,delta)
        result.memory=next_memory
        result.boundary_features=boundary_features
        return result


def validate_reference_context(model, context):
    """One context guard for field-only and coupled inference adapters."""
    field=getattr(model,'field',model)
    if (context.units!=field.units or context.modes!=(0,)
            or configuration_digest(context.configuration['model']) != configuration_digest(model.checkpoint_configuration)):
        raise ValueError('native field step/context configuration mismatch')
    context.validate()
    boundary=context.configuration['boundary']
    expected_policy=('learned_unverified' if field.update_mapping.boundary_rows is None else
        getattr(field.update_mapping.boundary_rows,'policy','dirichlet_subset_open_unverified'))
    if boundary.get('policy',boundary.get('options',{}).get('policy'))!=expected_policy:
        raise ValueError('physical/unspecified boundary policy is unsupported; explicitly register learned_unverified')
    if field.update_mapping.boundary_rows is not None:
        native=boundary.get("native",{})
        if not np.array_equal(native.get("boundary"),field.update_mapping.boundary_rows.configuration["boundary"]):
            raise ValueError("native fluid boundary codes do not match constrained rows")
    mesh=context.configuration['mesh']
    if 'x' in mesh:
        pos=np.asarray(mesh['x'])[:,0,0,:].T
        vertex=np.asarray(mesh['vertex'])-1  # native HDF5 IDs
    elif 'pos' in mesh and 'vertex' in mesh:
        pos,vertex=np.asarray(mesh['pos']),np.asarray(mesh['vertex'])
    else:
        raise ValueError('missing native context geometry')
    if not np.array_equal(pos,field.geometry.pos) or not np.array_equal(vertex,field.geometry.vertex):
        raise ValueError('native field step/context geometry mismatch')
    if field.feature_path is not None:
        from .native_features import BASIS_VERSION
        if 'x' not in mesh or 'size' not in mesh:
            raise ValueError('feature path requires full native context geometry')
        digest=configuration_digest(dict(x=np.asarray(mesh['x']),size=np.asarray(mesh['size']),
            vertex=np.asarray(mesh['vertex'])-1,units=field.units,basis=BASIS_VERSION))
        if digest!=field.feature_path.evaluator.geometry_digest:
            raise ValueError('full native feature/context geometry mismatch')
    indices=mesh.get('index')
    if indices is None and hasattr(model,'coupled_inputs'):
        raise ValueError('coupled context requires explicit native shared indices')
    if indices is not None and not np.array_equal(np.asarray(indices),field.update_mapping.indices.cpu().numpy()):
        raise ValueError('native context shared indices mismatch')


class NativeReferenceSolver:
    """Only adaptation; scheduling and atomic commit remain ReferenceRollout."""
    def __init__(self, model, context):
        from copy import deepcopy
        validate_reference_context(model,context)
        self.model=model
        self.supports_field_event=hasattr(model,'coupled_inputs') and model.mode=='joint_learning'
        self.supports_field_memory=model.temporal_core is not None
        self.context=deepcopy(context)
        self._context_fingerprint=self.context.fingerprint()
        self.feature_parameters=None
        if model.feature_path is not None:
            from .native_features import select_feature_parameters,_parameter_values
            rows=self.context.configuration['physics'].get('feature_parameter_rows')
            if rows is None: raise ValueError('checkpoint context requires confirmed feature parameter rows')
            self.feature_parameters=select_feature_parameters(rows,scenario_enabled=model.feature_path.scenario_enabled,
                physics_enabled=model.feature_path.physics_enabled)
            _parameter_values(self.feature_parameters,model.encoder.weight)

    def __call__(self, values, shards, feedback, dt, previous_dt, *, fluid_deltas, time_coefficients, field_memory=None, field_event=None):
        from .marker_state import FluidSolveResult
        if self.context.fingerprint()!=self._context_fingerprint:
            raise ValueError('native solver context configuration changed')
        validate_reference_context(self.model,self.context)
        source_inputs={}
        if hasattr(self.model,'coupled_inputs'):
            source_inputs=self.model.coupled_inputs(shards,feedback,self.model.encoder.weight.device)
            if self.supports_field_event:source_inputs['field_event']=field_event
        else:
            if (feedback is None or feedback.layout!='native_full' or feedback.deltas is None
                    or not np.isfinite(feedback.values).all() or not np.isfinite(feedback.deltas).all()
                    or feedback.values.any() or feedback.deltas.any()):
                raise ValueError('marker feedback consumer unimplemented; reference solver requires explicit zero feedback/history')
        scheme=self.context.configuration['numerics'].get('scheme')
        if scheme is None:
            raise ValueError('explicit reference time scheme required')
        device=self.model.encoder.weight.device
        with torch.no_grad():
            out=self.model(torch.as_tensor(values,device=device),torch.as_tensor(fluid_deltas,device=device),
                dt_si=dt,previous_dt_si=previous_dt,scheme=scheme,time_coefficients=time_coefficients,feature_parameters=self.feature_parameters,
                memory=None if field_memory is None else torch.as_tensor(field_memory,device=device),**source_inputs)
        return FluidSolveResult(out.values.cpu().numpy().copy(),out.fluid_deltas.cpu().numpy().copy(),
            None if out.memory is None else out.memory.cpu().numpy().copy())
