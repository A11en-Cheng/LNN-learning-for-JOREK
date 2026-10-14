"""Two-update no-truth-field-feedback pilot; source/shard gradients are stopped.

Run with ``python -m tools.run_native_autonomous_pilot`` under an external
hard deadline. Importing this module never reads real data or starts training.
Public seam: PilotTrainer.update/evaluate/save/load on complete PilotWindows.
"""
from copy import copy, deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.models.native_state import configuration_digest
from src.training.base_coupled import TorchRecomputedSourceEvents
from tools.diagnose_native_real_chain import _array_digest as state_digest
from src.training.coupled_metrics import recorded_supervision

FIELDS = ('psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e')


@dataclass
class PilotWindow:
    name: str
    adapter: object
    runtime: object
    controls: tuple
    nodes: tuple
    split: str
    feedback_mode: str


def read_pilot_labels(run, event_indices, *, gamma):
    """Labels only: original radius liveness and projection at index minus one."""
    import h5py
    from src.data.base_coupled import read_recorded_node
    root=Path(run); nodes=[]; metadata=[]
    for index in event_indices:
        field=root/f'jorek{index:06d}.h5'
        with h5py.File(field,'r') as f:
            radius=f['spi_radius_arr'][...]
            if radius.ndim!=1 or not np.isfinite(radius).all():
                raise ValueError('recorded finite shard radius required for true liveness')
        node,meta=read_recorded_node(field,gamma=gamma,
            projection_path=root/f'projections{index-1:06d}.h5',
            shard_identity=dict(ids=list(range(len(radius))),alive=(radius>0).tolist(),
                                provenance='recorded_fixed_slot_ids_and_true_spi_radius_arr'))
        if meta['field_index']!=index:
            raise ValueError('recorded label internal event index mismatch')
        nodes.append(node); metadata.append(meta)
    return tuple(nodes),tuple(metadata)


def _rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(rng, *, restore_cuda=True):
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    if restore_cuda and rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])


def _guard(window):
    from src.models.native_psi_teacher import RecordedPsiFeedback, NativePsiTeacherShardModel
    from src.models.native_equilibrium import NativeEquilibriumProvider
    adapter = window.adapter
    if (window.feedback_mode != 'autonomous' or not isinstance(adapter, TorchRecomputedSourceEvents)
            or isinstance(adapter.equilibrium_provider, RecordedPsiFeedback)
            or isinstance(adapter.shard_model, NativePsiTeacherShardModel)
            or (adapter.equilibrium_provider is not None and not isinstance(adapter.equilibrium_provider,NativeEquilibriumProvider))
            or adapter.reference.equilibrium_provider is not adapter.equilibrium_provider
            or 'psi_teacher_event' in window.runtime.shards):
        raise ValueError('teacher/recorded feedback forbidden in autonomous pilot')
    if window.split != 'train':
        raise ValueError('pilot accepts train windows only; no validation/test fitting or automatic evaluation')
    if len(window.controls) != 2 or len(window.nodes) != 2:
        raise ValueError('exactly two complete AR events and field labels required')
    if window.runtime.marker.clock.startup_pending:
        raise ValueError('poststartup initialization required')


def _fit_initials(windows):
    """Explicit equal-record train fit; Gauss moments are volume weighted per run.

    Zero-variance groups get an explicitly recorded invertible unity convention.
    Source closure scales remain named physical references, not fitted statistics.
    """
    us = [np.asarray(w.runtime.fields_current) for w in windows]
    ds = [np.asarray(w.runtime.fluid_deltas) for w in windows]
    center = np.mean([u.mean(-1, keepdims=True) for u in us], axis=0)
    variance = np.mean([((u-center)**2).mean(-1, keepdims=True) for u in us], axis=0)
    history = np.mean([(d*d).mean(-1, keepdims=True) for d in ds], axis=0)
    positive = lambda a: np.where(a > 0, a, 1.)
    sums=[]; squares=[]; gradients=[]; scenarios=[]; magnetics=[]
    with torch.no_grad():
        for w in windows:
            model=w.adapter.model; evaluator=model.feature_path.evaluator
            u=torch.as_tensor(w.runtime.fields_current, dtype=model.encoder.weight.dtype,
                              device=model.encoder.weight.device)
            g=evaluator(u, units=model.units); weight=evaluator.volume/evaluator.volume.sum()
            sums.append((g.values*weight[...,None]).sum((0,1)).detach().cpu().numpy())
            squares.append((g.values.square()*weight[...,None]).sum((0,1)).detach().cpu().numpy())
            gradients.append((g.gradients.square()*weight[...,None,None]).sum((0,1)).detach().cpu().numpy())
            rows=w.runtime.native_context.configuration['physics']['feature_parameter_rows']
            parameters={r['name']:r['value'] for r in rows}
            scenarios.append(np.square([parameters['n_spi'], parameters['pellet_density']]))
            magnetics.append((g.gradients[:,:,0,:].square()*parameters['eta']**2*weight[...,None]).sum((0,1)).detach().cpu().numpy())
    gc=np.mean(sums,0)
    return dict(version='pilot-train-initial-statistics-v1', fit_support=[w.name for w in windows],
        fit_objects='initial_fields_and_raw_history_only', zero_group_policy='unity_only_for_zero_empirical_scale',
        source_scales='explicit_physical_reference_NOT_fitted', center=center.tolist(),
        state_scale=positive(np.sqrt(variance)).tolist(), history_scale=positive(np.sqrt(history)).tolist(),
        gauss=dict(value_center=gc.tolist(), value_scale=positive(np.sqrt(np.maximum(0,np.mean(squares,0)-gc**2))).tolist(),
            gradient_scale=positive(np.sqrt(np.mean(gradients,0))).tolist(),
            scenario_scale=positive(np.sqrt(np.mean(scenarios,0))).tolist(),
            magnetic_scale=positive(np.sqrt(np.mean(magnetics,0))).tolist()))


def _install_normalization(window, statistics):
    model=window.adapter.model
    with torch.no_grad():
        for dest, key in [('center','center'), ('state_scale','state_scale'), ('increment_scale','history_scale')]:
            buffer=getattr(model.field,dest)
            buffer.copy_(torch.as_tensor(statistics[key],dtype=buffer.dtype,device=buffer.device))
        for name, value in statistics['gauss'].items():
            buffer=getattr(model.feature_path,name)
            buffer.copy_(torch.as_tensor(value,dtype=buffer.dtype,device=buffer.device))
    model.feature_path.training_support=[[name, 'initial_only'] for name in statistics['fit_support']]
    # Replace the context, not the input physical state or its cloud/history.
    runtime=copy(window.runtime)
    runtime.native_context=deepcopy(window.runtime.native_context)
    runtime.native_context.configuration['model']=deepcopy(model.checkpoint_configuration)
    window.runtime=runtime


class PilotTrainer:
    """Shared learnable weights; per-run geometry/source buffers stay distinct."""
    def __init__(self, windows, *, split_config, seed=1, learning_rate=1e-4, device='cpu'):
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('pilot supports explicit cpu or cuda device only')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('requested CUDA unavailable in this process; check GPU permissions')
            index=torch.cuda.current_device() if self.device.index is None else self.device.index
            if index>=torch.cuda.device_count():raise ValueError('requested CUDA device index unavailable')
            self.device=torch.device('cuda',index)
        self.windows=list(windows)
        if not self.windows or len({w.name for w in self.windows})!=len(self.windows):
            raise ValueError('distinct train windows required')
        for w in self.windows: _guard(w)
        if seed != 1 or learning_rate != 1e-4:
            raise ValueError('fixed pilot seed=1 and learning_rate=1e-4; no search')
        self.split_config=deepcopy(split_config); self.seed=seed; self.cursor=0
        # Canonical CPU fit/init keeps old CPU checkpoint statistics and seed
        # semantics unchanged. Move BEFORE parameter sharing and Adam creation.
        # NumPy runtime/source state remains CPU; only Torch models are migrated.
        for w in self.windows:w.adapter.model.cpu()
        self.normalization=_fit_initials(self.windows)
        for w in self.windows: _install_normalization(w,self.normalization)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=self.windows[0].adapter.model
        # Replace diagnostic prescribed sinusoidal weights with seeded init.
        with torch.no_grad():
            for p in model.parameters(): p.normal_(0., .02)
            model.field.decoder.weight.mul_(1e-3)
            model.field.decoder.bias.zero_()
        for w in self.windows:w.adapter.model.to(self.device)
        shared=dict(model.named_parameters())
        for w in self.windows[1:]:
            other=dict(w.adapter.model.named_parameters())
            if shared.keys()!=other.keys() or any(shared[n].shape!=other[n].shape for n in shared):
                raise ValueError('learnable architecture differs across runs')
            for name,p in shared.items():
                parent,_,leaf=name.rpartition('.')
                setattr(w.adapter.model.get_submodule(parent),leaf,p)
        self.optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
        self.gradient_scope=TorchRecomputedSourceEvents.gradient_scope

    def _forward(self,index):
        w=self.windows[index]; _guard(w)
        state=w.adapter.initialize(w.runtime); predictions=[]; evidence=[]
        for control in w.controls:
            incoming=state_digest(state.values)
            state,diag=w.adapter.advance(state,control)
            predictions.append(w.adapter.prediction(state))
            if 'psi_feedback_intervention' in predictions[-1] or predictions[-1].get('autonomous_prediction') is False:
                raise ValueError('teacher intervention discovered in event output')
            evidence.append(dict(input_field_digest=incoming,predicted_field_digest=state_digest(state.values),
                source_consumes='detached_current_prediction' if len(evidence) else 'initial_nine_fields',
                diagnostics=dict(gradient_scope=diag.get('gradient_scope'),
                    marker_event_time_si=state.runtime.marker.feedback.time_si,
                    particle_time_si=state.runtime.marker.clock.particle_time_si,
                    residual_si=state.runtime.marker.clock.step_rest_time)))
        # Labels become observable only after both complete AR events.
        losses=[]; events=[]
        scales=dict(coefficients=self.normalization['state_scale'],raw_history=self.normalization['history_scale'],
            gauss_values=self.normalization['gauss']['value_scale'],
            compatible_gradients=self.normalization['gauss']['gradient_scale'],
            provenance='train_initials_only:'+','.join(self.normalization['fit_support']))
        for prediction,node,control,proof in zip(predictions,w.nodes,w.controls,evidence):
            if abs(node.time_si-control.end_si)>1e-14:
                raise ValueError('field label endpoint mismatch')
            label=node.targets['field']
            if label['units']!=w.adapter.model.units:
                raise ValueError('field label units mismatch')
            value=prediction['field']['values']
            truth=torch.as_tensor(label['values'],dtype=value.dtype,device=value.device)
            if truth.shape!=value.shape or not torch.isfinite(truth).all():
                raise ValueError('complete finite field labels required')
            observation=recorded_supervision(prediction,node,{k:True for k in node.targets},
                evaluator=w.adapter.model.feature_path.evaluator,scales=scales,
                weights=dict(coefficients=1.,raw_history=1.,gauss_values=1.,compatible_gradients=1.),
                normalization_source=scales['provenance'],region_name='full_native_support',
                indices=w.adapter.model.field.update_mapping.indices.cpu().numpy())
            native_loss=observation['loss']; losses.append(native_loss.total)
            event=dict(time_si=node.time_si,fields={name:dict(loss=float(native_loss.per_field_contribution[i].detach()),
                **observation['metrics']['fields']['gauss_values'][name]) for i,name in enumerate(FIELDS)},
                loss_report=native_loss.report(),metrics=observation['metrics'],**proof)
            events.append(event)
        return torch.stack(losses).mean(),events

    def evaluate(self):
        """Same two train windows only; evaluation does not consume training RNG."""
        rng=_rng(); modes=[w.adapter.model.training for w in self.windows]
        try:
            with torch.no_grad():
                out=[]
                for i,w in enumerate(self.windows):
                    eval_seed=self.seed+i
                    random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)
                    w.adapter.model.eval(); loss,events=self._forward(i)
                    out.append(dict(window=w.name,loss=float(loss),events=events,evaluation_seed=eval_seed))
                return out
        finally:
            _restore_rng(rng)
            for w,mode in zip(self.windows,modes): w.adapter.model.train(mode)

    def update(self,index):
        rng=_rng()
        parameters=list(self.windows[0].adapter.model.parameters())
        before=[p.detach().clone() for p in parameters]
        optimizer=deepcopy(self.optimizer.state_dict()); cursor=self.cursor
        try:
            return self._update(index)
        except Exception:
            with torch.no_grad():
                for p,saved in zip(parameters,before): p.copy_(saved)
            self.optimizer.load_state_dict(optimizer); self.cursor=cursor
            _restore_rng(rng)
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def _update(self,index):
        if self.cursor>=min(2,len(self.windows)) or index!=self.cursor:
            raise ValueError('fixed one-update-per-run order and two-update budget')
        w=self.windows[index]; _guard(w); w.adapter.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss,events=self._forward(index)
        if not torch.isfinite(loss): raise ValueError('nonfinite loss; no optimizer update')
        loss.backward()
        gradients=[p.grad for p in w.adapter.model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError('missing/nonfinite intended parameter gradient; no optimizer update')
        norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
        if not torch.isfinite(norm) or not bool(norm>0): raise ValueError('nonfinite/zero gradient; no optimizer update')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in w.adapter.model.parameters()):
            raise ValueError('nonfinite updated parameters; rollback update')
        self.cursor+=1
        return dict(window=w.name,update=self.cursor,loss=float(loss.detach()),gradient_norm=float(norm),
                    gradient_scope=self.gradient_scope,events=events)

    def save(self,path):
        """Trusted local checkpoint at a completed window/update boundary only."""
        from dataclasses import asdict
        payload=dict(schema='native-autonomous-pilot-v1', cursor=self.cursor,seed=self.seed,
            execution_device=str(self.device),
            split_config=self.split_config, normalization=self.normalization,
            windows=[dict(name=w.name,controls=[asdict(c) for c in w.controls],
                          configuration=w.adapter.configuration_fingerprint,
                          initial_state_digest=state_digest(w.runtime)) for w in self.windows],
            parameters={n:p.detach().cpu().clone() for n,p in self.windows[0].adapter.model.named_parameters()},
            buffers=[{n:b.detach().cpu().clone() for n,b in w.adapter.model.named_buffers()} for w in self.windows],
            training_modes=[w.adapter.model.training for w in self.windows],
            optimizer=self.optimizer.state_dict(),rng=_rng(),gradient_scope=self.gradient_scope,
            resume_semantics='next_window_reinitialized_from_its_registered_initial_state_not_mid_event')
        with Path(path).open('xb') as f: torch.save(payload,f)

    def load(self,path):
        """Restore shared weights only; never copy one run's geometry into another."""
        from dataclasses import asdict
        payload=torch.load(path,map_location='cpu',weights_only=False)
        expected=[dict(name=w.name,controls=[asdict(c) for c in w.controls],
                       configuration=w.adapter.configuration_fingerprint,
                       initial_state_digest=state_digest(w.runtime)) for w in self.windows]
        if (payload['schema']!='native-autonomous-pilot-v1' or payload['seed']!=self.seed
                or configuration_digest(payload['normalization'])!=configuration_digest(self.normalization)
                or configuration_digest(payload['split_config'])!=configuration_digest(self.split_config)
                or payload['windows']!=expected or payload['gradient_scope']!=self.gradient_scope
                or not 0<=payload['cursor']<=min(2,len(self.windows))):
            raise ValueError('checkpoint configuration/normalization/split mismatch')
        parameters=dict(self.windows[0].adapter.model.named_parameters())
        saved=payload['parameters']
        if parameters.keys()!=saved.keys() or any(saved[n].shape!=p.shape or not torch.isfinite(saved[n]).all() for n,p in parameters.items()):
            raise ValueError('checkpoint parameter mismatch/nonfinite values')
        modes=payload['training_modes']; buffers=payload['buffers']
        if len(modes)!=len(self.windows) or any(type(m) is not bool for m in modes) or len(buffers)!=len(self.windows):
            raise ValueError('checkpoint per-run mode/buffer count mismatch')
        for w,stored in zip(self.windows,buffers):
            current=dict(w.adapter.model.named_buffers())
            if current.keys()!=stored.keys() or any(current[n].shape!=b.shape or current[n].dtype!=b.dtype
                or not torch.isfinite(b).all() for n,b in stored.items()):
                raise ValueError('checkpoint per-run buffer mismatch')
        with torch.no_grad():
            for name,p in parameters.items(): p.copy_(saved[name])
            for w,stored,mode in zip(self.windows,buffers,modes):
                current=dict(w.adapter.model.named_buffers())
                for name,b in stored.items(): current[name].copy_(b)
                w.adapter.model.train(mode)
        self.optimizer.load_state_dict(payload['optimizer']); self.cursor=payload['cursor']
        # Device is execution metadata, not a physics fingerprint. CPU-only
        # readers may restore a CUDA checkpoint without initializing CUDA.
        # Cross-device trajectories are not promised bitwise identical.
        _restore_rng(payload['rng'],restore_cuda=self.device.type=='cuda' or torch.cuda.is_initialized())
