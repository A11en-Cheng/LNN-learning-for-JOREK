"""Two full causal events with one supervised backward; never an optimizer.

NumPy source/shard remain stop-gradient. Future labels are opened only after
both forward events. Timing modes are explicit and must not be mixed.
"""
import argparse
import hashlib
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from functools import wraps
import json
import os
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np
import torch

from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest
from tools.profile_native_coupled_latency import restore_bound_model, rng_state, timed_call, write_json, component_digests, spatial_mesh_binding

FIELDS = ('psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e')
CRITICAL_STAGES = {'prefix', 'burnin_and_forecast', 'forward_event_1',
                   'forward_event_2', 'loss_assembly', 'backward'}


def exclusive_stage_summary(rows):
    """Partition critical computation plus label reads using direct children.

    Setup, initialize, and the separate connectivity probe are excluded. Parent
    timers include children; subtract only direct children, never all descendants.
    """
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate timing IDs')
    children = {key: [] for key in indexed}
    for row in rows:
        if not row['completed']:
            raise ValueError('incomplete timing tree')
        parent = row['parent_id']
        if parent is not None:
            if parent not in indexed:
                raise ValueError('missing timing parent')
            p = indexed[parent]
            if not (p['started_at_seconds'] <= row['started_at_seconds']
                    <= row['ended_at_seconds'] <= p['ended_at_seconds']):
                raise ValueError('child timing outside parent')
            children[parent].append(row)
    categories = dict(source_advance_inclusive='marker_transport', marker_births='marker_births',
        shard_density='shard_density', shard_update='shard_update', shard_refresh='shard_refresh',
        model_forward='field_forward', field_solver_prepare='field_prepare',
        coupled_inputs_prepare='input_prepare', reference_validation='state_validation',
        torch_state_validation='state_validation', loss_assembly='loss_metrics',
        native_supervision='loss_metrics', coupled_node_metrics='loss_metrics',
        loss_report='loss_metrics', backward='backward', label_read='label_read')
    groups = {}; included = []; total = 0.
    selected_roots = CRITICAL_STAGES | {'label_read'}
    for row in rows:
        direct = sorted(children[row['id']], key=lambda r: r['started_at_seconds'])
        if any(a['ended_at_seconds'] > b['started_at_seconds'] for a,b in zip(direct,direct[1:])):
            raise ValueError('overlapping sibling timings')
        exclusive = row['seconds'] - sum(child['seconds'] for child in direct)
        if exclusive < -1e-8:
            raise ValueError('children exceed parent duration')
        ancestor = row; visited = set()
        while ancestor['parent_id'] is not None:
            if ancestor['id'] in visited:
                raise ValueError('cyclic timing tree')
            visited.add(ancestor['id']); ancestor = indexed[ancestor['parent_id']]
        if ancestor['stage'] not in selected_roots:
            continue
        if row['parent_id'] is None:
            total += row['seconds']
        key = categories.get(row['stage'], 'event_other')
        groups[key] = groups.get(key, 0.) + max(0., exclusive)
        included.append(dict(id=row['id'], stage=row['stage'], category=key,
                             exclusive_seconds=max(0., exclusive)))
    if abs(sum(groups.values()) - total) > 1e-7 * max(1., total):
        raise ValueError('exclusive timing accounting does not close')
    return dict(scope='critical_computation_plus_label_read_excludes_setup_and_extra_probes',
                total_seconds=total, categories_seconds=groups,
                categories_percent={key:100.*value/total if total else 0. for key,value in groups.items()},
                exclusive_stages=included, accounting_closed=True)


class Timing:
    def __init__(self, device, memory_mib):
        self.device = torch.device(device)
        self.memory_bytes = memory_mib*1024**2
        self.rows = []
        self._stack = []
        self._next_id = 0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        self.sync(); start = time.perf_counter()
        print(json.dumps(dict(stage=name, status='started')), flush=True)
        row = dict(stage=name, completed=False, id=self._next_id,
                   parent_id=self._stack[-1] if self._stack else None, started_at_seconds=start)
        self._next_id += 1
        self._stack.append(row['id'])
        try:
            yield
            self.sync()
            row['completed'] = True
        finally:
            ended = time.perf_counter()
            self._stack.pop()
            row.update(seconds=ended-start, ended_at_seconds=ended,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if self.device.type == 'cuda':
                row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            self.rows.append(row)
            print(json.dumps(row), flush=True)
        if max(row['peak_rss_bytes'], row.get('cuda_peak_allocated_bytes', 0)) > self.memory_bytes:
            raise MemoryError('backward probe memory budget exceeded')


@contextmanager
def component_timing(adapter, timing, *, fine=False):
    """Instance-only transparent wrappers; parent/child times overlap."""
    saved = []
    def install(obj, name, label):
        if not hasattr(obj, name):
            return
        prior = obj.__dict__.get(name)
        owned = name in obj.__dict__
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            with timing.stage(label):
                return original(*args, **kwargs)
        saved.append((obj, name, owned, prior))
        setattr(obj, name, measured)
    try:
        for obj, name, label in [
            (adapter.source, 'advance', 'source_advance_inclusive'),
            (adapter.shard_model, 'density_source', 'shard_density'),
            (adapter.shard_model, 'update', 'shard_update'),
            (adapter.shard_model, 'refresh_diagnostics', 'shard_refresh'),
            (adapter.model, 'forward', 'model_forward'),
            (adapter, 'validate', 'torch_state_validation'),
            (adapter.reference, 'validate', 'reference_validation')]:
            install(obj, name, label)
        if fine:
            for obj, name, label in [
                (adapter.shard_model, 'births', 'marker_births'),
                (adapter.model, 'reference_solver', 'field_solver_prepare'),
                (adapter.model, 'coupled_inputs', 'coupled_inputs_prepare')]:
                install(obj, name, label)
        yield
    finally:
        for obj, name, owned, prior in reversed(saved):
            if owned:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)


def gradient_metrics(gradient):
    if gradient is None:
        return dict(present=False, finite=False, nonzero=False, norm=None, max_abs=None)
    g = gradient.detach()
    finite = bool(torch.isfinite(g).all())
    return dict(present=True, finite=finite, nonzero=bool(torch.count_nonzero(g)),
        norm=float(torch.linalg.vector_norm(g)) if finite else None,
        max_abs=float(g.abs().max()) if finite and g.numel() else None)


def coupling_provenance(reference_file=None):
    """Explicit trusted-local diagnostic import, before any production consumer."""
    if reference_file is None:
        path=Path(__file__).resolve().parents[1]/'src/models/native_coupling.py'
        kind='workspace_production'
    else:
        if any(name in sys.modules for name in ('src.models.native_coupling',
                'tools.native_real_chain_builder','src.training.base_coupled')):
            raise ValueError('frozen coupling requires fresh process before production consumer imports')
        path=Path(reference_file).resolve(strict=True)
        if not path.is_file() or path.suffix!='.py':
            raise ValueError('trusted local coupling Python file required')
        kind='trusted_local_frozen_reference'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if reference_file is not None:
        import src.models as package
        spec=importlib.util.spec_from_file_location('src.models.native_coupling',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name,None)
            raise
        setattr(package,'native_coupling',module)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen coupling file changed during import')
    return dict(kind=kind,path=str(path.resolve()),sha256=digest,
        scope='diagnostic_process_only_physical_source_configuration_unchanged')


@contextmanager
def endpoint_capture(adapter, root, timing, result, controls, curriculum):
    """Observe complete events, write one full-slot endpoint then release it.

NPZ is uncompressed, object-free numerical data. The small torch pickle is
trusted-local metadata/RNG only; NumPy pickle payloads are not mmap arrays.
Capture is nested in driver timing, so the entire run is NOT a timing baseline.
"""
    original = adapter.advance
    owned = 'advance' in adapter.__dict__; previous = adapter.__dict__.get('advance')
    def captured(state, control):
        transient = {}
        def density(module, inputs, kwargs):
            with timing.stage('capture_forward_density'):
                transient['native_density_source'] = kwargs['shard_source'].values.detach().cpu().numpy().copy()
        hook = adapter.model.register_forward_pre_hook(density, with_kwargs=True)
        try:
            pair = original(state, control)
        finally:
            hook.remove()
        out = pair[0]; r = out.runtime
        number = len(result['events'])+1
        segment = ('prefix' if number<=2 else ('forecast' if torch.is_grad_enabled() else 'burnin')) if curriculum else 'forecast'
        with timing.stage('capture_endpoint'):
            clocks = dict(marker=vars(r.marker.clock).copy(),
                fluid=dict(time_si=r.fluid_time_si,index=r.fluid_index,dt_previous_si=r.fluid_dt_previous_si,
                    plan_cursor=r.plan_cursor,step_plan_si=list(r.step_plan_si)),
                feedback=dict(time_si=r.marker.feedback.time_si,gather_time_si=r.marker.feedback.gather_time_si,
                    history_provenance=r.marker.feedback.history_provenance))
            rng = rng_state()
            row = dict(event=number,segment=segment,phase=control.phase,time_si=r.fluid_time_si,
                state_digest=_array_digest(out),rng_digest=_array_digest(rng),clocks=clocks,
                components=component_digests(out),model_digest=_array_digest(adapter.model.state_dict()),
                marker_slots=len(r.marker.cloud.weight),live_markers=int(r.marker.cloud.alive.sum()),
                numeric_capture_schema='native-full-slot-event-arrays-v1',
                transient_density_digest=_array_digest(transient['native_density_source']),
                initial_binding=result['initial_binding'])
            arrays = {k:getattr(out,k).detach().cpu().numpy() for k in ('values','raw_deltas','memory')}
            arrays['fields_previous'] = r.fields_previous
            arrays.update({'cloud.'+k:getattr(r.marker.cloud,k) for k in r.marker.cloud.array_names()})
            arrays['cloud.group'] = np.asarray(r.marker.cloud.group)
            arrays.update({'feedback.'+k:getattr(r.marker.feedback,k) for k in ('values','deltas')})
            arrays.update({'shards.'+k:v for k,v in r.shards.items() if isinstance(v,np.ndarray)})
            arrays.update(transient)
            with (root/f'event-{number}-arrays.npz').open('xb') as stream:
                np.savez(stream,**arrays)
            omitted={'marker','shards','fields_current','fields_previous','fluid_deltas','field_memory'}
            metadata=dict(schema='trusted-native-event-metadata-v1',state_digest=row['state_digest'],
                clocks=clocks,rng=rng,initial_binding=row['initial_binding'],
                runtime={k:v for k,v in vars(r).items() if k not in omitted},
                marker={k:v for k,v in vars(r.marker).items() if k not in ('cloud','feedback','clock')},
                feedback={k:v for k,v in vars(r.marker.feedback).items() if k not in ('values','deltas')},
                shards={k:v for k,v in r.shards.items() if not isinstance(v,np.ndarray)})
            with (root/f'event-{number}-state.pt').open('xb') as stream:
                torch.save(metadata,stream)
            write_json(root/f'event-{number}.json',row)
            result['events'].append(row)
        return pair
    adapter.advance = captured
    try:
        yield
    finally:
        if owned:
            adapter.advance = previous
        else:
            del adapter.advance


def curriculum_forward(adapter, runtime, controls, burnin, timing, root, profile):
    from tools.native_memory_burnin import make_prefix, forecast
    with timing.stage('prefix'):
        prefix, _ = timed_call('prefix', lambda: make_prefix(adapter, runtime, controls, 2),
            root, profile=profile, device=str(timing.device))
    endpoints = []
    original = adapter.advance
    owned = 'advance' in adapter.__dict__
    previous = adapter.__dict__.get('advance')
    preparation = None
    warm_steps = 0
    def measured(state, control):
        nonlocal preparation, warm_steps
        grad = torch.is_grad_enabled()
        if grad and preparation is not None:
            preparation.__exit__(None, None, None); preparation = None
        if not grad:
            warm_steps += 1
        name = f'forecast_event_{len(endpoints)+1}' if grad else f'burnin_event_{warm_steps}'
        with timing.stage(name):
            pair, _ = timed_call(name, lambda: original(state, control), root,
                profile=profile, device=str(timing.device))
        if grad:
            out = pair[0]
            endpoints.append((out.values, out.raw_deltas, out.memory))
        return pair
    adapter.advance = measured
    try:
        with timing.stage('burnin_and_forecast'):
            try:
                preparation = timing.stage('burnin'); preparation.__enter__()
                predictions, proof = forecast(adapter, prefix, burnin=burnin, horizon=2, grad=True)
            finally:
                if preparation is not None:
                    preparation.__exit__(None, None, None); preparation = None
        return endpoints, predictions, proof
    finally:
        if preparation is not None:
            preparation.__exit__(None, None, None)
        if owned:
            adapter.advance = previous
        else:
            del adapter.advance


def curriculum_observe(adapter, predictions, nodes, normalization, timing):
    """Same four terms and object metrics as current curriculum.observe."""
    from src.training.coupled_metrics import native_supervision, coupled_node_metrics
    s = normalization
    scales = dict(coefficients=s['state_scale'], raw_history=s['history_scale'],
        gauss_values=s['gauss']['value_scale'], compatible_gradients=s['gauss']['gradient_scale'],
        provenance='two_train_initials_only')
    indices = adapter.model.field.update_mapping.indices.detach().cpu().numpy()
    losses = []; reports = []
    for prediction, node in zip(predictions, nodes):
        target = dict(node.targets)
        for obj, key in [('marker_projection', 'marker_time_si'), ('shard_density', 'shard_density_time_si'),
                         ('shard_positions', 'shard_time_si')]:
            if obj in target:
                target[key] = node.object_times_si[obj]
        with timing.stage('native_supervision'):
            native = native_supervision(prediction['field'], target['field'],
                evaluator=adapter.model.feature_path.evaluator, scales=scales,
                weights=dict(coefficients=1., raw_history=1., gauss_values=1., compatible_gradients=1.), indices=indices)
        with timing.stage('coupled_node_metrics'):
            metrics = coupled_node_metrics(prediction, target, evaluator=adapter.model.feature_path.evaluator,
                normalization_source=scales['provenance'], region_name='full_native_support', indices=indices)
        with timing.stage('loss_report'):
            reports.append(dict(time_si=node.time_si, loss=native.report(), metrics=metrics))
        losses.append(native.total)
    return losses, reports


def restore_curriculum_window(window, payload, run_index):
    """Restore parameters/buffers/normalization only; never create an optimizer.

The saved full runtime and configuration fingerprints remain strict gates.
This is a diagnostic frozen-weight load, not a curriculum-controller resume.
"""
    from dataclasses import asdict
    from tools.train_native_autonomous_pilot import _install_normalization
    if payload.get('schema') != 'native-autonomous-pilot-v1' or payload['gradient_scope'] != window.adapter.gradient_scope:
        raise ValueError('unsupported trained checkpoint/gradient scope')
    recorded = payload['windows'][run_index]
    if window.name != recorded['name'] or [asdict(c) for c in window.controls] != recorded['controls']:
        raise ValueError('trained checkpoint run/control binding mismatch')
    window.runtime.step_plan_si = tuple(c.dt_si for c in window.controls)
    _install_normalization(window, payload['normalization'])
    model = window.adapter.model
    stored = dict(payload['parameters'], **payload['buffers'][run_index])
    current = model.state_dict()
    if stored.keys() != current.keys() or any(stored[n].shape != v.shape or stored[n].dtype != v.dtype
            or not torch.isfinite(stored[n]).all() for n,v in current.items()):
        raise ValueError('trained complete parameter/buffer schema mismatch')
    model.load_state_dict(stored, strict=True)
    model.train(payload['training_modes'][run_index])
    window.runtime.native_context.configuration['model'] = deepcopy(model.checkpoint_configuration)
    if window.adapter.configuration_fingerprint != recorded['configuration']:
        raise ValueError('trained checkpoint configuration fingerprint mismatch after exact restoration')
    actual = _array_digest(window.runtime)
    if actual != recorded['initial_state_digest']:
        raise ValueError('trained checkpoint initial runtime digest mismatch after exact restoration: '+actual)
    return dict(configuration_fingerprint=recorded['configuration'], initial_runtime_digest=actual,
        actual_model_digest=_array_digest(model.state_dict()), normalization_digest=_array_digest(payload['normalization']),
        model_components={n:_array_digest(v) for n,v in model.state_dict().items()},
        training_mode=model.training, optimizer_loaded=False, controller_resumed=False)


def profile_backward(adapter, runtime, controls, label_loader, output_dir, *, device='cpu',
                     profile_forward=False, profile_backward_ops=False, memory_mib=8192,
                     component_hooks=True, curriculum=None, capture_endpoints=False, implementation=None,
                     fine_components=False):
    """Two complete events; labels returned as values/time_si/provenance dicts."""
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=False)
    timing = Timing(device, memory_mib)
    result = dict(completed=False, success=False, backward_calls=0, optimizer_instantiated=False,
        optimizer_updates=0, gradient_scope=adapter.gradient_scope,
        capture_endpoints=capture_endpoints,events=[],
        coupling_implementation=implementation,
        eligible_timing_baseline=not(capture_endpoints or profile_forward or profile_backward_ops),
        capture_timing_note='capture stages separately recorded but included in driver parents; never use capture run for speedup',
        future_truth_feedback=False, label_read_policy='after_both_forward_events_only',
        profile_forward=profile_forward, profile_backward=profile_backward_ops,
        component_hooks=component_hooks, timing_scope='instrumented_parent_child_inclusive_do_not_sum',
        fine_components=fine_components, stage_tree_schema='ids_direct_parent_perf_counter_intervals_v1',
        memory_scope='cumulative_process_RSS_and_CUDA_peak_not_additive_stage_allocations',
        autograd_diagnostic='after_backward_second_endpoint_loss_to_first_state_and_memory_separate_timing',
        backward_retain_graph=True,
        backward_timing_caveat='retains_saved_graph_for_post_backward_cross_step_probe; actual_training_backward_uses_retain_graph_false; not_identical_release_cost_or_peak_memory',
        initial_rng_digest=_array_digest(rng_state()),
        initial_model_digest=_array_digest(adapter.model.state_dict()),
        model_fingerprint=adapter.fingerprint, source_configuration=adapter.source.configuration)
    before = _array_digest(adapter.model.state_dict())
    initial_runtime_digest = _array_digest(runtime)
    result.update(initial_binding=_array_digest(dict(runtime=initial_runtime_digest,controls=controls,
        model=before,configuration=adapter.configuration_fingerprint,
        curriculum=curriculum,initial_rng=result['initial_rng_digest'])),
        initial_runtime_digest=initial_runtime_digest,
        initial_model_components={k:_array_digest(v) for k,v in adapter.model.state_dict().items()},
        initial_spatial_mesh_binding=spatial_mesh_binding(adapter.source.mesh),
        cpu_spatial_implementation=getattr(adapter.source,'benchmark_cpu_spatial_implementation','production_cpu_batched_v1'))
    if capture_endpoints:
        with (root/'initial-rng.pt').open('xb') as stream:
            torch.save(dict(rng=rng_state(),initial_binding=result['initial_binding']),stream)
    params = {name: p for name, p in adapter.model.named_parameters() if p.requires_grad}
    if any(p.grad is not None for p in params.values()):
        raise ValueError('fresh empty gradients required; will not overwrite existing parameter gradients')
    try:
        if len(controls) != (4 if curriculum is not None else 2):
            raise ValueError('two forecast events plus explicit curriculum prefix required')
        adapter.model.train()
        manager = component_timing(adapter, timing, fine=fine_components) if component_hooks else nullcontext()
        capture = endpoint_capture(adapter,root,timing,result,controls,curriculum) if capture_endpoints else nullcontext()
        with torch.enable_grad(), manager, capture:
            if curriculum is not None:
                endpoints, predictions, proof = curriculum_forward(adapter, runtime, controls,
                    curriculum['burnin'], timing, root, profile_forward)
                result['curriculum'] = proof
                label_controls = controls[2:]
            else:
                with timing.stage('initialize'):
                    state = adapter.initialize(runtime)
                endpoints = []
                for i, control in enumerate(controls, 1):
                    with timing.stage(f'forward_event_{i}'):
                        pair, _ = timed_call(f'forward-event-{i}', lambda: adapter.advance(state, control),
                            root, profile=profile_forward, device=device)
                        state, _ = pair
                    endpoints.append((state.values, state.raw_deltas, state.memory))
                result['forward_final_state_digest'] = _array_digest(state)
                label_controls = controls
            result['forward_final_rng_digest'] = _array_digest(rng_state())
            with timing.stage('label_read'):
                labels = label_loader(label_controls)
                if len(labels) != 2:
                    raise ValueError('exactly two matching field labels required')
            with timing.stage('loss_assembly'):
              if curriculum is not None:
                losses, reports = curriculum_observe(adapter, predictions, labels, curriculum['normalization'], timing)
                loss = torch.stack(losses).mean()
                result.update(loss=float(loss.detach()), loss_reports=reports,
                    per_field_loss=[dict(zip(FIELDS, r['loss']['per_field_total_contribution'])) for r in reports],
                    loss_definition='current_curriculum_observe_four_terms_mean_over_two_forecast_endpoints')
              else:
                scale = adapter.model.field.state_scale.detach()
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError('positive finite checkpoint state_scale required')
                losses = []
                result['per_field_loss'] = []
                result['labels'] = []
                for tensors, label, control in zip(endpoints, labels, label_controls):
                    if abs(label['time_si']-control.end_si) > 16*abs(np.spacing(control.end_si)):
                        raise ValueError('supervision label endpoint time mismatch')
                    target = torch.as_tensor(label['values'], dtype=tensors[0].dtype, device=tensors[0].device)
                    if target.shape != tensors[0].shape or not torch.isfinite(target).all():
                        raise ValueError('finite complete nine-field target required')
                    per_field = ((tensors[0]-target)/scale).square().flatten(1).mean(1)
                    losses.append(per_field.mean())
                    result['per_field_loss'].append(dict(zip(FIELDS, per_field.detach().cpu().tolist())))
                    result['labels'].append(dict(time_si=label['time_si'], provenance=label['provenance']))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite supervised loss')
                result.update(loss=float(loss.detach()),
                    loss_definition='mean_over_two_endpoints_and_nine_fields_of_mean_native_coeff_squared_error_div_checkpoint_state_scale_squared',
                    loss_scale=scale.detach().cpu().tolist())
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.device(device).type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) if profile_backward_ops else None
            with profiler if profiler is not None else nullcontext():
                with timing.stage('backward'):
                    # One real loss backward; keep its graph only for the
                    # separately timed post-backward cross-step diagnostic.
                    loss.backward(retain_graph=True)
                    result['backward_calls'] += 1
            if profiler is not None:
                profiler.export_chrome_trace(str(root/'backward-trace.json'))
                rows = []
                for e in profiler.key_averages():
                    rows.append(dict(name=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                        cpu_self_us=e.self_cpu_time_total, device_total_us=e.device_time_total,
                        device_self_us=e.self_device_time_total))
                write_json(root/'backward-operators.json', dict(
                    top_cpu_self=sorted(rows,key=lambda r:r['cpu_self_us'],reverse=True)[:80],
                    top_device_self=sorted(rows,key=lambda r:r['device_self_us'],reverse=True)[:80]))
            with timing.stage('cross_step_autograd_grad'):
                cross = torch.autograd.grad(losses[1], endpoints[0], allow_unused=True)
                result['cross_step_gradients'] = {k: gradient_metrics(v) for k,v in zip(('values','raw_deltas','memory'), cross)}
            with timing.stage('gradient_validation'):
                result['parameter_gradients'] = {n: gradient_metrics(p.grad) for n,p in params.items()}
                result['intended_named_parameters'] = list(params)
                result['unconnected_or_zero_parameters'] = [n for n,g in result['parameter_gradients'].items() if not g['nonzero']]
                result['all_parameter_gradients_finite_nonzero'] = all(g['finite'] and g['nonzero'] for g in result['parameter_gradients'].values())
                result['all_parameter_gradients_connected_finite'] = all(g['present'] and g['finite'] for g in result['parameter_gradients'].values())
                result['cross_step_verified'] = all(g['finite'] and g['nonzero'] for g in result['cross_step_gradients'].values())
            result['completed'] = True
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(stages=timing.rows, model_unchanged=before == _array_digest(adapter.model.state_dict()),
            initial_runtime_unchanged=initial_runtime_digest == _array_digest(runtime))
        result['success'] = bool(result['completed'] and result['model_unchanged'] and result['initial_runtime_unchanged']
            and result.get('cross_step_verified') and result.get('all_parameter_gradients_connected_finite'))
        result['training_critical_path_seconds'] = sum(r['seconds'] for r in timing.rows if r['stage'] in CRITICAL_STAGES)
        result['extra_cross_step_diagnostic_seconds'] = sum(r['seconds'] for r in timing.rows if r['stage']=='cross_step_autograd_grad')
        if result['completed']:
            result['exclusive_timing'] = exclusive_stage_summary(timing.rows)
        write_json(root/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('curriculum', 'initial-two'), default='curriculum')
    parser.add_argument('--run-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--anchor', type=int, choices=(4110, 5680))
    parser.add_argument('--model-state', type=Path, default=Path(
        'output/native-cuda-curriculum-2026-10-09/seed1-run3/epoch-3-parameters-optimizer.pt'))
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=900.)
    parser.add_argument('--memory-mib', type=int, default=8192)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--birth-max-proposals', type=int, default=1000000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--profile-forward', action='store_true')
    parser.add_argument('--profile-backward', action='store_true')
    parser.add_argument('--no-component-hooks', action='store_true')
    parser.add_argument('--fine-components', action='store_true', help='also time marker births and field/input preparation; same physical chain')
    parser.add_argument('--capture-endpoints', action='store_true', help='full-slot numerical evidence for every prefix/forecast event; not a timing baseline')
    parser.add_argument('--deterministic', action='store_true', help='explicit CUDA deterministic mode; requires fresh process and separate timing/capture runs')
    parser.add_argument('--reference-coupling-file',type=Path,help='execute trusted local frozen native_coupling.py before production imports; diagnostic process only')
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 900 or not 0 < args.memory_mib <= 8192:
        parser.error('bounded wall <=900 seconds, memory <=8192 MiB required')
    if min(args.chunk_size, args.birth_max_proposals, args.threads) <= 0:
        parser.error('positive execution resource settings required')
    if args.fine_components and args.no_component_hooks:
        parser.error('fine components require component hooks')
    args.spatial_backend = 'batched_v1'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(success=False, optimizer_updates=0, optimizer_instantiated=False,
        diagnostic_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage='cuda_initialization', mode=args.mode,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        training='backward_only_no_parameter_updates', future_truth_feedback=False,
        environment={k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS',
            'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
    start = time.perf_counter()
    def deadline(*_):
        raise TimeoutError('whole backward probe wall deadline exceeded')
    signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    timing = Timing('cuda', args.memory_mib)
    try:
        result['stage']='coupling_import'
        result['coupling_implementation']=coupling_provenance(args.reference_coupling_file)
        result['stage']='cuda_initialization'
        if args.deterministic:
            if torch.cuda.is_initialized():
                raise RuntimeError('deterministic CLI requires a fresh process before CUDA initialization')
            existing=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
            if existing not in (None, ':4096:8', ':16:8'):
                raise ValueError('unsupported deterministic CUBLAS_WORKSPACE_CONFIG')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        result['environment']['CUBLAS_WORKSPACE_CONFIG']=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        torch.use_deterministic_algorithms(args.deterministic)
        result['deterministic']=args.deterministic
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback for real backward probe')
        torch.set_num_threads(args.threads)
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        result.update(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            numpy_version=np.__version__, deterministic=torch.are_deterministic_algorithms_enabled())
        from tools.native_real_chain_builder import build_real
        from src.data.base_coupled import StepControl
        from src.training.base_coupled import TorchRecomputedSourceEvents
        from tools.train_native_autonomous_pilot import PilotWindow, read_pilot_labels, _restore_rng
        from dataclasses import asdict
        result['stage'] = 'checkpoint_read'
        with timing.stage('checkpoint_read'):
            payload = torch.load(args.model_state, map_location='cpu', weights_only=False, mmap=True)
        if args.mode == 'curriculum':
            origin = payload['split_config']['origin']
            if (payload['split_config'].get('forecast_cursor') != 2 or
                payload['split_config'].get('horizons') != [1,2,2] or
                origin.get('teacher') is not False or origin.get('truth_field_feedback') is not False):
                raise ValueError('unsupported current curriculum contract')
            specs = [w for w in origin['windows'] if w['split']=='train']
            spec = specs[args.run_index]; run = Path(spec['run']); anchor = spec['anchor']
        else:
            if args.run is None or args.anchor is None:
                raise ValueError('initial-two mode requires explicit --run and --anchor')
            run, anchor = args.run, args.anchor
        if (args.run is not None and args.run != run) or (args.anchor is not None and args.anchor != anchor):
            raise ValueError('explicit run/anchor disagrees with checkpoint train window')
        seed = args.seed if args.seed is not None else 1030+args.run_index
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        result.update(seed=seed, run=str(run), anchor=anchor, checkpoint=str(args.model_state),
            rng_policy='fixed_diagnostic_seed_then_builder_consumes_rng_no_reseed_after_restore; latest_frozen_weights_not_exact_epoch_rng_or_optimizer_resume')
        stages = Stages(); result['stage'] = 'build'
        with timing.stage('build'):
            rows = _initial_pairing(run, anchor, stages)
            model, source, shard, runtime, provider = build_real(run, anchor, rows, stages, args, feedback_mode='autonomous')
        controls = tuple(StepControl(r['start_si'],r['dt_si'],r['end_si'],r['origin'],r['source'],phase=r['phase']) for r in rows)
        adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=provider)
        result['stage'] = 'strict_model_restore'
        with timing.stage('strict_model_restore'):
            if args.mode == 'curriculum':
                recorded = payload['windows'][args.run_index]
                saved_controls = tuple(StepControl(**c) for c in recorded['controls'])
                if [asdict(c) for c in controls] != recorded['controls'][:2] or len(saved_controls) != 4:
                    raise ValueError('initial controls disagree with trained window')
                import h5py
                registry = json.loads(Path(origin['restart_registry']).read_text())['runs'][run.name]['restart_events']
                for index, control in zip((anchor+4,anchor+5),saved_controls[2:]):
                    if index in registry or Path(control.provenance) != run/f'jorek{index:06d}.h5':
                        raise ValueError('forecast restart/path mismatch')
                    with h5py.File(control.provenance,'r') as f:
                        scalar=lambda k:float(np.asarray(f[k][...]).ravel()[0])
                        if (scalar('index_now') != index or scalar('t_now')*scalar('t_norm') != control.end_si
                            or scalar('tstep')*scalar('t_norm') != control.dt_si):
                            raise ValueError('forecast recorded control metadata mismatch')
                controls = saved_controls
                window = PilotWindow(recorded['name'],adapter,runtime,controls,(), 'train','autonomous')
                result['restore'] = restore_curriculum_window(window,payload,args.run_index)
                runtime = window.runtime
                selection_path = args.selection or args.model_state.parent/'selection.json'
                selection = json.loads(selection_path.read_text())
                if type(selection['burnin']) is not int or not 0<=selection['burnin']<=2:
                    raise ValueError('invalid saved burnin selection')
                curriculum = dict(burnin=selection['burnin'],normalization=payload['normalization'])
                result.update(selection=selection, selection_path=str(selection_path),
                    weight_scope='latest_trained_parameters_and_run_specific_buffers_no_optimizer_loaded')
            else:
                binding = _array_digest(dict(geometry=source.mesh.configuration['geometry'],units=source.mesh.units,
                    fields=runtime.fields_current,raw=runtime.fluid_deltas,previous=runtime.fields_previous,
                    clock=runtime.marker.clock,controls=controls))
                if payload.get('schema') != 'native-latency-initial-model-v1' or payload['initial_binding'] != binding:
                    raise ValueError('trusted latency initial-model snapshot binding mismatch')
                restore_bound_model(model,payload)
                _restore_rng(payload['rng'])
                runtime.native_context.configuration['model'] = deepcopy(model.checkpoint_configuration)
                curriculum = None
        with timing.stage('cuda_transfer'):
            model.to('cuda'); model.train()
        result['stage'] = 'complete_chain_backward'
        def labels(cs):
            if args.mode=='curriculum':
                nodes,_ = read_pilot_labels(run, (anchor+4,anchor+5), gamma=model.units.gamma)
                return nodes
            import h5py
            out=[]
            for c in cs:
                with h5py.File(c.provenance,'r') as f:
                    t=float(np.asarray(f['t_now'][...]).ravel()[0])*float(np.asarray(f['t_norm'][...]).ravel()[0])
                    out.append(dict(values=f['values'][...],time_si=t,provenance=c.provenance))
            return out
        result['probe'] = profile_backward(adapter,runtime,controls,labels,args.output_dir/'chain',device='cuda',
            profile_forward=args.profile_forward,profile_backward_ops=args.profile_backward,
            memory_mib=args.memory_mib,component_hooks=not args.no_component_hooks,curriculum=curriculum,
            capture_endpoints=args.capture_endpoints,implementation=result['coupling_implementation'],
            fine_components=args.fine_components)
        result.update(success=result['probe']['success'], stage='completed')
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(elapsed_seconds=time.perf_counter()-start,setup_stages=timing.rows)
        write_json(args.output_dir/'result.json',result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
