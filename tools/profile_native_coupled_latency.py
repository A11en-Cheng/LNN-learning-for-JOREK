"""Bounded two-event autonomous latency probe; no labels, training or optimizer.

Run one registered train anchor per process. cProfile times are instrumented
wall times, not an uninstrumented speedup baseline. Inclusive parents and their
children MUST NOT be summed. --no-profile retains identical event execution.
"""
import argparse
import cProfile
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import pstats
import random
import resource
import signal
import time

import numpy as np
import torch

from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def component_digests(state):
    r = state.runtime
    cloud = r.marker.cloud
    return dict(cloud={k: _array_digest(getattr(cloud, k)) for k in cloud.array_names()},
        transport_history=_array_digest(getattr(r.marker,'transport_history',None)),
        fields={k: _array_digest(getattr(r, k)) for k in
                ('fields_current', 'fields_previous', 'fluid_deltas', 'field_memory')},
        torch_state={k: _array_digest(getattr(state, k)) for k in ('values', 'raw_deltas', 'memory')},
        feedback={k: _array_digest(getattr(r.marker.feedback, k)) for k in ('values', 'deltas')},
        shards={k: _array_digest(v) for k, v in r.shards.items() if isinstance(v, np.ndarray)},
        clocks=_array_digest(dict(marker=r.marker.clock, fluid_time_si=r.fluid_time_si,
            fluid_index=r.fluid_index, fluid_dt_previous_si=r.fluid_dt_previous_si,
            plan_cursor=r.plan_cursor, step_plan_si=r.step_plan_si,
            feedback_time_si=r.marker.feedback.time_si,
            gather_time_si=r.marker.feedback.gather_time_si)),
        rng={k: _array_digest(v) for k, v in rng_state().items()})


def timed_call(name, operation, output_dir, *, profile, device):
    """Synchronize only at measurement boundaries; dump partial profile on failure."""
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    profiler = cProfile.Profile() if profile else None
    start = time.perf_counter()
    try:
        if profiler:
            profiler.enable()
        value = operation()
        if cuda:
            torch.cuda.synchronize(device)
        return value, time.perf_counter() - start
    finally:
        elapsed = time.perf_counter() - start
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(output_dir / (name + '.prof')))
            stats = pstats.Stats(profiler)
            rows = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0],
                         calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
                    for k, v in stats.stats.items()]
            write_json(output_dir / (name + '-profile.json'), dict(
                timing='instrumented wall; inclusive parents overlap children; CUDA boundary synchronized',
                elapsed_seconds=elapsed,
                top_cumulative=sorted(rows, key=lambda r: r['cumulative_seconds'], reverse=True)[:80],
                top_self=sorted(rows, key=lambda r: r['self_seconds'], reverse=True)[:80],
                source_components=[r for r in rows if any(s in r['file'] for s in
                    ('native_source_', 'native_shard_', 'native_equilibrium', 'native_coupling'))]))


def field_repeat_probe(model, fresh, values, raw_deltas, kwargs, output_dir, *, repeats=4):
    """Frozen public joint-field forwards only; no source advance or training."""
    from tools.native_replay_metrics import relative_metrics
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    fresh.load_state_dict(model.state_dict(), strict=True); fresh.eval(); model.eval()
    digest = _array_digest(model.state_dict())
    numerical_inputs = dict(values=values, raw_deltas=raw_deltas, memory=kwargs['memory'],
        marker_feedback=kwargs['marker_feedback'], projection_deltas=kwargs['projection_deltas'],
        native_density_source=kwargs['shard_source'].values)
    array = lambda t: t.detach().cpu().numpy().copy()
    inputs_digest = _array_digest(numerical_inputs)
    np.savez(output_dir/'frozen-inputs.npz', **{k: array(v) for k,v in numerical_inputs.items()})
    result = dict(scope='conditional_joint_field_fixed_initial_zero_feedback_not_complete_event',
        model_digest=digest, input_digest=inputs_digest,
        density_input_digest=_array_digest(kwargs['shard_source'].values),
        device=str(values.device), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        observations=[])
    original = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            mode = 'deterministic' if deterministic else 'default'
            reference = None
            for label, candidate in (('original', model), ('fresh', fresh)):
                for i in range(repeats):
                    with torch.no_grad():
                        out = candidate(values, raw_deltas, **kwargs)
                    arrays = dict(values=array(out.values), raw_deltas=array(out.fluid_deltas), memory=array(out.memory))
                    np.savez(output_dir/f'{mode}-{label}-{i}.npz', **arrays)
                    if reference is None:
                        reference = arrays
                    metrics = {k: [relative_metrics(reference[k][f], arrays[k][f]) for f in range(9)]
                               for k in ('values', 'raw_deltas')}
                    metrics['memory'] = relative_metrics(reference['memory'], arrays['memory'])
                    row = dict(mode=mode, model=label, repeat=i, metrics=metrics,
                        exact_to_mode_reference=all(np.array_equal(reference[k], arrays[k]) for k in arrays))
                    result['observations'].append(row)
                    print(json.dumps(dict(stage='frozen_field_repeat', mode=mode, model=label, repeat=i,
                        exact=row['exact_to_mode_reference'], memory_relative_l2=metrics['memory']['relative_l2'])), flush=True)
        result.update(model_unchanged=all(_array_digest(m.state_dict()) == digest for m in (model, fresh)),
                      inputs_unchanged=_array_digest(numerical_inputs) == inputs_digest)
    finally:
        torch.use_deterministic_algorithms(original, warn_only=warn_only)
        write_json(output_dir/'result.json', result)
    return result


def restore_bound_model(model, saved, *, allow_spatial_backend_change=False, mesh=None):
    """Validate configuration against restored buffers; failed loads are atomic."""
    previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.load_state_dict(saved['state_dict'], strict=True)
        old, new = deepcopy(saved['configuration']), deepcopy(model.checkpoint_configuration)
        if allow_spatial_backend_change:
            # Explicit benchmark-only migration. This never changes a production
            # checkpoint loader or discards any non-spatial source semantics.
            from tools.compare_native_spatial_rollouts import _spatial_configuration_valid
            for config in (old, new):
                source = config['source']
                backend = source.get('spatial_backend', 'scalar')
                if (mesh is None or backend not in ('scalar', 'batched_v1', 'cuda_fp64_v1')
                    or source.get('mesh', {}).get('geometry') != mesh.configuration['geometry']
                    or not _spatial_configuration_valid(source, backend, spatial_mesh_binding(mesh))):
                    raise ValueError('spatial configuration migration geometry/backend mismatch')
                spatial = source.get('spatial_configuration')
                if spatial is not None and spatial['mesh_binding'] != spatial_mesh_binding(mesh):
                    raise ValueError('spatial configuration full mesh binding mismatch')
                source.pop('spatial_backend', None)
                source.pop('spatial_configuration', None)
        if _array_digest(old) != _array_digest(new):
            raise ValueError('initial model snapshot configuration mismatch; backend migration requires explicit authorization')
        model.eval()
    except BaseException:
        model.load_state_dict(previous, strict=True)
        for module, training in modes:
            module.training = training
        raise


def spatial_mesh_binding(mesh):
    """Live CPU geometry evidence using the native FP64 spatial binding schema."""
    from src.models.native_state import configuration_digest
    return configuration_digest(dict(configuration=mesh.configuration, units=vars(mesh.units),
        x=mesh.coordinates, size=mesh.size, vertex=mesh.vertex, indices=mesh.indices,
        bounds=mesh._location_bounds, centers=mesh._location_centers,
        radius=mesh._location_radius, tree=mesh._location_tree.data))


def profile_events(adapter, runtime, controls, output_dir, *, profile=True, device='cpu', model_state=None, capture_arrays=False, allow_spatial_backend_change=False):
    """Public complete-event seam; CPU allowed for small non-production tests only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(controls) != 2:
        raise ValueError('exactly two complete events required')
    binding = _array_digest(dict(geometry=adapter.source.mesh.configuration['geometry'],
        units=adapter.source.mesh.units, fields=runtime.fields_current, raw=runtime.fluid_deltas,
        previous=runtime.fields_previous, clock=runtime.marker.clock, controls=controls))
    if model_state is not None:
        saved = torch.load(model_state, map_location='cpu', weights_only=False)
        if saved.get('schema') != 'native-latency-initial-model-v1' or saved['initial_binding'] != binding:
            raise ValueError('initial model snapshot geometry/state/control binding mismatch')
        restore_bound_model(adapter.model, saved, allow_spatial_backend_change=allow_spatial_backend_change,
                            mesh=adapter.source.mesh)
        from tools.train_native_autonomous_pilot import _restore_rng
        _restore_rng(saved['rng'])
        # Config includes spectral buffers: synchronize after exact restoration.
        runtime = copy(runtime)
        runtime.native_context = deepcopy(runtime.native_context)
        runtime.native_context.configuration['model'] = adapter.model.checkpoint_configuration
    weights = {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}
    payload = dict(schema='native-latency-initial-model-v1', initial_binding=binding,
        state_dict=weights, configuration=adapter.model.checkpoint_configuration, rng=rng_state())
    torch.save(payload, output_dir / 'initial-model.pt')
    result = dict(success=False, events=[], profiled=profile, optimizer_updates=0,
                  capture_arrays=capture_arrays,
                  cpu_spatial_implementation=getattr(adapter.source, 'benchmark_cpu_spatial_implementation',
                      'production_shared_geometry_v1' if adapter.source.spatial_backend == 'batched_v1' else 'not_applicable'),
                  allow_spatial_backend_change=allow_spatial_backend_change,
                  future_truth_feedback=False, model_fingerprint=adapter.fingerprint,
                  source_configuration=adapter.source.configuration,
                  initial_spatial_mesh_binding=spatial_mesh_binding(adapter.source.mesh),
                  initial_binding=binding, initial_model_digest=_array_digest(weights),
                  initial_model_components={k: _array_digest(v) for k, v in weights.items()},
                  initial_rng_digest=_array_digest(payload['rng']))
    before = _array_digest(adapter.model.state_dict())
    try:
        with torch.no_grad():
            state, result['initialize_seconds'] = timed_call('initialize', lambda: adapter.initialize(runtime),
                output_dir, profile=profile, device=device)
            result['initial_components'] = component_digests(state)
            for i, control in enumerate(controls, 1):
                print(json.dumps(dict(stage='complete_event', event=i, status='started', phase=control.phase)), flush=True)
                captured = {}
                def capture(module, inputs, kwargs):
                    captured['native_density_source'] = kwargs['shard_source'].values.detach().cpu().numpy().copy()
                hook = adapter.model.register_forward_pre_hook(capture, with_kwargs=True) if capture_arrays else None
                try:
                    (state, diagnostic), elapsed = timed_call(f'event-{i}', lambda: adapter.advance(state, control),
                        output_dir, profile=profile, device=device)
                finally:
                    if hook is not None:
                        hook.remove()
                row = dict(event=i, phase=control.phase, seconds=elapsed, time_si=state.runtime.fluid_time_si,
                    marker_slots=len(state.runtime.marker.cloud.weight), live_markers=int(state.runtime.marker.cloud.alive.sum()),
                    components=component_digests(state))
                if capture_arrays:
                    r = state.runtime
                    row['clocks'] = dict(marker=vars(r.marker.clock).copy(),
                        fluid=dict(time_si=r.fluid_time_si, index=r.fluid_index, dt_previous_si=r.fluid_dt_previous_si,
                                   plan_cursor=r.plan_cursor, step_plan_si=list(r.step_plan_si)),
                        feedback=dict(time_si=r.marker.feedback.time_si, gather_time_si=r.marker.feedback.gather_time_si,
                                      history_provenance=r.marker.feedback.history_provenance))
                    arrays = {k: getattr(state, k).detach().cpu().numpy() for k in ('values', 'raw_deltas', 'memory')}
                    arrays['fields_previous'] = r.fields_previous
                    arrays.update({'cloud.'+k: getattr(r.marker.cloud, k) for k in r.marker.cloud.array_names()})
                    arrays['cloud.group'] = np.asarray(r.marker.cloud.group)
                    arrays.update({'feedback.'+k: getattr(r.marker.feedback, k) for k in ('values', 'deltas')})
                    arrays.update({'shards.'+k: v for k,v in r.shards.items() if isinstance(v, np.ndarray)})
                    np.savez(output_dir / f'event-{i}-arrays.npz', **arrays, **captured)
                    row['numeric_capture_schema'] = 'native-full-slot-event-arrays-v1'
                    row['transient_density_digest'] = _array_digest(captured['native_density_source'])
                    row['timing_note'] = 'includes optional forward pre-hook density D2H capture; not default timing mode'
                result['events'].append(row)
                write_json(output_dir / f'event-{i}.json', row)
                print(json.dumps(dict(stage='complete_event', event=i, status='completed', seconds=elapsed)), flush=True)
            result.update(final_state_digest=_array_digest(state), final_rng_digest=_array_digest(rng_state()),
                model_unchanged=before == _array_digest(adapter.model.state_dict()), success=True)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch.device(device).type == 'cuda':
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        write_json(output_dir / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--anchor', type=int, choices=(4110, 5680), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--wall-seconds', type=float, default=900.)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--spatial-backend', choices=('scalar', 'batched_v1', 'cuda_fp64_v1'), default='batched_v1')
    p.add_argument('--marker-locator', choices=('native','jorek_nearby_v1'), default='native',
        help='opt-in saved-neighbour JOREK nearby workflow; CPU batched_v1 only')
    p.add_argument('--marker-transport',choices=('native_fixed','local_subcycling_v1','local_subcycling_boundary_v2'),default='native_fixed',
        help='explicit numerical approximation with per-marker history; not checkpoint-compatible with fixed transport')
    p.add_argument('--allow-spatial-backend-change', action='store_true', help='benchmark-only: allow validated spatial backend/config migration after exact model restoration')
    p.add_argument('--cpu-shared-reference', action='store_true', help='benchmark-only frozen pre-sharing CPU points/deposit; batched_v1 only, physical config unchanged')
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--model-state', type=Path, help='trusted initial-model.pt from same run/anchor/controls; restores parameters, buffers and RNG')
    p.add_argument('--capture-arrays', action='store_true', help='diagnostic only: save numerical field/history/memory and transient forward density; adds D2H capture overhead')
    p.add_argument('--field-repeat-probe', action='store_true', help='no complete events: repeat frozen conditional joint-field CUDA forwards, default versus deterministic')
    p.add_argument('--deterministic', action='store_true', help='opt-in deterministic algorithms for complete events; changes timing mode, never compare against default-mode timings')
    args = p.parse_args()
    if args.cpu_shared_reference and (args.spatial_backend != 'batched_v1' or args.field_repeat_probe):
        p.error('--cpu-shared-reference requires batched_v1 complete-event mode')
    if not 0 < args.wall_seconds <= 900 or min(args.memory_mib, args.threads, args.chunk_size, args.birth_max_proposals) <= 0:
        p.error('positive budgets and wall-seconds <=900 required')
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400' + ('_injInner' if args.anchor == 5680 else '')
    if args.run.name != expected:
        p.error('run/anchor must match the declared train window')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(success=False, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cpu_spatial_implementation=('frozen_pre_shared_geometry_literal_v1' if args.cpu_shared_reference else
            'production_shared_geometry_v1' if args.spatial_backend == 'batched_v1' else 'not_applicable'),
        training=False, future_truth_feedback=False, deterministic=args.deterministic, control_origin='recorded_schedule_replay',
        weights='builder_prescribed_nonzero_untrained_fixed_parameters', profile_scope='build separate; initialize and each complete event independently',
        environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')})
    start = time.monotonic()
    def deadline(signum, frame):
        raise TimeoutError('whole probe wall deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    stages = Stages()
    try:
        if args.field_repeat_probe or args.deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        result['environment']['CUBLAS_WORKSPACE_CONFIG'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        torch.use_deterministic_algorithms(args.deterministic)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; sandbox visibility failure is not CPU fallback permission')
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        result.update(device=torch.cuda.get_device_name(), torch_version=torch.__version__, numpy_version=np.__version__)
        rows, result['pairing_seconds'] = timed_call('pairing', lambda: _initial_pairing(args.run, args.anchor, stages),
            args.output_dir, profile=not args.no_profile, device='cpu')
        from tools.native_real_chain_builder import build_real
        bundle, result['build_seconds'] = timed_call('build', lambda: build_real(args.run, args.anchor, rows, stages, args,
            feedback_mode='autonomous'), args.output_dir, profile=not args.no_profile, device='cpu')
        model, source, shard, runtime, provider = bundle
        del bundle
        if args.cpu_shared_reference:
            from tools.native_spatial_reference import bind_cpu_shared_reference
            bind_cpu_shared_reference(source)
        result['source_configuration'] = source.configuration
        result['equilibrium_configuration'] = provider.configuration
        from src.models.native_state import configuration_digest
        result['source_fingerprint'] = configuration_digest(source.configuration)
        from src.training.base_coupled import TorchRecomputedSourceEvents
        from src.data.base_coupled import StepControl
        _, result['cuda_transfer_seconds'] = timed_call('cuda-transfer', lambda: model.to('cuda'),
            args.output_dir, profile=False, device='cuda')
        model.eval()
        adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=provider)
        controls = tuple(StepControl(r['start_si'], r['dt_si'], r['end_si'], r['origin'], r['source'], phase=r['phase']) for r in rows)
        result['controls'] = rows
        if args.field_repeat_probe:
            if args.model_state is None:
                raise ValueError('field probe requires explicit bound baseline model snapshot')
            saved = torch.load(args.model_state, map_location='cpu', weights_only=False)
            binding = _array_digest(dict(geometry=source.mesh.configuration['geometry'], units=source.mesh.units,
                fields=runtime.fields_current, raw=runtime.fluid_deltas, previous=runtime.fields_previous,
                clock=runtime.marker.clock, controls=controls))
            if saved['initial_binding'] != binding:
                raise ValueError('field probe initial snapshot binding mismatch')
            restore_bound_model(model, saved, allow_spatial_backend_change=args.allow_spatial_backend_change,
                                mesh=source.mesh)
            runtime.native_context.configuration['model'] = model.checkpoint_configuration
            from src.models.native_source_features import FieldEventContext
            density = shard.density_source(runtime.shards)
            shards = dict(runtime.shards, native_density_source=density)
            inputs = model.coupled_inputs(shards, runtime.marker.feedback, 'cuda')
            tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device='cuda')
            kwargs = dict(dt_si=controls[0].dt_si, previous_dt_si=runtime.fluid_dt_previous_si,
                scheme=runtime.time_evol_scheme, memory=tensor(runtime.field_memory),
                feature_parameters=model.reference_solver(runtime.native_context).feature_parameters,
                field_event=FieldEventContext(runtime.fluid_time_si, runtime.marker.feedback.time_si,
                    'registered_restart', 'diagnostic_fixed_initial_zero_feedback_not_actual_ordinary_event'), **inputs)
            fresh_bundle = build_real(args.run, args.anchor, rows, stages, args, feedback_mode='autonomous')
            fresh = fresh_bundle[0].to('cuda')
            fresh_density = fresh_bundle[2].density_source(fresh_bundle[3].shards)
            result['fresh_density_exact'] = torch.equal(density.values, fresh_density.values)
            del fresh_bundle, fresh_density
            result['field_repeat'] = field_repeat_probe(model, fresh, tensor(runtime.fields_current),
                tensor(runtime.fluid_deltas), kwargs, args.output_dir/'field-repeat')
            result['success'] = True
            return 0
        chain = profile_events(adapter, runtime, controls, args.output_dir / 'chain', profile=not args.no_profile,
            device='cuda', model_state=args.model_state, capture_arrays=args.capture_arrays,
            allow_spatial_backend_change=args.allow_spatial_backend_change)
        result.update(success=chain['success'], final_state_digest=chain['final_state_digest'],
            final_rng_digest=chain['final_rng_digest'], model_fingerprint=chain['model_fingerprint'])
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__, error=str(exc), stage=stages.current)
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(args.output_dir / 'result.json', result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
