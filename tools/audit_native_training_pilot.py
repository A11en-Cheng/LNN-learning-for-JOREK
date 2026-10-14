"""Read-only registered-poststartup pilot audit, NOT a training runner.

Public seam: audit_window(run, anchor, *, registered_events, events=2).
All physical slots are validated by the existing marker reader. Future field
values/projections are labels, never replacement inputs. Schedule metadata is
an explicitly recorded control, not an autonomous timestep-selection claim.
"""
from pathlib import Path
import json

import h5py
import numpy as np

from src.data.marker_checkpoint import read_marker_snapshot
from src.data.namelist import parse_namelist
from src.models.marker_state import MarkerClock, particle_window_schedule


def _scalar(f, name):
    value = np.asarray(f[name][...])
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f'invalid scalar {name}: {f.filename}')
    return float(value.ravel()[0])


def _file(path):
    return dict(path=str(path), bytes=path.stat().st_size)


def _field(root, index):
    path = root / f'jorek{index:06d}.h5'
    with h5py.File(path, 'r') as f:
        shape = f['values'].shape
        if len(shape) != 4 or shape[:3] != (9, 4, 1) or f['deltas'].shape != shape:
            raise ValueError('complete nine-field cubic values/raw deltas required')
        tn = _scalar(f, 't_norm')
        return dict(index=index, time_si=_scalar(f, 't_now')*tn,
                    time_ju=_scalar(f, 't_now'), t_norm_si=tn,
                    imported_tstep_ju=_scalar(f, 'tstep'), shape=list(shape),
                    shards=int(_scalar(f, 'n_spi_tot')) if 'n_spi_tot' in f else None,
                    **_file(path))


def _projection(root, index, shape):
    path = root / f'projections{index:06d}.h5'
    with h5py.File(path, 'r') as f:
        values = f['values'][...]
        if values.shape != (5, *shape[1:]) or not np.isfinite(values).all():
            raise ValueError('complete finite five-channel projection required')
        return dict(index=index, time_si=_scalar(f, 't_now'),
                    exactly_zero=not bool(np.any(values)), **_file(path))


def audit_window(run, anchor, *, registered_events, events=2):
    """Audit one finite full-cloud window; raise on missing/invalid evidence.

    This certifies data availability/pairing, NOT autonomous model closure,
    gradient support, runtime budget sufficiency, or production-build identity.
    """
    if type(events) is not int or events < 1:
        raise ValueError('positive finite event count required')
    if anchor+1 not in registered_events:
        raise ValueError('initialization requires a registered poststartup anchor')
    root = Path(run)
    fields = [_field(root, i) for i in range(anchor, anchor+events+2)]
    imported, initial = fields[:2]
    params_path = root / 'jorek_namelist_spitest'
    nominal = float(parse_namelist(params_path.read_text())['tstep_particles'])
    imported_dt = imported['imported_tstep_ju']*imported['t_norm_si']
    if not np.isfinite(nominal) or nominal <= 0 or imported_dt < nominal:
        raise ValueError('invalid imported/nominal particle step')
    dt = imported_dt / int(imported_dt/nominal)
    marker_path = root / f'part{anchor:05d}.h5'
    cloud, particle_time, metadata = read_marker_snapshot(marker_path)
    if (abs(imported['time_si']-particle_time-imported_dt)>1e-14
            or abs(initial['time_si']-imported['time_si'])>1e-14):
        raise ValueError('particle/poststartup time disagrees with imported restart gap')
    if any(f['t_norm_si'] != imported['t_norm_si'] or f['shape'] != imported['shape'] for f in fields):
        raise ValueError('field units/layout changed within pilot window')
    slots, live = len(cloud.weight), int(np.count_nonzero(cloud.alive))
    payload = sum(getattr(cloud, n).nbytes for n in cloud.array_names())
    del cloud
    feedback = _projection(root, anchor, initial['shape'])
    if not feedback['exactly_zero'] or abs(feedback['time_si']-particle_time)>1e-14:
        raise ValueError('registered original zero projection at marker time required')
    c = MarkerClock(particle_time, 0., dt, particle_time, initial['time_si'])
    rows = []
    for start, end in zip(fields[1:-1], fields[2:]):
        source = _projection(root, start['index'], initial['shape'])
        if abs(source['time_si']-start['time_si'])>1e-14:
            raise ValueError('source label must end at field event START time')
        particle_start, n, gather = particle_window_schedule(c, start['time_si'])
        control = end['time_si']-start['time_si']
        if n < 1 or control <= 0:
            raise ValueError('positive source and field windows required')
        c.sim_time_si = start['time_si']
        c.step_rest_time = max(0., start['time_si']-(particle_start+gather))
        rows.append(dict(start_index=start['index'], end_index=end['index'],
            start_time_si=start['time_si'], end_time_si=end['time_si'],
            control_dt_si=control, control_origin='recorded_field_internal_time_difference',
            source_label=source, field_label=end, particle_start_si=particle_start,
            particle_end_si=particle_start+gather, particle_substeps=n,
            gather_time_si=gather, particle_residual_si=c.step_rest_time,
            feedback_input='registered_original_zero' if len(rows)==0 else 'previous_predicted_projection',
            field_input='initial_full_nine_fields' if len(rows)==0 else 'previous_predicted_nine_fields'))
    return dict(schema='native-autonomous-pilot-data-audit-v1', run=str(root), anchor=anchor,
        data_feasible=True, qualification='files_and_time_pairing_only_not_model_or_training_qualification',
        future_truth_use='labels_only; schedule_metadata_is_recorded_control',
        initial_fields=[imported, initial], initial_feedback=feedback,
        cloud=dict(slots=slots, live=live, array_payload_bytes=payload,
                   metadata=metadata, **_file(marker_path)),
        clock=dict(particle_time_si=particle_time, nominal_particle_dt_si=nominal,
                   particle_dt_si=dt, initial_residual_si=0., imported_dt_si=imported_dt,
                   reconstruction='registered_restart_adjust_once_then_retain_residual'),
        events=rows, support=dict(indices=list(range(anchor, anchor+events+2)),
            shared_static_inputs=[str(params_path)], independent_restart_not_causal_independence_of_plasma=True),
        cost=dict(total_particle_substeps=sum(r['particle_substeps'] for r in rows),
            initial_live_times_substeps_proxy=live*sum(r['particle_substeps'] for r in rows),
            proxy_excludes_births_losses_gauss_projection_and_field_solver=True,
            two_cloud_copies_bytes=2*payload, shards=initial['shards'],
            wall_time_prediction_seconds=None))


def audit_pilot(config):
    """Return finite JSON evidence, retaining input failures and split conflicts."""
    registry_path = Path(config['restart_registry'])
    registry = json.loads(registry_path.read_text())['runs']
    windows, failures, conflicts = [], [], []
    for spec in config['windows']:
        try:
            events = registry[Path(spec['run']).name]['restart_events']
            row = audit_window(spec['run'], spec['anchor'], events=spec['events'],
                               registered_events=events)
            row['split'] = spec['split']
            windows.append(row)
        except (ValueError, OSError, KeyError) as exc:
            failures.append(dict(window=spec, stage='window_input_audit',
                                 error=f'{type(exc).__name__}: {exc}'))
    for i, left in enumerate(windows):
        for right in windows[i+1:]:
            if Path(left['run']).resolve() != Path(right['run']).resolve():
                continue
            shared = sorted(set(left['support']['indices']) & set(right['support']['indices']))
            if shared:
                conflicts.append(dict(run=left['run'], left_split=left['split'],
                                      right_split=right['split'], indices=shared))
    return dict(schema='native-autonomous-pilot-plan-audit-v1',
        data_feasible=bool(windows) and not failures and not conflicts,
        full_support_disjoint=not conflicts, windows=windows,
        failures=failures, support_conflicts=conflicts,
        split_semantics='disjoint dynamic file/index support; shared namelist/geometry/constants allowed; not statistically independent plasma histories',
        qualification='data audit only; autonomous closure, differentiability and execution remain runner checks',
        restart_registry=_file(registry_path),
        total_initial_live_times_substeps_proxy=sum(w['cost']['initial_live_times_substeps_proxy'] for w in windows))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/native-autonomous-training-pilot.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = audit_pilot(config)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['data_feasible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
