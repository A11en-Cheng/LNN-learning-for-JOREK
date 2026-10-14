#!/usr/bin/env python3
"""Read-only, bounded evidence audit. Never advances JOREK or a learned model.

This is a research artifact generator, not a training data adapter. Future
records are read only for labels and explicitly qualified schedule replay.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from src.data.marker_checkpoint import initialize_native_checkpoint, read_marker_snapshot
from src.data.namelist import parse_namelist
from src.data.output_contract import parse_macroscopic_times
from src.data.native_run import _read_native_shard

BASE = Path('data/EHL-2')
NAME = 'phaseA_dataset_1d22_ang15_massloss_neon_v400'
TOL_JU = 1e-7


def witness(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def field(run, index, table):
    path = run / f'jorek{index:06d}.h5'
    with h5py.File(path, 'r') as f:
        scalar = lambda key: float(np.asarray(f[key][()]).ravel()[0])
        row = {key: scalar(key) for key in ('index_now', 't_now', 't_norm', 'tstep')}
        assert row['index_now'] == index
        assert abs(row['t_now'] - table[index]) < TOL_JU
        for key in ('values', 'deltas'):
            values = f[key][()]
            assert values.shape == (9, 4, 1, 4125) and np.isfinite(values).all()
        row['shard_keys'] = [key for key in f if key.startswith('spi_')]
        row['shape'] = list(f['values'].shape)
        row['saved_scalar_parameters'] = {}
        for key in f:
            if f[key].size == 1:
                value = np.asarray(f[key][()]).ravel()[0]
                if isinstance(value, bytes): value = value.decode().strip()
                elif isinstance(value, np.generic): value = value.item()
                row['saved_scalar_parameters'][key] = value
    row['source'] = witness(path)
    row['previous_fluid_dt_ju'] = row['t_now'] - table[index-1]
    return row


def datum(status, source, algorithm, phase, value=None):
    return dict(status=status, source=source, algorithm=algorithm,
                tolerance=dict(time_ju=TOL_JU, time_si_floor=1e-14), phase=phase, value=value)


def audit_anchor(suffix, index, restart_events):
    run = BASE / (NAME + suffix)
    table = parse_macroscopic_times(run / 'macroscopic_vars.dat')
    restart = index + 1 in restart_events
    # All initialization evidence is at or before index. index+1/+2 are used
    # solely as schedule replay witnesses, never to fill physical history.
    last = index+10 if not suffix else index+7
    records = {i: field(run, i, table) for i in range(index-2, last+1)}
    current = records[index]
    _read_native_shard(run / f'jorek{index:06d}.h5', index, side='checkpoint')
    nl = run / 'jorek_namelist_spitest'
    parameters = parse_namelist(nl.read_text())
    files = [witness(run / 'macroscopic_vars.dat'), witness(nl),
             witness(run / f'part{index:05d}.h5')]
    if restart:
        checkpoint = initialize_native_checkpoint(run / f'jorek{index:06d}.h5',
            run / f'part{index:05d}.h5', event_index=index,
            times_path=run / 'macroscopic_vars.dat', namelist_path=nl,
            boundary='jorek_restart', previous_fluid_path=run / f'jorek{index-1:06d}.h5')
        cloud = checkpoint.marker.cloud
        marker_time = checkpoint.marker.clock.sim_time_si
        clock = vars(checkpoint.marker.clock).copy()
        assert not checkpoint.marker.feedback.values.any()
        assert not checkpoint.marker.feedback.deltas.any()
    else:
        cloud, marker_time, _ = read_marker_snapshot(run / f'part{index:05d}.h5')
        clock = None
    assert abs(marker_time-table[index-1]*current['t_norm']) < 1e-14
    projections = []
    for i in range(index-2, index+1):
        path = run / f'projections{i:06d}.h5'
        with h5py.File(path, 'r') as f:
            values = f['values'][()]
            assert values.shape == (5, 4, 1, 4125) and np.isfinite(values).all()
            projections.append(dict(index=i, time_si=float(np.ravel(f['t_now'][()])[0]),
                absmax=float(np.abs(values).max()), dtype=str(values.dtype),
                scaleoffset=f['values'].scaleoffset, deltas_saved='deltas' in f,
                rcs=str(np.ravel(f['RCS_version'][()])[0]), source=witness(path)))
    if restart:
        assert projections[-1]['absmax'] == 0
        assert abs(projections[-1]['time_si']-marker_time) < 1e-14
        assert abs(records[index+1]['t_now']-current['t_now']) < TOL_JU
    else:
        assert abs(projections[-2]['time_si']-marker_time) < 1e-14
    labels = []
    for i in range(index+1,last+1):
        path = run/f'projections{i:06d}.h5'
        with h5py.File(path,'r') as f:
            values = f['values'][()]
            assert values.shape == (5,4,1,4125) and np.isfinite(values).all()
            time_si = float(np.ravel(f['t_now'][()])[0])
            assert abs(time_si-records[i]['t_now']*records[i]['t_norm']) < 1e-14
            labels.append(dict(index=i,time_si=time_si,source=witness(path),
                role='conditional_source_or_supervision_only_not_initialization',
                deltas_saved='deltas' in f))
    # Restart startup writes the imported tstep in the duplicate output.
    # Its first scheduling control is witnessed by the NEXT positive interval.
    control_witness = index+2 if restart else index+1
    dt = records[control_witness]['t_now'] - records[control_witness-1]['t_now']
    assert dt > 0 and abs(dt-records[control_witness]['tstep']) < TOL_JU
    control = dict(origin='recorded_schedule_replay', dt_ju=records[control_witness]['tstep'],
        dt_si=records[control_witness]['tstep']*current['t_norm'],
        source=[records[control_witness-1]['source'], records[control_witness]['source']],
        algorithm='positive field interval crosschecked against saved tstep; restart startup inherits first segment scheduling control',
        phase='startup' if restart else 'ordinary', status='verified_reconstructed',
        autonomous_input_eligible=False, event_review_eligible=True,
        limitation='future recorded control witness; no external control supplied; production startup implementation equivalence pending')
    schedule = []
    for event in range(index+1,last+1):
        wi = index+2 if restart and event == index+1 else event
        rec = records[wi]
        assert rec['previous_fluid_dt_ju'] > 0
        assert abs(rec['previous_fluid_dt_ju']-rec['tstep']) < TOL_JU
        schedule.append(dict(event_index=event,dt_ju=rec['tstep'],dt_si=rec['tstep']*rec['t_norm'],
            witness_index=wi,source=rec['source'],origin='recorded_schedule_replay',autonomous_input_eligible=False))
    phase = 'registered_restart_before_startup' if restart else 'ordinary_after_fluid_before_particle'
    states = {}
    for key in ('values', 'raw_fluid_deltas', 'shard'):
        states[key] = datum('saved', current['source'], 'native HDF5; no endpoint differencing', phase)
    states['previous_field'] = datum('saved', records[index-1]['source'], 'full native values', phase)
    states['marker'] = datum('saved', files[-1], 'full all-slot validation; alive iff i_elm>0', phase,
                            dict(slots=len(cloud.weight), active=int(cloud.alive.sum())))
    history_source = ['data/restart_events.json', projections[-1]['source'],
                      'docs/adr/0020-marker-source-contract.md#决定'] if restart else [p['source'] for p in projections[:2]]
    states['projection_values'] = datum('verified_reconstructed' if restart else 'saved',
        history_source, 'registered zero feedback' if restart else 'previous projection at marker event time', phase)
    states['projection_history'] = datum('verified_reconstructed' if restart else 'pending_evidence',
        history_source, 'registered restart zero values and raw deltas; no ordinary-window extension' if restart else
        'candidate P2709-P2708; NOT evaluated or admitted without production update/output witness', phase)
    states['clock'] = datum('verified_reconstructed' if restart else 'pending_evidence',
        [files[0], files[1], current['source']],
        'restart residual=0; imported tstep divides nominal particle dt; startup_pending' if restart else
        'requires causal stepping from reset2704, effective particle dt, all scheduler events and actual gather; absent saved clock', phase, clock)
    states['next_control'] = datum('verified_reconstructed', control['source'], control['algorithm'], control['phase'], control)
    states['external_next_control'] = datum('unavailable', None,
        'no external controller supplied; recorded schedule is a separate qualification', phase)
    states['imported_tstep'] = datum('saved', current['source'], 'native scalar tstep', phase, current['tstep'])
    states['previous_fluid_step'] = datum('verified_reconstructed',
        [records[index-1]['source'],current['source']], 'internal t_now difference crosschecked with imported tstep',
        'before restart setup' if restart else phase,current['previous_fluid_dt_ju'])
    assert abs(current['previous_fluid_dt_ju']-current['tstep']) < TOL_JU
    for key, value in [('marker_substep', None if clock is None else clock['particle_dt_si']),
                       ('residual', None if clock is None else clock['step_rest_time'])]:
        states[key] = datum('verified_reconstructed' if restart else 'pending_evidence',
            [files[1],current['source'],'data/restart_events.json'],
            'registered restart initialization' if restart else
            'see ../clock-witnesses.json; clock-only candidate, not certified state replay',phase,value)
    states['actual_gather'] = datum('pending_evidence',None,
        'not equal by default to fluid dt; requires actual integer substep count times particle_dt',
        'next particle event')
    states['event_phase'] = datum('verified_reconstructed' if restart else 'pending_evidence', history_source,
        'registered event + internal times + zero projection' if restart else 'time pairing alone cannot certify ordinary scheduler phase', phase)
    states['configuration'] = datum('pending_evidence', [files[1], current['source']],
        'bind full namelist bytes plus all HDF5 scalar parameters; effective production defaults/build/dirty patch not witnessed', phase)
    states['production_equivalence'] = datum('pending_evidence',
        'report/native-closure-production-provenance-2026-10-08.md', 'subsystem evidence, not HEAD equality', phase)
    blockers = ['rocket6', 'nonlocal_ablation', 'production_birth_and_kernel',
                'production_filtered_constrained_projection', 'active_PRB_radiation', 'physical_boundary']
    if int(cloud.alive.sum()) > 10000: blockers.append('reference_live_marker_limit_10000')
    result = dict(run=run.name, index=index, boundary=phase, state=states,
        configuration_binding=dict(namelist=files[1], parsed_namelist=parameters,
            saved_h5_scalars=current['saved_scalar_parameters'], effective_configuration_complete=False),
        time_contract=dict(imported_tstep_ju=current['tstep'],
            previous_fluid_dt_ju=current['previous_fluid_dt_ju'],
            previous_dt_semantics='startup solver uses next control as tstep_prev' if restart else 'ordinary previous fluid step',
            next_control=control, marker_substep_si=None if clock is None else clock['particle_dt_si'],
            residual_si=None if clock is None else 0., actual_gather_si=None,
            gather_status='pending_evidence', projection_clock=[{k:p[k] for k in ('index','time_si')} for p in projections]),
        recorded_control_schedule=schedule,
        data_state_qualification=dict(core_registered_initialization=restart,
            fully_production_certified=False, reasons=['effective_configuration_and_production_equivalence_pending'] +
            ([] if restart else ['ordinary_clock_gather_history_pending'])),
        model_support_qualification=dict(eligible=False, reasons=blockers),
        control_qualification=dict(recorded_event_review=True, autonomous=False, origin='recorded_schedule_replay'),
        projections=projections, future_projection_labels=labels, field_records=records, witnesses=files,
        no_training=True, no_particle_clipping=True, causal_replay_executed=False)
    return result


def windows(anchors):
    # Split assignment exists BEFORE support construction. The later test
    # candidates must reject replay from a training anchor even if local labels fit.
    blocks = [(NAME, 'train', 4108, 4115), (NAME, 'test', 4117, 4120),
              (NAME+'_injInner', 'validation', 2707, 2716),
              (NAME+'_injInner', 'validation', 5678, 5686)]
    rows = []
    for anchor in anchors:
        a = anchor['index']; run = anchor['run']
        starts = [a+j for j in ((2,3,4,5,7,8) if run == NAME else (2,3,4,5))]
        for start in starts:
            end = start+2
            # includes prior U, projection history, anchor, startup/replay,
            # one warmup transition, and two prediction transitions.
            first = a-1 if anchor['data_state_qualification']['core_registered_initialization'] else a-2
            support = list(range(first, end+1))
            target_blocks = [b for b in blocks if b[0] == run and b[2] <= start < end <= b[3]]
            split = target_blocks[0][1] if target_blocks else None
            isolated = bool(target_blocks and all(target_blocks[0][2] <= i <= target_blocks[0][3] for i in support))
            common = [] if isolated else ['full_initialization_replay_history_warmup_support_outside_split']
            data_reasons = common + anchor['data_state_qualification']['reasons'] + ['causal_replay_to_window_start_not_executed']
            if anchor['boundary'].startswith('registered_restart'):
                data_reasons.append('post_restart_projection_history_requires_causal_replay')
            model_reasons = anchor['model_support_qualification']['reasons']
            rows.append(dict(run=run, anchor=a, prediction_start=start, prediction_end=end,
                split=split, warmup_transitions=1, horizon=2, support_indices=support,
                support_roles=dict(history=list(range(first,a)), initialization=[a],
                    startup=[a,a+1] if anchor['boundary'].startswith('registered_restart') else [],
                    replay=list(range(a+1,start-1)), warmup=[start-1,start], prediction=list(range(start,end+1))),
                full_support_isolated=isolated,
                saved_conditional_source_labels=[p for p in anchor['future_projection_labels']
                    if start <= p['index'] < end],
                qualifications=dict(data_state=dict(eligible=False,reasons=data_reasons),
                    model_support=dict(eligible=False,reasons=model_reasons),
                    control_source=dict(recorded_event_review=True, autonomous=False,
                        origin='recorded_schedule_replay',reasons=['external_control_not_supplied']),
                    conditional_source=dict(eligible=False,reasons=data_reasons),
                    self_generated_closed_loop=dict(eligible=False,reasons=data_reasons+model_reasons+['external_control_not_supplied']))))
    return dict(blocks=[dict(run=b[0],split=b[1],first=b[2],last=b[3]) for b in blocks],
        windows=rows, counts=dict(candidates=len(rows), full_support_isolated=sum(r['full_support_isolated'] for r in rows),
            core_registered_initializations=sum(a['data_state_qualification']['core_registered_initialization'] for a in anchors),
            recorded_control_source=sum(r['qualifications']['control_source']['recorded_event_review'] for r in rows),
            autonomous_control_source=sum(r['qualifications']['control_source']['autonomous'] for r in rows),
            **{q:sum(r['qualifications'][q]['eligible'] for r in rows)
               for q in ('data_state','model_support','conditional_source','self_generated_closed_loop')}),
        rejection_counts=dict(Counter(reason for row in rows for reason in row['qualifications']['data_state']['reasons'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise ValueError('refuse to overwrite evidence directory')
    registry = json.loads(Path('data/restart_events.json').read_text())['runs']
    anchors = [audit_anchor(s, i, registry[NAME+s]['restart_events'])
               for s,i in [('',4110),('_injInner',2710),('_injInner',5680)]]
    manifest = windows(anchors)
    args.output_dir.mkdir(parents=True)
    for name, payload in [('anchors.json', anchors), ('windows.json', manifest)]:
        (args.output_dir/name).write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__': main()
