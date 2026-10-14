"""Pilot feasibility at the public file-audit seam; no optimizer or rollout."""
import json

import h5py
import numpy as np
import pytest


def recorded_restart(root):
    """Independent time oracle: imported gap 1s, nominal dt .3s, then .5s."""
    root.mkdir(exist_ok=True)
    (root / 'jorek_namelist_spitest').write_text('tstep_particles = 0.3\n')
    for index, time in [(10, 2.), (11, 2.), (12, 2.5), (13, 3.)]:
        with h5py.File(root / f'jorek{index:06d}.h5', 'w') as f:
            for key, value in dict(t_now=time, t_norm=1., tstep=1. if index<12 else .5).items():
                f[key] = [value]
            f['values'] = np.zeros((9, 4, 1, 2))
            f['deltas'] = np.zeros((9, 4, 1, 2))
        with h5py.File(root / f'projections{index:06d}.h5', 'w') as f:
            f['t_now'] = [1. if index == 10 else time]
            f['values'] = np.full((5, 4, 1, 2), 0. if index == 10 else 7.)
    from src.models.marker_state import MarkerCloud
    cloud = MarkerCloud.empty(capacity=3)
    cloud.i_elm[:2] = 1
    cloud.weight[:2] = 2.
    with h5py.File(root / 'part00010.h5', 'w') as f:
        f['time'] = [1.]
        g = f.create_group('groups/001')
        for name in cloud.array_names():
            g['v' if name == 'V_prev' else name] = getattr(cloud, name)
        for key, value in dict(type=b'particle_marker', Z=10, mass=20., adas_suffix=b'ne').items():
            g[key] = [value]
    return root


def test_audit_pairs_source_at_event_start_and_preserves_fixed_particle_clock(tmp_path):
    from tools.audit_native_training_pilot import audit_window
    root = recorded_restart(tmp_path)
    result = audit_window(root, 10, registered_events=[11], events=2)
    assert result['data_feasible']
    assert result['cloud']['slots'] == 3
    assert result['cloud']['live'] == 2
    # floor(1/.3)=3: imported adjustment yields 1/3s, never resets to .3.
    assert result['clock']['particle_dt_si'] == pytest.approx(1/3)
    assert [e['particle_substeps'] for e in result['events']] == [3, 1]
    assert [e['source_label']['index'] for e in result['events']] == [11, 12]
    assert [e['field_label']['index'] for e in result['events']] == [12, 13]
    assert result['events'][0]['source_label']['time_si'] == 2.
    assert result['events'][1]['particle_residual_si'] == pytest.approx(1/6)
    assert result['initial_feedback']['index'] == 10
    assert result['support']['indices'] == [10, 11, 12, 13]
    assert result['cost']['initial_live_times_substeps_proxy'] == 8
    assert result['future_truth_use'] == 'labels_only; schedule_metadata_is_recorded_control'
    json.dumps(result, allow_nan=False)


def test_registered_name_does_not_excuse_wrong_original_particle_clock(tmp_path):
    from tools.audit_native_training_pilot import audit_window
    root = recorded_restart(tmp_path)
    # Zero aux and marker agree with each other, but not imported restart gap.
    with h5py.File(root / 'part00010.h5', 'r+') as f:
        f['time'][0] = .5
    with h5py.File(root / 'projections000010.h5', 'r+') as f:
        f['t_now'][0] = .5
    with pytest.raises(ValueError, match='imported restart gap'):
        audit_window(root, 10, registered_events=[11])


def test_pilot_rejects_shared_full_initialization_support_across_splits(tmp_path):
    from tools.audit_native_training_pilot import audit_pilot
    root = recorded_restart(tmp_path / 'run')
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'runs': {'run': {'restart_events': [11]}}}))
    plan = dict(restart_registry=str(registry), windows=[
        dict(run=str(root), anchor=10, events=2, split=split)
        for split in ('train', 'val', 'test')])
    result = audit_pilot(plan)
    assert not result['data_feasible']
    assert not result['full_support_disjoint']
    assert result['support_conflicts'][0]['indices'] == [10, 11, 12, 13]


@pytest.mark.parametrize('invalid', ['missing_part', 'nonzero_initial_projection', 'zero_live_probability', 'future_source_time', 'unregistered'])
def test_audit_never_manufactures_initialization_or_uses_future_source(tmp_path, invalid):
    from tools.audit_native_training_pilot import audit_window
    root = recorded_restart(tmp_path)
    registered = [11]
    if invalid == 'missing_part':
        (root / 'part00010.h5').unlink()
    elif invalid == 'nonzero_initial_projection':
        with h5py.File(root / 'projections000010.h5', 'r+') as f:
            f['values'][0, 0, 0, 0] = 1.
    elif invalid == 'zero_live_probability':
        with h5py.File(root / 'part00010.h5', 'r+') as f:
            f['groups/001/P_imp'][0] = 0.
    elif invalid == 'future_source_time':
        with h5py.File(root / 'projections000011.h5', 'r+') as f:
            f['t_now'][0] = 2.5
    else:
        registered = []
    with pytest.raises((ValueError, FileNotFoundError)):
        audit_window(root, 10, registered_events=registered)
