from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from src.models.native_coupling import NativeSourceMesh
from tests.native_relocation_reference import locate as reference_locate
from tests.test_native_locator_events import event_fixture, assert_runtime_exact
from tools.benchmark_native_neighbor_locator import _rng, _tree_equal
from tools.validate_native_relocation_history import validate_sequence, main


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture():
    source, marker, fields = event_fixture(backend='batched_v1')
    windows = [dict(index=4111+i, previous=fields.copy(), current=fields.copy(),
                    raw=np.full_like(fields, .01+i), target=(i+1)*1e-9, dt=1e-9,
                    shards={'source_volume_m3': np.array([1.+i])}) for i in range(2)]
    return source, marker, windows


@pytest.mark.parametrize('offset', [13e-18, 2e-16])
def test_registered_first_rounding_pair_and_second_structural_shards(offset):
    from pathlib import Path
    from types import SimpleNamespace
    from tools.validate_native_relocation_history import recorded_windows
    source, marker, events = fixture()
    dt = 1e-6; logical = .01
    structural = dict(ids=np.array([7]), injector_phi=np.array([.2]),
                      nonlocal_diagnostic='stale', old_equilibrium='stale')
    runtime = SimpleNamespace(fields_previous=events[0]['previous'],
        fields_current=events[0]['current'], fluid_deltas=events[0]['raw'],
        fluid_time_si=logical, fluid_dt_previous_si=dt, shards=structural, marker=marker)
    recorded_time = logical+offset
    calls = []
    volume = np.array([3.])
    def loader(run, index, units, registered):
        calls.append((index, registered))
        target = recorded_time if index == 4111 else logical+dt
        shards = dict(positions=np.array([[1., 2., 3.]]), source_volume_m3=volume,
            source_volume_time_si=target, source_volume_provenance='actual_h5::spi_vol_arr', time_si=target)
        return runtime.fields_current, runtime.fluid_deltas, target, dt, shards
    registered = [4111]
    windows, alignment = recorded_windows(Path('/tmp/fixture'), 4110, source, runtime, registered, loader)
    assert calls == [(4111, []), (4112, registered)]
    assert windows[0]['target'] == logical
    assert windows[0]['shards']['source_volume_time_si'] == logical
    assert windows[0]['shards']['source_volume_m3'] is volume
    assert alignment['recorded_source_volume_time_si'] == recorded_time
    assert alignment['recorded_minus_logical_si'] == recorded_time-logical
    assert windows[1]['shards']['source_volume_time_si'] == logical+dt
    assert windows[1]['shards']['source_volume_provenance'] == 'actual_h5::spi_vol_arr'
    np.testing.assert_array_equal(windows[1]['shards']['ids'], structural['ids'])
    assert 'nonlocal_diagnostic' not in windows[1]['shards']
    assert 'old_equilibrium' not in windows[1]['shards']
    assert runtime.fluid_time_si == logical


@pytest.mark.parametrize('violation', ['large_gap', 'dt', 'fields', 'raw', 'second_restart'])
def test_first_pairing_rejects_mismatch_and_second_keeps_restart_guard(violation):
    from pathlib import Path
    from types import SimpleNamespace
    from tools.validate_native_relocation_history import recorded_windows
    source, marker, events = fixture()
    runtime = SimpleNamespace(fields_previous=events[0]['previous'], fields_current=events[0]['current'],
        fluid_deltas=events[0]['raw'], fluid_time_si=.01, fluid_dt_previous_si=1e-6, shards={})
    def loader(run, index, units, registered):
        if index in registered:
            raise ValueError('ordinary window crosses restart')
        fields = runtime.fields_current.copy(); raw = runtime.fluid_deltas.copy()
        dt = runtime.fluid_dt_previous_si; target = runtime.fluid_time_si
        if violation == 'large_gap': target += 1e-10
        elif violation == 'dt': dt = np.nextafter(dt, np.inf)
        elif violation == 'fields': fields.flat[0] += 1
        elif violation == 'raw': raw.flat[0] += 1
        return fields, raw, target, dt, dict(source_volume_m3=np.array([1.]),
            source_volume_time_si=target, source_volume_provenance='fixture')
    with pytest.raises(ValueError, match='pairing|crosses restart'):
        recorded_windows(Path('/tmp/fixture'), 4110, source, runtime, [4111, 4112], loader)


def test_two_events_carry_nonzero_history_and_serialized_resume_exactly():
    source, marker, windows = fixture()
    original = deepcopy(marker)
    rng = _rng()
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    seen = []
    def births(owner, state, window, sampler):
        if window['index'] == 4112:
            seen.append(deepcopy(state))
            assert np.any(state.feedback.deltas)
            assert state.clock.sim_time_si == 1e-9
            assert state.clock.field_start_si == 1e-9
            assert state.clock.field_end_si == 2e-9
            assert state.cloud.x[0, 0] > original.cloud.x[0, 0]
        random.random(); np.random.random()
        return None
    result = validate_sequence(source, marker, windows, births, lambda *args: None)
    assert result['success']
    assert len(seen) == 4  # Two direct arms and two serialized resumes.
    for row in result['windows'] + result['serialization_resumes']:
        assert row['comparison']['all_arrays_exact']
        assert row['comparison']['diagnostics_all_exact']
        assert row['comparison']['births_exact']
    assert result['windows'][1]['arms']['baseline']['carried_marker_digest'] == (
        result['windows'][0]['arms']['baseline']['final_marker_digest'])
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
    assert_runtime_exact(marker, original)
    assert _tree_equal(_rng(), rng)


@pytest.mark.parametrize('violation', ['history', 'clock', 'diagnostics', 'rng'])
def test_second_event_strictly_rejects_candidate_change(violation):
    source, marker, windows = fixture()
    advance = source.advance
    def changed(state, target, *args, **kwargs):
        out, diag = advance(state, target, *args, **kwargs)
        if target == 2e-9 and NativeSourceMesh.locate is not reference_locate:
            if violation == 'history':
                out.feedback.deltas.flat[0] = np.nextafter(out.feedback.deltas.flat[0], np.inf)
            elif violation == 'clock':
                out.clock.window_index += 1
            elif violation == 'diagnostics':
                diag['moments'] = np.nextafter(diag['moments'], np.inf)
            else:
                random.random()
        return out, diag
    source.advance = changed
    result = validate_sequence(source, marker, windows, lambda *args: None, lambda *args: None)
    assert not result['success']
    assert not result['windows'][1]['comparison']['success']


def test_bad_gap_and_exception_restore_methods_and_rng():
    source, marker, windows = fixture()
    windows[1]['dt'] = 2e-9
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    rng = _rng()
    with pytest.raises(ValueError, match='gap differs'):
        validate_sequence(source, marker, windows, lambda *args: None, lambda *args: None)
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
    assert _tree_equal(_rng(), rng)


def test_births_are_generated_in_each_arm_with_preserved_rng():
    from src.models.marker_state import MarkerCloud
    source, marker, windows = fixture()
    calls = []
    def births(owner, state, window, sampler):
        born = MarkerCloud.empty(10, 1)
        for name in born.array_names():
            getattr(born, name)[0] = getattr(state.cloud, name)[0]
        born.weight[0] *= np.random.uniform(.5, 1.)
        calls.append((window['index'], float(born.weight[0])))
        return born
    result = validate_sequence(source, marker, windows, births, lambda *args: None)
    assert result['success']
    assert len(calls) == 6
    assert calls[0] == calls[1]
    assert calls[2] == calls[3] == calls[4] == calls[5]


def test_cli_help_and_exclusive_output(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert '--run-index' in capsys.readouterr().out
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    output = tmp_path/'existing.json'
    output.touch()
    with pytest.raises(FileExistsError):
        main(['--output', str(output)])
    assert output.read_bytes() == b''
