"""Fast complete-source fixtures only; no recorded million-slot jobs."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from src.models.native_coupling import NativeSourceMesh
from src.models.native_nearby_locator import NativeNearbyMarkerSource
from tests.test_native_locator_events import assert_runtime_exact
from tests.test_native_neighbor_locator import shared_event
from tests.test_native_nearby_locator import saved_chain_neighbours
from tools.benchmark_native_neighbor_locator import _rng, _tree_equal
from tools.validate_native_nearby_history import main, validate_sequence


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture():
    source, marker, fields = shared_event(n_cells=4)
    candidate = NativeNearbyMarkerSource(source, neighbours=saved_chain_neighbours(4))
    windows = [dict(index=4111+i, previous=fields.copy(), current=fields.copy(),
                    raw=np.full_like(fields, .01+i), target=(i+1)*1e-9, dt=1e-9,
                    shards={'source_volume_m3': np.array([1.+i])}) for i in range(2)]
    return source, candidate, marker, windows


def test_complete_two_windows_and_per_arm_serialization_without_method_patches():
    source, candidate, marker, windows = fixture()
    before = deepcopy(marker); rng = _rng()
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    seen = []
    def births(owner, state, window, sampler):
        random.random(); np.random.random(); torch.rand(1)
        seen.append((owner is candidate, window['index'], state.clock.sim_time_si))
        if window['index'] == 4112:
            assert np.any(state.feedback.deltas)
            assert state.clock.field_start_si == 1e-9
            assert state.clock.field_end_si == 2e-9
        return None
    result = validate_sequence(source, candidate, marker, windows, births, lambda *args: None)
    assert result['success']
    assert len(seen) == 6
    assert len(result['serialization_resumes']) == 2
    for row in result['serialization_resumes']:
        comparison = row['comparison']
        assert comparison['complete_marker_exact']
        assert comparison['all_arrays_exact']
        assert comparison['diagnostics_all_exact']
        assert comparison['required_exact']['rng']
        assert comparison['births']['full_births_exact']
        assert comparison['locator_stats_exact']
    for row in result['windows']:
        assert all(row['comparison']['required_exact'].values())
        for key in ('values', 'deltas'):
            assert set(row['comparison']['feedback_channels'][key]) == set(map(str, range(5)))
            assert all('relative_l2' in m for m in row['comparison']['feedback_channels'][key].values())
    for mode in ('current_native', 'nearby'):
        assert result['windows'][1]['arms'][mode]['carried_marker_digest'] == result['windows'][0]['arms'][mode]['final_marker_digest']
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
    assert_runtime_exact(marker, before)
    assert _tree_equal(_rng(), rng)


def test_second_window_increment_subtracts_each_arms_own_start():
    from types import SimpleNamespace
    source, _, marker, windows = fixture()
    starts={False:[],True:[]};ends={False:[],True:[]}
    def owner(candidate):
        def advance(state,target,*args,**kwargs):
            starts[candidate].append(deepcopy(state))
            out,diag=source.advance(state,target,*args,**kwargs)
            if candidate and target==1e-9:
                out.cloud.x[0,0]+=1e-6
                out.cloud.V_prev[0,0]+=100.
            ends[candidate].append(deepcopy(out))
            return out,diag
        return SimpleNamespace(advance=advance)
    result=validate_sequence(owner(False),owner(True),marker,windows,lambda *args:None,lambda *args:None)
    a0,b0=starts[False][1],starts[True][1]
    a1,b1=ends[False][1],ends[True][1]
    assert not np.array_equal(a0.cloud.x,b0.cloud.x)
    support=a1.cloud.alive&b1.cloud.alive&a0.cloud.alive&b0.cloud.alive
    a=a1.cloud.x[support]-a0.cloud.x[support]
    b=b1.cloud.x[support]-b0.cloud.x[support]
    expected=np.linalg.norm(b-a)/np.linalg.norm(a)
    actual=result['windows'][1]['comparison']['increments']['x']['relative_l2']
    assert actual==pytest.approx(expected,rel=1e-12,abs=0.)


def test_births_generated_independently_and_reported_without_future_reuse():
    from src.models.marker_state import MarkerCloud
    source, candidate, marker, windows = fixture()
    calls = []
    def births(owner, state, window, sampler):
        born = MarkerCloud.empty(10, 1)
        for name in born.array_names():
            getattr(born, name)[0] = getattr(state.cloud, name)[0]
        born.weight[0] *= np.random.uniform(.5, 1.)
        if owner is candidate and window['index'] == 4112:
            born.weight[0] *= 2
        calls.append((owner is candidate, window['index'], float(born.weight[0])))
        return born
    result = validate_sequence(source, candidate, marker, windows, births, lambda *args: None)
    assert len(calls) == 6
    assert calls[0][2] == calls[1][2]
    assert calls[2][2] != calls[3][2]
    assert calls[2] == calls[4] and calls[3] == calls[5]
    assert not result['success']
    assert not result['windows'][1]['comparison']['births']['full_births_exact']
    assert not result['windows'][1]['comparison']['required_exact']['cloud.weight']
    assert all(row['comparison']['success'] for row in result['serialization_resumes'])


@pytest.mark.parametrize('arm', ['current_native', 'nearby'])
@pytest.mark.parametrize('violation', ['state', 'diagnostics', 'rng', 'birth'])
def test_same_arm_resume_requires_exact_everything(arm, violation):
    from types import SimpleNamespace
    source, candidate, marker, windows = fixture()
    # Keep the underlying source intact: the nearby adapter copies that source.
    source = SimpleNamespace(advance=source.advance)
    owner = source if arm == 'current_native' else candidate
    advance = owner.advance
    counts = {'second': 0}
    def changed(state, target, *args, **kwargs):
        out, diag = advance(state, target, *args, **kwargs)
        if target == 2e-9:
            counts['second'] += 1
            if counts['second'] == 2:
                if violation == 'state':
                    out.feedback.deltas.flat[0] = np.nextafter(out.feedback.deltas.flat[0], np.inf)
                elif violation == 'diagnostics':
                    diag['moments'] = np.nextafter(diag['moments'], np.inf)
                elif violation == 'rng':
                    random.random()
        return out, diag
    owner.advance = changed
    birth_calls = {False: 0, True: 0}
    def births(runner, state, window, sampler):
        if window['index'] == 4112:
            key = runner is candidate
            birth_calls[key] += 1
            if violation == 'birth' and runner is owner and birth_calls[key] == 2:
                np.random.random()
        return None
    result = validate_sequence(source, candidate, marker, windows, births, lambda *args: None)
    assert not result['success']
    rows = {row['mode']: row['comparison'] for row in result['serialization_resumes']}
    assert not rows[arm]['success']


@pytest.mark.parametrize('violation', ['gap', 'birth_exception', 'memory'])
def test_failure_restores_rng_and_leaves_input_and_methods_unchanged(violation):
    source, candidate, marker, windows = fixture()
    rng = _rng(); before = deepcopy(marker)
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    if violation == 'gap':
        windows[1]['dt'] = 2e-9
    def births(*args):
        random.random(); np.random.random()
        if violation == 'birth_exception':
            raise RuntimeError('birth failed')
        return None
    with pytest.raises((ValueError, RuntimeError, MemoryError)):
        validate_sequence(source, candidate, marker, windows, births, lambda *args: None,
                          memory_bytes=1 if violation == 'memory' else None)
    assert _tree_equal(_rng(), rng)
    assert_runtime_exact(marker, before)
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods


def test_cli_exclusive_output_fixed_bounds_and_no_real_job(tmp_path, monkeypatch):
    import tools.validate_native_nearby_history as tool
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    calls = []
    def fake_execute(args, result):
        calls.append(args)
        result['success'] = True
    monkeypatch.setattr(tool, 'execute', fake_execute)
    output = tmp_path/'fixture.json'
    assert main(['--run-index', '1', '--output', str(output)]) == 0
    args = calls[0]
    assert (args.run_index, args.seed, args.chunk_size, args.wall_seconds, args.memory_mib,
            args.birth_cells_per_width, args.marker_locator) == (1, 1, 2048, 600, 8192, 64, 'native')
    result = json.loads(output.read_text())
    assert result['code_unchanged'] and result['success'] and not result['training']
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(['--output', str(output)])
    assert output.read_bytes() == before
    assert len(calls) == 1


def test_cli_help_and_thread_requirement(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert '--run-index' in capsys.readouterr().out
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    with pytest.raises(SystemExit) as exc:
        main(['--output', '/tmp/not-created-nearby-history.json'])
    assert exc.value.code == 2


def test_execute_reads_actual_initial_field_neighbours_with_unpatched_baseline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import h5py
    import tools.validate_native_nearby_history as tool
    import tools.native_real_chain_builder as builder
    import tools.diagnose_native_real_chain as diagnostic
    source, candidate, marker, windows = fixture()
    run = tmp_path/'phaseA_dataset_1d22_ang15_massloss_neon_v400'
    run.mkdir()
    table = saved_chain_neighbours(4)
    # These are tiny synthetic records; build/source execution is mocked.
    for index in (4110, 4111, 4112):
        with h5py.File(run/f'jorek{index:06d}.h5', 'w') as f:
            f['neighbours'] = table if index == 4111 else np.zeros_like(table)
    for name in ('jorek_namelist_spitest', 'part04110.h5', 'projections004110.h5'):
        (run/name).write_bytes(b'fixture')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'windows': [dict(split='train', run=str(run), anchor=4110),
                                             dict(split='train', run='unused', anchor=5680)]}))
    mock_marker = SimpleNamespace(cloud=SimpleNamespace(weight=range(1000000), alive=np.array([True])),
                                  provenance={'fixture': True})
    runtime = SimpleNamespace(marker=mock_marker, shards={'fixture': np.array([1.])},
                              fluid_deltas=windows[0]['raw'])
    birth = SimpleNamespace(configuration={'fixture': True})
    calls = []
    def fake_build(run_arg, anchor, controls, stages, args, **options):
        assert args.marker_locator == 'native' and options['feedback_mode'] == 'autonomous'
        return object(), source, SimpleNamespace(birth=birth), runtime, object()
    monkeypatch.setattr(builder, 'build_real', fake_build)
    monkeypatch.setattr(diagnostic, '_initial_pairing', lambda *args: [])
    for window in windows:
        window['shards']['source_volume_provenance'] = 'synthetic_fixture'
    monkeypatch.setattr(tool, 'recorded_windows', lambda *args: (windows, {'fixture': True}))
    def fake_validate(baseline, nearby, initial, events, births, sampler, **options):
        calls.append(nearby)
        assert baseline is source and initial is mock_marker
        np.testing.assert_array_equal(nearby.locator.neighbours.T, table)
        assert options['memory_bytes'] == 8192*1024**2
        return dict(success=True)
    monkeypatch.setattr(tool, 'validate_sequence', fake_validate)
    args = SimpleNamespace(seed=1, config=config, run_index=0, marker_locator='native', memory_mib=8192)
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    result = {'success': False}
    tool.execute(args, result)
    assert len(calls) == 1 and result['success']
    assert result['neighbours']['origin'].endswith('jorek004111.h5::neighbours')
    assert result['input_files_unchanged'] and result['other_inputs_unchanged']
    assert result['slots'] == 1000000
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
