from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from src.models.native_coupling import NativeSourceMesh
from tests.native_relocation_reference import locate as reference_locate
from tests.test_native_locator_events import event_fixture, assert_runtime_exact
from tools.benchmark_native_relocation import benchmark_source_window, main
from tools.benchmark_native_neighbor_locator import _rng, _tree_equal


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('scenario', ['normal', 'crossing', 'lost'])
def test_exact_window_order_profiles_and_restoration(scenario):
    source, state, fields = event_fixture(backend='batched_v1', scenario=scenario)
    before = deepcopy(state)
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    rng = _rng()
    result = benchmark_source_window(source, state, 1e-9, fields, fields)
    assert result['success']
    assert result['coordinates_c_float64'] == bool(
        source.mesh.coordinates.dtype == np.dtype(np.float64)
        and source.mesh.coordinates.flags.c_contiguous)
    assert result['cache_flag_scope'] == 'eligibility_not_observed_cache_hits'
    assert [(r['mode'], r['warmup']) for r in result['iterations']] == [
        ('baseline', True), ('candidate', True), ('baseline', False),
        ('candidate', False), ('candidate', False), ('baseline', False)]
    assert len(result['timeline']) == 8
    assert set(result['profiles']) == {'baseline', 'candidate'}
    assert not result['profile_timing_eligible']
    for row in result['profiles'].values():
        assert 'seconds' not in row
        assert row['locator_counts']['locate_batch']['calls'] > 0
    for row in result['iterations'] + list(result['profiles'].values()):
        assert row['comparison']['all_arrays_exact']
        assert row['comparison']['diagnostics_all_exact']
        assert set(row['comparison']['arrays']) == {
            'cloud.' + name for name in state.cloud.array_names()
        } | {'cloud.alive', 'feedback.values', 'feedback.deltas'}
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
    assert_runtime_exact(state, before)
    assert _tree_equal(_rng(), rng)


@pytest.mark.parametrize('change', ['array', 'diagnostics', 'rng', 'input'])
def test_exact_contract_rejects_candidate_changes(change):
    source, state, fields = event_fixture(backend='batched_v1')
    advance = source.advance
    def altered(marker, *args, **kwargs):
        out, diag = advance(marker, *args, **kwargs)
        if NativeSourceMesh.locate is not reference_locate:
            if change == 'array':
                out.cloud.st[0, 0] = np.nextafter(out.cloud.st[0, 0], np.inf)
            elif change == 'diagnostics':
                diag['energy'] = np.nextafter(diag['energy'], np.inf)
            elif change == 'rng':
                random.random()
            else:
                marker.cloud.x[0, 0] += .1
        return out, diag
    source.advance = altered
    result = benchmark_source_window(source, state, 1e-9, fields, fields,
                                     repeats=1, skip_profile=True)
    assert not result['success']


def test_exception_restores_class_and_rng():
    source, state, fields = event_fixture(backend='batched_v1')
    methods = NativeSourceMesh.locate, NativeSourceMesh.locate_batch
    rng = _rng()
    def fail(*args, **kwargs):
        assert NativeSourceMesh.locate is reference_locate
        random.random()
        raise RuntimeError('fixture failure')
    source.advance = fail
    with pytest.raises(RuntimeError, match='fixture failure'):
        benchmark_source_window(source, state, 1e-9, fields, fields, repeats=1)
    assert (NativeSourceMesh.locate, NativeSourceMesh.locate_batch) == methods
    assert _tree_equal(_rng(), rng)


def test_generic_fixed_birth_object_shared_every_pass():
    from src.models.marker_state import MarkerCloud
    source, state, fields = event_fixture(backend='batched_v1')
    # Synthetic capacity/count; real runs retain every generated birth.
    cloud = MarkerCloud.empty(10, 70)
    for name in cloud.array_names():
        getattr(cloud, name)[0] = getattr(state.cloud, name)[0]
    state.cloud = cloud
    births = MarkerCloud.empty(10, 64)
    for name in births.array_names():
        getattr(births, name)[:] = getattr(cloud, name)[0]
    original = deepcopy(births)
    calls = []
    advance = source.advance
    def observed(*args, **kwargs):
        calls.append(kwargs['births'])
        return advance(*args, **kwargs)
    source.advance = observed
    result = benchmark_source_window(source, state, 1e-9, fields, fields,
                                     births=births, repeats=1, skip_profile=True)
    assert result['success']
    assert len(calls) == 4 and all(b is births for b in calls)
    for name in births.array_names():
        np.testing.assert_array_equal(getattr(births, name), getattr(original, name))


def test_empty_input_and_zero_window_rejected():
    source, state, fields = event_fixture(backend='batched_v1')
    with pytest.raises(ValueError, match='positive source window'):
        benchmark_source_window(source, state, 0., fields, fields)
    state.cloud.i_elm[:] = 0
    with pytest.raises(ValueError, match='nonzero live'):
        benchmark_source_window(source, state, 1e-9, fields, fields)


def test_cli_help_and_output_never_overwritten(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert '--skip-profile' in capsys.readouterr().out
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    output = tmp_path/'existing.json'
    output.touch()
    with pytest.raises(FileExistsError):
        main(['--output', str(output)])
    assert output.read_bytes() == b''
