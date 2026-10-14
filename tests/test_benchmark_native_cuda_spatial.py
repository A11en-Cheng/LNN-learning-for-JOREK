import numpy as np

from tests.test_native_coupling import native_fixture


def test_spatial_probe_keeps_every_point_and_reports_public_outputs():
    from tools.benchmark_native_cuda_spatial import benchmark_spatial
    mesh, _, _, fields = native_fixture()
    local = np.array([[.1, .2], [.3, .4], [.8, .9]])
    positions = np.column_stack([2+local[:, 0], local[:, 1], np.zeros(3)])
    hints = np.ones(3, dtype=np.int32)
    result = benchmark_spatial(mesh, mesh, fields, positions, hints, local,
        chunk_size=2, repeats=1, synchronize=lambda: None)
    assert result['point_count'] == 3
    assert result['points']['values_exact'] and result['points']['gradients_exact']
    assert set(result['points']['per_field']) == {'psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e'}
    for case in result['locate']:
        assert case['point_count'] == 3
        assert case['element_ids_exact'] and case['local_coordinate_max_abs_difference'] == 0.
    assert result['locate'][-1]['forced_invalid_hints'] == 1
    assert result['hot_resident']['status'] == 'not_available_no_residency_API'


def test_spatial_probe_exercises_owned_resident_fields_on_cpu_test_device():
    from tools.benchmark_native_cuda_spatial import benchmark_spatial
    from src.models.native_cuda_spatial import NativeCudaSpatial
    mesh, _, _, fields = native_fixture()
    st = np.array([[.2, .4], [.8, .7]])
    positions = np.column_stack([2+st[:, 0], st[:, 1], np.zeros(2)])
    candidate = NativeCudaSpatial(mesh, device='cpu')
    result = benchmark_spatial(mesh, candidate, fields, positions, np.ones(2, dtype=np.int32), st,
        chunk_size=1, repeats=1)
    assert result['hot_resident']['status'] == 'measured'
    for channel in result['hot_resident']['per_field'].values():
        for metric in channel.values():
            assert metric['relative_l2'] < 1e-13
    assert all(case['element_ids_exact'] for case in result['locate'])
    import json
    json.dumps(result, allow_nan=False)
