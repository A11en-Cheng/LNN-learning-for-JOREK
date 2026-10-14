"""Benchmark-only frozen reference at existing mesh/source public seams."""
from copy import deepcopy
import numpy as np


def test_reference_binding_is_instance_local_and_exact_for_fields_and_deposition():
    from tests.test_native_coupling import native_fixture
    from tools.native_spatial_reference import bind_cpu_shared_reference
    mesh, source, state, fields = native_fixture()
    source.spatial_backend = 'batched_v1'
    other = deepcopy(mesh)
    e = np.array([0, 0]); st = np.array([[.2, .7], [.6, .1]])
    arrays = (fields, fields[:5, :, :, ::-1])
    want = mesh.points_many(e, st, arrays)
    before = deepcopy(source.configuration)
    bind_cpu_shared_reference(source)
    assert source.configuration == before
    assert 'points_many' not in other.__dict__
    for outputs, expected in zip(mesh.points_many(e, st, arrays), want):
        for a, b in zip(outputs, expected):
            np.testing.assert_array_equal(a, b)
    ids = np.flatnonzero(state.cloud.alive)
    loads = np.ones((len(ids), 5))
    np.testing.assert_array_equal(mesh.deposit_batch(state.cloud, ids, loads), other.deposit_batch(state.cloud, ids, loads))


def test_reference_complete_source_event_preserves_state_and_rng():
    from tests.test_native_coupling import native_fixture
    from tools.native_spatial_reference import bind_cpu_shared_reference
    from tools.diagnose_native_real_chain import _array_digest
    mesh, source, state, fields = native_fixture()
    source.spatial_backend = 'batched_v1'
    reference = deepcopy(source)
    bind_cpu_shared_reference(reference)
    rng = np.random.get_state()
    want = source.advance(state, 1e-9, fields, fields)
    expected_rng = _array_digest(np.random.get_state())
    np.random.set_state(rng)
    got = reference.advance(state, 1e-9, fields, fields)
    assert _array_digest(got[0]) == _array_digest(want[0])
    for name in want[1]:
        if name == 'execution':
            for counter in ('chunks', 'processed_particle_steps', 'slots'):
                assert got[1][name][counter] == want[1][name][counter]
        else:
            assert _array_digest(got[1][name]) == _array_digest(want[1][name])
    assert _array_digest(np.random.get_state()) == expected_rng
