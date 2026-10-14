"""Diagnostic worker pool acceptance through complete public source events."""
from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_locator_events import event_fixture, assert_runtime_exact
from tools.benchmark_native_cpu_threads import ThreadedKernels, stable_diagnostics


@pytest.mark.parametrize('workers', [2, 4, 8])
@pytest.mark.parametrize('scenario', ['normal', 'crossing', 'lost'])
def test_threaded_complete_event_preserves_state_and_history(workers, scenario):
    source, state, fields = event_fixture(backend='batched_v1', scenario=scenario)
    # Eight live slots force every worker through the public sample/locate/atomic path.
    for name in state.cloud.array_names():
        a = getattr(state.cloud, name)
        setattr(state.cloud, name, np.repeat(a[:1], 10, axis=0))
    state.cloud.i_life[:8] = np.arange(1, 9)
    state.cloud.weight[:8] *= np.linspace(.5, 1.5, 8)
    before = deepcopy(state)
    expected, diagnostic = source.advance(state, 1e-9, fields, fields)
    with ThreadedKernels(source, workers) as kernels:
        actual, got_diagnostic = kernels.adapter.advance(state, 1e-9, fields, fields)
        assert len(kernels.worker_ids) == workers
        assert kernels.rows['sample'] >= 8
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(state, before)
    from tools.diagnose_native_real_chain import _array_digest
    assert _array_digest(stable_diagnostics(got_diagnostic)) == _array_digest(stable_diagnostics(diagnostic))


def test_worker_failure_does_not_commit_and_retry_uses_current_fields():
    source, state, fields = event_fixture(backend='batched_v1')
    for name in state.cloud.array_names():
        a = getattr(state.cloud, name)
        setattr(state.cloud, name, np.repeat(a[:1], 10, axis=0))
    original = source.atomic
    def fail(te, ne):
        raise RuntimeError('prescribed worker failure')
    before = deepcopy(state)
    source.atomic = fail
    with ThreadedKernels(source, 4) as kernels:
        with pytest.raises(RuntimeError, match='prescribed worker failure'):
            kernels.adapter.advance(state, 1e-9, fields, fields)
    assert_runtime_exact(state, before)
    source.atomic = original
    fields[4, 0] = 2.
    expected, _ = source.advance(state, 1e-9, fields, fields)
    with ThreadedKernels(source, 4) as kernels:
        actual, _ = kernels.adapter.advance(state, 1e-9, fields, fields)
    assert_runtime_exact(actual, expected)
