"""Public replay evidence comparison, not a physics-accuracy gate."""
import json
import numpy as np


def test_state_report_separates_raw_increment_and_per_channel_relative_errors():
    from copy import deepcopy
    from tools.diagnose_native_production_chain import build_teacher_fixture
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.report_native_replay_pair import compare_states
    model,source,shard,runtime,teacher=build_teacher_fixture(2)
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=teacher)
    before=adapter.initialize(runtime);reference=deepcopy(before);candidate=deepcopy(before)
    reference.runtime.fluid_deltas.fill(1.);candidate.runtime.fluid_deltas.fill(1.)
    candidate.runtime.fluid_deltas[0]*=1.01
    reference.runtime.marker.feedback.values.fill(2.)
    candidate.runtime.marker.feedback.values.fill(2.)
    candidate.runtime.marker.feedback.values[1]*=1.1
    result=compare_states(before,reference,candidate)
    assert result['fields']['psi']['values']['relative_l2']==0.
    assert np.isclose(result['fields']['psi']['raw_deltas']['relative_l2'],.01)
    assert np.isclose(result['projection']['rad_sink']['values']['relative_l2'],.1)
    assert result['projection']['dE_ion']['values']['relative_l2']==0.
    json.dumps(result,allow_nan=False)
