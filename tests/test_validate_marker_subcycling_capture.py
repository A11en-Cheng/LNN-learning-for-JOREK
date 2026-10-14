from copy import deepcopy
import numpy as np
import pytest
from tests.test_marker_state import runtime
from tools.validate_marker_subcycling_capture import compare


def test_increment_report_separates_rz_phi_weight_and_all_projection_channels():
    initial=runtime();initial.cloud.P_imp=np.pad(initial.cloud.P_imp,((0,0),(0,9)))
    initial.feedback.values=np.zeros((5,4,1,1));initial.feedback.deltas=np.zeros_like(initial.feedback.values)
    ref=deepcopy(initial);got=deepcopy(initial)
    ref.cloud.x[0]+=[.1,.2,.3];got.cloud.x[0]+=[.101,.2,.3]
    ref.cloud.V_prev[0]+=[1.,2.,3.];got.cloud.V_prev[0]+=[1.1,2.,3.]
    ref.feedback.values[:]=1.;got.feedback.values[:]=1.01
    result=compare(ref,got,initial)
    assert result['identity_exact']
    assert result['increments']['x']['R']['relative_l2']==pytest.approx(.01)
    assert result['increments']['x']['phi']['relative_l2']==0
    assert 'RZ_vector' in result['increments']['x']
    assert 'vector' not in result['increments']['x']
    assert result['increments']['V_prev']['atom_weighted_vector']['relative_l2']==pytest.approx(.1/np.sqrt(14.))
    assert len(result['projection']['values'])==len(result['projection']['deltas'])==5
