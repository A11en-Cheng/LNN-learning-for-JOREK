from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.models.marker_state import (MarkerCloud, MarkerRuntime, MarkerClock,
    ProjectionFeedback, advance_window)
from tools.validate_native_birth_windows import (relative_error, distribution,
    bootstrap_ratios, eion_diagnostics, load_record)


def test_independent_marker_fixture_linear_conditional_decomposition():
    cloud = MarkerCloud.empty(10,8)
    cloud.i_elm[:2] = 1
    cloud.weight[:2] = [2,3]
    cloud.x[:2,0] = 1
    cloud.V_prev[:2,0] = 1
    born = MarkerCloud.empty(10,1)
    born.i_elm[:] = 1
    born.weight[:] = 4
    born.x[:,0] = 1
    born.V_prev[:,0] = 1
    feedback = ProjectionFeedback(np.ones((5,2)),np.zeros((5,2)),0)
    clock = MarkerClock(0,0,.01,0,.04)
    state = MarkerRuntime(cloud,feedback,clock)
    def sample(t,c,ids,old):
        assert np.array_equal(old.values,feedback.values)
        return dict(velocity=np.tile([1.,0,0],(len(ids),1)),ne_si=np.ones(len(ids)),
            scd=np.full((len(ids),10),.1),acd=np.zeros((len(ids),10)),
            radiation_w_per_atom=np.full((len(ids),11),.2))
    def deposit(c,ids,loads):
        basis = np.column_stack((1+c.x[ids,0],2-c.x[ids,0]))
        return loads.T@basis
    def locate(x):
        return np.ones(len(x),int),np.zeros((len(x),2))
    def project(rhs):
        return rhs@np.array([[2.,.1],[.1,3.]])
    def advance(s,births=None):
        return advance_window(s,.04,sample=sample,locate=locate,deposit=deposit,
            project=project,potentials_ev=np.arange(1,11),t_norm=1,
            central_density=1,births=births)
    whole,_ = advance(state,born)
    existing,_ = advance(state)
    compact,_ = advance(MarkerRuntime(deepcopy(born),feedback,deepcopy(clock)))
    assert relative_error(whole.feedback.values,existing.feedback.values+compact.feedback.values)<1e-14
    assert len(whole.cloud.weight)==8
    assert whole.cloud.i_life[2]==1
    assert np.array_equal(state.cloud.i_life,np.zeros(8))
    assert np.allclose(whole.cloud.P_imp[2],compact.cloud.P_imp[0])


def test_relative_error_has_no_epsilon_zero_normalization():
    assert relative_error(np.zeros(2),np.ones(2)) is None
    assert relative_error(np.array([1e-30]),np.array([2e-30])) == 1


def test_statistics_and_bootstrap():
    assert distribution([1,2,3,None])['median']==2
    assert distribution([None])==dict(undefined=1)
    result = bootstrap_ratios([2,4,6],[1,2,3],2917)
    assert result['ratio_of_mean_errors']==2
    assert result['percentile95']==[2,2]
    assert bootstrap_ratios([1,2],[0,0],1)['status']=='undefined_zero_or_missing_noise'


def test_eion_signed_and_shape_diagnostics():
    a=np.array([1.,-1])
    row=eion_diagnostics(a,2*a,np.zeros(2),a,np.ones(2))
    assert row['signed_denominator_flag']=='zero'
    assert row['newborn_signed_integral_fraction'] is None
    assert row['cancellation']==1
    assert row['best_scale']==2
    assert row['shape_residual_relative']==0


def test_source_record_unknown_units_restart_and_missing_fail_closed(tmp_path):
    units=SimpleNamespace(t_norm_si=1.,n_norm_m3=1e20,central_mass=1.)
    with pytest.raises(ValueError,match='missing physical'):
        load_record(tmp_path,2,units,[])
    with h5py.File(tmp_path/'jorek000002.h5','w') as f:
        for key,value in dict(index_now=2,t_norm=1,central_density=1,central_mass=1,
                              t_now=2,tstep=1).items():
            f[key]=[value]
    with pytest.raises(ValueError,match='restart'):
        load_record(tmp_path,2,units,[2])
    with pytest.raises(ValueError,match='units'):
        load_record(tmp_path,2,SimpleNamespace(t_norm_si=2,n_norm_m3=1e20,central_mass=1),[])
