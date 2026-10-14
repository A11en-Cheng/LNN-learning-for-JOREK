"""Public complete-source evidence and explicit opt-in CLI contracts."""
from copy import deepcopy

import numpy as np
import pytest

from src.models.native_nearby_locator import NativeNearbyMarkerSource
from tests.test_native_nearby_locator import saved_chain_neighbours
from tests.test_native_relocation_work import near_edge_event
from tools.benchmark_native_neighbor_locator import benchmark_source_window
from tools.diagnose_native_nearby_difference import diagnose


def test_nearby_benchmark_reports_all_native_axes_and_independent_counters():
    source,state,fields=near_edge_event()
    candidate=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    result=benchmark_source_window(source,candidate,state,1e-9,fields,fields,repeats=1,count_pass=True)
    assert result['success'] and result['inputs_unchanged']
    assert result['locator_counts']['neighbor']['global_fallback_points']==0
    assert result['locator_counts']['neighbor']['edge_crossings']==1
    assert set(result['iterations'][-1]['comparison']['feedback_channels']['values'])==set(map(str,range(5)))
    assert result['count_pass_comparisons']['neighbor']['required_exact']['cloud.i_elm']


def test_diagnosis_compares_same_targets_without_changing_source_path():
    source,state,fields=near_edge_event();before=deepcopy(state)
    candidate=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    result=diagnose(source,candidate,state,1e-9,fields,fields,record={})
    assert result['success'] and result['inputs_unchanged']
    assert result['same_target_checked_visits']==1
    assert result['same_target_element_mismatch_visits']==0
    assert result['discrete_final_mismatch_count']==0
    np.testing.assert_array_equal(state.cloud.x,before.cloud.x)


def test_diagnosis_retains_expected_lost_semantic_difference():
    source,state,fields=near_edge_event();fields[1]*=1000;state.cloud.V_prev*=1000
    candidate=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    result=diagnose(source,candidate,state,1e-9,fields,fields,record={})
    assert not result['success']
    assert result['inputs_unchanged']
    assert result['discrete_final_mismatch_count']==1
    assert result['same_target_element_mismatch_visits']==1
    assert result['same_target_examples'][0]['baseline']['element']==0
    assert result['same_target_examples'][0]['nearby']['element']==-2


def test_benchmark_help_exposes_bounded_paired_comparison(capsys):
    from tools.benchmark_native_nearby import main
    with pytest.raises(SystemExit) as error:main(['--help'])
    assert error.value.code==0
    output=capsys.readouterr().out
    assert '--run-index' in output and '--seed' in output and '--repeats' in output


def test_diagnosis_cli_provides_reused_builder_arguments(tmp_path,monkeypatch):
    import tools.diagnose_native_nearby_difference as tool
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        monkeypatch.setenv(name,'1')
    def fake_execute(args,result):
        assert args.repeats==1 and args.marker_locator=='native'
        assert args.chunk_size==2048
        result['success']=True
    monkeypatch.setattr(tool.benchmark,'execute',fake_execute)
    assert tool.main(['--output',str(tmp_path/'probe.json')])==0


def test_all_point_diagnosis_does_not_exclude_successful_same_element_updates():
    source,state,fields=near_edge_event();fields[1]*=.01;state.cloud.V_prev*=.01
    candidate=NativeNearbyMarkerSource(source,neighbours=saved_chain_neighbours())
    result=diagnose(source,candidate,state,1e-9,fields,fields,record={},all_points=True)
    assert result['success'] and result['inputs_unchanged']
    assert result['all_points_compared']
    assert result['same_target_checked_visits']==1
    assert result['top_same_element_local_differences']
    assert 'sampled_velocity_relative_l2' in result['top_same_element_local_differences'][0]
