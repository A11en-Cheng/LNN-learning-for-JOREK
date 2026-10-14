"""Diagnostic control/fork/output contracts without real-cloud timing jobs."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from tools.benchmark_native_element_functions import acceptance, execute


def test_zero_reference_differences_and_discrete_failures_cannot_pass():
    from tools.native_replay_metrics import relative_metrics
    r=relative_metrics(np.zeros(2), np.ones(2))
    report=dict(fields={'field':r}, shards={}, feedback={'values':{},'deltas':{}},
        cloud_full_slots={k:{'relative_metrics':relative_metrics(np.ones(2),np.ones(2))}
                          for k in ('x','st','V_prev','P_imp')},
        integer_events_exact=True,clocks_exact=True,transport_history_exact=True)
    assert not acceptance(report)['passed']
    report['fields']['field']=relative_metrics(np.ones(2),np.ones(2))
    assert acceptance(report)['passed']
    report['transport_history_exact']=False
    assert not acceptance(report)['passed']


def test_real_event_forks_use_unique_artifact_names_and_do_not_mutate_prefix(tmp_path, monkeypatch):
    from tests.test_base_coupled_training import event_fixture
    from src.models.native_coupling import NativeMarkerSource
    import tools.benchmark_native_element_functions as benchmark
    import tools.native_real_chain_builder as builder
    def fixture_builder(*args,**kwargs):
        model,old,shard,runtime=event_fixture(n=3)
        source=NativeMarkerSource(old.mesh,atomic=old.atomic,
            atomic_version=old.configuration['atomic'],potentials_ev=old.potentials_ev,
            F0=old.F0,diffusion_ju=old.diffusion_ju,spatial_backend='batched_v1')
        model.source_configuration=deepcopy(source.configuration)
        runtime.native_context.configuration['model']=model.checkpoint_configuration
        return model,source,shard,runtime,None
    rows=[dict(start_si=(i+1)*1e-9,end_si=(i+2)*1e-9,dt_si=1e-9,
        origin='external_control',source='fixture',phase='ordinary') for i in range(3)]
    monkeypatch.setattr(builder,'build_real',fixture_builder)
    monkeypatch.setattr(benchmark,'recorded_controls',lambda *args: rows)
    monkeypatch.setattr(benchmark,'sampling_probe',lambda *args: dict(warm_queries=[],coefficient_bytes=0))
    args=SimpleNamespace(device='cpu',threads=1,seed=4810,run_index=0,output_dir=tmp_path,repeats=2)
    result={}
    execute(args,result)
    assert len(result['iterations'])==4
    assert len(result['comparisons'])==2
    assert all(len(row['events'])==2 for row in result['iterations'])
    assert all(row['rng_exact'] for row in result['comparisons'])
    assert result['prefix_unchanged'] and result['weights_unchanged']
    assert len(list(tmp_path.glob('progress-*.json')))==4
