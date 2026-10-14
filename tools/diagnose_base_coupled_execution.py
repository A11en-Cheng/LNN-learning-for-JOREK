"""Synthetic mechanism diagnostic using actual Adapter; no optimizer or real eligibility."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from tests.test_base_coupled_training import event_fixture,sequence
from src.training.base_coupled import TorchRecomputedSourceEvents,run_sequence,save_session,load_session
from src.training.coupled_metrics import coupled_node_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():raise ValueError('new artifact directory required')
    a.output_dir.mkdir(parents=True);torch.manual_seed(17)
    model,source,shard,runtime=event_fixture(n=2);adapter=TorchRecomputedSourceEvents(model,source,shard)
    seq=sequence(n=2,burnin=0);initial=adapter.initialize(runtime)
    initial.values.requires_grad_();initial.memory.requires_grad_()
    started=time.perf_counter();first,_=adapter.advance(initial,seq.controls[0]);first.values.retain_grad();first.memory.retain_grad()
    second,_=adapter.advance(first,seq.controls[1]);second.values.square().sum().backward()
    gradient=dict(first_values=float(first.values.grad.abs().sum()),first_memory=float(first.memory.grad.abs().sum()),
        initial_memory=float(initial.memory.grad.abs().sum()),core_parameters=sum(float(p.grad.abs().sum()) for p in model.temporal_core.parameters() if p.grad is not None))
    whole=run_sequence(seq,adapter,adapter.initialize(runtime));part=run_sequence(seq,adapter,adapter.initialize(runtime),stop_cursor=1)
    save_session(a.output_dir/'session.pt',seq,adapter,part)
    fresh=TorchRecomputedSourceEvents(deepcopy(model),deepcopy(source),deepcopy(shard))
    resumed=run_sequence(seq,fresh,session=load_session(a.output_dir/'session.pt',seq,fresh))
    assert torch.equal(whole.state.values,resumed.state.values) and torch.equal(whole.state.memory,resumed.state.memory)
    predictions=adapter.prediction(second)
    # No synthetic truth is manufactured: missing object labels stay missing.
    metrics=coupled_node_metrics(predictions,{},evaluator=model.feature_path.evaluator,
        normalization_source='no_fitted_statistics',region_name='all_elements')
    result=dict(schema='base-coupled-execution-diagnostic-v1',scope='small_cloud_fixture_actual_adapter_not_phaseA',
        optimizer_updates=0,events_completed=2,gradient_scope=adapter.gradient_scope,gradient=gradient,
        exact_new_model_restore=True,events=[dict(start_si=c.start_si,end_si=c.end_si,dt_si=c.dt_si) for c in seq.controls],
        metrics=metrics,parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (a.output_dir/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
