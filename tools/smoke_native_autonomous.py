"""Full-cloud autonomous two-event smoke. No optimizer or truth-field feedback."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_native_real_chain import Stages,_initial_pairing,_trace_public_stages,_compact_event,_array_digest
from tools.native_real_chain_builder import build_real


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--anchor',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,default=900.)
    p.add_argument('--memory-mib',type=int,default=8192)
    p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--birth-max-proposals',type=int,default=1000000)
    p.add_argument('--birth-position-mode',choices=['native_probability_n0','native_probability_periodic','native_rejection_reference'],
        default='native_probability_n0')
    p.add_argument('--birth-cells-per-width',type=int,default=64)
    p.add_argument('--birth-max-cells',type=int,default=16_000_000)
    p.add_argument('--birth-phi-cells',type=int,default=64)
    p.add_argument('--seed',type=int,default=20261009,help='Initialize Python/NumPy/Torch once; checkpoints retain the evolving RNG state.')
    p.add_argument('--spatial-backend',choices=['scalar','batched_v1'],default='batched_v1')
    args=p.parse_args()
    if not 0<=args.seed<2**32:p.error('--seed must be in [0, 2**32)')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    stages=Stages();start=time.monotonic()
    result=dict(scope='autonomous_full_cloud_untrained_smoke',optimizer_updates=0,
        future_truth_feedback=False,success=False,events=[],run=str(args.run),anchor=args.anchor,
        seed=args.seed,rng_policy='seed_once_at_run_start_checkpoint_full_state_no_per_step_reset',
        initial_rng_digest=_array_digest(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())))
    try:
        rows=_initial_pairing(args.run,args.anchor,stages)
        model,source,shard,runtime,provider=build_real(args.run,args.anchor,rows,stages,args,feedback_mode='autonomous')
        result['equilibrium_configuration']=provider.configuration
        result['source_configuration']=source.configuration
        from src.training.base_coupled import TorchRecomputedSourceEvents
        from src.data.base_coupled import StepControl,read_recorded_node
        from src.training.coupled_metrics import coupled_node_metrics
        adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=provider)
        _trace_public_stages(model,source,shard,stages)
        state=adapter.initialize(runtime);del runtime
        for row in rows:
            control=StepControl(row['start_si'],row['dt_si'],row['end_si'],row['origin'],row['source'],phase=row['phase'])
            with stages.track('autonomous_complete_event',event_index=row['event_index']):
                previous=state
                with torch.no_grad():state,diagnostic=adapter.advance(state,control)
                stages.completed_events+=1
            evidence=_compact_event(previous.runtime,state,diagnostic)
            result['events'].append(evidence)
            with (args.output_dir/f"committed-{stages.completed_events}.json").open('x') as stream:
                json.dump(evidence,stream,indent=2,allow_nan=False)
            if evidence['branch_intervention'] is not None:raise ValueError('truth feedback forbidden')
            # Labels are first opened AFTER the complete event has committed.
            stages.current='labels_and_metrics'
            import h5py
            with h5py.File(row['source'],'r') as labels:
                true_alive=(labels['spi_radius_arr'][...]>0).tolist()
            node,label_metadata=read_recorded_node(row['source'],gamma=source.mesh.units.gamma,
                projection_path=args.run/f"projections{row['event_index']-1:06d}.h5",
                shard_identity=dict(ids=state.runtime.shards['ids'].tolist(),
                    alive=true_alive,provenance='fixed_initial_slot_identity_recorded_radius_liveness'))
            target=dict(node.targets,marker_time_si=node.object_times_si['marker_projection'],
                shard_time_si=node.object_times_si['shard_positions'])
            prediction=adapter.prediction(state)
            evidence['metrics']=coupled_node_metrics(prediction,target,
                evaluator=model.field.feature_path.evaluator,normalization_source='native_JOREK_units_no_fitted_scales',
                region_name='all',indices=source.mesh.indices)
            with (args.output_dir/f"event-{stages.completed_events}.json").open('x') as stream:
                json.dump(evidence,stream,indent=2,allow_nan=False)
            del previous
        result['success']=True
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__,error=str(exc),stage=stages.current)
        traceback.print_exc()
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,completed_events=stages.completed_events)
        result['final_rng_digest']=_array_digest(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()))
        with (args.output_dir/'result.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(success=result['success'],completed_events=stages.completed_events,
            elapsed_seconds=result['elapsed_seconds'],error=result.get('error'))),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
