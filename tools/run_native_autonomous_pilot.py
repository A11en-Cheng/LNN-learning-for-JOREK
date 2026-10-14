"""Execute the fixed two-update coupled pilot AFTER both autonomous smokes."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from tools.diagnose_native_real_chain import Stages,_initial_pairing
from tools.native_real_chain_builder import build_real
from tools.train_native_autonomous_pilot import PilotWindow,PilotTrainer,read_pilot_labels
from src.data.base_coupled import StepControl
from src.training.base_coupled import TorchRecomputedSourceEvents


def build_windows(config,stages):
    args=SimpleNamespace(chunk_size=2048,wall_seconds=900.,memory_mib=8192,
        birth_max_proposals=1000000,spatial_backend='batched_v1')
    windows=[]
    for spec in config['windows']:
        if spec['split']!='train':continue
        run=Path(spec['run']);anchor=spec['anchor']
        rows=_initial_pairing(run,anchor,stages)
        model,source,shard,runtime,provider=build_real(run,anchor,rows,stages,args,feedback_mode='autonomous')
        controls=tuple(StepControl(c['start_si'],c['dt_si'],c['end_si'],c['origin'],c['source'],phase=c['phase']) for c in rows)
        nodes,_=read_pilot_labels(run,[row['event_index'] for row in rows],gamma=source.mesh.units.gamma)
        if any(node.targets['shard_positions']['ids']!=runtime.shards['ids'].tolist() for node in nodes):
            raise ValueError('recorded shard identity differs from initial fixed slots')
        adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=provider)
        windows.append(PilotWindow(f'{run.name}:{anchor}',adapter,runtime,controls,tuple(nodes),
            split='train',feedback_mode='autonomous'))
    if len(windows)!=2:raise ValueError('exactly the two declared train windows required')
    return windows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/native-autonomous-training-pilot.json'))
    parser.add_argument('--smoke-root',type=Path,default=Path('output/native-autonomous-smoke-2026-10-09'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    config=json.loads(args.config.read_text());stages=Stages();start=time.monotonic()
    result=dict(success=False,optimizer_updates=0,teacher=False,future_truth_feedback=False,
        seed=1,burnin_steps=0,burnin_selection='not_performed_this_two_update_pilot',
        validation_evaluated=False,test_evaluated=False,training_plan='two_full_base_events_AR_one_update_per_run',
        scope='bounded_coupled_training_mechanism_not_convergence_or_C0',config=config,updates=[])
    trainer=None;phase='smoke_gate'
    def save_json(name,value):
        with (args.output_dir/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    try:
        if config.get('teacher') is not False or config.get('truth_field_feedback') is not False:
            raise ValueError('explicit no-teacher training configuration required')
        declared={(str(Path(w['run']).resolve()),w['anchor']) for w in config['windows'] if w['split']=='train'}
        smoked=set()
        for name,anchor in [('v400-run2',4110),('injInner-run2',5680)]:
            proof=json.loads((args.smoke_root/name/'result.json').read_text())
            if (not proof['success'] or proof['completed_events']!=2 or proof['future_truth_feedback']
                or proof['anchor']!=anchor or proof['optimizer_updates']!=0):
                raise ValueError('matching completed autonomous smoke required before optimization')
            smoked.add((str(Path(proof['run']).resolve()),proof['anchor']))
        if declared!=smoked:raise ValueError('training windows differ from completed smoke windows')
        phase='build_training_windows'
        trainer=PilotTrainer(build_windows(config,stages),split_config=config,seed=1)
        save_json('normalization.json',trainer.normalization)
        result['parameter_count']=sum(p.numel() for p in trainer.windows[0].adapter.model.parameters())
        result['gradient_scope']=trainer.gradient_scope
        phase='pre_evaluation';stages.emit(phase,'started')
        result['before']=trainer.evaluate();save_json('before.json',result['before'])
        stages.emit(phase,'completed',losses=[r['loss'] for r in result['before']])
        phase='train_update_1';stages.emit(phase,'started')
        row=trainer.update(0);result['updates'].append(row);result['optimizer_updates']=trainer.cursor
        save_json('update-1.json',row);trainer.save(args.output_dir/'update-1.pt')
        stages.emit(phase,'completed',loss=row['loss'],gradient_norm=row['gradient_norm'])
        phase='fresh_constructor_restore';stages.emit(phase,'started')
        del trainer;gc.collect()
        trainer=PilotTrainer(build_windows(config,stages),split_config=config,seed=1)
        trainer.load(args.output_dir/'update-1.pt')
        if trainer.cursor!=1:raise ValueError('saved update cursor not restored')
        result['fresh_constructor_resume']=True
        stages.emit(phase,'completed')
        phase='train_update_2';stages.emit(phase,'started')
        row=trainer.update(1);result['updates'].append(row);result['optimizer_updates']=trainer.cursor
        save_json('update-2.json',row);trainer.save(args.output_dir/'update-2.pt')
        stages.emit(phase,'completed',loss=row['loss'],gradient_norm=row['gradient_norm'])
        phase='post_evaluation';stages.emit(phase,'started')
        result['after']=trainer.evaluate();save_json('after.json',result['after'])
        stages.emit(phase,'completed',losses=[r['loss'] for r in result['after']])
        result['success']=True
    except Exception as exc:
        import traceback
        result.update(error_type=type(exc).__name__,error=str(exc),failure_phase=phase)
        traceback.print_exc()
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        save_json('result.json',result)
        print(json.dumps({k:result.get(k) for k in ('success','optimizer_updates','elapsed_seconds','failure_phase','error')}),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
