#!/usr/bin/env python3
"""Budgeted native field-only supervision; no feedback consumer or legacy trainer."""
import argparse,json,random,time,resource,traceback
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_training import prepare_training_data,build_training_model
from src.data.native_window_io import atomic_json
from src.models.native_state import configuration_digest
from src.training.native_curriculum import SCOPE,LossWeights,CurriculumStage,NativeCurriculumTrainer
from src.training.native_evaluation import evaluate_banks,evaluate_window,dt_limit_diagnostic,subdivision_diagnostic


def make_trainer(bundle,cfg,seed):
    model=build_training_model(bundle,cfg['model'])
    banks={k:[w for w in ws if w.split=='train'] for k,ws in bundle.banks.items()}
    return NativeCurriculumTrainer(model,banks,weights=LossWeights(**cfg['loss']),
        stages=tuple(CurriculumStage(**dict(s,detach_before=tuple(s['detach_before']))) for s in cfg['stages']),
        seed=seed,**cfg['optimizer'],experiment_configuration=dict(scope=SCOPE,
            config_digest=configuration_digest(cfg),data_spec_digest=bundle.spec_digest,
            normalizer_version=bundle.scaler.version,training_support=bundle.scaler.training_support))


def validate_configuration(cfg):
    if cfg.get('schema')!='native-field-curriculum-experiment-v1' or cfg.get('scope')!=SCOPE:
        raise ValueError('explicit native field-only experiment required')
    if (len(cfg['seeds'])!=2 or len(set(cfg['seeds']))!=2 or any(type(s)!=int for s in cfg['seeds'])
        or cfg['threads']!=1):
        raise ValueError('two fixed integer seeds and single CPU thread required')
    stages=[CurriculumStage(**dict(s,detach_before=tuple(s['detach_before']))) for s in cfg['stages']]
    total=sum(s.updates for s in stages)
    if total>200 or stages[0].name!='smoke' or stages[0].updates!=1 or stages[0].horizon!=1:
        raise ValueError('budget <=200 including one smoke update required')
    if cfg['eval_updates']!=sorted(set(cfg['eval_updates'])) or cfg['eval_updates'][0]!=0 or cfg['eval_updates'][-1]!=total:
        raise ValueError('fixed evaluation updates must include initial and final')
    if not 0<cfg['resume_at']<total:raise ValueError('explicit intermediate optimizer boundary required')
    LossWeights(**cfg['loss'])


def run_seed(bundle,cfg,seed,out,*,resume_checkpoint=None):
    out.mkdir(parents=True,exist_ok=resume_checkpoint is not None)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    trainer=make_trainer(bundle,cfg,seed)
    evaluations=[];resume_events=[]
    if resume_checkpoint:
        trainer.load(resume_checkpoint)
        if (out/'evaluations.json').exists():
            evaluations=[r for r in json.loads((out/'evaluations.json').read_text())['evaluations'] if r['update']<=trainer.update_count]
        resume_events.append(dict(update=trainer.update_count,path=str(resume_checkpoint),kind='explicit_cli_resume'))
    total=sum(s.updates for s in trainer.stages);start=time.perf_counter();training_seconds=0.;evaluation_seconds=0.
    def evaluate():
        nonlocal evaluation_seconds
        t=time.perf_counter()
        records=evaluate_banks(trainer.model,bundle.banks,trainer.weights,
            training_run=cfg['data']['training_run'],max_windows=cfg['max_eval_windows'],include_test=trainer.update_count==total)
        evaluations.append(dict(scope=SCOPE,update=trainer.update_count,records=records))
        evaluation_seconds+=time.perf_counter()-t
        atomic_json(out/'evaluations.json',dict(scope=SCOPE,evaluations=evaluations,
            checkpoint_selection='fixed_final_budget_no_validation_or_test_selection'))
    if trainer.update_count==total:
        raise ValueError('training checkpoint already completed its fixed budget; use load() for inspection/inference')
    if trainer.update_count==0:evaluate()
    try:
        while trainer.update_count<total:
            t=time.perf_counter();log=trainer.advance();training_seconds+=time.perf_counter()-t
            if trainer.update_count==1:
                smoke_window=trainer.window_banks[1][log['window_index']]
                after=evaluate_window(trainer.model,smoke_window,trainer.weights,teacher_forced=True)
                atomic_json(out/'smoke.json',dict(scope=SCOPE,optimizer_updates_in_budget=1,before=log,
                    after=after,optimizer_changed_parameters=log['parameter_change_l2']>0))
            atomic_json(out/'training-curves.json',dict(scope=SCOPE,updates=trainer.history))
            if trainer.update_count in cfg['eval_updates']:evaluate()
            if trainer.update_count==cfg['resume_at']:
                checkpoint=out/'boundary.pt';trainer.save(checkpoint)
                # Actually reconstruct a new model/optimizer and continue the real run.
                fresh=make_trainer(bundle,cfg,seed);fresh.load(checkpoint);trainer=fresh
                resume_events.append(dict(update=trainer.update_count,path=str(checkpoint),kind='new_model_optimizer_resume'))
            if trainer.update_count%10==0:print(seed,trainer.update_count,log['stage'],'loss',log['loss'],flush=True)
        trainer.save(out/'final.pt')
        diagnostic=dt_limit_diagnostic(trainer.model,trainer.window_banks[4][0],cfg['dt_probe_factors'])
        atomic_json(out/'dt-limit.json',diagnostic)
        atomic_json(out/'subdivision.json',subdivision_diagnostic(trainer.model,trainer.window_banks[4][0]))
        result=dict(scope=SCOPE,seed=seed,optimizer_updates=trainer.update_count,
            budgets={s.name:s.updates for s in trainer.stages},parameters=sum(p.numel() for p in trainer.model.parameters()),
            core_parameters=sum(p.numel() for p in trainer.model.temporal_core.parameters()),
            training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
            elapsed_seconds=time.perf_counter()-start,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resume_events=resume_events,device='cpu',dtype='torch.float64',torch_version=torch.__version__,
            feedback_policy='not_provided_not_consumed_not_zero_imputed',source_policy='none',
            checkpoint_selection='fixed_final_budget',physical_BC='unverified',
            caveat='field history may encode omitted drive effects; no learned source-response, stability, accuracy or C0 claim')
        atomic_json(out/'summary.json',result);return result
    except Exception as e:
        atomic_json(out/'failure.json',dict(scope=SCOPE,seed=seed,completed_updates=trainer.update_count,
            exception=repr(e),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume-checkpoint',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate_configuration(cfg)
    if args.seed is not None and args.seed not in cfg['seeds']:parser.error('seed must belong to fixed configuration')
    if args.resume_checkpoint is not None and args.seed is None:parser.error('resume requires explicit configured seed')
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'experiment.json').exists() and configuration_digest(json.loads((args.output_dir/'experiment.json').read_text()))!=configuration_digest(cfg):
        raise ValueError('existing experiment configuration mismatch')
    atomic_json(args.output_dir/'experiment.json',cfg)
    bundle=prepare_training_data(cfg['data'],args.output_dir/'data')
    print(json.dumps(dict(scope=SCOPE,records=bundle.coverage['raw_records'],
        windows=bundle.coverage['window_counts'],exclusions=bundle.coverage['exclusions']),ensure_ascii=False),flush=True)
    if args.audit_only:return
    seeds=cfg['seeds'] if args.seed is None else [args.seed]
    results=[run_seed(bundle,cfg,s,args.output_dir/f'seed-{s}',resume_checkpoint=args.resume_checkpoint) for s in seeds]
    atomic_json(args.output_dir/'run-summary.json',dict(scope=SCOPE,seeds=results))


if __name__=='__main__':main()
