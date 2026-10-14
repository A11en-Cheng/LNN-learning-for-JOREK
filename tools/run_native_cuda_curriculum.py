"""Bounded CUDA curriculum with validation-only memory-recovery comparison."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from tools.run_native_autonomous_pilot import build_windows
from tools.native_real_chain_builder import build_real
from tools.diagnose_native_real_chain import Stages,_initial_pairing
from tools.train_native_autonomous_pilot import PilotWindow,PilotTrainer,read_pilot_labels,_install_normalization,_rng,_restore_rng
from tools.native_memory_burnin import make_prefix,forecast
from src.data.base_coupled import StepControl
from src.training.base_coupled import TorchRecomputedSourceEvents
from src.training.coupled_metrics import recorded_supervision


def extend(window,spec,registry):
    root=Path(spec['run']);anchor=spec['anchor'];controls=list(window.controls)
    for index in range(anchor+4,anchor+6):
        if index in registry[root.name]['restart_events']:raise ValueError('forecast crosses registered restart')
        path=root/f'jorek{index:06d}.h5'
        with h5py.File(path,'r') as f:
            scalar=lambda k:float(np.asarray(f[k][...]).ravel()[0])
            end=scalar('t_now')*scalar('t_norm');dt=scalar('tstep')*scalar('t_norm')
            if scalar('index_now')!=index:raise ValueError('field internal index mismatch')
        controls.append(StepControl(controls[-1].end_si,dt,end,'recorded_schedule_replay',str(path)))
    window.controls=tuple(controls)
    window.runtime.step_plan_si=tuple(c.dt_si for c in controls)
    window.nodes,_=read_pilot_labels(root,range(anchor+2,anchor+6),gamma=window.adapter.model.units.gamma)
    window.runtime.validate_fluid_history()


def observe(trainer,w,predictions):
    s=trainer.normalization
    scales=dict(coefficients=s['state_scale'],raw_history=s['history_scale'],gauss_values=s['gauss']['value_scale'],
        compatible_gradients=s['gauss']['gradient_scale'],provenance='two_train_initials_only')
    losses=[];rows=[]
    for step,(prediction,node) in enumerate(zip(predictions,w.nodes[2:]),1):
        o=recorded_supervision(prediction,node,{k:True for k in node.targets},
            evaluator=w.adapter.model.feature_path.evaluator,scales=scales,
            weights=dict(coefficients=1.,raw_history=1.,gauss_values=1.,compatible_gradients=1.),
            normalization_source=scales['provenance'],region_name='full_native_support',
            indices=w.adapter.model.field.update_mapping.indices.cpu().numpy())
        losses.append(o['loss'].total)
        rows.append(dict(forecast_step=step,time_si=node.time_si,loss=o['loss'].report(),metrics=o['metrics']))
    return torch.stack(losses).mean(),rows


def instrument(model,source,stats):
    old_source=source.advance;old_forward=model.forward
    def source_call(*a,**kw):
        start=time.perf_counter()
        try:return old_source(*a,**kw)
        finally:stats['source_seconds']+=time.perf_counter()-start
    def forward(*a,**kw):
        start=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();out=old_forward(*a,**kw);end.record();end.synchronize()
        stats['field_cuda_stream_ms']+=begin.elapsed_time(end)
        stats['field_wall_seconds']+=time.perf_counter()-start;stats['field_calls']+=1
        return out
    source.advance=source_call;model.forward=forward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3)
    args=p.parse_args()
    if args.epochs!=3:raise ValueError('fixed first curriculum budget: three epochs, two windows each')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; never silently fall back to CPU')
    args.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    config=json.loads(Path('config/native-autonomous-training-pilot.json').read_text())
    registry=json.loads(Path(config['restart_registry']).read_text())['runs'];stages=Stages()
    specs=[w for w in config['windows'] if w['split']=='train'];vspecs=[w for w in config['windows'] if w['split']=='val']
    stats=dict(source_seconds=0.,field_wall_seconds=0.,field_cuda_stream_ms=0.,field_calls=0,backward_wall_seconds=0.)
    result=dict(success=False,optimizer_updates=0,epochs=[],burnin_comparison=[],device=torch.cuda.get_device_name(0),
        dtype='float64',teacher=False,test_evaluated=False,epoch_definition='one_pass_over_two_declared_train_windows_NOT_full_phaseA',
        source_gradient='stop_gradient_numpy',profiling=stats)
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            try:
                s=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=5)
                samples.append(dict(time=time.time(),global_gpu_csv=s.stdout.strip(),returncode=s.returncode))
            except Exception as exc:samples.append(dict(time=time.time(),error=str(exc)))
            stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.monotonic();phase='build'
    def write(name,value):
        with (args.output_dir/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    try:
        trainer=PilotTrainer(build_windows(config,stages),split_config=config,device='cuda')
        trainer.load('output/native-autonomous-training-pilot-2026-10-09/seed1/update-2.pt')
        trainer.cursor=0  # New envelope counts curriculum updates, legacy pilot cursor unused.
        for w,spec in zip(trainer.windows,specs):
            extend(w,spec,registry);instrument(w.adapter.model,w.adapter.source,stats)
        validation=[]
        options=SimpleNamespace(chunk_size=2048,wall_seconds=900.,memory_mib=12288,birth_max_proposals=1000000,spatial_backend='batched_v1')
        for spec,train in zip(vspecs,trainer.windows):
            root=Path(spec['run']);controls=_initial_pairing(root,spec['anchor'],stages)
            model,source,shard,runtime,provider=build_real(root,spec['anchor'],controls,stages,options,feedback_mode='autonomous')
            cs=tuple(StepControl(c['start_si'],c['dt_si'],c['end_si'],c['origin'],c['source']) for c in controls)
            adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=provider)
            w=PilotWindow(f"{root.name}:{spec['anchor']}",adapter,runtime,cs,(), 'val','autonomous')
            _install_normalization(w,trainer.normalization);model.to('cuda')
            if model.feature_path.evaluator.geometry_digest!=train.adapter.model.feature_path.evaluator.geometry_digest:
                raise ValueError('cannot transfer spatial coefficients across different validation geometry')
            shared=dict(train.adapter.model.named_parameters())
            for name,param in shared.items():
                parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,param)
            # Static spatial iteration coefficients belong to the trained model, not validation initialization.
            with torch.no_grad():
                reference=dict(train.adapter.model.field.stencil_layers.named_buffers())
                for name,buffer in model.field.stencil_layers.named_buffers():buffer.copy_(reference[name])
            w.runtime.native_context.configuration['model']=deepcopy(model.checkpoint_configuration)
            extend(w,spec,registry);instrument(model,source,stats);validation.append(w)
        trainer.split_config=dict(origin=config,curriculum='cuda_memory_recovery_v1',forecast_cursor=2,
            epochs=3,horizons=[1,2,2],learning_rate=1e-4,no_truth_feedback=True)
        phase='validation_memory_recovery';print(phase,flush=True)
        rng=_rng()
        for i,w in enumerate(validation):
            w.adapter.model.eval();np.random.seed(100+i);torch.manual_seed(100+i);random.seed(100+i)
            prefix=make_prefix(w.adapter,w.runtime,w.controls,2)
            for b,reset in [(0,False),(0,True),(1,True),(2,True)]:
                predictions,proof=forecast(w.adapter,prefix,burnin=b,horizon=2,reset_memory=reset)
                with torch.no_grad():loss,rows=observe(trainer,w,predictions)
                row=dict(window=w.name,burnin=b,reset_memory=reset,loss=float(loss),proof=proof,events=rows)
                result['burnin_comparison'].append(row)
                write(f'val-{i}-burnin-{b}-reset-{reset}.json',row)
                print(json.dumps(dict(phase=phase,window=w.name,burnin=b,reset_memory=reset,loss=float(loss))),flush=True)
            del prefix,predictions
        _restore_rng(rng)
        scores={b:sum(r['loss'] for r in result['burnin_comparison'] if r['reset_memory'] and r['burnin']==b)/2 for b in (0,1,2)}
        best=min(scores.values());selected=min(b for b,v in scores.items() if v<=best+1e-6*max(abs(best),1e-12))
        result['selection']=dict(burnin=selected,validation_scores=scores,relative_tie_tolerance=1e-6,
            interpretation='provisional_memory_recovery_setting_not_optimal_physical_history_or_BPTT_length')
        write('selection.json',result['selection'])
        phase='initial_train_evaluation';result['initial_train_evaluation']=[]
        for i,w in enumerate(trainer.windows):
            w.adapter.model.eval();np.random.seed(200+i);torch.manual_seed(200+i)
            prefix=make_prefix(w.adapter,w.runtime,w.controls,2)
            predictions,proof=forecast(w.adapter,prefix,burnin=selected,horizon=2)
            with torch.no_grad():loss,rows=observe(trainer,w,predictions)
            row=dict(window=w.name,loss=float(loss),proof=proof,events=rows)
            result['initial_train_evaluation'].append(row);write(f'initial-train-{i}.json',row)
            print(json.dumps(dict(phase=phase,run=i,loss=float(loss))),flush=True)
            del prefix,predictions
        for epoch,horizon in enumerate((1,2,2),1):
            phase=f'epoch_{epoch}';epoch_rows=[]
            for i,w in enumerate(trainer.windows):
                w.adapter.model.train();np.random.seed(1000+epoch*10+i);torch.manual_seed(1000+epoch*10+i);random.seed(1000+epoch*10+i)
                trainer.optimizer.zero_grad(set_to_none=True)
                prefix=make_prefix(w.adapter,w.runtime,w.controls,2)
                predictions,proof=forecast(w.adapter,prefix,burnin=selected,horizon=horizon,grad=True)
                loss,rows=observe(trainer,w,predictions)
                backward_start=time.perf_counter();loss.backward();torch.cuda.synchronize()
                stats['backward_wall_seconds']+=time.perf_counter()-backward_start
                parameters=list(w.adapter.model.parameters())
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('invalid gradient')
                norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
                if not torch.isfinite(norm) or norm<=0:raise ValueError('invalid gradient norm')
                saved=[p.detach().clone() for p in parameters];saved_optimizer=deepcopy(trainer.optimizer.state_dict())
                try:
                    trainer.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):raise ValueError('invalid updated weights')
                except Exception:
                    with torch.no_grad():
                        for p,old in zip(parameters,saved):p.copy_(old)
                    trainer.optimizer.load_state_dict(saved_optimizer)
                    raise
                result['optimizer_updates']+=1
                row=dict(epoch=epoch,window=w.name,forecast_steps=horizon,loss=float(loss.detach()),
                    gradient_norm=float(norm),proof=proof,events=rows)
                epoch_rows.append(row);write(f'epoch-{epoch}-run-{i}.json',row)
                print(json.dumps(dict(phase=phase,run=i,loss=row['loss'],updates=result['optimizer_updates'])),flush=True)
                del prefix,predictions,loss
            result['epochs'].append(epoch_rows)
            trainer.save(args.output_dir/f'epoch-{epoch}-parameters-optimizer.pt')
            write(f'epoch-{epoch}-controller.json',dict(completed_epoch=epoch,updates=result['optimizer_updates'],
                next_epoch=epoch+1,selection=result['selection'],pilot_cursor_unused=True,
                restart_boundary='rebuild_full_prefix_from_certified_initial_state_after_each_weight_update'))
        phase='final_train_evaluation';result['final_train_evaluation']=[]
        for i,w in enumerate(trainer.windows):
            w.adapter.model.eval();np.random.seed(200+i);torch.manual_seed(200+i)
            prefix=make_prefix(w.adapter,w.runtime,w.controls,2)
            predictions,proof=forecast(w.adapter,prefix,burnin=selected,horizon=2)
            with torch.no_grad():loss,rows=observe(trainer,w,predictions)
            row=dict(window=w.name,loss=float(loss),proof=proof,events=rows)
            result['final_train_evaluation'].append(row);write(f'final-train-{i}.json',row)
            del prefix,predictions
        result['success']=True
    except Exception as exc:
        import traceback
        result.update(failure_phase=phase,error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
    finally:
        stop.set();thread.join(timeout=6)
        result.update(elapsed_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_monitor_scope='whole_device_including_other_processes_not_exclusive_training_utilization')
        write('gpu-samples.json',samples);write('result.json',result)
        print(json.dumps(dict(success=result['success'],updates=result['optimizer_updates'],error=result.get('error'))),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
