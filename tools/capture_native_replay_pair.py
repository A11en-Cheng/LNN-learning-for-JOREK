"""Bounded real continuous/restored pair with component evidence; no training.

Retains three endpoint states, not a full cloud trajectory. Strict replay verdict
is unchanged: a mismatch is reported after evidence capture, never hidden.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from tools.diagnose_native_real_chain import Stages,_initial_pairing,_array_digest,execute_coupled_events
from tools.native_real_chain_builder import build_real


def attach_source_evidence(source,directory,prefix):
    """Observe the second public source event without altering its inputs."""
    advance=source.advance;project=source.mesh.project;point=source.mesh.point
    active=False;sample_count=0
    def observed_point(*args,**kwargs):
        nonlocal sample_count
        result=point(*args,**kwargs)
        if active and sample_count<2:
            np.savez_compressed(directory/f'{prefix}-sample-{sample_count}.npz',
                element=np.asarray(args[0]),st=np.asarray(args[1]),fields=np.asarray(args[2]),
                values=result[0],gradient=result[1],rz=result[2])
            sample_count+=1
        return result
    def observed_project(rhs):
        result=project(rhs)
        if active:np.savez_compressed(directory/f'{prefix}-projection.npz',rhs=rhs,values=result)
        return result
    def observed_advance(state,target,previous,current,**kwargs):
        nonlocal active
        active=state.clock.window_index==1
        if active:
            before=dict(marker=state,previous=previous,current=current,target=target,**kwargs)
            if before.get('time_sampler') is not None:
                before['time_sampler']=vars(before['time_sampler'])
            with (directory/f'{prefix}-source-input.json').open('x') as f:
                json.dump({k:_array_digest(v) for k,v in before.items()},f,indent=2)
            with (directory/f'{prefix}-births.pt').open('xb') as f:torch.save(kwargs['births'],f)
        try:return advance(state,target,previous,current,**kwargs)
        finally:active=False
    source.advance=observed_advance;source.mesh.project=observed_project;source.mesh.point=observed_point


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--anchor',type=int,default=4110)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=900.)
    parser.add_argument('--memory-mib',type=int,default=8192)
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--birth-max-proposals',type=int,default=1000000)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    random.seed(20261009);np.random.seed(20261009);torch.manual_seed(20261009);torch.set_num_threads(2)
    stages=Stages();run=args.run.resolve();controls=_initial_pairing(run,args.anchor,stages)
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import detach_state
    public=tuple(StepControl(c['start_si'],c['dt_si'],c['end_si'],c['origin'],c['source'],phase=c['phase']) for c in controls)
    def build(prefix):
        bundle=build_real(run,args.anchor,controls,stages,args)
        attach_source_evidence(bundle[1],args.output_dir,prefix)
        return bundle
    def observe(phase,state):
        with (args.output_dir/f'{phase}.pt').open('xb') as f:torch.save(detach_state(state),f)
        stages.emit('component_snapshot','completed',phase=phase,state_digest=_array_digest(state))
    result=dict(scope='recorded_psi_intervention_untrained_pair_not_physics_accuracy',optimizer_updates=0,
        run=str(run),anchor=args.anchor,seed=20261009,controls=controls,configuration=vars(args).copy())
    result['configuration']={k:str(v) if isinstance(v,Path) else v for k,v in result['configuration'].items()}
    try:
        result['strict_result']=execute_coupled_events(build('continuous'),public,stages,args.output_dir,
            rebuild=lambda:build('restored'),scope=str(run),replay_observer=observe)
        result['strict_replay_passed']=True
    except AssertionError as exc:
        result.update(strict_replay_passed=False,error=str(exc))
        if str(exc)!='checkpoint complete state/weights/RNG replay differs':raise
    finally:
        with (args.output_dir/'capture.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(capture_complete=True,strict_replay_passed=result['strict_replay_passed'])),flush=True)


if __name__=='__main__':main()
