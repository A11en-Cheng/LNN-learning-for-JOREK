#!/usr/bin/env python3
"""Bounded real full-grid/all-slot psi-teacher event attempt, never training.

No ordinary history manufacture, timestep shrinking or cloud reduction.
Failures retain the precise last public event stage and parent wall/RSS proof.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


class Stages:
    def __init__(self):
        self.current='worker_start';self.completed_events=0;self.startup_events=0

    def emit(self,stage,status,**details):
        print(json.dumps(dict(stage=stage,status=status,completed_events=self.completed_events,
            startup_fluid_events=self.startup_events,ordinary_source_field_events=self.completed_events,
            monotonic_seconds=time.monotonic(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            **details),allow_nan=False),flush=True)

    @contextmanager
    def track(self,stage,**details):
        parent=self.current;self.current=stage;start=time.monotonic()
        self.emit(stage,'started',**details)
        try:
            yield details
        except Exception as exc:
            self.emit(stage,'failed',elapsed_seconds=time.monotonic()-start,
                error_type=type(exc).__name__,reason=str(exc),**details)
            raise
        else:
            self.emit(stage,'completed',elapsed_seconds=time.monotonic()-start,**details)
            self.current=parent


def _scalar(handle,key):
    import numpy as np
    if key not in handle or handle[key].size!=1:raise ValueError('missing scalar '+key)
    value=float(np.asarray(handle[key][()]).ravel()[0])
    if not np.isfinite(value):raise ValueError('nonfinite scalar '+key)
    return value


def _initial_pairing(run,index,stages):
    import h5py
    registry=json.loads((ROOT/'data/restart_events.json').read_text())['runs']
    with stages.track('initial_event_pairing',run=str(run),anchor_index=index) as evidence:
        if run.name not in registry:raise ValueError('run requires explicit registered restart evidence')
        field=run/f'jorek{index:06d}.h5';marker=run/f'part{index:05d}.h5'
        with h5py.File(field,'r') as f:
            tn=_scalar(f,'t_norm');current=_scalar(f,'t_now')*tn
            evidence.update(field_time_si=current,imported_dt_si=_scalar(f,'tstep')*tn,
                field_shape=list(f['values'].shape),n_nodes=f['values'].shape[-1])
        with h5py.File(marker,'r') as f:
            particle=_scalar(f,'time')
            evidence.update(marker_time_si=particle,marker_slots=f['groups/001/i_elm'].size)
        if index+1 not in registry[run.name]['restart_events']:
            raise ValueError('ordinary continuous anchor lacks saved clock/residual and raw projection history; no zero replacement')
        # This is a registered POSTstartup import, not an ordinary snapshot
        # given invented continuous history, nor a network startup execution.
        import numpy as np
        with h5py.File(run/f'jorek{index+1:06d}.h5','r') as f:
            post_time=_scalar(f,'t_now')*tn;startup_dt=_scalar(f,'tstep')*tn
            if abs(particle+startup_dt-post_time)>max(1e-14,tn*1e-7):
                raise ValueError('registered startup clock does not match imported particle clock')
        with h5py.File(run/f'projections{index:06d}.h5','r') as f:
            projection=np.asarray(f['values'][...]);projection_time=_scalar(f,'t_now')
            if projection.shape!=(5,4,1,evidence['n_nodes']) or not np.isfinite(projection).all() or projection.any():
                raise ValueError('registered initial projection is not the complete recorded zero projection')
            if abs(projection_time-particle)>1e-14:raise ValueError('registered zero projection time mismatch')
        evidence.update(initialization='recorded_poststartup',initial_field_index=index+1,
            imported_field_time_si=current,field_time_si=post_time,field_start_si=particle,field_end_si=post_time,
            projection_max_abs=0.,projection_time_si=projection_time,startup_dt_si=startup_dt,
            raw_projection_history_origin='registered_restart_zero_projection_not_ordinary_history',
            field_previous_origin='imported_restart_field_at_registered_startup_logical_start_preserving_record_time')
        controls=[];previous=post_time
        for offset in (2,3):
            with h5py.File(run/f'jorek{index+offset:06d}.h5','r') as f:
                end=_scalar(f,'t_now')*tn;dt=_scalar(f,'tstep')*tn
                if _scalar(f,'index_now')!=index+offset:raise ValueError('HDF event index mismatch')
            logical=previous
            if dt<=0 or abs(logical+dt-end)>max(1e-14,tn*1e-7):
                raise ValueError('recorded HDF dt does not reach its event endpoint from the declared solve clock')
            controls.append(dict(start_si=previous,solve_start_si=logical,end_si=end,dt_si=dt,
                phase='ordinary',event_index=index+offset,
                origin='recorded_schedule_replay',source=str(run/f'jorek{index+offset:06d}.h5')))
            previous=end
        evidence['controls']=controls
        return controls


def _worker(args):
    stages=Stages()
    try:
        run=args.run.resolve();controls=_initial_pairing(run,args.anchor,stages)
        import random
        import numpy as np
        import torch
        seed=20261009
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
        stages.emit('execution_configuration','completed',seed=seed,torch_num_threads=torch.get_num_threads(),
            training=False,optimizer_updates=0,bptt_requested=False)
        with stages.track('full_chain_build'):
            from tools.native_real_chain_builder import build_real
            bundle=build_real(run,args.anchor,controls,stages,args)
        from src.data.base_coupled import StepControl
        public_controls=tuple(StepControl(row['start_si'],row['dt_si'],row['end_si'],
            row['origin'],row['source'],phase=row['phase']) for row in controls)
        execute_coupled_events(bundle,public_controls,stages,args.output_dir,
            rebuild=lambda:build_real(run,args.anchor,controls,stages,args),scope=str(run))
    except Exception as exc:
        stages.emit(stages.current,'attempt_failed',error_type=type(exc).__name__,reason=str(exc))
        traceback.print_exc()
        return 1
    return 0


def _array_digest(value):
    """Bounded exact digest, including every slot, cache, clock and Tensor mirror."""
    import hashlib
    from dataclasses import fields,is_dataclass
    import numpy as np
    import torch
    result=hashlib.sha256()
    def token(value):
        payload=str(value).encode();result.update(len(payload).to_bytes(8,'little'));result.update(payload)
    def visit(item):
        if isinstance(item,torch.Tensor):item=item.detach().cpu().numpy()
        if isinstance(item,np.ndarray):
            if item.dtype.hasobject:raise TypeError('object arrays cannot be exact checkpoint evidence')
            array=np.ascontiguousarray(item);token('array');token(array.dtype.str);token(array.shape)
            result.update(memoryview(array).cast('B'))
        elif is_dataclass(item):
            token(type(item).__qualname__)
            for field in fields(item):token(field.name);visit(getattr(item,field.name))
        elif isinstance(item,dict):
            token('dict');token(len(item))
            for key in sorted(item,key=repr):visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            token(type(item).__name__);token(len(item))
            for entry in item:visit(entry)
        elif isinstance(item,np.generic):visit(item.item())
        elif item is None or isinstance(item,(str,bool,int,float,bytes)):
            token(type(item).__name__);token(repr(item))
        else:raise TypeError('unsupported exact state evidence '+type(item).__qualname__)
    visit(value)
    return result.hexdigest()


def _rng_digest():
    import random
    import numpy as np
    import torch
    # The real builder executes on CPU. CUDA RNG is not consumed by this runner.
    return _array_digest((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def _compact_event(before,after,diagnostics):
    import numpy as np
    runtime=after.runtime;shards=runtime.shards;cloud=runtime.marker.cloud
    delta=np.max(np.abs(runtime.fields_current-before.fields_current),axis=(1,2,3))
    if not np.isfinite(delta).all() or not (delta>0).all():
        raise ValueError('all nine committed fields must have finite nonzero actual output, not zero_decoder')
    nonlocal_diag=shards.get('nonlocal_diagnostic')
    return dict(event_index=runtime.fluid_index,field_time_si=runtime.fluid_time_si,
        field_delta_max_abs=delta.tolist(),field_digest=_array_digest(runtime.fields_current),
        raw_deltas_digest=_array_digest(runtime.fluid_deltas),memory_digest=_array_digest(runtime.field_memory),
        marker_slots=len(cloud.weight),live_markers=int(cloud.alive.sum()),
        new_births=int(np.sum(cloud.i_life.astype(np.int64)-before.marker.cloud.i_life.astype(np.int64))),
        marker_clock=dict(vars(runtime.marker.clock)),feedback_time_si=runtime.marker.feedback.time_si,
        actual_gather_time_si=runtime.marker.feedback.gather_time_si,
        feedback_digest=_array_digest(runtime.marker.feedback),cloud_digest=_array_digest(cloud),
        shard_digest=_array_digest(shards),
        shard_position_delta_max_abs=float(np.max(np.abs(shards['positions']-before.shards['positions']))),
        shard_radius_delta_max_abs=float(np.max(np.abs(shards['radius']-before.shards['radius']))),
        shard_rate_delta_max_abs=float(np.max(np.abs(shards['ablation_atoms_s']-before.shards['ablation_atoms_s']))),
        nonlocal_time_si=None if nonlocal_diag is None else nonlocal_diag.time_si,
        nonlocal_field_digest=None if nonlocal_diag is None else nonlocal_diag.field_digest,
        source_volume_time_si=shards.get('source_volume_time_si'),
        branch_intervention=shards.get('psi_teacher_event'),
        diagnostic_field_binding=shards.get('diagnostic_field_binding'),
        source_execution=diagnostics.get('execution'),gradient_scope=diagnostics.get('gradient_scope'))


def _trace_public_stages(model,source,shard,stages):
    """Read-only public-call tracing; no checkpointed physics or RNG changes."""
    import numpy as np
    def attach(owner,name,label,summarize):
        original=getattr(owner,name)
        def call(*args,**kwargs):
            with stages.track(label) as evidence:
                out=original(*args,**kwargs)
                evidence.update(summarize(out))
                return out
        setattr(owner,name,call)
    attach(shard,'density_source','independent_shard_density',lambda out: {})
    attach(shard,'births','marker_birth_sampling',lambda out: {'new_markers':len(out.weight)})
    attach(source,'advance','marker_transport_and_projection',lambda out: {'execution':out[1].get('execution')})
    attach(model,'forward','joint_nine_field_forward',lambda out: {
        'increment_max_abs':out.increment.detach().abs().amax(dim=(1,2,3)).tolist()})
    attach(shard,'update','OLD_field_shard_update',lambda out: {'rate_sum_atoms_s':float(np.sum(out['ablation_atoms_s']))})
    attach(shard,'refresh_diagnostics','nonlocal_and_volume_refresh',lambda out: {'time_si':out['time_si']})


def execute_coupled_events(bundle,controls,stages,directory,*,rebuild,scope,verify_bptt=False,replay_observer=None):
    """Complete event/checkpoint seam, shared by real entry and explicit fixtures.

    Keep no full-cloud trajectory. The first checkpoint contains ALL slot
    arrays; the final comparison uses an exact streaming digest. No labels,
    optimizer, training or truth reinstallation occurs during advancement.
    """
    import gc
    import numpy as np
    import torch
    from src.data.base_coupled import Support,make_sequence
    from src.training.base_coupled import TorchRecomputedSourceEvents,SequenceSession,save_session,load_session
    model,source,shard,runtime,teacher=bundle;del bundle
    if len(controls)!=2 or any(c.phase!='ordinary' for c in controls):
        raise ValueError('two ordinary source-field events required after explicit poststartup import')
    directory=Path(directory)
    weights_before=_array_digest(model.state_dict())
    _trace_public_stages(model,source,shard,stages)
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=teacher)
    with stages.track('torch_state_initialize'):
        state=adapter.initialize(runtime)
    first_si=runtime.marker.clock.sim_time_si;del runtime
    for tensor in (state.values,state.raw_deltas,state.memory):tensor.requires_grad_()
    # These three tensors retain the graph, not the full cloud runtime.
    initial_graph=(state.values,state.raw_deltas,state.memory)
    support=Support(scope,first_si,controls[-1].end_si,'registered_poststartup_and_psi_intervention',scope)
    sequence=make_sequence(run=scope,task='self_generated',controls=controls,nodes=(),burnin_steps=0,
        support=(support,),blocks=(Support(scope,first_si,controls[-1].end_si,'diagnostic_only',scope),),
        state_evidence={'complete_state':'verified_reconstructed'},required_state=('complete_state',),
        model_reasons=(),parameters={'scope':scope,'qualification_granted':False,'recorded_psi_intervention':True})
    rows=[];first_graph=None
    for index,control in enumerate(controls):
        with stages.track('ordinary_complete_event',ordinary_index=index+1,
                field_start_si=control.start_si,field_end_si=control.end_si,recorded_dt_si=control.dt_si):
            previous=state
            state,diagnostics=adapter.advance(previous,control)
            stages.completed_events+=1
        with stages.track('committed_event_evidence',ordinary_index=index+1) as evidence:
            row=_compact_event(previous.runtime,state,diagnostics);rows.append(row);evidence.update(row)
            np.savez_compressed(directory/f'event-{index+1}.npz',
                values=state.runtime.fields_current,raw_deltas=state.runtime.fluid_deltas,memory=state.runtime.field_memory,
                feedback=state.runtime.marker.feedback.values,projection_deltas=state.runtime.marker.feedback.deltas,
                shard_positions=state.runtime.shards['positions'],shard_radius=state.runtime.shards['radius'],
                shard_ablation_atoms_s=state.runtime.shards['ablation_atoms_s'])
        if replay_observer is not None:
            replay_observer('before_second' if index==0 else 'continuous_second',state)
        for tensor in (state.values,state.raw_deltas,state.memory):tensor.retain_grad()
        if index==0:
            first_graph=(state.values,state.raw_deltas,state.memory)
            with stages.track('first_event_checkpoint_save') as evidence:
                path=directory/'first-event.pt'
                if path.exists():raise ValueError('refusing to overwrite first event checkpoint')
                session=SequenceSession(state,cursor=1,sequence_fingerprint=sequence.fingerprint,
                    adapter_fingerprint=adapter.fingerprint)
                save_session(path,sequence,adapter,session)
                evidence.update(checkpoint_bytes=path.stat().st_size,marker_slots=len(state.runtime.marker.cloud.weight),
                    full_state_digest=_array_digest(state),rng_digest=_rng_digest())
                del session
        del previous
    final_digest=_array_digest(state);final_rng=_rng_digest()
    # BPTT is a mechanism check only; source/shard remain stop-gradient NumPy.
    bptt=dict(status='not_executed',gradient_scope=adapter.gradient_scope)
    if verify_bptt:
        with stages.track('cross_event_bptt') as evidence:
            state.values.square().mean().backward()
            gradients={}
            arrays={}
            for prefix,group in (('initial',initial_graph),('first',first_graph)):
                for name,tensor in zip(('values','raw_deltas','memory'),group):
                    key=prefix+'_'+name
                    if tensor.grad is None or not torch.isfinite(tensor.grad).all() or not torch.any(tensor.grad):
                        raise AssertionError('cross-event graph disconnected: '+key)
                    gradients[key]=float(tensor.grad.abs().sum())
                    arrays[key]=tensor.grad.detach().cpu().numpy()
            np.savez_compressed(directory/'bptt.npz',**arrays)
            bptt=dict(status='verified',gradient_scope=adapter.gradient_scope,
                cross_event_gradient_l1=gradients,loss_input='last_committed_fields_only_no_labels_no_optimizer')
            evidence.update(bptt_result=bptt)
    unchanged=weights_before==_array_digest(model.state_dict())
    if not unchanged:raise AssertionError('complete events changed model parameters/buffers')
    del state,initial_graph,first_graph
    gc.collect()
    with stages.track('new_model_restore_build'):
        fresh,fresh_source,fresh_shard,fresh_runtime,fresh_teacher=rebuild()
        del fresh_runtime
        _trace_public_stages(fresh,fresh_source,fresh_shard,stages)
        with torch.no_grad():fresh.field.decoder.bias.add_(.1)
        different=_array_digest(fresh.state_dict())!=weights_before
        other=TorchRecomputedSourceEvents(fresh,fresh_source,fresh_shard,equilibrium_provider=fresh_teacher)
    with stages.track('first_event_checkpoint_load') as evidence:
        loaded=load_session(directory/'first-event.pt',sequence,other)
        weights_equal=_array_digest(fresh.state_dict())==weights_before
        evidence.update(cursor=loaded.cursor,marker_slots=len(loaded.state.runtime.marker.cloud.weight),
            loaded_weights_equal=weights_equal,fresh_constructor_weights_differ=different)
    with stages.track('restored_second_complete_event'):
        resumed,diagnostics=other.advance(loaded.state,controls[1])
        del loaded
    if replay_observer is not None:replay_observer('restored_second',resumed)
    with stages.track('checkpoint_exact_comparison') as evidence:
        restored_digest=_array_digest(resumed);restored_rng=_rng_digest()
        checkpoint=dict(complete_state_bitwise_equal=restored_digest==final_digest,full_rng_equal=restored_rng==final_rng,
            fresh_constructor_weights_differ=different,loaded_weights_equal=weights_equal,
            direct_state_digest=final_digest,restored_state_digest=restored_digest,
            direct_rng_digest=final_rng,restored_rng_digest=restored_rng,
            replayed_ordinary_events=1,rng_scope='python_numpy_MT19937_torch_CPU_no_CUDA_execution')
        evidence.update(checkpoint)
        if not all((restored_digest==final_digest,restored_rng==final_rng,different,weights_equal)):
            raise AssertionError('checkpoint complete state/weights/RNG replay differs')
    result=dict(schema='native-real-chain-events-v1',scope=scope,ordinary_source_field_events=len(rows),
        startup_fluid_events=0,optimizer_updates=0,model_weights_unchanged=unchanged,
        retains_full_cloud_trajectory=False,events=rows,checkpoint=checkpoint,bptt=bptt,
        field_mode=model.mode,zero_decoder=False,qualification_granted=False,
        source_configuration=source.configuration,shard_configuration=shard.configuration)
    with (directory/'events.json').open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    stages.emit('complete_chain_verified','completed',model_weights_unchanged=unchanged,checkpoint=checkpoint,bptt=bptt)
    return result


def attempt_real_chain(run,anchor_index,output_dir,*,wall_seconds=900.,memory_mib=8192,
                       birth_max_proposals=200000,chunk_size=256):
    """Public complete-event attempt with parent-enforced finite resource bounds."""
    if (not 0<wall_seconds<=900 or not 1024<=memory_mib<=16384 or birth_max_proposals<1 or chunk_size<1):
        raise ValueError('explicit finite wall/RSS/birth/chunk budgets required')
    directory=Path(output_dir).resolve();directory.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','tools.diagnose_native_real_chain','--worker','--run',str(Path(run).resolve()),
        '--anchor',str(anchor_index),'--output-dir',str(directory),'--wall-seconds',str(wall_seconds),
        '--memory-mib',str(memory_mib),'--birth-max-proposals',str(birth_max_proposals),'--chunk-size',str(chunk_size)]
    start=time.monotonic();peak=0;stopped=None
    with (directory/'worker.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,env=os.environ.copy())
        while child.poll() is None:
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0)
                peak=max(peak,rss)
            except OSError:pass
            if peak>memory_mib*1024**2:stopped='rss_limit'
            if time.monotonic()-start>wall_seconds:stopped='wall_limit'
            if stopped:
                child.kill();break
            time.sleep(.05)
        child.wait(timeout=10)
    records=[]
    for line in (directory/'worker.log').read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and 'stage' in row:records.append(row)
    failed=[row for row in records if row['status']=='attempt_failed']
    reason=failed[-1]['reason'] if failed else stopped
    stage=failed[-1]['stage'] if failed else records[-1]['stage'] if records else 'worker_start'
    completed=max((row.get('completed_events',0) for row in records),default=0)
    result=dict(schema='native-real-chain-attempt-v1',run=str(Path(run).resolve()),anchor_index=anchor_index,
        scope='real_full_grid_all_slots_recorded_psi_intervention_not_autonomous_not_physics_certification',
        success=child.returncode==0 and completed==2,full_coupled_events=completed,
        startup_fluid_events=max((row.get('startup_fluid_events',0) for row in records),default=0),
        ordinary_source_field_events=completed,target_ordinary_source_field_events=2,
        initialization='recorded_poststartup',
        seed=20261009,torch_num_threads=2,
        failure_stage=None if completed==2 and child.returncode==0 else stage,
        failure_reason=reason,stopped=stopped,returncode=child.returncode,stages=records,
        elapsed_seconds=time.monotonic()-start,observed_peak_rss_bytes=peak,
        wall_limit_seconds=wall_seconds,memory_limit_mib=memory_mib,birth_max_proposals=birth_max_proposals,
        optimizer_updates=0,cloud_cropped=False,timestep_shrunk=False,qualification_granted=False)
    with (directory/'diagnostic.json').open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--anchor',type=int,default=4110)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--wall-seconds',type=float,default=900.);parser.add_argument('--memory-mib',type=int,default=8192)
    parser.add_argument('--birth-max-proposals',type=int,default=200000);parser.add_argument('--chunk-size',type=int,default=256)
    args=parser.parse_args()
    if args.worker:raise SystemExit(_worker(args))
    result=attempt_real_chain(args.run,args.anchor,args.output_dir,wall_seconds=args.wall_seconds,
        memory_mib=args.memory_mib,birth_max_proposals=args.birth_max_proposals,chunk_size=args.chunk_size)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
