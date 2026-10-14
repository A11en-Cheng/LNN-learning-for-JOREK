"""All-slot cost evidence with parent-enforced wall/RSS limits; no qualification.

The cost mode only samples geometry operations, not an event or truncated cloud
rollout. The attempt mode uses the public NativeMarkerSource on the whole cloud
after its budget bridge is wired. It is a frozen-field source diagnostic, not a
production coupled restart (birth/nonlocal/initialization are not certified here).
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def emit(**record):
    print(json.dumps(record),flush=True)


def worker(args):
    import resource
    import h5py
    import numpy as np
    from src.data.marker_checkpoint import read_marker_snapshot
    from src.models.native_coupling import NativeSourceMesh,NativeMarkerSource,NativeNeonAtomicRates
    from src.models.native_state import NativeUnits
    from src.models.marker_state import MarkerRuntime,MarkerClock,ProjectionFeedback,MarkerExecutionBudget
    anchor=json.loads(args.anchors.read_text())[args.row]
    marker_path=Path(anchor['state']['marker']['source']['path'])
    field_path=marker_path.parent/f"jorek{anchor['index']:06d}.h5"
    started=time.monotonic()
    cloud,marker_time,_=read_marker_snapshot(marker_path)
    ids=np.flatnonzero(cloud.alive)
    payload=sum(getattr(cloud,k).nbytes for k in cloud.array_names())
    emit(stage='cloud_loaded',slots=len(cloud.weight),live=len(ids),payload_bytes=payload,
         seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with h5py.File(field_path,'r') as f:
        scalar=lambda k:float(np.asarray(f[k][()]).ravel()[0])
        units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
        values=f['values'][...];F0=scalar('F0')
        build=time.monotonic()
        mesh=NativeSourceMesh(f['x'][...],f['size'][...],f['vertex'][...]-1,f['index'][...],
            units=units,projection='none' if args.mode=='cost' else 'sparse',max_elements=5000)
    emit(stage='geometry_ready',elements=len(mesh.gauss),unique_dofs=mesh.n_dofs,
         seconds=time.monotonic()-build,projection=mesh.projection)
    if args.mode=='cost':
        # Point cost at existing st uses all saved degrees; one exact locator
        # operation is measured independently, without modifying any particle.
        sample_ids=ids[:min(4,len(ids))]
        begin=time.monotonic()
        for i in sample_ids:mesh.point(int(cloud.i_elm[i])-1,cloud.st[i],values)
        point_seconds=time.monotonic()-begin
        emit(stage='field_point_cost',sample_count=len(sample_ids),seconds=point_seconds,
             estimated_full_live_seconds=point_seconds/max(1,len(sample_ids))*len(ids))
        begin=time.monotonic()
        elm,st=mesh.locate(cloud.x[sample_ids[:1]])
        emit(stage='locator_cost',sample_count=len(elm),seconds=time.monotonic()-begin,
             original_element=cloud.i_elm[sample_ids[:1]].tolist(),located_element=elm.tolist(),
             local_coordinate_max_difference=float(abs(st-cloud.st[sample_ids[:1]]).max()),
             estimated_full_live_seconds=(time.monotonic()-begin)*len(ids))
        return
    if 'execution_budget' not in inspect.signature(NativeMarkerSource.__init__).parameters:
        emit(stage='attempt_unavailable',reason='NativeMarkerSource execution_budget bridge not wired',
             completed_source_windows=0,full_coupled_events=0)
        return
    from src.models.source_module import load_neon_tables
    from src.physics.adas_closure import parse_adf11,ADAS_DIR
    tables=load_neon_tables()
    rates=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=parse_adf11(ADAS_DIR/'prb96_ne.dat'))
    # Diagnostic field history is explicitly frozen; no assertion of a legal
    # production startup clock or feedback initialization is made.
    dt=1e-9
    state=MarkerRuntime(cloud,ProjectionFeedback(np.zeros((5,4,1,mesh.n_nodes)),
        np.zeros((5,4,1,mesh.n_nodes)),marker_time),MarkerClock(marker_time,0.,dt,marker_time,marker_time+dt),
        {'scope':'frozen_field_resource_diagnostic_not_production_restart'})
    policy=MarkerExecutionBudget(max_slots=len(cloud.weight),max_live=max(len(ids),1),
        chunk_size=min(args.chunk_size,max(len(ids),1)),max_particle_steps=1,
        max_seconds=args.seconds,max_rss_bytes=args.memory_mib*1024**2)
    source=NativeMarkerSource(mesh,atomic=rates,atomic_version=rates.version,
        potentials_ev=tables.ionization_potentials_ev,F0=F0,diffusion_ju=0.,execution_budget=policy)
    emit(stage='full_source_attempt_started',slots=len(cloud.weight),live=len(ids),chunk_size=policy.chunk_size,
         frozen_field=True,projection='unfiltered_resource_reference',births=False)
    out,diagnostic=source.advance(state,marker_time+dt,values,values)
    emit(stage='full_source_attempt_completed',execution=diagnostic['execution'],
         retained_slots=len(out.cloud.weight),live_after=int(out.cloud.alive.sum()),
         completed_source_windows=1,full_coupled_events=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--rows',type=int,nargs='+',default=[0,2]);p.add_argument('--row',type=int,default=0)
    p.add_argument('--mode',choices=('cost','attempt'),default='cost')
    p.add_argument('--seconds',type=float,default=20.);p.add_argument('--memory-mib',type=int,default=1536)
    p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if not 0<args.seconds<=60 or not 1<=args.memory_mib<=4096 or args.chunk_size<1:
        raise ValueError('finite wall/RSS/chunk limits required')
    if args.worker:return worker(args)
    if args.output is None or args.output.exists():raise ValueError('new evidence output required')
    rows=[]
    for row in args.rows:
        command=[sys.executable,__file__,'--worker','--anchors',str(args.anchors),'--row',str(row),
            '--mode',args.mode,'--seconds',str(args.seconds),'--memory-mib',str(args.memory_mib),
            '--chunk-size',str(args.chunk_size)]
        started=time.monotonic();peak=0;stopped=None
        # Redirect bounded child output to a temporary file so supervision never
        # blocks on a full stdout pipe. Children emit only small stage records.
        import tempfile
        with tempfile.TemporaryFile(mode='w+') as log:
            child=subprocess.Popen(command,stdout=log,stderr=log,env=os.environ.copy())
            while child.poll() is None:
                try:
                    status=Path(f'/proc/{child.pid}/status').read_text()
                    rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                except (OSError,StopIteration):pass
                if peak>args.memory_mib*1024**2:stopped='rss_limit'
                if time.monotonic()-started>args.seconds:stopped='wall_limit'
                if stopped:
                    child.kill();break
                time.sleep(.02)
            child.wait(timeout=5);log.seek(0);output=log.read()
        stages=[];errors=[]
        for line in output.splitlines():
            try:stages.append(json.loads(line))
            except ValueError:errors.append(line)
        rows.append(dict(anchor_row=row,returncode=child.returncode,stopped=stopped,
            elapsed_seconds=time.monotonic()-started,observed_peak_rss_bytes=peak,stages=stages,errors=errors))
    result=dict(schema='native-cloud-cost-bounded-v1',mode=args.mode,training=False,optimizer_updates=0,
        wall_limit_per_process_seconds=args.seconds,memory_limit_per_process_mib=args.memory_mib,
        full_coupled_events=0,qualification_granted=False,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
