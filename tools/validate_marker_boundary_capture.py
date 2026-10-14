"""Trusted selected real marker: legacy-limit red / verified-exit green.

Loads a LOCAL trusted Source capture and the initial part file only; one base
interval, no births, no training, no cropped-cloud production qualification.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from src.data.marker_checkpoint import read_marker_snapshot
from src.models.marker_transport import LocalSubcycling,initialize_transport_history
from src.models.native_coupling import NativeMarkerSource
from src.models.native_nearby_locator import NativeNearbyMarkerSource
from tools.diagnose_native_nearby_motion import load_capture,select_cloud
from tools.diagnose_native_real_chain import _array_digest
from tools.benchmark_native_lookup import _json
from tools.validate_native_nearby_history import _file_hash


def execute(capture_path, marker_path, slot):
    payload=load_capture(capture_path)
    original=payload['source']
    cloud,t,meta=read_marker_snapshot(marker_path)
    initial=deepcopy(payload['initial']);initial.cloud=cloud
    if abs(t-initial.clock.sim_time_si)>1e-14:
        raise ValueError('selected initial snapshot clock mismatch')
    initial=select_cloud(initial,[slot]);immutable=_array_digest(initial)
    target=initial.clock.sim_time_si+initial.clock.particle_dt_si
    rows=[];outputs={}
    for name,policy,nearby in [('legacy_limit',LocalSubcycling(),False),
        ('verified_native',LocalSubcycling(boundary_exit=True),False),
        ('verified_nearby',LocalSubcycling(boundary_exit=True),True),
        ('verified_finer',LocalSubcycling(boundary_exit=True,max_level=11,min_level=7),True)]:
        source=NativeMarkerSource(original.mesh,atomic=original.atomic,atomic_version=original.configuration['atomic'],
            potentials_ev=original.potentials_ev,F0=original.F0,diffusion_ju=original.diffusion_ju,
            time_sampling=original.time_sampling,field_static=original.field_static,
            execution_budget=original.execution_budget,spatial_backend='batched_v1',transport_policy=policy,
            saved_neighbours=payload['neighbours'] if policy.boundary_exit else None)
        runner=NativeNearbyMarkerSource(source,neighbours=payload['neighbours']) if nearby else source
        state=initialize_transport_history(initial,policy,previous_dt_si=initial.clock.particle_dt_si)
        before=_array_digest(state);tick=time.perf_counter()
        try:
            out,diag=runner.advance(state,target,payload['previous'],payload['current'],
                time_sampler=deepcopy(payload['sampler']))
            events=diag.get('boundary_exits',[])
            terminal=bool(len(events)==1 and not out.cloud.alive[0])
            elapsed=events[0]['time_si']-initial.clock.sim_time_si if terminal else None
            weights_ok=terminal and np.isclose(diag['moments'][0,2],out.cloud.weight[0]*elapsed,rtol=1e-9,atol=0)
            outputs[name]=(out,diag)
            row=dict(mode=name,passed=bool(terminal and weights_ok and name!='legacy_limit'),
                events=events,terminal_x=out.cloud.x,terminal_st=out.cloud.st,
                accepted_dt_si=out.transport_history.previous_dt_si,
                terminal_level=out.transport_history.level,execution=diag['execution'],
                residence_time_relative_check=bool(weights_ok),configuration=policy.configuration)
        except ValueError as error:
            row=dict(mode=name,passed=name=='legacy_limit' and 'refinement limit' in str(error),error=str(error))
        row.update(seconds=time.perf_counter()-tick,inputs_unchanged=before==_array_digest(state))
        rows.append(row)
    from tools.native_replay_metrics import relative_metrics
    comparisons={}
    if all(name in outputs for name in ('verified_native','verified_nearby','verified_finer')):
        a,da=outputs['verified_native'];b,db=outputs['verified_nearby'];f,df=outputs['verified_finer']
        for name,left,right,dleft,dright in [('locator',a,b,da,db),('refinement',f,b,df,db)]:
            comparisons[name]=dict(position_RZ=relative_metrics(left.cloud.x[:,:2],right.cloud.x[:,:2]),
                velocity_increment=relative_metrics(left.cloud.V_prev-initial.cloud.V_prev,right.cloud.V_prev-initial.cloud.V_prev),
                residence=relative_metrics(dleft['moments'][:,2],dright['moments'][:,2]),
                P_imp=relative_metrics(left.cloud.P_imp,right.cloud.P_imp),
                projection={str(c):relative_metrics(left.feedback.values[c],right.feedback.values[c]) for c in range(5)})
    return dict(success=bool(all(r['passed'] and r['inputs_unchanged'] for r in rows)
        and _array_digest(initial)==immutable),scope='one_initial_real_particle_one_base_step_no_births',
        original_slot=slot,training=False,production_qualified=False,runs=rows,comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--marker',type=Path,required=True)
    p.add_argument('--slot',type=int,default=973)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    paths=[Path(__file__),*sorted(Path('src').rglob('*.py'))]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    inputs={str(path):_file_hash(path) for path in (args.capture,args.marker)}
    tick=time.perf_counter()
    result=execute(args.capture,args.marker,args.slot)
    result.update(seconds=time.perf_counter()-tick,source_code_sha256=hashes,input_sha256=inputs,
        code_unchanged=all(_file_hash(path)==value for path,value in hashes.items()),
        input_files_unchanged=all(_file_hash(path)==value for path,value in inputs.items()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,default=_json,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('success','seconds','runs','comparisons')},default=_json),flush=True)
    return 0 if result['success'] and result['code_unchanged'] and result['input_files_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
