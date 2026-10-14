"""Trusted-local selected-cloud refinement study through Source.advance.

Conditional real fields; not complete-cloud projection or a JOREK truth oracle.
The same explicit native velocity-interval import is used in every arm.
"""
import argparse
from copy import copy,deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from src.models.marker_transport import LocalSubcycling,initialize_transport_history
from src.models.native_nearby_locator import NativeNearbyMarkerSource
from tools.diagnose_native_nearby_motion import load_capture
from tools.diagnose_native_real_chain import _array_digest
from tools.native_replay_metrics import relative_metrics
from tools.benchmark_native_lookup import _json,_rss


def compare(reference, candidate, initial):
    support=reference.cloud.alive&candidate.cloud.alive&initial.cloud.alive
    support&=(reference.cloud.i_life==candidate.cloud.i_life)&(reference.cloud.i_life==initial.cloud.i_life)
    rows={}
    for key,components in [('x',('R','Z','phi')),('V_prev',('R','Z','phi_velocity'))]:
        a=getattr(reference.cloud,key)[support]-getattr(initial.cloud,key)[support]
        b=getattr(candidate.cloud,key)[support]-getattr(initial.cloud,key)[support]
        rows[key]={name:relative_metrics(a[:,i],b[:,i]) for i,name in enumerate(components)}
        if key=='V_prev':
            w=np.sqrt(initial.cloud.weight[support])[:,None]
            rows[key]['atom_weighted_vector']=relative_metrics(a*w,b*w)
            rows[key]['vector']=relative_metrics(a,b)
        else:
            rows[key]['RZ_vector']=relative_metrics(a[:,:2],b[:,:2])
    return dict(common_persistent_count=int(support.sum()),increments=rows,
        P_imp=relative_metrics(reference.cloud.P_imp[support],candidate.cloud.P_imp[support]),
        identity_exact=bool(np.array_equal(reference.cloud.i_elm,candidate.cloud.i_elm)
            and np.array_equal(reference.cloud.i_life,candidate.cloud.i_life)),
        projection={name:{str(c):relative_metrics(getattr(reference.feedback,name)[c],
            getattr(candidate.feedback,name)[c]) for c in range(5)} for name in ('values','deltas')})


def execute(payload):
    original=payload['initial'];digest=_array_digest((original,payload['previous'],payload['current']))
    outs={};rows=[]
    for name,minimum,nearby in [('native_local64',6,False),('nearby_local64',6,True),('nearby_local128',7,True)]:
        source=copy(payload['source']);source.configuration=deepcopy(source.configuration)
        policy=LocalSubcycling(min_level=minimum)
        source.transport_policy=policy;source.configuration['transport_policy']=policy.configuration
        runner=NativeNearbyMarkerSource(source,neighbours=payload['neighbours']) if nearby else source
        state=initialize_transport_history(original,policy,previous_dt_si=original.clock.particle_dt_si)
        before=_array_digest(state);tick=time.perf_counter()
        out,diag=runner.advance(state,payload['target'],payload['previous'],payload['current'],
            **({} if payload['sampler'] is None else dict(time_sampler=deepcopy(payload['sampler']))))
        history=out.transport_history
        rows.append(dict(mode=name,seconds=time.perf_counter()-tick,execution=diag['execution'],
            final_levels={str(level):int(np.sum(history.level[out.cloud.alive]==level)) for level in np.unique(history.level)},
            inputs_unchanged=before==_array_digest(state),configuration=policy.configuration))
        outs[name]=out
        print(json.dumps(rows[-1]),flush=True)
    return dict(success=all(r['inputs_unchanged'] for r in rows),training=False,optimizer_updates=0,
        production_qualified=False,scope='selected_real_cloud_conditional_fields_no_births_NOT_full_cloud_projection',
        original_slots=payload['original_slots'].tolist(),particle_steps=rows[0]['execution']['processed_particle_steps'],
        runs=rows,locator_comparison=compare(outs['native_local64'],outs['nearby_local64'],original),
        refinement_comparison=compare(outs['nearby_local128'],outs['nearby_local64'],original),
        original_inputs_unchanged=digest==_array_digest((original,payload['previous'],payload['current'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':p.error(key+'=1 required')
    paths=[Path(__file__),Path('src/models/marker_transport.py'),Path('src/models/marker_state.py'),
        Path('src/models/native_coupling.py'),Path('src/models/native_nearby_locator.py')]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    def deadline(*_):raise TimeoutError('600s selected replay deadline exceeded')
    old=signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    tick=time.perf_counter();result={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:result=execute(load_capture(args.capture))
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        result.update(seconds=time.perf_counter()-tick,peak_rss_bytes=_rss(),source_code_sha256=hashes,
            code_unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path)] for path in paths))
        with args.output.open('x') as stream:json.dump(result,stream,default=_json,indent=2,allow_nan=False)
    return 0 if result.get('success') and result['code_unchanged'] and result['original_inputs_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
