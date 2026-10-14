"""Bounded read-only cloud preflight; never grants execution qualification."""
import argparse
import json
from pathlib import Path
import resource
import time
import h5py
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunk-slots',type=int,default=65536)
    parser.add_argument('--memory-budget-mib',type=int,default=1024)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite evidence')
    if not 1<=args.chunk_slots<=65536 or args.memory_budget_mib<=0:raise ValueError('bounded explicit budget required')
    started=time.perf_counter();rows=[]
    for anchor in json.loads(args.anchors.read_text()):
        marker=anchor['state']['marker'];path=Path(marker['source']['path']);t0=time.perf_counter()
        with h5py.File(path,'r') as f:
            group=f['groups/001'];ids=group['i_elm'];slots=len(ids);active=0
            for start in range(0,slots,args.chunk_slots):active+=int(np.count_nonzero(ids[start:start+args.chunk_slots]>0))
            payload=sum(int(d.size*d.dtype.itemsize) for d in group.values() if isinstance(d,h5py.Dataset))
        claimed=marker['value']
        if (slots,active)!=(claimed['slots'],claimed['active']):raise ValueError('anchor live/slot evidence changed')
        pending={k:v['status'] for k,v in anchor['state'].items()
            if v['status'] not in ('saved','verified_reconstructed','causal_replayed')}
        rows.append(dict(run=anchor['run'],index=anchor['index'],path=str(path),slots=slots,active=active,
            payload_bytes=payload,minimum_three_state_copies_bytes=3*payload,
            three_copies_within_memory_budget=3*payload<=args.memory_budget_mib*1024**2,
            count_seconds=time.perf_counter()-t0,data_pending=pending,
            particle_execution='not_attempted',full_events_completed=0,
            reasons=['production provider not fully bound: nonlocal/birth/density kernel remain unsupported',
                'fluid open/Mach1 boundary support incomplete',
                *(['active cloud exceeds existing 10000 reference guard; no bounded pusher certified'] if active>10000 else []),
                *(['ordinary clock/projection history pending'] if any(k in pending for k in ('projection_history','clock','residual','event_phase')) else [])]))
    result=dict(schema='native-execution-resource-preflight-v1',training=False,optimizer_updates=0,
        scope='all_slots_count_only_not_cloud_push_or_closed_loop',chunk_slots=args.chunk_slots,
        memory_budget_mib=args.memory_budget_mib,rows=rows,elapsed_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
