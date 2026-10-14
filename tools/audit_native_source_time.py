"""Read-only full-native phaseA sampling audit; no particles or rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import h5py
import numpy as np
from src.models.native_source_time import NativeSourceTimeSampler


def read(path):
    with h5py.File(path,'r') as f:
        scalar=lambda key:float(np.ravel(f[key][()])[0])
        return dict(values=np.asarray(f['values'][()]),raw=np.asarray(f['deltas'][()]),
            time=scalar('t_now')*scalar('t_norm'),dt=scalar('tstep')*scalar('t_norm'),
            index=int(scalar('index_now')),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',required=True,help='run_path:current_index; previous is index-1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite audit artifact')
    started=time.perf_counter();rows=[]
    for case in args.case:
        path,index=case.rsplit(':',1);run=Path(path);index=int(index)
        p=run/f'jorek{index:06d}.h5';q=run/f'jorek{index-1:06d}.h5'
        now,prev=read(p),read(q)
        if now['index']!=index or prev['index']!=index-1 or now['dt']<=0 or now['time']<=prev['time']:
            raise ValueError('ordinary ordered internally indexed records required')
        results=[]
        for static in (True,False):
            sampler=NativeSourceTimeSampler(now['values'],now['raw'],time_prev_si=prev['time'],
                time_now_si=now['time'],tstep_si=now['dt'],static=static)
            for fraction in (0.,.5,1.,1.1):
                t=now['time']-fraction*(now['time']-prev['time'])
                sampled=sampler.fields_at(t)
                results.append(dict(static=static,time_si=t,fraction_before_current=fraction,
                    finite=bool(np.isfinite(sampled).all()),
                    max_abs_change_by_field_degree=np.max(np.abs(sampled-now['values']),axis=(2,3)).tolist()))
        log=run/'logfile_spitest'
        log_evidence=[]
        with log.open(errors='replace') as stream:
            for i,line in enumerate(stream,1):
                if 'mod_fields_linear : (t_start, t_norm, t_now, static)' in line:
                    log_evidence.append(dict(line=i,text=line.strip()))
                    break  # Startup evidence only; do not load hundreds of MB of logs.
        rows.append(dict(run=run.name,current=dict(path=str(p),sha256=now['sha256'],index=now['index']),
            previous_for_clock_and_comparison=dict(path=str(q),sha256=prev['sha256'],index=prev['index']),
            shape=list(now['values'].shape),saved_rawD=True,imported_tstep_si=now['dt'],
            clock_span_si=now['time']-prev['time'],startup_static_log=log_evidence,
            max_abs_endpoint_difference_minus_rawD_by_field_degree=
                np.max(np.abs(now['values']-prev['values']-now['raw']),axis=(2,3)).tolist(),
            samples=results,scope='static_true_inspected_live_path; false_counterfactual_branch_not_production_certification'))
    result=dict(schema='native-source-time-audit-v1',training=False,optimizer_updates=0,
        scope='full_native_sampling_only_no_cloud_no_shard_no_coupled_rollout',
        source_rcs='v2.25.12.0-1042-gb4165893f-dirty',
        inspected_head='54c675bcc39c33b082c0c35046e28122dc53b3aa',
        sampler_version=NativeSourceTimeSampler.version,rows=rows,
        elapsed_seconds=time.perf_counter()-started,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),runs=len(rows),elapsed_seconds=result['elapsed_seconds'],
        process_peak_rss_kib=result['process_peak_rss_kib'])))


if __name__=='__main__':main()
