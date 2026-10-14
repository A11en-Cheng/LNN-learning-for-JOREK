"""Per-field epoch curves, fixed-task comparisons and measured GPU evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

FIELDS=('psi','Phi','Z_j','omega','rho','T_i','v_par','rho_imp','T_e')


def rows_for(window,phase,epoch=None):
    rows=[]
    for event in window['events']:
        for i,name in enumerate(FIELDS):
            m=event['metrics']['fields']['gauss_values'][name]
            rows.append(dict(phase=phase,epoch=epoch,window=window['window'],forecast_step=event['forecast_step'],
                time_si=event['time_si'],field=name,loss_contribution=event['loss']['per_field_total_contribution'][i],
                relative_l2=m['relative_l2'],pearson_r=m['pearson_r']))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();r=json.loads((args.directory/'result.json').read_text());rows=[]
    for phase,key in [('initial','initial_train_evaluation'),('final','final_train_evaluation')]:
        for w in r.get(key,[]):rows+=rows_for(w,phase)
    for epoch in r.get('epochs',[]):
        for w in epoch:rows+=rows_for(w,'training',w['epoch'])
    if not rows:raise ValueError('no completed training/evaluation rows')
    with (args.directory/'per-field.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    names=sorted({row['window'] for row in rows})
    for ax,field in zip(axes.flat,FIELDS):
        for name in names:
            group=[x for x in rows if x['phase']=='training' and x['forecast_step']==1 and x['field']==field and x['window']==name]
            ax.plot([x['epoch'] for x in group],[x['relative_l2'] for x in group],marker='o',label='inner' if 'injInner' in name else 'v400')
        ax.set_title(field);ax.set_xlabel('epoch (2 declared windows)');ax.set_ylabel('first forecast node relative L2');ax.legend()
    fig.suptitle('Training observations: same forecast node; source RNG varies by epoch')
    fig.savefig(args.directory/'per-field-epoch-errors.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for ax,field in zip(axes.flat,FIELDS):
        for name in names:
            group=[x for x in rows if x['phase']=='training' and x['forecast_step']==1 and x['field']==field and x['window']==name]
            ax.plot([x['epoch'] for x in group],[x['loss_contribution'] for x in group],marker='o',label='inner' if 'injInner' in name else 'v400')
        ax.set_title(field);ax.set_xlabel('epoch');ax.set_ylabel('first forecast node loss contribution');ax.legend()
    fig.suptitle('Per-field contribution to four-term normalized data loss')
    fig.savefig(args.directory/'per-field-epoch-losses.png',dpi=140);plt.close(fig)
    summary=[]
    for name in names:
        for field in FIELDS:
            d=dict(window=name,field=field)
            for phase in ('initial','final'):
                group=[x for x in rows if x['phase']==phase and x['window']==name and x['field']==field]
                if group:
                    d[phase+'_loss_contribution']=float(np.mean([x['loss_contribution'] for x in group]))
                    d[phase+'_last_relative_l2']=group[-1]['relative_l2']
            summary.append(d)
    samples=json.loads((args.directory/'gpu-samples.json').read_text());util=[]
    for s in samples:
        try:util.append(float(s['global_gpu_csv'].split(',')[0]))
        except (KeyError,ValueError):pass
    out=dict(per_field=summary,profiling=r['profiling'],elapsed_seconds=r['elapsed_seconds'],
        whole_device_util_mean=None if not util else float(np.mean(util)),
        whole_device_util_p95=None if not util else float(np.percentile(util,95)),
        whole_device_util_max=None if not util else max(util),
        utilization_scope=r['gpu_monitor_scope'],peak_cuda_allocated_bytes=r['peak_cuda_allocated_bytes'])
    with (args.directory/'field-summary.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
