#!/usr/bin/env python3
"""Summarize fixed-budget field-only artifacts; no new training or selection."""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_window_io import atomic_json
from src.training.native_curriculum import SCOPE

FIELDS=['psi','Phi','Z_j','omega','rho','T_i','v_par','rho_imp','T_e']


def summarize(root):
    evaluations=[];runs=[];curves=[];probes=[]
    for path in sorted(root.glob('seed-*')):
        if not (path/'summary.json').exists():raise ValueError('unfinished seed')
        summary=json.loads((path/'summary.json').read_text());runs.append(summary);seed=summary['seed']
        history=json.loads((path/'training-curves.json').read_text())['updates']
        curves.append((seed,history))
        probes.append((seed,json.loads((path/'dt-limit.json').read_text())['rows']))
        for evaluation in json.loads((path/'evaluations.json').read_text())['evaluations']:
            groups=defaultdict(list)
            for record in evaluation['records']:
                for step in record['steps']:
                    key=(record['evaluation_group'],record['policy'],record['horizon'],step['offset'],step['dt_bucket'],step['ratio_bucket'])
                    groups[key].append(step)
            for key,steps in groups.items():
                group,policy,horizon,offset,dt,ratio=key
                row=dict(scope=SCOPE,seed=seed,update=evaluation['update'],group=group,policy=policy,
                    horizon=horizon,offset=offset,dt_bucket=dt,ratio_bucket=ratio,n=len(steps),
                    elapsed_ju=float(np.mean([s['elapsed_ju'] for s in steps])),
                    mean_loss=float(np.mean([s['total'] for s in steps])))
                for name in ('unweighted','weighted','normalized_group_mse','native_unit_squared_mse'):
                    row[name]={k:np.mean([np.asarray(s[name][k]) for s in steps],axis=0).tolist() for k in steps[0][name]}
                evaluations.append(row)
    atomic_json(root/'metrics-summary.json',dict(scope=SCOPE,fields=FIELDS,runs=runs,evaluation=evaluations,
        aggregation='equal fixed selected windows, separate horizon/offset/dt/ratio/group/policy',
        checkpoint_selection='fixed_final_budget',caveat='no accuracy, source-response, long-term stability or C0 acceptance'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for seed,h in curves:
        axes[0].semilogy([r['update'] for r in h],[r['loss'] for r in h],label=f'seed {seed}: sampled train')
        for policy,style in [('TF','--'),('AR','-')]:
            vals=[r for r in evaluations if r['seed']==seed and r['group']=='same_run_validation' and r['policy']==policy and r['horizon']==4]
            by=defaultdict(list)
            for r in vals:by[r['update']].append(r['mean_loss'])
            axes[1].semilogy(sorted(by),[np.mean(by[i]) for i in sorted(by)],style,label=f'{seed} {policy}')
        vals=[r for r in evaluations if r['seed']==seed and r['group']=='same_run_validation' and r['policy']=='AR'
            and r['horizon']==4 and r['update']==runs[[x['seed'] for x in runs].index(seed)]['optimizer_updates']]
        axes[2].semilogy([r['offset'] for r in vals],[r['mean_loss'] for r in vals],'-o',label=f'{seed} final AR')
    for boundary in (20,40):
        axes[0].axvline(boundary,color='gray',linestyle=':',alpha=.5)
        axes[1].axvline(boundary,color='gray',linestyle=':',alpha=.5)
    axes[0].set(xlabel='optimizer update',ylabel='weighted loss',title='Sampled training curve')
    axes[1].set(xlabel='optimizer update',ylabel='weighted loss',title='Fixed same-run validation K=4')
    axes[2].set(xlabel='prediction step',ylabel='weighted loss',title='Fixed endpoint error growth')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle(SCOPE);fig.tight_layout();fig.savefig(root/'field-only-curves.png',dpi=140,metadata={'Description':SCOPE});plt.close(fig)
    fig,axes=plt.subplots(len(runs),2,figsize=(11,4*len(runs)),squeeze=False)
    for j,run in enumerate(runs):
        vals=[r for r in evaluations if r['seed']==run['seed'] and r['update']==run['optimizer_updates']
            and r['group']=='same_run_validation' and r['policy']=='AR' and r['horizon']==4]
        for i,key in enumerate(('coefficients','raw_history')):
            rmse=np.sqrt(np.mean([r['normalized_group_mse'][key] for r in vals],axis=0))
            im=axes[j,i].imshow(np.log10(np.maximum(rmse,1e-15)),aspect='auto')
            axes[j,i].set(yticks=range(9),yticklabels=FIELDS,xticks=range(4),
                xlabel='native degree',title=f'seed {run["seed"]}: {key}, log10 normalized RMSE')
            fig.colorbar(im,ax=axes[j,i])
    with (root/'field-degree-errors.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['scope','seed','field','degree','coefficient_normalized_rmse',
            'raw_history_normalized_rmse','gauss_value_normalized_rmse',
            'R_gradient_normalized_rmse','Z_gradient_normalized_rmse'])
        for run in runs:
            vals=[r for r in evaluations if r['seed']==run['seed'] and r['update']==run['optimizer_updates']
                and r['group']=='same_run_validation' and r['policy']=='AR' and r['horizon']==4]
            terms={k:np.sqrt(np.mean([r['normalized_group_mse'][k] for r in vals],axis=0))
                for k in vals[0]['normalized_group_mse']}
            for i,field in enumerate(FIELDS):
                for degree in range(4):
                    writer.writerow([SCOPE,run['seed'],field,degree,terms['coefficients'][i,degree],
                        terms['raw_history'][i,degree],terms['gauss_values'][i],
                        *terms['compatible_gradients'][i]])
    fig.suptitle(SCOPE+' / final same-run AR K=4');fig.tight_layout()
    fig.savefig(root/'field-only-degree-errors.png',dpi=140,metadata={'Description':SCOPE});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed,rows in probes:
        positive=[r for r in rows if r['factor']>0]
        for ax,key in zip(axes,('increment_scaled_rms','memory_change_rms')):
            ax.loglog([r['dt_ju'] for r in positive],[r[key] for r in positive],'-o',label=f'seed {seed}')
            ax.set(xlabel='positive dt [JU]',ylabel=key);ax.legend();ax.grid(alpha=.2)
    fig.suptitle(SCOPE+' / exact dt=0 is separately zero');fig.tight_layout()
    fig.savefig(root/'field-only-dt-limit.png',dpi=140,metadata={'Description':SCOPE});plt.close(fig)
    return evaluations


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();summarize(args.output_dir)
