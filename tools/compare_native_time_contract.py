#!/usr/bin/env python3
"""Fixed-budget field-only comparison; no optimization or checkpoint selection."""
import argparse,json,sys
from pathlib import Path
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_window_io import atomic_json
from src.training.native_curriculum import SCOPE


def compare(old,new):
    old,new=Path(old),Path(new)
    configs=[json.loads((p/'experiment.json').read_text()) for p in (old,new)]
    expected=deepcopy(configs[0]);expected['model']['output_semantics']='rate_memory_v2'
    if expected!=configs[1]:raise ValueError('comparison changed more than declared time adapter')
    for file in ('manifest.json','normalizer.json','feature-statistics.json','parameter-routes.json'):
        if json.loads((old/'data'/file).read_text())!=json.loads((new/'data'/file).read_text()):
            raise ValueError('comparison data/statistics mismatch: '+file)
    metrics=[json.loads((p/'metrics-summary.json').read_text())['evaluation'] for p in (old,new)]
    key=lambda r:tuple(r[k] for k in ('seed','update','group','policy','horizon','offset','dt_bucket','ratio_bucket'))
    baseline={key(r):r for r in metrics[0]}
    if set(baseline)!={key(r) for r in metrics[1]}:raise ValueError('different fixed evaluation support')
    rows=[]
    for r in metrics[1]:
        b=baseline[key(r)]
        rows.append(dict(scope=SCOPE,**{k:r[k] for k in ('seed','update','group','policy','horizon','offset','dt_bucket','ratio_bucket','elapsed_ju')},
            old_loss=b['mean_loss'],new_loss=r['mean_loss'],difference=r['mean_loss']-b['mean_loss'],
            old_normalized_group_mse=b['normalized_group_mse'],new_normalized_group_mse=r['normalized_group_mse']))
    seeds=[]
    for seed in configs[0]['seeds']:
        paths=[p/f'seed-{seed}' for p in (old,new)]
        curves=[json.loads((p/'training-curves.json').read_text())['updates'] for p in paths]
        if len(curves[0])!=60 or len(curves[1])!=60:raise ValueError('comparison budget mismatch')
        # Compare actual sampler and teacher decisions, independent of losses.
        keys=('window_index','teacher_mask','stage','horizon')
        decisions_equal=all(all(a.get(k)==b.get(k) for k in keys) for a,b in zip(*curves))
        if not decisions_equal:raise ValueError('sampler/teacher decisions differ')
        seeds.append(dict(seed=seed,updates_per_arm=60,sampler_teacher_decisions_equal=True,
            old_summary=json.loads((paths[0]/'summary.json').read_text()),
            new_summary=json.loads((paths[1]/'summary.json').read_text()),
            old_dt=json.loads((paths[0]/'dt-limit.json').read_text()),
            new_dt=json.loads((paths[1]/'dt-limit.json').read_text()),
            new_subdivision=json.loads((paths[1]/'subdivision.json').read_text())))
    atomic_json(new/'comparison.json',dict(scope=SCOPE,baseline=str(old),new=str(new),
        identical_data_statistics=True,configuration_change='only model.output_semantics=rate_memory_v2',
        seeds=seeds,evaluation=rows,caveat='cross-run also changes dt; not pure parameter generalization'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for seed in configs[0]['seeds']:
        for arm,name,style in ((0,'old','--'),(1,'rate/memory','-')):
            for policy,ax in zip(('TF','AR'),axes[:2]):
                rs=[r for r in metrics[arm] if r['seed']==seed and r['group']=='same_run_validation' and r['horizon']==4 and r['policy']==policy]
                updates=sorted(set(r['update'] for r in rs))
                ax.semilogy(updates,[np.mean([r['mean_loss'] for r in rs if r['update']==i]) for i in updates],style,label=f'{seed} {name}')
                ax.set(xlabel='optimizer update',ylabel='weighted loss',title=f'Fixed same-run K4 {policy}')
            rs=[r for r in metrics[arm] if r['seed']==seed and r['group']=='same_run_validation' and r['horizon']==4 and r['policy']=='AR' and r['update']==60]
            axes[2].semilogy([r['offset'] for r in rs],[r['mean_loss'] for r in rs],style+'o',label=f'{seed} {name}')
    axes[2].set(xlabel='prediction step',ylabel='weighted loss',title='Final AR endpoint growth')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle(SCOPE+' / same data, seeds and 60 updates');fig.tight_layout()
    fig.savefig(new/'time-contract-comparison.png',dpi=140);plt.close(fig)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True);p.add_argument('--new',required=True)
    args=p.parse_args();compare(args.baseline,args.new)
