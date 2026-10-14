"""Render per-field contributions and errors without mixing physical fields."""
import argparse
import csv
import json
from pathlib import Path


def field_rows(result):
    rows=[]
    for phase in ('before','after'):
        for window in result[phase]:
            for step,event in enumerate(window['events'],1):
                for name,metrics in event['fields'].items():
                    rows.append(dict(phase=phase,window=window['window'],forecast_step=step,
                        field=name,loss_contribution=metrics['loss'],relative_l2=metrics['relative_l2'],
                        pearson_r=metrics['pearson_r']))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();result=json.loads(args.result.read_text());rows=field_rows(result)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'per-field.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=list(result['after'][0]['events'][0]['fields'])
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for i,window in enumerate(result['after']):
        name=window['window']
        for phase,offset in [('before',-.18),('after',.18)]:
            group=[r for r in rows if r['window']==name and r['phase']==phase]
            losses=[sum(r['loss_contribution'] for r in group if r['field']==f)/len(window['events']) for f in fields]
            errors=[next(r['relative_l2'] for r in group if r['field']==f and r['forecast_step']==len(window['events'])) for f in fields]
            x=[j+offset for j in range(9)]
            axes[i,0].bar(x,losses,width=.36,label=phase)
            axes[i,1].bar(x,errors,width=.36,label=phase)
        for j in range(2):
            axes[i,j].set_xticks(range(9),fields,rotation=35);axes[i,j].set_yscale('log');axes[i,j].legend()
            axes[i,j].set_title(name.split(':')[0].replace('phaseA_dataset_1d22_ang15_massloss_neon_','')+
                (' mean field loss contribution' if j==0 else ' final-step Gauss relative L2'))
    fig.suptitle('Two optimizer updates only; not a full phaseA epoch or convergence result')
    fig.savefig(args.output_dir/'per-field.png',dpi=150);plt.close(fig)
    for window in result['after']:
        print(window['window'])
        for name in fields:
            es=window['events'];loss=sum(e['fields'][name]['loss'] for e in es)/len(es)
            print(f"{name:8s} mean_loss={loss:.9g} final_relative_L2={es[-1]['fields'][name]['relative_l2']:.9g}")


if __name__=='__main__':main()
