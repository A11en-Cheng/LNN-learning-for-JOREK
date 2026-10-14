"""Read-only attribution of archived fixed-field projection differences.

Uses saved native Gauss norms for physical ratios and coefficient subtraction
only to test equal-background/additive structure, never as a physical norm.
No marker execution, field advance, RNG draws or training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.validate_native_birth_projection import saved_coefficients


def attribute_pair(whole, newborn, reference_whole, candidate_whole,
                   reference_newborn, candidate_newborn):
    background_a = reference_whole-reference_newborn
    background_b = candidate_whole-candidate_newborn
    difference = candidate_whole-reference_whole
    identity_residual = difference-(candidate_newborn-reference_newborn)
    denominator = float(np.linalg.norm(difference))
    result = dict(background_identity_residual_relative=(
        float(np.linalg.norm(identity_residual)/denominator) if denominator else None),
        background_identity_status='defined' if denominator else 'undefined_zero_difference',
        background_subtraction_bitwise_equal=bool(np.array_equal(background_a,background_b)),
        channels=[])
    coefficient_differences=np.atleast_2d(difference)
    coefficient_residuals=np.atleast_2d(identity_residual)
    for index,(full, born) in enumerate(zip(whole,newborn)):
        total_norm = full['reference_norm']
        fraction = born['reference_norm']/total_norm if total_norm else None
        predicted = (fraction*born['relative_l2'] if fraction is not None
                     and born['relative_l2'] is not None else None)
        whole_integral = full['reference_integral']
        result['channels'].append(dict(channel=full['channel'],
            coefficient_identity_residual_relative=(
                float(np.linalg.norm(coefficient_residuals[index])/np.linalg.norm(coefficient_differences[index]))
                if np.linalg.norm(coefficient_differences[index]) else None),
            whole_reference_norm=total_norm,newborn_reference_norm=born['reference_norm'],
            newborn_norm_fraction=fraction,
            whole_relative_l2=full['relative_l2'],newborn_relative_l2=born['relative_l2'],
            additive_prediction_relative_l2=predicted,
            whole_integral_relative_error=full['integral_relative_error'],
            newborn_integral_fraction=(born['reference_integral']/whole_integral if whole_integral else None),
            reference_whole_integral=whole_integral,
            reference_newborn_integral=born['reference_integral'],
            reference_existing_integral=whole_integral-born['reference_integral']))
    return result


def analyze(directory):
    summary_path=directory/'summary.json';gauss_path=directory/'gauss-metrics.json'
    summary=json.loads(summary_path.read_text());gauss=json.loads(gauss_path.read_text())
    comparisons=gauss['comparisons']
    rows=[]
    for whole in comparisons:
        if whole['scope']!='whole':continue
        matches=[r for r in comparisons if r['scope']=='newborn'
                 and r['seed']==whole['seed'] and r['other']==whole['other']]
        if len(matches)!=1:raise ValueError('one matching newborn comparison required')
        seed=whole['seed'];other=whole['other']
        a=saved_coefficients(directory,f'reference_{seed}','whole')
        b=saved_coefficients(directory,f'{other}_{seed}','whole')
        na=saved_coefficients(directory,f'reference_{seed}','newborn')
        nb=saved_coefficients(directory,f'{other}_{seed}','newborn')
        result=attribute_pair(whole['channels'],matches[0]['channels'],a,b,na,nb)
        rows.append(dict(seed=seed,other=other,**result))
    return dict(anchor=summary['anchor'],directory=str(directory),
                cells_per_width=summary['cells_per_width'],
                scope='archived_fixed_field_window_conditional_birth_attribution_not_new_execution',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (summary_path,gauss_path)},pairs=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directories',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output file required')
    result=dict(no_training=True,marker_advances=0,field_advances=0,
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cases=[analyze(d) for d in args.directories])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    for case in result['cases']:
        print(json.dumps(dict(anchor=case['anchor'],cells_per_width=case['cells_per_width'],
            Eion=[dict(seed=r['seed'],other=r['other'],
                      background_identity_residual_relative=r['background_identity_residual_relative'],
                      **r['channels'][0]) for r in case['pairs']])),flush=True)


if __name__=='__main__':main()
