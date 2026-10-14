#!/usr/bin/env python3
"""No-training production-config, real-amplitude and isolated-clock diagnostics.

No absent projection history is filled. Clock-only linear contributions are
not a source-conditioned forward on real data and do not certify an anchor.
"""
import argparse,json
from pathlib import Path
import h5py,numpy as np,torch
from src.data.namelist import parse_namelist


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite evidence')
    from tests.test_native_joint_source import joint_fixture
    model,*_=joint_fixture(seed=23)
    weight=model.source_encoding.projection.weight.detach()[:,99].numpy()
    rows=[]
    for suffix,index in [('',4112),('_injInner',2710)]:
        run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        nl=parse_namelist((run/'jorek_namelist_spitest').read_text())
        keys=('spi_abl_model','spi_rkt_model','nonlocal_abl','filter_perp','filter_hyper','filter_par',
              'filter_perp_n0','filter_hyper_n0','filter_par_n0','ns_radius_ratio','ns_radius_min')
        config={k:nl.get(k) for k in keys}
        with h5py.File(run/f'jorek{index:06d}.h5') as f:
            tn=float(np.asarray(f['t_norm'][...]).ravel()[0]);ablation=f['spi_abl_arr'][...]
            boundary,cnt=np.unique(f['boundary'][...],return_counts=True)
            bcs={str(int(k)):int(n) for k,n in zip(boundary,cnt)}
        with h5py.File(run/f'projections{index:06d}.h5') as f:
            a=f['values'][...];t=float(np.asarray(f['t_now'][...]).ravel()[0])
            saved='deltas' in f;rcs=str(np.asarray(f['RCS_version'][...]).ravel()[0])
        ju=t/tn;linear=weight*ju
        rows.append(dict(run=run.name,index=index,production_parameters=config,
            projection_absmax_field_degree=np.abs(a).max(axis=(2,3)).tolist(),
            projection_min_field_degree=a.min(axis=(2,3)).tolist(),
            projection_max_field_degree=a.max(axis=(2,3)).tolist(),
            projection_history_saved=saved,projection_rcs=rcs,boundary_counts=bcs,
            ablation_atoms_s=dict(min=float(ablation.min()),max=float(ablation.max()),sum=float(ablation.sum())),
            isolated_clock=dict(event_ju=ju,reference_scale_ju=1.,seed=23,
                linear_contribution=linear.tolist(),tanh_saturation_fraction=float((np.abs(np.tanh(linear))>.999).mean()),
                warning='isolated absolute-clock contribution only; not full real source encoding'),
            support={'ablation3':'local_pure_Ne_approximation_only',
                'rocket6':'unimplemented','nonlocal':'unimplemented',
                'birth':'one_center_reference_not_rejection_sampling',
                'density_kernel':'resolved_Gaussian_reference_not_certified_target_kernel',
                'projection':'scalable_unfiltered_unconstrained_reference',
                'radiation':'PLT_only_approximation_missing_active_PRB',
                'CX':'not_required_by_inspected_get_Lz_branch',
                'physical_boundary':'unimplemented_no_production_admission'}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'training':False,'rows':rows},indent=2)+'\n');print(args.output)

if __name__=='__main__':main()
