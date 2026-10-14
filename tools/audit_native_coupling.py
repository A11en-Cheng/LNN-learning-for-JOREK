#!/usr/bin/env python3
"""Finite phaseA source-coupling qualification audit; never inference inputs.

Run from repo root with PYTHONPATH=. No payload replay or directory scan.
"""
import argparse
import json
from pathlib import Path
from src.data.native_state import audit_native_segment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    registry=json.loads(Path('data/restart_events.json').read_text())['runs']
    selected={
        'phaseA_dataset_1d22_ang15_massloss_neon_v400':(4109,4110,4111,4112),
        'phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner':(2702,2703,2704,2705),
    }
    report={'scope':'finite_metadata_and_local_load_components_not_full_cloud_coupling','runs':{}}
    for run,indices in selected.items():
        rows=audit_native_segment(Path('data/EHL-2')/run,indices,restart_events=registry[run]['restart_events'])
        report['runs'][run]={
            'records':rows,'ordinary_projection_history':'missing_in_projection_HDF5',
            'full_cloud_coupled_qualification':'not_established',
            'reference_limits':['H=1','small_dense_unique_DOF_budget_128','small_cloud_10000_live_limit',
                'no_rocket_no_nonlocal','PLT_only_no_PRB_CX','unfiltered_projection'],
            'local_load_test_only':{'current':indices[2],'previous_projection':indices[1],
                'history_basis':'saved_previous_projection_is_entirely_zero; exact for_this_pair_only'},
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
