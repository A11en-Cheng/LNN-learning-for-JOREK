#!/usr/bin/env python3
"""Bounded metadata/active-slot audit; no replay, training or qualification by fiat."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from src.data.native_state import audit_native_segment


def audit():
    registry=json.loads(Path('data/restart_events.json').read_text())['runs']
    selected={'phaseA_dataset_1d22_ang15_massloss_neon_v400':(4109,4110,4111,4112),
              'phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner':(2702,2703,2704,2705)}
    result={'scope':'bounded_entry_audit_not_window_or_anchor_certification','training':False,'runs':{}}
    for name,indices in selected.items():
        run=Path('data/EHL-2')/name
        records=audit_native_segment(run,indices,restart_events=registry[name]['restart_events'])
        for row in records:
            i=row['index']
            with h5py.File(run/f'jorek{i:06d}.h5') as f:
                row['field_history_saved']='deltas' in f and f['deltas'].shape==f['values'].shape
                row['unique_dofs']=int(np.unique(f['index'][...]).size)
            with h5py.File(run/f'projections{i:06d}.h5') as f:
                row['projection_history_saved']='deltas' in f
            row['marker_status']='unavailable_at_selected_path'
            if row['marker_present']:
                with h5py.File(run/f'part{i:05d}.h5') as f:
                    g=f['groups/001']
                    row['marker_slots']=int(g['i_elm'].size)
                    row['marker_active']=int(np.count_nonzero(g['i_elm'][...]>0))
                    row['marker_payload_bytes']=sum(g[k].size*g[k].dtype.itemsize for k in
                        ('x','st','i_elm','weight','P_imp','v','i_life','t_birth'))
                    row['marker_status']='saved_time_checked_not_anchor_certified'
            reasons=['real_grid_projection_not_implemented','production_source_branches_not_supported',
                     'physical_boundary_not_verified','external_next_step_control_not_supplied',
                     'complete_clock_and_projection_history_not_certified']
            if not row['marker_present']:reasons.append('matching_marker_snapshot_not_present_at_selected_path')
            row['qualification']={'field_records':'full_axes_and_saved_history' if row['field_history_saved'] else 'rejected_missing_history',
                'conditional_source_window':'not_certified','self_generated_source_window':'rejected',
                'reasons':reasons,'projection_difference_reconstruction':'pending_evidence_not_automatically_rejected'}
        result['runs'][name]=records
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite audit evidence')
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
