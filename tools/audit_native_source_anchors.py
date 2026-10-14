#!/usr/bin/env python3
"""Two explicit candidate anchors; unavailable state stays unavailable. No replay."""
import argparse,json,time,resource
from pathlib import Path
import h5py,numpy as np
from src.data.marker_checkpoint import read_marker_snapshot,initialize_native_checkpoint
from src.data.native_state import read_native_field_state
from src.models.native_state import NativeUnits
from src.data.native_run import _read_native_shard
from src.data.output_contract import parse_macroscopic_times


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite evidence')
    result={'scope':'candidate_state_inventory_not_certified_coupled_anchors','training':False,'anchors':[]}
    for suffix,index in [('',4110),('_injInner',2710)]:
        run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        table=parse_macroscopic_times(run/'macroscopic_vars.dat')
        start=time.perf_counter()
        registry=json.loads(Path('data/restart_events.json').read_text())['runs']
        restart=index+1 in registry[run.name]['restart_events']
        clock_evidence=None
        if restart:
            checkpoint=initialize_native_checkpoint(run/f'jorek{index:06d}.h5',run/f'part{index:05d}.h5',
                event_index=index,times_path=run/'macroscopic_vars.dat',namelist_path=run/'jorek_namelist_spitest',
                boundary='jorek_restart',previous_fluid_path=run/f'jorek{index-1:06d}.h5')
            cloud=checkpoint.marker.cloud;t=checkpoint.marker.clock.sim_time_si
            clock_evidence=vars(checkpoint.marker.clock).copy()
        else:
            cloud,t,metadata=read_marker_snapshot(run/f'part{index:05d}.h5')
        with h5py.File(run/f'jorek{index:06d}.h5') as f:
            scalar=lambda k:float(np.asarray(f[k][...]).ravel()[0])
            tn=scalar('t_norm')
            units=NativeUnits(tn,scalar('central_density')*1e20,scalar('central_mass'),5/3)
            expected=table[index-1]*tn
            assert abs(t-expected)<max(1e-14,tn*1e-7)
            fields={'values':('saved' if 'values' in f else 'unavailable'),
                    'raw_fluid_deltas':('saved' if 'deltas' in f else 'unavailable'),
                    'previous_field':('saved' if (run/f'jorek{index-1:06d}.h5').exists() else 'unavailable'),
                    'shard':('saved' if 'spi_R_arr' in f else 'unavailable')}
            source_step=float(np.asarray(f['tstep'][...]).ravel()[0])
        current=read_native_field_state(run/f'jorek{index:06d}.h5',units=units,modes=(0,))
        previous=read_native_field_state(run/f'jorek{index-1:06d}.h5',units=units,modes=(0,))
        assert abs(previous.time_si-t)<max(1e-14,tn*1e-7)
        _read_native_shard(run/f'jorek{index:06d}.h5',index,side='checkpoint')
        projection_meta=[]
        for i in ((index-1,index) if restart else (index-2,index-1)):
            with h5py.File(run/f'projections{i:06d}.h5') as f:
                if restart and i==index:
                    assert not np.asarray(f['values'][...]).any(), 'registered zero projection witness missing'
                projection_meta.append(dict(index=i,time_si=float(np.asarray(f['t_now'][...]).ravel()[0]),
                    dtype=str(f['values'].dtype),saved_deltas='deltas' in f,
                    scaleoffset=f['values'].scaleoffset,rcs=str(np.asarray(f['RCS_version'][...]).ravel()[0])))
        gaps=['source_model_rocket6_nonlocal_filter_PRB_not_supported',
              'physical_boundary_rules_not_verified','external_next_control_not_supplied']
        fields.update(marker='saved',projection_values='saved',
            projection_history='verified_reconstructed' if restart else 'unavailable',
            clock='verified_reconstructed' if restart else 'unavailable',
            next_control='unavailable',event_phase='verified_reconstructed' if restart else 'unavailable')
        if not restart:gaps+=['continuous_particle_dt_residual_actual_gather_not_certified',
                              'projection_history_build_and_output_chain_witness_missing']
        result['anchors'].append(dict(run=run.name,index=index,marker_time_si=t,
            field_time_si=table[index]*tn,imported_tstep_ju=source_step,marker_slots=len(cloud.weight),
            marker_active=int(cloud.alive.sum()),marker_payload_bytes=sum(getattr(cloud,k).nbytes for k in cloud.array_names()),
            read_validate_seconds=time.perf_counter()-start,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            state_status=fields,projection_files=projection_meta,registered_restart_clock=clock_evidence,
            projection_history_evidence='registered JOREK restart zeros only' if restart else
                'float64 unquantized saved values; production RCS is dirty and differs from inspected HEAD; difference remains uncertified',
            qualifying_boundary='registered_restart_next_event4111' if restart else 'ordinary_continuation_only',
            coupled_eligible=False,reasons=gaps))
        del cloud
        if restart:del checkpoint
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)

if __name__=='__main__':main()
