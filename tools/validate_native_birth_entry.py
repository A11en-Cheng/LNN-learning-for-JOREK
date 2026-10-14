"""Full real mesh/all-slot birth entry validation, no advance, loss or training.

Uses registered poststartup initial states and actual saved fluid history. It
does not claim a whole coupled event or predicted-field accuracy validation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.diagnose_native_real_chain import Stages,_initial_pairing
from tools.native_real_chain_builder import build_real
from src.models.marker_state import particle_window_schedule
from src.models.native_source_birth import NativeProductionBirths
from src.models.native_source_time import NativeSourceTimeSampler


def cloud_digest(cloud):
    digest=hashlib.sha256()
    for name in cloud.array_names():
        value=getattr(cloud,name)
        digest.update(name.encode());digest.update(value.tobytes())
    return digest.hexdigest()


def validate_case(run,anchor,*,seed=2917,cells_per_width=64):
    stages=Stages();rows=_initial_pairing(run,anchor,stages)
    options=SimpleNamespace(chunk_size=2048,wall_seconds=120.,memory_mib=8192,
        birth_max_proposals=1000000,spatial_backend='batched_v1',
        birth_cells_per_width=cells_per_width,birth_max_cells=16_000_000)
    model,source,shard,state,_=build_real(run,anchor,rows,stages,options,feedback_mode='autonomous')
    assert shard.birth.position_sampler is not None
    before=cloud_digest(state.marker.cloud)
    clock=state.marker.clock;target=state.fluid_time_si
    _,steps,gather=particle_window_schedule(clock,target)
    sampler=NativeSourceTimeSampler(state.fields_current,state.fluid_deltas,
        time_prev_si=clock.field_start_si,time_now_si=clock.field_end_si,
        tstep_si=state.fluid_dt_previous_si,static=source.field_static)
    settings=shard.birth.configuration
    reference=NativeProductionBirths(shard.density,**{k:settings[k] for k in
        ('physical_particles','n_particle_slots','n_cpu','rank','R_geo','n_plane','max_proposals')})
    rate=state.shards['ablation_atoms_s'].sum()/settings['n_cpu']
    result=dict(run=str(run),anchor=anchor,slots=len(state.marker.cloud.weight),
        original_live=int(state.marker.cloud.alive.sum()),particle_steps=steps,gather_si=gather,
        field_updates=0,marker_advance_events=0,no_future_truth_feedback=True,
        builder_mode=shard.birth.position_sampler.geometry,seed=seed,cases=[])
    saved=deepcopy(np.random.get_state())
    try:
        for name,birth in (('corrected_rejection_reference',reference),('native_probability',shard.birth)):
            np.random.seed(seed)
            cloud=birth.births(state.shards,state.marker,target,state.fields_previous,
                state.fields_current,source,time_sampler=sampler)
            cloud.validate()
            expected_count=int(rate*gather/birth.weight)
            if rate*gather>1e14:expected_count=max(expected_count,1)
            assert len(cloud.weight)==expected_count and cloud.alive.all()
            assert np.isfinite(cloud.V_prev).all()
            assert np.all(cloud.P_imp[:,0]==1) and not cloud.P_imp[:,1:].any()
            assert np.all((cloud.st>=0)&(cloud.st<=1))
            assert cloud_digest(state.marker.cloud)==before
            result['cases'].append(dict(mode=name,birth_count=len(cloud.weight),
                total_weight_relative_error=abs(cloud.weight.sum()/(rate*gather)-1),
                original_cloud_unchanged=True,neutral_charge_state=True,finite_initial_velocity=True,
                valid_elements=int(np.count_nonzero(cloud.i_elm>0)),configuration=birth.configuration,
                position_sha256=hashlib.sha256(cloud.x.tobytes()).hexdigest()))
    finally:np.random.set_state(saved)
    assert result['cases'][0]['birth_count']==result['cases'][1]['birth_count']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2917)
    parser.add_argument('--cells-per-width',type=int,default=64)
    args=parser.parse_args()
    if not 0<=args.seed<2**32:parser.error('--seed must be in [0, 2**32)')
    if args.output.exists():raise ValueError('new output file required')
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/'tools/native_real_chain_builder.py',root/'src/models/native_birth_acceptance.py',
        root/'src/models/native_source_birth.py',root/'src/models/native_birth_positions.py']
    result=dict(scope='real_full_mesh_all_slot_birth_entry_only_no_complete_event',no_training=True,seed=args.seed,
        source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},cases=[])
    for suffix,anchor in (('',4110),('_injInner',5680)):
        run=root/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        result['cases'].append(validate_case(run,anchor,seed=args.seed,cells_per_width=args.cells_per_width))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),birth_counts=[c['cases'][1]['birth_count'] for c in result['cases']])))


if __name__=='__main__':main()
