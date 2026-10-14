#!/usr/bin/env python3
"""Clock-only research: independently check later logged arithmetic, no cloud replay."""
import argparse
import json
from pathlib import Path
import re
import h5py
import numpy as np

from src.data.namelist import parse_namelist


RUN = Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner')


def scalar(path, key):
    with h5py.File(path, 'r') as f:
        return float(np.ravel(f[key][()])[0])


def projection_time(index):
    return scalar(RUN/f'projections{index:06d}.h5','t_now')


def candidate(reset_index, last_index):
    h5 = RUN/f'jorek{reset_index:06d}.h5'
    imported = scalar(h5,'tstep')*scalar(h5,'t_norm')
    nominal = float(parse_namelist((RUN/'jorek_namelist_spitest').read_text())['tstep_particles'])
    dt = imported/int(imported/nominal)
    sim = projection_time(reset_index)
    residual = 0.
    rows = []
    for index in range(reset_index+1,last_index+1):
        target = projection_time(index)
        start = sim-residual
        count = int((target-start)/dt)
        gather = count*dt
        residual = target-start-gather
        rows.append(dict(index=index,sim_time_si=sim,particle_start_si=start,target_time_si=target,
            projection_source=str(RUN/f'projections{index:06d}.h5'),
            n_steps=count,gather_si=gather,residual_after_si=residual,particle_dt_si=dt))
        sim = target
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('refuse to overwrite evidence')
    # Stream only first three complete particle blocks, avoiding full 447MB log.
    events=[]; block={}; lines={}
    pattern=re.compile(r'PARTICLE : (sim%time|particle_start_time|particle_step_time|n_steps|step_rest_time)\s*:\s*([^\s]+)')
    with (RUN/'logfile_spitest').open() as f:
        for lineno,line in enumerate(f,1):
            match=pattern.search(line)
            if match:
                key,value=match.groups(); block[key]=float(value); lines[key]=lineno
            written=re.search(r'Written projection to projections(\d+)\.h5',line)
            if written and block:
                events.append(dict(index=int(written[1]),logged=block,lines=lines,output_line=lineno))
                block={};lines={}
                if len(events)==3: break
    assert len(events)==3 and events[0]['index']==9291
    dt=scalar(RUN/'jorek009290.h5','tstep')*scalar(RUN/'jorek009290.h5','t_norm')
    nominal=float(parse_namelist((RUN/'jorek_namelist_spitest').read_text())['tstep_particles'])
    dt/=int(dt/nominal)
    for event in events:
        logged=event['logged']; count=int(logged['particle_step_time']/dt)
        rest=logged['particle_step_time']-count*dt
        assert count==logged['n_steps']
        assert abs(rest-logged['step_rest_time'])<1e-22
        assert abs(projection_time(event['index']-1)-logged['sim%time'])<1e-14
        # particle_step_time has 17-digit log precision. target_time's printed
        # 8 digits is insufficient to derive an integer substep count.
        assert abs(projection_time(event['index'])-(logged['particle_start_time']+logged['particle_step_time']))<1e-14
        event['verified_arithmetic']=dict(particle_dt_si=dt,n_steps=count,
            residual_si=rest,gather_si=count*dt)
    rows=candidate(2703,2709)
    marker_clock=scalar(RUN/'part02710.h5','time')
    assert abs(rows[-1]['target_time_si']-marker_clock)<1e-14
    result=dict(training=False,cloud_replay=False,
        independent_logged_cases=events,log_source=str(RUN/'logfile_spitest'),
        validation_scope='9291-9293 logged arithmetic and projection/event times only; not target dirty-build equivalence',
        candidate_2710=dict(status='pending_evidence',reset_anchor=2703,
            initialization_sources=[str(RUN/'jorek002703.h5'),str(RUN/'projections002703.h5'),
                str(RUN/'jorek_namelist_spitest'),'data/restart_events.json'],
            algorithm='current marker_spi_source: imported dt partition; zero restart residual; each saved projection event time; int truncation and residual carry',
            rows=rows,clock_at_snapshot=dict(sim_time_si=marker_clock,
                particle_dt_si=rows[-1]['particle_dt_si'],step_rest_time=rows[-1]['residual_after_si']),
            prerequisites=['effective target nominal substep configuration',
                'production source equivalence for clock and scheduler',
                'complete event chain/no skipped particle updates; projection output time is scheduler event time',
                'projection float64 output transform certified'],
            not_certified_reason='later logs verify rule locally, not 2704-2709 production event completeness; integer truncation sensitive to rounding',
            no_future_physical_history=True),
        projection_delta_rule=dict(status='pending_evidence',
            candidate='rawDP_k=P_k-P_previous_update',
            independently_verified_case='registered restart: zero values witness and restart initialization returns zero rawDP',
            ordinary_validation='no independent saved rawDP or update-level production witness found',
            prerequisites=['same node object/layout and consecutive actual updates',
                'no missing/duplicate update/restart', 'post-solve filtered/BC stage matches rawDP operands',
                'lossless output transform and production source path', 'projection and marker event pairing'],
            permitted_scope='registered zero-history only; ordinary difference never admitted by this audit'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['candidate_2710']['clock_at_snapshot'],indent=2))


if __name__=='__main__': main()
