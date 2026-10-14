"""Position-generation-only snapshot benchmark; no locate/fields/cloud/training.

Legacy proposal reproduction is deliberately isolated from mesh rejection.
All shards and their recorded volumes are retained. The n0 table is a declared
new angular-average mode, not identical legacy RNG or continuous-source replay.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import h5py
import numpy as np

from src.models.native_birth_positions import NativeTabulatedBirthSampler
from src.models.native_source_density import NativeShardDensityKernel
from src.models.native_state import NativeUnits
from src.models.native_birth_acceptance import NATIVE_BIRTH_ACCEPTANCE,native_birth_threshold


def load_snapshot(path):
    """Only source/normalization metadata: intentionally no FEM mesh creation."""
    from src.data.namelist import parse_namelist
    path=Path(path)
    params=parse_namelist((path.parent/'jorek_namelist_spitest').read_text())
    with h5py.File(path,'r') as f:
        scalar=lambda k:float(np.ravel(f[k][...])[0])
        if scalar('n_tor')!=1 or scalar('n_plane')!=1 or scalar('n_period')!=1:
            raise ValueError('benchmark requires inspected n0 snapshot')
        units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
        coordinates=f['x'][...]
        domain=coordinates[:,0,0].T
        bounds=np.stack([domain.min(0),domain.max(0)])
        now=scalar('t_now')*units.t_norm_si
        s=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
            radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],species=f['spi_species_arr'][...],
            source_volume_m3=f['spi_vol_arr'][...],source_volume_time_si=now,
            source_volume_provenance=str(path)+'::spi_vol_arr')
        dt=scalar('tstep')*units.t_norm_si
    mesh=SimpleNamespace(units=units,configuration=dict(
        geometry=hashlib.sha256(coordinates.tobytes()).hexdigest(),scope='birth_metadata_only_no_FEM'))
    numerical=params.get('spi_num_vol')
    numerical_origin=str(path.parent/'jorek_namelist_spitest')+'::spi_num_vol'
    if numerical is None:
        log_path=path.parent/'logfile_spitest'
        recorded=set(re.findall(r'^\s*spi_num_vol\s*=\s*([TF])\s*$',log_path.read_text(),re.MULTILINE))
        if len(recorded)!=1:raise ValueError('unambiguous logged spi_num_vol required')
        numerical=recorded.pop()=='T'
        numerical_origin=str(log_path)+'::spi_num_vol'
    if not isinstance(numerical,bool):
        if str(numerical).lower() not in ('.t.','.true.','.f.','.false.'):
            raise ValueError('explicit spi_num_vol required')
        numerical=str(numerical).lower() in ('.t.','.true.')
    kernel=NativeShardDensityKernel(mesh,ns_radius_min=float(params['ns_radius_min']),
        ns_radius_ratio=float(params['ns_radius_ratio']),ns_deltaphi=float(params['ns_deltaphi']),
        spi_num_vol=numerical,ns_delta_minor_rad=0.,injection_time_si=float(params['t_ns'])*units.t_norm_si,
        toroidal_angles=(0.,))
    widths=kernel.widths(s)
    lower=np.min(s['positions'][:,:2]-5*widths[:,None],axis=0)
    upper=np.max(s['positions'][:,:2]+5*widths[:,None],axis=0)
    for d in range(2):
        if upper[d]>bounds[0,d] and bounds[1,d]>lower[d]:
            bounds[0,d]=max(bounds[0,d],lower[d]);bounds[1,d]=min(bounds[1,d],upper[d])
    rate=float(np.sum(s['ablation_atoms_s']*s['species']))
    norm=rate*units.t_norm_si/units.n_norm_m3
    dv=float(np.float32(float(np.float32(.01)*np.float32(.01))*float(params['r_geo'])*2*np.pi))
    weight=float(params['spi_quantity'])/1_000_000
    count=int(rate*dt/weight)
    if rate*dt>1e14:count=max(count,1)
    if count<1:raise ValueError('snapshot fluid-step birth budget has no births')
    return kernel,s,dict(time_si=now,bounds=bounds,dv=dv,norm=norm),count,dict(
        snapshot=str(path),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shard_count=len(s['radius']),live_source_shards=int(np.count_nonzero((s['radius']>0)&(s['ablation_atoms_s']>0))),
        count=count,count_origin='recorded_tstep_times_ablation_over_million_slot_nominal_weight_not_particle_gather_replay',
        ns_deltaphi=kernel.deltaphi,ns_radius_min=kernel.minimum,spi_num_vol_origin=numerical_origin)


def legacy_positions(kernel,s,options,count,*,max_proposals=1_000_000,max_seconds=30.):
    """Literal seven-uniform physical-source gate; excludes mesh gate by scope."""
    tick=time.perf_counter()
    prepared=kernel.prepare_rates(s,time_si=options['time_si'])
    lo,hi=options['bounds'];points=np.zeros((count,3));pending=np.arange(count)
    proposals=0
    while len(pending):
        rejected=[]
        for j in pending:
            if proposals>=max_proposals or (proposals%256==0 and time.perf_counter()-tick>max_seconds):
                raise RuntimeError('bounded legacy birth generation budget exceeded')
            ran=np.random.random(7);proposals+=1
            x=np.array([np.sqrt(ran[0]*(hi[0]**2-lo[0]**2)+lo[0]**2),
                        lo[1]+ran[1]*(hi[1]-lo[1]),2*np.pi*ran[2]])
            a=float(native_birth_threshold(prepared.rates_at(x[None])[0],dv=options['dv'],norm=options['norm']))
            if ran[3]>=a:rejected.append(j)
            else:points[j]=x
        pending=np.asarray(rejected,dtype=int)
    return points,proposals


def probabilities_on_support(table,r_edges,z_edges):
    """Exact n0 bilinear-table integral on arbitrary target rectangular bins."""
    if table.probabilities.shape[-1]!=1:raise ValueError('n0 support integral required')
    def basis_integrals(target,source):
        lo=np.maximum(target[:-1,None],source[None,:-1])
        hi=np.minimum(target[1:,None],source[None,1:])
        length=np.maximum(hi-lo,0.)
        # Local differences avoid cancellation when a tiny interval is far
        # from coordinate zero (notably native R**2 volume coordinates).
        mid=np.clip(((lo-source[None,:-1])+(hi-source[None,:-1]))/(2*np.diff(source)[None,:]),0.,1.)
        right=length*mid
        return length-right,right
    a0,a1=basis_integrals(.5*np.asarray(r_edges)**2,.5*table.r_edges**2)
    b0,b1=basis_integrals(np.asarray(z_edges),table.z_edges)
    nodes=table.acceptance_nodes[:,:,0]
    integral=(a0@nodes[:-1,:-1]@b0.T+a1@nodes[1:,:-1]@b0.T
              +a0@nodes[:-1,1:]@b1.T+a1@nodes[1:,1:]@b1.T)*2*np.pi
    return integral/table.diagnostics['acceptance_integral_m3']


def benchmark_case(path,*,repeats=3,cells_per_width=8,seed=2917):
    kernel,s,options,count,record=load_snapshot(path)
    sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,cells_per_width=cells_per_width,max_cells=16_000_000)
    saved=deepcopy(np.random.get_state())
    rows=[];table=None
    try:
        for repeat in range(repeats):
            for mode in (('legacy','table') if repeat%2==0 else ('table','legacy')):
                np.random.seed(seed+repeat)
                tick=time.perf_counter();cpu=time.process_time()
                if mode=='legacy':
                    _,proposals=legacy_positions(kernel,s,options,count)
                    build_seconds=None;sample_seconds=None
                else:
                    table=sampler.prepare(kernel,s,**options)
                    build_seconds=time.perf_counter()-tick
                    sample_tick=time.perf_counter();table.sample(count)
                    sample_seconds=time.perf_counter()-sample_tick;proposals=count
                rows.append(dict(mode=mode,repeat=repeat,seconds=time.perf_counter()-tick,
                    process_cpu_seconds=time.process_time()-cpu,build_seconds=build_seconds,
                    sample_seconds=sample_seconds,source_proposals=proposals))
                print(json.dumps(dict(snapshot=str(path),**rows[-1])),flush=True)
        summary={mode:statistics.median(r['seconds'] for r in rows if r['mode']==mode)
                 for mode in ('legacy','table')}
        # One higher-resolution diagnostic is outside all performance timings.
        refined=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,
            cells_per_width=2*cells_per_width,max_cells=64_000_000).prepare(kernel,s,**options)
        # Both directions are needed: coarse-cell integrals alone can conceal
        # in-cell curve distortion. Integrate the actual linear reconstruction.
        truth=probabilities_on_support(refined,table.r_edges,table.z_edges)
        current=table.probabilities[:,:,0]
        relative=float(np.linalg.norm(current-truth)/np.linalg.norm(truth))
        fine=refined.probabilities[:,:,0]
        coarse_on_fine=probabilities_on_support(table,refined.r_edges,refined.z_edges)
        record.update(scope='source_position_generation_only_build_included_no_mesh_no_field_no_marker_advance',
            production_qualified=False,iterations=rows,median_seconds=summary,
            generation_speedup=summary['legacy']/summary['table'],configuration=sampler.configuration,
            diagnostics=table.diagnostics,
            refinement=dict(coarse_cells=table.diagnostics['table_cells'],fine_cells=refined.diagnostics['table_cells'],
                cell_probability_relative_l2=relative,total_variation=.5*float(np.sum(np.abs(current-truth))),
                refined_support_relative_l2=float(np.linalg.norm(coarse_on_fine-fine)/np.linalg.norm(fine)),
                refined_support_total_variation=.5*float(np.sum(np.abs(coarse_on_fine-fine))),
                acceptance_integral_relative_error=abs(table.diagnostics['acceptance_integral_m3']/refined.diagnostics['acceptance_integral_m3']-1),
                reference='2x resolution table_not_exact_continuous_solution'),
            maximum_kernel_toroidal_relative_variation=1-float(np.exp(-(np.pi/kernel.deltaphi)**2)))
        return record
    finally:
        np.random.set_state(saved)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--cells-per-width',type=int,default=64)
    parser.add_argument('--seed',type=int,default=2917)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new output file required; previous evidence must remain')
    if not 0<=args.seed<2**32-3:parser.error('--seed must leave room for repeat offsets')
    root=Path(__file__).resolve().parents[1]/'data/EHL-2'
    repo=Path(__file__).resolve().parents[1]
    code_paths=[Path(__file__),repo/'src/models/native_birth_positions.py',repo/'src/models/native_source_birth.py',
                repo/'src/models/native_source_density.py',repo/'src/models/native_birth_acceptance.py']
    result=dict(schema='native-birth-position-benchmark-v3',acceptance=NATIVE_BIRTH_ACCEPTANCE,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),no_training=True,
        mesh_location_changed=False,threads=1,cases=[],numpy_version=np.__version__,seed=args.seed,
        source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths})
    # Set before Python/NumPy startup; don't silently install a new dependency
    # or mutate process thread settings after libraries have been initialized.
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('launch benchmark with '+key+'=1')
    from tools.benchmark_native_cpu_threads import native_thread_counts
    result['native_thread_counts']=native_thread_counts()
    for suffix,index in (('',4111),('_injInner',5681)):
        path=root/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)/f'jorek{index:06d}.h5'
        result['cases'].append(benchmark_case(path,repeats=args.repeats,cells_per_width=args.cells_per_width,seed=args.seed))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(result=str(args.output),cases=[dict(snapshot=r['snapshot'],
        speedup=r['generation_speedup'],relative_l2=r['refinement']['cell_probability_relative_l2']) for r in result['cases']])),flush=True)


if __name__=='__main__':main()
