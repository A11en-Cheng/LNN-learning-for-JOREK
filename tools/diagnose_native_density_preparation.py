"""Bounded CPU-only real density differential; no markers, events or training.

The reference preserves the former per-element (16 points), per-shard scalar
expression and accumulation order. Both branches use the SAME sparse factor.
"""
import argparse
import hashlib
import json
from math import pi
import os
from pathlib import Path
import resource
import signal
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/native-cpu-latency-2026-10-09/density-input-check.json'))
    parser.add_argument('--wall-seconds',type=int,default=180)
    parser.add_argument('--memory-mib',type=int,default=2048)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('new evidence path required')
    if args.wall_seconds<=0 or args.memory_mib<=0: raise ValueError('positive explicit resource budget required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    import h5py
    import numpy as np
    import torch
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_source_density import NativeShardDensityKernel
    from src.models.native_state import NativeUnits
    from src.data.namelist import parse_namelist
    from tools.native_replay_metrics import relative_metrics
    from tools.diagnose_native_real_chain import _array_digest
    torch.set_num_threads(1)
    started=time.monotonic()
    result=dict(schema='native-density-input-differential-v1',success=False,runs=[],
        scope='CPU_all_846_shards_all_Gauss_points_no_cloud_no_events_no_training_no_GPU',
        budget=dict(wall_seconds=args.wall_seconds,memory_mib=args.memory_mib,threads=1),
        oracle='independent_old_scalar_rates_at_per_element_16_points_per_shard_order',
        volume_policy='refresh_from_actual_initial_shards_once_as_real_builder_then_freeze_identical_input',
        production_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path('src/models/native_source_density.py'),Path('src/models/native_coupling.py'))})
    stage='start'
    def deadline(*unused): raise TimeoutError('density-only explicit wall budget exceeded')
    signal.signal(signal.SIGALRM,deadline); signal.alarm(args.wall_seconds)
    def check():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>args.memory_mib*1024**2:
            raise MemoryError('density-only process RSS budget exceeded')
        if time.monotonic()-started>args.wall_seconds: deadline()
    def evidence(a,b):
        a=np.asarray(a);b=np.asarray(b)
        diff=np.abs(a-b); different=a!=b
        return dict(exact=bool(np.array_equal(a,b)),reference_digest=_array_digest(a),
            optimized_digest=_array_digest(b),reference_strides=list(a.strides),optimized_strides=list(b.strides),
            differing_count=int(np.count_nonzero(different)),shape=list(a.shape),
            first_differences=np.argwhere(different)[:8].tolist(),metrics=relative_metrics(a,b))
    def matrix_evidence(mesh):
        matrix=mesh.mass
        return dict(mass=_array_digest((matrix.data,matrix.indices,matrix.indptr)),
                    equilibration=_array_digest(mesh._equilibration),
                    factor=_array_digest(tuple((m.data,m.indices,m.indptr) for m in (mesh._factor.L,mesh._factor.U))),
                    permutations=_array_digest((mesh._factor.perm_r,mesh._factor.perm_c)))
    try:
        for suffix,anchor in [('',4110),('_injInner',5680)]:
            root=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
            path=root/f'jorek{anchor+1:06d}.h5';params_path=root/'jorek_namelist_spitest'
            params=parse_namelist(params_path.read_text());row=dict(run=str(root),anchor=anchor,stages_seconds={})
            result['runs'].append(row)
            stage=f'{anchor}:read_initial'
            with h5py.File(path,'r') as f:
                scalar=lambda key:float(f[key][...].ravel()[0])
                if any(scalar(k)!=1 for k in ('n_plane','n_tor','n_period')): raise ValueError('H1 single-plane data required')
                units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
                geometry=dict(coordinates=f['x'][...],size=f['size'][...],vertex=f['vertex'][...]-1,indices=f['index'][...])
                event_time=scalar('t_now')*units.t_norm_si
                shards=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
                    radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],species=f['spi_species_arr'][...],
                    time_si=event_time,source_volume_m3=f['spi_vol_arr'][...],source_volume_time_si=event_time,
                    source_volume_provenance=str(path)+'::spi_vol_arr')
            if len(shards['radius'])!=846: raise ValueError('all 846 initial shards required')
            row.update(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       namelist_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),shards=846)
            build=lambda:NativeSourceMesh(**geometry,units=units,projection='sparse',max_elements=5000)
            stage=f'{anchor}:mesh';stamp=time.monotonic();mesh=build();check()
            row['stages_seconds']['mesh_build']=time.monotonic()-stamp
            stage=f'{anchor}:density_configuration'
            flag_value=params.get('spi_num_vol',True)
            if (not isinstance(flag_value,bool) and
                str(flag_value).lower() not in ('.t.','.true.','.f.','.false.')):
                raise ValueError('explicit boolean configuration required: spi_num_vol')
            numerical=(flag_value if isinstance(flag_value,bool)
                       else str(flag_value).lower() in ('.t.','.true.'))
            row['density_configuration_provenance']=dict(
                spi_num_vol=numerical,
                spi_num_vol_origin=('namelist' if 'spi_num_vol' in params else
                    'native_real_chain_builder explicit True default; preset_parameters'),
                ns_delta_minor_rad=0.)
            if params.get('ns_delta_minor_rad',0.)!=0:
                raise ValueError('unsupported target elongated source branch')
            kernel=NativeShardDensityKernel(mesh,ns_radius_min=float(params['ns_radius_min']),
                ns_radius_ratio=float(params['ns_radius_ratio']),ns_deltaphi=float(params['ns_deltaphi']),
                spi_num_vol=numerical,ns_delta_minor_rad=0.,
                injection_time_si=float(params['t_ns'])*units.t_norm_si,toroidal_angles=(0.,))
            stage=f'{anchor}:volume_refresh';stamp=time.monotonic()
            shards=kernel.refresh_volumes(shards,time_si=event_time,provenance='same_initial_refresh_as_build_real')
            row['stages_seconds']['volume_refresh']=time.monotonic()-stamp
            row['input_digest']=_array_digest(shards); row['projection']=matrix_evidence(mesh)
            points=[np.column_stack([g.coordinates,np.zeros(len(g.coordinates))]) for g in mesh.gauss]
            row['elements']=len(points); row['gauss_points']=sum(len(p) for p in points)
            stage=f'{anchor}:literal_scalar_rates';stamp=time.monotonic();literal=[]
            scale=units.t_norm_si/units.n_norm_m3*20/units.central_mass
            for e,xyz in enumerate(points):
                if e%32==0:check()
                values=np.zeros(len(xyz))
                if event_time>kernel.injection_time_si:
                    for j in range(len(shards['radius'])):
                        if shards['radius'][j]==0 or shards['ablation_atoms_s'][j]==0:continue
                        pos=shards['positions'][j]
                        width=max(shards['radius'][j]*kernel.ratio,kernel.minimum)
                        volume=pi*pos[0]*kernel.toroidal_integral*width**2
                        if kernel.numerical and shards['source_volume_m3'][j]>0:
                            volume=shards['source_volume_m3'][j]
                        pol=np.exp(-np.sum((xyz[:,:2]-pos[:2])**2,axis=1)/width**2)
                        angle=np.abs((kernel.angles-pos[2]+pi)%(2*pi)-pi)
                        tor=np.mean(np.exp(-(angle/kernel.deltaphi)**2))
                        values+=(shards['ablation_atoms_s'][j]*scale/volume)*pol*tor
                literal.append(values)
            literal=np.stack(literal);row['stages_seconds']['literal_scalar_rates']=time.monotonic()-stamp
            stage=f'{anchor}:prepared_rates';stamp=time.monotonic()
            prepared=kernel.prepare_rates(shards,time_si=event_time)
            optimized=np.stack([prepared.rates_at(x,toroidal_average=True) for x in points]);check()
            row['stages_seconds']['prepared_rates']=time.monotonic()-stamp
            row['rates']=evidence(literal,optimized)
            stage=f'{anchor}:same_factor_projection';stamp=time.monotonic()
            old_rhs=mesh.weak_load(literal[:,:,None]);new_rhs=mesh.weak_load(optimized[:,:,None])
            row['weak_load']=evidence(old_rhs,new_rhs)
            old_coeff=mesh.project(old_rhs);new_coeff=mesh.project(new_rhs)
            row['projected_coefficients']=evidence(old_coeff,new_coeff)
            actual=kernel.density_source(shards).values.numpy()
            expected=np.concatenate((np.zeros_like(old_coeff),old_coeff),axis=0)
            row['public_native_density_source']=evidence(expected,actual)
            row['stages_seconds']['same_factor_projection_and_public_source']=time.monotonic()-stamp
            row['input_unchanged']=row['input_digest']==_array_digest(shards)
            if args.wall_seconds-(time.monotonic()-started)>35:
                stage=f'{anchor}:fresh_projection_mesh';stamp=time.monotonic();fresh=build();check()
                row['fresh_mesh']=dict(status='completed',operator=matrix_evidence(fresh),
                    same_rhs_coefficients=evidence(old_coeff,fresh.project(old_rhs)),
                    fresh_weak_load=evidence(old_rhs,fresh.weak_load(literal[:,:,None])))
                row['stages_seconds']['fresh_mesh']=time.monotonic()-stamp
                del fresh
            else:row['fresh_mesh']=dict(status='skipped_remaining_budget_less_than_35_seconds')
            print(json.dumps(dict(anchor=anchor,rates_exact=row['rates']['exact'],
                coefficients_exact=row['projected_coefficients']['exact'],stage_seconds=row['stages_seconds'])),flush=True)
            del mesh,kernel,prepared
        result['success']=True
        result['all_density_stages_bitwise_equal']=all(all(r[k]['exact'] for k in
            ('rates','weak_load','projected_coefficients','public_native_density_source')) for r in result['runs'])
    except Exception as exc:
        result.update(failed_stage=stage,error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(success=result['success'],output=str(args.output),failed_stage=result.get('failed_stage'))),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
