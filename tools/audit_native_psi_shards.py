"""Bounded raw-record psi-teacher/shard/birth audit, NOT autonomous rollout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import hashlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def emit(stage, **values):
    print(json.dumps(dict(stage=stage,**values)),flush=True)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def worker(args):
    import h5py
    import numpy as np
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.data.namelist import parse_namelist
    from src.models.native_coupling import NativeSourceMesh,NativeMarkerSource,NativeNeonAtomicRates
    from src.models.native_source_nonlocal import NativeNonlocalDiagnostic
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    from src.models.native_source_density import NativeShardDensityKernel
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_source_time import NativeSourceTimeSampler
    from src.models.source_module import load_neon_tables
    from src.models.marker_state import MarkerRuntime,MarkerClock,ProjectionFeedback
    from src.data.marker_checkpoint import read_marker_snapshot
    path=args.path;target=args.target
    if path.parent.resolve()!=target.parent.resolve():raise ValueError('same-run comparison required')
    record=read_native_psi_teacher(path)
    params=parse_namelist(Path(record.namelist_path).read_text())
    def load(p):
        with h5py.File(p,'r') as f:
            scalar=lambda k:float(f[k][()].ravel()[0])
            s=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
                velocities=np.column_stack([f[k][...] for k in ('spi_Vel_R_arr','spi_Vel_Z_arr','spi_Vel_RxZ_arr')]),
                radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],
                injector_phi=f['spi_phi_init_arr'][...],species=f['spi_species_arr'][...],
                ids=np.arange(int(scalar('n_spi_tot'))),positions_coordinate_system='RZphi',
                velocities_coordinate_system='injector_R_Z_RxZ',time_si=scalar('t_now')*scalar('t_norm'),
                source_volume_m3=f['spi_vol_arr'][...],source_volume_time_si=scalar('t_now')*scalar('t_norm'),
                source_volume_provenance=str(p)+' saved spi_vol_arr')
            return s,f['values'][...],f['deltas'][...],scalar('tstep')*scalar('t_norm'),scalar('F0')
    s,u,raw,imported_fluid_dt,F0=load(path);truth,_,_,control_dt,_=load(target)
    nominal_particle_dt=float(params['tstep_particles'])
    imported_particle_dt=imported_fluid_dt/max(1,int(imported_fluid_dt/nominal_particle_dt))
    emit('records',path=str(path),target=str(target),shards=len(s['radius']),
        source_sha256=record.source_sha256,field_digest=record.field_digest,
        source_time_si=s['time_si'],target_time_si=truth['time_si'],
        timestamp_dt_si=truth['time_si']-s['time_si'],target_control_dt_si=control_dt,
        event_classification=('positive_time_step' if truth['time_si']-s['time_si']>1e-12 else
                              'restart_equal_time_edge_not_continuous_event'),
        initial_fields='all9_real_single_step_only',future_record='comparison_only')
    emit('particle_clock_evidence',nominal_particle_dt_si=nominal_particle_dt,
        imported_fluid_dt_si=imported_fluid_dt,imported_restart_particle_dt_si=imported_particle_dt,
        algorithm='imported_fluid_dt/max(1,int(imported_fluid_dt/nominal_particle_dt))',
        actual_probe_dt_si=5e-9,scope='probe_does_not_replay_imported_clock_or_startup_event')
    mesh=NativeSourceMesh(record.coordinates,record.size,record.vertex,record.indices,
        units=record.units,projection='none',max_elements=5000)
    if mesh.configuration['geometry']!=record.geometry_digest:raise ValueError('geometry digest mismatch')
    # Complete position/old-rate erosion diagnostics survive later nonlocal failure.
    dt=control_dt;delta=s['positions'][:,2]-s['injector_phi']
    k=s['positions'].copy();k[:,0]+=(s['velocities'][:,0]*np.cos(delta)+s['velocities'][:,2]*np.sin(delta))*dt
    k[:,1]+=s['velocities'][:,1]*dt
    k[:,2]=np.mod(k[:,2]+(s['velocities'][:,2]*np.cos(delta)-s['velocities'][:,0]*np.sin(delta))/s['positions'][:,0]*dt,2*np.pi)
    radius=s['radius'].copy();alive=radius>0
    radius[alive]=np.maximum(0,radius[alive]-dt*s['ablation_atoms_s'][alive]/(4*np.pi*radius[alive]**2*float(params['pellet_density'])*1e20))
    emit('all_shard_kinematics',shards=len(radius),changed_positions=int(np.any(k!=s['positions'],axis=1).sum()),
        changed_radius=int((radius!=s['radius']).sum()),
        position_max_abs_by_coordinate={name:float(abs(k[:,j]-truth['positions'][:,j]).max())
            for j,name in enumerate(('R_m','Z_m','phi_rad'))},
        radius_max_abs_error=float(abs(radius-truth['radius']).max()))
    feedback_path=args.feedback_path or path.with_name(f'projections{record.step_index:05d}.h5')
    if args.feedback_path is None and not feedback_path.exists():
        feedback_path=path.with_name(f'projections{record.step_index:06d}.h5')
    try:
        if feedback_path.parent.resolve()!=path.parent.resolve():raise ValueError('same-run feedback required')
        with h5py.File(feedback_path,'r') as f:p=f['values'][...];pt=float(f['t_now'][()].ravel()[0])
        feedback_sha256=file_sha256(feedback_path)
        registration=None
        if args.feedback_policy=='registered-startup-zero':
            # Read the existing independently audited registration; never infer
            # restart status from a zero array or the selected filename alone.
            anchors=Path(__file__).resolve().parents[1]/'output/native-closure-certification-2026-10-08/anchors.json'
            matching=[a for a in json.loads(anchors.read_text())
                if a['run']==path.parent.name and a['index']+1==record.step_index
                and a['boundary']=='registered_restart_before_startup'
                and a['state']['clock']['status']=='verified_reconstructed'
                and a['state']['clock']['value']['startup_pending'] is True]
            if len(matching)!=1:raise ValueError('registered poststartup field index required for startup-zero policy')
            anchor=matching[0]
            sources=[q for q in anchor['projections'] if q['index']==anchor['index']
                and Path(q['source']['path']).resolve()==feedback_path.resolve()
                and q['source']['sha256']==feedback_sha256 and q['absmax']==0.]
            if len(sources)!=1 or np.count_nonzero(p):
                raise ValueError('registered original zero projection file/hash required; no fabricated zero')
            registration=dict(path=str(anchors),sha256=file_sha256(anchors),restart_index=anchor['index'],
                field_index=record.step_index,phase='poststartup_int3d_before_next_projection',
                actual_zero_file=True)
        emit('feedback_selection',policy=args.feedback_policy,explicit_path=args.feedback_path is not None,
            path=str(feedback_path),source_sha256=feedback_sha256,absmax=float(abs(p).max()),registration=registration)
        emit('nonlocal_inputs',feedback_path=str(feedback_path),feedback_time_si=pt,
            feedback_minus_field_seconds=pt-record.time_si,n_flux=int(params['n_flux']),n_table=int(params['n_nonlocal_array']))
        table=NativeNonlocalDiagnostic.build(mesh,u,p,record.equilibrium,n_flux=int(params['n_flux']),
            n_table=int(params['n_nonlocal_array']),feedback_time_si=pt,
            provenance=f'raw-record psi teacher cache reconstruction; aux_policy={args.feedback_policy}; '
                       f'aux_path={feedback_path}; aux_sha256={feedback_sha256}; registration={registration}')
        s['nonlocal_diagnostic']=table;s['old_equilibrium']=record.equilibrium
        emit('nonlocal_ready',minimum_shell_volume_m3=float(table.shell_volume_m3.min()))
        dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=float(params['pellet_density'])*1e20,
            F0=F0,R_geo=float(params['r_geo']),magnetic_reduction=False,nonlocal_abl=True,
            ns_radius_min=float(params['ns_radius_min']),ns_radius_ratio=float(params['ns_radius_ratio']))
        out=dynamics.update(s,u,record.time_si,dt)
        metrics={'positions':{name:dict(max_abs=float(abs(out['positions'][:,j]-truth['positions'][:,j]).max()),
            rms=float(np.sqrt(np.mean((out['positions'][:,j]-truth['positions'][:,j])**2))),
            unit='rad' if j==2 else 'm') for j,name in enumerate(('R','Z','phi'))}}
        for name in ('radius','velocities','ablation_atoms_s'):
            error=np.abs(out[name]-truth[name])
            nonzero=truth[name]!=0
            metrics[name]=dict(max_abs=float(error.max()),rms=float(np.sqrt(np.mean(error**2))),
                unit={'radius':'m','velocities':'m/s_injector_R_Z_RxZ','ablation_atoms_s':'atoms/s'}[name],
                max_relative_nonzero=float((error[nonzero]/abs(truth[name][nonzero])).max()) if nonzero.any() else None,
                relative_coverage=int(nonzero.sum()),zero_count=int((~nonzero).sum()),
                zero_max_abs=float(error[~nonzero].max()) if (~nonzero).any() else None,
                changed_entries=int(np.count_nonzero(out[name]!=s[name])))
        emit('all_shard_production_update',shards=len(out['radius']),metrics=metrics,qualification_granted=False)
    except Exception as e:
        emit('shard_failure',exception=type(e).__name__,message=str(e))
    if args.shards_only:return
    # Independent real birth window; no existing particle removed or advanced.
    try:
        kernel=NativeShardDensityKernel(mesh,ns_radius_min=float(params['ns_radius_min']),
            ns_radius_ratio=float(params['ns_radius_ratio']),ns_deltaphi=float(params['ns_deltaphi']),
            spi_num_vol=True,ns_delta_minor_rad=0.,injection_time_si=float(params['t_ns'])*record.units.t_norm_si,
            toroidal_angles=(0.,))
        particle_path=args.particle_path or path.with_name(f'part{record.step_index:05d}.h5')
        cloud,particle_time,_=read_marker_snapshot(particle_path)
        if particle_path.parent.resolve()!=path.parent.resolve() or particle_time>record.time_si:
            raise ValueError('birth probe requires same-run nonfuture particle snapshot')
        emit('birth_cloud_loaded',slots=len(cloud.weight),live_before=int(cloud.alive.sum()),
            particle_path=str(particle_path),particle_time_si=particle_time,
            source_volume='all846_saved_spi_vol_no_refresh',window_seconds=5e-9,max_proposals=args.max_proposals,
            scope='single_5ns_birth_mechanism_probe_not_imported_base_or_gather_replay',
            configured_particle_dt_si=float(params['tstep_particles']),field_static=True)
        marker=MarkerRuntime(cloud,ProjectionFeedback(np.zeros((5,4,1,mesh.n_nodes)),
            np.zeros((5,4,1,mesh.n_nodes)),particle_time),
            MarkerClock(particle_time,0.,5e-9,particle_time,record.time_si),{'scope':'birth_only_feedback_not_consumed'})
        tables=load_neon_tables()
        rates=NativeNeonAtomicRates(tables)
        source=NativeMarkerSource(mesh,atomic=rates,atomic_version=rates.version,
            potentials_ev=tables.ionization_potentials_ev,F0=F0,diffusion_ju=0.,
            time_sampling='jorek_raw_deltas_v1',field_static=True)
        sampler=NativeSourceTimeSampler(u,raw,time_prev_si=particle_time,time_now_si=record.time_si,
            tstep_si=record.time_si-particle_time,static=True)
        births=NativeProductionBirths(kernel,physical_particles=float(params['spi_quantity']),
            n_particle_slots=len(cloud.weight),n_cpu=1,rank=0,R_geo=float(params['r_geo']),n_plane=1,
            max_proposals=args.max_proposals)
        atoms=float(s['ablation_atoms_s'].sum())*5e-9
        truncated_count=int(atoms/births.weight)
        emit('birth_sampling_started',rate_atoms_s=float(s['ablation_atoms_s'].sum()),
            truncated_nominal_count=truncated_count,threshold_nominal_count=max(truncated_count,1) if atoms>1e14 else truncated_count,
            R_geo=float(params['r_geo']),
            package_target_weight=births.weight,rng='MT19937_distribution_not_PCG32_production_replay')
        np.random.seed(20261009)
        new=births.births(s,marker,particle_time+5e-9,u,u,source,time_sampler=sampler)
        merged=cloud.copy() if hasattr(cloud,'copy') else __import__('copy').deepcopy(cloud)
        merged.insert(new)
        emit('birth_sampling_completed',new_count=len(new.weight),retained_slots=len(merged.weight),
            live_before=int(cloud.alive.sum()),live_after=int(merged.alive.sum()),
            weights=new.weight.tolist(),positions=new.x.tolist(),elements=new.i_elm.tolist(),
            finite_velocities=bool(np.isfinite(new.V_prev).all()),neutral=bool((new.P_imp[:,0]==1).all()))
    except Exception as e:emit('birth_failure',exception=type(e).__name__,message=str(e))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--particle-path',type=Path)
    parser.add_argument('--feedback-path',type=Path)
    parser.add_argument('--feedback-policy',choices=('recorded','registered-startup-zero'),default='recorded')
    parser.add_argument('--shards-only',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seconds',type=float,default=60.);parser.add_argument('--memory-mib',type=int,default=1536)
    parser.add_argument('--max-proposals',type=int,default=2000)
    args=parser.parse_args()
    if not 0<args.seconds<=60 or not 1<=args.memory_mib<=1536 or not 0<args.max_proposals<=10000:
        raise ValueError('explicit finite audit resource limits required')
    if args.worker:return worker(args)
    if args.output is None or args.output.exists():raise ValueError('new evidence output required')
    started=time.monotonic();peak=0;stop=None
    with tempfile.TemporaryFile(mode='w+') as log:
        command=[sys.executable,__file__,'--worker','--path',str(args.path),'--target',str(args.target),
            '--max-proposals',str(args.max_proposals)]
        if args.particle_path is not None:command+=['--particle-path',str(args.particle_path)]
        if args.feedback_path is not None:command+=['--feedback-path',str(args.feedback_path)]
        command+=['--feedback-policy',args.feedback_policy]
        if args.shards_only:command+=['--shards-only']
        child=subprocess.Popen(command,stdout=log,stderr=log,env=os.environ.copy())
        while child.poll() is None:
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0)
                peak=max(peak,rss)
            except (OSError,StopIteration):pass
            if peak>args.memory_mib*1024**2:stop='rss_limit'
            if time.monotonic()-started>args.seconds:stop='wall_limit'
            if stop:child.kill();break
            time.sleep(.02)
        child.wait(timeout=5);log.seek(0);lines=log.read().splitlines()
    stages=[];errors=[]
    for line in lines:
        try:stages.append(json.loads(line))
        except ValueError:errors.append(line)
    result=dict(scope='raw_record_psi_teacher_single_step_not_autonomous',training=False,optimizer_updates=0,
        wall_limit_seconds=args.seconds,memory_limit_mib=args.memory_mib,max_proposals=args.max_proposals,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=peak,stop=stop,returncode=child.returncode,
        stages=stages,errors=errors,qualification_granted=False)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
