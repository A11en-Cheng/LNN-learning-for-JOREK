"""Bounded deterministic shard residual probes; no birth, RNG or qualification."""
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rate_metrics(predicted, reference, dt):
    import numpy as np
    predicted = np.asarray(predicted); reference = np.asarray(reference)
    error = abs(predicted-reference); nonzero = reference != 0
    relative = error[nonzero]/abs(reference[nonzero])
    total = float(reference.sum())
    result = dict(shards=len(reference), nonzero_truth_count=int(nonzero.sum()),
        sum_pred_atoms_s=float(predicted.sum()), sum_truth_atoms_s=total,
        signed_total_rate_relative=float(predicted.sum()/total-1) if total else None,
        release_weighted_l1_relative=float(error.sum()/total) if total else None,
        max_relative_nonzero=float(relative.max()) if len(relative) else None,
        zero_truth_max_abs_atoms_s=float(error[~nonzero].max()) if (~nonzero).any() else None,
        absolute_error_quantiles_atoms_s=dict(zip(('p50','p90','p95','p99','max'),
            np.quantile(error,[.5,.9,.95,.99,1]).tolist())) if len(error) else {},
        relative_error_quantiles_nonzero=dict(zip(('p50','p90','p95','p99','max'),
            np.quantile(relative,[.5,.9,.95,.99,1]).tolist())) if len(relative) else {},
        reference_new_rate_integral_atoms=total*dt,
        predicted_new_rate_integral_atoms=float(predicted.sum())*dt,
        integral_scope='constant_NEW_rate_times_dt_not_actual_OLD_rate_release_this_step')
    for threshold,label in ((1e-8,'1e8'),(1e-6,'1e6'),(1e-3,'1e3'),(.01,'1pct')):
        count=int((relative>threshold).sum())
        result['affected_gt_'+label+'_relative_count']=count
        result['affected_gt_'+label+'_all_shard_fraction']=count/len(reference) if len(reference) else None
    return result


def shell_inputs(mesh, fields, feedback, equilibrium, *, mass_correction=True):
    """Keep evaluated U/P fixed while probing ONLY shell/table arithmetic."""
    import numpy as np
    rows=[];ratio=mesh.units.central_mass/20*((2/2.01455279245) if mass_correction else 1.)
    for ids,g in zip(mesh.vertex.T,mesh.gauss):
        b=g.test_basis[:,:,:,0]
        u=np.einsum('qvd,fvd->qf',b,fields[:,:,0,ids].transpose(0,2,1))
        p=np.einsum('qvd,fvd->qf',b,feedback[:,:,0,ids].transpose(0,2,1))
        density=np.where(p[:,4]<2e-4,1e-4+1e-4*np.exp(np.minimum((p[:,4]-2e-4)/1e-4,0)),p[:,4])
        charge=np.minimum(np.maximum(p[:,3],0)/density,10.)
        rows.append(np.column_stack((equilibrium.normalized_flux(u[:,0],g.coordinates[:,1]),
            u[:,4]+(ratio*charge-1)*u[:,7],u[:,8],g.weights*g.jacobian*g.coordinates[:,0]*2*np.pi)))
    return np.concatenate(rows)


def table_probe(cache, rows, mesh, *, rounded_axis=False, corrected_shell=False):
    """Diagnostic counterfactual, never silently installed as production policy."""
    import numpy as np
    from src.models.marker_feedback import EL_CHG,MU_ZERO
    nf=len(cache.shell_volume_m3);nt=len(cache.psi);eq=cache.equilibrium
    fraction=lambda n: (np.arange(n,dtype=np.float32)/np.float32(n)).astype(float) if rounded_axis else np.arange(n)/n
    coarse=eq.psi_axis+fraction(nf)*(eq.psi_bnd-eq.psi_axis)
    fine=eq.psi_axis+fraction(nt)*(eq.psi_bnd-eq.psi_axis)
    inside=(rows[:,0]>=0)&(rows[:,0]<1);data=rows[inside]
    index=np.floor(data[:,0]*nf).astype(int)
    if corrected_shell:
        index=np.searchsorted(eq.normalized_flux(coarse),data[:,0],side='right')-1
    volume=np.bincount(index,weights=data[:,3],minlength=nf)
    numbers=np.bincount(index,weights=data[:,1]*data[:,3],minlength=nf)
    energy=np.bincount(index,weights=data[:,1]*data[:,2]*data[:,3],minlength=nf)
    te=energy/numbers/(EL_CHG*MU_ZERO*mesh.units.n_norm_m3)
    ne=numbers/volume*mesh.units.n_norm_m3
    values=[]
    for psi in fine:
        i=int(np.argmin(abs(coarse-psi)))
        j=i+1 if abs(psi-eq.psi_axis)>=abs(coarse[i]-eq.psi_axis) else i-1
        if j>=nf:j=nf-2
        values.append([(a[i]-a[j])/(coarse[i]-coarse[j])*(psi-coarse[i])+a[i] for a in (te,ne)])
    values=np.asarray(values)
    return replace(cache,psi=fine,temperature_ev=values[:,0],electron_density_m3=values[:,1],shell_volume_m3=volume),index


def rank_es_probe(cache, rows, mesh, previous, n_cpu, fields, feedback):
    """Explicit candidate current-source MPI path, NOT a production claim.

    Rank0 uses current ES. Others use prior ES before allreduce; each rank
    subsequently interpolates the reduced shell sums on its own psi grid.
    Return rank0's table only (the exported shard arrays are rank0's state).
    """
    import numpy as np
    nf=len(cache.shell_volume_m3);eq=cache.equilibrium
    changed=rows.copy();chunk=(len(mesh.gauss)+n_cpu-1)//n_cpu;start=chunk*len(mesh.gauss[0].weights)
    # Shell rows hold private-corrected normalized flux. Re-evaluate literal
    # previous ES at actual full cubic psi/Z rather than undo private reflection.
    previous_rows=shell_inputs(mesh,fields,feedback,previous)
    changed[start:,0]=previous_rows[start:,0]
    # Double fractions in both ES states make floor equivalent here; separately
    # tested actual coarse-edge correction found no membership changes.
    result,index=table_probe(cache,changed,mesh)
    def membership(data):
        inside=(data[:,0]>=0)&(data[:,0]<1)
        index=np.full(len(data),-1);index[inside]=np.floor(data[inside,0]*nf).astype(int)
        return index
    return result,int(np.count_nonzero(membership(rows)!=membership(changed)))


def fortran_cache(cache, mesh, fields, feedback, previous, n_cpu):
    """Compile a small literal-source oracle, never compile or modify JOREK."""
    import numpy as np
    from src.models.marker_feedback import EL_CHG,MU_ZERO
    if any(e.xcase!='double' or e.lcfs_lost for e in (cache.equilibrium,previous)):
        raise ValueError('independent oracle explicitly supports target double-X/LCFS-false only')
    nq=sum(len(g.weights) for g in mesh.gauss)
    stream=io.StringIO()
    stream.write(f'{len(cache.shell_volume_m3)} {len(cache.psi)} {nq} {n_cpu} {len(mesh.gauss[0].weights)}\n')
    for eq in (cache.equilibrium,previous):
        stream.write(' '.join(map(str,(eq.psi_axis,eq.psi_bnd,*eq.psi_xpoints,*eq.z_xpoints)))+'\n')
    stream.write(str(mesh.units.central_mass/20*(2/2.01455279245))+'\n')
    minima=np.full(5,np.inf);negative=np.zeros(5,dtype=int);floor_changed=0;floor_visited=0
    for ids,g in zip(mesh.vertex.T,mesh.gauss):
        b=g.test_basis[:,:,:,0]
        u=np.einsum('qvd,fvd->qf',b,fields[:,:,0,ids].transpose(0,2,1))
        p=np.einsum('qvd,fvd->qf',b,feedback[:,:,0,ids].transpose(0,2,1))
        inside=(cache.equilibrium.normalized_flux(u[:,0],g.coordinates[:,1])>=0)&(cache.equilibrium.normalized_flux(u[:,0],g.coordinates[:,1])<1)
        raw=np.column_stack((u[:,4],u[:,7],u[:,8],p[:,3],p[:,4]))[inside]
        if len(raw):
            minima=np.minimum(minima,raw.min(axis=0));negative+=(raw<0).sum(axis=0)
        def zcharge(scale):
            density=scale*p[:,4]
            density=np.where(density<2e-4,1e-4+1e-4*np.exp(np.minimum((density-2e-4)/1e-4,0)),density)
            return np.minimum(np.maximum(scale*p[:,3],0)/density,10.)
        floor_changed+=int(np.count_nonzero((abs(zcharge(1)-zcharge(2))>1e-12)&inside))
        floor_visited+=int(np.count_nonzero((p[:,4]<2e-4)&inside))
        data=np.column_stack((u[:,0],g.coordinates[:,1],u[:,4],u[:,7],u[:,8],p[:,3],p[:,4],
            g.weights*g.jacobian*g.coordinates[:,0]*2*np.pi))
        np.savetxt(stream,data,fmt='%.17e')
    source=Path(__file__).resolve().parents[1]/'tests/native_nonlocal_residual_reference.f90'
    with tempfile.TemporaryDirectory(prefix='native-nonlocal-oracle-') as directory:
        exe=Path(directory)/'oracle'
        flags=['-O0','-fdefault-real-8','-fdefault-double-8','-ffree-line-length-none']
        subprocess.run(['gfortran',*flags,str(source),'-o',str(exe)],check=True,capture_output=True,timeout=10)
        result=subprocess.run([str(exe)],input=stream.getvalue(),capture_output=True,text=True,check=True,timeout=15)
    table=np.loadtxt(io.StringIO(result.stdout))
    return replace(cache,psi=table[:,0],temperature_ev=table[:,1]/(EL_CHG*MU_ZERO*mesh.units.n_norm_m3),
        electron_density_m3=table[:,2]*mesh.units.n_norm_m3),dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),compiler_flags=flags,
        rank_policy='candidate_rank0_current_other_ranks_prior_ES',n_cpu=n_cpu,
        aux_policy='literal_use_marker_and_present_aux_branches',gauss_rows=nq,
        raw_inside_gauss_min=dict(zip(('rho_JU','rhoimp_JU','Te_JU','P4_JU','P5_JU'),minima.tolist())),
        raw_inside_gauss_negative_count=dict(zip(('rho_JU','rhoimp_JU','Te_JU','P4_JU','P5_JU'),negative.tolist())),
        single_aux_inside_density_correction_count=floor_visited,
        doubled_aux_inside_charge_changed_above_1e_minus12_absolute=floor_changed,
        production_build_equivalence_claimed=False)


def diagnose(args):
    import h5py
    import numpy as np
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.data.namelist import parse_namelist
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_source_nonlocal import NativeNonlocalDiagnostic
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    from src.models.marker_feedback import EL_CHG,MU_ZERO
    from tools.audit_native_psi_shards import file_sha256
    if not (args.path.parent.resolve()==args.target.parent.resolve()==args.feedback_path.parent.resolve()):
        raise ValueError('same-run explicit causal records required')
    record=read_native_psi_teacher(args.path)
    params=parse_namelist(Path(record.namelist_path).read_text())
    def read_state(path):
        with h5py.File(path,'r') as f:
            scalar=lambda k:float(np.asarray(f[k][()]).ravel()[0])
            state=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
                velocities=np.column_stack([f[k][...] for k in ('spi_Vel_R_arr','spi_Vel_Z_arr','spi_Vel_RxZ_arr')]),
                radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],
                injector_phi=f['spi_phi_init_arr'][...],species=f['spi_species_arr'][...],
                ids=np.arange(int(scalar('n_spi_tot'))),positions_coordinate_system='RZphi',
                velocities_coordinate_system='injector_R_Z_RxZ')
            return state,f['values'][...],scalar('tstep')*scalar('t_norm'),scalar('F0'),f['spi_psi_arr'][...]
    state,u,_,F0,_=read_state(args.path);truth,_,dt,_,saved_psi=read_state(args.target)
    with h5py.File(args.feedback_path,'r') as f:
        feedback=f['values'][...];pt=float(np.asarray(f['t_now'][()]).ravel()[0])
    if pt>record.time_si:raise ValueError('future auxiliary feedback forbidden')
    print(json.dumps(dict(stage='loaded',path=str(args.path),target=str(args.target),
        feedback_path=str(args.feedback_path),field_sha256=record.source_sha256,
        feedback_sha256=file_sha256(args.feedback_path),target_sha256=file_sha256(args.target),
        old_time_si=record.time_si,feedback_time_si=pt,dt_si=dt,shards=len(state['radius']),
        future_record_usage='independent_comparison_only')),flush=True)
    mesh=NativeSourceMesh(record.coordinates,record.size,record.vertex,record.indices,
        units=record.units,projection='none',max_elements=5000)
    cache=NativeNonlocalDiagnostic.build(mesh,u,feedback,record.equilibrium,n_flux=int(params['n_flux']),
        n_table=int(params['n_nonlocal_array']),feedback_time_si=pt,provenance='explicit causal residual audit')
    state['nonlocal_diagnostic']=cache;state['old_equilibrium']=record.equilibrium
    dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=float(params['pellet_density'])*1e20,
        F0=F0,R_geo=float(params['r_geo']),magnetic_reduction=False,nonlocal_abl=True,
        ns_radius_min=float(params['ns_radius_min']),ns_radius_ratio=float(params['ns_radius_ratio']))
    output=dynamics.update(state,u,record.time_si,dt)
    again=dynamics.update(state,u,record.time_si,dt)
    repeat={key:bool(np.array_equal(output[key],again[key])) for key in ('positions','radius','velocities','ablation_atoms_s')}
    rows=shell_inputs(mesh,u,feedback,record.equilibrium)
    variants={'production_baseline':cache};shell_counts={};baseline_index=None
    for name,rounded,corrected in [('double_rebuild',False,False),('float32_axis_only',True,False),
                                   ('double_shell_correction_only',False,True),('float32_axis_and_shell',True,True)]:
        variants[name],index=table_probe(cache,rows,mesh,rounded_axis=rounded,corrected_shell=corrected)
        if baseline_index is None:baseline_index=index
        shell_counts[name]=int(np.count_nonzero(index!=baseline_index))
    prior_path=args.path.with_name(f'jorek{record.step_index-1:06d}.h5')
    prior=read_native_psi_teacher(prior_path)
    if prior.geometry_digest!=record.geometry_digest:
        raise ValueError('prior/current geometry mismatch in rank-cache diagnostic')
    previous=prior.equilibrium  # Keep true prior binding; do not relabel it current.
    for n_cpu in (10,):
        variants[f'candidate_rank0_current_others_prior_ES_ncpu{n_cpu}'],changed=rank_es_probe(cache,rows,mesh,previous,n_cpu,u,feedback)
        shell_counts[f'candidate_rank0_current_others_prior_ES_ncpu{n_cpu}']=changed
    uncorrected=shell_inputs(mesh,u,feedback,record.equilibrium,mass_correction=False)
    variants['candidate_mass_ratio_without_deuterium_correction'],_=table_probe(cache,uncorrected,mesh)
    variants['candidate_mass_ratio_without_correction_and_rank_ES'],changed=rank_es_probe(cache,uncorrected,mesh,previous,10,u,feedback)
    shell_counts['candidate_mass_ratio_without_correction_and_rank_ES']=changed
    doubled=shell_inputs(mesh,u,2*feedback,record.equilibrium)
    variants['candidate_both_marker_and_present_aux_branches'],_=table_probe(cache,doubled,mesh)
    variants['candidate_double_aux_and_rank_ES'],changed=rank_es_probe(cache,doubled,mesh,previous,10,u,2*feedback)
    shell_counts['candidate_double_aux_and_rank_ES']=changed
    oracle_evidence=None
    if args.fortran_oracle:
        variants['independent_fortran_literal_cache'],oracle_evidence=fortran_cache(cache,mesh,u,feedback,previous,10)
        shell_counts['independent_fortran_literal_cache']=shell_counts['candidate_double_aux_and_rank_ES']
    predictions={name:output['ablation_atoms_s'].copy() for name in variants}
    traces=[];lookup_changes={name:0 for name in variants}
    def lookup(table,psi):
        i=int(np.argmin(abs(table.psi-psi)))
        if abs(psi-record.equilibrium.psi_axis)<abs(table.psi[i]-record.equilibrium.psi_axis):i=max(0,i-1)
        return i
    for i,radius in enumerate(state['radius']):
        if radius<=0:continue
        elm,st=mesh.locate(output['positions'][i:i+1])
        if elm[0]<=0:continue
        v,_,_,_,_=mesh.point(int(elm[0])-1,st[0],u)
        te=max(v[8]/(EL_CHG*MU_ZERO*mesh.units.n_norm_m3),0.)
        ne=max(max(v[4]*mesh.units.n_norm_m3,0.)-max(v[7]*mesh.units.n_norm_m3,0.),0.)
        r=output['radius'][i];local=dynamics._ablation(r,te,ne)
        idx=lookup(cache,v[0]);nlrates={}
        for name,table in variants.items():
            tn,nn=table.sample(v[0],output['positions'][i,1],equilibrium=record.equilibrium,
                old_fields=u,time_si=record.time_si,mesh=mesh)
            nl=dynamics._ablation(r,tn,nn);nlrates[name]=nl
            predictions[name][i]=max(local,nl)
            lookup_changes[name]+=lookup(table,v[0])!=idx
        actual=float(truth['ablation_atoms_s'][i]);pred=float(output['ablation_atoms_s'][i])
        traces.append(dict(shard_id=i,fortran_shard_index_1based=i+1,element_1based=int(elm[0]),st=st[0].tolist(),
            psi=float(v[0]),saved_next_shard_psi=float(saved_psi[i]),psi_error=float(v[0]-saved_psi[i]),
            normalized_psi=float(record.equilibrium.normalized_flux(v[0],output['positions'][i,1])),
            table_index_0based=idx,te_local_ev=float(te),ne_local_m3=float(ne),
            te_nonlocal_ev=float(cache.temperature_ev[idx]),ne_nonlocal_m3=float(cache.electron_density_m3[idx]),
            local_rate_atoms_s=float(local),nonlocal_rates_atoms_s={k:float(vv) for k,vv in nlrates.items()},
            branch='nonlocal' if nlrates['production_baseline']>local else 'local',
            predicted_atoms_s=pred,truth_atoms_s=actual,abs_error_atoms_s=abs(pred-actual),
            variant_predicted_atoms_s={name:float(predictions[name][i]) for name in predictions},
            relative_error=abs(pred-actual)/abs(actual) if actual else None))
    branch_metrics={}
    for branch in ('local','nonlocal'):
        ids=[row['shard_id'] for row in traces if row['branch']==branch]
        branch_metrics[branch]=rate_metrics(output['ablation_atoms_s'][ids],truth['ablation_atoms_s'][ids],dt)
    errors=abs(output['ablation_atoms_s']-truth['ablation_atoms_s'])
    relative_order=sorted(traces,key=lambda row:row['relative_error'] or 0,reverse=True)
    abs_order=sorted(traces,key=lambda row:row['abs_error_atoms_s'],reverse=True)
    return dict(stage='residual',repeat_bitwise_equal=repeat,rng_used=False,birth_executed=False,
        independent_fortran_oracle=oracle_evidence,
        candidate_rank_evidence=dict(n_cpu=10,source='same-directory logfile_spitest:2969',
            limitation='logfile_spitest is later restart 9290+, not exact 5680 production-build certificate',
            prior_record_path=str(prior_path),prior_record_sha256=prior.source_sha256,
            prior_ES_time_si=prior.time_si,current_ES_time_si=record.time_si),
        metrics=rate_metrics(output['ablation_atoms_s'],truth['ablation_atoms_s'],dt),branch_metrics=branch_metrics,
        worst_relative_shards=relative_order[:10],largest_absolute_shards=abs_order[:10],
        top10_fraction_of_l1_error=float(np.sort(errors)[-10:].sum()/errors.sum()),
        saved_next_psi_comparison_max_abs=float(max(abs(row['psi_error']) for row in traces)),
        probes={name:dict(metrics=rate_metrics(pred,truth['ablation_atoms_s'],dt),
            worst_nonzero_shard_id=(int(np.flatnonzero(truth['ablation_atoms_s']!=0)[np.argmax(
                abs(pred[truth['ablation_atoms_s']!=0]-truth['ablation_atoms_s'][truth['ablation_atoms_s']!=0])/
                abs(truth['ablation_atoms_s'][truth['ablation_atoms_s']!=0]))])
                if np.any(truth['ablation_atoms_s']!=0) else None),
            shell_membership_changed=shell_counts.get(name,0),lookup_indices_changed=lookup_changes[name],
            max_table_te_change_ev=float(abs(variants[name].temperature_ev-cache.temperature_ev).max()),
            max_table_ne_change_m3=float(abs(variants[name].electron_density_m3-cache.electron_density_m3).max()))
            for name,pred in predictions.items()},trace=traces)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--feedback-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--memory-mib',type=int,default=1536)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--fortran-oracle',action='store_true')
    args=parser.parse_args()
    if not 0<args.seconds<=60 or not 0<args.memory_mib<=1536:
        raise ValueError('resource ceiling is 60s/1536MiB')
    if args.worker:
        print(json.dumps(diagnose(args)),flush=True); return
    if args.output.exists(): raise ValueError('do not overwrite existing evidence')
    started=time.monotonic();peak=0;stop=None
    with tempfile.TemporaryFile(mode='w+') as log:
        command=[sys.executable,__file__,'--worker','--path',str(args.path),'--target',str(args.target),
            '--feedback-path',str(args.feedback_path),'--output',str(args.output)]
        if args.fortran_oracle:command+=['--fortran-oracle']
        child=subprocess.Popen(command,stdout=log,stderr=log)
        while child.poll() is None:
            try:
                status=Path(f'/proc/{child.pid}/status').read_text()
                rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
                peak=max(peak,rss)
            except OSError: pass
            if peak>args.memory_mib*1024**2:stop='rss_limit'
            if time.monotonic()-started>args.seconds:stop='wall_limit'
            if stop:child.kill();break
            time.sleep(.02)
        child.wait(timeout=5);log.seek(0);lines=log.read().splitlines()
    evidence=dict(scope='deterministic_real_single_step_residual_no_birth_no_training',
        wall_limit_seconds=args.seconds,memory_limit_mib=args.memory_mib,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=peak,stop=stop,
        returncode=child.returncode,stages=[],errors=[],qualification_granted=False)
    for line in lines:
        try:evidence['stages'].append(json.loads(line))
        except ValueError:evidence['errors'].append(line)
    with args.output.open('x') as stream:json.dump(evidence,stream,indent=2)
    print(json.dumps({k:v for k,v in evidence.items() if k!='stages'}))


if __name__=='__main__':main()
