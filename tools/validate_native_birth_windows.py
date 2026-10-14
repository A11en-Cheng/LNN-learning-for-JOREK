"""Recorded physical source windows, common full cloud, conditional birth ensembles."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np

from tools.validate_native_birth_projection import (CHANNELS, digest, rng_digest,
    metrics, gauss_values, write_json)
from tools.validate_native_birth_entry import cloud_digest
from tools.diagnose_native_real_chain import Stages, _initial_pairing
from tools.native_real_chain_builder import build_real
from src.models.marker_state import MarkerRuntime, particle_window_schedule
from src.models.native_source_birth import NativeProductionBirths
from src.models.native_birth_positions import NativeTabulatedBirthSampler
from src.models.native_source_time import NativeSourceTimeSampler


def relative_error(reference, candidate):
    norm = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate-reference)/norm) if norm else None


def distribution(values):
    valid = [v for v in values if v is not None]
    return dict(undefined=len(values)-len(valid), **({
        'mean': float(np.mean(valid)), 'median': float(np.median(valid)),
        'p95': float(np.percentile(valid, 95)), 'max': float(np.max(valid))} if valid else {}))


def eion_diagnostics(a, b, old, whole, weights):
    norm = lambda x: float(np.sqrt(np.sum(weights*x*x)))
    integral = lambda x: float(x@weights)
    aa = float(np.sum(weights*a*a))
    scale = float(np.sum(weights*a*b)/aa) if aa else None
    total = integral(whole)
    return dict(newborn_norm_fraction=norm(a)/norm(whole) if norm(whole) else None,
        newborn_signed_integral_fraction=integral(a)/total if total else None,
        signed_denominator_flag='positive' if total > 0 else 'negative' if total < 0 else 'zero',
        old_background_norm=norm(old), old_background_integral=integral(old),
        best_scale=scale, shape_residual_relative=norm(b-scale*a)/norm(b) if scale is not None and norm(b) else None,
        amplitude_relative=abs(scale-1) if scale is not None else None,
        positive_norm=norm(np.maximum(a,0)), negative_norm=norm(np.minimum(a,0)),
        cancellation=1-abs(integral(a))/integral(abs(a)) if integral(abs(a)) else None)


def bootstrap_ratios(new, noise, seed):
    new, noise = np.asarray(new), np.asarray(noise)
    valid = np.isfinite(new) & np.isfinite(noise)
    new, noise = new[valid], noise[valid]
    if not len(new) or np.mean(noise) == 0:
        return dict(status='undefined_zero_or_missing_noise')
    ids = np.random.default_rng(seed).integers(0,len(new),(2000,len(new)))
    denominator = noise[ids].mean(1)
    ratios = new[ids].mean(1)[denominator != 0]/denominator[denominator != 0]
    return dict(status='descriptive_within_window_paired_seed_resampling',
        ratio_of_mean_errors=float(new.mean()/noise.mean()),
        percentile95=np.percentile(ratios,[2.5,97.5]).tolist(),resamples=2000)


def capture_advance(source, marker, target, previous, current, sampler, births=None, born_ids=None):
    deposit, project = source.mesh.deposit_batch, source.mesh.project
    captured = None
    whole = None
    def capture(cloud, ids, loads):
        nonlocal captured
        if born_ids is not None:
            selected = np.isin(ids,born_ids)
            if selected.any():
                rhs = deposit(cloud,ids[selected],loads[selected])
                captured = rhs if captured is None else captured+rhs
        return deposit(cloud,ids,loads)
    def save(rhs):
        nonlocal whole
        whole = rhs.copy()
        return project(rhs)
    source.mesh.deposit_batch, source.mesh.project = capture, save
    try:
        out, diagnostic = source.advance(marker,target,previous,current,births=births,time_sampler=sampler)
    finally:
        source.mesh.deposit_batch, source.mesh.project = deposit, project
    if whole is None:
        whole = np.zeros((5,source.mesh.n_dofs))
    newborn = np.zeros_like(whole) if captured is None else captured/diagnostic['gather_time_seconds']
    return out, whole, newborn, diagnostic


def compact_response(source, marker, newborn, target, previous, current, sampler):
    if getattr(marker,'transport_history',None) is not None or getattr(source,'transport_policy',None) is not None:
        raise ValueError('compact birth-only response is not registered for local transport history')
    compact = MarkerRuntime(deepcopy(newborn), marker.feedback, deepcopy(marker.clock), deepcopy(marker.provenance))
    out, rhs, _, diagnostic = capture_advance(source,compact,target,previous,current,sampler)
    return out, rhs, diagnostic


def load_record(run, index, units, registered):
    path = run/f'jorek{index:06d}.h5'
    if not path.exists():
        raise ValueError('missing physical source record: '+str(path))
    with h5py.File(path,'r') as f:
        scalar = lambda k: float(np.asarray(f[k][...]).ravel()[0])
        if scalar('index_now') != index or scalar('t_norm') != units.t_norm_si or scalar('central_density')*1e20 != units.n_norm_m3 or scalar('central_mass') != units.central_mass:
            raise ValueError('unknown/mismatched source index or units')
        now, dt = scalar('t_now')*units.t_norm_si, scalar('tstep')*units.t_norm_si
        if dt <= 0 or index in registered:
            raise ValueError('ordinary window crosses restart or nonpositive tstep')
        shards = dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
            radius=f['spi_radius_arr'][...], ablation_atoms_s=f['spi_abl_arr'][...],species=f['spi_species_arr'][...],
            source_volume_m3=f['spi_vol_arr'][...],source_volume_time_si=now,
            source_volume_provenance=str(path)+'::spi_vol_arr',time_si=now)
        return f['values'][...],f['deltas'][...],now,dt,shards


def fingerprint(marker, previous, current, raw, shards):
    return dict(cloud=cloud_digest(marker.cloud),feedback=digest(marker.feedback.values),
        feedback_history=digest(marker.feedback.deltas),clock=vars(marker.clock).copy(),
        previous=digest(previous),current=digest(current),rawD=digest(raw),
        shards={k:digest(v) for k,v in shards.items() if isinstance(v,np.ndarray)})


def spotcheck_saved(args):
    """Supplement existing first-window evidence with direct per-channel errors."""
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    summary=json.loads(args.spotcheck_saved.read_text())
    run=Path(summary['run'])
    options=SimpleNamespace(chunk_size=2048,wall_seconds=120,memory_mib=8192,
        birth_max_proposals=1000000,spatial_backend='batched_v1',
        birth_cells_per_width=64,birth_max_cells=16000000)
    stages=Stages()
    _,source,shard,state,_=build_real(run,summary['anchor'],
        _initial_pairing(run,summary['anchor'],stages),stages,options,feedback_mode='autonomous')
    index=summary['anchor']+1
    with h5py.File(run/f'jorek{index:06d}.h5','r') as f:
        shards=dict(state.shards,source_volume_m3=f['spi_vol_arr'][...],
            source_volume_time_si=state.fluid_time_si,
            source_volume_provenance=str(run/f'jorek{index:06d}.h5')+'::spi_vol_arr')
    initial=fingerprint(state.marker,state.fields_previous,state.fields_current,state.fluid_deltas,shards)
    if initial!=summary['windows'][0]['initial']:
        raise ValueError('supplemental direct check initial context differs')
    sampler=NativeSourceTimeSampler(state.fields_current,state.fluid_deltas,
        time_prev_si=state.marker.clock.field_start_si,time_now_si=state.marker.clock.field_end_si,
        tstep_si=state.fluid_dt_previous_si,static=source.field_static)
    np.random.seed(2917+2000003)
    born=shard.birth.births(shards,state.marker,state.fluid_time_si,
        state.fields_previous,state.fields_current,source,time_sampler=sampler)
    out,rhs,_,diagnostic=capture_advance(source,state.marker,state.fluid_time_si,
        state.fields_previous,state.fields_current,sampler,born)
    with np.load(args.spotcheck_saved.parent/f'j{index}'/'new64_2917.npz') as f:
        candidate=f['whole_coeff']
    direct_values,weights=gauss_values(source.mesh,out.feedback.values)
    candidate_values,_=gauss_values(source.mesh,candidate)
    result=dict(index=index,seed=2917,arm='new64',slots=len(out.cloud.weight),
        initial_matches=True,execution=diagnostic['execution'],
        channels=[dict(channel=name,coefficient_relative_error=relative_error(a,b),
            gauss_relative_error=relative_error(a*np.sqrt(weights),b*np.sqrt(weights)))
            for name,a,b in zip(CHANNELS,direct_values,candidate_values)])
    # Coefficient errors use actual coefficients rather than Gauss samples.
    for row,a,b in zip(result['channels'],out.feedback.values,candidate):
        row['coefficient_relative_error']=relative_error(a,b)
    if any(r['gauss_relative_error'] is None or r['gauss_relative_error']>1e-10 for r in result['channels']):
        raise ValueError('supplemental per-channel decomposition check failed')
    result['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.spotcheck_saved.parent/'direct-channel-spotcheck.json',result)
    print(json.dumps(result),flush=True)


def execute(args):
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    run = root/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.anchor==5680 else ''))
    registered = json.loads((root/'data/restart_events.json').read_text())['runs'][run.name]['restart_events']
    options = SimpleNamespace(chunk_size=2048,wall_seconds=120,memory_mib=8192,birth_max_proposals=1000000,
        spatial_backend='batched_v1',birth_cells_per_width=64,birth_max_cells=16000000)
    stages = Stages()
    _,source,shard,state,_ = build_real(run,args.anchor,_initial_pairing(run,args.anchor,stages),stages,options,feedback_mode='autonomous')
    settings = shard.birth.configuration
    kwargs = {k:settings[k] for k in ('physical_particles','n_particle_slots','n_cpu','rank','R_geo','n_plane','max_proposals')}
    reference = NativeProductionBirths(shard.density,**kwargs)
    new = NativeProductionBirths(shard.density,**kwargs,position_sampler=NativeTabulatedBirthSampler(
        geometry='axisymmetric_n0',n_tor=1,cells_per_width=64,max_cells=16000000))
    marker, previous, current, raw = state.marker,state.fields_previous,state.fields_current,state.fluid_deltas
    result = dict(anchor=args.anchor,run=str(run),physical_windows=args.windows,seeds_per_window=args.seeds,
        seedbase=2917,cells_per_width=64,no_training=True,field_advances=0,windows=[],
        command=sys.argv,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),root/'tools/native_real_chain_builder.py',*sorted((root/'src/models').glob('native*.py')),root/'src/models/marker_state.py']})
    write_json(args.output_dir/'snapshot.json',result)
    for window in range(args.windows):
        started = time.monotonic()
        index = args.anchor+1+window
        if window == 0:
            target,dt = state.fluid_time_si,state.fluid_dt_previous_si
            # Builder recomputes diagnostics; restore the recorded source-volume cache.
            with h5py.File(run/f'jorek{index:06d}.h5','r') as f:
                shards = dict(state.shards,source_volume_m3=f['spi_vol_arr'][...],
                    source_volume_time_si=target,source_volume_provenance=str(run/f'jorek{index:06d}.h5')+'::spi_vol_arr')
        else:
            previous = current
            current,raw,target,dt,shards = load_record(run,index,source.mesh.units,registered)
            if not np.isclose(target-marker.clock.sim_time_si,dt,rtol=1e-8,atol=0):
                raise ValueError('physical source step gap: recorded t_now difference != tstep')
            marker.clock.field_start_si = marker.clock.sim_time_si
            marker.clock.field_end_si = target
        sampler = NativeSourceTimeSampler(current,raw,time_prev_si=marker.clock.field_start_si,
            time_now_si=marker.clock.field_end_si,tstep_si=dt,static=source.field_static)
        initial = fingerprint(marker,previous,current,raw,shards)
        start,steps,gather = particle_window_schedule(marker.clock,target)
        folder = args.output_dir/f'j{index}'
        folder.mkdir()
        record = dict(index=index,window_index=window,initial=initial,tstep_si=dt,target_si=target,
            particle_start_si=start,particle_steps=steps,gather_si=gather,field_static=source.field_static,
            source_volume_provenance=shards['source_volume_provenance'],cases=[],comparisons=[],ensemble_means=[],statistics=[],eion=[])
        print(json.dumps(dict(stage='window_start',anchor=args.anchor,index=index,steps=steps)),flush=True)
        np.random.seed(2917)
        common = reference.births(shards,marker,target,previous,current,source,time_sampler=sampler)
        born_ids = np.flatnonzero(~marker.cloud.alive)[:len(common.weight)]
        common_out,common_rhs,common_birth_rhs,diagnostic = capture_advance(source,marker,target,previous,current,sampler,common,born_ids)
        existing_rhs = common_rhs-common_birth_rhs
        old_values,weights = gauss_values(source.mesh,source.mesh.project(existing_rhs))
        record['common'] = dict(birth_count=len(common.weight),slots_before=len(marker.cloud.weight),
            slots_after=len(common_out.cloud.weight),before=cloud_digest(marker.cloud),after=cloud_digest(common_out.cloud),
            feedback=digest(common_out.feedback.values),feedback_history=digest(common_out.feedback.deltas),clock=vars(common_out.clock).copy(),
            execution=diagnostic['execution'],cumulative_eion_change_j=float(diagnostic['energy'].sum()))
        arrays = {}
        for seed in range(2917,2917+args.seeds):
            for arm,birth,offset in [('refA',reference,0),('refB',reference,1000003),('new64',new,2000003)]:
                actual_seed = seed+offset
                np.random.seed(actual_seed)
                before_rng = rng_digest()
                newborn = birth.births(shards,marker,target,previous,current,source,time_sampler=sampler)
                birth_rng = rng_digest()
                compact_out,rhs,small_diagnostic = compact_response(source,marker,newborn,target,previous,current,sampler)
                coefficients = source.mesh.project(rhs)
                whole_coefficients = source.mesh.project(existing_rhs+rhs)
                values,_ = gauss_values(source.mesh,coefficients)
                whole_values,_ = gauss_values(source.mesh,whole_coefficients)
                arrays[(seed,arm)] = (values,whole_values)
                case = dict(seed=seed,arm=arm,actual_seed=actual_seed,rng_before=before_rng,rng_after_birth=birth_rng,
                    rng_after_window=rng_digest(),birth_count=len(newborn.weight),birth_digest=cloud_digest(newborn),
                    final_newborn_digest=cloud_digest(compact_out.cloud),cumulative_eion_change_j=float(small_diagnostic['energy'].sum()))
                if window==0 and seed==2917 and arm=='new64':
                    direct,direct_rhs,_,direct_diagnostic = capture_advance(source,marker,target,previous,current,sampler,newborn)
                    error = relative_error(direct_rhs,existing_rhs+rhs)
                    projected_error = relative_error(direct.feedback.values,whole_coefficients)
                    record['direct_spotcheck'] = dict(arm=arm,seed=seed,slots=len(direct.cloud.weight),
                        rhs_relative_error=error,projection_relative_error=projected_error,execution=direct_diagnostic['execution'])
                    if error is None or projected_error is None or max(error,projected_error)>1e-10:
                        raise ValueError('direct conditional decomposition equivalence failed')
                    del direct
                record['cases'].append(case)
                np.savez_compressed(folder/f'{arm}_{seed}.npz',newborn_rhs=rhs,newborn_coeff=coefficients,whole_coeff=whole_coefficients)
            for arm in ('refB','new64'):
                for scope,position in [('newborn',0),('whole',1)]:
                    a,b = arrays[(seed,'refA')][position],arrays[(seed,arm)][position]
                    record['comparisons'].append(dict(seed=seed,arm=arm,scope=scope,channels=metrics(a,b,a@weights,b@weights,weights)))
                a,b = arrays[(seed,'refA')][0][0],arrays[(seed,arm)][0][0]
                record['eion'].append(dict(seed=seed,arm=arm,**eion_diagnostics(a,b,old_values[0],arrays[(seed,'refA')][1][0],weights)))
        for scope,position in [('newborn',0),('whole',1)]:
            for arm in ('refB','new64'):
                a,b = [np.mean([arrays[(seed,mode)][position] for seed in range(2917,2917+args.seeds)],axis=0) for mode in ('refA',arm)]
                record['ensemble_means'].append(dict(arm=arm,scope=scope,channels=metrics(a,b,a@weights,b@weights,weights)))
            for channel_index,channel in enumerate(CHANNELS):
                rows = {arm:[c['channels'][channel_index] for c in record['comparisons'] if c['scope']==scope and c['arm']==arm] for arm in ('refB','new64')}
                for arm in rows:
                    record['statistics'].append(dict(scope=scope,arm=arm,channel=channel,
                        **{key:distribution([row[key] for row in rows[arm]]) for key in ('relative_l2','pearson','integral_relative_error','integral_absolute_error')}))
                record.setdefault('bootstrap',[]).append(dict(scope=scope,channel=channel,
                    **bootstrap_ratios([r['relative_l2'] if r['relative_l2'] is not None else np.nan for r in rows['new64']],
                        [r['relative_l2'] if r['relative_l2'] is not None else np.nan for r in rows['refB']],2917+window)))
        if fingerprint(marker,previous,current,raw,shards)!=initial:
            raise ValueError('conditional initial context mutated')
        marker = common_out
        record['elapsed_seconds'] = time.monotonic()-started
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(folder/'summary.json',record)
        result['windows'].append(record)
        print(json.dumps(dict(stage='window_complete',anchor=args.anchor,index=index,elapsed=record['elapsed_seconds'],births=record['common']['birth_count'])),flush=True)
    write_json(args.output_dir/'summary.json',result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor',type=int,choices=(4110,5680))
    parser.add_argument('--spotcheck-saved',type=Path)
    parser.add_argument('--windows',type=int,default=4)
    parser.add_argument('--seeds',type=int,default=8)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.spotcheck_saved is not None:
        saved=deepcopy(np.random.get_state())
        try:spotcheck_saved(args)
        finally:np.random.set_state(saved)
        return
    if args.anchor is None or args.output_dir is None:
        parser.error('--anchor and --output-dir required for execution')
    if not 1<=args.windows<=4 or not 2<=args.seeds<=8:
        parser.error('windows 1..4 and seeds 2..8 required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    saved = deepcopy(np.random.get_state())
    try:
        execute(args)
    except BaseException as exc:
        write_json(args.output_dir/'failure.json',dict(type=type(exc).__name__,reason=str(exc),command=sys.argv))
        raise
    finally:
        np.random.set_state(saved)


if __name__=='__main__':
    main()
