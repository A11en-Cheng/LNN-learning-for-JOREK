"""Bounded real-cloud birth A/B, fixed fields, no network advance or training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np

from tools.validate_native_birth_entry import cloud_digest
from tools.diagnose_native_real_chain import Stages, _initial_pairing
from tools.native_real_chain_builder import build_real
from src.models.native_source_birth import NativeProductionBirths
from src.models.native_birth_positions import NativeTabulatedBirthSampler
from src.models.native_source_time import NativeSourceTimeSampler
from src.models.marker_state import particle_window_schedule


CHANNELS = ('ionization_energy', 'radiation', 'charge_squared', 'charge', 'number')


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def rng_digest():
    state = np.random.get_state()
    return hashlib.sha256(state[0].encode()+state[1].tobytes()+repr(state[2:]).encode()).hexdigest()


def metrics(reference, candidate, reference_integral, candidate_integral, weights=None):
    rows = []
    for name, a, b, ai, bi in zip(CHANNELS, reference, candidate, reference_integral, candidate_integral):
        a, b = a.ravel(), b.ravel()
        w = np.ones_like(a) if weights is None else np.asarray(weights)
        norm = float(np.sqrt(np.sum(w*a*a)))
        centered_a, centered_b = a-np.average(a,weights=w), b-np.average(b,weights=w)
        denominator = float(np.sqrt(np.sum(w*centered_a**2)*np.sum(w*centered_b**2)))
        rows.append(dict(channel=name, reference_norm=norm,
            relative_l2=float(np.sqrt(np.sum(w*(b-a)**2))/norm) if norm else None,
            relative_l2_status='defined' if norm else 'undefined_zero_reference',
            pearson=float(np.sum(w*centered_a*centered_b)/denominator) if denominator else None,
            pearson_status='defined' if denominator else 'undefined_constant_channel',
            reference_integral=float(ai), candidate_integral=float(bi),
            integral_absolute_error=float(abs(bi-ai)),
            integral_relative_error=float(abs((bi-ai)/ai)) if ai else None,
            integral_status='defined' if ai else 'undefined_zero_reference'))
    return rows


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def compact_metrics(result):
    """Descriptive means/ranges only; two seeds are not a statistical certificate."""
    groups = {}
    for comparison in result['comparisons']:
        key = (comparison['other'],comparison['scope'],comparison['representation'])
        groups.setdefault(key,[]).append(comparison)
    rows = []
    for (other,scope,representation), comparisons in groups.items():
        for index,channel in enumerate(CHANNELS):
            row = dict(other=other,scope=scope,representation=representation,channel=channel,
                seeds=[c['seed'] for c in comparisons])
            for metric in ('relative_l2','pearson','integral_relative_error','integral_absolute_error'):
                values = [c['channels'][index][metric] for c in comparisons]
                finite = [v for v in values if v is not None]
                row[metric] = dict(mean=float(np.mean(finite)),minimum=min(finite),maximum=max(finite)) if finite else None
                row[metric+'_undefined_count'] = len(values)-len(finite)
            rows.append(row)
    return dict(anchor=result['anchor'],exact_replay=result['exact_replay'],
        interpretation='descriptive finite Monte Carlo comparisons; no significance or equivalence certification',metrics=rows)


def gauss_values(mesh, coefficients):
    values, weights = [], []
    for ids,g in zip(mesh.vertex.T,mesh.gauss):
        local = coefficients[:,:,0,ids].transpose(0,2,1)
        values.append(np.einsum('qvd,fvd->qf',g.test_basis[:,:,:,0],local).T)
        weights.append(g.weights*g.jacobian*g.coordinates[:,0]*2*np.pi)
    return np.concatenate(values,axis=1),np.concatenate(weights)


def saved_coefficients(directory, label, scope):
    path = directory/(label+'.npz')
    if scope == 'birth' and not path.exists():
        with np.load(directory/(label+'_birth.npz')) as archive:return archive['coeff']
    with np.load(path) as archive:return archive[f'{label}_{scope}_coeff']


def ensemble_comparisons(mesh, paths):
    summaries = [(p,json.loads(p.read_text())) for p in paths]
    initial = summaries[0][1]['initial']
    assert all(s['initial']==initial for _,s in summaries), 'ensemble requires identical complete initial state'
    result,skipped = [],[]
    seeds = sorted({c['seed'] for _,s in summaries for c in s['comparisons']})
    for other in ('reference_noise','new'):
        for scope in ('birth','newborn','whole'):
            pairs = []
            for seed in seeds:
                candidates = [(p,s) for p,s in summaries if any(c['seed']==seed and c['other']==other
                    and c['scope']==scope and c['representation']=='coeff' for c in s['comparisons'])]
                if not candidates:continue
                p,_ = candidates[-1]
                pairs.append((saved_coefficients(p.parent,f'reference_{seed}',scope),
                    saved_coefficients(p.parent,f'{other}_{seed}',scope),seed))
            if len(pairs)<2:
                skipped.append(dict(other=other,scope=scope,reason='fewer_than_two_complete_saved_pairs'))
                continue
            a = np.mean([pair[0] for pair in pairs],axis=0)
            b = np.mean([pair[1] for pair in pairs],axis=0)
            av,w = gauss_values(mesh,a)
            bv,_ = gauss_values(mesh,b)
            result.append(dict(other=other,scope=scope,representation='gauss_field_ensemble_mean',
                seed='mean_of_saved_seeds',ensemble_seeds=[pair[2] for pair in pairs],
                channels=metrics(av,bv,av@w,bv@w,weights=w)))
    return result,skipped


def gauss_from_summary(path, ensemble_paths=None):
    """Rebuild only the filtered mesh; never construct a model or advance markers."""
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.data.namelist import parse_namelist
    from src.models.native_coupling import NativeSourceMesh
    import h5py
    result = json.loads(path.read_text())
    run,anchor = Path(result['run']),result['anchor']
    root = Path(__file__).resolve().parents[1]
    registered = tuple(json.loads((root/'data/restart_events.json').read_text())['runs'][run.name]['restart_events'])
    read = lambda index:read_native_psi_teacher(run/f'jorek{index:06d}.h5',lcfs_restart_index=anchor,registered_restart_indices=registered)
    initial,imported = read(anchor+1),read(anchor)
    params = parse_namelist((run/'jorek_namelist_spitest').read_text())
    with h5py.File(run/f'jorek{anchor+1:06d}.h5','r') as f:F0=float(np.asarray(f['F0'][...]).ravel()[0])
    spec = dict(filter_perp=float(params['filter_perp_n0']),filter_hyper=float(params['filter_hyper_n0']),
        filter_parallel=float(params['filter_par_n0']),F0=F0,psi=imported.psi,
        field_time_si=imported.time_si,field_provenance=str(run/f'jorek{anchor:06d}.h5')+' projection creation',
        boundary=initial.boundary,axis_nodes=np.zeros(len(initial.boundary),bool),fix_axis_nodes=False,apply_dirichlet=True)
    started = time.monotonic()
    mesh = NativeSourceMesh(initial.coordinates,initial.size,initial.vertex,initial.indices,
        units=initial.units,projection='sparse',max_elements=5000,projection_n0=spec)
    if ensemble_paths is not None:
        comparisons,skipped = ensemble_comparisons(mesh,[path,*ensemble_paths])
        output = dict(anchor=anchor,exact_replay=result['exact_replay'],comparisons=comparisons,skipped=skipped,
            interpretation='Compare ensemble-mean fields, NOT the average per-seed errors; two seeds cannot certify bias or equivalence',
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,*ensemble_paths]},
            measure='g.weights * g.jacobian * R * 2pi',field_advances=0,marker_advances=0)
        destination = path.parent/'ensemble-gauss-metrics.json'
        write_json(destination,output)
        print(json.dumps(dict(output=str(destination),comparisons=len(comparisons),skipped=skipped)))
        return
    comparisons = []
    for comparison in result['comparisons']:
        if comparison['representation']!='coeff':continue
        seed,scope,other = comparison['seed'],comparison['scope'],comparison['other']
        alabel,blabel = f'reference_{seed}',f'{other}_{seed}'
        a = saved_coefficients(path.parent,alabel,scope)
        b = saved_coefficients(path.parent,blabel,scope)
        av,w = gauss_values(mesh,a)
        bv,bw = gauss_values(mesh,b)
        assert np.array_equal(w,bw) and (w>0).all()
        comparisons.append(dict(seed=seed,other=other,scope=scope,representation='gauss_field_volume_weighted',
            channels=metrics(av,bv,av@w,bv@w,weights=w)))
    output = dict(anchor=anchor,exact_replay=result['exact_replay'],comparisons=comparisons,
        projection_configuration=mesh.configuration, elapsed_seconds=time.monotonic()-started,
        measure='g.weights * g.jacobian * R * 2pi; initial full native 16-point element quadrature',
        source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        field_advances=0,marker_advances=0)
    write_json(path.parent/'gauss-metrics.json',output)
    write_json(path.parent/'gauss-metrics-summary.json',compact_metrics(output))
    print(json.dumps(dict(anchor=anchor,gauss_comparisons=len(comparisons),elapsed_seconds=output['elapsed_seconds'])))


def execute(args):
    import torch
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    run = root/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+('_injInner' if args.anchor == 5680 else ''))
    stages = Stages()
    started = time.monotonic()
    options = SimpleNamespace(chunk_size=2048, wall_seconds=args.window_seconds, memory_mib=8192,
        birth_max_proposals=1000000, spatial_backend='batched_v1',
        birth_cells_per_width=args.cells_per_width, birth_max_cells=args.birth_max_cells)
    _, source, shard, state, _ = build_real(run, args.anchor, _initial_pairing(run, args.anchor, stages),
        stages, options, feedback_mode='autonomous')
    build_seconds = time.monotonic()-started
    settings = shard.birth.configuration
    kwargs = {k: settings[k] for k in ('physical_particles','n_particle_slots','n_cpu','rank','R_geo','n_plane','max_proposals')}
    reference = NativeProductionBirths(shard.density, **kwargs)
    probability = NativeProductionBirths(shard.density, **kwargs, position_sampler=NativeTabulatedBirthSampler(
        geometry='axisymmetric_n0', n_tor=1, cells_per_width=args.cells_per_width, max_cells=args.birth_max_cells))
    table_evidence = {}
    prepare = probability.position_sampler.prepare
    def measured_prepare(*positional, **keywords):
        started = time.monotonic()
        prepared = prepare(*positional, **keywords)
        table_evidence.clear()
        table_evidence.update(prepared.diagnostics, build_seconds=time.monotonic()-started)
        return prepared
    probability.position_sampler.prepare = measured_prepare
    clock = state.marker.clock
    sampler = NativeSourceTimeSampler(state.fields_current, state.fluid_deltas,
        time_prev_si=clock.field_start_si, time_now_si=clock.field_end_si,
        tstep_si=state.fluid_dt_previous_si, static=source.field_static)
    target = state.fluid_time_si
    _, steps, gather = particle_window_schedule(clock, target)
    initial = dict(cloud=cloud_digest(state.marker.cloud), current=digest(state.fields_current),
        previous=digest(state.fields_previous), rawD=digest(state.fluid_deltas),
        feedback=digest(state.marker.feedback.values), feedback_history=digest(state.marker.feedback.deltas),
        clock=vars(clock), shards={k:digest(v) for k,v in state.shards.items() if isinstance(v,np.ndarray)})
    result = dict(anchor=args.anchor, run=str(run), initial=initial, build_seconds=build_seconds,
        slots=len(state.marker.cloud.weight), initial_live=int(state.marker.cloud.alive.sum()),
        particle_steps=steps, gather_si=gather, cells_per_width=args.cells_per_width,
        birth_max_cells=args.birth_max_cells, no_training=True, field_advances=0,
        rng_protocol='MT19937 distribution comparison, independent offsets; seed once per case', cases=[], comparisons=[])
    paths = [Path(__file__), root/'tools/native_real_chain_builder.py']+list((root/'src/models').glob('native_birth*.py'))+[root/'src/models/native_source_birth.py',root/'src/models/native_coupling.py']
    result['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(args.output_dir/'initial.json',result)
    arrays = {}
    seeds = args.seeds if args.seeds is not None else [args.seed]
    plans = [(f'{mode}_{seed}', birth, seed+offset) for seed in seeds for mode,birth,offset in
        [('reference',reference,0),('reference_noise',reference,1000003),('new',probability,2000003)]]
    plans.append(('reference_replay',reference,seeds[0]))
    original_deposit = source.mesh.deposit_batch
    original_project = source.mesh.project
    free = np.flatnonzero(~state.marker.cloud.alive)
    for label, birth, seed in plans:
        np.random.seed(seed)
        before_rng = rng_digest()
        t = time.monotonic()
        newborn = birth.births(state.shards,state.marker,target,state.fields_previous,state.fields_current,source,time_sampler=sampler)
        row = dict(label=label,seed=seed,rng_before=before_rng,rng_after_birth=rng_digest(),
            birth_seconds=time.monotonic()-t,birth_count=len(newborn.weight),newborn_digest=cloud_digest(newborn),
            birth_configuration=birth.configuration)
        if birth is probability:row['table'] = dict(table_evidence)
        load = np.zeros((len(newborn.weight),5))
        load[:,4] = newborn.weight/source.mesh.units.n_norm_m3
        birth_rhs = original_deposit(newborn,np.arange(len(newborn.weight)),load)
        arrays[label+'_birth_weak'] = birth_rhs
        arrays[label+'_birth_coeff'] = original_project(birth_rhs)
        write_json(args.output_dir/(label+'_birth.json'),row)
        np.savez_compressed(args.output_dir/(label+'_birth.npz'),
            weak=birth_rhs,coeff=arrays[label+'_birth_coeff'])
        if args.window_labels is not None and label not in args.window_labels:
            row.update(window_status='not_requested',window_seconds=0.,rng_after_window=rng_digest())
            result['cases'].append(row)
            write_json(args.output_dir/(label+'.json'),row)
            continue
        # Neutral newborns have only number at birth; energy/charge emerge during the window.
        born_slots = free[:len(newborn.weight)]
        captured = np.zeros_like(birth_rhs)
        def deposit(cloud, ids, loads):
            selected = np.isin(ids,born_slots)
            if selected.any():
                captured[:] += original_deposit(cloud,ids[selected],loads[selected])
            return original_deposit(cloud,ids,loads)
        def project(rhs):
            arrays[label+'_whole_weak'] = rhs.copy()
            return original_project(rhs)
        source.mesh.deposit_batch, source.mesh.project = deposit, project
        t = time.monotonic()
        try:
            advanced, diagnostic = source.advance(state.marker,target,state.fields_previous,state.fields_current,
                births=newborn,time_sampler=sampler)
            arrays[label+'_whole_coeff'] = advanced.feedback.values.copy()
            arrays[label+'_newborn_weak'] = captured/gather
            arrays[label+'_newborn_coeff'] = original_project(captured/gather)
            row.update(window_status='completed',window_steps=diagnostic['n_steps'],
                final_cloud_digest=cloud_digest(advanced.cloud),final_slots=len(advanced.cloud.weight))
        except (RuntimeError, ValueError) as exc:
            row.update(window_status='unfinished',window_reason=str(exc))
        finally:
            source.mesh.deposit_batch, source.mesh.project = original_deposit, original_project
        row.update(window_seconds=time.monotonic()-t,rng_after_window=rng_digest(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert cloud_digest(state.marker.cloud) == initial['cloud']
        assert digest(state.fields_current) == initial['current']
        assert digest(state.fields_previous) == initial['previous']
        assert digest(state.fluid_deltas) == initial['rawD']
        assert digest(state.marker.feedback.values) == initial['feedback']
        assert digest(state.marker.feedback.deltas) == initial['feedback_history']
        assert vars(state.marker.clock) == initial['clock']
        assert all(digest(state.shards[k]) == v for k,v in initial['shards'].items())
        row['initial_state_unchanged'] = True
        result['cases'].append(row)
        write_json(args.output_dir/(label+'.json'),row)
        np.savez_compressed(args.output_dir/(label+'.npz'),**{k:v for k,v in arrays.items() if k.startswith(label+'_')})
        print(json.dumps(row),flush=True)
    for seed in seeds:
        for other in ('reference_noise','new'):
            for scope in ('birth','newborn','whole'):
                for representation in ('weak','coeff'):
                    akey, bkey = f'reference_{seed}_{scope}_{representation}',f'{other}_{seed}_{scope}_{representation}'
                    if akey not in arrays or bkey not in arrays:continue
                    a,b = arrays[akey],arrays[bkey]
                    # Weak integral is action on constant test (derivative DOFs zero).
                    if representation == 'weak':
                        constant = np.zeros(source.mesh.n_dofs)
                        constant[source.mesh.inverse[0]] = 1
                        ai,bi = a@constant,b@constant
                    else:ai,bi = source.mesh.integrate(a),source.mesh.integrate(b)
                    result['comparisons'].append(dict(seed=seed,other=other,scope=scope,representation=representation,
                        channels=metrics(a,b,ai,bi)))
    replay = result['cases'][-1]
    first = result['cases'][0]
    result['exact_replay'] = dict(birth=first['newborn_digest']==replay['newborn_digest'],
        rng=first['rng_after_window']==replay['rng_after_window'],
        arrays=all(np.array_equal(v,arrays[k.replace(f'reference_{seeds[0]}_','reference_replay_',1)])
            for k,v in arrays.items() if k.startswith(f'reference_{seeds[0]}_') and k.replace(f'reference_{seeds[0]}_','reference_replay_',1) in arrays))
    result['exact_replay']['full_window'] = (
        first['window_status'] == replay['window_status'] == 'completed'
        and first['final_cloud_digest'] == replay['final_cloud_digest'])
    write_json(args.output_dir/'summary.json',result)
    write_json(args.output_dir/'metrics-summary.json',compact_metrics(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor',type=int,choices=(4110,5680))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--summary-only',type=Path,help='Read an existing summary.json without building or advancing')
    parser.add_argument('--gauss-from-summary',type=Path,help='Postprocess saved coefficients with a rebuilt mesh only')
    parser.add_argument('--ensemble-summaries',type=Path,nargs='*',help='Enable ensemble means, optionally adding other saved summaries; no windows rerun')
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed',type=int,default=2917)
    seeds.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--cells-per-width',type=int,choices=(64,128),default=64)
    parser.add_argument('--birth-max-cells',type=int,default=16000000)
    parser.add_argument('--threads',type=int,choices=(1,2),default=1)
    parser.add_argument('--window-seconds',type=float,default=60)
    parser.add_argument('--window-labels',nargs='+',help='Optional exact case labels to attempt; all births still run')
    args = parser.parse_args()
    if args.summary_only is not None:
        print(json.dumps(compact_metrics(json.loads(args.summary_only.read_text())),indent=2,allow_nan=False))
        return
    if args.gauss_from_summary is not None:
        gauss_from_summary(args.gauss_from_summary,args.ensemble_summaries)
        return
    if args.anchor is None or args.output_dir is None:
        parser.error('--anchor and --output-dir are required for execution')
    selected_seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(selected_seeds)) != len(selected_seeds) or any(not 0<=s<=2**32-1-2000003 for s in selected_seeds):
        parser.error('unique seeds in [0, 2**32-1-2000003] required for independent offsets')
    if not np.isfinite(args.window_seconds) or args.window_seconds<=0 or args.birth_max_cells<1:
        parser.error('positive finite window time and positive cell cap required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    saved = deepcopy(np.random.get_state())
    try:execute(args)
    finally:np.random.set_state(saved)


if __name__ == '__main__':
    main()
