"""Deterministic motion differential and trusted-local selected-particle replay.

Diagnostic only: no production methods, physics or defaults are edited. Full
capture runs both public Source.advance paths. Replay removes other particles
only (feedback is fixed inside a source window), retaining the real sampling,
charge, pusher, history, location and sparse projection callbacks.
Never load a pickle supplied by an untrusted party.
"""
import argparse
from copy import copy, deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time
from unittest.mock import patch

import numpy as np

from tools.benchmark_native_lookup import _json, _rss
from tools.benchmark_native_neighbor_locator import _compare, _rng, _restore_rng
from tools.diagnose_native_real_chain import _array_digest


def select_cloud(state, slots):
    result = deepcopy(state)
    for key in state.cloud.array_names():
        setattr(result.cloud, key, getattr(state.cloud, key)[slots].copy())
    if getattr(result,'transport_history',None) is not None:
        for key in ('previous_dt_si','level','lifetime'):
            setattr(result.transport_history,key,getattr(state.transport_history,key)[slots].copy())
    result.cloud.validate()
    return result


def concentration(reference, actual, initial, key, support):
    delta = getattr(reference.cloud, key)[support]-getattr(initial.cloud, key)[support]
    error = getattr(actual.cloud, key)[support]-getattr(reference.cloud, key)[support]
    squared = np.sum(error**2, axis=1)
    order = np.argsort(squared)[::-1]
    total = float(squared.sum())
    denominator = float(np.sum(delta**2))
    slots = np.flatnonzero(support)
    return dict(relative_l2=np.sqrt(total/denominator) if denominator else None,
        error_squared=total, reference_increment_squared=denominator,
        error_energy_fractions={str(k):float(squared[order[:k]].sum()/total) if total else 0.
                                for k in (1,4,16,64,256)},
        top_slots=slots[order[:16]].tolist())


def capture(source, candidate, initial, target, previous, current, *, births=None,
            sampler_factory=None, record=None, capture_path=None, **unused):
    rng = _rng(); immutable = _array_digest((initial, previous, current, births))
    results = []
    try:
        for name, runner in [('baseline', source), ('neighbor', candidate)]:
            print(json.dumps(dict(phase='capture', mode=name, status='started')), flush=True)
            _restore_rng(rng)
            sampler = sampler_factory() if sampler_factory else None
            tick = time.perf_counter()
            out, diag = runner.advance(initial, target, previous, current, births=births,
                                       **({} if sampler is None else dict(time_sampler=sampler)))
            results.append((out, diag, _rng()))
            print(json.dumps(dict(phase='capture', mode=name, status='completed',
                                  seconds=time.perf_counter()-tick)), flush=True)
        a, da, ra = results[0]; b, db, rb = results[1]
        support = a.cloud.alive & b.cloud.alive & initial.cloud.alive
        support &= (a.cloud.i_life == initial.cloud.i_life) & (b.cloud.i_life == initial.cloud.i_life)
        rows = {k:concentration(a,b,initial,k,support) for k in ('x','V_prev')}
        slots = np.unique(rows['x']['top_slots']+rows['V_prev']['top_slots'])
        # All retained identities existed at the start: births cannot overwrite
        # them. Other markers affect projection, not motion inside this window.
        selected = select_cloud(initial, slots)
        mesh = copy(source.mesh); mesh._factor = None
        saved_source = copy(source); saved_source.mesh = mesh
        payload = dict(schema='trusted-nearby-motion-replay-v1',source=saved_source,
            neighbours=candidate.locator.neighbours.T.copy(),initial=selected,target=target,
            previous=previous,current=current,sampler=sampler_factory() if sampler_factory else None,
            original_slots=slots,expected={name:{k:getattr(out.cloud,k)[slots].copy()
                for k in out.cloud.array_names()} for name,out in [('baseline',a),('nearby',b)]},
            original_input_digest=immutable)
        with capture_path.open('xb') as stream:pickle.dump(payload,stream,protocol=5)
        comparison = _compare(a,b,initial,da,db,ra,rb)
        record.update(success=True,iterations=[dict(mode='baseline'),dict(mode='neighbor')],
            comparison=comparison,motion=rows,capture_path=str(capture_path),
            selected_slots=slots.tolist(),inputs_unchanged=immutable==_array_digest((initial,previous,current,births)),
            timing_scope='diagnostic_only_not_benchmark',experiment_executed=True)
        record['success'] &= record['inputs_unchanged']
        return record
    finally:_restore_rng(rng)


def sample_detail(source, cloud, ids, fields, feedback):
    mesh = source.mesh; u = mesh.units
    v,g,_,rz = mesh._marker_points(cloud.i_elm[ids].astype(np.int64)-1,cloud.st[ids],fields,feedback.values)
    ni_raw = v[:,7]*u.n_norm_m3*u.central_mass/20
    denominator = np.maximum(np.maximum(ni_raw,0),1e12)
    exb = np.column_stack((-g[:,1,1],g[:,1,0],np.zeros(len(ids))))*rz[:,0,None]/u.t_norm_si
    parallel = v[:,6,None]/u.t_norm_si*np.column_stack((g[:,0,1],-g[:,0,0],np.full(len(ids),source.F0)))/rz[:,0,None]
    diffusion = -source.diffusion_ju/u.t_norm_si*g[:,7]*u.n_norm_m3*u.central_mass/20/denominator[:,None]
    return dict(field=v,gradient=g,RZ=rz,ni_raw=ni_raw,ni_denominator=denominator,
                exb=exb,parallel=parallel,diffusion=diffusion)


def traced_advance(runner, state, target, previous, current, sampler):
    """Observe actual callbacks, returning their values unmodified."""
    from src.models.marker_state import advance_window
    rows = []
    def observed_advance(*args, **kwargs):
        sample, locate = kwargs['sample'], kwargs['locate_with_hints']
        latest = None
        def observe_sample(t,cloud,ids,feedback):
            nonlocal latest
            actual = sample(t,cloud,ids,feedback)
            fields = sampler.fields_at(t) if sampler is not None else previous+(current-previous)*((t-state.clock.field_start_si)/(state.clock.field_end_si-state.clock.field_start_si))
            latest = dict(time_si=t,ids=ids.copy(),x=cloud.x[ids].copy(),st=cloud.st[ids].copy(),
                element=cloud.i_elm[ids].copy(),V_prev=cloud.V_prev[ids].copy(),
                velocity=actual['velocity'].copy(),detail=sample_detail(runner,cloud,ids,fields,feedback))
            return actual
        def observe_locate(x,**hints):
            actual = locate(x,**hints)
            latest.update(next_x=x.copy(),next_element=actual[0].copy(),next_st=actual[1].copy())
            rows.append(latest)
            return actual
        kwargs.update(sample=observe_sample,locate_with_hints=observe_locate)
        return advance_window(*args,**kwargs)
    with patch('src.models.native_coupling.advance_window', observed_advance):
        out,diag = runner.advance(state,target,previous,current,
            **({} if sampler is None else dict(time_sampler=deepcopy(sampler))))
    return out,diag,rows


def load_capture(path):
    with path.open('rb') as stream:payload = pickle.load(stream)
    if payload['schema'] != 'trusted-nearby-motion-replay-v1':raise ValueError('wrong replay schema')
    source = payload['source']; mesh = source.mesh
    if mesh.projection == 'sparse':
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        scale = diags(mesh._equilibration)
        mesh._factor = splu((scale@mesh.mass@scale).tocsc())
    return payload


def refine_baseline_mesh(mesh):
    """Diagnostic only: refine accepted old-cell inverse maps, never choose cells."""
    result=copy(mesh);original=mesh.locate_batch
    def locate(positions,**hints):
        elements,local=original(positions,**hints)
        ids=np.flatnonzero(elements>0)
        for _ in range(5):
            if not len(ids):break
            rz,jac=mesh._location_geometries(elements[ids].astype(np.int64)-1,local[ids])
            residual=rz-positions[ids,:2]
            active=np.linalg.norm(residual,axis=1)>1e-14
            ids=ids[active]
            if len(ids):
                local[ids]-=np.linalg.solve(jac[active],residual[active,...,None])[...,0]
                if not np.isfinite(local[ids]).all() or ((local[ids]<0)|(local[ids]>1)).any():
                    raise ValueError('diagnostic refinement left accepted element; not a tolerance-only probe')
        ids=np.flatnonzero(elements>0)
        if len(ids):
            rz,_=mesh._location_geometries(elements[ids].astype(np.int64)-1,local[ids])
            if (np.linalg.norm(rz-positions[ids,:2],axis=1)>1e-14).any():
                raise ValueError('diagnostic refinement did not converge')
        return elements,local
    result.locate_batch=locate
    return result


def replay(payload, *, slot=None, diffusion_scale=1., dt_divisor=1,steps=None,refine_baseline=False,local_subcycling=False):
    if local_subcycling and dt_divisor!=1:
        raise ValueError('local subcycling import requires original native velocity history interval; dt_divisor must be 1')
    from src.models.native_nearby_locator import NativeNearbyMarkerSource
    source = copy(payload['source'])
    source.configuration = deepcopy(source.configuration)
    source.diffusion_ju *= diffusion_scale
    source.configuration['diffusion_ju'] = source.diffusion_ju
    state = payload['initial']
    if slot is not None:
        indices = np.flatnonzero(payload['original_slots'] == slot)
        if len(indices) != 1:raise ValueError('slot absent from capture')
        state = select_cloud(state,indices)
    state = deepcopy(state); state.clock.particle_dt_si /= dt_divisor
    if local_subcycling:
        from src.models.marker_transport import LocalSubcycling,initialize_transport_history
        source.transport_policy=LocalSubcycling()
        source.configuration['transport_policy']=source.transport_policy.configuration
        state=initialize_transport_history(state,source.transport_policy,previous_dt_si=state.clock.particle_dt_si)
    target=payload['target']
    if steps is not None:
        target=state.clock.particle_time_si+steps*state.clock.particle_dt_si
        if target>payload['target']+1e-14:raise ValueError('diagnostic steps exceed captured window')
    nearby = NativeNearbyMarkerSource(source,neighbours=payload['neighbours'])
    baseline=source
    if refine_baseline:
        baseline=copy(source);baseline.mesh=refine_baseline_mesh(source.mesh)
    outs={}; traces={}; rows=[]
    for name,runner in [('baseline',baseline),('nearby',nearby)]:
        start=time.perf_counter()
        out,diag,trace=traced_advance(runner,state,target,payload['previous'],payload['current'],payload['sampler'])
        outs[name]=out;traces[name]=trace
        rows.append(dict(mode=name,seconds=time.perf_counter()-start,n_steps=diag['n_steps'],execution=diag['execution']))
    a,b=outs['baseline'],outs['nearby'];support=a.cloud.alive&b.cloud.alive&state.cloud.alive
    result=dict(success=True,training=False,optimizer_updates=0,timings=rows,
        source_diffusion_ju=source.diffusion_ju,particle_dt_si=state.clock.particle_dt_si,
        baseline_refined=refine_baseline,target_si=target,
        local_subcycling=local_subcycling,
        motion={k:concentration(a,b,state,k,support) for k in ('x','V_prev')},
        traces=traces,original_slots=([slot] if slot is not None else payload['original_slots'].tolist()))
    if slot is None and diffusion_scale==1 and dt_divisor==1 and steps is None and not refine_baseline and not local_subcycling:
        result['full_cloud_replay_comparison']={name:{k:dict(array_equal=bool(np.array_equal(getattr(out.cloud,k),expected[k])),
            max_absolute=float(np.max(np.abs(getattr(out.cloud,k)-expected[k]))))
            for k in out.cloud.array_names()} for name,out in outs.items() for expected in [payload['expected'][name]]}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path)
    p.add_argument('--run-index',type=int,choices=(0,1),default=1)
    p.add_argument('--slot',type=int)
    p.add_argument('--diffusion-scale',type=float,default=1.)
    p.add_argument('--dt-divisor',type=int,choices=(1,2,4,8,16,32,64),default=1)
    p.add_argument('--steps',type=int)
    p.add_argument('--refine-baseline',action='store_true')
    p.add_argument('--local-subcycling',action='store_true',help='explicit numerical approximation; not native fixed particle_dt')
    p.add_argument('--assert-consistent',action='store_true',help='diagnostic red loop: relative velocity increment error <=1e-4 (not a production acceptance threshold)')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':p.error(key+'=1 required')
    if not 0<=args.diffusion_scale<=1 or (args.steps is not None and not 1<=args.steps<=10000):p.error('bounded diagnostic diffusion scale/steps required')
    result={}; start=time.perf_counter(); args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__).resolve(),Path('src/models/native_coupling.py'),Path('src/models/native_nearby_locator.py'),Path('src/models/marker_state.py'),Path('src/models/marker_transport.py')]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    def timeout(*_):raise TimeoutError('diagnostic deadline exceeded')
    old=signal.signal(signal.SIGALRM,timeout); signal.alarm(600)
    try:
        if args.capture:
            result=replay(load_capture(args.capture),slot=args.slot,diffusion_scale=args.diffusion_scale,
                dt_divisor=args.dt_divisor,steps=args.steps,refine_baseline=args.refine_baseline,local_subcycling=args.local_subcycling)
        else:
            import tools.benchmark_native_nearby as benchmark
            args.config=Path('config/native-autonomous-training-pilot.json');args.seed=1;args.chunk_size=2048
            args.wall_seconds=600;args.memory_mib=8192;args.birth_max_proposals=1000000
            args.spatial_backend='batched_v1';args.marker_locator='native';args.repeats=1
            result.update(training=False,optimizer_updates=0,production_qualified=False)
            with patch.object(benchmark,'benchmark_source_window',partial(capture,capture_path=args.output.with_suffix('.pickle'))):
                benchmark.execute(args,result)
        error=result['motion']['V_prev']['relative_l2']
        result['diagnostic_increment_consistent']=error is not None and error<=1e-4
        result['assert_consistent_requested']=args.assert_consistent
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        result.update(total_seconds=time.perf_counter()-start,peak_rss_bytes=_rss(),source_code_sha256=hashes,
            code_unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path)] for path in paths))
        with args.output.open('x') as stream:json.dump(result,stream,default=_json,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('motion','total_seconds','diagnostic_increment_consistent')},default=_json),flush=True)
    return 1 if args.assert_consistent and not result['diagnostic_increment_consistent'] else 0


if __name__=='__main__':raise SystemExit(main())
