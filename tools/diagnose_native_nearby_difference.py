"""Bounded same-target discrete locator diagnosis; not a speed benchmark.

Runs complete public source windows, preserving target evolution. The new
locator is observed without overriding native geometry. Crossed/lost targets
are also passed to the unchanged old locator with identical hints.
"""
import argparse
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import signal
import time
import traceback
from functools import partial

import numpy as np

import tools.benchmark_native_nearby as benchmark
from tools.benchmark_native_lookup import _json, _rss
from tools.benchmark_native_neighbor_locator import _compare, _rng, _restore_rng
from tools.diagnose_native_real_chain import _array_digest


def point_evidence(mesh, target, element, st):
    out=dict(element=int(element),st=np.asarray(st).tolist())
    if element>0:
        rz,jac=mesh._location_geometry(int(element)-1,np.asarray(st))
        out.update(RZ=rz.tolist(),RZ_residual=float(np.linalg.norm(rz-target[:2])),
                   jacobian_determinant=float(np.linalg.det(jac)),
                   jacobian_condition=float(np.linalg.cond(jac)))
    return out


class ObservedNearby:
    def __init__(self,locator,initial,births,*,all_points=False,source=None,fields=None):
        self.locator=locator
        self.cloud=deepcopy(initial.cloud)
        if births is not None:self.cloud.insert(deepcopy(births),grow=False)
        self.ids=np.flatnonzero(self.cloud.alive);self.offset=0;self.step=0
        self.rows=[];self.mismatch_visits=0;self.checked_visits=0
        self.all_points=all_points;self.source=source;self.fields=fields
        self.feedback=initial.feedback.values
        self.top_local=[];self.first_significant_local=None

    def __getattr__(self,name):return getattr(self.locator,name)

    def locate_batch(self,positions,*,element_hints=None,local_hints=None):
        if self.offset==len(self.ids):
            self.ids=np.flatnonzero(self.cloud.alive);self.offset=0;self.step+=1
        ids=self.ids[self.offset:self.offset+len(positions)]
        self.offset+=len(positions)
        actual=self.locator.locate_batch(positions,element_hints=element_hints,local_hints=local_hints)
        elm,st=actual
        check=(np.arange(len(positions)) if self.all_points else
               np.flatnonzero((elm!=element_hints)|(self.locator.last_ifail!=0)))
        self.checked_visits+=len(check)
        if len(check):
            old,old_st=self.locator.mesh.locate_batch(positions[check],
                element_hints=element_hints[check],local_hints=local_hints[check])
            mismatched=np.flatnonzero(old!=elm[check])
            self.mismatch_visits+=len(mismatched)
            for j in mismatched:
                i=check[j]
                if len(self.rows)<100:
                    self.rows.append(dict(slot=int(ids[i]),particle_substep=self.step,
                        target=positions[i].tolist(),old_hint=int(element_hints[i]),
                        old_local_hint=local_hints[i].tolist(),nearby_ifail=int(self.locator.last_ifail[i]),
                        baseline=point_evidence(self.locator.mesh,positions[i],old[j],old_st[j]),
                        nearby=point_evidence(self.locator.mesh,positions[i],elm[i],st[i])))
            if self.all_points:
                same=(old==elm[check])&(old>0)
                delta=np.linalg.norm(old_st-st[check],axis=1)
                eligible=np.flatnonzero(same)
                significant=eligible[delta[eligible]>1e-8]
                selected=list(eligible[np.argsort(delta[eligible])[-2:]])
                if self.first_significant_local is None and len(significant):
                    selected.append(int(significant[0]))
                for j in set(selected):
                    i=check[j]
                    row=dict(slot=int(ids[i]),particle_substep=self.step,
                        target=positions[i].tolist(),old_hint=int(element_hints[i]),
                        old_local_hint=local_hints[i].tolist(),local_difference_norm=float(delta[j]),
                        baseline=point_evidence(self.locator.mesh,positions[i],old[j],old_st[j]),
                        nearby=point_evidence(self.locator.mesh,positions[i],elm[i],st[i]))
                    if self.source is not None:
                        v,g,_,rz=self.locator.mesh._marker_points(
                            np.array([old[j]-1,elm[i]-1]),np.stack([old_st[j],st[i]]),
                            self.fields,self.feedback)
                        u=self.source.mesh.units
                        b=np.column_stack((g[:,0,1],-g[:,0,0],np.full(2,self.source.F0)))/rz[:,0,None]
                        velocity=np.column_stack((-g[:,1,1],g[:,1,0],np.zeros(2)))*rz[:,0,None]/u.t_norm_si+v[:,6,None]/u.t_norm_si*b
                        ni=np.maximum(v[:,7]*u.n_norm_m3*u.central_mass/20,0.)
                        grad=g[:,7]*u.n_norm_m3*u.central_mass/20
                        velocity[:,:2]-=self.source.diffusion_ju/u.t_norm_si*grad/np.maximum(ni,1e12)[:,None]
                        denom=np.linalg.norm(velocity[0])
                        row.update(sampled_velocity_relative_l2=float(np.linalg.norm(velocity[1]-velocity[0])/denom) if denom else None)
                    if self.first_significant_local is None and j in significant:
                        self.first_significant_local=row.copy()
                    self.top_local.append(row)
                self.top_local=sorted(self.top_local,key=lambda r:r['local_difference_norm'],reverse=True)[:32]
        self.cloud.i_elm[ids]=elm
        return actual


def diagnose(source,candidate,initial,target,previous,current,*,births=None,
             sampler_factory=None,record=None,all_points=False,**unused):
    rng=_rng();immutable=_array_digest((initial,previous,current,births))
    reference=ref_diag=ref_rng=None;out=None
    rows=[]
    try:
        for mode,runner in [('baseline',source),('jorek_nearby',candidate)]:
            marker=deepcopy(initial);born=deepcopy(births);_restore_rng(rng)
            sampler=sampler_factory() if sampler_factory else None
            if mode=='jorek_nearby':
                if all_points and sampler is not None and not sampler.static:
                    raise ValueError('all-point field sensitivity probe requires static sampler')
                observer=ObservedNearby(candidate.locator,initial,births,all_points=all_points,
                    source=source,fields=current)
                candidate.locator=observer
            out,diag=runner.advance(marker,target,previous,current,births=born,
                **({} if sampler is None else dict(time_sampler=sampler)))
            current_rng=_rng()
            if reference is None:reference,ref_diag,ref_rng=out,diag,current_rng
            comparison=_compare(reference,out,initial,ref_diag,diag,ref_rng,current_rng)
            rows.append(dict(mode=mode,comparison=comparison))
        slots=np.flatnonzero(reference.cloud.i_elm!=out.cloud.i_elm)
        final=[]
        for i in slots[:100]:
            final.append(dict(slot=int(i),baseline=point_evidence(source.mesh,reference.cloud.x[i],
                reference.cloud.i_elm[i],reference.cloud.st[i]),
                nearby=point_evidence(source.mesh,out.cloud.x[i],out.cloud.i_elm[i],out.cloud.st[i]),
                baseline_position=reference.cloud.x[i].tolist(),nearby_position=out.cloud.x[i].tolist()))
        record.update(success=all(r['comparison']['success'] for r in rows),
            iterations=rows,discrete_final_mismatch_count=len(slots),final_mismatches=final,
            same_target_checked_visits=observer.checked_visits,
            same_target_element_mismatch_visits=observer.mismatch_visits,
            same_target_examples=observer.rows,inputs_unchanged=immutable==_array_digest((initial,previous,current,births)),
            all_points_compared=all_points,top_same_element_local_differences=observer.top_local,
            first_local_difference_above_1e_8=observer.first_significant_local,
            timing_scope='diagnostic_only_additional_locator_comparisons_not_speed')
        return record
    finally:
        _restore_rng(rng)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index',type=int,choices=(0,1),default=1)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--all-points',action='store_true',help='also compare successful same-element updates; adds old locator work, never timing eligible')
    args=p.parse_args(argv)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(k)!='1':p.error(k+'=1 required')
    args.config=Path('config/native-autonomous-training-pilot.json');args.seed=1
    args.chunk_size=2048;args.wall_seconds=600;args.memory_mib=8192
    args.birth_max_proposals=1000000;args.spatial_backend='batched_v1';args.marker_locator='native';args.repeats=1
    result=dict(schema='native-nearby-discrete-diagnosis-v1',success=False,training=False,
        optimizer_updates=0,production_qualified=False,configuration=vars(args))
    repo=Path(__file__).resolve().parents[1]
    paths=[Path(__file__).resolve(),repo/'tools/benchmark_native_nearby.py',
        repo/'tools/native_real_chain_builder.py',repo/'src/models/native_nearby_locator.py',
        repo/'src/models/native_coupling.py',repo/'src/models/marker_state.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['source_code_sha256']=hashes
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def timeout(*_):raise TimeoutError('cooperative 600s deadline exceeded')
    with args.output.open('x') as f:
        old=signal.signal(signal.SIGALRM,timeout);signal.alarm(600);start=time.perf_counter()
        saved=benchmark.benchmark_source_window
        try:
            benchmark.benchmark_source_window=partial(diagnose,all_points=args.all_points)
            benchmark.execute(args,result)
        except Exception as exc:
            result.update(success=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        finally:
            benchmark.benchmark_source_window=saved
            signal.alarm(0);signal.signal(signal.SIGALRM,old)
            result['code_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)
            result['success'] &= result['code_unchanged']
            result.update(total_seconds=time.perf_counter()-start,peak_rss_bytes=_rss())
            json.dump(result,f,default=_json,indent=2,allow_nan=False);f.write('\n')
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
