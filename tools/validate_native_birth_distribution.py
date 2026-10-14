"""Independent traditional-law/table comparison, no mesh, fields or training.

Real unsaturated cases use analytic Gaussian integrals and a different-proposal
rejection sampler with the same continuous target law. Actual scalar legacy
replays are retained separately. A saturated overlap fixture also exercises
literal cylindrical-uniform/source-gate rejection in vectorized batches.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from math import pi,sqrt
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm,ks_2samp

from src.models.native_birth_positions import NativeTabulatedBirthSampler
from tools.benchmark_native_birth_generation import load_snapshot,legacy_positions,probabilities_on_support
from src.models.native_birth_acceptance import NATIVE_BIRTH_ACCEPTANCE


def coefficients(kernel,s,options):
    """Reuse only owned source coefficients, never the new probability table."""
    prepared=kernel.prepare_rates(s,time_si=options['time_si'])
    if not prepared.blocks:raise ValueError('active reference source required')
    positions=np.concatenate([b[0] for b in prepared.blocks])
    widths=np.sqrt(np.concatenate([b[1][:,0] for b in prepared.blocks]))
    amplitudes=np.concatenate([b[2] for b in prepared.blocks])
    return positions,widths,amplitudes,prepared.deltaphi


def gaussian_integrals(edges,centers,widths,*,radial=False):
    """Integrate exp(-(x-center)^2/w^2), with optional exact R Jacobian."""
    t=(np.asarray(edges)[None,:]-centers[:,None])/widths[:,None]
    result=widths[:,None]*sqrt(pi)/2*np.diff(erf(t),axis=1)
    if radial:
        result=centers[:,None]*result+widths[:,None]**2/2*(np.exp(-t[:,:-1]**2)-np.exp(-t[:,1:]**2))
    # Vanishing far-tail erf differences can incur roundoff cancellation.
    return np.maximum(result,0.)


def analytic_probabilities(kernel,s,options,r_edges,z_edges,*,historical_number_scale=False):
    pos,w,a,dphi=coefficients(kernel,s,options)
    scale=kernel.mesh.units.central_mass/20 if historical_number_scale else 1.
    upper=float(a.sum()*scale*options['dv']/options['norm'])
    if upper>=1:raise ValueError('analytic unsaturated reference requires global threshold upper bound <1')
    jr=gaussian_integrals(r_edges,pos[:,0],w,radial=True)
    jz=gaussian_integrals(z_edges,pos[:,1],w)
    mass=jr.T@(a[:,None]*jz)
    probability=mass/mass.sum()
    toroidal_integral=dphi*sqrt(pi)*erf(pi/dphi)
    integral=float(mass.sum()*toroidal_integral*scale*options['dv']/options['norm'])
    volume=pi*(options['bounds'][1,0]**2-options['bounds'][0,0]**2)*np.diff(options['bounds'][:,1])[0]
    # Conservative normalized-TV rounding bound, ignoring FP64 evaluation error.
    delta=2**-24*integral+2**-150*volume
    return probability,dict(global_acceptance_upper_bound=upper,
        continuous_acceptance_integral_m3=integral,float32_rounding_TV_bound=delta/(integral-delta))


def certify_native_gate_bound(kernel,s,options,r_edges,z_edges):
    """Conservative all-phi, within-cell upper bound on JOREK's mass-source gate.

    Independent Gaussian maxima, not table values. The old extra number-source
    conversion has been removed from both corrected Python birth paths.
    """
    pos,w,a,_=coefficients(kernel,s,options)
    def maxima(edges,centers):
        distance=np.maximum(np.maximum(edges[:-1][None,:]-centers[:,None],
                                      centers[:,None]-edges[1:][None,:]),0.)
        return np.exp(-(distance/w[:,None])**2)
    upper=maxima(r_edges,pos[:,0]).T@(a[:,None]*maxima(z_edges,pos[:,1]))
    bound=float(upper.max()*options['dv']/options['norm'])
    return dict(central_mass_over_20=kernel.mesh.units.central_mass/20,
        native_literal_gate_cell_upper_bound=bound,
        native_literal_gate_unsaturated_proved=bool(bound<1-2**-24),
        scope='all_RZ_cells_all_phi_gaussian_maxima_no_mesh_no_table_values',
        production_correction_applied=True)


def native_literal_probabilities(kernel,s,options,r_edges,z_edges,*,order=24,phi_order=8):
    """Independent composite quadrature of the literal Fortran mass-source gate.

    Gaussian separability accelerates evaluation only; total clipping is done
    at every quadrature point. No birth table, clipped-node interpolation or
    extra number-source conversion. The angular rule integrates actual phi.
    """
    x,weights=np.polynomial.legendre.leggauss(order)
    r=(r_edges[:-1,None]+r_edges[1:,None])/2+np.diff(r_edges)[:,None]/2*x
    z=(z_edges[:-1,None]+z_edges[1:,None])/2+np.diff(z_edges)[:,None]/2*x
    pos,w,a,dphi=coefficients(kernel,s,options)
    pr=np.exp(-((r.ravel()[None,:]-pos[:,0,None])/w[:,None])**2)
    pz=np.exp(-((z.ravel()[None,:]-pos[:,1,None])/w[:,None])**2)
    px,pw=np.polynomial.legendre.leggauss(phi_order)
    integrated=np.zeros((r.size,z.size));maximum=0.
    for phi,pweight in zip(pi*(px+1),pw):
        angle=np.abs((phi-pos[:,2]+pi)%(2*pi)-pi)
        rates=pr.T@((a*np.exp(-(angle/dphi)**2))[:,None]*pz)
        gate=(rates*options['dv']/options['norm']).astype(np.float32).astype(float)
        maximum=max(maximum,float(gate.max()))
        integrated+=pweight*np.minimum(gate,1.)
    integrated=integrated.reshape(len(r_edges)-1,order,len(z_edges)-1,order)
    wr=np.diff(r_edges)[:,None]/2*weights*r
    wz=np.diff(z_edges)[:,None]/2*weights
    mass=np.einsum('iajb,ia,jb->ij',integrated,wr,wz)*pi
    return mass/mass.sum(),dict(order=order,phi_order=phi_order,
        max_native_gate_at_quadrature=maximum,acceptance_integral_m3=float(mass.sum()),
        approximation='independent_composite_Gauss_quadrature_literal_mass_source_clip')


def sample_unsaturated_reference(kernel,s,options,count,rng):
    pos,_,a,_=coefficients(kernel,s,options)
    upper=float(a.sum()*options['dv']/options['norm'])
    if upper>=1:raise ValueError('reference mixture requires proved unsaturated source')
    return sample_source_mixture(kernel,s,options,count,rng)


def sample_source_mixture(kernel,s,options,count,rng):
    """Independent truncated Gaussian mixture + R/Rmax rejection, no table.

    Choose shard by its integrated cylindrical source mass. Keep that shard
    fixed while rejecting radial proposals, then draw Z and periodic delta-phi.
    Different RNG protocol; only continuous-target distribution equivalence.
    """
    pos,w,a,dphi=coefficients(kernel,s,options);lo,hi=options['bounds']
    jr=gaussian_integrals([lo[0],hi[0]],pos[:,0],w,radial=True)[:,0]
    jz=gaussian_integrals([lo[1],hi[1]],pos[:,1],w)[:,0]
    weights=a*jr*jz;weights/=weights.sum()
    cdf=np.cumsum(weights);cdf[-1]=1.
    ids=np.searchsorted(cdf,rng.random(count),side='right')
    sigma=w[ids]/sqrt(2);center=pos[ids]
    points=np.empty((count,3));pending=np.arange(count);proposals=0
    for _ in range(1000):
        if not len(pending):break
        proposed=truncnorm.ppf(rng.random(len(pending)),
            (lo[0]-center[pending,0])/sigma[pending],(hi[0]-center[pending,0])/sigma[pending],
            loc=center[pending,0],scale=sigma[pending])
        accept=rng.random(len(pending))<proposed/hi[0]
        points[pending[accept],0]=proposed[accept]
        proposals+=len(pending);pending=pending[~accept]
    if len(pending):raise RuntimeError('radial reference rejection budget exceeded')
    points[:,1]=truncnorm.ppf(rng.random(count),(lo[1]-center[:,1])/sigma,
        (hi[1]-center[:,1])/sigma,loc=center[:,1],scale=sigma)
    tau=dphi/sqrt(2)
    delta=truncnorm.ppf(rng.random(count),-pi/tau,pi/tau,scale=tau)
    points[:,2]=(center[:,2]+delta)%(2*pi)
    if not np.isfinite(points).all():raise ValueError('nonfinite reference birth point')
    return points,proposals


def sample_native_reference(kernel,s,options,count,rng):
    """Independent Gaussian mixture followed by TOTAL-source plateau correction.

    Proposal density is proportional to f; accept min(1,1/f), so the final
    continuous target is proportional to min(f,1). No new table is used.
    Real*4 rounding is excluded here and independently tested at the old gate.
    """
    _,_,a,_=coefficients(kernel,s,options)
    if a.sum()*options['dv']/options['norm']<1:
        return sample_unsaturated_reference(kernel,s,options,count,rng)
    prepared=kernel.prepare_rates(s,time_si=options['time_si'])
    accepted=[];found=0;proposals=0
    for _ in range(100):
        if found>=count:break
        points,radial=sample_source_mixture(kernel,s,options,count-found,rng)
        proposals+=radial
        f=prepared.rates_at(points)*options['dv']/options['norm']
        keep=rng.random(len(points))<1/np.maximum(f,1.)
        accepted.append(points[keep]);found+=np.count_nonzero(keep)
    if found<count:raise RuntimeError('total-source plateau correction budget exceeded')
    return np.concatenate(accepted)[:count],proposals


def vectorized_source_acceptance(kernel,s,options,positions):
    """Literal total-source gate, shard-order FP64 sum then real*4 threshold."""
    pos,w,amp,dphi=coefficients(kernel,s,options)
    result=np.zeros(len(positions))
    for center,width,amplitude in zip(pos,w,amp):
        pol=np.exp(-np.sum((positions[:,:2]-center[:2])**2,axis=1)/width**2)
        angle=np.abs((positions[:,2]-center[2]+pi)%(2*pi)-pi)
        result+=amplitude*pol*np.exp(-(angle/dphi)**2)
    return np.minimum(np.maximum(result*options['dv']/options['norm'],0.).astype(np.float32).astype(float),1.)


def sample_vectorized_legacy(kernel,s,options,count,rng,*,batch=8192,max_proposals=2_000_000):
    """Uniform cylindrical proposals + literal source gate, no grid lookup."""
    lo,hi=options['bounds'];accepted=[];found=0;proposals=0
    while found<count:
        n=min(batch,max_proposals-proposals)
        if n<1:raise RuntimeError('vectorized legacy proposal budget exceeded')
        draws=rng.random((n,7));proposals+=n
        x=np.column_stack((np.sqrt(lo[0]**2+draws[:,0]*(hi[0]**2-lo[0]**2)),
            lo[1]+draws[:,1]*(hi[1]-lo[1]),2*pi*draws[:,2]))
        a=vectorized_source_acceptance(kernel,s,options,x)
        keep=x[draws[:,3]<a]
        accepted.append(keep);found+=len(keep)
    return np.concatenate(accepted)[:count],proposals


def metrics(reference,actual):
    reference=np.asarray(reference);actual=np.asarray(actual)
    denom=float(np.linalg.norm(reference))
    if denom<=0:raise ValueError('nonzero reference required')
    correlation=(float(np.corrcoef(reference.ravel(),actual.ravel())[0,1])
                 if np.ptp(reference)>0 and np.ptp(actual)>0 else None)
    return dict(relative_l2=float(np.linalg.norm(actual-reference)/denom),
        correlation=correlation,total_variation=.5*float(np.sum(np.abs(actual-reference))))


def histogram(points,r_edges,z_edges):
    counts=np.histogram2d(points[:,0],points[:,1],bins=(r_edges,z_edges))[0]
    if counts.sum()!=len(points):raise ValueError('birth sample outside comparison support')
    return counts/len(points)


def run_real(path,*,sample_count=50000,seeds=(31,47,89,113),literal_repeats=1,cells_per_width=8,literal_seed=2917):
    kernel,s,options,birth_count,record=load_snapshot(path)
    lo,hi=options['bounds']
    # Fixed comparison support, unrelated to either algorithm's table nodes.
    r_edges=np.linspace(lo[0],hi[0],25);z_edges=np.linspace(lo[1],hi[1],25)
    # Historical target is preserved as an explicitly named diagnostic only.
    old_truth,old_proof=analytic_probabilities(kernel,s,options,r_edges,z_edges,historical_number_scale=True)
    native16,diag16=native_literal_probabilities(kernel,s,options,r_edges,z_edges,order=16)
    native32,diag32=native_literal_probabilities(kernel,s,options,r_edges,z_edges,order=32)
    native64,diag64=native_literal_probabilities(kernel,s,options,r_edges,z_edges,order=64)
    truth=native64;proof=diag64
    if np.sum(coefficients(kernel,s,options)[2])*options['dv']/options['norm']<1:
        truth,proof=analytic_probabilities(kernel,s,options,r_edges,z_edges)
    sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,cells_per_width=cells_per_width,max_cells=16_000_000)
    table=sampler.prepare(kernel,s,**options)
    modeled=probabilities_on_support(table,r_edges,z_edges)
    native_audit=certify_native_gate_bound(kernel,s,options,table.r_edges,table.z_edges)
    native_audit.update(quadrature=diag64,quadrature_refinement_error=metrics(native64,native32),
        coarse_quadrature_refinement_error=metrics(native32,native16),
        historical_number_gate_vs_native_literal=metrics(native64,old_truth),
        new_table_vs_native_literal=metrics(native64,modeled),
        near_uniform_angular_kernel=True)
    refinements=[]
    for resolution in sorted(set((8,16,32,cells_per_width))):
        candidate=table if resolution==cells_per_width else NativeTabulatedBirthSampler(
            geometry='axisymmetric_n0',n_tor=1,cells_per_width=resolution,max_cells=16_000_000).prepare(kernel,s,**options)
        distribution=probabilities_on_support(candidate,r_edges,z_edges)
        refinements.append(dict(cells_per_width=resolution,**metrics(truth,distribution)))
    record.update(reference='JOREK_literal_mass_source_gate_analytic_or_independent_composite_quadrature_not_refined_table',
        proof=proof,deterministic_distribution_error=metrics(truth,modeled),resolution_sweep=refinements,
        native_mass_gate_audit=native_audit,
        comparison_bins=[24,24],sample_count_per_seed=sample_count,seeds=list(seeds),
        table_configuration=sampler.configuration,ensembles=[],literal_replays=[],
        no_mesh=True,no_field=True,no_marker_advance=True,production_qualified=False)
    for seed in seeds:
        tick=time.perf_counter()
        reference,proposals=sample_native_reference(kernel,s,options,sample_count,np.random.default_rng(seed))
        reference2,_=sample_native_reference(kernel,s,options,sample_count,np.random.default_rng(seed+10000))
        actual=table.sample(sample_count,rng=np.random.default_rng(seed+20000))
        refhist=histogram(reference,r_edges,z_edges);refhist2=histogram(reference2,r_edges,z_edges)
        newhist=histogram(actual,r_edges,z_edges)
        row=dict(seed=seed,seconds=time.perf_counter()-tick,reference_radial_proposals=proposals,
            reference_vs_reference=metrics(refhist,refhist2),reference_vs_table_sample=metrics(refhist,newhist),
            reference_vs_independent_truth=metrics(truth,refhist),table_sample_vs_independent_truth=metrics(truth,newhist),
            table_sample_vs_own_distribution=metrics(modeled,newhist),
            marginal_KS={str(j):float(ks_2samp(reference[:,j],actual[:,j]).statistic) for j in range(3)},
            RZ_mean_relative_error=float(np.linalg.norm(actual[:,:2].mean(0)-reference[:,:2].mean(0))/np.linalg.norm(reference[:,:2].mean(0))))
        record['ensembles'].append(row)
        print(json.dumps(dict(stage='ensemble',snapshot=str(path),**row)),flush=True)
    original=deepcopy(np.random.get_state())
    try:
        for i in range(literal_repeats):
            np.random.seed(literal_seed+i);tick=time.perf_counter()
            points,proposals=legacy_positions(kernel,s,options,birth_count)
            actual=table.sample(birth_count,rng=np.random.default_rng(literal_seed+i))
            # Deliberately report low-N diagnostics, not same-ID position error.
            row=dict(seed=literal_seed+i,count=birth_count,proposals=proposals,seconds=time.perf_counter()-tick,
                legacy_positions=points.tolist(),table_positions=actual.tolist(),
                legacy_hist_vs_analytic=metrics(truth,histogram(points,r_edges,z_edges)),
                table_hist_vs_analytic=metrics(truth,histogram(actual,r_edges,z_edges)))
            record['literal_replays'].append(row)
    finally:np.random.set_state(original)
    record['sampling_noise_summary']={key:dict(
        min=float(min(e[key]['relative_l2'] for e in record['ensembles'])),
        max=float(max(e[key]['relative_l2'] for e in record['ensembles'])),
        median=float(np.median([e[key]['relative_l2'] for e in record['ensembles']])))
        for key in ('reference_vs_reference','reference_vs_table_sample','reference_vs_independent_truth','table_sample_vs_independent_truth')}
    return record


def saturated_probabilities(kernel,s,options,r_edges,z_edges,*,order=32,per_shard_cap=False):
    """Independent Gauss quadrature: no new table interpolation or sampler.

    The overlap fixture's near-uniform phi is integrated by a separate 8-point
    angular quadrature. A per-shard-cap-then-sum target is diagnostic only.
    """
    x,weights=np.polynomial.legendre.leggauss(order)
    r=(r_edges[:-1,None]+r_edges[1:,None])/2+np.diff(r_edges)[:,None]/2*x
    z=(z_edges[:-1,None]+z_edges[1:,None])/2+np.diff(z_edges)[:,None]/2*x
    rr,zz=np.meshgrid(r.ravel(),z.ravel(),indexing='ij')
    positions=np.column_stack((rr.ravel(),zz.ravel(),np.zeros(rr.size)))
    pos,w,a,dphi=coefficients(kernel,s,options)
    px,pw=np.polynomial.legendre.leggauss(8)
    angular=np.zeros(rr.size)
    for phi,pweight in zip(pi*(px+1),pw):
        total=np.zeros(rr.size)
        for center,width,amplitude in zip(pos,w,a):
            angle=abs((phi-center[2]+pi)%(2*pi)-pi)
            f=amplitude*np.exp(-np.sum((positions[:,:2]-center[:2])**2,axis=1)/width**2-(angle/dphi)**2)
            f*=options['dv']/options['norm']
            total+=np.minimum(f,1.) if per_shard_cap else f
        # Match the original real*4 gate; a sum of individual plateaus is an
        # alternative source measure, NOT an executable probability >1.
        angular+=pweight*(total if per_shard_cap else np.minimum(total.astype(np.float32).astype(float),1.))
    angular=angular.reshape(len(r_edges)-1,order,len(z_edges)-1,order)
    wr=np.diff(r_edges)[:,None]/2*weights*r
    wz=np.diff(z_edges)[:,None]/2*weights
    mass=np.einsum('iajb,ia,jb->ij',angular,wr,wz)*pi
    return mass/mass.sum()


def run_overlap(sample_count=50000):
    from tests.test_native_production_density import source_fixture
    from src.models.native_source_density import NativeShardDensityKernel
    old,one=source_fixture()
    kernel=NativeShardDensityKernel(old.mesh,ns_radius_min=.05,ns_radius_ratio=1.4,
        ns_deltaphi=1e3,spi_num_vol=False,ns_delta_minor_rad=0.,injection_time_si=0.,toroidal_angles=(0.,))
    s={key:np.repeat(value,2,axis=0) if isinstance(value,np.ndarray) else value for key,value in one.items()}
    s['positions'][1,:2]=[2.54,.48]
    options=dict(time_si=1.,bounds=np.array([[2.35,.35],[2.65,.65]]),dv=.01,
        norm=float(s['ablation_atoms_s'].sum())*kernel.mesh.units.t_norm_si/kernel.mesh.units.n_norm_m3)
    table=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,cells_per_width=8).prepare(kernel,s,**options)
    lo,hi=options['bounds'];r_edges=np.linspace(lo[0],hi[0],17);z_edges=np.linspace(lo[1],hi[1],17)
    a,proposals=sample_vectorized_legacy(kernel,s,options,sample_count,np.random.default_rng(167))
    b,_=sample_vectorized_legacy(kernel,s,options,sample_count,np.random.default_rng(267))
    new=table.sample(sample_count,rng=np.random.default_rng(367))
    quadrature=saturated_probabilities(kernel,s,options,r_edges,z_edges,order=64)
    quadrature32=saturated_probabilities(kernel,s,options,r_edges,z_edges,order=32)
    alternative=saturated_probabilities(kernel,s,options,r_edges,z_edges,order=64,per_shard_cap=True)
    return dict(scope='synthetic_two_shard_overlap_literal_uniform_source_gate_not_real_production',
        count=sample_count,proposals=proposals,table_diagnostics=table.diagnostics,
        independent_quadrature_order=64,quadrature_refinement_error=metrics(quadrature,quadrature32),
        table_vs_independent_quadrature=metrics(quadrature,probabilities_on_support(table,r_edges,z_edges)),
        per_shard_cap_then_sum_vs_total_cap=metrics(quadrature,alternative),
        reference_vs_reference=metrics(histogram(a,r_edges,z_edges),histogram(b,r_edges,z_edges)),
        reference_vs_table_sample=metrics(histogram(a,r_edges,z_edges),histogram(new,r_edges,z_edges)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-count',type=int,default=50000)
    parser.add_argument('--literal-repeats',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--seeds',type=int,nargs='+',default=[31,47,89,113],help='Independent ensemble base seeds; reference B/new use +10000/+20000.')
    parser.add_argument('--seed',type=int,default=2917,help='Low-count literal reference replay seed.')
    parser.add_argument('--cells-per-width',type=int,default=64)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('new evidence output file required')
    if not 1000<=args.sample_count<=200000:raise ValueError('bounded sample count required')
    if not 0<=args.seed<2**32-3 or any(not 0<=s<2**32-20000 for s in args.seeds):
        parser.error('seeds must be nonnegative and leave room for comparison offsets')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+'=1 required before startup')
    from tools.benchmark_native_cpu_threads import native_thread_counts
    repo=Path(__file__).resolve().parents[1];root=repo/'data/EHL-2'
    files=(Path(__file__),repo/'src/models/native_birth_positions.py',repo/'src/models/native_source_density.py',
           repo/'src/models/native_source_birth.py',repo/'src/models/native_birth_acceptance.py',repo/'tools/benchmark_native_birth_generation.py')
    result=dict(schema='native-birth-distribution-comparison-v2',acceptance=NATIVE_BIRTH_ACCEPTANCE,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        no_training=True,seed=args.seed,seeds=args.seeds,source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        native_thread_counts=native_thread_counts(),cases=[])
    for suffix,index in (('',4111),('_injInner',5681)):
        path=root/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)/f'jorek{index:06d}.h5'
        result['cases'].append(run_real(path,sample_count=args.sample_count,literal_repeats=args.literal_repeats,
            seeds=args.seeds,cells_per_width=args.cells_per_width,literal_seed=args.seed))
    result['saturated_overlap']=run_overlap(args.sample_count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),deterministic_errors=[r['deterministic_distribution_error'] for r in result['cases']])),flush=True)


if __name__=='__main__':main()
