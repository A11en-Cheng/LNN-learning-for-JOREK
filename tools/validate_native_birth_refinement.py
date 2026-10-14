"""Deterministic probability-table refinement and cost audit; no training.

Comparison supports are fixed independently of table resolution. The native
mass-source reference is analytic when unsaturated and composite Gauss otherwise.
The target applies to probability bin masses, NOT finite-marker projection noise.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from src.models.native_birth_positions import NativeTabulatedBirthSampler
from tools.benchmark_native_birth_generation import load_snapshot, probabilities_on_support
from tools.validate_native_birth_distribution import analytic_probabilities, coefficients, metrics, native_literal_probabilities


def audit_case(path, *, resolutions=(8,32,64,128), bins=96, seed=2917, target=1e-4):
    kernel, shards, options, count, record = load_snapshot(path)
    lo, hi = options['bounds']
    r = np.linspace(lo[0],hi[0],bins+1)
    z = np.linspace(lo[1],hi[1],bins+1)
    reference_start = time.perf_counter()
    upper = coefficients(kernel,shards,options)[2].sum()*options['dv']/options['norm']
    if upper < 1:
        truth, proof = analytic_probabilities(kernel,shards,options,r,z)
        uncertainty = proof['float32_rounding_TV_bound']
        reference = dict(method='analytic_unsaturated_cylindrical_Gaussian_integrals',proof=proof,
                         refinement_relative_l2=None,rounding_TV_bound=uncertainty)
    else:
        low, _ = native_literal_probabilities(kernel,shards,options,r,z,order=16)
        truth, proof = native_literal_probabilities(kernel,shards,options,r,z,order=32)
        convergence = metrics(truth,low)
        # An a posteriori diagnostic, not a rigorous quadrature error bound.
        if convergence['relative_l2'] > target/5:
            refined, proof = native_literal_probabilities(kernel,shards,options,r,z,order=64)
            convergence = metrics(refined,truth)
            truth = refined
        reference = dict(method='independent_literal_real4_total_gate_composite_Gauss',proof=proof,
                         refinement_relative_l2=convergence['relative_l2'],rigorous_error_bound=False)
    record.update(scope='deterministic_fixed_support_birth_probability_and_cost_only',
                  no_training=True,projection_error_measured=False,seed=seed,
                  seed_policy='local_Generator_for_replay_probe_global_RNG_untouched',
                  comparison_bins=[bins,bins],reference=reference,
                  reference_seconds=time.perf_counter()-reference_start,
                  target_relative_l2=target,resolution_sweep=[])
    for resolution in resolutions:
        sampler = NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,
                    cells_per_width=resolution,max_cells=16_000_000)
        tick = time.perf_counter(); cpu = time.process_time()
        table = sampler.prepare(kernel,shards,**options)
        build = time.perf_counter()-tick; build_cpu = time.process_time()-cpu
        tick = time.perf_counter()
        points = table.sample(count,rng=np.random.default_rng(seed))
        sample_seconds = time.perf_counter()-tick
        replay = table.sample(count,rng=np.random.default_rng(seed))
        different = table.sample(count,rng=np.random.default_rng(seed+1))
        if not np.array_equal(points,replay):raise AssertionError('same-seed replay failed')
        if count and np.array_equal(points,different):raise AssertionError('different seed did not change positions')
        distribution = probabilities_on_support(table,r,z)
        error = metrics(truth,distribution)
        # Avoid claiming certification when the independent reference is unresolved.
        converged = reference['refinement_relative_l2'] is None or reference['refinement_relative_l2'] <= target/5
        row = dict(cells_per_width=resolution,configuration=sampler.configuration,
                   diagnostics=table.diagnostics,build_seconds=build,build_cpu_seconds=build_cpu,
                   sample_seconds=sample_seconds,**error,
                   empirical_target_met=bool(error['relative_l2']<=target and converged),
                   reference_resolved_for_target=converged,
                   same_seed_bitwise_positions=True,different_seed_changes_positions=True,
                   positions_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                   process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        record['resolution_sweep'].append(row)
        print(json.dumps(dict(snapshot=str(path),**{k:row[k] for k in
              ('cells_per_width','build_seconds','relative_l2','total_variation','empirical_target_met')})),flush=True)
        del table
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolutions',type=int,nargs='+',default=[8,32,64,128])
    p.add_argument('--bins',type=int,default=96)
    p.add_argument('--seed',type=int,default=2917)
    p.add_argument('--target',type=float,default=1e-4)
    args=p.parse_args()
    if args.output.exists():p.error('new output file required; existing evidence is preserved')
    if not 0<=args.seed<2**32-1:p.error('seed must leave room for one replay offset')
    if not 8<=args.bins<=128 or not 1e-6<=args.target<=1e-3:p.error('bounded bins and target required')
    if any(not 2<=r<=128 for r in args.resolutions):p.error('resolution must be in [2,128]')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':p.error(key+'=1 required before startup')
    repo=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),repo/'tools/validate_native_birth_distribution.py',
           repo/'tools/benchmark_native_birth_generation.py',repo/'src/models/native_birth_positions.py',
           repo/'src/models/native_birth_acceptance.py',repo/'src/models/native_source_density.py']
    result=dict(schema='native-birth-refinement-v1',no_training=True,seed=args.seed,
                source_code_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},cases=[])
    for suffix,index in (('',4111),('_injInner',5681)):
        path=repo/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)/f'jorek{index:06d}.h5'
        result['cases'].append(audit_case(path,resolutions=args.resolutions,bins=args.bins,
                                      seed=args.seed,target=args.target))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output))),flush=True)


if __name__=='__main__':main()
