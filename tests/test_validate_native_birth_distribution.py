"""Independent traditional birth-law verification; no field or training."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import quad

from tests.test_native_production_density import source_fixture
from tools.benchmark_native_birth_generation import legacy_positions
from tools.validate_native_birth_distribution import (
    analytic_probabilities, certify_native_gate_bound, gaussian_integrals,
    histogram, metrics, native_literal_probabilities, run_overlap, sample_unsaturated_reference,
    sample_vectorized_legacy, vectorized_source_acceptance,
)


def inputs(dv=.001):
    kernel,s=source_fixture()
    options=dict(time_si=1.,bounds=np.array([[2.35,.35],[2.65,.65]]),dv=dv,
        norm=float(s['ablation_atoms_s'].sum())*kernel.mesh.units.t_norm_si/kernel.mesh.units.n_norm_m3)
    return kernel,s,options


@pytest.mark.parametrize('radial',[False,True])
def test_gaussian_integrals_match_independent_quadrature(radial):
    edges=np.array([2.35,2.48,2.52,2.65]);center=2.5;width=.05
    actual=gaussian_integrals(edges,np.array([center]),np.array([width]),radial=radial)[0]
    expected=[quad(lambda x:np.exp(-((x-center)/width)**2)*(x if radial else 1),lo,hi)[0]
              for lo,hi in zip(edges[:-1],edges[1:])]
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=1e-16)


@pytest.mark.parametrize('dv',[.001,.2])
def test_vectorized_gate_matches_original_scalar_source_query(dv):
    kernel,s,options=inputs(dv)
    s={k:np.repeat(v,2,axis=0) if isinstance(v,np.ndarray) else v for k,v in s.items()}
    s['positions'][1,:2]=[2.54,.48]
    points=np.random.default_rng(301).uniform([2.35,.35,0],[2.65,.65,2*np.pi],(300,3))
    rate=kernel.prepare_rates(s,time_si=1.).rates_at(points)
    expected=np.minimum((rate*dv/options['norm']).astype(np.float32),1.)
    np.testing.assert_array_equal(vectorized_source_acceptance(kernel,s,options,points),expected)


def test_vectorized_rejection_first_accept_replays_original_seven_draw_proposal():
    kernel,s,options=inputs(.1)
    before=deepcopy(np.random.get_state())
    try:
        np.random.seed(2917)
        old,_=legacy_positions(kernel,s,options,1)
    finally:np.random.set_state(before)
    new,_=sample_vectorized_legacy(kernel,s,options,1,np.random.RandomState(2917),batch=128)
    np.testing.assert_array_equal(new,old)


def test_unsaturated_reference_histogram_and_cylindrical_mean_match_analytic_law():
    kernel,s,options=inputs()
    r=np.linspace(2.35,2.65,9);z=np.linspace(.35,.65,9)
    truth,proof=analytic_probabilities(kernel,s,options,r,z)
    points,_=sample_unsaturated_reference(kernel,s,options,120000,np.random.default_rng(41))
    assert truth.sum()==pytest.approx(1.)
    assert proof['global_acceptance_upper_bound']<1
    assert metrics(truth,histogram(points,r,z))['relative_l2']<.02
    numerator=quad(lambda x:x*x*np.exp(-((x-2.5)/.05)**2),2.35,2.65)[0]
    denominator=quad(lambda x:x*np.exp(-((x-2.5)/.05)**2),2.35,2.65)[0]
    assert abs(points[:,0].mean()-numerator/denominator)<.0003


def test_analytic_reference_rejects_unproved_unsaturated_case():
    kernel,s,options=inputs(10.)
    with pytest.raises(ValueError,match='upper bound'):
        analytic_probabilities(kernel,s,options,np.linspace(2.35,2.65,3),np.linspace(.35,.65,3))
    with pytest.raises(ValueError,match='unsaturated'):
        sample_unsaturated_reference(kernel,s,options,3,np.random.default_rng(2))


def test_native_cell_bound_contains_pointwise_mass_source_gate():
    kernel,s,options=inputs()
    r=np.linspace(2.35,2.65,13);z=np.linspace(.35,.65,13)
    result=certify_native_gate_bound(kernel,s,options,r,z)
    points=np.array([[2.5,.5,0.],[2.49,.47,0.]])
    actual=kernel.prepare_rates(s,time_si=1.).rates_at(points)*options['dv']/options['norm']
    assert result['native_literal_gate_cell_upper_bound']>=max(actual)
    assert result['central_mass_over_20']==pytest.approx(kernel.mesh.units.central_mass/20)
    assert result['production_correction_applied'] is True


def test_unsaturated_native_literal_quadrature_cancels_global_mass_scale():
    kernel,s,options=inputs()
    r=np.linspace(2.35,2.65,9);z=np.linspace(.35,.65,9)
    truth,_=analytic_probabilities(kernel,s,options,r,z)
    native,diag=native_literal_probabilities(kernel,s,options,r,z,order=24,phi_order=16)
    assert diag['max_native_gate_at_quadrature']<1
    assert metrics(truth,native)['relative_l2']<1e-7


def test_native_literal_saturation_changes_normalized_distribution_without_table():
    kernel,s,options=inputs(.01)
    r=np.linspace(2.35,2.65,9);z=np.linspace(.35,.65,9)
    truth,_=analytic_probabilities(kernel,s,options,r,z,historical_number_scale=True)
    native,diag=native_literal_probabilities(kernel,s,options,r,z,order=32,phi_order=16)
    assert diag['max_native_gate_at_quadrature']>1
    assert metrics(native,truth)['relative_l2']>.05


def test_saturated_overlap_exposes_per_shard_plateau_then_sum_difference():
    result=run_overlap(12000)
    assert result['table_diagnostics']['saturated_cells']>0
    assert result['quadrature_refinement_error']['relative_l2']<.0003
    assert result['table_vs_independent_quadrature']['relative_l2']<.015
    assert result['per_shard_cap_then_sum_vs_total_cap']['relative_l2']>.05


def test_constant_correlation_and_invalid_histogram_are_explicit():
    assert metrics(np.ones(3),np.ones(3))['correlation'] is None
    with pytest.raises(ValueError,match='nonzero'):metrics(np.zeros(3),np.ones(3))
    with pytest.raises(ValueError,match='outside'):histogram(np.array([[4.,5.,0.]]),[0,1],[0,1])
