"""Hand-calculated Gauss formula components; no FEM or fluid closure claim."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest

from src.models.marker_feedback import (
    ProjectionUnits, MarkerFeedbackParameters, GaussMarkerFeedback, GaussMetric,
    GaussFluid, consume_marker_feedback, electron_energy_weak_terms,
)
from src.models.marker_state import fluid_time_coefficients


def fixture():
    # Choose mass for m_i/m_imp=0.1 including the source's isotope correction.
    p = MarkerFeedbackParameters(1., 2.01455279245, 5/3, 'Ne', 10,
                                 'element_matrix', True, True, True)
    mu = 4e-7*np.pi
    units = ProjectionUnits('jorek_marker_projection', 1e20,
                            np.sqrt(mu*2.01455279245*1.660539040e-27*1e20),
                            1/(mu*(2/3)))
    a = lambda x: np.array([x], dtype=float)
    feedback = GaussMarkerFeedback(
        np.array([[-2., 3., 20., 6., 2.]]),
        np.array([[0., 0., 0., 8., 2.]]),
        np.array([[0., 0., 0., 5., 1.]]),
        np.array([[0., 0., 0., 4., .5]]),
        np.array([[7., 9., 11., .8, .2]]), units, 'gauss', 'raw_projection_increment')
    fluid = GaussFluid(a(10), a(2), a(4), a(10), a(2), a(4), 'model502_native')
    metric = GaussMetric(a(2), a(1), a(1), a(3))  # determinant5, nonorthogonal
    return feedback, fluid, metric, p


def run(f, u, g, p, ratio=2., scheme='Gears', h=.5):
    # SI scale cancels from the ratio, the direct rate uses h in JU.
    return consume_marker_feedback(f, u, g, p,
        fluid_time_coefficients(scheme, h, h/ratio),
        dt_si=h*(f.units.t_norm_si if f.units is not None else 1.))


def assert_scalar(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)


def test_independent_hand_calculated_charge_gradient_closures_and_history():
    f, u, g, p = fixture()
    old = deepcopy((f, u, g))
    out = run(f, u, g, p)
    # Hand oracle: q=6/2=3, q2=20/2=10, alpha=.1*3-1=-.7.
    assert_scalar(out.mean_charge, 3)
    assert_scalar(out.mean_square_charge, 10)
    assert_scalar(out.alpha_e, -.7)
    assert_scalar(out.alpha_e_bis, -.7)
    assert_scalar(out.alpha_e_tri, 0)
    for zero in vars(out.native_linearization).values():
        np.testing.assert_array_equal(zero, np.zeros(1))
    assert_scalar(out.ne_before_correction, 8.6)
    assert_scalar(out.ne_jorek, 8.6)
    assert_scalar(out.ne_si, 8.6e20)
    assert_scalar(out.effective_charge, 140/43)  # (20+8)/(43/5)
    assert_scalar(out.electron_pressure, 34.4)
    # dq=(dM1-q*dN)/N; transform x=(3*s-t)/5,y=(-s+2*t)/5.
    for name, value in [('s', 1), ('t', 1), ('phi', 1.25), ('x', .4), ('y', .2)]:
        assert_scalar(getattr(out.charge_derivatives, name), value)
        assert_scalar(getattr(out.alpha_derivatives, name), value/10)
    # ratio2: dq_delta=(1.6-3*.4)/2=.2; correction=-.1*.2*4*2=-.16.
    assert_scalar(out.delta_charge, .2)
    assert_scalar(out.delta_alpha_e, .02)
    assert_scalar(out.projection_history_jorek, -.16)
    for before, after in zip(old, (f, u, g)):
        for name, v in vars(before).items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(v, getattr(after, name))
    assert not np.shares_memory(out.scaled_projection_deltas, f.raw_deltas)


@pytest.mark.parametrize('c1,c2,rate', [(2., 3., -5.), (-2., 3., -1.),
                                      (-4., -3., 4.), (0., 0., 0.)])
def test_energy_sign_units_and_weak_measure(c1, c2, rate):
    f, u, g, p = fixture()
    f.values[0, :2] = c1, c2
    out = run(f, u, g, p)
    assert_scalar(out.energy_rate_jorek, rate)
    # Convert back to physical power; integrating for h*t_norm gives rate*h*E_norm.
    assert_scalar(out.energy_rate_si, rate*f.units.energy_norm_j_m3/f.units.t_norm_si)
    assert_scalar(out.energy_rate_si*(.5*f.units.t_norm_si), rate*.5*f.units.energy_norm_j_m3)
    direct, history = electron_energy_weak_terms(out, test_value=np.array([2.]),
        big_r=np.array([3.]), xjac=np.array([5.]), quadrature_weight=np.array([.25]))
    assert_scalar(direct, rate*.5*7.5)
    assert_scalar(history, -.16*7.5)  # NO dt, theta, zeta multiplication


@pytest.mark.parametrize('density', [0., -.2, .25, .5])
def test_density_floor_preserves_native_gradient_and_delta_rules(density):
    f, u, g, p = fixture()
    f.values[0, 4] = density
    f.values[0, 3] = 1.
    out = run(f, u, g, p)
    assert_scalar(out.density, .5)
    assert_scalar(out.mean_charge, 2)
    # Floor is NOT differentiated here: raw dN_s=2 and deltaN=.2 survive.
    assert_scalar(out.density_derivatives.s, 2)
    assert_scalar(out.charge_derivatives.s, 8)
    assert_scalar(out.delta_charge, 1.6)


def test_negative_moments_clip_values_but_not_projection_derivatives():
    f, u, g, p = fixture()
    f.values[0, 2:5] = -4., -3., -2.
    out = run(f, u, g, p)
    assert_scalar(out.mean_charge, 0)
    assert_scalar(out.mean_square_charge, 0)
    assert_scalar(out.charge_derivatives.s, 16)  # raw8 / floor.5
    assert_scalar(out.delta_charge, 3.2)         # raw.8 * ratio2 / floor.5
    assert_scalar(out.effective_charge, 1)


def test_zero_projection_is_zero_energy_but_not_zero_fluid_density():
    f, u, g, p = fixture()
    f = replace(f, values=np.zeros((1, 5)), d_s=np.zeros((1, 5)),
                d_t=np.zeros((1, 5)), d_phi=np.zeros((1, 5)), raw_deltas=np.zeros((1, 5)))
    out = run(f, u, g, p)
    assert_scalar(out.density, .5)
    assert_scalar(out.ne_jorek, 8.)
    assert_scalar(out.energy_rate_jorek, 0)
    assert_scalar(out.projection_history_jorek, 0)


@pytest.mark.parametrize('moment,capped,grad', [(20., 10., -6.), (22., 10., 0.)])
def test_charge_cap_does_not_cap_precomputed_history_or_square_moment(moment, capped, grad):
    f, u, g, p = fixture()
    f.values[0, 3] = moment
    f.values[0, 2] = 2000.
    out = run(f, u, g, p)
    assert_scalar(out.mean_charge, capped)
    assert_scalar(out.charge_derivatives.s, grad)  # cap condition strict >10
    assert_scalar(out.delta_charge, (.8-(moment/2)*.2))
    assert_scalar(out.mean_square_charge, 1000.) # NOT capped at Z_nuc^2
    assert out.effective_charge[0] > 10          # total Zeff not upper capped either


def test_low_temperature_override_order_matches_source():
    f, u, g, p = fixture()
    temp = .05*(1.602176565e-19*4e-7*np.pi*1e20)
    out = run(f, replace(u, te_corrected=np.array([temp])), g, p)
    assert_scalar(out.mean_charge, 0)
    assert_scalar(out.alpha_e, -1)
    assert_scalar(out.charge_derivatives.s, 1) # source does NOT clear spatial gradient
    assert_scalar(out.delta_charge, .2)       # source does NOT clear delta either


@pytest.mark.parametrize('ne', [-1., 0., .00015, .0002])
def test_separate_jorek_exponential_density_correction_and_si_floor(ne):
    f, u, g, p = fixture()
    f.values[0, 3] = 0  # alpha=-1; ne=rho_corrected-rho_imp_corrected
    # Small positive densities avoid avoidable subtraction cancellation in the oracle.
    u = replace(u, rho_corrected=np.array([.001+ne]) if ne > -1 else np.array([1.]),
                rho_imp_corrected=np.array([.001]) if ne > -1 else np.array([2.]))
    out = run(f, u, g, p)
    # Independent scalar exponential correction with a=b=0.0001.
    expected = ne if ne >= .0002 else .0001+.0001*np.exp((ne-.0002)/.0001)
    assert_scalar(out.ne_jorek, expected)
    assert_scalar(out.ne_si, max(ne*1e20, 1e16))
    if ne == 0:
        assert out.ne_jorek[0] != out.ne_si[0]/1e20


@pytest.mark.parametrize('ratio', [.25, 1., 2., 4.])
@pytest.mark.parametrize('scheme', ['Gears', 'implicit Euler', 'Crank-Nicholson'])
def test_variable_step_scaled_once_and_history_not_multiplied_by_zeta(ratio, scheme):
    f, u, g, p = fixture()
    raw = f.raw_deltas.copy()
    out = run(f, u, g, p, ratio=ratio, scheme=scheme)
    assert_scalar(out.delta_charge, .1*ratio)
    assert_scalar(out.projection_history_jorek, -.08*ratio)
    np.testing.assert_array_equal(f.raw_deltas, raw)
    np.testing.assert_array_equal(out.scaled_projection_deltas, raw*ratio)


def test_only_energy_channels_set_direct_energy_rate():
    f, u, g, p = fixture()
    out = run(f, u, g, p)
    f.values[0, 2:] *= 3
    changed = run(f, u, g, p)
    np.testing.assert_array_equal(out.energy_rate_jorek, changed.energy_rate_jorek)
    assert not np.array_equal(out.effective_charge, changed.effective_charge)
    # Fluid rho_imp is distinct from c5. History uses raw rho_imp=2, not c5=6.
    assert_scalar(changed.projection_history_jorek, -.16/3)


@pytest.mark.parametrize('bad', ['units', 'scaled', 'nodes', 'delta', 'gradient',
                                'fft', 'species', 'normalization', 'metric', 'fluid'])
def test_missing_or_unsupported_inputs_fail(bad):
    f, u, g, p = fixture()
    if bad == 'units': f = replace(f, units=None)
    if bad == 'scaled': f = replace(f, delta_convention='scaled')
    if bad == 'nodes': f = replace(f, location='node_dof0')
    if bad == 'delta': f = replace(f, raw_deltas=None)
    if bad == 'gradient': f = replace(f, d_phi=None)
    if bad == 'fft': p = replace(p, assembly_path='element_matrix_fft')
    if bad == 'species': p = replace(p, impurity='unknown')
    if bad == 'normalization': f = replace(f, units=replace(f.units, n_norm_m3=1))
    if bad == 'metric': g = replace(g, x_s=np.array([1.]), x_t=np.array([1.]),
                                  y_s=np.array([1.]), y_t=np.array([1.]))
    if bad == 'fluid': u = replace(u, rho_corrected=None)
    with pytest.raises(ValueError):
        run(f, u, g, p)


def test_multiple_gauss_points_use_pointwise_not_global_closures():
    f, u, g, p = fixture()
    feedback = replace(f, values=np.array([[2., 3., 20., 6., 2.],
                                          [-4., -1., -2., -3., 0.]]),
        d_s=np.zeros((2, 5)), d_t=np.zeros((2, 5)), d_phi=np.zeros((2, 5)),
        raw_deltas=np.zeros((2, 5)))
    fluid = replace(u, **{k: np.repeat(v, 2) for k, v in vars(u).items() if isinstance(v, np.ndarray)})
    metric = GaussMetric(*(np.repeat(v, 2) for v in vars(g).values()))
    out = run(feedback, fluid, metric, p)
    np.testing.assert_array_equal(out.mean_charge, [3., 0.])
    np.testing.assert_array_equal(out.density, [2., .5])
    np.testing.assert_array_equal(out.energy_rate_jorek, [-5., 4.])
    np.testing.assert_allclose(out.ne_jorek, [8.6, 8.], rtol=2e-14)


@pytest.mark.parametrize('flag', ['use_marker', 'with_impurities', 'with_TiTe'])
def test_unsupported_model_flags_fail(flag):
    f, u, g, p = fixture()
    with pytest.raises(ValueError, match='requires model502'):
        run(f, u, g, replace(p, **{flag: False}))


@pytest.mark.parametrize('bad', ['nonfinite', 'shape', 'time', 'step'])
def test_invalid_arrays_and_time_fail(bad):
    from src.models.marker_state import FluidTimeCoefficients
    f, u, g, p = fixture()
    coefficients = fluid_time_coefficients('Gears', 2., 1.)
    step = f.units.t_norm_si
    if bad == 'nonfinite': f = replace(f, values=np.full((1, 5), np.nan))
    if bad == 'shape': f = replace(f, d_t=np.zeros((2, 5)))
    if bad == 'time': coefficients = FluidTimeCoefficients(1., .5, 2.)
    if bad == 'step': step = 0.
    with pytest.raises(ValueError):
        consume_marker_feedback(f, u, g, p, coefficients, dt_si=step)
