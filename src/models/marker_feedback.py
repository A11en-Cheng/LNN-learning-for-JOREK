"""Deterministic model502 *ordinary* assembly formulas at Gauss points.

This is a NumPy reference consumer, not FEM evaluation, an assembler, or a
nine-field step. Inputs must already be evaluated from the full projection
basis. No file reads, learning parameters, or node-dof0 interpolation occur.
See report/marker-feedback-consumption-2026-10-07.md for the source map.
"""
from dataclasses import dataclass
import numpy as np

from .marker_state import FluidTimeCoefficients

# Match this JOREK checkout's models/constants.f90, including its older amu.
MU_ZERO = 4e-7 * np.pi
ATOMIC_MASS_UNIT = 1.660539040e-27
EL_CHG = 1.602176565e-19


@dataclass(frozen=True)
class ProjectionUnits:
    convention: str               # jorek_marker_projection
    n_norm_m3: float
    t_norm_si: float
    energy_norm_j_m3: float


@dataclass(frozen=True)
class MarkerFeedbackParameters:
    central_density: float
    central_mass: float
    gamma: float
    impurity: str
    nuclear_charge: int
    assembly_path: str             # element_matrix, never FFT
    use_marker: bool
    with_impurities: bool
    with_TiTe: bool

    def validate(self, units):
        if not isinstance(units, ProjectionUnits) or units.convention != 'jorek_marker_projection':
            raise ValueError('explicit JOREK marker projection units required')
        if self.assembly_path != 'element_matrix':
            raise ValueError('only ordinary element_matrix is supported; FFT has different formulas')
        if not (self.use_marker is True and self.with_impurities is True and self.with_TiTe is True):
            raise ValueError('requires model502 use_marker, with_impurities and with_TiTe')
        species = {'D2': (2., 1), 'C': (12., 6), 'Ne': (20., 10),
                   'Ar': (40., 18), 'Fe': (56., 26), 'W': (184., 74)}
        if self.impurity not in species or self.nuclear_charge != species[self.impurity][1]:
            raise ValueError('unsupported impurity or inconsistent ADAS nuclear charge')
        vals = (self.central_density, self.central_mass, self.gamma,
                units.n_norm_m3, units.t_norm_si, units.energy_norm_j_m3)
        if not np.isfinite(vals).all() or min(vals) <= 0 or self.gamma <= 1:
            raise ValueError('invalid physical normalization parameters')
        expected = (self.central_density*1e20,
                    np.sqrt(MU_ZERO*self.central_mass*ATOMIC_MASS_UNIT*self.central_density*1e20),
                    1/(MU_ZERO*(self.gamma-1)))
        actual = (units.n_norm_m3, units.t_norm_si, units.energy_norm_j_m3)
        if not np.allclose(actual, expected, rtol=1e-9, atol=0):
            raise ValueError('projection units inconsistent with physical parameters')
        # mod_elt_matrix :885-906; NOT simply central_mass/impurity_mass.
        return self.central_mass/species[self.impurity][0] * (2/2.01455279245)


@dataclass(frozen=True)
class GaussMarkerFeedback:
    values: np.ndarray             # [q,5], frozen shared channel order
    d_s: np.ndarray                # [q,5], reference-coordinate derivatives
    d_t: np.ndarray
    d_phi: np.ndarray              # radians, not divided by R
    raw_deltas: np.ndarray         # [q,5], NOT yet multiplied by dt/dt_prev
    units: ProjectionUnits
    location: str                  # gauss (not node_dof0)
    delta_convention: str          # raw_projection_increment


@dataclass(frozen=True)
class GaussMetric:
    x_s: np.ndarray                # [q], x=R [m], y=Z [m]
    x_t: np.ndarray
    y_s: np.ndarray
    y_t: np.ndarray


@dataclass(frozen=True)
class GaussFluid:
    rho: np.ndarray                # raw native total mass density / rho_norm
    rho_imp: np.ndarray            # raw native impurity MASS density / rho_norm
    te: np.ndarray                 # raw native electron temperature
    rho_corrected: np.ndarray      # upstream corr_neg_dens result, NOT reconstructed
    rho_imp_corrected: np.ndarray
    te_corrected: np.ndarray       # upstream temperature correction result, JU
    convention: str                # model502_native


@dataclass(frozen=True)
class SpatialDerivatives:
    s: np.ndarray
    t: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class NativeMarkerLinearization:
    """Source-initialized zeros, NOT derivatives of the clipped formulas.

    mod_elt_matrix :910-934 initializes these; the marker branch never assigns
    nonzero values. Native resistivity/energy-exchange consumers receive them.
    Keeping these zeros is source conformance, not an unimplemented closure.
    """
    dcharge_dte: np.ndarray
    d2charge_dte2: np.ndarray
    deffective_charge_dte: np.ndarray
    deffective_charge_drho: np.ndarray
    deffective_charge_drho_imp: np.ndarray
    dmean_square_charge_dte: np.ndarray


@dataclass(frozen=True)
class ConsumedMarkerFeedback:
    density: np.ndarray            # floored c5, NOT rho_imp or injection source
    density_derivatives: SpatialDerivatives  # raw projection derivatives per source
    mean_charge_before_cap: np.ndarray
    mean_charge: np.ndarray        # cap then low-Te override, in source order
    mean_square_charge: np.ndarray # Z_eff_imp; no nuclear-charge-squared cap
    charge_derivatives: SpatialDerivatives
    delta_charge: np.ndarray       # quotient linearization BEFORE charge cap/Te override
    alpha_e: np.ndarray
    alpha_e_bis: np.ndarray        # exactly alpha_e: marker dZ/dTe=0
    alpha_e_tri: np.ndarray        # exactly zero in this source branch
    alpha_derivatives: SpatialDerivatives
    delta_alpha_e: np.ndarray
    ne_before_correction: np.ndarray
    ne_jorek: np.ndarray           # exponential correction, not SI floor/n_norm
    ne_si: np.ndarray              # separate SI hard floor 1e16 m^-3
    effective_charge: np.ndarray   # total mixture Z_eff, minimum1, NO upper cap
    electron_pressure: np.ndarray # uses RAW fluid densities/temperature
    energy_rate_jorek: np.ndarray # -(c1+max(c2,0)), no rho/ne multiplier
    energy_rate_si: np.ndarray    # W/m^3
    energy_step_jorek: np.ndarray # rate * normalized fluid dt
    projection_history_jorek: np.ndarray # -delta_alpha_e*raw Te*raw rho_imp; NO zeta/dt
    scaled_projection_deltas: np.ndarray  # fresh assembly-local copy
    native_linearization: NativeMarkerLinearization


def _array(value, shape, name):
    if value is None:
        raise ValueError(f'missing {name}')
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return a


def consume_marker_feedback(feedback: GaussMarkerFeedback, fluid: GaussFluid,
                            metric: GaussMetric, parameters: MarkerFeedbackParameters,
                            time: FluidTimeCoefficients, *, dt_si: float):
    """Consume already evaluated quantities, preserving original arrays.

    Marker delta scaling happens HERE exactly once. dt_si/t_norm is used for
    the direct source only. The assembly callback must not rescale the returned
    projection history or multiply it by zeta. Fluid nodal deltas stay separate.
    Corrected fluid fields must come from the same upstream Gauss evaluation.
    """
    if not isinstance(feedback, GaussMarkerFeedback) or not isinstance(fluid, GaussFluid):
        raise ValueError('explicit Gauss marker and fluid inputs required')
    if not isinstance(metric, GaussMetric) or not isinstance(parameters, MarkerFeedbackParameters):
        raise ValueError('explicit Gauss metric and physical parameters required')
    if feedback.location != 'gauss' or fluid.convention != 'model502_native':
        raise ValueError('requires Gauss evaluation and model502 native fluid units')
    if feedback.delta_convention != 'raw_projection_increment':
        raise ValueError('raw unscaled projection deltas required')
    ratio = parameters.validate(feedback.units)
    if not isinstance(time, FluidTimeCoefficients):
        raise ValueError('FluidTimeCoefficients required')
    if (not np.isfinite([time.theta, time.zeta, time.delta_scale, dt_si]).all()
            or time.delta_scale <= 0 or dt_si <= 0
            or not ((time.theta == .5 and time.zeta == 0)
                    or (time.theta == 1 and (time.zeta == 0 or
                        np.isclose(time.zeta, time.delta_scale/(1+time.delta_scale), rtol=1e-13, atol=0))))):
        raise ValueError('unsupported/inconsistent time coefficients or timestep')
    v = np.asarray(feedback.values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 5 or len(v) == 0:
        raise ValueError('Gauss values must have shape [q,5], q>0')
    q = len(v)
    v = _array(v, (q, 5), 'values')
    ds, dt, dp, raw = [_array(getattr(feedback, name), (q, 5), name)
                       for name in ('d_s', 'd_t', 'd_phi', 'raw_deltas')]
    xs, xt, ys, yt = [_array(getattr(metric, name), (q,), name)
                      for name in ('x_s', 'x_t', 'y_s', 'y_t')]
    jac = xs*yt-xt*ys
    if (jac == 0).any() or not np.isfinite(jac).all():
        raise ValueError('singular/nonfinite Gauss metric')
    rho, ri, te, rc, ric, tec = [_array(getattr(fluid, name), (q,), name)
                                for name in ('rho', 'rho_imp', 'te', 'rho_corrected',
                                             'rho_imp_corrected', 'te_corrected')]
    if (rc <= 0).any() or (ric <= 0).any() or (tec <= 0).any():
        raise ValueError('corrected fluid density and temperature must be positive')
    n = np.maximum(v[:, 4], .5)
    z = np.maximum(v[:, 3], 0)/n
    z2 = np.maximum(v[:, 2], 0)/n
    scaled = raw * time.delta_scale  # allocation; never overwrite persisted data
    dz = (scaled[:, 3]-z*scaled[:, 4])/n
    def gradients(channel):
        return SpatialDerivatives(ds[:, channel].copy(), dt[:, channel].copy(), dp[:, channel].copy(),
                                  (yt*ds[:, channel]-ys*dt[:, channel])/jac,
                                  (-xt*ds[:, channel]+xs*dt[:, channel])/jac)
    dn = gradients(4)
    dq = gradients(3)
    # Deliberately use RAW derivatives even when values were clipped/floored.
    quotient = [(getattr(dq, k)-z*getattr(dn, k))/n for k in ('s', 't', 'phi', 'x', 'y')]
    capped = z > parameters.nuclear_charge  # strict >; equality retains gradients
    dz_grad = SpatialDerivatives(*(np.where(capped, 0., g) for g in quotient))
    z_final = np.minimum(z, parameters.nuclear_charge)
    te_ev = tec/(EL_CHG*MU_ZERO*feedback.units.n_norm_m3)
    z_final = np.where(te_ev < .1, 0., z_final)
    # Source's low-Te override clears Z and Te derivatives, NOT spatial/delta_Z.
    alpha = ratio*z_final-1
    da = SpatialDerivatives(*(ratio*getattr(dz_grad, k) for k in ('s', 't', 'phi', 'x', 'y')))
    delta_alpha = ratio*dz
    ne = rc+alpha*ric
    # corr_neg_dens(ne, (/ .1,.1 /), 1e-3); floor and tail are both 1e-4.
    ne_ju = ne.copy()
    mask = ne < 2e-4
    ne_ju[mask] = 1e-4+1e-4*np.exp((ne[mask]-2e-4)/1e-4)
    ne_si = np.maximum(ne*feedback.units.n_norm_m3, 1e16)
    zeff = np.maximum((z2*n+np.maximum(rc-ric, 0))/ne_ju, 1)
    energy = -(v[:, 0]+np.maximum(v[:, 1], 0))
    history = -delta_alpha*te*ri
    result = ConsumedMarkerFeedback(
        density=n, density_derivatives=dn, mean_charge_before_cap=z,
        mean_charge=z_final, mean_square_charge=z2, charge_derivatives=dz_grad,
        delta_charge=dz, alpha_e=alpha, alpha_e_bis=alpha.copy(), alpha_e_tri=np.zeros(q),
        alpha_derivatives=da, delta_alpha_e=delta_alpha, ne_before_correction=ne,
        ne_jorek=ne_ju, ne_si=ne_si, effective_charge=zeff,
        electron_pressure=(rho+alpha*ri)*te, energy_rate_jorek=energy,
        energy_rate_si=energy*feedback.units.energy_norm_j_m3/feedback.units.t_norm_si,
        energy_step_jorek=energy*(dt_si/feedback.units.t_norm_si),
        projection_history_jorek=history, scaled_projection_deltas=scaled,
        native_linearization=NativeMarkerLinearization(*(np.zeros(q) for _ in range(6))))
    # Finite inputs can still overflow; never return silent invalid closures.
    for name, a in vars(result).items():
        arrays = vars(a).values() if isinstance(a, (SpatialDerivatives, NativeMarkerLinearization)) else (a,)
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError(f'nonfinite consumed feedback {name}')
    return result


def electron_energy_weak_terms(consumed: ConsumedMarkerFeedback, *, test_value,
                               big_r, xjac, quadrature_weight):
    """Two separate equation9 RHS contributions, not a nine-field forcing.

    Values are per basis/Gauss point. Summation, harmonic basis, constraints,
    and LHS remain the assembler's responsibility. xjac is signed as in JOREK.
    """
    shape = consumed.energy_rate_jorek.shape
    v, r, j, w = [_array(a, shape, name) for a, name in
                  ((test_value, 'test_value'), (big_r, 'big_r'),
                   (xjac, 'xjac'), (quadrature_weight, 'quadrature_weight'))]
    if (r <= 0).any() or (j == 0).any() or (w <= 0).any():
        raise ValueError('invalid weak-form geometry/weight')
    measure = v*r*j*w
    return (measure*consumed.energy_step_jorek,
            measure*consumed.projection_history_jorek)
