"""Full cubic native basis evaluation for ordinary model502 local assembly.

No node-dof0 slicing, bilinear surrogate, file reads, or fluid solve occurs.
Native array axes are retained at the boundary; q is flattened (plane,s,t).
"""
from dataclasses import dataclass
import numpy as np

from .marker_feedback import (GaussMetric, GaussFluid, GaussMarkerFeedback,
                              ConsumedMarkerFeedback, consume_marker_feedback,
                              electron_energy_weak_terms)


@dataclass(frozen=True)
class NativeElementGeometry:
    coordinates: np.ndarray       # native x: [2,4,1,4] (R/Z,degree,coord-mode,vertex)
    size: np.ndarray              # native local size: [degree,vertex], including signs
    n_order: int


@dataclass(frozen=True)
class GaussRule:
    order: int                    # explicit compile-time GAUSS_ORDER

    def points_weights(self):
        if self.order != 4:
            raise ValueError('unsupported Gauss order')
        return (np.array([.0694318442029735, .3300094782075720,
                          .6699905217924280, .9305681557970265]),
                np.array([.173927422568727, .326072577431273,
                          .326072577431273, .173927422568727]))


@dataclass(frozen=True)
class ToroidalSampling:
    modes: tuple[int, ...]        # explicit native [0,cos n,sin n,...] mode numbers
    n_plane: int
    n_period: int

    def basis(self):
        modes = np.asarray(self.modes)
        if (modes.ndim != 1 or not len(modes) or len(modes)%2 != 1
                or not np.issubdtype(modes.dtype, np.integer) or modes[0] != 0
                or not isinstance(self.n_plane, int) or self.n_plane < 1
                or not isinstance(self.n_period, int) or self.n_period < 1
                or (modes[1:] <= 0).any()
                or not np.array_equal(modes[1::2], modes[2::2])
                or (modes % self.n_period != 0).any()):
            raise ValueError('invalid native toroidal modes/planes/period')
        phi = 2*np.pi*np.arange(self.n_plane)/self.n_plane/self.n_period
        hz = np.ones((self.n_plane, len(modes)))
        hp = np.zeros_like(hz)
        for i in range(1, len(modes), 2):
            n = modes[i]
            hz[:, i], hz[:, i+1] = np.cos(n*phi), np.sin(n*phi)
            hp[:, i], hp[:, i+1] = -n*np.sin(n*phi), n*np.cos(n*phi)
        return phi, hz, hp


@dataclass(frozen=True)
class NativeGaussEvaluation:
    values: np.ndarray            # [q,field], q flattened (plane,s,t)
    d_s: np.ndarray
    d_t: np.ndarray
    d_phi: np.ndarray             # derivative wrt radians, no 1/R
    d_x: np.ndarray               # R derivative [1/m]
    d_y: np.ndarray               # Z derivative [1/m]
    coordinates: np.ndarray      # [q,2], R/Z in metres
    metric: GaussMetric
    jacobian: np.ndarray          # signed, not abs
    test_basis: np.ndarray        # [q,vertex,degree,harmonic], size and HZ included
    weights: np.ndarray           # w_s*w_t; NO toroidal quadrature multiplier
    s: np.ndarray
    t: np.ndarray
    phi: np.ndarray


def _finite(a, shape, name):
    if a is None:
        raise ValueError(f'missing {name}')
    a = np.asarray(a, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return a


def _cubic_basis(s, t):
    def one(x):
        h = np.stack([(1-x)**2*(1+2*x), 3*x*(1-x)**2,
                      x*x*(3-2*x), 3*(1-x)*x*x], axis=1).reshape(-1, 2, 2)
        d = np.stack([6*x*(x-1), 3*(x-1)*(3*x-1),
                      -6*x*(x-1), -3*x*(3*x-2)], axis=1).reshape(-1, 2, 2)
        return h, d
    hs, ds = one(s)
    ht, dt = one(t)
    arrays = [np.empty((len(s), 4, 4)) for _ in range(3)]
    for v, (a, b) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        for j, (c, d) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            arrays[0][:, v, j] = hs[:, a, c]*ht[:, b, d]
            arrays[1][:, v, j] = ds[:, a, c]*ht[:, b, d]
            arrays[2][:, v, j] = hs[:, a, c]*dt[:, b, d]
    return arrays


def evaluate_native_element(geometry: NativeElementGeometry, values, *,
                            rule: GaussRule, toroidal: ToroidalSampling):
    """Evaluate ALL native field degrees/harmonics and axisymmetric geometry.

    values: [field,degree,harmonic,vertex], same axes as a four-node native H5
    selection. Coordinate-mode1 is the ordinary assembly's geometry boundary;
    nonaxisymmetric geometry is rejected rather than silently sliced.
    """
    if not isinstance(geometry, NativeElementGeometry) or geometry.n_order != 3:
        raise ValueError('requires complete native cubic n_order=3 geometry')
    coords = _finite(geometry.coordinates, (2, 4, 1, 4), 'native coordinates')
    size = _finite(geometry.size, (4, 4), 'native element size').T
    if not isinstance(rule, GaussRule) or not isinstance(toroidal, ToroidalSampling):
        raise ValueError('explicit Gauss rule and toroidal sampling required')
    phi, hz, hp = toroidal.basis()
    a = np.asarray(values, dtype=float)
    if a.ndim != 4 or a.shape[0] < 1 or a.shape[1:] != (4, len(toroidal.modes), 4):
        raise ValueError('values require full [field,4,harmonic,4] native layout')
    a = _finite(a, a.shape, 'native values').transpose(3, 1, 2, 0)
    x, w = rule.points_weights()
    pp, s, t = np.meshgrid(np.arange(len(phi)), x, x, indexing='ij')
    pp, s, t = pp.ravel(), s.ravel(), t.ravel()
    h, hs, ht = [v*size for v in _cubic_basis(s, t)]
    basis, bs, bt, bp = [pol[:, :, :, None]*tor[pp, None, None, :]
                         for pol, tor in ((h, hz), (hs, hz), (ht, hz), (h, hp))]
    val, ds, dt, dp = [np.einsum('qvdh,vdhf->qf', b, a) for b in (basis, bs, bt, bp)]
    xy, xy_s, xy_t = [np.einsum('qvd,cdv->qc', b, coords[:, :, 0, :]) for b in (h, hs, ht)]
    xs, ys, xt, yt = xy_s[:, 0], xy_s[:, 1], xy_t[:, 0], xy_t[:, 1]
    jac = xs*yt-xt*ys
    if not np.isfinite(jac).all() or (jac == 0).any() or (xy[:, 0] <= 0).any():
        raise ValueError('singular/nonfinite metric or nonpositive R')
    dx = (yt[:, None]*ds-ys[:, None]*dt)/jac[:, None]
    dy = (-xt[:, None]*ds+xs[:, None]*dt)/jac[:, None]
    if not all(np.isfinite(a).all() for a in (val, ds, dt, dp, dx, dy, xy, basis)):
        raise ValueError('nonfinite evaluated native fields/basis/derivatives')
    weights = np.broadcast_to(w[:, None]*w[None, :], (len(phi), len(x), len(x))).ravel().copy()
    return NativeGaussEvaluation(val, ds, dt, dp, dx, dy, xy,
        GaussMetric(xs, xt, ys, yt), jac, basis, weights, s, t, phi[pp])


@dataclass(frozen=True)
class FluidCorrectionParameters:
    t_min: float                  # native JU, explicitly resolved from run inputs
    te_1: float                   # native JU; numerical profile may override preset


@dataclass(frozen=True)
class NativeMarkerLocalLoads:
    gauss: NativeGaussEvaluation   # all9 raw fluid fields and first derivatives
    feedback: GaussMarkerFeedback # raw evaluated projection deltas retained here
    fluid: GaussFluid             # raw and corrected fields kept separately
    consumed: ConsumedMarkerFeedback
    direct: np.ndarray            # [vertex,degree,harmonic], equation9 only
    projection_history: np.ndarray
    rhs_indices: np.ndarray       # zero-based ordinary full-block RHS row indices


def marker_electron_energy_local_loads(geometry, fluid_values, projection_values,
                                      projection_deltas, *, rule, toroidal, units,
                                      parameters, correction, time, dt_si):
    """Pure local reference chain, no nine-equation/global assembly.

    Full native arrays have [field,4,harmonic,4] axes and native normalization.
    Projection deltas are raw, not dt-scaled. Only consume_marker_feedback
    applies dt/dt_prev; only its direct energy term applies dt_si/t_norm.
    Test functions include signed size and HZ once. Plane contributions sum
    as in the ordinary path, with no dphi/2pi/plane-count normalization.
    """
    if not isinstance(correction, FluidCorrectionParameters):
        raise ValueError('explicit resolved native fluid correction parameters required')
    if not np.isfinite([correction.t_min, correction.te_1]).all() or max(correction.t_min, correction.te_1) <= 0:
        raise ValueError('invalid native temperature correction thresholds')
    g = evaluate_native_element(geometry, fluid_values, rule=rule, toroidal=toroidal)
    p = evaluate_native_element(geometry, projection_values, rule=rule, toroidal=toroidal)
    d = evaluate_native_element(geometry, projection_deltas, rule=rule, toroidal=toroidal)
    if g.values.shape[1] != 9 or p.values.shape[1] != 5 or d.values.shape[1] != 5:
        raise ValueError('requires all9 native fluid fields and all5 projection channels/deltas')
    def corrected(a, low, tail):
        out = a.copy()
        mask = a < low+tail
        out[mask] = low+tail*np.exp((a[mask]-low-tail)/tail)
        return out
    rho, ri, te = g.values[:,4].copy(), g.values[:,7].copy(), g.values[:,8].copy()
    temp_floor = max(correction.t_min, correction.te_1)
    fluid = GaussFluid(rho, ri, te, corrected(rho, 1e-12, 1e-8),
        corrected(ri, 1e-12, 1e-8), corrected(te, temp_floor/2, temp_floor/2), 'model502_native')
    feedback = GaussMarkerFeedback(p.values, p.d_s, p.d_t, p.d_phi, d.values,
                                   units, 'gauss', 'raw_projection_increment')
    consumed = consume_marker_feedback(feedback, fluid, g.metric, parameters, time, dt_si=dt_si)
    shape = g.test_basis.shape[1:]
    direct, history = np.empty(shape), np.empty(shape)
    for index in np.ndindex(shape):
        a, b = electron_energy_weak_terms(consumed, test_value=g.test_basis[(slice(None),)+index],
            big_r=g.coordinates[:,0], xjac=g.jacobian, quadrature_weight=g.weights)
        direct[index], history[index] = np.sum(a), np.sum(b)
    nh = len(toroidal.modes)
    indices = (np.arange(16).reshape(4,4,1)*9*nh + 8*nh + np.arange(nh))
    return NativeMarkerLocalLoads(g, feedback, fluid, consumed, direct, history, indices)
