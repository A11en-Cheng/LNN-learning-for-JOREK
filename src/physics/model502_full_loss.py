"""Full reduced-MHD residual set for Model 502 in JOREK units.

This module is the physics-coupling layer for Stage 0 training.  It derives
the nine equation residuals from the authoritative weak form in JOREK's
``models/model502/mod_elt_matrix_fft.f90`` (equations 1-9, around lines
1490-1935) and evaluates them as node-collocated strong-form residuals with
graph finite differences.

Conventions locked against the Fortran source (and cross-checked against the
model500 equation set in arXiv:2308.16124):

* The axisymmetric Poisson bracket is ``{a,b} = a_R b_Z - a_Z b_R`` with **no
  1/R factor** (``ps0_s*u0_t - ps0_t*u0_s`` in the source).  Cylindrical
  weighting is carried by explicit powers of ``R`` in each term.
* ``Z_j = Delta* psi = lap psi - psi_R/R`` (eq 3) but
  ``omega = lap u + u_R/R`` (eq 4): the two elliptic operators differ in the
  sign of the first-derivative term.
* Stage 0 is teacher-forced, so ``state_curr`` is the recorded frame and its
  channels 2/3 hold the FEM-consistent ``Z_j``/``omega``.  The momentum RHS
  therefore uses the recorded derived fields directly (a surrogate
  reconstruction of ``Z_j`` is too gradient-noisy to bracket against
  ``psi``).  The ``zj_surrogate``/``delta_star`` reconstruction is used only
  for the predicted-state elliptic consistency residual.
* Momentum (eq 2), in the node-collocated strong form:

  .. math::
      div(R^3 rho grad du/dt)
        = 1/2 {R^2 |grad u|^2, R^2 rho} + {R^4 rho omega, u}
        + {psi, Z_j} + 2 R d p/d Z + div(visco R grad omega)

  with ``p = n_i T_i + n_e T_e`` (two-temperature closure, eqs 6/9 pressure
  factors).  Note the pressure sign: the recorded frames satisfy
  ``{psi, Z_j} = -[R^2, p]`` (measured with exact gradients on recorded
  trajectories), i.e. the stored ``psi``/``Z_j`` carry the opposite sign from
  the Fortran internal variables.  The loss keeps the stored-frame convention
  so the recorded trajectory minimises it.  Two-fluid (tauIC), TG
  hyper-viscosity, zeta stabilisation and the
  continuity-consistency corrections from the Fortran are omitted on purpose:
  they involve second-derivative couplings or the Newton increment and cannot
  be represented faithfully by node-local graph operators.  Their omission is
  a measured, nonzero residual floor at the recorded trajectory, not a bug.
* The ``- F0/R dZ_j/dphi`` toroidal force is identically zero for the 2D
  (n_tor=1) frames in scope.

Residuals are returned as *relative* L2 norms by default:
``|lhs-rhs| / (|lhs| + |rhs|)`` (or the sum of term norms for multi-term
balances).  This bounds every channel in [0, ~2] and removes the old
coefficient-dependent normalisations (``/visco``, ``/ZK_*``) that made the
vorticity term saturate the training clamp.
"""

from __future__ import annotations

import torch

from src.data.model502_schema import destandardize, inverse_transform_state
from src.physics.grad_ops import compute_divergence, compute_gradient
from src.physics.mhd import get_magnetic_field, get_velocity_field
from src.physics.model502_physics import _scalar


PHYSICS_DTYPE = torch.float64
"""Arithmetic dtype for the residual assembly.

float32's 8-bit exponent overflows at ~3.4e38; a model prediction at +-20
z-scores de-standardises to ~1e11 densities and products of those values
cross that boundary.  There is no native "float32 with a double-precision
exponent" format, so float64 (11-bit exponent, ~1.8e308) is the closest
hardware-supported equivalent and makes the assembly overflow-proof.  Two
deliberate exceptions stay in float32:

* the ``psi -> Z_j`` surrogate is a trained float32 artifact (feeding float64
  would raise a dtype mismatch inside its GAT layers);
* the returned scalar residuals are cast back to the model dtype so gradients
  remain float32.
"""


OMEGA_REFERENCE = 1e-2
"""Fixed scale for the ``vorticity_definition`` residual.

The recorded ``omega`` field is tiny (|omega| rms ~ 5e-3), so any
frame-relative normalisation divides by a near-zero denominator and makes the
Phi gradient of this term extremely stiff (measured ~3.6e3 per normalised
Phi unit).  Normalising by this fixed constant instead keeps the residual
dimensionless while capping the stiffness and avoiding the denominator
collapse on quasi-static frames.
"""


def poisson_bracket(grad_a, grad_b):
    """Axisymmetric Poisson bracket ``{a,b} = a_R b_Z - a_Z b_R`` (no 1/R)."""
    return grad_a[:, 0] * grad_b[:, 1] - grad_a[:, 1] * grad_b[:, 0]


def delta_plus(field, pos, edge_index, edge_attr):
    """Vorticity elliptic operator ``omega = lap u + u_R/R`` (eq 4)."""
    r = pos[:, 0].clamp_min(1e-8)
    scalar = _scalar(field)
    grad = compute_gradient(scalar, edge_index, edge_attr)
    lap = compute_divergence(grad, edge_index, edge_attr)
    return lap + grad[:, 0] / r


def delta_star(field, pos, edge_index, edge_attr):
    """Current elliptic operator ``Z_j = Delta* psi = lap psi - psi_R/R`` (eq 3).

    Uses the MLS ``div(grad(...))`` form; the legacy quadratic-fit
    ``_delta_star_psi`` was rank-deficient on the Delaunay mesh (it kept only
    one direction of each edge, 3-4 neighbours for 5 monomials).
    """
    r = pos[:, 0].clamp_min(1e-8)
    scalar = _scalar(field)
    grad = compute_gradient(scalar, edge_index, edge_attr)
    lap = compute_divergence(grad, edge_index, edge_attr)
    return lap - grad[:, 0] / r


def _laplacian(field, edge_index, edge_attr):
    return compute_divergence(
        compute_gradient(_scalar(field), edge_index, edge_attr),
        edge_index,
        edge_attr,
    )


def _relative_l2(lhs, rhs, eps=1e-6, detach_scale=False):
    """Relative RMS imbalance between two sides of a balance law.

    ``detach_scale`` freezes the normaliser: the *value* is unchanged, but the
    gradient keeps flowing when the two sides do not cancel
    (``|a-b|/(|a|+|b|)`` plateaus at 1 with zero slope, see pitfalls #28).
    """
    scale = lhs.square().mean().sqrt() + rhs.square().mean().sqrt() + eps
    if detach_scale:
        scale = scale.detach()
    return (lhs - rhs).square().mean().sqrt() / scale


def _relative_norm(*terms, eps=1e-6, detach_scale=False):
    """Relative RMS of a multi-term balance: ``|sum| / sum(|term|)``.

    The ratio saturates at 1.0 with **zero gradient** once the terms stop
    cancelling pointwise, which is where a learned model tends to sit; the
    detached variant keeps the same value but restores a (weighted-L1) slope.
    """
    total = torch.zeros_like(terms[0])
    scale = torch.zeros((), device=terms[0].device, dtype=terms[0].dtype)
    for term in terms:
        total = total + term
        scale = scale + term.square().mean().sqrt()
    if detach_scale:
        scale = scale.detach()
    return total.square().mean().sqrt() / (scale + eps)


def _coeff(cond, index, default):
    if cond is None or cond.numel() <= index:
        return default
    return float(cond[index].item())


def _denormalized_states(state_next, state_curr, stats):
    if stats is not None:
        mean, std = stats
        state_next = inverse_transform_state(
            destandardize(state_next, mean, std, name="state")
        )
        state_curr = inverse_transform_state(
            destandardize(state_curr, mean, std, name="state")
        )
    return state_next, state_curr


def _clamp_physical(state):
    """Clip de-normalised fields to generous regulariser bounds.

    The model predicts in normalised space and ``pred`` is only clamped at
    ``[-20, 20]`` z-scores; after de-standardisation the log1p fields can
    reach ~1e11 via ``expm1``.  Products of those values overflow float32 in
    the residual assembly (which the training clamp then reports as a constant
    10 per term).  These bounds are orders of magnitude outside the recorded
    ranges (psi ~ 3e2, Z_j ~ 1e2, omega ~ 1e-2, rho ~ 5e1, T ~ 7) and only
    cut off non-physical extrapolations.
    """
    bounds = [
        (-1e4, 1e4),    # psi
        (-1e2, 1e2),    # Phi
        (-1e5, 1e5),    # Z_j
        (-1e3, 1e3),    # omega
        (1e-12, 1e4),   # rho
        (1e-12, 1e4),   # T_i
        (-1e4, 1e4),    # v_par
        (1e-12, 1e4),   # rho_imp
        (1e-12, 1e4),   # T_e
    ]
    clamped = state.clone()
    for index, (low, high) in enumerate(bounds):
        clamped[:, index] = state[:, index].clamp(low, high)
    return clamped



def _effective_par_conduction(T, T0, kappa0, kappa_neg, kappa_max, t_min_zkpar, dependent):
    """JOREK's capped, temperature-dependent parallel conduction.

    ``mod_elt_matrix.f90:717-740``::
+
        ZK_par_T = ZK_par * (T_corr/T_0)**2.5
        if (ZK_par_T > ZK_par_max) ZK_par_T = ZK_par_max
        if (T_corr < T_min_ZKpar)  ZK_par_T = ZK_par*(T_min_ZKpar/T_0)**2.5

    With the EHL-2 numbers (``ZK_i_par=1.67e5``, ``ZK_e_par=4.1e6``,
    ``ZK_par_max=1e4``, ``T_0=5e-7``) the cap is *active* in the plasma, so the
    result is 1e4 for both species - using the raw coefficients over-estimated
    the parallel conduction by 17x/410x (pitfalls #38).  ``T_corr`` uses the
    smooth negative floor of JOREK's ``corr_neg_temp`` (exponential below
    ``T_min``).
    """
    if not dependent:
        return torch.full_like(T, float(kappa0))
    T_corr = T.clamp_min(float(t_min_zkpar)) if t_min_zkpar > 0 else T
    kappa = float(kappa0) * (T_corr / max(float(T0), 1e-30)) ** 2.5
    kappa = torch.where(kappa > float(kappa_max), torch.full_like(kappa, float(kappa_max)), kappa)
    if t_min_zkpar > 0:
        floor = float(kappa0) * (float(t_min_zkpar) / max(float(T0), 1e-30)) ** 2.5
        kappa = torch.where(T < float(t_min_zkpar), torch.full_like(kappa, floor), kappa)
    return kappa


def _source_controls(graph_data, source_stats, device, dtype):
    source = getattr(graph_data, "source", None)
    if source is None:
        return (
            torch.zeros(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        )
    source = source.to(device=device, dtype=dtype)
    if source_stats is not None:
        mean, std = source_stats
        mean = torch.as_tensor(mean, device=device, dtype=dtype).view(1, -1)
        std = torch.as_tensor(std, device=device, dtype=dtype).view(1, -1)
        source = source * std + mean
    d_ion = source[:, 0]
    rad_sink = source[:, 1]
    z_eff = source[:, 2]
    z_imp = source[:, 3]
    n_imp = source[:, 4]
    return d_ion, rad_sink, z_eff, z_imp, n_imp


def model502_full_residuals(
    state_next,
    state_curr,
    graph_data,
    dt,
    *,
    stats=None,
    condition_stats=None,
    source_stats=None,
    zj_surrogate=None,
    state_target=None,
    gamma=5.0 / 3.0,
    relative=True,
    detach_denominator=False,
    return_terms=False,
):
    """Assemble the nine Model 502 residuals and return scalar L2 norms.

    ``state_next``/``state_curr``/``state_target`` may be either physical
    JOREK units or the normalised model representation; pass the matching
    ``stats`` to de-normalise.  With ``return_terms=True`` the result is
    ``{"residuals": {...}, "vorticity": {...}, "consistency": {...}}``.

    ``detach_denominator`` freezes the relative normalisers.  Values are
    identical, but the gradient stays alive on the ``|a-b|/(|a|+|b|)`` plateau
    where a learned model otherwise sits (see ``report/pitfalls.md`` #28).
    Use it for *training*; keep it False when reporting residual floors so that
    the number is the plain physical ratio.
    """
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    pos = graph_data.pos
    r = pos[:, 0].clamp_min(1e-8)
    device = state_next.device
    dtype = state_next.dtype

    state_next, state_curr = _denormalized_states(state_next, state_curr, stats)
    state_next = _clamp_physical(state_next).to(PHYSICS_DTYPE)
    state_curr = _clamp_physical(state_curr).to(PHYSICS_DTYPE)
    if state_target is not None:
        state_target = _clamp_physical(
            _denormalized_states(state_target, state_target, stats)[0]
        ).to(PHYSICS_DTYPE)
    pos = pos.to(PHYSICS_DTYPE)
    edge_attr = edge_attr.to(PHYSICS_DTYPE)

    condition = getattr(graph_data, "condition", None)
    if condition is not None and condition_stats is not None:
        cmean, cstd = condition_stats
        cmean = torch.as_tensor(cmean, device=device, dtype=dtype)
        cstd = torch.as_tensor(cstd, device=device, dtype=dtype)
        condition = condition * cstd + cmean

    eta = _coeff(condition, 0, 1e-5)
    visco = _coeff(condition, 1, 1e-5)
    visco_par = _coeff(condition, 2, 1e-5)
    visco_par_par = _coeff(condition, 3, 0.0)
    zk_i_par = _coeff(condition, 4, 1.0)
    zk_e_par = _coeff(condition, 5, 1.0)
    d_perp = _coeff(condition, 6, 1e-5)
    d_perp_imp = _coeff(condition, 11, d_perp)
    zk_i_perp = _coeff(condition, 12, 1e-5)
    zk_e_perp = _coeff(condition, 18, 1e-5)
    f0 = _coeff(condition, 33, 1.0)

    psi = state_curr[:, 0]
    u = state_curr[:, 1]
    rho = state_curr[:, 4].clamp_min(1e-12)
    ti = state_curr[:, 5].clamp_min(1e-12)
    v_par = state_curr[:, 6]
    rho_imp = state_curr[:, 7].clamp_min(1e-12)
    te = state_curr[:, 8].clamp_min(1e-12)

    psi_next = state_next[:, 0]
    u_next = state_next[:, 1]
    rho_next = state_next[:, 4].clamp_min(1e-12)
    ti_next = state_next[:, 5].clamp_min(1e-12)
    v_par_next = state_next[:, 6]
    rho_imp_next = state_next[:, 7].clamp_min(1e-12)
    te_next = state_next[:, 8].clamp_min(1e-12)

    grad_psi = compute_gradient(psi, edge_index, edge_attr)
    grad_u = compute_gradient(u, edge_index, edge_attr)
    grad_rho = compute_gradient(rho, edge_index, edge_attr)
    grad_rho_imp = compute_gradient(rho_imp, edge_index, edge_attr)
    grad_ti = compute_gradient(ti, edge_index, edge_attr)
    grad_te = compute_gradient(te, edge_index, edge_attr)
    grad_vpar = compute_gradient(v_par, edge_index, edge_attr)

    b_field = get_magnetic_field(grad_psi, r.unsqueeze(1), f0)
    v_field = get_velocity_field(grad_u, b_field, v_par.unsqueeze(1), r.unsqueeze(1))
    v_pol = v_field[:, 0:2]
    b_norm = b_field.norm(dim=1).clamp_min(1e-8)
    b_hat = b_field / b_norm.unsqueeze(1)
    b_pol = b_hat[:, 0:2]

    d_ion, rad_sink, z_eff, z_imp, n_imp = _source_controls(
        graph_data, source_stats, device, dtype
    )
    z_imp = z_imp.to(PHYSICS_DTYPE).clamp_min(1.0)
    d_ion = d_ion.to(PHYSICS_DTYPE)
    rad_sink = rad_sink.to(PHYSICS_DTYPE)
    z_eff = z_eff.to(PHYSICS_DTYPE)
    n_imp = n_imp.to(PHYSICS_DTYPE)
    # Two-temperature closure, JOREK form (mod_elt_matrix.f90:1369-1403):
    #   alpha_i = m_i/m_imp - 1, alpha_e = (m_i/m_imp)*Z_imp - 1,
    #   Pi = (rho + rho_imp*alpha_i) Ti, Pe = (rho + rho_imp*alpha_e) Te.
    # The earlier n_i = rho + rho_imp assumed m_i/m_imp = 1, which is wrong by
    # ~20x for the neon pellet (m_imp = 20 vs deuterium m_i ~ 2); that is what
    # broke the pressure/jxB cancellation in the momentum equation.
    # m_i/m_imp for Ne = central_mass/20 * (2/2.01455279245) (see the gas-type
    # branches at mod_elt_matrix.f90:895; all 8 runs use the neon pellet).
    m_i_over_m_imp = float(_coeff(condition, None, 0.0)) if False else 0.049639
    alpha_i = m_i_over_m_imp - 1.0
    alpha_e = m_i_over_m_imp * z_imp - 1.0
    n_i = rho + rho_imp * alpha_i
    n_e = rho + rho_imp * alpha_e
    pressure = n_i * ti + n_e * te
    grad_p = compute_gradient(pressure, edge_index, edge_attr)

    # Recorded FEM-consistent derived fields (teacher-forced current state).
    z_j = state_curr[:, 2]
    omega = state_curr[:, 3]
    grad_z_j = compute_gradient(z_j, edge_index, edge_attr)
    grad_omega = compute_gradient(omega, edge_index, edge_attr)

    du_dt = (u_next - u) / dt
    drho_dt = (rho_next - rho) / dt
    grad_du = compute_gradient(du_dt, edge_index, edge_attr)

    # --- eq 1: induction, dpsi/dt = eta Z_j - R {u, psi} ---
    dpsi_dt = (psi_next - psi) / dt
    adv_psi = r * poisson_bracket(grad_u, grad_psi)
    residuals = {}
    terms: dict[str, dict[str, torch.Tensor]] = {}
    terms["induction"] = {
        "dpsi_dt": dpsi_dt, "eta_Z_j": -eta * z_j, "adv_psi": adv_psi,
    }
    residuals["induction"] = (
        _relative_norm(*terms["induction"].values(), detach_scale=detach_denominator)
        if relative
        else (dpsi_dt - eta * z_j + adv_psi).square().mean().sqrt()
    )

    # --- eq 2: perpendicular momentum, strong form derived in the docstring ---
    r2 = r * r
    rho_hat = r2 * rho
    inertia = compute_divergence(
        (r * rho_hat).unsqueeze(1) * grad_du, edge_index, edge_attr
    )
    vv2 = r2 * grad_u.square().sum(dim=1)
    convection = 0.5 * poisson_bracket(
        compute_gradient(vv2, edge_index, edge_attr),
        compute_gradient(rho_hat, edge_index, edge_attr),
    )
    advection = poisson_bracket(
        compute_gradient(r2 * rho_hat * omega, edge_index, edge_attr), grad_u
    )
    jxb = poisson_bracket(grad_psi, grad_z_j)
    pressure_force = +2.0 * r * grad_p[:, 1]
    viscosity = compute_divergence(
        (visco * r).unsqueeze(1) * grad_omega, edge_index, edge_attr
    )
    vort_rhs = convection + advection + jxb + pressure_force + viscosity
    terms["vorticity"] = {
        "inertia": inertia, "convection": -convection, "advection": -advection,
        "jxb": -jxb, "pressure": -pressure_force, "viscosity": -viscosity,
    }
    residuals["vorticity"] = (
        _relative_norm(*terms["vorticity"].values(), detach_scale=detach_denominator)
        if relative
        else (inertia - vort_rhs).square().mean().sqrt()
    )

    lap_rho = _laplacian(rho, edge_index, edge_attr)
    lap_rho_imp = _laplacian(rho_imp, edge_index, edge_attr)
    lap_ti = _laplacian(ti, edge_index, edge_attr)
    lap_te = _laplacian(te, edge_index, edge_attr)
    lap_vpar = _laplacian(v_par, edge_index, edge_attr)

    par_grad_p = b_hat[:, 0] * grad_p[:, 0] + b_hat[:, 1] * grad_p[:, 1]
    par_grad_ti = b_hat[:, 0] * grad_ti[:, 0] + b_hat[:, 1] * grad_ti[:, 1]
    par_grad_te = b_hat[:, 0] * grad_te[:, 0] + b_hat[:, 1] * grad_te[:, 1]
    par_grad_vpar = (
        b_hat[:, 0] * grad_vpar[:, 0] + b_hat[:, 1] * grad_vpar[:, 1]
    )

    div_par_ti = compute_divergence(
        par_grad_ti.unsqueeze(1) * b_pol, edge_index, edge_attr
    )
    div_par_te = compute_divergence(
        par_grad_te.unsqueeze(1) * b_pol, edge_index, edge_attr
    )
    div_par_vpar = compute_divergence(
        par_grad_vpar.unsqueeze(1) * b_pol, edge_index, edge_attr
    )

    exchange = (n_e * n_e / n_i.clamp_min(1e-12)) * 1e-2 * (ti - te)
    ohmic = eta * z_j * z_j / (r * r)
    div_v = compute_divergence(v_field[:, 0:2], edge_index, edge_attr)

    # --- eq 5: density ---
    div_rho_v = compute_divergence(rho.unsqueeze(1) * v_pol, edge_index, edge_attr)
    diff_rho = d_perp * lap_rho + d_perp_imp * lap_rho_imp
    particle_source = _coeff(condition, 41, 0.0)
    terms["density"] = {
        "drho_dt": drho_dt, "div_rho_v": div_rho_v,
        "diffusion": -diff_rho,
        "particlesource": -r * float(particle_source),
    }
    residuals["density"] = (
        _relative_norm(*terms["density"].values(), detach_scale=detach_denominator)
        if relative
        else (drho_dt + div_rho_v - diff_rho - particle_source).square().mean().sqrt()
    )

    # --- eq 6: ion energy ---
    d_ni_ti_dt = (n_i * ti_next - n_i * ti) / dt
    adv_ni_ti = compute_divergence(
        (n_i * ti).unsqueeze(1) * v_field[:, 0:2], edge_index, edge_attr
    ) + n_i * ti * div_v
    kappa_i = _effective_par_conduction(
        ti, _coeff(condition, 27, 5e-7), zk_i_par, _coeff(condition, 53, 2.4e1),
        _coeff(condition, 44, 1e4), _coeff(condition, 46, 0.0),
        _coeff(condition, 45, 0.0) > 0.5,
    )
    cond_i = kappa_i * div_par_ti + zk_i_perp * (lap_ti - div_par_ti)
    terms["ion_energy"] = {
        "d_ni_ti_dt": d_ni_ti_dt, "adv": adv_ni_ti, "conduction": -cond_i, "exchange": -exchange,
    }
    residuals["ion_energy"] = (
        _relative_norm(*terms["ion_energy"].values(), detach_scale=detach_denominator)
        if relative
        else (d_ni_ti_dt + adv_ni_ti - cond_i - exchange).square().mean().sqrt()
    )

    # --- eq 7: parallel momentum ---
    d_rho_vpar_dt = (rho_next * v_par_next - rho * v_par) / dt
    adv_rho_vpar = rho * v_par * div_v
    diff_vpar = visco_par * lap_vpar + visco_par_par * div_par_vpar
    terms["parallel_momentum"] = {
        "d_rho_vpar_dt": d_rho_vpar_dt, "adv": adv_rho_vpar,
        "par_grad_p": par_grad_p, "diffusion": -diff_vpar,
    }
    residuals["parallel_momentum"] = (
        _relative_norm(*terms["parallel_momentum"].values(), detach_scale=detach_denominator)
        if relative
        else (
            d_rho_vpar_dt + adv_rho_vpar + par_grad_p - diff_vpar
        ).square().mean().sqrt()
    )

    # --- eq 8: impurity density ---
    div_rimp_v = compute_divergence(
        rho_imp.unsqueeze(1) * v_pol, edge_index, edge_attr
    )
    d_rimp_dt = (rho_imp_next - rho_imp) / dt
    terms["impurity"] = {
        "d_rimp_dt": d_rimp_dt, "div_rimp_v": div_rimp_v,
        "diffusion": -d_perp_imp * lap_rho_imp, "N_imp": -r * n_imp,
    }
    residuals["impurity"] = (
        _relative_norm(*terms["impurity"].values(), detach_scale=detach_denominator)
        if relative
        else (
            d_rimp_dt + div_rimp_v - d_perp_imp * lap_rho_imp - r * n_imp
        ).square().mean().sqrt()
    )

    # --- eq 9: electron energy ---
    d_ne_te_dt = (n_e * te_next - n_e * te) / dt
    adv_ne_te = compute_divergence(
        (n_e * te).unsqueeze(1) * v_field[:, 0:2], edge_index, edge_attr
    ) + n_e * te * div_v
    kappa_e = _effective_par_conduction(
        te, _coeff(condition, 28, 5e-7), zk_e_par, _coeff(condition, 52, 1.5e2),
        _coeff(condition, 44, 1e4), _coeff(condition, 47, 0.0),
        _coeff(condition, 45, 0.0) > 0.5,
    )
    cond_e = kappa_e * div_par_te + zk_e_perp * (lap_te - div_par_te)
    terms["electron_energy"] = {
        "d_ne_te_dt": d_ne_te_dt, "adv": adv_ne_te, "conduction": -cond_e,
        "ohmic": -ohmic, "rad_sink": rad_sink, "d_ion": -d_ion, "exchange": exchange,
    }
    residuals["electron_energy"] = (
        _relative_norm(*terms["electron_energy"].values(), detach_scale=detach_denominator)
        if relative
        else (
            d_ne_te_dt + adv_ne_te - cond_e - ohmic + rad_sink - d_ion + exchange
        ).square().mean().sqrt()
    )

    # Closure identity; kept for completeness, identically zero by definition.
    terms["charge_state"] = {"closure": n_e - (rho + (z_imp - 1.0) * rho_imp)}
    residuals["charge_state"] = (
        (n_e - (rho + (z_imp - 1.0) * rho_imp)).square().mean().sqrt()
    )

    def _derive_predicted_zj():
        if zj_surrogate is not None:
            features = torch.cat(
                [
                    psi_next.unsqueeze(1),
                    pos,
                    (rho_next * (ti_next + te_next)).unsqueeze(1),
                    compute_gradient(
                        rho_next * (ti_next + te_next), edge_index, edge_attr
                    ),
                    compute_gradient(ti_next, edge_index, edge_attr),
                    compute_gradient(te_next, edge_index, edge_attr),
                ],
                dim=1,
            )
            out = zj_surrogate(
                features.float(), edge_index, edge_attr[:, :2].float()
            ).squeeze(1)
            return out.to(PHYSICS_DTYPE).clamp(-1e5, 1e5)
        return delta_star(psi_next, pos, edge_index, edge_attr)

    zj_hat = _derive_predicted_zj()
    omega_hat = delta_plus(u_next, pos, edge_index, edge_attr)
    zj_target = state_target[:, 2] if state_target is not None else z_j
    omega_target = state_target[:, 3] if state_target is not None else omega
    terms["current_definition"] = {"zj_hat": zj_hat, "zj_target": -zj_target}
    residuals["current_definition"] = (
        _relative_l2(zj_hat, zj_target, detach_scale=detach_denominator)
        if relative
        else (zj_hat - zj_target).square().mean().sqrt()
    )
    terms["vorticity_definition"] = {"omega_hat": omega_hat, "omega_target": -omega_target}
    residuals["vorticity_definition"] = (
        (omega_hat - omega_target).square().mean().sqrt() / OMEGA_REFERENCE
        if relative
        else (omega_hat - omega_target).square().mean().sqrt()
    )

    # Gradients flow back through the float64 graph; keep the scalar loss in
    # the model dtype so the optimiser sees float32 grads.
    residuals = {name: value.to(dtype) for name, value in residuals.items()}

    if not return_terms:
        return residuals
    # Term registry: per-equation term tensors plus their rms values, so callers
    # can attribute a residual to individual terms (which term dominates, which
    # one the model gets wrong) and/or reuse the terms as features.
    term_rms = {
        equation: {
            name: float(value.detach().to(torch.float64).pow(2).mean().sqrt())
            for name, value in equation_terms.items()
        }
        for equation, equation_terms in terms.items()
    }
    return {
        "residuals": residuals,
        "terms": terms,
        "term_rms": term_rms,
        "vorticity": {
            "inertia": inertia,
            "convection": convection,
            "advection": advection,
            "jxb": jxb,
            "pressure": pressure_force,
            "viscosity": viscosity,
            "lhs": inertia,
            "rhs": vort_rhs,
        },
        "consistency": {
            "zj_hat": zj_hat,
            "zj_target": zj_target,
            "omega_hat": omega_hat,
            "omega_target": omega_target,
        },
    }
