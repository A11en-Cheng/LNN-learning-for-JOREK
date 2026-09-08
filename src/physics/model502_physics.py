"""Model 502 derived-field reconstruction and seven-equation residuals.

``Z_j`` and ``omega`` are derived from ``psi`` and ``u`` via the elliptic
operators ``Z_j = Delta* psi`` and ``omega = Delta u`` (definitions.md §1.3),
so they are reconstructed here rather than predicted.  The residual module
returns reduced residuals for the seven prognostic fields; it is the physics
term that ``JorekLoss`` deliberately lacked.  All operators are graph-local
finite differences via :mod:`src.physics.grad_ops`.
"""

from __future__ import annotations

import torch

from src.physics.grad_ops import compute_divergence, compute_gradient
from src.physics.mhd import get_magnetic_field, get_velocity_field
from src.data.model502_schema import destandardize, inverse_transform_state


def reconstruct_derived(psi, u, pos, edge_index, edge_attr):
    """Return ``(Z_j, omega)`` from ``psi`` and ``u`` via elliptic operators."""
    if psi.dim() == 2 and psi.shape[1] == 1:
        psi = psi.squeeze(1)
    if u.dim() == 2 and u.shape[1] == 1:
        u = u.squeeze(1)
    r = pos[:, 0].clamp_min(1e-8)
    grad_psi = compute_gradient(psi, edge_index, edge_attr)
    flux_weighted = grad_psi / (r * r).unsqueeze(1)
    z_j = (r * r) * compute_divergence(flux_weighted, edge_index, edge_attr)
    grad_u = compute_gradient(u, edge_index, edge_attr)
    omega = compute_divergence(grad_u, edge_index, edge_attr)
    return z_j.unsqueeze(1), omega.unsqueeze(1)


def _scalar(x):
    return x.squeeze(1) if x.dim() == 2 else x


def _poisson(grad_a, grad_b, r):
    return (grad_a[:, 0] * grad_b[:, 1] - grad_a[:, 1] * grad_b[:, 0]) / r


def _coeff(cond, index, default):
    if cond is None or cond.numel() <= index:
        return default
    return float(cond[index].item())


def model502_residuals(state_next, state_curr, graph_data, dt, F0=1.0, stats=None, condition_stats=None):
    """Reduced Model 502 residuals for the seven prognostic fields.

    Channel order follows the canonical 9-field tensor: psi(0), Phi(1), rho(4),
    T_i(5), v_par(6), rho_imp(7), T_e(8).  Returns a dict of scalar L2 norms.
    """
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    pos = graph_data.pos
    r = pos[:, 0].clamp_min(1e-8)

    if stats is not None:
        mean, std = stats
        state_next = inverse_transform_state(destandardize(state_next, mean, std, name="state"))
        state_curr = inverse_transform_state(destandardize(state_curr, mean, std, name="state"))

    condition = getattr(graph_data, "condition", None)
    if condition is not None and condition_stats is not None:
        cmean, cstd = condition_stats
        condition = condition * cstd + cmean
    eta = _coeff(condition, 0, 1e-5)
    visco = _coeff(condition, 1, 1e-5)
    zk_i = _coeff(condition, 4, 1.0)
    zk_e = _coeff(condition, 5, 1.0)
    d_perp = _coeff(condition, 6, 1e-5)
    f0 = _coeff(condition, 33, F0)

    psi = state_curr[:, 0:1]
    u = state_curr[:, 1:2]
    rho = state_curr[:, 4:5]
    rho_imp = state_curr[:, 7:8]
    t_i = state_curr[:, 5:6]
    t_e = state_curr[:, 8:9]
    v_par = state_curr[:, 6:7]
    psi_next = state_next[:, 0:1]
    u_next = state_next[:, 1:2]
    rho_next = state_next[:, 4:5]
    rho_imp_next = state_next[:, 7:8]
    t_i_next = state_next[:, 5:6]
    t_e_next = state_next[:, 8:9]
    v_par_next = state_next[:, 6:7]

    grad_psi = compute_gradient(_scalar(psi), edge_index, edge_attr)
    grad_u = compute_gradient(_scalar(u), edge_index, edge_attr)
    b_field = get_magnetic_field(grad_psi, r.unsqueeze(1), f0)
    v_field = get_velocity_field(grad_u, b_field, v_par, r.unsqueeze(1))
    grad_rho = compute_gradient(_scalar(rho), edge_index, edge_attr)
    grad_ti = compute_gradient(_scalar(t_i), edge_index, edge_attr)
    grad_te = compute_gradient(_scalar(t_e), edge_index, edge_attr)
    grad_rho_imp = compute_gradient(_scalar(rho_imp), edge_index, edge_attr)
    z_j, omega = reconstruct_derived(psi, u, pos, edge_index, edge_attr)
    grad_z_j = compute_gradient(_scalar(z_j), edge_index, edge_attr)
    grad_omega = compute_gradient(_scalar(omega), edge_index, edge_attr)
    pressure = rho * (t_i + t_e) + rho_imp * (t_i + t_e)
    grad_p = compute_gradient(_scalar(pressure), edge_index, edge_attr)
    b_norm = b_field.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b_hat = b_field / b_norm
    r2 = (r * r).unsqueeze(1)

    residuals = {}
    adv_psi = _poisson(grad_u, grad_psi, r).unsqueeze(1)
    residuals["induction"] = float(
        (((psi_next - psi) / dt - eta * z_j + r.unsqueeze(1) * adv_psi).square()).mean().sqrt().item()
    )
    adv_omega = _poisson(grad_omega, grad_u, r).unsqueeze(1)
    curl_j_psi = _poisson(grad_z_j, grad_psi, r).unsqueeze(1)
    curl_p_r2 = (grad_p[:, 1] * 2.0 * r).unsqueeze(1)
    lap_omega = compute_divergence(grad_omega, edge_index, edge_attr).unsqueeze(1)
    residuals["vorticity"] = float(
        ((rho * adv_omega - curl_j_psi + curl_p_r2 + visco * lap_omega).square()).mean().sqrt().item()
    )
    div_rho_v = compute_divergence(rho * v_field[:, 0:2], edge_index, edge_attr).unsqueeze(1)
    diff_rho = d_perp * compute_divergence(grad_rho, edge_index, edge_attr).unsqueeze(1)
    source_rho = torch.zeros_like(rho)
    if hasattr(graph_data, "background_source") and graph_data.background_source is not None:
        source_rho = _scalar(graph_data.background_source[:, 0]).unsqueeze(1)
    residuals["density"] = float(
        ((rho_next - rho) / dt + div_rho_v - diff_rho - source_rho).square().mean().sqrt().item()
    )
    div_v = compute_divergence(v_field[:, 0:2], edge_index, edge_attr).unsqueeze(1)
    div_ti_v = compute_divergence(t_i * v_field[:, 0:2], edge_index, edge_attr).unsqueeze(1)
    cond_i = zk_i * compute_divergence(grad_ti, edge_index, edge_attr).unsqueeze(1)
    residuals["ion_pressure"] = float(
        ((t_i_next - t_i) / dt + div_ti_v + t_i * div_v - cond_i).square().mean().sqrt().item()
    )
    div_te_v = compute_divergence(t_e * v_field[:, 0:2], edge_index, edge_attr).unsqueeze(1)
    cond_e = zk_e * compute_divergence(grad_te, edge_index, edge_attr).unsqueeze(1)
    ohmic = eta * (z_j.squeeze(1) ** 2) / r2.squeeze(1)
    residuals["electron_pressure"] = float(
        ((t_e_next - t_e) / dt + div_te_v + t_e * div_v - cond_e - ohmic.unsqueeze(1)).square().mean().sqrt().item()
    )
    par_grad_p = (b_hat[:, 0:1] * grad_p[:, 0:1] + b_hat[:, 1:2] * grad_p[:, 1:2])
    lap_vpar = compute_divergence(compute_gradient(_scalar(v_par), edge_index, edge_attr), edge_index, edge_attr).unsqueeze(1)
    residuals["parallel_momentum"] = float(
        ((rho_next * v_par_next - rho * v_par) / dt + par_grad_p + rho * v_par * div_v - visco * lap_vpar).square().mean().sqrt().item()
    )
    div_imp_v = compute_divergence(rho_imp * v_field[:, 0:2], edge_index, edge_attr).unsqueeze(1)
    diff_imp = d_perp * compute_divergence(grad_rho_imp, edge_index, edge_attr).unsqueeze(1)
    residuals["impurity"] = float(
        ((rho_imp_next - rho_imp) / dt + div_imp_v - diff_imp).square().mean().sqrt().item()
    )
    residuals["derived_zj_omega_norm"] = float(
        (z_j.square().mean() + omega.square().mean()).sqrt().item()
    )
    return residuals
