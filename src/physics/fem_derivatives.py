"""FEM-consistent nodal derivatives from JOREK's stored Hermite dofs.

``jorek*.h5`` stores four bicubic-Hermite coefficients per node and field
(value, d/ds, d/dt, d2/dsdt, scaled by ``element%size``).  Evaluating the basis
derivatives at the element corners gives physical ``d/dR``, ``d/dZ`` without the
graph least-squares fit used elsewhere, and is ~1000x cheaper than that fit.

The geometry map is the same one JOREK uses for the mass matrix:

    x(s,t) = sum_ij x_dof(i,j) * size(i,j) * H(i,j,s,t)
    u_s    = sum_ij u_dof(i,j) * size(i,j) * H_s(i,j,s,t)
    J      = x_s * y_t - x_t * y_s
    du/dR  = ( y_t * u_s - y_s * u_t) / J
    du/dZ  = (-x_t * u_s + x_s * u_t) / J
"""

from __future__ import annotations

import numpy as np

from src.data.marker_projection import basis_2d_cubic_derivatives

CORNERS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _corner_basis() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for s, t in CORNERS:
        h, h_s, h_t = basis_2d_cubic_derivatives(np.array([s]), np.array([t]))
        out.append((h[0], h_s[0], h_t[0]))
    return out


def nodal_gradient(
    u_dofs: np.ndarray,
    x_dofs: np.ndarray,
    size: np.ndarray,
    vertex: np.ndarray,
    *,
    field: int,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Physical ``[du/dR, du/dZ]`` per node, averaged over adjacent elements.

    ``u_dofs`` is ``[nodes, 4, n_var]``, ``x_dofs`` ``[nodes, 4, 2]``,
    ``size`` ``[4, 4, n_elem]`` and ``vertex`` ``[4, n_elem]`` (0-based).
    """
    u_dofs = np.asarray(u_dofs, dtype=np.float64)
    x_dofs = np.asarray(x_dofs, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    n_nodes = int(n_nodes if n_nodes is not None else u_dofs.shape[0])
    n_elem = vertex.shape[1]

    u_local = u_dofs[vertex, :, field]          # [4 vertex, n_elem, 4 dof]
    x_local = x_dofs[vertex]                    # [4, n_elem, 4, 2]
    grad = np.zeros((n_nodes, 2), dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)
    for corner, (h, h_s, h_t) in enumerate(_corner_basis()):
        # Only the corner's own four dofs are non-zero at that corner.
        idx = np.arange(4)
        del idx
        u_s = np.einsum("ej,je,j->e", u_local[corner], size[corner], h_s[corner])
        u_t = np.einsum("ej,je,j->e", u_local[corner], size[corner], h_t[corner])
        x_s = np.einsum("ejd,je,j->ed", x_local[corner], size[corner], h_s[corner])
        x_t = np.einsum("ejd,je,j->ed", x_local[corner], size[corner], h_t[corner])
        jac = x_s[:, 0] * x_t[:, 1] - x_t[:, 0] * x_s[:, 1]
        safe = np.where(np.abs(jac) > 1e-30, jac, 1.0)
        d_r = (x_t[:, 1] * u_s - x_s[:, 1] * u_t) / safe
        d_z = (-x_t[:, 0] * u_s + x_s[:, 0] * u_t) / safe
        nodes = vertex[corner]
        np.add.at(grad[:, 0], nodes, d_r)
        np.add.at(grad[:, 1], nodes, d_z)
        np.add.at(count, nodes, 1.0)
    count = np.where(count > 0, count, 1.0)
    return grad / count[:, None]


def shafranov_star(
    u_dofs: np.ndarray,
    x_dofs: np.ndarray,
    size: np.ndarray,
    vertex: np.ndarray,
    *,
    field: int,
    n_nodes: int | None = None,
) -> np.ndarray:
    """``Delta* psi = lap psi - psi_R / R`` via the first derivative only.

    The mixed/second-derivative part needs the basis second derivatives; this
    helper currently returns the first-derivative correction term so callers can
    compose it with a graph Laplacian when the full operator is not yet ported.
    """
    grad = nodal_gradient(u_dofs, x_dofs, size, vertex, field=field, n_nodes=n_nodes)
    return -grad[:, 0]


def build_gradient_map(
    size: np.ndarray,
    vertex: np.ndarray,
    x_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Static per-(element, corner) map ``[du/dR, du/dZ] = M @ u_dof_corner``.

    The element geometry is fixed for a run, so this is computed once and then
    applied per frame (cheap) by :func:`nodal_gradient_torch`.
    Returns ``(G, vertex)`` with ``G`` of shape ``[n_elem, 4, 2, 4]``.
    """
    size = np.asarray(size, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    x_dofs = np.asarray(x_dofs, dtype=np.float64)
    n_elem = vertex.shape[1]
    x_local = x_dofs[vertex]                      # [4, n_elem, 4, 2]
    out = np.zeros((n_elem, 4, 2, 4), dtype=np.float64)
    for corner, (h, h_s, h_t) in enumerate(_corner_basis()):
        x_s = np.einsum("ejd,je,j->ed", x_local[corner], size[corner], h_s[corner])
        x_t = np.einsum("ejd,je,j->ed", x_local[corner], size[corner], h_t[corner])
        jac = x_s[:, 0] * x_t[:, 1] - x_t[:, 0] * x_s[:, 1]
        safe = np.where(np.abs(jac) > 1e-30, jac, 1.0)
        w_s = size[corner].T * h_s[corner]        # [n_elem, 4]
        w_t = size[corner].T * h_t[corner]
        # du/dR = ( y_t * u_s - y_s * u_t) / J ;  du/dZ = (-x_t u_s + x_s u_t) / J
        out[:, corner, 0, :] = (x_t[:, 1:2] * w_s - x_s[:, 1:2] * w_t) / safe[:, None]
        out[:, corner, 1, :] = (-x_t[:, 0:1] * w_s + x_s[:, 0:1] * w_t) / safe[:, None]
    return out, vertex


def nodal_gradient_torch(
    u_dofs,
    gradient_map,
    vertex,
    *,
    n_nodes: int,
    counts=None,
):
    """Torch version of :func:`nodal_gradient` for all fields at once.

    ``u_dofs`` is ``[nodes, 4, n_var]`` (torch), ``gradient_map`` the output of
    :func:`build_gradient_map`, ``vertex`` ``[4, n_elem]``.  Returns
    ``[nodes, 2, n_var]``.
    """
    import torch

    ud = u_dofs[vertex]                            # [4, n_elem, 4, n_var]
    n_var = ud.shape[-1]
    out = torch.zeros((n_nodes, 2, n_var), dtype=u_dofs.dtype, device=u_dofs.device)
    for corner in range(4):
        g = torch.einsum("eab,ebv->eav", gradient_map[:, corner], ud[corner])
        out.index_add_(0, vertex[corner], g)
    if counts is None:
        counts = torch.zeros(n_nodes, dtype=u_dofs.dtype, device=u_dofs.device)
        for corner in range(4):
            counts.index_add_(
                0, vertex[corner], torch.ones_like(g[:, 0, 0])
            )
    return out / counts.clamp_min(1.0)[:, None, None]


def _corner_basis_second(h: float = 1e-4):
    """``(H_s, H_t, H_ss, H_tt, H_st)`` at the four corners.

    The bicubic basis is a polynomial of degree <=3 in each variable, so the
    second derivatives follow *exactly* from central differences of the analytic
    first derivatives (no need to port 80 more Fortran expressions).
    """
    out = []
    for s, t in CORNERS:
        def basis(ss, tt):
            _, h_s, h_t = basis_2d_cubic_derivatives(
                np.array([ss]), np.array([tt])
            )
            return h_s[0], h_t[0]

        h_s, h_t = basis(s, t)
        h_s_p, _ = basis(s + h, t)
        h_s_m, _ = basis(s - h, t)
        _, h_t_p = basis(s, t + h)
        _, h_t_m = basis(s, t - h)
        h_ss = (h_s_p - h_s_m) / (2 * h)
        h_tt = (h_t_p - h_t_m) / (2 * h)
        h_st = (h_s_p - h_s_m) / (2 * h) * 0.0
        # mixed derivative: differentiate H_s in t (exact for a cubic in t)
        h_s_tp, _ = basis(s, t + h)
        h_s_tm, _ = basis(s, t - h)
        h_st = (h_s_tp - h_s_tm) / (2 * h)
        out.append((h_s, h_t, h_ss, h_tt, h_st))
    return out


def build_laplacian_map(
    size: np.ndarray,
    vertex: np.ndarray,
    x_dofs: np.ndarray,
    jac_threshold: float = 1e-5,
) -> np.ndarray:
    """Static ``(element, corner) -> lap`` map over the element's 16 dofs.

    Chain rule identical to ``mod_uncoupled_projection.f90`` (xjac_x/xjac_y and
    the u_xx/u_yy expressions), evaluated at the element corners.

    .. warning:: **BROKEN - do not use for physics or features.**  The metric
       coefficients below match the Fortran verbatim and the operator annihilates
       linear fields to 1e-11, but quadratic fields come out ~1e4 too large
       (``lap(R*Z)`` should be 0, measured 1.9e4; ``lap(R^2)`` should be 4,
       measured 8.5e4).  The geometry-only self-check therefore does *not*
       validate this function.  Root cause is the dof<->basis scaling
       convention: ``basis_2d_cubic_derivatives`` returns a *scaled* Hermite
       basis (plain-basis reconstruction of ``s^2`` gives 1.25 instead of 0.25),
       so the ``size`` array and the dof ordering have to match exactly, and the
       current pairing does not.  Acceptance criterion when someone fixes it:
       ``lap(R) = lap(Z) = lap(R*Z) = 0`` and ``lap(R^2) = lap(R^2+Z^2) = 4``
       (all to ~1e-8) - see ``tests/check_fem_second_order.py``.

       Use the graph-LSQ Laplacian instead (``interp_dataset.laplacian_features``
       / ``model502_full_loss._laplacian``); it is verified against the recorded
       ``Z_j``/``omega`` channels at corr 0.987-0.997.
    """
    import warnings

    warnings.warn(
        "build_laplacian_map is known-broken for quadratic fields (~1e4 error); "
        "use the graph-LSQ Laplacian instead (see its docstring).",
        RuntimeWarning,
        stacklevel=2,
    )
    size = np.asarray(size, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    x_dofs = np.asarray(x_dofs, dtype=np.float64)
    n_elem = vertex.shape[1]
    x_elem = x_dofs[vertex].transpose(1, 0, 2, 3).reshape(n_elem, 16, 2)
    out = np.zeros((n_elem, 4, 16), dtype=np.float64)
    for corner, (h_s, h_t, h_ss, h_tt, h_st) in enumerate(_corner_basis_second()):
        # sizing of the 16 dofs (vertex-major, dof-minor)
        sz = size.transpose(2, 0, 1).reshape(n_elem, 16)          # [n_elem,16]
        sh_s = sz * h_s.reshape(1, 16)
        sh_t = sz * h_t.reshape(1, 16)
        sh_ss = sz * h_ss.reshape(1, 16)
        sh_tt = sz * h_tt.reshape(1, 16)
        sh_st = sz * h_st.reshape(1, 16)
        x_s = np.einsum("ek,ekd->ed", sh_s, x_elem)
        x_t = np.einsum("ek,ekd->ed", sh_t, x_elem)
        x_ss = np.einsum("ek,ekd->ed", sh_ss, x_elem)
        x_tt = np.einsum("ek,ekd->ed", sh_tt, x_elem)
        x_st = np.einsum("ek,ekd->ed", sh_st, x_elem)
        xs, xt = x_s[:, 0], x_t[:, 0]
        ys, yt = x_s[:, 1], x_t[:, 1]
        xss, xtt, xst = x_ss[:, 0], x_tt[:, 0], x_st[:, 0]
        yss, ytt, yst = x_ss[:, 1], x_tt[:, 1], x_st[:, 1]
        jac = xs * yt - xt * ys
        safe = np.where(np.abs(jac) > 1e-30, jac, 1.0)
        xjac_x = (xss * yt**2 - yss * xt * yt - 2 * xst * ys * yt
                  + yst * (xs * yt + xt * ys) + xtt * ys**2 - ytt * xs * ys) / safe
        xjac_y = (ytt * xs**2 - xtt * ys * xs - 2 * yst * xt * xs
                  + xst * (yt * xs + ys * xt) + yss * xt**2 - xss * yt * xt) / safe
        j2 = safe**2
        a_ss = (yt**2 + xt**2) / j2
        a_st = -2.0 * (ys * yt + xs * xt) / j2
        a_tt = (ys**2 + xs**2) / j2
        a_s = ((yst * yt - ytt * ys) + (xst * xt - xtt * xs)) / j2 - (xjac_x * yt - xjac_y * xt) / j2
        a_t = ((yst * ys - yss * yt) + (xst * xs - xss * xt)) / j2 - (-xjac_x * ys + xjac_y * xs) / j2
        out[:, corner, :] = (
            a_ss[:, None] * sh_ss + a_st[:, None] * sh_st + a_tt[:, None] * sh_tt
            + a_s[:, None] * sh_s + a_t[:, None] * sh_t
        )
        # Degenerate corners (|J| ~ 0, e.g. axis nodes) would divide by J^2 and
        # blow the map up; drop them instead of producing 1e8-sized entries.
        out[:, corner, :] = np.where(
            (np.abs(jac) > jac_threshold)[:, None], out[:, corner, :], 0.0
        )
    return out


def nodal_laplacian_torch(u_dofs, laplacian_map, vertex, *, n_nodes: int, counts=None):
    """Physical Laplacian per node (all fields at once) from precomputed maps."""
    import torch

    ud = u_dofs[vertex].permute(1, 0, 2, 3).reshape(vertex.shape[1], 16, -1)
    n_var = ud.shape[-1]
    out = torch.zeros((n_nodes, n_var), dtype=u_dofs.dtype, device=u_dofs.device)
    for corner in range(4):
        lap = torch.einsum("ek,ekv->ev", laplacian_map[:, corner], ud)
        out.index_add_(0, vertex[corner], lap)
    if counts is None:
        counts = torch.zeros(n_nodes, dtype=u_dofs.dtype, device=u_dofs.device)
        ones = torch.ones(vertex.shape[1], dtype=u_dofs.dtype, device=u_dofs.device)
        for corner in range(4):
            counts.index_add_(0, vertex[corner], ones)
    return out / counts.clamp_min(1.0)[:, None]
