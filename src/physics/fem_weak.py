"""Weak-form (finite-element) Laplacian on the nodal values.

Motivation (pitfalls #21 / #37 and the term attribution of 2026-09-30): the
physics residuals need a second-derivative operator, but applying a *pointwise*
Laplacian to a decoded nodal field amplifies node-level noise by ~1/h^2 - the
conduction term on the model's field measured 137x its value on the recorded
field, and the physics-loss gradient was ~1e5 times the data gradient.

The weak form replaces "differentiate at each node" by "multiply by a test
function and integrate over the element"::

    (N_i, lap u) = - (grad N_i, grad u)        (Green, natural BCs)
    A_ij = int_elem grad N_i . grad N_j dV     (stiffness, symmetric PSD)
    lap u |_i  ~= - (A u)_i / m_i              (m = lumped mass = row sums of A's mass)

Small node-to-node wiggles average out in the integral, so the operator's
sensitivity to them is O(1) instead of O(1/h^2), and the same matrix is what an
implicit time step would invert.

Discretisation note: JOREK itself solves the *cubic Hermite* weak form (its
state carries the derivative dofs).  Here we build the operator on the nodal
**values** the model actually predicts, using the 4-node bilinear shape
functions of each quadrilateral.  It is a different but consistent
discretisation of the same operator, chosen because (a) it does not need the
derivative dofs and (b) it has the noise-averaging property we are after.
"""

from __future__ import annotations

import numpy as np

# 2x2 Gauss points/weights on the reference square [0,1]^2.
_GAUSS = 0.5 + 0.5 / np.sqrt(3.0) * np.array([-1.0, 1.0])
_GAUSS_W = np.array([0.5, 0.5])


def _gradients(s: float, t: float) -> np.ndarray:
    """``[dN/ds, dN/dt]`` for N1..N4 of the bilinear quadrilateral."""
    return np.array(
        [
            [-(1.0 - t), -(1.0 - s)],   # N1 = (1-s)(1-t)
            [+(1.0 - t), -s],           # N2 = s(1-t)
            [+t, +s],                   # N3 = s t
            [-t, +(1.0 - s)],           # N4 = (1-s) t
        ]
    )


def element_stiffness(coords: np.ndarray):
    """Bilinear element stiffness and lumped mass for one quadrilateral.

    ``coords`` is ``[4, 2]`` (R, Z) in element-node order.  Returns
    ``(A_e[4,4], m_e[4])`` with ``A_e`` the usual symmetric PSD stiffness and
    ``m_e`` the row-sum (lumped) mass, both in physical units.
    """
    a_e = np.zeros((4, 4))
    for gs, ws in zip(_GAUSS, _GAUSS_W):
        for gt, wt in zip(_GAUSS, _GAUSS_W):
            dnds = _gradients(gs, gt)               # [4,2]
            x_s = dnds[:, 0] @ coords               # [2]
            x_t = dnds[:, 1] @ coords               # [2]
            jac = np.array([[x_s[0], x_t[0]], [x_s[1], x_t[1]]])
            det_j = abs(np.linalg.det(jac))
            if det_j < 1e-30:
                continue
            inv = np.linalg.inv(jac)
            # physical gradients: dN/dx = dN/ds * s_x + dN/dt * t_x
            grad = dnds @ inv.T                     # [4,2] -> (dN/dR, dN/dZ)
            weight = ws * wt * det_j
            a_e += weight * (grad @ grad.T)
    # lumped mass: integrate the shape functions (bilinear => analytic per node)
    m_e = _lumped_mass(coords)
    return a_e, m_e


def _lumped_mass(coords: np.ndarray) -> np.ndarray:
    """Row-sum lumped mass ``m_i = int N_i dV`` (exact for a bilinear quad)."""
    m = np.zeros(4)
    for gs, ws in zip(_GAUSS, _GAUSS_W):
        for gt, wt in zip(_GAUSS, _GAUSS_W):
            n = np.array([(1 - gs) * (1 - gt), gs * (1 - gt), gs * gt, (1 - gs) * gt])
            dnds = _gradients(gs, gt)
            x_s = dnds[:, 0] @ coords
            x_t = dnds[:, 1] @ coords
            det_j = abs(x_s[0] * x_t[1] - x_t[0] * x_s[1])
            m += ws * wt * det_j * n
    return m




def triangle_stiffness(coords):
    """P1 (linear) triangle stiffness and lumped mass.

    JOREK represents the axis region as collapsed quadrilaterals whose two
    corners coincide; ``mod_import_gvec.f90:471-478`` keeps the element but sets
    the collapsed direction's derivative dofs to zero (``size(:,3)=size(:,4)=0``).
    The equivalent on the nodal values is a linear triangle over the three
    distinct corners - which is exactly what this returns.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-30:
        return np.zeros((3, 3)), np.zeros(3)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    grad = np.stack([b, c], axis=1) / area2          # [3,2]
    area = abs(area2) / 2.0
    a_e = area * (grad @ grad.T)
    m_e = area / 3.0 * np.ones(3)
    return a_e, m_e


def _min_det_j(coords: np.ndarray) -> float:
    """Smallest Gauss-point Jacobian of a bilinear quadrilateral (<0 if flipped)."""
    smallest = np.inf
    for gs in _GAUSS:
        for gt in _GAUSS:
            dnds = _gradients(gs, gt)
            x_s = dnds[:, 0] @ coords
            x_t = dnds[:, 1] @ coords
            smallest = min(smallest, x_s[0] * x_t[1] - x_t[0] * x_s[1])
    return float(smallest)



def cluster_coincident_nodes(pos: np.ndarray, tol: float = 1e-9):
    """Merge coincident nodes into one degree of freedom.

    JOREK ties every node on the magnetic axis to a single dof set
    (``mod_import_gvec.f90``: ``index(1:4) = 1,2,3,4`` for all axis nodes) and
    zeroes the collapsed direction's derivative dofs.  The collapsed
    quadrilaterals therefore carry no information along the degenerate
    direction.  Our nodal analogue is to give the coincident nodes **one shared
    value**, which is what this clustering does.

    Returns ``(cluster, inverse)``: ``cluster[node]`` is the cluster id and
    ``inverse[c]`` the list of nodes in cluster ``c``.
    """
    keys: dict[tuple, int] = {}
    cluster = np.full(pos.shape[0], -1, dtype=np.int64)
    inverse: list[list[int]] = []
    for index, (r, z) in enumerate(pos):
        key = (round(float(r) / tol), round(float(z) / tol))
        if key not in keys:
            keys[key] = len(inverse)
            inverse.append([])
        cid = keys[key]
        cluster[index] = cid
        inverse[cid].append(index)
    return cluster, inverse



def build_p1_laplacian(vertex: np.ndarray, pos: np.ndarray, cluster=None):
    """P1 (linear) FEM Laplacian on a triangulation of the quadrilateral mesh.

    Each quadrilateral is split into two triangles using its four corner
    coordinates.  This is the piece that makes the operator *exact for linear
    fields*: a bilinear quad only reproduces linear functions when it is a
    parallelogram (our wall grid is not), while a P1 triangle always does.  The
    collapsed quadrilaterals on the magnetic axis lose one triangle to the
    degeneracy check, which is the nodal analogue of JOREK zeroing the collapsed
    direction's derivative dofs (``mod_import_gvec.f90``).

    Returns ``(A, m, cluster)`` with ``A`` symmetric PSD in cluster space and
    ``lap(u) ~= -(A @ u_c) / m``.
    """
    import scipy.sparse as sp

    n_nodes = pos.shape[0]
    if cluster is None:
        cluster, _ = cluster_coincident_nodes(pos)
    else:
        cluster = np.asarray(cluster, dtype=np.int64)
    n_clusters = int(cluster.max()) + 1
    rows, cols, vals = [], [], []
    mass = np.zeros(n_clusters)
    n_elem = vertex.shape[1]
    degenerate = 0
    for e in range(n_elem):
        nodes = cluster[vertex[:, e]]
        coords = pos[vertex[:, e]]
        for tri in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(tri)
            a_e, m_e = triangle_stiffness(coords[idx])
            if not np.any(a_e):
                degenerate += 1
                continue
            used = nodes[idx]
            for i in range(3):
                for j in range(3):
                    rows.append(used[i])
                    cols.append(used[j])
                    vals.append(a_e[i, j])
            mass[used] += m_e
    print(f"[fem_weak] P1 triangles: {n_elem * 2 - degenerate} valid, "
          f"{degenerate} degenerate ({degenerate / max(2 * n_elem, 1):.2%})")
    a = sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()
    return a, mass, cluster




def triangle_parallel_stiffness(coords, b_hat):
    """Parallel (field-aligned) P1 stiffness ``int (b.grad N_i)(b.grad N_j) dV``.

    ``b_hat`` is the unit magnetic field for this element (2 components); the
    operator is the weak form of ``(b.grad)^2``, i.e. the parallel conduction
    ``kappa_par * div_par(grad_par T)`` without the metric terms.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-30:
        return np.zeros((3, 3))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    grad = np.stack([b, c], axis=1) / area2          # [3,2]
    g_par = grad @ b_hat                             # [3]
    area = abs(area2) / 2.0
    return area * np.outer(g_par, g_par)


def build_p1_parallel_laplacian(vertex, pos, b_hat_nodes, cluster=None):
    """Assemble ``(A_par, m, cluster)``; ``b_hat_nodes`` is ``[n_nodes, 2]``."""
    import scipy.sparse as sp

    n_nodes = pos.shape[0]
    if cluster is None:
        cluster, _ = cluster_coincident_nodes(pos)
    else:
        cluster = np.asarray(cluster, dtype=np.int64)
    n_clusters = int(cluster.max()) + 1
    b_nodes = np.asarray(b_hat_nodes, dtype=np.float64)
    rows, cols, vals = [], [], []
    mass = np.zeros(n_clusters)
    n_elem = vertex.shape[1]
    for e in range(n_elem):
        nodes = cluster[vertex[:, e]]
        coords = pos[vertex[:, e]]
        b_e = b_nodes[vertex[:, e]].mean(axis=0)
        norm = np.linalg.norm(b_e)
        if norm < 1e-12:
            continue
        b_e = b_e / norm
        for tri in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(tri)
            a_e = triangle_parallel_stiffness(coords[idx], b_e)
            if not np.any(a_e):
                continue
            _, m_e = triangle_stiffness(coords[idx])
            used = nodes[idx]
            for i in range(3):
                for j in range(3):
                    rows.append(used[i])
                    cols.append(used[j])
                    vals.append(a_e[i, j])
            mass[used] += m_e
    a = sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()
    return a, mass, cluster



def build_p1_advection(vertex, pos, v_nodes, cluster=None, div_v=None):
    """Lumped P1 advection operator for ``A(f) = -div(f v)``.

    Weak form: ``int N_i A(f) = int grad N_i . (f v) - boundary``, so with the
    lumped element average ``f|_e ~ (1/3) sum_j f_j``::

        adv_e[i, j] = area_e / 3 * (v_e . grad N_i)      (same for every j)

    and ``A(f)_i ~ (M_adv f)_i / m_i``.  ``div_v`` (optional, per cluster) adds
    the ``-f div(v)`` part of the energy equations as a diagonal contribution.

    Returns ``(M_adv, m, cluster)``.  The matrix is **not** symmetric, so it is
    meant to be used inside an implicit (backward-Euler) solve, which is the
    stabilisation our explicit centred discretisation needs: centred advection
    has no numerical dissipation and is unconditionally unstable explicitly.
    """
    import scipy.sparse as sp

    n_nodes = pos.shape[0]
    if cluster is None:
        cluster, _ = cluster_coincident_nodes(pos)
    else:
        cluster = np.asarray(cluster, dtype=np.int64)
    n_clusters = int(cluster.max()) + 1
    v_nodes = np.asarray(v_nodes, dtype=np.float64)
    rows, cols, vals = [], [], []
    mass = np.zeros(n_clusters)
    n_elem = vertex.shape[1]
    for e in range(n_elem):
        nodes = cluster[vertex[:, e]]
        coords = pos[vertex[:, e]]
        v_e = v_nodes[vertex[:, e]].mean(axis=0)
        for tri in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(tri)
            x1, y1 = coords[idx[0]]
            x2, y2 = coords[idx[1]]
            x3, y3 = coords[idx[2]]
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(area2) < 1e-30:
                continue
            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            grad = np.stack([b, c], axis=1) / area2      # [3,2]
            area = abs(area2) / 2.0
            gv = grad @ v_e                              # [3] = v . grad N_i
            used = nodes[idx]
            for i in range(3):
                for j in range(3):
                    rows.append(used[i])
                    cols.append(used[j])
                    vals.append(area / 3.0 * gv[i])
            mass[used] += area / 3.0 * np.ones(3)
    m_adv = sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()
    if div_v is not None:
        dv = np.asarray(div_v, dtype=np.float64)
        m_adv = m_adv - sp.diags(dv * np.asarray(
            [mass[c] for c in range(n_clusters)], dtype=np.float64
        ))
    return m_adv, mass, cluster



def build_p1_anisotropic_stiffness(vertex, pos, metric_nodes, cluster=None):
    """``A_ij = int grad N_i . M grad N_j`` with a per-element 2x2 metric.

    Used for JOREK's numerical stabilisation term
    (``mod_elt_matrix.f90``, ``TG_num2``): its weak form is proportional to
    ``(grad w x grad u) (grad v x grad u)``, i.e. an anisotropic diffusion of
    ``omega`` with the metric ``M = [[u_y^2, -u_x u_y], [-u_x u_y, u_x^2]]``
    built from the potential's gradient - a rank-one metric along ``grad u``.
    """
    import scipy.sparse as sp

    n_nodes = pos.shape[0]
    if cluster is None:
        cluster, _ = cluster_coincident_nodes(pos)
    else:
        cluster = np.asarray(cluster, dtype=np.int64)
    n_clusters = int(cluster.max()) + 1
    metric = np.asarray(metric_nodes, dtype=np.float64)
    rows, cols, vals = [], [], []
    for e in range(vertex.shape[1]):
        nodes = cluster[vertex[:, e]]
        coords = pos[vertex[:, e]]
        m_e = metric[vertex[:, e]].mean(axis=0).reshape(2, 2)
        for tri in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(tri)
            x1, y1 = coords[idx[0]]
            x2, y2 = coords[idx[1]]
            x3, y3 = coords[idx[2]]
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(area2) < 1e-30:
                continue
            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            grad = np.stack([b, c], axis=1) / area2
            g = grad @ m_e                                  # [3,2]
            a_e = (abs(area2) / 2.0) * (g @ grad.T)
            used = nodes[idx]
            for i in range(3):
                for j in range(3):
                    rows.append(used[i])
                    cols.append(used[j])
                    vals.append(a_e[i, j])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()


def boundary_mask(vertex: np.ndarray) -> np.ndarray:
    """Nodes attached to fewer elements than the mesh maximum (the wall).

    The weak Laplacian equals the interior Laplacian only up to the boundary
    term ``integral N_i du/dn``, which is nonzero for the fields we differentiate.
    Callers that want the *interior* operator should mask these nodes, which is
    what JOREK's own boundary conditions handle implicitly.
    """
    n_nodes = int(vertex.max()) + 1
    count = np.zeros(n_nodes, dtype=np.int64)
    for corner in range(vertex.shape[0]):
        np.add.at(count, vertex[corner], 1)
    return count < count.max()


def build_nodal_laplacian(vertex: np.ndarray, pos: np.ndarray, cluster=None):
    """Assemble ``(A, m)`` for a mesh given element connectivity and node coords.

    ``A`` is sparse ``[n_clusters, n_clusters]`` (symmetric PSD stiffness) and
    ``m`` the lumped mass ``[n_clusters]``; ``lap(u) ~= -(A @ u_c) / m`` with
    ``u_c`` the per-cluster value (coincident nodes share one dof, as JOREK ties
    the axis nodes).  Returns ``(A, m, cluster)``.
    """
    import scipy.sparse as sp

    n_nodes = pos.shape[0]
    if cluster is None:
        cluster, _ = cluster_coincident_nodes(pos)
    else:
        cluster = np.asarray(cluster, dtype=np.int64)
    n_clusters = int(cluster.max()) + 1
    rows, cols, vals = [], [], []
    mass = np.zeros(n_clusters)
    skipped = 0
    n_elem = vertex.shape[1]
    for e in range(n_elem):
        nodes = cluster[vertex[:, e]]
        if nodes.min() < 0 or nodes.max() >= n_nodes:
            continue
        coords = pos[nodes]
        # Collapsed quadrilaterals (magnetic axis) have coincident corners: JOREK
        # kills the collapsed direction's derivative dofs (mod_import_gvec.f90),
        # the nodal analogue is to treat the element as a linear triangle over
        # the distinct corners.
        keep = [0]
        for i in range(1, 4):
            if all(np.linalg.norm(coords[i] - coords[j]) > 1e-9 for j in keep):
                keep.append(i)
        keep = np.asarray(keep)
        if len(keep) == 3:
            a_e, m_e = triangle_stiffness(coords[keep])
        elif len(keep) == 4:
            span = coords.max(axis=0) - coords.min(axis=0)
            scale = max(span[0] * span[1], 1e-30)
            if scale < 1e-12 or _min_det_j(coords) < 1e-6 * scale:
                skipped += 1
                continue
            a_e, m_e = element_stiffness(coords)
        else:
            skipped += 1
            continue
        used = nodes[keep]
        for i in range(len(keep)):
            for j in range(len(keep)):
                rows.append(used[i])
                cols.append(used[j])
                vals.append(a_e[i, j])
        mass[used] += m_e
    if skipped:
        print(f"[fem_weak] skipped {skipped} elements "
              f"({skipped / max(n_elem, 1):.2%})")
    a = sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()
    return a, mass, cluster
