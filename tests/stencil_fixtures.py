"""Synthetic quadrilateral meshes shared by the B0 stencil/boundary tests.

These are plain helpers (not pytest modules): they build the same
``vertex``/``pos`` contract the real ``jorek*.h5`` frames carry, plus a
collapsed magnetic-axis fan that reproduces JOREK's coincident axis nodes.
"""

from __future__ import annotations

import numpy as np


def structured_mesh(nx: int = 5, nz: int = 5, *, r0: float = 1.0, r1: float = 2.0,
                    z0: float = -0.5, z1: float = 0.5):
    """Cartesian ``(R, Z)`` grid: ``(nx - 1) * (nz - 1)`` quadrilaterals."""

    rs = np.linspace(r0, r1, nx)
    zs = np.linspace(z0, z1, nz)
    pos = np.array([[r, z] for z in zs for r in rs], dtype=np.float64)
    vertex = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n3 = n0 + nx
            n2 = n3 + 1
            vertex.append([n0, n1, n2, n3])
    return np.asarray(vertex, dtype=np.int64).T, pos


def structured_boundary_codes(nx: int = 5, nz: int = 5):
    """Assign one JOREK BC code per node on the structured mesh border.

    bottom -> 1 (target/side 2, inward, Mach-1 sheath), top -> 2 (tangent,
    Dirichlet-all), left -> 20 (fixed corner), right -> 11 (outward target).
    """

    codes = np.zeros(nx * nz, dtype=np.int64)
    for j in range(nz):
        for i in range(nx):
            node = j * nx + i
            if j == 0:
                codes[node] = 1
            elif j == nz - 1:
                codes[node] = 2
            elif i == 0:
                codes[node] = 20
            elif i == nx - 1:
                codes[node] = 11
    return codes


def collapsed_axis_mesh(n_arc: int = 8, radius: float = 1.0):
    """Fan mesh with ``n_arc`` coincident magnetic-axis nodes at ``R = 0``.

    Node ids ``0..n_arc-1`` all sit at ``(0, 0)`` (JOREK ties them to one dof
    set); ids ``n_arc..2*n_arc-1`` lie on the arc.  Each quadrilateral has two
    coincident corners, i.e. exactly the collapsed element JOREK builds at the
    axis (``mod_import_gvec.f90``).
    """

    angles = np.linspace(0.0, 2.0 * np.pi, n_arc, endpoint=False)
    axis = np.zeros((n_arc, 2), dtype=np.float64)
    outer = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    pos = np.concatenate([axis, outer], axis=0)
    vertex = []
    for i in range(n_arc):
        j = (i + 1) % n_arc
        vertex.append([i, j, n_arc + j, n_arc + i])
    return np.asarray(vertex, dtype=np.int64).T, pos


def deterministic_state(n_nodes: int, n_fields: int = 9) -> np.ndarray:
    """Smooth deterministic ``[N, 9]`` field without any RNG."""

    node = np.arange(n_nodes, dtype=np.float64)
    base = np.stack(
        [
            np.sin(0.1 * node + 0.3 * field) + 0.05 * field * np.cos(0.07 * node)
            for field in range(n_fields)
        ],
        axis=1,
    )
    return base


def fine_structured_mesh(nx: int = 9, nz: int = 9, *, r0: float = 1.0, spacing: float = 1e-3):
    """Fine structured mesh with a geometric spectrum of order ``1/h^2 ~ 1e6``.

    The B2 implicit/explicit discussion is about the physical transport
    coefficients (``ZK ~ 4e6``, ``visco ~ 2.4e-3``): only a mesh whose P1
    stiffness reaches the recorded order (B0: ``lambda_max = 5.75e6``) makes
    ``dt = 2/(alpha*lambda)`` land on the documented side of ``dt = 0.05``.
    """

    return structured_mesh(
        nx,
        nz,
        r0=r0,
        r1=r0 + spacing * (nx - 1),
        z0=-0.5 * spacing * (nz - 1),
        z1=0.5 * spacing * (nz - 1),
    )


def confirmed_boundary_codes(nx: int = 9, nz: int = 9):
    """Boundary codes that need no JOREK-side confirmation (0/1/2/5/11).

    bottom -> 1 (inward Mach-1 target), top -> 2 (tangent, Dirichlet-all),
    left -> 5 (side-3 inward target), right -> 11 (side-2 outward target).
    """

    codes = np.zeros(nx * nz, dtype=np.int64)
    for j in range(nz):
        for i in range(nx):
            node = j * nx + i
            if j == 0:
                codes[node] = 1
            elif j == nz - 1:
                codes[node] = 2
            elif i == 0:
                codes[node] = 5
            elif i == nx - 1:
                codes[node] = 11
    return codes


def dirichlet_only_codes(nx: int = 9, nz: int = 9):
    """All wall nodes carry code 2 (Dirichlet-all), no sheath/Mach-1 rows."""

    codes = structured_boundary_codes(nx, nz)
    return np.where(codes != 0, 2, 0).astype(np.int64)


def uniform_b_hat(n_nodes: int, angle: float = 0.7) -> np.ndarray:
    """Uniform unit field direction used by the synthetic parallel operators."""

    return np.tile(np.array([np.cos(angle), np.sin(angle)]), (n_nodes, 1))
