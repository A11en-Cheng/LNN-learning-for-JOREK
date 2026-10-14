"""JOREK quadrilateral-FEM stencil convolution for the nine Model-502 fields.

Phase B0 component: a sparse local operator built from one frame's
``size`` / ``vertex`` / ``contain_node`` / ``neighbours`` / ``x`` arrays, i.e.
exactly the connectivity JOREK solves on (ADR-0018 decision 5).  The operator
transfers flux with a shared local stencil matmul over the four corners of every
quadrilateral and scatters the result back to the nodes with the FEM lumped-mass
normalisation::

    msg[e, i] = sum_j K[i, j] * x[vertex[j, e]]        (local 4x4 matmul)
    update_i  = - sum_e msg[e, i] / m_i                (lumped-mass flux)
    y         = x + dt * update

The geometry is assembled with the same P1 triangularisation / coincident-node
clustering as :mod:`src.physics.fem_weak` (``cluster_coincident_nodes`` mirrors
JOREK's magnetic-axis dof sharing in ``mod_import_gvec.f90``), so with the
identity kernel the operator *is* the verified weak-form Laplacian:
``update = -A x / m``, with ``A`` symmetric, row-sum zero and therefore
conservative.  The discrete relations ``Z_j ~ Delta* psi`` and
``omega ~ Delta+ Phi`` are exposed (``laplacian``, ``delta_star``,
``delta_plus``) and checked against the recorded channels at the graph-LSQ
convention (corr 0.994/0.997, relL2 0.113/0.080).

Degenerate corners (coincident axis nodes, |J| ~ 0) are clustered/dropped so no
weight can blow up; ``R -> 0`` is guarded in the ``1/R`` metric terms.  Every
layer output is reset by :mod:`src.models.jorek_boundary` when a boundary
context is supplied (Dirichlet held, Neumann one-sided, Mach-1 sheath).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.data.model502_schema import FIELD_NAMES, NUM_STATE_FIELDS
from src.models import jorek_boundary as jb
from src.physics import fem_weak

R_MIN = 1e-8


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class FrameGeometry:
    """Raw per-run (or per-frame) JOREK geometry read from ``jorek*.h5``."""

    path: str
    n_nodes: int
    n_elements: int
    vertex: np.ndarray               # [4, E] 0-based node ids
    pos: np.ndarray                  # [N, 2] (R, Z) node values
    size: np.ndarray | None          # [4, 4, E] Hermite dof scaling
    x_dofs: np.ndarray | None        # [N, 4, 2] geometry dofs
    neighbours: np.ndarray | None    # [4, E] 1-based element ids (-1/0 = none)
    boundary_code: np.ndarray        # [N] JOREK boundary code
    freeboundary: bool
    axis_node: np.ndarray            # [N] bool
    contain_node: np.ndarray | None  # [k, E] optional cross-check
    contain_node_matches_vertex: bool | None = None
    values: np.ndarray | None = None      # [N, 4, 9] state dofs (optional)
    deltas: np.ndarray | None = None      # [N, 4, 9] state deltas (optional)

    @property
    def node_values(self) -> np.ndarray | None:
        """Value dof of all nine channels ``[N, 9]`` (None when not loaded)."""

        if self.values is None:
            return None
        return self.values[:, 0, :]

    def stencil(self, *, cluster_tol: float = 1e-9) -> "FemStencilGeometry":
        return build_fem_stencil_geometry(
            self.vertex, self.pos, cluster_tol=cluster_tol, neighbours=self.neighbours
        )

    def boundary(self, *, psi_gradient=None, no_mach1_bc: bool = False) -> jb.BoundaryContext:
        return jb.build_boundary_context(
            self.boundary_code,
            self.pos,
            self.vertex,
            freeboundary=self.freeboundary,
            axis_node=self.axis_node,
            psi_gradient=psi_gradient,
            no_mach1_bc=no_mach1_bc,
        )


def load_frame_geometry(h5_path: str | Path, *, with_values: bool = False) -> FrameGeometry:
    """Read the B0 geometry fields from one ``jorek*.h5`` frame."""

    import h5py

    path = Path(h5_path)
    with h5py.File(path, "r") as handle:
        vertex = np.asarray(handle["vertex"][()], dtype=np.int64) - 1
        x = np.asarray(handle["x"][()], dtype=np.float64)
        if x.ndim == 4:
            if x.shape[2] != 1:
                raise ValueError(
                    f"2D-only contract: 'x' harmonic dimension must be 1, got {x.shape[2]}"
                )
            pos = x[:, 0, 0, :].T
            x_dofs = np.transpose(x[:, :, 0, :], (2, 1, 0))
        elif x.ndim == 3:
            pos = x[:, 0, :].T
            x_dofs = np.transpose(x, (2, 1, 0))
        else:
            pos = x.T
            x_dofs = None
        size = np.asarray(handle["size"][()], dtype=np.float64) if "size" in handle else None
        neighbours = (
            np.asarray(handle["neighbours"][()], dtype=np.int64) if "neighbours" in handle else None
        )
        contain_node = (
            np.asarray(handle["contain_node"][()], dtype=np.int64)
            if "contain_node" in handle
            else None
        )
        contain_node_matches = None
        if contain_node is not None and np.any(contain_node):
            # When the frame actually stores the element->node table, it must
            # agree with ``vertex`` (JOREK writes both from the same element
            # list).  Record the outcome instead of failing: some export
            # variants write ``contain_node`` in a different (padded) layout.
            stored = contain_node[:4].astype(np.int64)
            if stored.shape == vertex.shape and np.all(stored > 0):
                contain_node_matches = bool(
                    np.array_equal(stored - 1, vertex) or np.array_equal(stored, vertex)
                )
        boundary_code = np.asarray(handle["boundary"][()], dtype=np.int64).reshape(-1)
        freeboundary = jb._parse_logical(handle["freeboundary"][()]) if "freeboundary" in handle else False
        axis_node = np.zeros(pos.shape[0], dtype=bool)
        if "axis_node" in handle:
            raw = np.asarray(handle["axis_node"][()]).reshape(-1)
            if raw.dtype.kind in {"S", "U"}:
                axis_node = np.array([str(item).strip().upper() in {"T", "TRUE", "1"} for item in raw])
            else:
                axis_node = raw.astype(bool)
        values = None
        deltas = None
        if with_values:
            raw_values = np.asarray(handle["values"][()], dtype=np.float64)
            if raw_values.ndim != 4:
                raise ValueError(f"values must be [n_var, 4, harmonics, nodes], got {raw_values.shape}")
            if raw_values.shape[2] != 1:
                raise ValueError(
                    f"2D-only contract: 'values' harmonic dimension must be 1, got {raw_values.shape[2]}"
                )
            values = np.transpose(raw_values[:, :, 0, :], (2, 1, 0))  # [N, 4, 9]
            if "deltas" in handle:
                deltas = np.transpose(
                    np.asarray(handle["deltas"][()], dtype=np.float64)[:, :, 0, :], (2, 1, 0)
                )
    return FrameGeometry(
        path=str(path),
        n_nodes=int(pos.shape[0]),
        n_elements=int(vertex.shape[1]),
        vertex=vertex,
        pos=pos,
        size=size,
        x_dofs=x_dofs,
        neighbours=neighbours,
        boundary_code=boundary_code,
        freeboundary=bool(freeboundary),
        axis_node=axis_node,
        contain_node=contain_node,
        contain_node_matches_vertex=contain_node_matches,
        values=values,
        deltas=deltas,
    )


@dataclass
class FemStencilGeometry:
    """Assembled FEM stencil: element kernels + lumped mass + cluster map."""

    vertex: np.ndarray               # [4, E] original 0-based ids
    vertex_cluster: np.ndarray       # [4, E] cluster ids
    cluster: np.ndarray              # [N] node -> cluster
    cluster_inverse: np.ndarray      # [C] cluster -> representative node
    element_kernel: np.ndarray       # [E, 4, 4] P1 stiffness embedded per quad
    element_mass: np.ndarray         # [E, 4]
    stiffness: sp.csr_matrix         # [C, C] assembled stiffness
    mass: np.ndarray                 # [C] lumped mass
    degenerate: np.ndarray           # [E] bool
    pos: np.ndarray | None = None    # [N, 2] (R, Z) node coordinates (B2 term assembly)
    neighbours: np.ndarray | None = None
    radius: np.ndarray | None = None
    _torch_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_nodes(self) -> int:
        return int(self.cluster.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.vertex.shape[1])

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_inverse.shape[0])

    # -- numpy convenience -------------------------------------------------
    def node_mass(self) -> np.ndarray:
        """Lumped mass lifted back to node space (axis duplicates share it)."""

        return self.mass[self.cluster]

    def to_cluster_numpy(self, values: np.ndarray) -> np.ndarray:
        """Node values ``[N]``/``[N, C]`` -> cluster values ``[C]``/``[C, C]``.

        Coincident nodes (JOREK's magnetic-axis dof sharing) are averaged; for
        identical duplicates this is the exact value they carry.
        """

        values = np.asarray(values, dtype=np.float64)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"values must be [N] or [N, C], got {values.shape}")
        x = np.zeros((self.n_clusters, values.shape[1]), dtype=np.float64)
        np.add.at(x, self.cluster, values)
        counts = np.bincount(self.cluster, minlength=self.n_clusters).astype(np.float64)
        x = x / np.maximum(counts[:, None], 1.0)
        return x[:, 0] if squeeze else x

    def from_cluster_numpy(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values[self.cluster]

    def to_cluster(self, values: torch.Tensor) -> torch.Tensor:
        """Torch version of :meth:`to_cluster_numpy` (keeps the autograd graph)."""

        values = torch.as_tensor(values)
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(1)
        if values.ndim != 2:
            raise ValueError(f"values must be [N] or [N, C], got {tuple(values.shape)}")
        buffers = self.torch_buffers(values.device, values.dtype)
        cluster = buffers["cluster"]
        counts = torch.bincount(cluster, minlength=self.n_clusters).clamp_min(1).to(values.dtype)
        x = torch.zeros(
            (self.n_clusters, values.shape[1]), dtype=values.dtype, device=values.device
        ).index_add(0, cluster, values) / counts[:, None]
        return x[:, 0] if squeeze else x

    def from_cluster(self, values: torch.Tensor) -> torch.Tensor:
        buffers = self.torch_buffers(values.device, values.dtype)
        return values.index_select(0, buffers["cluster"])

    def laplacian_numpy(self, values: np.ndarray) -> np.ndarray:
        x = self.to_cluster_numpy(values)
        lap = -(self.stiffness @ x)
        if x.ndim == 1:
            lap = lap / np.maximum(self.mass, 1e-30)
            return lap[self.cluster]
        lap = lap / np.maximum(self.mass[:, None], 1e-30)
        return lap[self.cluster]

    # -- torch operators ---------------------------------------------------
    def torch_buffers(self, device=None, dtype=torch.float64) -> dict:
        key = (str(device), dtype)
        if key not in self._torch_cache:
            coo = self.stiffness.tocoo()
            indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
            values = torch.as_tensor(coo.data, dtype=dtype, device=device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
                stiffness = torch.sparse_coo_tensor(
                    indices, values, coo.shape, device=device, check_invariants=False
                ).coalesce()
            self._torch_cache[key] = {
                "stiffness": stiffness,
                "mass": torch.as_tensor(self.mass, dtype=dtype, device=device),
                "cluster": torch.as_tensor(self.cluster, dtype=torch.long, device=device),
                "vertex_cluster": torch.as_tensor(self.vertex_cluster, dtype=torch.long, device=device),
                "element_kernel": torch.as_tensor(self.element_kernel, dtype=dtype, device=device),
            }
        return self._torch_cache[key]

    def laplacian(self, values: torch.Tensor) -> torch.Tensor:
        """Weak-form Laplacian ``-(A u)/m`` in node space (constant-free)."""

        values = torch.as_tensor(values)
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(1)
        if values.ndim != 2:
            raise ValueError(f"laplacian expects [n_nodes] or [n_nodes, C], got {tuple(values.shape)}")
        buffers = self.torch_buffers(values.device, values.dtype)
        x = self.to_cluster(values)
        lap = -torch.sparse.mm(buffers["stiffness"], x) / buffers["mass"].clamp_min(1e-30)[:, None]
        out = self.from_cluster(lap)
        return out.squeeze(1) if squeeze else out

    def delta_star(self, psi: torch.Tensor, psi_r: torch.Tensor) -> torch.Tensor:
        """``Z_j ~ Delta* psi = lap psi - psi_R / R`` (eq. 3 convention)."""

        radius = self._radius_torch(psi.device, psi.dtype)
        return self.laplacian(psi) - psi_r / radius

    def delta_plus(self, phi: torch.Tensor, phi_r: torch.Tensor) -> torch.Tensor:
        """``omega ~ Delta+ Phi = lap Phi + Phi_R / R`` (eq. 4 convention)."""

        radius = self._radius_torch(phi.device, phi.dtype)
        return self.laplacian(phi) + phi_r / radius

    def _radius_torch(self, device, dtype):
        if self.radius is None:
            raise ValueError("geometry has no radius; rebuild it from a frame")
        key = ("radius", str(device), dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.as_tensor(
                np.clip(self.radius, R_MIN, None), dtype=dtype, device=device
            )
        return self._torch_cache[key]

    def stability_limit(self) -> float:
        """Explicit-Euler stability bound for ``d x/dt = -A x / m``.

        Gershgorin on ``M^{-1} A`` gives ``|lambda| <= max_i sum_j |A_ij| / m_i``,
        so ``dt <= 2 / lambda_max`` keeps the flux update non-expansive.  JOREK
        solves this operator implicitly; the bound is what an explicit stencil
        layer must respect when it is rolled out layer by layer.
        """

        abs_row = np.asarray(abs(self.stiffness).sum(axis=1)).ravel()
        scale = abs_row / np.maximum(self.mass, 1e-30)
        lam_max = float(scale.max()) if scale.size else 0.0
        return 2.0 / max(lam_max, 1e-30)


def build_fem_stencil_geometry(
    vertex: np.ndarray,
    pos: np.ndarray,
    *,
    cluster_tol: float = 1e-9,
    neighbours: np.ndarray | None = None,
) -> FemStencilGeometry:
    """Assemble the P1 stiffness / lumped mass and the per-element 4x4 kernels.

    Mirrors ``src.physics.fem_weak.build_p1_laplacian``: each quadrilateral is
    split into the two triangles ``(0, 1, 2)`` / ``(0, 2, 3)``; coincident nodes
    are merged so collapsed axis quads collapse to a triangle instead of a
    zero-mass row; triangles with ``|J|`` below the degeneracy check are
    dropped.  The result is verified equal to the fem_weak assembly in
    ``tests/test_stencil_operator.py``.
    """

    vertex = np.asarray(vertex, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.float64)
    if vertex.ndim != 2 or vertex.shape[0] != 4:
        raise ValueError(f"vertex must be [4, n_elements], got {vertex.shape}")
    if vertex.min() < 0 or vertex.max() >= pos.shape[0]:
        raise ValueError("vertex must hold 0-based node ids inside [0, n_nodes)")

    cluster, _ = fem_weak.cluster_coincident_nodes(pos, tol=cluster_tol)
    n_clusters = int(cluster.max()) + 1
    cluster_inverse = np.array(
        [np.flatnonzero(cluster == c)[0] for c in range(n_clusters)], dtype=np.int64
    )
    vertex_cluster = cluster[vertex]

    n_elem = vertex.shape[1]
    element_kernel = np.zeros((n_elem, 4, 4), dtype=np.float64)
    element_mass = np.zeros((n_elem, 4), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    mass = np.zeros(n_clusters, dtype=np.float64)
    degenerate = np.zeros(n_elem, dtype=bool)

    for element in range(n_elem):
        local_nodes = vertex_cluster[:, element]
        coords = pos[vertex[:, element]]  # geometry from the node values
        valid = False
        for triangle in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(triangle)
            a_e, m_e = fem_weak.triangle_stiffness(coords[idx])
            if not np.any(a_e):
                continue
            valid = True
            for i in range(3):
                for j in range(3):
                    element_kernel[element, idx[i], idx[j]] += a_e[i, j]
                    rows.append(int(local_nodes[idx[i]]))
                    cols.append(int(local_nodes[idx[j]]))
                    vals.append(float(a_e[i, j]))
            element_mass[element, idx] += m_e
            np.add.at(mass, local_nodes[idx], m_e)
        degenerate[element] = not valid

    stiffness = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_clusters, n_clusters), dtype=np.float64
    ).tocsr()
    mass = np.maximum(mass, 0.0)
    return FemStencilGeometry(
        vertex=vertex,
        vertex_cluster=vertex_cluster,
        cluster=cluster,
        cluster_inverse=cluster_inverse,
        element_kernel=element_kernel,
        element_mass=element_mass,
        stiffness=stiffness,
        mass=mass,
        degenerate=degenerate,
        pos=pos.copy(),
        neighbours=None if neighbours is None else np.asarray(neighbours),
        radius=pos[:, 0].copy(),
    )


# ---------------------------------------------------------------------------
# Learnable stencil-conv layer
# ---------------------------------------------------------------------------
def _identity_delta(device, dtype):
    return torch.zeros((4, 4), device=device, dtype=dtype)


class StencilConv(nn.Module):
    """One stencil-conv layer over the JOREK quadrilateral adjacency.

    ``y = x + dt * (flux exchange)`` where the flux exchange is a shared local
    matmul over each element's four corners, weighted by the assembled P1
    element stiffness and normalised by the lumped mass.  With the default
    deterministic initialisation (``kernel_delta = 0``, ``gain = 1``) the layer
    reproduces ``-dt * A x / m`` exactly: conservative, constant-preserving and
    matching the verified weak-form operator.  ``learnable=True`` exposes
    ``kernel_delta`` (a shared 4x4 stencil perturbation) and ``gain`` (per
    channel flux scale) for the Phase B2 model; no random initialisation is
    used anywhere.

    When ``boundary`` is given, every layer output is reset through
    :func:`src.models.jorek_boundary.apply_boundary_conditions` (roadmap_v2
    §5.10): BC-coded boundary rows are not written by the interior stencil and
    are then assigned their BC-consistent values (Dirichlet held, Neumann
    one-sided, Mach-1 sheath).
    """

    def __init__(
        self,
        geometry: FemStencilGeometry,
        *,
        n_channels: int = NUM_STATE_FIELDS,
        dt: float = 0.05,
        learnable: bool = True,
        mix_channels: bool = False,
        boundary: jb.BoundaryContext | None = None,
        gamma: float = 5.0 / 3.0,
        c_angle: float = 0.0174524,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.n_channels = int(n_channels)
        self.dt = float(dt)
        # Explicit layers must respect the diffusion stability bound (JOREK
        # itself solves this operator implicitly).  The cap only limits the
        # transient step; the fixed point (steady state) is unchanged.
        self.stability_limit = geometry.stability_limit()
        self.dt_effective = min(self.dt, 0.5 * self.stability_limit)
        self.dt_capped = bool(self.dt_effective < self.dt)
        self.boundary = boundary
        self.gamma = float(gamma)
        self.c_angle = float(c_angle)

        buffers = geometry.torch_buffers(torch.device("cpu"), torch.float64)
        self.register_buffer("element_kernel", buffers["element_kernel"].clone())
        self.register_buffer("mass", buffers["mass"].clone())
        self.register_buffer("cluster", buffers["cluster"].clone())
        self.register_buffer("vertex_cluster", buffers["vertex_cluster"].clone())
        node_mass = torch.as_tensor(geometry.node_mass(), dtype=torch.float64)
        self.register_buffer("node_mass", node_mass)
        if boundary is None:
            boundary_rows = np.zeros(geometry.n_nodes, dtype=bool)
        else:
            boundary_rows = boundary.bc_node_mask
        self.register_buffer("boundary_rows", torch.as_tensor(boundary_rows))

        self.kernel_delta = nn.Parameter(_identity_delta(torch.device("cpu"), torch.float64), requires_grad=learnable)
        self.gain = nn.Parameter(torch.ones(self.n_channels, dtype=torch.float64), requires_grad=learnable)
        self.mix_channels = bool(mix_channels)
        if self.mix_channels:
            self.channel_mix = nn.Parameter(torch.eye(self.n_channels, dtype=torch.float64), requires_grad=learnable)
        else:
            self.register_parameter("channel_mix", None)

    # -- helpers -----------------------------------------------------------
    def _kernel(self) -> torch.Tensor:
        return self.element_kernel * (1.0 + self.kernel_delta.unsqueeze(0))

    def _element_flux(self, x_cluster: torch.Tensor) -> torch.Tensor:
        kernel = self._kernel()
        # ``vertex_cluster`` is [4, E]; flattening it row-major would group the
        # *same corner* of four consecutive elements.  The local matmul needs
        # the element-major order that matches ``message.reshape(E, 4, C)``.
        flat = self.vertex_cluster.transpose(0, 1).reshape(-1)
        local = x_cluster.index_select(0, flat).reshape(
            self.geometry.n_elements, 4, self.n_channels
        )
        message = torch.einsum("eij,ejc->eic", kernel, local)
        flat = message.reshape(-1, self.n_channels)
        acc = torch.zeros(
            (self.geometry.n_clusters, self.n_channels), dtype=x_cluster.dtype, device=x_cluster.device
        )
        acc = acc.index_add(0, self.vertex_cluster.transpose(0, 1).reshape(-1), flat)
        return acc * self.gain.unsqueeze(0)

    def flux(self, values: torch.Tensor) -> torch.Tensor:
        """Raw flux exchange ``-A x / m`` in node space (no dt, no BC)."""

        values = torch.as_tensor(values)
        x_cluster = self.geometry.to_cluster(values)
        acc = self._element_flux(x_cluster)
        exchange = -acc / self.mass.clamp_min(1e-30).unsqueeze(1)
        return self.geometry.from_cluster(exchange)

    # -- forward -----------------------------------------------------------
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(
                f"stencil-conv expects [n_nodes, {self.n_channels}], got {tuple(values.shape)}"
            )
        if values.shape[0] != self.geometry.n_nodes:
            raise ValueError(
                f"stencil-conv geometry has {self.geometry.n_nodes} nodes, got {values.shape[0]}"
            )
        if self.boundary is not None:
            # A boundary value (or NaN) must not leak into the interior update:
            # BC rows enter the stencil already sanitised and BC-consistent.
            values = jb.sanitize_boundary_inputs(
                values, self.boundary, gamma=self.gamma
            )
        x_cluster = self.geometry.to_cluster(values)
        if self.channel_mix is not None:
            x_cluster = x_cluster @ self.channel_mix
        acc = self._element_flux(x_cluster)
        update = -acc / self.mass.clamp_min(1e-30).unsqueeze(1)
        out_cluster = x_cluster + self.dt_effective * update
        out = self.geometry.from_cluster(out_cluster)
        # Boundary rows are not written by the interior stencil (roadmap_v2
        # §5.10); the BC layer below assigns them.
        if bool(self.boundary_rows.any()):
            out = torch.where(self.boundary_rows.unsqueeze(1), values, out)
        if self.boundary is not None:
            out = jb.apply_boundary_conditions(
                out,
                self.boundary,
                reference=values,
                dt=self.dt_effective,
                gamma=self.gamma,
                c_angle=self.c_angle,
                mass=self.node_mass,
            )
        return out


class StencilConvStack(nn.Module):
    """``n_layers`` identical stencil-conv layers (B2 rollout helper/tests)."""

    def __init__(self, layer: StencilConv, n_layers: int) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def gradient_map_from_frame(geometry: FrameGeometry):
    """Static Hermite-dof gradient map ``[E, 4, 2, 4]`` (JOREK convention)."""

    if geometry.size is None or geometry.x_dofs is None:
        raise ValueError("frame geometry has no size/x_dofs")
    from src.physics.fem_derivatives import build_gradient_map

    gradient_map, vertex = build_gradient_map(geometry.size, geometry.vertex, geometry.x_dofs)
    return gradient_map, vertex


def nodal_gradient(geometry: FrameGeometry, field: int = FIELD_NAMES.index("psi")) -> np.ndarray:
    """First-order physical gradient ``[N, 2]`` from the frame's Hermite dofs."""

    if geometry.x_dofs is None or geometry.size is None or geometry.values is None:
        raise ValueError("frame geometry needs size/x_dofs/values for a dof gradient")
    from src.physics.fem_derivatives import nodal_gradient as _nodal_gradient

    return _nodal_gradient(
        geometry.values, geometry.x_dofs, geometry.size, geometry.vertex, field=field
    )


def geometry_report(geometry: FemStencilGeometry) -> dict:
    """Checks that tie the assembled operator back to JOREK's arrays."""

    row_sums = np.asarray(geometry.element_kernel.sum(axis=2))
    valid = ~geometry.degenerate
    kernel_row_sum = float(np.abs(row_sums[valid]).max()) if valid.any() else 0.0
    symmetric = float(
        np.abs(geometry.element_kernel - geometry.element_kernel.transpose(0, 2, 1))[valid].max()
    ) if valid.any() else 0.0
    stiffness_symmetric = float(
        abs(geometry.stiffness - geometry.stiffness.T).max()
    ) if geometry.stiffness.nnz else 0.0
    report = {
        "n_nodes": geometry.n_nodes,
        "n_elements": geometry.n_elements,
        "n_clusters": geometry.n_clusters,
        "degenerate_elements": int(geometry.degenerate.sum()),
        "mass_min": float(geometry.mass.min()),
        "mass_nonpositive": int((geometry.mass <= 0).sum()),
        "stiffness_nnz": int(geometry.stiffness.nnz),
        "element_kernel_rowsum_max": kernel_row_sum,
        "element_kernel_symmetry": symmetric,
        "stiffness_symmetry": stiffness_symmetric,
        "vertex_in_range": bool(geometry.vertex.max() < geometry.n_nodes and geometry.vertex.min() >= 0),
    }
    if geometry.neighbours is not None:
        report["neighbours_shape"] = list(np.asarray(geometry.neighbours).shape)
    return report


def neighbour_consistency(geometry: FemStencilGeometry) -> dict:
    """Compare ``neighbours`` with the shared-edge adjacency from ``vertex``.

    Both arrays are JOREK's; the check answers "is the element adjacency we use
    the same one JOREK stores".  Mismatches are expected only for degenerate
    (X-point/axis-collapsed) sides, which the operator drops.
    """

    if geometry.neighbours is None:
        return {"available": False}
    from collections import defaultdict

    neighbours = np.asarray(geometry.neighbours)
    vertex = geometry.vertex
    pos = None
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element in range(vertex.shape[1]):
        for side in range(4):
            a = int(vertex[side, element])
            b = int(vertex[(side + 1) % 4, element])
            edges[tuple(sorted((a, b)))].append(element + 1)

    matches = 0
    mismatches = []
    degenerate_sides = 0
    for element in range(vertex.shape[1]):
        for side in range(4):
            a = int(vertex[side, element])
            b = int(vertex[(side + 1) % 4, element])
            if geometry.cluster[a] == geometry.cluster[b]:
                degenerate_sides += 1
                continue
            key = tuple(sorted((a, b)))
            candidates = [e for e in edges[key] if e != element + 1]
            actual = candidates[0] if candidates else 0
            stored = int(neighbours[side, element])
            stored = 0 if stored <= 0 else stored
            if stored == actual:
                matches += 1
            else:
                mismatches.append((element + 1, side, stored, actual))
    checked = matches + len(mismatches)
    return {
        "available": True,
        "sides_checked": int(4 * vertex.shape[1]),
        "sides_with_noncoincident_corners": checked,
        "degenerate_sides": degenerate_sides,
        "matches": matches,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:8],
        "mismatch_fraction": len(mismatches) / float(max(checked, 1)),
    }
