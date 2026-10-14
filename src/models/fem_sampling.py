"""Deterministic FEM sampling/deposition on the JOREK quadrilateral mesh.

Phase B1 needs the two directions of the JOREK finite-element mesh:

* **sampling**: read local plasma quantities (``rho``, ``rho_imp``, ``T_e``,
  ``psi``) at an arbitrary shard position ``(R, Z)``.  JOREK uses
  ``find_RZ`` + ``interp_PRZ`` (element search + polynomial/Hermite
  interpolation); Phase B2 established that the nodal ``dof-0`` data on this
  mesh is bilinear-quad compatible (``src/physics/fem_weak.py``), so the
  deterministic stand-in here is the 4-node bilinear shape function of the
  containing quadrilateral.
* **deposition**: scatter per-shard source integrals back onto the nodes with
  the same shape functions (``b_i = sum_p N_i(s_p, t_p) q_p``) and divide by
  the lumped nodal measure (``m_i = int N_i 2 pi R dR dZ``), which is the
  diagonal (lumped) L2 projection of the deposited quantity.  JOREK solves the
  consistent-mass projection system (``projections*.h5``); the lumped form used
  here keeps the *budget exactly closed by construction*
  (``sum_i x_i m_i == sum_p q_p``) at the cost of a registered O(h^2)
  truncation difference, which is one of the proxy-vs-``inj_source``
  registration items of the Phase B1 report.

Everything here is deterministic and differentiable w.r.t. the point values
(the element search uses the nearest-element candidate; the shape weights
themselves are differentiated through the Newton solve).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ElementLocation",
    "FemMeshSampler",
    "bilinear_shape",
]


def bilinear_shape(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Bilinear shape functions ``N1..N4`` on ``[0,1]^2``.

    Node order follows the JOREK ``vertex`` order (counter-clockwise corners):
    ``N1=(1-s)(1-t)``, ``N2=s(1-t)``, ``N3=s t``, ``N4=(1-s)t``.
    """
    return torch.stack(
        [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t], dim=-1
    )


@dataclass(frozen=True)
class ElementLocation:
    """Element indices and reference coordinates for a batch of points."""

    element: torch.Tensor  # [B, P] long
    s: torch.Tensor  # [B, P]
    t: torch.Tensor  # [B, P]
    inside: torch.Tensor  # [B, P] bool

    @property
    def weights(self) -> torch.Tensor:
        """Bilinear shape weights ``[B, P, 4]`` at the located points."""
        return bilinear_shape(self.s, self.t)


class FemMeshSampler:
    """Element search + bilinear interpolation/deposition on one JOREK mesh.

    Parameters
    ----------
    pos:
        ``[N, 2]`` node coordinates ``(R, Z)`` in metres.
    vertex:
        ``[4, E]`` element connectivity, **0-based** (the HDF5 files store
        1-based indices; the caller converts once).
    n_candidates:
        Number of nearest-element candidates tried per point before falling
        back to the nearest node.
    """

    def __init__(
        self,
        pos: np.ndarray | torch.Tensor,
        vertex: np.ndarray | torch.Tensor,
        *,
        n_candidates: int = 4,
        newton_iters: int = 8,
        tol: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        pos = torch.as_tensor(np.asarray(pos, dtype=np.float64), dtype=dtype)
        vertex = torch.as_tensor(np.asarray(vertex), dtype=torch.long)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"pos must be [N, 2], got {tuple(pos.shape)}")
        if vertex.ndim != 2 or vertex.shape[0] != 4:
            raise ValueError(f"vertex must be [4, E], got {tuple(vertex.shape)}")
        if int(vertex.min()) < 0:
            raise ValueError("vertex must be 0-based; convert the 1-based JOREK file first")
        if int(vertex.max()) >= pos.shape[0]:
            raise ValueError("vertex references nodes outside pos")
        self.dtype = dtype
        self.pos = pos
        self.vertex = vertex
        self.n_candidates = max(1, int(n_candidates))
        self.newton_iters = max(1, int(newton_iters))
        self.tol = float(tol)
        self.n_nodes = int(pos.shape[0])
        self.n_elem = int(vertex.shape[1])
        # [E, 4, 2] corner coordinates, [E, 2] centroids.
        self.corners = pos[vertex.t()].contiguous()
        self.centroids = self.corners.mean(dim=1)
        self._measure: torch.Tensor | None = None
        # Node -> incident elements (padded); used to build robust element
        # candidates: a point inside an element has that element among the
        # incidences of its nearest nodes.  Nearest-centroid candidates alone
        # miss stretched elements (measured: 25% of the shard positions on the
        # real mesh at frame 3000).
        incidence: list[list[int]] = [[] for _ in range(self.n_nodes)]
        vertex_np = self.vertex.cpu().numpy()
        for element in range(self.n_elem):
            for corner in range(4):
                incidence[int(vertex_np[corner, element])].append(element)
        max_inc = max((len(items) for items in incidence), default=1)
        padded = np.zeros((self.n_nodes, max_inc), dtype=np.int64)
        for node, items in enumerate(incidence):
            padded[node, : len(items)] = items
            padded[node, len(items) :] = items[0] if items else 0
        self.node_elements = torch.as_tensor(padded, dtype=torch.long)
        self.max_incidence = int(max_inc)

    # ------------------------------------------------------------------ helpers

    def _candidate_elements(self, points: torch.Tensor, *, n_nodes: int = 4) -> torch.Tensor:
        """``[P, k]`` element candidates ordered by likelihood.

        The first block contains the incident elements of the ``n_nodes``
        nearest nodes; the second block appends the nearest-centroid elements
        (covers points in stretched elements whose centroid is far away).
        """
        flat = points.reshape(-1, 2)
        node_distance = torch.cdist(flat, self.pos)
        k_nodes = min(max(1, int(n_nodes)), self.n_nodes)
        nearest_nodes = torch.topk(node_distance, k=k_nodes, largest=False, dim=1).indices
        incident = self.node_elements.to(flat.device)[nearest_nodes].reshape(
            flat.shape[0], k_nodes * self.max_incidence
        )
        k_centroid = min(self.n_candidates, self.n_elem)
        centroid_distance = torch.cdist(flat, self.centroids)
        nearest_elements = torch.topk(
            centroid_distance, k=k_centroid, largest=False, dim=1
        ).indices
        candidates = torch.cat([incident, nearest_elements], dim=1)
        return candidates

    def _newton_reference(self, points: torch.Tensor, elements: torch.Tensor):
        """Inverse bilinear map for one candidate element per point.

        Returns ``(s, t, converged, residual)`` with ``[P]`` tensors.
        """
        corners = self.corners[elements]  # [P, 4, 2]
        x0 = corners[:, 0]
        x1 = corners[:, 1]
        x2 = corners[:, 2]
        x3 = corners[:, 3]
        s = torch.full_like(points[:, 0], 0.5)
        t = torch.full_like(points[:, 1], 0.5)
        for _ in range(self.newton_iters):
            n1 = (1.0 - s) * (1.0 - t)
            n2 = s * (1.0 - t)
            n3 = s * t
            n4 = (1.0 - s) * t
            mapped = (
                n1[:, None] * x0 + n2[:, None] * x1 + n3[:, None] * x2 + n4[:, None] * x3
            )
            dx = points - mapped
            # dN/ds = [-(1-t), (1-t), t, -t]; dN/dt = [-(1-s), -s, s, (1-s)]
            one_m_t = 1.0 - t
            one_m_s = 1.0 - s
            dxds = (
                -one_m_t[:, None] * x0
                + one_m_t[:, None] * x1
                + t[:, None] * x2
                - t[:, None] * x3
            )
            dxdt = (
                -one_m_s[:, None] * x0
                - s[:, None] * x1
                + s[:, None] * x2
                + one_m_s[:, None] * x3
            )
            ds = dxds
            dt = dxdt
            jac = torch.stack([ds, dt], dim=-1)  # [P, 2, 2]
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            safe = det.abs() < 1e-30
            inv_det = torch.where(safe, torch.ones_like(det), 1.0 / torch.where(safe, torch.ones_like(det), det))
            # J^{-1} dx
            delta_s = (jac[:, 1, 1] * dx[:, 0] - jac[:, 0, 1] * dx[:, 1]) * inv_det
            delta_t = (-jac[:, 1, 0] * dx[:, 0] + jac[:, 0, 0] * dx[:, 1]) * inv_det
            delta_s = torch.where(safe, torch.zeros_like(delta_s), delta_s)
            delta_t = torch.where(safe, torch.zeros_like(delta_t), delta_t)
            s = s + delta_s
            t = t + delta_t
        n1 = (1.0 - s) * (1.0 - t)
        n2 = s * (1.0 - t)
        n3 = s * t
        n4 = (1.0 - s) * t
        mapped = n1[:, None] * x0 + n2[:, None] * x1 + n3[:, None] * x2 + n4[:, None] * x3
        residual = (mapped - points).norm(dim=-1)
        inside = (
            (s >= -self.tol)
            & (s <= 1.0 + self.tol)
            & (t >= -self.tol)
            & (t <= 1.0 + self.tol)
            & ~safe
        )
        return s, t, inside, residual

    def _nearest_node(self, points: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, 2)
        distance = torch.cdist(flat, self.pos)
        return distance.argmin(dim=1)

    # ------------------------------------------------------------------- locate

    def locate(self, points: torch.Tensor) -> ElementLocation:
        """Locate ``[B, P, 2]`` points in the mesh, one element per point."""
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"points must be [B, P, 2], got {tuple(points.shape)}")
        batch, n_points = points.shape[0], points.shape[1]
        dtype = self.pos.dtype
        points = points.to(dtype=dtype)
        if n_points == 0:
            empty = points.new_zeros(batch, 0)
            return ElementLocation(
                element=empty.long(), s=empty, t=empty, inside=empty.bool()
            )
        flat = points.reshape(-1, 2)
        candidates = self._candidate_elements(points)  # [P, k]
        best_element = torch.full((flat.shape[0],), -1, dtype=torch.long, device=flat.device)
        best_s = flat.new_zeros(flat.shape[0])
        best_t = flat.new_zeros(flat.shape[0])
        best_residual = torch.full_like(best_s, float("inf"))
        found = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
        inside_flag = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
        for column in range(candidates.shape[1]):
            element = candidates[:, column]
            s, t, inside, residual = self._newton_reference(flat, element)
            take = inside & ~found
            best_element = torch.where(take, element, best_element)
            best_s = torch.where(take, s, best_s)
            best_t = torch.where(take, t, best_t)
            best_residual = torch.where(take, residual, best_residual)
            inside_flag = inside_flag | take
            found = found | take
            if bool(found.all()):
                break
        # Fallback: the candidate with the smallest residual (flagged outside).
        if not bool(found.all()):
            for column in range(candidates.shape[1]):
                element = candidates[:, column]
                s, t, _inside, residual = self._newton_reference(flat, element)
                take = ~found & (residual < best_residual)
                best_element = torch.where(take, element, best_element)
                best_s = torch.where(take, s, best_s)
                best_t = torch.where(take, t, best_t)
                best_residual = torch.where(take, residual, best_residual)
                found = found | take
        outside = best_element < 0
        if bool(outside.any()):
            nearest = self._nearest_node(flat)
            # Emulate a point sitting exactly on the nearest node of a candidate
            # element so the deposition weights stay well defined.
            for candidate_column in range(candidates.shape[1]):
                element = candidates[:, candidate_column]
                on_node = (self.vertex.t()[element] == nearest[:, None]).any(dim=1)
                take = outside & on_node
                best_element = torch.where(take, element, best_element)
                best_s = torch.where(take, torch.full_like(best_s, 0.5), best_s)
                best_t = torch.where(take, torch.full_like(best_t, 0.5), best_t)
                outside = outside & ~take
            # Last resort: no candidate element contains the nearest node
            # (degenerate mesh region) -> keep the nearest candidate element.
            if bool(outside.any()):
                fallback = candidates[:, 0]
                best_element = torch.where(outside, fallback, best_element)
        # Keep extrapolation bounded: points outside every candidate element are
        # clamped to the reference square (registered as ``inside=False``).
        clamp_s = best_s.clamp(-self.tol, 1.0 + self.tol)
        clamp_t = best_t.clamp(-self.tol, 1.0 + self.tol)
        best_s = torch.where(inside_flag, best_s, clamp_s)
        best_t = torch.where(inside_flag, best_t, clamp_t)
        return ElementLocation(
            element=best_element.reshape(batch, n_points),
            s=best_s.reshape(batch, n_points),
            t=best_t.reshape(batch, n_points),
            inside=inside_flag.reshape(batch, n_points),
        )

    # ------------------------------------------------------------ sample/deposit

    def _point_nodes(self, location: ElementLocation) -> torch.Tensor:
        """``[..., P, 4]`` node indices for the located elements."""
        return self.vertex.t()[location.element]

    def sample(self, values: torch.Tensor, location: ElementLocation) -> torch.Tensor:
        """Bilinear sample of ``values [B, N, C]`` at the located points."""
        if values.ndim != 3:
            raise ValueError(f"values must be [B, N, C], got {tuple(values.shape)}")
        node_index = self._point_nodes(location)  # [B, P, 4]
        batch, n_points = node_index.shape[0], node_index.shape[1]
        index = node_index.reshape(batch, n_points * 4, 1).expand(batch, n_points * 4, values.shape[2])
        gathered = torch.gather(values, 1, index).reshape(
            batch, n_points, 4, values.shape[2]
        )
        weights = location.weights.to(dtype=values.dtype).unsqueeze(-1)
        return (gathered * weights).sum(dim=2)

    def sample_gradient(
        self, values: torch.Tensor, location: ElementLocation
    ) -> torch.Tensor:
        """Analytic bilinear field gradient at the located points.

        Returns ``[B, P, C, 2]`` with the two physical components ordered as the
        mesh coordinates ``(R, Z)``.  The reference-space derivatives are mapped
        through ``J^{-T}`` of the element map, so no finite-difference sampling
        is needed (deterministic, one locate per point).
        """
        if values.ndim != 3:
            raise ValueError(f"values must be [B, N, C], got {tuple(values.shape)}")
        corners = self.corners.to(dtype=values.dtype)[location.element]  # [B, P, 4, 2]
        s = location.s.to(dtype=values.dtype)
        t = location.t.to(dtype=values.dtype)
        one_m_s = 1.0 - s
        one_m_t = 1.0 - t
        # dN/ds and dN/dt for the bilinear reference square.
        dnds = torch.stack([-one_m_t, one_m_t, t, -t], dim=-1)  # [B, P, 4]
        dndt = torch.stack([-one_m_s, -s, s, one_m_s], dim=-1)
        x_s = (dnds.unsqueeze(-1) * corners).sum(dim=2)  # [B, P, 2]
        x_t = (dndt.unsqueeze(-1) * corners).sum(dim=2)
        jac = torch.stack([x_s, x_t], dim=-1)  # [B, P, 2, 2]: jac[..., i, j] = dx_i/dxi_j
        det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
        safe = det.abs() < 1e-30
        inv_det = torch.where(safe, torch.ones_like(det), 1.0 / torch.where(safe, torch.ones_like(det), det))
        # J^{-1}: [[dxi/dx, dxi/dy], [deta/dx, deta/dy]]
        jinv = torch.stack(
            [
                torch.stack([jac[..., 1, 1] * inv_det, -jac[..., 0, 1] * inv_det], dim=-1),
                torch.stack([-jac[..., 1, 0] * inv_det, jac[..., 0, 0] * inv_det], dim=-1),
            ],
            dim=-2,
        )  # [B, P, 2, 2]
        dndxi = torch.stack([dnds, dndt], dim=-1)  # [B, P, 4, 2]
        # dN/dx_m = sum_j dN/dxi_j * jinv[j, m]; degenerate elements get zero.
        dndx = torch.einsum("bpkj,bpjm->bpkm", dndxi, jinv)
        dndx = torch.where(safe.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(dndx), dndx)
        node_index = self._point_nodes(location)  # [B, P, 4]
        batch, n_points, n_channels = values.shape[0], node_index.shape[1], values.shape[2]
        index = node_index.reshape(batch, n_points * 4, 1).expand(
            batch, n_points * 4, n_channels
        )
        gathered = torch.gather(values, 1, index).reshape(batch, n_points, 4, n_channels)
        # gradient[..., c, m] = sum_k dN_k/dx_m * value_k,c
        return torch.einsum("bpkm,bpkc->bpcm", dndx, gathered)

    def deposit(self, per_point: torch.Tensor, location: ElementLocation) -> torch.Tensor:
        """Scatter ``per_point [B, P, C]`` to nodes with shape weights."""
        if per_point.ndim != 3:
            raise ValueError(f"per_point must be [B, P, C], got {tuple(per_point.shape)}")
        batch, n_points, n_channels = per_point.shape
        node_index = self._point_nodes(location)  # [B, P, 4]
        weights = location.weights.to(dtype=per_point.dtype)  # [B, P, 4]
        contributions = (per_point.unsqueeze(2) * weights.unsqueeze(-1)).reshape(
            batch, n_points * 4, n_channels
        )
        flat_index = node_index.reshape(batch, n_points * 4)
        offset = torch.arange(batch, device=per_point.device).unsqueeze(1) * self.n_nodes
        out = per_point.new_zeros(batch * self.n_nodes, n_channels)
        out.index_add_(0, (flat_index + offset).reshape(-1), contributions.reshape(-1, n_channels))
        return out.reshape(batch, self.n_nodes, n_channels)

    # ------------------------------------------------------------------ measure

    def node_measure(self) -> torch.Tensor:
        """Lumped nodal measure ``int N_i 2 pi R dR dZ`` ``[N]`` (cached)."""
        if self._measure is None:
            corners = self.corners.detach().cpu().numpy()  # [E, 4, 2]
            weights_nodes = np.zeros((self.n_elem, 4), dtype=np.float64)
            gauss = 0.5 + 0.5 / np.sqrt(3.0) * np.array([-1.0, 1.0])
            gauss_w = np.array([0.5, 0.5])
            for gs, ws in zip(gauss, gauss_w):
                for gt, wt in zip(gauss, gauss_w):
                    n1 = (1.0 - gs) * (1.0 - gt)
                    n2 = gs * (1.0 - gt)
                    n3 = gs * gt
                    n4 = (1.0 - gs) * gt
                    shape = np.array([n1, n2, n3, n4])
                    dnds = np.array([-(1.0 - gt), (1.0 - gt), gt, -gt])
                    dndt = np.array([-(1.0 - gs), -gs, gs, (1.0 - gs)])
                    x_s = dnds @ corners  # [E, 2]
                    x_t = dndt @ corners
                    det = np.abs(x_s[:, 0] * x_t[:, 1] - x_s[:, 1] * x_t[:, 0])
                    radius = corners[:, :, 0] @ shape  # [E]
                    weights_nodes += (ws * wt) * shape[None, :] * det[:, None] * radius[:, None]
            weights_nodes *= 2.0 * np.pi
            measure = np.zeros(self.n_nodes, dtype=np.float64)
            for corner in range(4):
                np.add.at(measure, self.vertex[corner].detach().cpu().numpy(), weights_nodes[:, corner])
            self._measure = torch.as_tensor(measure, dtype=self.pos.dtype)
        return self._measure

    def project(
        self,
        per_point: torch.Tensor,
        location: ElementLocation,
        *,
        measure_floor: float = 1e-9,
    ) -> torch.Tensor:
        """Lumped L2 projection: deposit then divide by the nodal measure."""
        deposited = self.deposit(per_point, location)
        measure = self.node_measure().to(device=deposited.device, dtype=deposited.dtype)
        floor = max(float(measure_floor), float(measure.mean().abs()) * 1e-6)
        return deposited / measure.clamp_min(floor).unsqueeze(0).unsqueeze(-1)
