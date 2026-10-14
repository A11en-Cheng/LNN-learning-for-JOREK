"""EXPERIMENT ONLY: static topology and event-local marker sampling cache.

Not a production backend or checkpoint-compatible replacement. Native cubic
geometry/fields are sampled on dense element-local grids. Inverse mapping of
each *bilinear microcell* is analytic, not iterative. Difficult points fall
back to native search. Field gradients differentiate the same bilinear value
representation, transformed with its geometry, rather than an unrelated table.
An optional confirmed path checks the candidate with cubic geometry and samples
the complete native cubic, caching only per-element coefficients. Native cubic
deposition/projection remain unchanged. No particle substep skips.
"""
from copy import copy, deepcopy
import hashlib
import time

import numpy as np
from scipy.spatial import cKDTree

from .native_gauss import _cubic_basis


def _cross(a, b):
    return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]


class ExperimentalLookupMesh:
    def __init__(self, mesh, *, grid_points=17, build_chunk=32768, max_bytes=4*1024**3,
                 sampling="bilinear"):
        if type(grid_points) is not int or grid_points < 3:
            raise ValueError("lookup requires at least three grid points")
        if type(build_chunk) is not int or build_chunk < 1 or max_bytes < 1:
            raise ValueError("positive lookup memory/build budgets required")
        if sampling not in ("bilinear", "native_cubic_confirmed"):
            raise ValueError("unsupported experimental lookup sampling")
        self.native = mesh
        self.sampling = sampling
        self.grid_points = grid_points
        self.cells = grid_points-1
        self.build_chunk = build_chunk
        self.n_elements = mesh.vertex.shape[1]
        self.geometry_sha256 = self._geometry_signature()
        self.estimated_bytes = self.n_elements*grid_points**2*192
        if self.estimated_bytes > max_bytes:
            raise MemoryError("lookup persistent memory estimate exceeds budget")
        q = np.linspace(0., 1., grid_points)
        s, t = np.meshgrid(q, q, indexing="ij")
        self.basis = _cubic_basis(s.ravel(), t.ravel())[0]
        self.basis_matrix = np.ascontiguousarray(self.basis.reshape(-1, 16))
        self.basis_matrix.flags.writeable = False
        self.node_ids = np.array(mesh.vertex.T, copy=True, order="C")
        self.local_sizes = np.array(mesh.size.transpose(2, 1, 0), copy=True, order="C")
        self.coordinate_coefficients = self._local_fields(mesh.coordinates)
        for a in (self.node_ids, self.local_sizes):
            a.flags.writeable = False
        self.configuration = deepcopy(mesh.configuration)
        self.configuration["experimental_lookup"] = dict(
            version="piecewise-bilinear-inverse-and-compatible-fields-v1", grid_points=grid_points,
            approximation="native_cubic_resampled_as_bilinear_microcells",
            fallback="native_location_for_unresolved_points", production_qualified=False,
            geometry_sha256=self.geometry_sha256, sampling=sampling,
            refresh_operator="cached_topology_basis_batched_matmul_v2",
            native_coefficient_layout="stride_preserving_element_vertex_field_degree_v3",
            confirmation=("one_native_newton_correction_then_residual_check_or_native_fallback"
                          if sampling == "native_cubic_confirmed" else "bilinear_only"))
        self.stats = dict(location_queries=0, hint_hits=0, tree_hits=0, fallback_queries=0,
                          sample_queries=0, field_table_builds=0, field_table_seconds=0.,
                          native_confirmation_queries=0, native_confirmation_accepted=0,
                          native_confirmation_rejected=0)
        started = time.perf_counter()
        self.coordinates_table = self._tabulate(mesh.coordinates)
        self.tree = cKDTree(self.coordinates_table.reshape(-1, 2), copy_data=False)
        self.geometry_build_seconds = time.perf_counter()-started
        self.field_table = self.feedback_table = None
        self.field_owner = self.feedback_owner = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _geometry_signature(self):
        digest = hashlib.sha256()
        for name in ("coordinates", "size", "vertex", "indices"):
            a = getattr(self.native, name)
            digest.update(str((name, a.dtype.str, a.shape)).encode())
            digest.update(a.tobytes(order="C"))
        return digest.hexdigest()

    def validate_fields(self, values, n_fields):
        # Public Source.advance checks snapshots before any particle work.
        # Concurrent geometry mutation inside an event is not supported.
        if self._geometry_signature() != self.geometry_sha256:
            raise ValueError("lookup geometry changed; rebuild its tables")
        return self.native.validate_fields(values, n_fields)

    @property
    def table_bytes(self):
        arrays = (self.basis, self.coordinates_table, self.field_table, self.feedback_table,
                  self.tree.indices, self.node_ids, self.local_sizes, self.coordinate_coefficients)
        # Tree node allocation is additional; report process RSS separately.
        return sum(a.nbytes for a in arrays if a is not None)

    def _local_fields(self, fields, out=None):
        coefficients = fields[:, :, 0, :][:, :, self.node_ids].transpose(2, 0, 3, 1)
        # Keep the native gather's vertex/field/degree strides. C-order copying
        # changes einsum's summation path and can perturb near-cancelling flows.
        local = coefficients if out is None else out
        if local.shape != coefficients.shape or local.dtype != np.float64:
            raise ValueError("invalid cached element coefficient buffer")
        local.flags.writeable = True
        try:
            local[:] = coefficients
            if not np.isfinite(local).all():
                raise ValueError("nonfinite cached element coefficients")
        finally:
            local.flags.writeable = False
        return local

    def _native_geometry(self, e, st):
        h, hs, ht = _cubic_basis(st[:, 0], st[:, 1])
        sizes = self.local_sizes[e]
        b, bs, bt = h*sizes, hs*sizes, ht*sizes
        x = self.coordinate_coefficients[e]
        xyz = np.einsum("bvd,bfvd->bf", b, x)
        xs = np.einsum("bvd,bfvd->bf", bs, x)
        xt = np.einsum("bvd,bfvd->bf", bt, x)
        jac = xs[:, 0]*xt[:, 1]-xt[:, 0]*xs[:, 1]
        valid = np.isfinite(jac) & (jac > 0) & (xyz[:, 0] > 0)
        return b, bs, bt, xyz, xs, xt, jac, valid

    def _confirm(self, x, e, st):
        """Bounded cubic correction; a failed confirmation is never accepted."""
        _, _, _, xyz, xs, xt, jac, valid = self._native_geometry(e, st)
        residual = xyz-x
        ds = np.divide(xt[:, 1]*residual[:, 0]-xt[:, 0]*residual[:, 1], jac,
                       out=np.zeros(len(st)), where=valid)
        dt = np.divide(-xs[:, 1]*residual[:, 0]+xs[:, 0]*residual[:, 1], jac,
                       out=np.zeros(len(st)), where=valid)
        corrected = st-np.column_stack((ds, dt))
        bounds = np.isfinite(corrected).all(axis=1) & (corrected >= 0).all(axis=1) & (corrected <= 1).all(axis=1)
        # Invalid candidates must not enter a polynomial evaluation with NaN.
        corrected[~(valid & bounds)] = st[~(valid & bounds)]
        *_, new_xyz, _xs, _xt, _jac, new_valid = self._native_geometry(e, corrected)
        accepted = valid & bounds & new_valid & (np.linalg.norm(new_xyz-x, axis=1) < 1e-10)
        self.stats["native_confirmation_queries"] += len(st)
        self.stats["native_confirmation_accepted"] += int(accepted.sum())
        self.stats["native_confirmation_rejected"] += int((~accepted).sum())
        return accepted, corrected

    def _tabulate(self, fields, out=None):
        n = len(fields)
        q = self.grid_points**2
        shape = (self.n_elements, self.grid_points, self.grid_points, n)
        storage = np.empty(shape, dtype=np.float64) if out is None else out
        if storage.shape != shape or storage.dtype != np.float64:
            raise ValueError("invalid cached field table buffer")
        storage.flags.writeable = True
        target = storage.reshape(self.n_elements, q, n)
        width = max(1, self.build_chunk//q)
        try:
            for start in range(0, self.n_elements, width):
                end = min(start+width, self.n_elements)
                coefficients = fields[:, :, 0, :][:, :, self.node_ids[start:end]].transpose(2, 3, 1, 0)
                coefficients = np.ascontiguousarray(coefficients*self.local_sizes[start:end, :, :, None])
                np.matmul(self.basis_matrix, coefficients.reshape(end-start, 16, n), out=target[start:end])
            if not np.isfinite(storage).all():
                raise ValueError("nonfinite lookup table")
        finally:
            storage.flags.writeable = False
        return storage

    @staticmethod
    def _corners(table, e, i, j):
        return table[e, i, j], table[e, i+1, j], table[e, i, j+1], table[e, i+1, j+1]

    @staticmethod
    def _bilinear(corners, u, v):
        a, b, c, d = corners
        val = ((1-u)*(1-v))[:, None]*a+(u*(1-v))[:, None]*b
        val += ((1-u)*v)[:, None]*c+(u*v)[:, None]*d
        du = (1-v)[:, None]*(b-a)+v[:, None]*(d-c)
        dv = (1-u)[:, None]*(c-a)+u[:, None]*(d-b)
        return val, du, dv

    def _inverse(self, x, e, i, j):
        """Stable quadratic inversion, testing both roots; no Newton loop."""
        a, p10, p01, p11 = self._corners(self.coordinates_table, e, i, j)
        b, c, d, y = p10-a, p01-a, p11-p10-p01+a, x-a
        aa = -_cross(b, d)
        bb = _cross(y, d)-_cross(b, c)
        cc = _cross(y, c)
        linear = np.abs(aa) <= 1e-12*np.maximum(np.abs(bb), np.finfo(float).tiny)
        disc = bb*bb-4*aa*cc
        root = np.sqrt(np.maximum(disc, 0.))
        q = -.5*(bb+np.copysign(root, bb))
        u1 = np.divide(q, aa, out=np.zeros_like(q), where=~linear & (aa != 0))
        u1[linear] = np.divide(-cc[linear], bb[linear], out=np.full(sum(linear), np.nan), where=bb[linear] != 0)
        u2 = np.divide(cc, q, out=np.full_like(q, np.nan), where=(q != 0) & ~linear)
        good = np.zeros(len(x), dtype=bool)
        uv = np.zeros((len(x), 2))
        for u in (u1, u2):
            cv = c+d*u[:, None]
            vv = np.divide(np.sum((y-b*u[:, None])*cv, axis=1), np.sum(cv*cv, axis=1),
                           out=np.full(len(x), np.nan), where=np.sum(cv*cv, axis=1) > 0)
            xs = b+d*vv[:, None]
            det = _cross(xs, cv)
            reconstructed = a+b*u[:, None]+c*vv[:, None]+d*(u*vv)[:, None]
            valid = (~good & np.isfinite(u) & np.isfinite(vv) & (linear | (disc >= 0))
                     & (u >= -1e-10) & (u <= 1+1e-10) & (vv >= -1e-10) & (vv <= 1+1e-10)
                     & (det > 0) & (x[:, 0] > 0)
                     & (np.linalg.norm(reconstructed-x, axis=1) < 1e-10))
            uv[valid] = np.column_stack((u[valid], vv[valid]))
            good |= valid
        st = (np.column_stack((i, j))+np.clip(uv, 0, 1))/self.cells
        return good, st

    def locate_batch(self, positions, *, element_hints=None, local_hints=None):
        x = np.asarray(positions)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("finite R/Z/phi positions required")
        n = len(x)
        elements = np.zeros(n, dtype=np.int32)
        local = np.zeros((n, 2))
        self.stats["location_queries"] += n
        if element_hints is not None:
            hints, st = np.asarray(element_hints), np.asarray(local_hints)
            if hints.shape != (n,) or st.shape != (n, 2) or not np.isfinite(st).all():
                raise ValueError("matching element/local hints required")
            ids = np.flatnonzero((hints > 0) & (hints <= self.n_elements))
            cell = np.minimum((np.clip(st[ids], 0, 1)*self.cells).astype(int), self.cells-1)
            good, found = self._inverse(x[ids, :2], hints[ids].astype(int)-1, cell[:, 0], cell[:, 1])
            elements[ids[good]], local[ids[good]] = hints[ids[good]], found[good]
            self.stats["hint_hits"] += int(good.sum())
        pending = np.flatnonzero(elements == 0)
        if len(pending):
            _, nearest = self.tree.query(x[pending, :2], workers=1)
            e = nearest//self.grid_points**2
            i = (nearest % self.grid_points**2)//self.grid_points
            j = nearest % self.grid_points
            # Four microcells meet at a grid node. Trying these is bounded
            # candidate selection, not an iterative nonlinear solver.
            for di, dj in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                active = np.flatnonzero(elements[pending] == 0)
                ii, jj = i[active]+di, j[active]+dj
                inside = (ii >= 0) & (ii < self.cells) & (jj >= 0) & (jj < self.cells)
                active, ii, jj = active[inside], ii[inside], jj[inside]
                good, found = self._inverse(x[pending[active], :2], e[active], ii, jj)
                ids = pending[active[good]]
                elements[ids], local[ids] = e[active[good]]+1, found[good]
                self.stats["tree_hits"] += int(good.sum())
        if self.sampling == "native_cubic_confirmed":
            ids = np.flatnonzero(elements > 0)
            accepted, corrected = self._confirm(x[ids, :2], elements[ids].astype(int)-1, local[ids])
            local[ids[accepted]] = corrected[accepted]
            elements[ids[~accepted]] = 0
        pending = np.flatnonzero(elements == 0)
        if len(pending):
            self.stats["fallback_queries"] += len(pending)
            options = {} if element_hints is None else dict(element_hints=np.asarray(element_hints)[pending],
                local_hints=np.asarray(local_hints)[pending])
            elements[pending], local[pending] = self.native.locate_batch(x[pending], **options)
        return elements, local

    def _marker_points(self, e, st, fields, feedback):
        # Source.advance owns read-only snapshots. Strong references prevent
        # identity recycling; new event snapshots rebuild all value tables.
        if fields is not self.field_owner or feedback is not self.feedback_owner:
            if fields.flags.writeable or feedback.flags.writeable:
                raise ValueError("lookup requires event-owned read-only field snapshots")
            started = time.perf_counter()
            # Invalidate owners first: failed preparation cannot associate a
            # partial new table with an old snapshot on a subsequent retry.
            previous_fields, previous_feedback = self.field_table, self.feedback_table
            self.field_owner = self.feedback_owner = None
            self.field_table = self.feedback_table = None
            prepare = self._local_fields if self.sampling == "native_cubic_confirmed" else self._tabulate
            self.field_table = prepare(fields, out=previous_fields)
            self.feedback_table = prepare(feedback, out=previous_feedback)
            self.field_owner, self.feedback_owner = fields, feedback
            self.stats["field_table_builds"] += 1
            self.stats["field_table_seconds"] += time.perf_counter()-started
        if self.sampling == "native_cubic_confirmed":
            b, bs, bt, xyz, xs, xt, jac, valid = self._native_geometry(e, st)
            if not valid.all():
                raise ValueError("invalid native point geometry")
            coefficients = self.field_table[e]
            values = np.einsum("bvd,bfvd->bf", b, coefficients)
            ds = np.einsum("bvd,bfvd->bf", bs, coefficients)
            dt = np.einsum("bvd,bfvd->bf", bt, coefficients)
            gradients = np.stack(((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/jac[:, None],
                                  (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/jac[:, None]), axis=-1)
            projected = np.einsum("bvd,bfvd->bf", b, self.feedback_table[e])
            self.stats["sample_queries"] += len(e)
            return values, gradients, projected, xyz
        scaled = np.clip(st, 0, 1)*self.cells
        cell = np.minimum(scaled.astype(int), self.cells-1)
        u, v = (scaled-cell).T
        i, j = cell.T
        values, ds, dt = self._bilinear(self._corners(self.field_table, e, i, j), u, v)
        xyz, xs, xt = self._bilinear(self._corners(self.coordinates_table, e, i, j), u, v)
        projected = self._bilinear(self._corners(self.feedback_table, e, i, j), u, v)[0]
        jac = _cross(xs, xt)
        if not np.isfinite(jac).all() or (jac <= 0).any() or (xyz[:, 0] <= 0).any():
            raise ValueError("invalid bilinear lookup geometry")
        gradients = np.stack(((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/jac[:, None],
                              (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/jac[:, None]), axis=-1)
        self.stats["sample_queries"] += len(e)
        return values, gradients, projected, xyz


def make_lookup_source(source, lookup):
    """Explicit diagnostic copy, never a silent production-mode mutation."""
    if source.mesh is not lookup.native:
        raise ValueError("lookup must be bound to this source mesh")
    if source.time_sampling == "jorek_raw_deltas_v1" and not source.field_static:
        raise ValueError("this experiment requires static per-window source fields")
    candidate = copy(source)
    candidate.mesh = lookup
    candidate.spatial_backend = "batched_v1"
    candidate._spatial = None
    candidate.configuration = deepcopy(source.configuration)
    candidate.configuration.update(version="experimental-native-source-lookup-v1",
        spatial_backend="experimental_lookup", mesh=lookup.configuration,
        approximation=("bilinear_candidate_with_native_cubic_confirmation_and_sampling"
                       if lookup.sampling == "native_cubic_confirmed" else
                       "piecewise_bilinear_geometry_values_and_compatible_gradients"))
    return candidate
