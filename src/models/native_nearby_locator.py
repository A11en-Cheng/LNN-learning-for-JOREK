"""Opt-in H=1 port of JOREK's edge-limited find_RZ_nearby workflow.

Uses SAVED native neighbours, not adjacency inferred from shared vertices.
Nearby Newton/edge/lost rules follow JOREK. Global fallback is the existing
Python locator, not JOREK find_RZ; this boundary is explicit in the fingerprint.
"""
from copy import copy, deepcopy
import threading

import numpy as np

from .native_coupling import NativeSourceMesh
from .native_state import configuration_digest


def _exit_face(st):
    """Literal quadrant comparisons in coord_in_neighbour (zero-based sides)."""
    s, t = st[:, 0], st[:, 1]
    return np.where(s > t, np.where(1-s > t, 0, 1),
                    np.where(1-s <= t, 2, 3))


def _edge_coordinates(face, along):
    return np.column_stack((
        np.choose(face, (along, np.ones(len(face)), 1-along, np.zeros(len(face)))),
        np.choose(face, (np.zeros(len(face)), along, np.ones(len(face)), 1-along))))


class NativeNearbyLocator:
    """Independent particle Newton updates in batches, eight rounds maximum."""

    def __init__(self, mesh, neighbours):
        if type(mesh) is not NativeSourceMesh:
            raise ValueError('nearby locator requires native H1 cubic mesh')
        n = np.asarray(neighbours)
        if (n.shape != mesh.vertex.shape or n.dtype.kind not in 'iu'
                or (n > len(mesh.gauss)).any()):
            raise ValueError('saved native neighbours must be integer [4,elements]')
        self.mesh = mesh
        self.neighbours = np.array(n, dtype=np.int64, copy=True).T
        self.faces = np.full(self.neighbours.shape, -1, dtype=np.int8)
        self.reverse = np.zeros(self.neighbours.shape, dtype=bool)
        # Preserve source handling: first reciprocal side, then vertex direction.
        for e, row in enumerate(self.neighbours):
            for face, j in enumerate(row):
                if j <= 0:
                    continue
                other = int(j)-1
                sides = np.flatnonzero(self.neighbours[other] == e+1)
                if other == e or not len(sides):
                    raise ValueError('saved neighbours are not reciprocal')
                side = int(sides[0])
                a, b = mesh.vertex[[face, (face+1) % 4], e]
                c, d = mesh.vertex[[side, (side+1) % 4], other]
                if a == d or b == c:
                    self.reverse[e, face] = True
                elif a != c and b != d:
                    raise ValueError('saved neighbours have no shared edge endpoint')
                self.faces[e, face] = side
        for a in (self.neighbours, self.faces, self.reverse):
            a.flags.writeable = False
        self.configuration = dict(
            version='jorek-find-RZ-nearby-H1-v1',
            saved_neighbours=configuration_digest(n),
            newton_iterations=8, squared_RZ_tolerance=1e-24,
            inverse_determinant_floor=1e-8, crossing_threshold=1-1e-12,
            boundary_policy='zero_neighbour_negative_old_element_lost',
            global_fallback='existing_Python_native_global_not_JOREK_find_RZ',
            invalid_geometry='Python_positive_geometry_guard_then_global',
            production_qualified=False)
        self.last_stats = None
        self.last_ifail = None

    def _geometry(self, e, st, stats):
        stats['geometry_batches'] += 1
        stats['geometry_points'] += len(e)
        try:
            rz, jac = self.mesh._location_geometries(e, st)
            return rz, jac, np.ones(len(e), dtype=bool)
        except ValueError:
            # A bad particle must not force every valid particle into fallback.
            rz = np.zeros((len(e), 2)); jac = np.zeros((len(e), 2, 2))
            good = np.zeros(len(e), dtype=bool)
            for i in range(len(e)):
                try:
                    rz[i], jac[i] = self.mesh._location_geometry(int(e[i]), st[i])
                    good[i] = True
                except ValueError:
                    pass
            return rz, jac, good

    def locate_batch(self, positions, *, element_hints=None, local_hints=None):
        x = np.asarray(positions)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError('finite R/Z/phi positions required')
        count = len(x)
        stats = dict(point_visits=count, geometry_batches=0, geometry_points=0,
                     newton_particle_rounds=0, edge_crossings=0, lost_points=0,
                     global_fallback_points=0, invalid_hint_points=0,
                     negative_neighbour_points=0, iteration_limit_points=0,
                     invalid_geometry_points=0, nearby_accepted=0)
        elements = np.zeros(count, dtype=np.int32)
        local = np.zeros((count, 2), dtype=np.float64)
        flags = np.zeros(count, dtype=np.int32)
        pending = np.ones(count, dtype=bool)
        if element_hints is None:
            hints = np.zeros(count, dtype=np.int32)
            st = np.zeros((count, 2), dtype=np.float64)
        else:
            hints = np.asarray(element_hints)
            st = np.asarray(local_hints)
            if (hints.shape != (count,) or hints.dtype.kind not in 'iu'
                    or st.shape != (count, 2) or st.dtype != np.dtype(np.float64)
                    or not np.isfinite(st).all()):
                raise ValueError('matching integer elements and float64 local hints required')
            st = st.copy()
        e = hints.astype(np.int64)-1
        active = np.flatnonzero((e >= 0) & (e < len(self.mesh.gauss)))
        stats['invalid_hint_points'] = count-len(active)
        rz, jac, good = self._geometry(e[active], st[active], stats)
        stats['invalid_geometry_points'] += int((~good).sum())
        flags[active[~good]] = 2
        active, rz, jac = active[good], rz[good], jac[good]
        for _ in range(8):
            if not len(active):
                break
            stats['newton_particle_rounds'] += len(active)
            delta_x = x[active, :2]-rz
            rs, rt, zs, zt = jac[:, 0, 0], jac[:, 0, 1], jac[:, 1, 0], jac[:, 1, 1]
            det = rs*zt-rt*zs
            inv = np.copysign(np.full(len(det), 1e8), det)
            np.divide(1., det, out=inv, where=np.abs(det) >= 1e-8)
            step = np.column_stack(((zt*delta_x[:, 0]-rt*delta_x[:, 1])*inv,
                                    (-zs*delta_x[:, 0]+rs*delta_x[:, 1])*inv))
            distance = np.where(step > 0, 1-st[active], st[active])
            factor = np.max(np.abs(step)/np.maximum(distance, 1e-30), axis=1)
            crossing = factor >= 1-1e-12
            next_st = st[active]+step/np.where(crossing, factor, 1)[:, None]
            st[active] = next_st
            crossed = active[crossing]
            keep = np.ones(len(active), dtype=bool)
            if len(crossed):
                face = _exit_face(st[crossed])
                neighbor = self.neighbours[e[crossed], face]
                lost, search, transfer = neighbor == 0, neighbor < 0, neighbor > 0
                ids = crossed[lost]
                elements[ids] = -(e[ids]+1); local[ids] = st[ids]
                flags[ids] = -1; pending[ids] = False
                stats['lost_points'] += len(ids)
                stats['negative_neighbour_points'] += int(search.sum())
                flags[crossed[search]] = 4
                keep[np.flatnonzero(crossing)[lost | search]] = False
                ids, f = crossed[transfer], face[transfer]
                if len(ids):
                    q = st[ids]
                    along = np.choose(f, (q[:, 0], q[:, 1], 1-q[:, 0], 1-q[:, 1]))
                    along = np.where(self.reverse[e[ids], f], 1-along, along)
                    st[ids] = _edge_coordinates(self.faces[e[ids], f], along)
                    e[ids] = neighbor[transfer]-1
                    stats['edge_crossings'] += len(ids)
            active = active[keep]
            rz, jac, good = self._geometry(e[active], st[active], stats)
            stats['invalid_geometry_points'] += int((~good).sum())
            flags[active[~good]] = 2
            active, rz, jac = active[good], rz[good], jac[good]
            residual = rz-x[active, :2]
            done = np.sum(residual*residual, axis=1) < 1e-24
            ids = active[done]
            elements[ids] = e[ids]+1; local[ids] = st[ids]; pending[ids] = False
            stats['nearby_accepted'] += len(ids)
            active, rz, jac = active[~done], rz[~done], jac[~done]
        flags[active] = 3
        stats['iteration_limit_points'] = len(active)
        ids = np.flatnonzero(pending)
        stats['global_fallback_points'] = len(ids)
        if len(ids):
            # A deliberate seam: reuse existing global implementation, not a
            # claim to port JOREK's rtree order/five initial guesses/lost flags.
            pe, ps = self.mesh.locate(x[ids])
            elements[ids], local[ids] = pe, ps
            flags[ids[pe <= 0]] = 99
        self.last_stats, self.last_ifail = stats, flags
        return elements, local


class NativeNearbyMarkerSource:
    """Opt-in complete-source adapter; no shared source/mesh method mutation."""

    def __init__(self, source, *, neighbours):
        self.source = source
        self._validate_native()
        self._mesh = source.mesh
        self._snapshot = {k: np.array(getattr(source.mesh, k), copy=True)
                          for k in ('coordinates', 'size', 'vertex', 'indices')}
        self._source_digest = configuration_digest(source.configuration)
        self.locator = NativeNearbyLocator(source.mesh, neighbours)
        self.configuration = deepcopy(source.configuration)
        self.configuration['nearby_locator'] = deepcopy(self.locator.configuration)
        self._lock = threading.Lock()
        self.last_stats = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _validate_native(self):
        mesh = self.source.mesh
        if (type(mesh) is not NativeSourceMesh or self.source.spatial_backend != 'batched_v1'
                or self.source._spatial is not None
                or any(k in mesh.__dict__ for k in ('locate', 'locate_batch', '_location_geometry', '_location_geometries'))):
            raise ValueError('nearby requires unoverridden native CPU batched_v1 source')

    def advance(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('nearby source cannot advance concurrently')
        stats = dict(batch_calls=0, point_visits=0, geometry_batches=0,
                     geometry_points=0, newton_particle_rounds=0, edge_crossings=0,
                     lost_points=0, global_fallback_points=0, invalid_hint_points=0,
                     negative_neighbour_points=0, iteration_limit_points=0,
                     invalid_geometry_points=0, nearby_accepted=0)
        try:
            self._validate_native()
            state = args[0] if args else kwargs.get('state')
            if state is not None and state.cloud.st.dtype != np.dtype(np.float64):
                raise ValueError('nearby requires float64 marker local coordinates')
            if (self.source.mesh is not self._mesh
                    or configuration_digest(self.source.configuration) != self._source_digest
                    or any(not np.array_equal(getattr(self._mesh, k), a) for k, a in self._snapshot.items())):
                raise ValueError('nearby static mesh/source changed; rebuild adapter')
            mesh, candidate = copy(self._mesh), copy(self.source)
            def locate_batch(positions, **hints):
                result = self.locator.locate_batch(positions, **hints)
                stats['batch_calls'] += 1
                for k, v in self.locator.last_stats.items():
                    stats[k] += v
                return result
            mesh.locate_batch = locate_batch
            candidate.mesh = mesh
            return candidate.advance(*args, **kwargs)
        finally:
            self.last_stats = stats
            self._lock.release()
