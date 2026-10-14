"""Opt-in topological neighbors before the native marker global fallback.

The native in-cell batch Newton, geometry, deposition and source substeps stay
unchanged. Only unresolved points try oriented shared edges first. This is not
a literal port of JOREK find_RZ_nearby and is not production-qualified.
"""
from copy import copy, deepcopy
import threading

import numpy as np

from .native_coupling import NativeSourceMesh
from .native_state import configuration_digest


class NativeNeighborMarkerSource:
    """Static native mesh adapter; no mutation of the wrapped source or mesh."""

    def __init__(self, source, *, max_hops=4):
        _validate_native(source)
        if type(max_hops) is not int or not 1 <= max_hops <= 8:
            raise ValueError('neighbor max_hops must be an integer in [1, 8]')
        self.source = source
        self.max_hops = max_hops
        self._mesh = source.mesh
        self._source_configuration = configuration_digest(source.configuration)
        self._snapshot = {name: np.array(getattr(self._mesh, name), copy=True)
                          for name in ('coordinates', 'size', 'vertex', 'indices')}
        self._neighbors, self._faces, self._reverse = _topology(self._mesh.vertex)
        self._topology_bytes = sum(a.nbytes for a in
            (self._neighbors, self._faces, self._reverse, *self._snapshot.values()))
        self.configuration = deepcopy(source.configuration)
        self.configuration['experimental_neighbor_locator'] = dict(
            version='native-batch-oriented-neighbor-fallback-v1',
            geometry=configuration_digest(self._snapshot), max_hops=max_hops,
            tolerance='native_RZ_norm_lt_1e-10',
            boundary_policy='original_global_search', production_qualified=False)
        self.last_stats = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def advance(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('neighbor source cannot advance concurrently')
        stats = dict(batch_calls=0, point_visits=0, neighbor_accepted=0,
                     global_fallback_points=0, edge_ambiguous=0,
                     topology_bytes=self._topology_bytes)
        try:
            _validate_native(self.source)
            state = args[0] if args else kwargs.get('state')
            if state is not None and state.cloud.st.dtype != np.dtype(np.float64):
                raise ValueError('neighbor experiment requires float64 marker local coordinates')
            if configuration_digest(self.source.configuration) != self._source_configuration:
                raise ValueError('native source configuration changed; rebuild adapter')
            if self.source.mesh is not self._mesh or any(
                    not np.array_equal(getattr(self._mesh, name), value)
                    for name, value in self._snapshot.items()):
                raise ValueError('static native neighbor mesh changed; rebuild adapter')
            mesh = copy(self._mesh)

            def locate(positions, *, element_hints=None, local_hints=None):
                return self._nearby(positions, element_hints, local_hints, stats)

            def locate_batch(positions, **hints):
                stats['batch_calls'] += 1
                stats['point_visits'] += len(positions)
                return NativeSourceMesh.locate_batch(mesh, positions, **hints)

            mesh.locate = locate
            mesh.locate_batch = locate_batch
            candidate = copy(self.source)
            candidate.mesh = mesh
            return candidate.advance(*args, **kwargs)
        finally:
            self.last_stats = stats.copy()
            self._lock.release()

    def _nearby(self, positions, hints, local_hints, stats):
        native = self._mesh
        x = np.asarray(positions)
        elements = np.zeros(len(x), dtype=np.int32)
        local = np.zeros((len(x), 2))
        if hints is not None:
            e = np.asarray(hints).astype(np.int64)-1
            st = np.clip(np.asarray(local_hints), 0, 1).astype(np.float64)
            active = np.flatnonzero((e >= 0) & (e < len(native.gauss)))
            previous = np.full(len(x), -1, dtype=np.int64)
            for _ in range(self.max_hops):
                if not len(active):
                    break
                # A Newton predictor selects an edge, not a physical solution.
                # Every candidate must still pass the unmodified native solve.
                try:
                    rz, jac = native._location_geometries(e[active], st[active])
                    delta = np.linalg.solve(jac, (x[active, :2]-rz)[..., None])[..., 0]
                except (ValueError, np.linalg.LinAlgError):
                    break
                with np.errstate(divide='ignore', invalid='ignore'):
                    distance = np.where(delta > 0, 1-st[active], -st[active])
                    fraction = np.where(delta != 0, distance/delta, np.inf)
                fraction = np.where(fraction >= -1e-12, np.maximum(fraction, 0), np.inf)
                axis = np.argmin(fraction, axis=1)
                first = fraction[np.arange(len(active)), axis]
                second = fraction[np.arange(len(active)), 1-axis]
                ambiguous = (~np.isfinite(first) | (first > 1+1e-9) |
                             (np.abs(second-first) < 1e-10))
                stats['edge_ambiguous'] += int(ambiguous.sum())
                face = np.where(axis == 0, np.where(delta[:, 0] > 0, 1, 3),
                                np.where(delta[:, 1] > 0, 2, 0))
                neighbor = self._neighbors[e[active], face]
                keep = (~ambiguous) & (neighbor >= 0) & (neighbor != previous[active])
                ids = active[keep]
                if not len(ids):
                    break
                edge = np.clip(st[ids]+first[keep, None]*delta[keep], 0, 1)
                f = face[keep]
                along = np.choose(f, (edge[:, 0], edge[:, 1], 1-edge[:, 0], 1-edge[:, 1]))
                along = np.where(self._reverse[e[ids], f], 1-along, along)
                new_face = self._faces[e[ids], f]
                mapped = _edge_coordinates(new_face, along)
                previous[ids] = e[ids]
                e[ids] = neighbor[keep]
                st[ids] = mapped
                candidate_e, candidate_st = _local_solve(native, x[ids], e[ids]+1, mapped)
                # Near edges retain the original global candidate ordering.
                interior = (candidate_st > 1e-8).all(axis=1) & (candidate_st < 1-1e-8).all(axis=1)
                accepted = (candidate_e > 0) & interior
                boundary = (candidate_e > 0) & ~interior
                stats['edge_ambiguous'] += int(boundary.sum())
                elements[ids[accepted]] = candidate_e[accepted]
                local[ids[accepted]] = candidate_st[accepted]
                stats['neighbor_accepted'] += int(accepted.sum())
                active = ids[(candidate_e == 0)]
        pending = np.flatnonzero(elements == 0)
        stats['global_fallback_points'] += len(pending)
        if len(pending):
            pe, ps = native.locate(x[pending], element_hints=None if hints is None else np.asarray(hints)[pending],
                                  local_hints=None if local_hints is None else np.asarray(local_hints)[pending])
            elements[pending] = pe
            local[pending] = ps
        return elements, local


def _local_solve(native, positions, hints, st):
    """Reuse native bounded Newton without recursively searching globally."""
    mesh = copy(native)
    mesh.locate = lambda x, **kwargs: (np.zeros(len(x), dtype=np.int32), np.zeros((len(x), 2)))
    return NativeSourceMesh.locate_batch(mesh, positions, element_hints=hints, local_hints=st)


def _validate_native(source):
    if (type(source.mesh) is not NativeSourceMesh or
            source.spatial_backend != 'batched_v1' or source._spatial is not None):
        raise ValueError('unmodified native batched CPU source required')
    if any(name in source.mesh.__dict__ for name in (
            '_location_geometry', '_location_geometries', 'locate', 'locate_batch')):
        raise ValueError('overridden native geometry/locator is unsupported')


def _edge_coordinates(face, along):
    return np.column_stack((np.choose(face, (along, np.ones(len(face)), 1-along, np.zeros(len(face)))),
                            np.choose(face, (np.zeros(len(face)), along, np.ones(len(face)), 1-along))))


def _topology(vertex):
    count = vertex.shape[1]
    neighbors = np.full((count, 4), -1, dtype=np.int64)
    faces = np.full((count, 4), -1, dtype=np.int8)
    reverse = np.zeros((count, 4), dtype=bool)
    edges = {}
    for e in range(count):
        for f in range(4):
            a, b = int(vertex[f, e]), int(vertex[(f+1) % 4, e])
            edges.setdefault(tuple(sorted((a, b))), []).append((e, f, a, b))
    for entries in edges.values():
        if len(entries) != 2 or entries[0][0] == entries[1][0]:
            continue  # Missing or non-manifold edges keep the global policy.
        left, right = entries
        if left[2] == left[3] or right[2] == right[3]:
            continue
        for old, new in ((left, right), (right, left)):
            e, f, a, b = old
            neighbors[e, f], faces[e, f] = new[:2]
            reverse[e, f] = a == new[3] and b == new[2]
    return neighbors, faces, reverse
