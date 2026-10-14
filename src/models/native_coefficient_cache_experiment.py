"""Experimental event-local coefficients; unchanged native geometry/locator.

No bilinear lookup, no dense field grid, no worker pool and no default wiring.
Only immutable snapshots created inside NativeMarkerSource.advance may be
cached. A fresh cache per event prevents reuse after the same input changes.
"""
from copy import copy, deepcopy
import threading
import time

import numpy as np

from .native_coupling import NativeSourceMesh


class NativeCoefficientCachedSource:
    def __init__(self, source, *, max_cache_bytes=64*1024**2):
        if (type(source.mesh) is not NativeSourceMesh or
                source.spatial_backend != 'batched_v1' or source._spatial is not None):
            raise ValueError('unmodified native batched CPU source required')
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError('positive coefficient cache byte budget required')
        self.source = source
        self.max_cache_bytes = max_cache_bytes
        self.configuration = deepcopy(source.configuration)
        self.configuration['experimental_coefficient_cache'] = dict(
            version='event-owned-native-element-coefficients-v1',
            lifetime='one_source_advance_immutable_sampling_snapshots',
            layout='native_gather_element_vertex_field_degree_keeporder',
            geometry='unchanged_native', locator='unchanged_native',
            max_cache_bytes=max_cache_bytes, production_qualified=False)
        self.last_stats = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def advance(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('coefficient cache source cannot advance concurrently')
        mesh = None
        try:
            mesh = _CoefficientMesh(self.source.mesh, self.max_cache_bytes)
            candidate = copy(self.source)
            candidate.mesh = mesh
            return candidate.advance(*args, **kwargs)
        finally:
            self.last_stats = None if mesh is None else mesh.stats.copy()
            self._lock.release()


class _CoefficientMesh:
    def __init__(self, native, max_bytes):
        self.native = native
        self.node_ids = np.array(native.vertex.T, copy=True)
        self.node_ids.flags.writeable = False
        self.max_bytes = max_bytes
        self._entries = {}
        self.stats = dict(field_builds=0, feedback_builds=0, sample_calls=0,
            build_seconds=0., sample_seconds=0., particle_gather_seconds=0.,
            coefficient_bytes=0, peak_coefficient_bytes=0, topology_bytes=self.node_ids.nbytes)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _local(self, name, fields, e):
        cached = self._entries.get(name)
        if cached is None or cached[0] is not fields:
            if fields.flags.writeable:
                raise ValueError('only event-owned immutable source snapshots may be cached')
            required = len(self.node_ids)*16*len(fields)*fields.dtype.itemsize
            # Include simultaneously live old/new buffers during dynamic rebuild.
            live = sum(entry[1].nbytes for entry in self._entries.values())
            if live+required+self.node_ids.nbytes > self.max_bytes:
                raise MemoryError('native coefficient cache budget exceeded')
            tick = time.perf_counter()
            # Do NOT C-pack E,F,V,D: that changes einsum's reduction order.
            # Native advanced-node gather owns E,V,F,D (or its F-input variant).
            coefficients = fields[:, :, 0, :][:, :, self.node_ids].transpose(2, 3, 0, 1)
            coefficients.flags.writeable = False
            self._entries[name] = (fields, coefficients)
            self.stats[name+'_builds'] += 1
            self.stats['build_seconds'] += time.perf_counter()-tick
            self.stats['coefficient_bytes'] = sum(entry[1].nbytes for entry in self._entries.values())
            self.stats['peak_coefficient_bytes'] = max(self.stats['peak_coefficient_bytes'], live+required)
        tick = time.perf_counter()
        # Select on the native vertex/field/degree layout, then expose E,F,V,D.
        local = self._entries[name][1][e].transpose(0, 2, 1, 3)
        self.stats['particle_gather_seconds'] += time.perf_counter()-tick
        return local

    def _marker_points(self, e, st, fields, feedback):
        tick = time.perf_counter()
        try:
            # Native geometry arithmetic/guards and all locator calls are untouched.
            ids, b, bs, bt, xyz, xs, xt, jac = self.native._points_geometry(e, st)
            field_local = self._local('field', fields, e)
            feedback_local = self._local('feedback', feedback, e)
            values = np.zeros((len(e), 9))
            gradients = np.zeros((len(e), 9, 2))
            projected = np.zeros((len(e), 5))
            for channels in (slice(4, 5), slice(6, 9)):
                values[:, channels] = np.einsum('bvd,bfvd->bf', b, field_local[:, channels])
            for channels in (slice(0, 2), slice(7, 8)):
                ds = np.einsum('bvd,bfvd->bf', bs, field_local[:, channels])
                dt = np.einsum('bvd,bfvd->bf', bt, field_local[:, channels])
                gradients[:, channels] = np.stack(
                    ((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/jac[:, None],
                     (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/jac[:, None]), axis=-1)
            projected[:, 3:5] = np.einsum('bvd,bfvd->bf', b, feedback_local[:, 3:5])
            return values, gradients, projected, xyz
        finally:
            self.stats['sample_calls'] += 1
            self.stats['sample_seconds'] += time.perf_counter()-tick
