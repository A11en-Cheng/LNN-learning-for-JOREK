"""Opt-in event-local bicubic functions; not a default production backend.

Full native degrees are converted once to polynomial coefficients. Values and
gradients use the same polynomial; the native locator/deposition/projection and
all particle steps are unchanged. Algebra is equivalent, not bitwise identical.
Compilation is explicit and outside all warm timings; no runtime installation.
"""
from copy import copy, deepcopy
import ctypes
import hashlib
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from .native_coupling import NativeSourceMesh

KERNEL_SOURCE = Path(__file__).with_suffix('.cpp')
COMPILER_FLAGS = ('-std=c++17', '-O3', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off')


def build_kernel(destination):
    """Explicit diagnostic build into a new caller-selected file."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['g++', *COMPILER_FLAGS, str(KERNEL_SOURCE), '-o', str(destination)],
                   check=True, capture_output=True, text=True)
    return destination


class CompiledElementFunctions:
    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        abi = self.library.native_element_function_abi
        abi.restype = ctypes.c_int64
        if abi() != 1:
            raise ValueError('unsupported element function kernel ABI')
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags=('C_CONTIGUOUS', 'ALIGNED'))
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags=('C_CONTIGUOUS', 'ALIGNED'))
        self.sample = self.library.native_element_function_sample
        self.sample.argtypes = [ctypes.c_int64, ip, fp, fp, fp, fp, fp, fp, fp, fp]
        self.sample.restype = ctypes.c_int64
        self.configuration = dict(abi=1, compiler='g++', compiler_flags=list(COMPILER_FLAGS),
            cpp_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


# Independent expanded native basis (Bezier-scaled derivative degrees).
_ONE = np.array([[[1.,0.,-3.,2.], [0.,3.,-6.,3.]],
                 [[0.,0.,3.,-2.], [0.,0.,3.,-3.]]])
_TRANSFORM = np.empty((4,4,4,4))
for _v, (_a, _b) in enumerate(((0,0),(1,0),(1,1),(0,1))):
    for _d, (_c, _k) in enumerate(((0,0),(1,0),(0,1),(1,1))):
        _TRANSFORM[_v,_d] = np.outer(_ONE[_a,_c], _ONE[_b,_k])
_TRANSFORM.flags.writeable = False


class NativeElementFunctionSource:
    """One source seam, fresh rebuildable cache per advance, no state mutation."""
    def __init__(self, source, *, kernel, max_cache_bytes=64*1024**2):
        from .native_nearby_locator import NativeNearbyMarkerSource
        self.nearby = isinstance(source, NativeNearbyMarkerSource)
        native = source.source if self.nearby else source
        if (type(native.mesh) is not NativeSourceMesh or native.spatial_backend != 'batched_v1'
                or native._spatial is not None):
            raise ValueError('native batched CPU source required for element functions')
        if not isinstance(kernel, CompiledElementFunctions):
            raise ValueError('explicit compiled element function kernel required')
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError('positive function cache budget required')
        self.source, self.native, self.kernel = source, native, kernel
        self.max_cache_bytes = max_cache_bytes
        self.configuration = deepcopy(source.configuration)
        self.configuration['experimental_element_functions'] = dict(
            version='event-owned-bicubic-monomial-horner-fp64-v1',
            algebra='full_native_cubic_equivalent_not_bitwise',
            lifetime='one_source_event_immutable_sampling_snapshots',
            locator='unchanged', deposition='unchanged', kernel=kernel.configuration,
            max_cache_bytes=max_cache_bytes, production_qualified=False)
        self.last_stats = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def sampling_cache(self):
        return _FunctionMesh(self.native.mesh, self.kernel, self.max_cache_bytes)

    def advance(self, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('element function source cannot advance concurrently')
        cache = None
        try:
            state = args[0] if args else kwargs.get('state')
            if state is not None and state.cloud.st.dtype != np.dtype(np.float64):
                raise ValueError('element function source requires float64 local coordinates')
            cache = self.sampling_cache()
            mesh = copy(self.native.mesh)
            mesh._marker_points = cache._marker_points
            native = copy(self.native); native.mesh = mesh
            if self.nearby:
                # Its delegating __getattr__ is not copy-protocol safe on an
                # uninitialized instance. Copy its dictionary without invoking it.
                candidate = object.__new__(type(self.source))
                candidate.__dict__.update(self.source.__dict__)
                candidate.source, candidate._mesh = native, mesh
            else:
                candidate = native
            return candidate.advance(*args, **kwargs)
        finally:
            self.last_stats = None if cache is None else cache.stats.copy()
            self._lock.release()


class _FunctionMesh:
    def __init__(self, native, kernel, budget):
        self.native, self.kernel, self.budget = native, kernel, budget
        self.node_ids = np.array(native.vertex.T, dtype=np.int64, copy=True)
        self.sizes = np.array(native.size.transpose(2,1,0), dtype=np.float64, copy=True)
        self._entries = {}
        self.geometry = None
        self.stats = dict(field_builds=0, feedback_builds=0, geometry_builds=0,
            build_seconds=0., sample_seconds=0., kernel_seconds=0., sample_calls=0,
            sample_points=0, coefficient_bytes=0, peak_coefficient_bytes=0,
            topology_bytes=self.node_ids.nbytes+self.sizes.nbytes)
        self.geometry = self._prepare(native.coordinates)
        self.stats['geometry_builds'] = 1

    def _prepare(self, fields):
        required = len(self.node_ids)*16*len(fields)*8
        live = sum(entry[1].nbytes for entry in self._entries.values())
        if self.geometry is not None:
            live += self.geometry.nbytes
        # Include native gather, scaled gather and the new coefficient buffer.
        peak = live+3*required+self.stats['topology_bytes']
        if peak > self.budget:
            raise MemoryError('element function cache budget exceeded')
        tick = time.perf_counter()
        local = np.asarray(fields, dtype=np.float64)[:,:,0,:][:,:,self.node_ids].transpose(2,0,3,1)
        local = local*self.sizes[:,None]
        coefficients = np.ascontiguousarray(np.einsum('efvd,vdij->efij', local, _TRANSFORM, optimize=True))
        if not np.isfinite(coefficients).all():
            raise ValueError('invalid full native function coefficients')
        coefficients.flags.writeable = False
        self.stats['build_seconds'] += time.perf_counter()-tick
        self.stats['peak_coefficient_bytes'] = max(self.stats['peak_coefficient_bytes'], peak)
        return coefficients

    def _local(self, name, fields):
        cached = self._entries.get(name)
        if cached is None or cached[0] is not fields:
            self.native.validate_fields(fields, 9 if name == 'field' else 5)
            if fields.flags.writeable:
                raise ValueError('only event-owned immutable sampling snapshots may be cached')
            coefficients = self._prepare(fields)
            self._entries[name] = (fields, coefficients)
            self.stats[name+'_builds'] += 1
            self.stats['coefficient_bytes'] = self.geometry.nbytes+sum(x[1].nbytes for x in self._entries.values())
        return self._entries[name][1]

    def _marker_points(self, e, st, fields, feedback):
        tick = time.perf_counter()
        try:
            e, st = np.asarray(e), np.asarray(st)
            if (e.ndim != 1 or e.dtype.kind not in 'iu' or (e<0).any()
                    or (e>=len(self.node_ids)).any() or st.shape != (len(e),2)
                    or not np.isfinite(st).all()):
                raise ValueError('invalid element/local function query')
            e = np.ascontiguousarray(e, dtype=np.int64)
            st = np.ascontiguousarray(st, dtype=np.float64)
            f, p = self._local('field', fields), self._local('feedback', feedback)
            values = np.zeros((len(e),9)); gradients = np.zeros((len(e),9,2))
            projected = np.zeros((len(e),5)); rz = np.empty((len(e),2))
            kernel_tick = time.perf_counter()
            invalid = self.kernel.sample(len(e), e, st, self.geometry, f, p,
                                          values, gradients, projected, rz)
            self.stats['kernel_seconds'] += time.perf_counter()-kernel_tick
            if invalid or any(not np.isfinite(a).all() for a in (values, gradients, projected, rz)):
                raise ValueError('invalid native function point geometry/result')
            return values, gradients, projected, rz
        finally:
            self.stats['sample_seconds'] += time.perf_counter()-tick
            self.stats['sample_calls'] += 1
            self.stats['sample_points'] += len(e) if e.ndim == 1 else 0
