"""Explicit opt-in Ne ADAS interpolation behind the existing callable seam.

The compiler removes repeated Python indexing/arithmetic, not table resolution
or particle steps. NumPy logarithms, per-charge powers, units and mapping stay
literal. Owned immutable snapshots support concurrent queries without caches.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .native_coupling import NativeNeonAtomicRates
from .native_state import configuration_digest

KERNEL_SOURCE = Path(__file__).with_suffix('.cpp')
COMPILER_FLAGS = ('-std=c++17', '-O3', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off')


def build_kernel(destination):
    """Caller requests one explicit build; never overwrite or auto-install."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['g++', *COMPILER_FLAGS, str(KERNEL_SOURCE), '-o', str(destination)],
        check=True, capture_output=True, text=True)
    return destination


class CompiledAtomicInterpolation:
    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        abi = self.library.native_atomic_interpolation_abi
        abi.restype = ctypes.c_int64
        if abi() != 1:
            raise ValueError('unsupported compiled atomic interpolation ABI')
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags=('C_CONTIGUOUS', 'ALIGNED'))
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags=('C_CONTIGUOUS', 'ALIGNED'))
        self.interpolate = self.library.native_atomic_log_interpolate
        self.interpolate.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ip, fp, fp, fp, fp]
        self.interpolate.restype = ctypes.c_int64
        self.configuration = dict(abi=1, compiler='g++', compiler_flags=list(COMPILER_FLAGS),
            cpp_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


class CompiledNeonAtomicRates:
    """Same SI/eV inputs and SCD/ACD/radiation outputs, different fingerprint.

    FP64 tables are frozen once, axes searched independently, and the existing NumPy
power kernel sees the same contiguous query vector length for each charge row.
This avoids silently replacing NumPy powers with platform scalar libm powers.
"""
    def __init__(self, reference, *, kernel, max_table_bytes=64*1024**2, max_queries=1000000):
        if type(reference) is not NativeNeonAtomicRates or not isinstance(kernel, CompiledAtomicInterpolation):
            raise ValueError('explicit native Ne provider and compiled atomic kernel required')
        if (type(max_table_bytes) is not int or max_table_bytes <= 0 or
            type(max_queries) is not int or max_queries <= 0):
            raise ValueError('positive atomic table/query budgets required')
        tables = [reference.tables.scd, reference.tables.acd, reference.tables.plt]
        if reference.prb is not None:
            tables.append(reference.prb)
        total = 0
        normalized = []
        for table in tables:
            if not isinstance(table, (tuple, list)) or len(table) != 3:
                raise ValueError('complete atomic table required')
            n, t, c = map(np.asarray, table)
            if (n.ndim != 1 or t.ndim != 1 or min(len(n), len(t)) < 2 or
                c.shape != (10, len(t), len(n)) or
                any(a.dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in (n, t, c)) or
                not (np.diff(n) > 0).all() or not (np.diff(t) > 0).all()):
                raise ValueError('invalid atomic table axes/charge rows')
            if any(a.dtype.kind != 'f' or a.dtype.itemsize != 8 for a in (n, t, c)):
                # Promoting a lower-precision axis before its subtraction can
                # change interval widths. Do not silently change table arithmetic.
                raise ValueError('compiled atomic interpolation requires FP64 table arithmetic')
            total += 8*(n.size+t.size+c.size)+5*8
            if total > max_table_bytes:
                raise MemoryError('atomic table snapshot exceeds budget')
            normalized.append((n, t, c))
        blocks = []; meta = []; cursor = 0
        for n, t, c in normalized:
            offsets = []
            for a in (n, t, c):
                offsets.append(cursor)
                owned = np.array(a, dtype=np.float64, order='C', copy=True).ravel()
                blocks.append(owned); cursor += owned.size
            meta.append((offsets[0], len(n), offsets[1], len(t), offsets[2]))
        self._data = np.concatenate(blocks)
        self._meta = np.array(meta, dtype=np.int64)
        self._data.flags.writeable = self._meta.flags.writeable = False
        self.kernel, self.max_queries = kernel, max_queries
        self.configuration = dict(version='compiled-Ne-log-bilinear-numpy-row-power-v1',
            reference_version=reference.version, tables=configuration_digest((self._meta, self._data)),
            table_count=len(tables), power='unchanged_numpy_contiguous_charge_rows',
            units='eV_m^-3_to_m^3_per_s_and_W_per_atom', charge_rows='PLT0:9_PRB1:10',
            table_bytes=self._data.nbytes+self._meta.nbytes,
            max_table_bytes=max_table_bytes, max_queries=max_queries,
            kernel=kernel.configuration.copy(), production_qualified=False)
        self.version = self.configuration['version']+':'+configuration_digest(self.configuration)

    def __call__(self, te_ev, ne_si):
        ne, te = np.asarray(ne_si), np.asarray(te_ev)
        if (te.shape != ne.shape or te.ndim != 1 or not len(te) or
            ne.dtype.kind not in 'fiub' or te.dtype.kind not in 'fiub' or
            not np.isfinite(te).all() or not np.isfinite(ne).all() or min(te.min(), ne.min()) <= 0):
            raise ValueError('positive finite atomic SI density/eV temperature required')
        if len(te) > self.max_queries:
            raise MemoryError('atomic query exceeds explicit budget')
        # Retain the reference's input dtype operations before float64 query
        # normalization, notably float32 density conversion and logarithms.
        logn = np.ascontiguousarray(np.log10(ne*1e-6), dtype=np.float64)
        logt = np.ascontiguousarray(np.log10(te), dtype=np.float64)
        logs = np.empty((len(self._meta), 10, len(te)), dtype=np.float64)
        if self.kernel.interpolate(len(te), len(self._meta), len(self._data),
            self._meta, self._data, logn, logt, logs):
            raise ValueError('invalid compiled atomic table metadata')
        # Literal reference power placement and array length per charge.
        values = [np.stack([10.**row for row in table], axis=1) for table in logs]
        scd, acd = values[0]*1e-6, values[1]*1e-6
        rad = np.zeros((len(te), 11))
        rad[:, :10] = values[2]*(ne*1e-6)[:, None]
        if len(values) == 4:
            rad[:, 1:] += values[3]*(ne*1e-6)[:, None]
        return scd, acd, rad
