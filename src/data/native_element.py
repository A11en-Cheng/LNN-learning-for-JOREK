"""Explicit one-file geometry read for the local native Gauss reference chain.

No field/projection search, synchronization by filename, or future frame read.
Fields and raw projection deltas come from the caller's persistent state.
"""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np

from src.models.native_gauss import NativeElementGeometry, ToroidalSampling


@dataclass(frozen=True)
class NativeElement:
    geometry: NativeElementGeometry
    node_indices: np.ndarray      # zero-based; native vertex order preserved
    n_nodes: int
    toroidal: ToroidalSampling
    provenance: dict

    def gather(self, values):
        """Select all degrees/harmonics; no change to native field normalization."""
        if values is None:
            raise ValueError('missing full native values/deltas')
        a = np.asarray(values)
        if (a.ndim != 4 or a.shape[0] < 1
                or a.shape[1:] != (4, len(self.toroidal.modes), self.n_nodes)
                or not np.isfinite(a).all()):
            raise ValueError('requires full finite native [field,degree,harmonic,node]')
        return a[:,:,:,self.node_indices].copy()


def read_native_element(path, *, element_index: int):
    """Read a zero-based element and explicit native sampling metadata.

    For multi-harmonic states mode must be saved explicitly; no inferred mode
    plan. Source ordinary geometry uses only axisymmetric coordinate mode1.
    """
    if not isinstance(element_index, int) or element_index < 0:
        raise ValueError('element_index must be a nonnegative zero-based integer')
    with h5py.File(path,'r') as f:
        def scalar(name):
            if name not in f or f[name].size != 1:
                raise ValueError(f'missing native scalar {name}')
            v = float(np.ravel(f[name][()])[0])
            if not np.isfinite(v) or v != int(v):
                raise ValueError(f'invalid native integer {name}')
            return int(v)
        if scalar('jorek_model') != 502 or scalar('n_order') != 3 or scalar('n_coord_tor') != 1:
            raise ValueError('requires model502 cubic axisymmetric native geometry')
        for k in ('x','size','vertex'):
            if k not in f:
                raise ValueError(f'missing native geometry {k}')
        vertices = f['vertex']
        if vertices.ndim != 2 or vertices.shape[0] != 4 or element_index >= vertices.shape[1]:
            raise ValueError('invalid native vertex layout/element index')
        ids = np.asarray(vertices[:,element_index])
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('native vertices must be integer IDs')
        ids = ids.astype(np.int64)-1
        x = f['x']
        if x.ndim != 4 or x.shape[:3] != (2,4,1):
            raise ValueError('requires full native x [2,4,1,node]')
        if (ids < 0).any() or (ids >= x.shape[3]).any() or len(set(ids)) != 4:
            raise ValueError('invalid/repeated native vertex IDs')
        if f['size'].shape != (4,4,vertices.shape[1]):
            raise ValueError('requires native size [degree,vertex,element]')
        # h5py fancy indexing requires sorted IDs; select explicitly in source order.
        coords = np.stack([x[:,:,:,int(i)] for i in ids], axis=-1)
        size = np.asarray(f['size'][:,:,element_index])
        if not np.isfinite(coords).all() or not np.isfinite(size).all():
            raise ValueError('nonfinite native geometry')
        nh = scalar('n_tor')
        if nh == 1:
            modes = (0,)
        elif 'mode' in f:
            raw_modes = np.ravel(f['mode'][()])
            if not np.issubdtype(raw_modes.dtype, np.integer):
                raise ValueError('native mode plan must contain integers')
            modes = tuple(int(m) for m in raw_modes)
            if len(modes) != nh:
                raise ValueError('native mode plan length mismatch')
        else:
            raise ValueError('missing explicit native toroidal mode plan')
        tor = ToroidalSampling(modes, scalar('n_plane'), scalar('n_period'))
        tor.basis()  # validate, without changing values
    return NativeElement(NativeElementGeometry(coords,size,3), ids, x.shape[3], tor,
        {'path': str(Path(path).resolve()), 'element_index': element_index,
         'layout': 'native_hdf5', 'geometry_coordinate_modes': 1})
