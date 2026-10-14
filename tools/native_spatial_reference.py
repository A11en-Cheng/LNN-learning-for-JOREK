"""Frozen pre-sharing CPU arithmetic for explicit benchmark baselines only.

Copied from the independent legacy oracle in test_native_shared_spatial.
No imports from tests, no production-class mutation, no configuration changes.
This is not a general-purpose replacement or checkpoint restoration policy.

Historical points had no per-call full-field finite/layout validation: source
advance validated complete fields once. This reference intentionally retains
that work. The new production helper's extra per-call validation remains in
the candidate timing; equivalence here is for legitimate finite input fields.
"""
from types import MethodType
import numpy as np
from src.models.native_coupling import _cubic_basis

IMPLEMENTATION = 'frozen_pre_shared_geometry_literal_v1'


def legacy_points(mesh, e, st, values):
    e = np.asarray(e); st = np.asarray(st); a = np.asarray(values)
    if (e.ndim != 1 or e.dtype.kind not in 'iu' or (e < 0).any()
            or (e >= len(mesh.gauss)).any() or st.shape != (len(e), 2)
            or not np.isfinite(st).all()):
        raise ValueError('invalid batch element/local coordinates')
    h, hs, ht = _cubic_basis(st[:, 0], st[:, 1])
    sizes = mesh.size[:, :, e].transpose(2, 1, 0)
    b = h*sizes; bs = hs*sizes; bt = ht*sizes; ids = mesh.vertex[:, e].T
    x = mesh.coordinates[:, :, 0, :][:, :, ids].transpose(2, 0, 3, 1)
    xyz = np.einsum('bvd,bfvd->bf', b, x)
    xs = np.einsum('bvd,bfvd->bf', bs, x)
    xt = np.einsum('bvd,bfvd->bf', bt, x)
    jac = xs[:, 0]*xt[:, 1]-xt[:, 0]*xs[:, 1]
    if not np.isfinite(jac).all() or (jac <= 0).any() or (xyz[:, 0] <= 0).any():
        raise ValueError('invalid native point geometry')
    local = a[:, :, 0, :][:, :, ids].transpose(2, 0, 3, 1)
    val = np.einsum('bvd,bfvd->bf', b, local)
    ds = np.einsum('bvd,bfvd->bf', bs, local)
    dt = np.einsum('bvd,bfvd->bf', bt, local)
    grad = np.stack(((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/jac[:, None],
                     (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/jac[:, None]), axis=-1)
    return val, grad, xyz, b, np.stack((xs, xt), axis=-1)


def legacy_deposit(mesh, cloud, ids, load):
    e = cloud.i_elm[ids].astype(np.int64)-1
    basis = legacy_points(mesh, e, cloud.st[ids], np.zeros((1, 4, 1, mesh.n_nodes)))[3]
    dofs = mesh.inverse[:, mesh.vertex[:, e].T].transpose(1, 2, 0).ravel()
    contributions = (load[:, :, None, None]*basis[:, None, :, :]).transpose(1, 0, 2, 3).reshape(5, -1)
    rhs = np.zeros((5, mesh.n_dofs))
    for channel in range(5):
        np.add.at(rhs[channel], dofs, contributions[channel])
    return rhs


def legacy_points_many(mesh, e, st, fields):
    return tuple(legacy_points(mesh, e, st, values) for values in fields)


def bind_cpu_shared_reference(source):
    """Install old literal work on this source mesh only; no GPU/scalar mixing."""
    if source.spatial_backend != 'batched_v1':
        raise ValueError('CPU shared reference requires batched_v1 backend')
    source.mesh.points_many = MethodType(legacy_points_many, source.mesh)
    source.mesh.deposit_batch = MethodType(legacy_deposit, source.mesh)
    source.benchmark_cpu_spatial_implementation = IMPLEMENTATION
    return IMPLEMENTATION
