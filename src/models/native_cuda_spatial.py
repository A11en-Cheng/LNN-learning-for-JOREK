"""Experimental FP64 spatial execution; NumPy boundaries, CPU search fallback.

Not numerically qualified for training/production: the 2026-10-09 full injInner
comparison found an element-identity divergence. Keep CPU as the default.
See report/native-cuda-spatial-performance-2026-10-09.md for evidence.

CUDA arithmetic is numerically comparable, not promised bitwise NumPy-identical.
An event_scope owns read-only mesh geometry for its lifetime. Named field slots
are device snapshots, replaced explicitly at each changed sampling time.
"""
from contextlib import contextmanager

import numpy as np
import torch

from .native_state import configuration_digest


class NativeCudaSpatial:
    """Standalone points/hinted-locator adapter; CPU device is test-only.

    Do not mutate mesh geometry/configuration during event_scope. Unscoped
    calls validate the binding each time. No particle or ndarray-identity cache.
    """

    def __init__(self, mesh, *, device='cuda'):
        self.device = torch.device(device)
        if self.device.type not in ('cuda', 'cpu'):
            raise ValueError('spatial device must be explicit CUDA or CPU test device')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA spatial backend requires available CUDA')
        self.mesh = mesh
        self.n_nodes = mesh.n_nodes
        self.n_elements = len(mesh.gauss)
        self._binding = self._binding_digest()
        self._fields = {}
        self._event = None
        self._in_event = False
        self._vertex = self._tensor(mesh.vertex.T, dtype=torch.int64)
        self._size = self._tensor(mesh.size.transpose(2, 1, 0))
        x = mesh.coordinates[:, :, 0, :][:, :, mesh.vertex.T].transpose(2, 0, 3, 1)
        self._coordinates = self._tensor(x)
        self._bounds = np.array(mesh._location_bounds, copy=True)
        self._s_index = self._tensor([[0, 1, 0, 1], [2, 3, 2, 3],
                                     [2, 3, 2, 3], [0, 1, 0, 1]], dtype=torch.int64)
        self._t_index = self._tensor([[0, 0, 1, 1], [0, 0, 1, 1],
                                     [2, 2, 3, 3], [2, 2, 3, 3]], dtype=torch.int64)
        self.configuration = dict(
            version='native-cuda-spatial-fp64-v1', backend='cuda_fp64_v1',
            device=str(self.device), test_only=self.device.type == 'cpu',
            geometry=mesh.configuration['geometry'], mesh_binding=self._binding,
            precision='float64', field_transfer='event_owned_named_snapshots_v1',
            locator='hinted_newton_cpu_scalar_fallback', max_iterations=25,
            damping=[1., .5, .25, .125, .0625, .03125], bounds_tolerance=1e-9,
            residual_tolerance=1e-10, roundoff_factor=256., edge_guard=1e-8)

    def _tensor(self, array, *, dtype=torch.float64):
        # Always own storage, even on the CPU testing device.
        return torch.tensor(np.asarray(array).copy(), dtype=dtype, device=self.device)

    def _binding_digest(self):
        m = self.mesh
        return configuration_digest(dict(
            configuration=m.configuration, units=vars(m.units),
            x=m.coordinates, size=m.size, vertex=m.vertex, indices=m.indices,
            bounds=m._location_bounds, centers=m._location_centers,
            radius=m._location_radius, tree=m._location_tree.data))

    def _validate_binding(self):
        if self._binding_digest() != self._binding:
            raise ValueError('spatial mesh/configuration binding changed; rebuild backend')

    def _guard(self):
        if not self._in_event:
            self._validate_binding()

    def reset_event_cache(self, event_binding=None):
        """Clear all field snapshots; validate mesh once before a new event."""
        if self._in_event:
            raise ValueError('cannot reset an active spatial event scope')
        self._fields.clear()
        self._event = None
        self._validate_binding()
        if event_binding is not None:
            self._event = configuration_digest(event_binding)

    @contextmanager
    def event_scope(self, event_binding):
        """Caller guarantees read-only mesh/config until scope exit; no nesting."""
        if self._in_event or event_binding is None:
            raise ValueError('explicit nonnested spatial event binding required')
        self.reset_event_cache(event_binding)
        self._in_event = True
        try:
            yield self
        finally:
            self._fields.clear()
            self._event = None
            self._in_event = False

    def _field_tensor(self, values):
        a = np.asarray(values)
        if (a.ndim != 4 or a.shape[1:] != (4, 1, self.n_nodes)
                or a.dtype.kind not in 'fiu' or not np.isfinite(a).all()):
            raise ValueError('finite full native field coefficients [F,4,1,N] required')
        return self._tensor(a[:, :, 0, :])

    def freeze_fields(self, key, values):
        """Upload an owned snapshot; same key replaces its previous time slot."""
        if not self._in_event or not isinstance(key, str) or not key:
            raise ValueError('named fields require an active spatial event scope')
        # Remove first, so a rejected replacement cannot silently reuse old data.
        self._fields.pop(key, None)
        self._fields[key] = self._field_tensor(values)

    def _inputs(self, e, st):
        e = np.asarray(e)
        st = np.asarray(st)
        if (e.ndim != 1 or e.dtype.kind not in 'iu' or (e < 0).any()
                or (e >= self.n_elements).any() or st.shape != (len(e), 2)
                or not np.isfinite(st).all()):
            raise ValueError('invalid batch element/local coordinates')
        return self._tensor(e, dtype=torch.int64), self._tensor(st)

    def _basis(self, st):
        def one(x):
            h = torch.stack(((1-x)**2*(1+2*x), 3*x*(1-x)**2,
                             x*x*(3-2*x), 3*(1-x)*x*x), dim=1)
            d = torch.stack((6*x*(x-1), 3*(x-1)*(3*x-1),
                             -6*x*(x-1), -3*x*(3*x-2)), dim=1)
            return h, d
        hs, ds = one(st[:, 0])
        ht, dt = one(st[:, 1])
        return (hs[:, self._s_index]*ht[:, self._t_index],
                ds[:, self._s_index]*ht[:, self._t_index],
                hs[:, self._s_index]*dt[:, self._t_index])

    def _geometry(self, e, st):
        h, hs, ht = self._basis(st)
        size = self._size[e]
        b, bs, bt = h*size, hs*size, ht*size
        x = self._coordinates[e]
        rz = torch.einsum('bvd,bfvd->bf', b, x)
        xs = torch.einsum('bvd,bfvd->bf', bs, x)
        xt = torch.einsum('bvd,bfvd->bf', bt, x)
        det = xs[:, 0]*xt[:, 1]-xt[:, 0]*xs[:, 1]
        invalid = ~torch.isfinite(det) | (det <= 0) | (rz[:, 0] <= 0)
        if bool(invalid.any()):
            raise ValueError('invalid native point geometry')
        return rz, torch.stack((xs, xt), dim=-1), det, b, bs, bt

    @torch.no_grad()
    def _points(self, e, st, fields):
        e, st = self._inputs(e, st)
        geometry = self._geometry(e, st)
        val, grad = self._sample_fields(e, geometry, fields)
        rz, jac, _, b, _, _ = geometry
        return tuple(a.cpu().numpy() for a in (val, grad, rz, b, jac))

    def _sample_fields(self, e, geometry, fields):
        _, jac, det, b, bs, bt = geometry
        local = fields[:, :, self._vertex[e]].permute(2, 0, 3, 1)
        val = torch.einsum('bvd,bfvd->bf', b, local)
        ds = torch.einsum('bvd,bfvd->bf', bs, local)
        dt = torch.einsum('bvd,bfvd->bf', bt, local)
        xs, xt = jac[:, :, 0], jac[:, :, 1]
        grad = torch.stack(((xt[:, 1, None]*ds-xs[:, 1, None]*dt)/det[:, None],
                            (-xt[:, 0, None]*ds+xs[:, 0, None]*dt)/det[:, None]), dim=-1)
        return val, grad

    def points(self, e, st, values):
        """Same five NumPy outputs as mesh.points; fresh field upload each call."""
        self._guard()
        return self._points(e, st, self._field_tensor(values))

    def points_frozen(self, key, e, st):
        """Sample an explicitly uploaded event/time snapshot without reupload."""
        if not self._in_event or key not in self._fields:
            raise ValueError('missing frozen field in active spatial event scope')
        return self._points(e, st, self._fields[key])

    @torch.no_grad()
    def points_frozen_many(self, keys, e, st):
        """Return five-output tuples in key order, sharing geometry within this call.

        Geometry NumPy outputs are shared between the returned tuples. Field
        contractions stay separate and use the same arithmetic as points_frozen.
        Nothing about the particle coordinates is retained after this call.
        """
        if not self._in_event:
            raise ValueError('frozen fields require an active spatial event scope')
        if not isinstance(keys, (tuple, list)) or any(not isinstance(k, str) for k in keys):
            raise ValueError('field keys must be a nonempty list or tuple of names')
        if not keys:
            raise ValueError('nonempty field keys required')
        if any(key not in self._fields for key in keys):
            raise ValueError('missing frozen field in active spatial event scope')
        e, st = self._inputs(e, st)
        geometry = self._geometry(e, st)
        sampled = [self._sample_fields(e, geometry, self._fields[key]) for key in keys]
        rz, jac, _, b, _, _ = geometry
        shared = tuple(a.cpu().numpy() for a in (rz, b, jac))
        return tuple((val.cpu().numpy(), grad.cpu().numpy(), *shared) for val, grad in sampled)

    @torch.no_grad()
    def locate_batch(self, positions, *, element_hints=None, local_hints=None):
        """Original hinted Newton policy; ambiguous/unresolved rows use CPU locate."""
        self._guard()
        x = np.asarray(positions)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError('finite R/Z/phi positions required')
        if element_hints is None:
            return self.mesh.locate(x)
        hints = np.asarray(element_hints)
        original_st = np.asarray(local_hints)
        if (hints.shape != (len(x),) or original_st.shape != (len(x), 2)
                or not np.isfinite(original_st).all()):
            raise ValueError('matching element/local hints required')
        elements = torch.zeros(len(x), dtype=torch.int32, device=self.device)
        local = torch.zeros((len(x), 2), dtype=torch.float64, device=self.device)
        e_np = hints.astype(np.int64)-1
        active_np = np.flatnonzero((e_np >= 0) & (e_np < self.n_elements))
        bounds = self._bounds[e_np[active_np]]
        inside = ((x[active_np, :2] >= bounds[:, 0]-1e-9)
                  & (x[active_np, :2] <= bounds[:, 1]+1e-9)).all(axis=1)
        active = self._tensor(active_np[inside], dtype=torch.int64)
        e = self._tensor(e_np, dtype=torch.int64)
        target = self._tensor(x[:, :2])
        st = self._tensor(np.clip(original_st, 0, 1))
        eps = 256*np.finfo(np.float64).eps
        margin = eps*torch.maximum(torch.ones(len(x), device=self.device, dtype=torch.float64),
                                   target.abs().amax(dim=1))
        for _ in range(25):
            if not len(active):
                break
            try:
                rz, jac, det, *_ = self._geometry(e[active], st[active])
                error = torch.linalg.vector_norm(rz-target[active], dim=1)
                # Do not let roundoff select a different convergence branch or
                # amplify through a nearly singular local inverse.
                uncertain = ((error-1e-10).abs() <= margin[active]) | (
                    det <= eps*jac.abs().amax(dim=(1, 2)).square())
                done = error < 1e-10
                edge = (st[active].amin(dim=1) <= 1e-8) | (st[active].amax(dim=1) >= 1-1e-8)
                uncertain |= done & edge
                ids = active[done & ~uncertain]
                elements[ids] = (e[ids]+1).to(torch.int32)
                local[ids] = st[ids]
                remain = ~done & ~uncertain
                active = active[remain]
                if not len(active):
                    break
                step, info = torch.linalg.solve_ex(jac[remain], (rz[remain]-target[active]).unsqueeze(-1))
                if bool((info != 0).any()):
                    break
                step = step[..., 0]
                error = error[remain]
                accepted = torch.zeros(len(active), dtype=torch.bool, device=self.device)
                ambiguous = torch.zeros_like(accepted)
                for damping in (1., .5, .25, .125, .0625, .03125):
                    todo = torch.nonzero(~accepted & ~ambiguous).flatten()
                    if not len(todo):
                        break
                    ids = active[todo]
                    trial = torch.clamp(st[ids]-damping*step[todo], 0, 1)
                    trial_rz = self._geometry(e[ids], trial)[0]
                    trial_error = torch.linalg.vector_norm(trial_rz-target[ids], dim=1)
                    ambiguous[todo] = (trial_error-error[todo]).abs() <= margin[ids]
                    better = (trial_error < error[todo]) & ~ambiguous[todo]
                    st[ids[better]] = trial[better]
                    accepted[todo[better]] = True
                active = active[accepted & ~ambiguous]
            except ValueError:
                break  # Same batch geometry failure -> scalar fallback policy.
        elements = elements.cpu().numpy()
        local = local.cpu().numpy()
        pending = np.flatnonzero(elements == 0)
        if len(pending):
            pe, ps = self.mesh.locate(x[pending], element_hints=hints[pending],
                                      local_hints=original_st[pending])
            elements[pending], local[pending] = pe, ps
        return elements, local
