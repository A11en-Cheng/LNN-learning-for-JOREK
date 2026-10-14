"""Stateless, current-cubic-psi double-X diagnostic, not the full JOREK ES.

Fixed-boundary H1 only. Restart LCFS history is supplied explicitly once per
epoch; no recorded field, future reader, hidden previous axis or fallback ES.
Numerical root search is stricter than JOREK's failed-search approximations.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import root
from .native_gauss import _cubic_basis
from .native_source_nonlocal import NativeFluxEquilibrium
from .native_state import configuration_digest


class NativeEquilibriumProvider:
    """Return fresh diagnostics from ``fields[0]`` at the requested time.

    ``axis_seed`` is a fixed initialization search center, not a flux value.
    ``restart_lcfs_lost`` summarizes JOREK's fixed index_start history prefix;
    a new restart epoch requires a newly configured provider. Outside this
    supported topology/search region, raise ValueError for event rollback.
    This NumPy diagnostic is an explicit stop-gradient path.
    """

    @classmethod
    def from_initial_record(cls,mesh,initial):
        """Initialize from one trusted record, reading ONLY its source_path.

        These HDFs do not save search parameters. The two audited namelist
        SHA256s explicitly select the EHL-2 fixed-boundary double-X profile
        (namelist lines 130/131/133/136/138). Unknown configurations fail;
        no directory traversal, adjacent HDF or namelist read is performed.
        """
        import hashlib
        from pathlib import Path
        import h5py
        supported={
            '0fca2b6a5d0cf94e92eb86a4ef11d961fba50a28d06520708e19dfec3d061811',
            'bf3de403f17ab713d54f51883eee72699fa9ff3e5dd0549588bfffbd6c6e2c0e',
        }
        if initial.namelist_sha256 not in supported:
            raise ValueError('unknown initialization namelist: explicit audited search parameters required')
        geometry=configuration_digest(dict(x=mesh.coordinates,size=mesh.size,vertex=mesh.vertex,indices=mesh.indices))
        if geometry!=initial.geometry_digest:
            raise ValueError('initial record and equilibrium mesh geometry mismatch')
        path=Path(initial.source_path)
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
        if sha.hexdigest()!=initial.source_sha256:
            raise ValueError('initial source file changed since record binding')
        with h5py.File(path,'r') as f:
            required=('index_now','t_now','t_norm','R_axis_t','Z_axis_t','freeboundary')
            if any(k not in f for k in required):raise ValueError('initial HDF missing equilibrium seed evidence')
            def scalar(key):
                a=np.asarray(f[key][...])
                if a.size!=1:raise ValueError('initial HDF scalar required: '+key)
                return a.ravel()[0]
            index=float(scalar('index_now'));time=float(scalar('t_now'))*float(scalar('t_norm'))
            if index!=initial.step_index or time!=initial.time_si:
                raise ValueError('initial seed index/time mismatch')
            r=f['R_axis_t'][...];z=f['Z_axis_t'][...]
            if r.shape!=(initial.step_index,) or z.shape!=r.shape:
                raise ValueError('initial axis history shape mismatch')
            flag=scalar('freeboundary')
            if isinstance(flag,bytes):flag=flag.decode('ascii')
            if str(flag).strip().upper()!='F':raise ValueError('initial free boundary is unsupported')
            seed=(float(r[-1]),float(z[-1]))
        result=cls(mesh,initial.boundary,axis_seed=seed,axis_search_radius=.5,
            z_xpoint_limits=(-1.,1.),R_geo=1.11,restart_lcfs_lost=initial.equilibrium.lcfs_lost,
            restart_history_provenance=initial.provenance,xcase=initial.equilibrium.xcase,freeboundary=False)
        result._config.update(initial_source_sha256=initial.source_sha256,
            search_parameter_namelist_sha256=initial.namelist_sha256,
            initialization_time_si=initial.time_si,initialization_step=initial.step_index,
            lcfs_history_length=initial.lcfs_history_length,
            search_parameter_profile='audited_EHL2_phaseA_v400_and_injInner_lines130_138')
        return result

    def __init__(self,mesh,boundary,*,axis_seed,axis_search_radius,z_xpoint_limits,
                 R_geo,restart_lcfs_lost,restart_history_provenance,
                 xcase='double',freeboundary=False,max_attempts=500,
                 gradient_tolerance=1e-9):
        if xcase!='double' or freeboundary is not False:
            raise ValueError('only fixed-boundary double X-point topology supported')
        if (type(restart_lcfs_lost) is not bool or not restart_history_provenance
            or np.shape(axis_seed)!=(2,) or np.shape(z_xpoint_limits)!=(2,)
            or not np.isfinite([*axis_seed,*z_xpoint_limits,axis_search_radius,R_geo,gradient_tolerance]).all()
            or axis_search_radius<=0 or R_geo<=0 or gradient_tolerance<=0
            or z_xpoint_limits[0]>=z_xpoint_limits[1]
            or type(max_attempts) is not int or max_attempts<1
            or np.shape(boundary)!=(mesh.n_nodes,) or not np.isfinite(boundary).all()):
            raise ValueError('explicit finite geometry/search/restart history configuration required')
        self.mesh=mesh
        self._config=dict(version='native-autonomous-double-X-v1',
            geometry=configuration_digest(dict(x=mesh.coordinates,size=mesh.size,vertex=mesh.vertex,indices=mesh.indices)),
            boundary=configuration_digest(np.asarray(boundary)),axis_seed=list(map(float,axis_seed)),
            axis_search_radius=float(axis_search_radius),z_xpoint_limits=list(map(float,z_xpoint_limits)),
            R_geo=float(R_geo),restart_lcfs_lost=restart_lcfs_lost,
            restart_history_provenance=str(restart_history_provenance),xcase=xcase,freeboundary=False,
            max_attempts=max_attempts,gradient_tolerance=float(gradient_tolerance),
            future_fields='inaccessible_no_reader',axis_search='fixed_initial_center_not_previous_ES',
            stationary_search='Gauss_ranked_scipy_root_physical_gradient_Hessian_classification',
            limiter='not_used_fixed_boundary_with_two_valid_far_Xpoints',
            lcfs='explicit_fixed_restart_prefix_not_current_topology_test',
            failure='raise_no_frozen_or_approximate_root_fallback',gradient='stop_gradient_numpy')
        q=(np.polynomial.legendre.leggauss(4)[0]+1)/2
        ss,tt=np.meshgrid(q,q,indexing='ij');self._st=np.column_stack((ss.ravel(),tt.ravel()))
        h,hs,ht=_cubic_basis(self._st[:,0],self._st[:,1])
        sizes=mesh.size.transpose(2,1,0)
        self._b=h[None]*sizes[:,None];self._bs=hs[None]*sizes[:,None];self._bt=ht[None]*sizes[:,None]
        geo=mesh.coordinates[:,:,0,:][:,:,mesh.vertex.T].transpose(2,0,3,1)
        self._rz=np.einsum('eqvd,efvd->eqf',self._b,geo)
        xs=np.einsum('eqvd,efvd->eqf',self._bs,geo)
        xt=np.einsum('eqvd,efvd->eqf',self._bt,geo)
        self._jac=xs[:,:,0]*xt[:,:,1]-xs[:,:,1]*xt[:,:,0]
        self._xs=xs;self._xt=xt

    @property
    def configuration(self):
        return deepcopy(self._config)

    def __call__(self,fields,time_si):
        m=self.mesh
        if configuration_digest(dict(x=m.coordinates,size=m.size,vertex=m.vertex,indices=m.indices))!=self._config['geometry']:
            raise ValueError('equilibrium fixed geometry changed after configuration binding')
        u=self.mesh.validate_fields(fields,9)
        if not np.isfinite(time_si):raise ValueError('finite diagnostic event time required')
        psi=np.ascontiguousarray(u[:1])
        local=psi[0,:,0,:][:,self.mesh.vertex.T].transpose(1,2,0)
        ds=np.einsum('eqvd,evd->eq',self._bs,local)
        dt=np.einsum('eqvd,evd->eq',self._bt,local)
        grad=np.stack(((self._xt[:,:,1]*ds-self._xs[:,:,1]*dt)/self._jac,
                       (-self._xt[:,:,0]*ds+self._xs[:,:,0]*dt)/self._jac),axis=-1)
        score=np.linalg.norm(grad,axis=-1)
        c=self._config;seed=np.asarray(c['axis_seed']);radius=c['axis_search_radius']

        def stationary(mask,kind,accept):
            ranking=np.argsort(np.where(mask,score,np.inf).ravel(),kind='stable')
            visited=set()
            for ix in ranking:
                e,q=divmod(int(ix),16)
                if not mask[e,q]:break
                if e in visited:continue
                if len(visited)>=c['max_attempts']:break
                visited.add(e)
                def fun(st):return self.mesh.point(e,st,psi)[1][0]
                try:
                    sol=root(fun,self._st[q],options={'maxfev':80,'xtol':1e-10})
                    st=sol.x
                    if not np.isfinite(st).all() or (st < -1e-8).any() or (st > 1+1e-8).any():continue
                    st=np.clip(st,0.,1.)
                    val,g,rz,_,j=self.mesh.point(e,st,psi)
                    if np.linalg.norm(g[0])>c['gradient_tolerance'] or not accept(rz):continue
                    step=1e-5
                    dg=np.column_stack([(fun(st+step*d)-fun(st-step*d))/(2*step) for d in np.eye(2)])
                    hess=dg@np.linalg.inv(j);eig=np.linalg.eigvalsh((hess+hess.T)/2)
                    if kind=='axis' and not (eig[0]*eig[1]>0):continue
                    if kind=='saddle' and not (eig[0]<0<eig[1]):continue
                    return float(val[0]),rz
                except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                    continue
            raise ValueError(f'autonomous equilibrium {kind} search failed: no valid stationary point')

        axis_mask=np.linalg.norm(self._rz-seed,axis=-1)<radius
        axis,axis_rz=stationary(axis_mask,'axis',lambda p:np.linalg.norm(p-seed)<radius)
        zlo,zhi=c['z_xpoint_limits'];margin=.06*c['R_geo']
        lo_limit=min(zlo,axis_rz[1]+.03*c['R_geo'])
        hi_limit=max(zhi,axis_rz[1]-.03*c['R_geo'])
        low,low_rz=stationary(self._rz[:,:,1]<lo_limit,'saddle',
            lambda p:p[1]<lo_limit and np.linalg.norm(p-axis_rz)>=margin)
        high,high_rz=stationary(self._rz[:,:,1]>hi_limit,'saddle',
            lambda p:p[1]>hi_limit and np.linalg.norm(p-axis_rz)>=margin)
        bnd=low if abs(axis-low)<abs(axis-high) else high
        if abs(bnd-axis)<1e-6:raise ValueError('autonomous equilibrium degenerate axis/boundary flux')
        return NativeFluxEquilibrium.bind(u,psi_axis=axis,psi_bnd=bnd,
            psi_xpoints=(low,high),z_xpoints=(float(low_rz[1]),float(high_rz[1])),
            xcase='double',lcfs_lost=c['restart_lcfs_lost'],time_si=float(time_si),
            provenance='current_predicted_cubic_psi;'+configuration_digest(c))
