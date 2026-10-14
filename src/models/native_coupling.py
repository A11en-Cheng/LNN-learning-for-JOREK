"""Small H=1 native source reference. No implicit production-cloud replay.

The full cubic mesh and explicit reference/inspected-n0 projection are versioned. Atomic
rates have an independently identified provider; the source path is NumPy.
"""
from copy import deepcopy
import numpy as np
from .native_gauss import _cubic_basis,NativeElementGeometry,evaluate_native_element,GaussRule,ToroidalSampling
from .native_state import configuration_digest
from .marker_state import advance_window
from .marker_feedback import MU_ZERO,EL_CHG,ATOMIC_MASS_UNIT


class NativeSourceMesh:
    """Full H1 cubic geometry; optional inspected n0 projection at creation.

    projection_n0 binds an explicit initialization psi snapshot, filters and
    particle-projection flags. It never defines nine-field fluid boundaries.
    """
    def __init__(self,coordinates,size,vertex,indices,*,units,max_dofs=128,projection="dense",max_elements=10000,projection_n0=None):
        units.validate();self.units=units
        self.coordinates=np.array(coordinates,copy=True);self.size=np.array(size,copy=True)
        self.vertex=np.array(vertex,copy=True);self.indices=np.array(indices,copy=True)
        self.n_nodes=self.coordinates.shape[-1]
        if (self.coordinates.shape!=(2,4,1,self.n_nodes) or self.vertex.ndim!=2 or self.vertex.shape[0]!=4
            or self.vertex.dtype.kind not in 'iu' or (self.vertex<0).any() or (self.vertex>=self.n_nodes).any()
            or self.size.shape!=(4,4,self.vertex.shape[1]) or self.indices.shape!=(4,self.n_nodes)
            or self.indices.dtype.kind not in 'iu' or (self.indices<=0).any()):
            raise ValueError('full cubic H1 mesh and native shared indices required')
        _,self.inverse=np.unique(self.indices,return_inverse=True)
        self.inverse=self.inverse.reshape(4,self.n_nodes);self.n_dofs=int(self.inverse.max())+1
        if projection not in ('dense','sparse','none'):
            raise ValueError('unsupported source projection backend')
        if self.vertex.shape[1]>max_elements:
            raise ValueError('source element resource budget exceeded')
        if projection=='dense' and self.n_dofs>max_dofs:
            raise ValueError('small dense reference DOF budget exceeded')
        self.projection=projection
        self.projection_n0=None
        if projection_n0 is not None:
            if projection!="sparse":raise ValueError("filtered n0 projection requires sparse backend")
            from .native_projection_n0 import validate_n0
            self.projection_n0=validate_n0(projection_n0,self.n_nodes)
        self.local_indices=self.inverse[:,self.vertex.T].transpose(1,2,0).reshape(-1,16)
        self.gauss=[];self.global_basis=[] if projection=='dense' else None
        rows=[];columns=[];data=[]
        self.mass=np.zeros((self.n_dofs,self.n_dofs)) if projection=='dense' else None
        for e,ids in enumerate(self.vertex.T):
            geo=NativeElementGeometry(self.coordinates[:,:,:,ids],self.size[:,:,e],3)
            g=evaluate_native_element(geo,np.zeros((1,4,1,4)),rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1))
            if (g.jacobian<=0).any():raise ValueError('positive oriented source mesh required')
            local=g.test_basis[:,:,:,0].reshape(16,16)
            vol=g.weights*g.jacobian*g.coordinates[:,0]
            self.gauss.append(g)
            if projection=='dense':
                b=np.zeros((16,self.n_dofs))
                for j,k in enumerate(self.local_indices[e]):b[:,k]+=local[:,j]
                self.mass+=b.T@(vol[:,None]*b);self.global_basis.append(b)
            elif projection=='sparse':
                k=self.local_indices[e]
                rows.append(np.repeat(k,16));columns.append(np.tile(k,16))
                if self.projection_n0 is None:
                    block=local.T@(vol[:,None]*local)
                else:
                    from .native_projection_n0 import local_operator
                    block=local_operator(self,e,g,self.projection_n0)
                data.append(block.ravel())
        self._factor=None
        if projection=='dense':
            if not np.isfinite(self.mass).all():raise ValueError('nonfinite source projection mass')
            np.linalg.cholesky(self.mass)
        elif projection=='sparse':
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import splu
            self.mass=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(columns))),
                shape=(self.n_dofs,self.n_dofs)).tocsc()
            if not np.isfinite(self.mass.data).all():raise ValueError('nonfinite source projection mass')
            from scipy.sparse import diags
            if self.projection_n0 is not None:
                from .native_projection_n0 import penalties
                self.mass+=diags(penalties(self,self.projection_n0))
            diagonal=self.mass.diagonal()
            if (diagonal<=0).any():raise ValueError('uncovered/invalid shared source DOF')
            self._equilibration=1/np.sqrt(diagonal)
            scale=diags(self._equilibration)
            self._factor=splu((scale@self.mass@scale).tocsc())
        self.configuration=dict(version='native-source-mesh-v2',units=vars(units),
            geometry=configuration_digest(dict(x=self.coordinates,size=self.size,vertex=self.vertex,indices=self.indices)),
            projection='consistent_cubic_unfiltered_H1_no_extra_2pi',backend=projection,
            boundary='unconstrained_reference_not_JOREK_projection_dirichlet',
            locator='bernstein_bounds_damped_newton_hints_v1',
            equilibration='symmetric_diagonal' if projection=='sparse' else 'none',
            max_dofs=max_dofs,max_elements=max_elements)
        if self.projection_n0 is not None:
            from .native_projection_n0 import fingerprint
            self.configuration.update(version="native-source-mesh-v3_n0",
                projection="jorek_inspected_n0_filtered",boundary="inspected_projection_penalties",
                projection_n0=fingerprint(self.projection_n0))
        # Convex hull of tensor Bernstein controls bounds the whole cubic map,
        # including curved edges (vertex-only boxes are not conservative).
        from scipy.spatial import cKDTree
        q=np.linspace(0.,1.,4)
        bernstein=np.stack([(1-q)**3,3*q*(1-q)**2,3*q*q*(1-q),q**3],axis=1)
        inverse=np.linalg.inv(bernstein)
        ss,tt=np.meshgrid(q,q,indexing='ij')
        basis=_cubic_basis(ss.ravel(),tt.ravel())[0]
        bounds=[]
        for e,ids in enumerate(self.vertex.T):
            coefficients=self.coordinates[:,:,0,ids].transpose(0,2,1)*self.size[:,:,e].T
            samples=np.einsum('qvd,fvd->qf',basis,coefficients).reshape(4,4,2)
            controls=np.einsum('ai,ijf,bj->abf',inverse,samples,inverse)
            bounds.append([controls.min(axis=(0,1)),controls.max(axis=(0,1))])
        self._location_bounds=np.asarray(bounds)
        self._location_centers=self._location_bounds.mean(axis=1)
        self._location_tree=cKDTree(self._location_centers)
        self._location_radius=float(np.linalg.norm(self._location_bounds[:,1]-self._location_centers,axis=1).max())+1e-9

    def weak_load(self,gauss_values):
        """Assemble test-space loads by shared index, without a global basis."""
        a=np.asarray(gauss_values)
        if a.ndim!=3 or a.shape[:2]!=(len(self.gauss),16) or not np.isfinite(a).all():
            raise ValueError('finite Gauss [E,16,channels] required')
        rhs=np.zeros((a.shape[-1],self.n_dofs))
        for e,g in enumerate(self.gauss):
            b=g.test_basis[:,:,:,0].reshape(16,16)
            local=b.T@((g.weights*g.jacobian*g.coordinates[:,0])[:,None]*a[e])
            for channel in range(len(rhs)):
                np.add.at(rhs[channel],self.local_indices[e],local[:,channel])
        return rhs

    def coefficients(self,vector):
        return vector[:,self.inverse][:,:,None,:]

    def validate_fields(self,values,n_fields):
        a=np.asarray(values)
        if a.shape!=(n_fields,4,1,self.n_nodes) or not np.isfinite(a).all():
            raise ValueError('missing/invalid full native coefficients/history')
        return a

    def point(self,e,st,values):
        if (type(e) not in (int,np.int32,np.int64) or not 0<=e<len(self.gauss)
            or np.shape(st)!=(2,) or not np.isfinite(st).all()):
            raise ValueError('invalid element/local coordinates')
        a=np.asarray(values);ids=self.vertex[:,e]
        h,hs,ht=_cubic_basis(np.array([st[0]]),np.array([st[1]]))
        b=h[0]*self.size[:,:,e].T;bs=hs[0]*self.size[:,:,e].T;bt=ht[0]*self.size[:,:,e].T
        x=self.coordinates[:,:,0,ids].transpose(0,2,1)
        xyz=np.einsum('vd,fvd->f',b,x)
        xs=np.einsum('vd,fvd->f',bs,x);xt=np.einsum('vd,fvd->f',bt,x)
        jac=xs[0]*xt[1]-xt[0]*xs[1]
        if not np.isfinite(jac) or jac<=0 or xyz[0]<=0:raise ValueError('invalid native point geometry')
        local=a[:,:,0,ids].transpose(0,2,1)
        val=np.einsum('vd,fvd->f',b,local)
        ds=np.einsum('vd,fvd->f',bs,local);dt=np.einsum('vd,fvd->f',bt,local)
        grad=np.stack([(xt[1]*ds-xs[1]*dt)/jac,(-xt[0]*ds+xs[0]*dt)/jac],axis=-1)
        return val,grad,xyz,b,np.stack([xs,xt],axis=1)

    def points(self,e,st,values):
        """Bounded batch of the same full cubic evaluations as point()."""
        return self.points_many(e,st,(values,))[0]

    def points_many(self,e,st,fields):
        """Five-output tuples in field order, sharing only this call's geometry."""
        if not isinstance(fields,(tuple,list)) or not fields:
            raise ValueError('nonempty tuple/list of native fields required')
        geometry=self._points_geometry(e,st)
        return tuple(self._points_from_geometry(geometry,values) for values in fields)

    def _points_geometry(self,e,st):
        """Existing batch geometry arithmetic and guards, with reusable basis."""
        e=np.asarray(e);st=np.asarray(st)
        if (e.ndim!=1 or e.dtype.kind not in 'iu' or (e<0).any() or (e>=len(self.gauss)).any()
            or st.shape!=(len(e),2) or not np.isfinite(st).all()):
            raise ValueError('invalid batch element/local coordinates')
        h,hs,ht=_cubic_basis(st[:,0],st[:,1]);sizes=self.size[:,:,e].transpose(2,1,0)
        b=h*sizes;bs=hs*sizes;bt=ht*sizes;ids=self.vertex[:,e].T
        x=self.coordinates[:,:,0,:][:,:,ids].transpose(2,0,3,1)
        xyz=np.einsum('bvd,bfvd->bf',b,x)
        xs=np.einsum('bvd,bfvd->bf',bs,x);xt=np.einsum('bvd,bfvd->bf',bt,x)
        jac=xs[:,0]*xt[:,1]-xt[:,0]*xs[:,1]
        if not np.isfinite(jac).all() or (jac<=0).any() or (xyz[:,0]<=0).any():
            raise ValueError('invalid native point geometry')
        return ids,b,bs,bt,xyz,xs,xt,jac

    def _points_from_geometry(self,geometry,values):
        ids,b,bs,bt,xyz,xs,xt,jac=geometry
        a=np.asarray(values)
        if a.ndim!=4:raise ValueError('missing/invalid full native coefficients/history')
        a=self.validate_fields(a,len(a))
        local=a[:,:,0,:][:,:,ids].transpose(2,0,3,1)
        val=np.einsum('bvd,bfvd->bf',b,local)
        ds=np.einsum('bvd,bfvd->bf',bs,local);dt=np.einsum('bvd,bfvd->bf',bt,local)
        grad=np.stack(((xt[:,1,None]*ds-xs[:,1,None]*dt)/jac[:,None],
            (-xt[:,0,None]*ds+xs[:,0,None]*dt)/jac[:,None]),axis=-1)
        return val,grad,xyz,b,np.stack((xs,xt),axis=-1)

    def _marker_points(self,e,st,fields,feedback):
        """Consume validated event snapshots, retaining cubic geometry arithmetic.

        This private source path does not change the complete public field
        evaluator. Only channels read by the marker formula are contracted;
        discarded entries in its internal full-width views are zero.
        """
        ids,b,bs,bt,xyz,xs,xt,jac=self._points_geometry(e,st)
        field_local=fields[:,:,0,:][:,:,ids].transpose(2,0,3,1)
        feedback_local=feedback[:,:,0,:][:,:,ids].transpose(2,0,3,1)
        values=np.zeros((len(e),9));gradients=np.zeros((len(e),9,2))
        projected=np.zeros((len(e),5))
        # Slice views preserve the original vd strides and reduction order.
        # Advanced field indexing packs a new array and changes einsum's
        # floating-point summation path, even with the same expression.
        for channels in (slice(4,5),slice(6,9)):
            values[:,channels]=np.einsum('bvd,bfvd->bf',b,field_local[:,channels])
        for channels in (slice(0,2),slice(7,8)):
            ds=np.einsum('bvd,bfvd->bf',bs,field_local[:,channels])
            dt=np.einsum('bvd,bfvd->bf',bt,field_local[:,channels])
            gradients[:,channels]=np.stack(
                ((xt[:,1,None]*ds-xs[:,1,None]*dt)/jac[:,None],
                 (-xt[:,0,None]*ds+xs[:,0,None]*dt)/jac[:,None]),axis=-1)
        projected[:,3:5]=np.einsum('bvd,bfvd->bf',b,feedback_local[:,3:5])
        return values,gradients,projected,xyz

    def _location_geometry(self,e,st):
        """point() geometry arithmetic and guards, without dummy-field work."""
        if (type(e) not in (int,np.int32,np.int64) or not 0<=e<len(self.gauss)
            or np.shape(st)!=(2,) or not np.isfinite(st).all()):
            raise ValueError('invalid element/local coordinates')
        ids=self.vertex[:,e]
        h,hs,ht=_cubic_basis(np.array([st[0]]),np.array([st[1]]))
        b=h[0]*self.size[:,:,e].T;bs=hs[0]*self.size[:,:,e].T;bt=ht[0]*self.size[:,:,e].T
        x=self.coordinates[:,:,0,ids].transpose(0,2,1)
        xyz=np.einsum('vd,fvd->f',b,x)
        xs=np.einsum('vd,fvd->f',bs,x);xt=np.einsum('vd,fvd->f',bt,x)
        jac=xs[0]*xt[1]-xt[0]*xs[1]
        if not np.isfinite(jac) or jac<=0 or xyz[0]<=0:raise ValueError('invalid native point geometry')
        return xyz,np.stack([xs,xt],axis=1)

    def _location_geometries(self,e,st):
        """points() geometry arithmetic and guards for bounded Newton batches."""
        e=np.asarray(e);st=np.asarray(st)
        if (e.ndim!=1 or e.dtype.kind not in 'iu' or (e<0).any() or (e>=len(self.gauss)).any()
            or st.shape!=(len(e),2) or not np.isfinite(st).all()):
            raise ValueError('invalid batch element/local coordinates')
        h,hs,ht=_cubic_basis(st[:,0],st[:,1]);sizes=self.size[:,:,e].transpose(2,1,0)
        b=h*sizes;bs=hs*sizes;bt=ht*sizes;ids=self.vertex[:,e].T
        x=self.coordinates[:,:,0,:][:,:,ids].transpose(2,0,3,1)
        xyz=np.einsum('bvd,bfvd->bf',b,x)
        xs=np.einsum('bvd,bfvd->bf',bs,x);xt=np.einsum('bvd,bfvd->bf',bt,x)
        jac=xs[:,0]*xt[:,1]-xt[:,0]*xs[:,1]
        if not np.isfinite(jac).all() or (jac<=0).any() or (xyz[:,0]<=0).any():
            raise ValueError('invalid native point geometry')
        return xyz,np.stack((xs,xt),axis=-1)

    def locate(self,positions,*,element_hints=None,local_hints=None):
        return self._locate(positions,element_hints=element_hints,local_hints=local_hints)

    def _locate(self,positions,*,element_hints=None,local_hints=None,trial_cache=None):
        """Original scalar order; optional geometry from this batch call only."""
        positions=np.asarray(positions)
        if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
            raise ValueError('finite R/Z/phi positions required')
        if element_hints is not None:
            if (np.shape(element_hints)!=(len(positions),) or np.shape(local_hints)!=(len(positions),2)
                or not np.isfinite(local_hints).all()):
                raise ValueError('matching element/local hints required')
        elements=np.zeros(len(positions),dtype=np.int32);local=np.zeros((len(positions),2))
        # Reuse only the unmodified native evaluator. Overrides keep their
        # original call/exception semantics; no geometry survives this call.
        reuse_geometry=(type(self) is NativeSourceMesh and '_location_geometry' not in self.__dict__)
        for i,x in enumerate(positions):
            candidates=self._location_tree.query_ball_point(x[:2],self._location_radius)
            hint=-1 if element_hints is None else int(element_hints[i])-1
            if reuse_geometry:
                from .native_location_candidates import location_candidates
                candidates=location_candidates(candidates,self._location_centers,self._location_bounds,x[:2],hint)
            else:
                candidates=sorted(candidates,key=lambda e:np.linalg.norm(self._location_centers[e]-x[:2]))
                if 0<=hint<len(self.gauss):candidates=[hint]+[e for e in candidates if e!=hint]
            for e in candidates:
                lower,upper=self._location_bounds[e]
                if np.any(x[:2]<lower-1e-9) or np.any(x[:2]>upper+1e-9):continue
                st=np.clip(local_hints[i],0,1).copy() if e==hint else np.array([.5,.5])
                geometry=None
                for _ in range(25):
                    try:
                        if geometry is None:
                            geometry=(None if trial_cache is None else trial_cache.get((e,st.tobytes())))
                            if geometry is None:geometry=self._location_geometry(e,st)
                        rz,jac=geometry
                        error=np.linalg.norm(rz-x[:2])
                        if error<1e-10:
                            elements[i]=e+1;local[i]=st;break
                        step=np.linalg.solve(jac,rz-x[:2])
                    except (ValueError,np.linalg.LinAlgError):break
                    # Keep trials in the candidate's physical reference domain.
                    accepted=False
                    for damping in (1.,.5,.25,.125,.0625,.03125):
                        trial=np.clip(st-damping*step,0,1)
                        try:
                            trial_geometry=(None if trial_cache is None else trial_cache.get((e,trial.tobytes())))
                            if trial_geometry is None:trial_geometry=self._location_geometry(e,trial)
                        except ValueError:continue
                        if np.linalg.norm(trial_geometry[0]-x[:2])<error:
                            st=trial;accepted=True
                            geometry=trial_geometry if reuse_geometry else None
                            break
                    if not accepted:break
                if elements[i]>0:break
        return elements,local

    def locate_batch(self,positions,*,element_hints=None,local_hints=None):
        """Fast in-element Newton; unresolved/crossing particles use scalar search."""
        x=np.asarray(positions)
        if (x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all()):
            raise ValueError('finite R/Z/phi positions required')
        if element_hints is None:return self.locate(x)
        hints=np.asarray(element_hints);st=np.asarray(local_hints).copy()
        if hints.shape!=(len(x),) or st.shape!=(len(x),2) or not np.isfinite(st).all():
            raise ValueError('matching element/local hints required')
        elements=np.zeros(len(x),dtype=np.int32);local=np.zeros((len(x),2))
        e=hints.astype(np.int64)-1
        active=np.flatnonzero((e>=0)&(e<len(self.gauss)))
        if len(active):
            bounds=self._location_bounds[e[active]]
            inside=((x[active,:2]>=bounds[:,0]-1e-9)&(x[active,:2]<=bounds[:,1]+1e-9)).all(axis=1)
            active=active[inside]
        st=np.clip(st,0,1)
        # Non-float64 hints can round/truncate trial coordinates on writeback;
        # the original algorithm reevaluates those stored coordinates.
        reuse_geometry=(type(self) is NativeSourceMesh and '_location_geometries' not in self.__dict__
                        and st.dtype==np.dtype(np.float64))
        # Both native evaluators contract the same float64 vd terms in the
        # same order. Retain only this call's initial and boundary-trial
        # geometries for failed hints, not a persistent spatial/field cache.
        reuse_trials=(reuse_geometry and '_location_geometry' not in self.__dict__
                      and 'locate' not in self.__dict__
                      and self.coordinates.dtype==np.dtype(np.float64)
                      and self.coordinates.flags.c_contiguous)
        trial_cache={} if reuse_trials else None
        initial_geometry=None
        geometry=None
        for _ in range(25):
            if not len(active):break
            try:
                rz,jac=(self._location_geometries(e[active],st[active]) if geometry is None else geometry)
                if reuse_trials and initial_geometry is None:
                    initial_geometry=(active.copy(),st[active].copy(),rz.copy(),jac.copy())
                error=np.linalg.norm(rz-x[active,:2],axis=1)
                done=error<1e-10;ids=active[done]
                elements[ids]=e[ids]+1;local[ids]=st[ids]
                remain=~done;active=active[remain]
                if not len(active):break
                step=np.linalg.solve(jac[remain],(rz[remain]-x[active,:2])[...,None])[...,0]
                error=error[remain];accepted=np.zeros(len(active),bool)
                if reuse_geometry:
                    next_rz=np.empty((len(active),2),dtype=rz.dtype)
                    next_jac=np.empty((len(active),2,2),dtype=jac.dtype)
                for damping in (1.,.5,.25,.125,.0625,.03125):
                    todo=np.flatnonzero(~accepted)
                    if not len(todo):break
                    ids=active[todo];trial=np.clip(st[ids]-damping*step[todo],0,1)
                    trial_rz,trial_jac=self._location_geometries(e[ids],trial)
                    if reuse_trials:
                        boundary=np.flatnonzero(((trial==0)|(trial==1)).any(axis=1))
                        for j in boundary:
                            if len(trial_cache)>=32768:break
                            trial_cache[(int(e[ids[j]]),trial[j].tobytes())]=(trial_rz[j].copy(),trial_jac[j].copy())
                    better=np.linalg.norm(trial_rz-x[ids,:2],axis=1)<error[todo]
                    st[ids[better]]=trial[better];accepted[todo[better]]=True
                    if reuse_geometry:
                        next_rz[todo[better]]=trial_rz[better]
                        next_jac[todo[better]]=trial_jac[better]
                active=active[accepted]
                if reuse_geometry:geometry=(next_rz[accepted],next_jac[accepted])
            except (ValueError,np.linalg.LinAlgError):
                break  # Scalar path retains the original per-particle failure policy.
        pending=np.flatnonzero(elements==0)
        if len(pending):
            if reuse_trials:
                if initial_geometry is not None:
                    ids,stored_st,stored_rz,stored_jac=initial_geometry
                    for j in np.flatnonzero(elements[ids]==0):
                        trial_cache[(int(e[ids[j]]),stored_st[j].tobytes())]=(stored_rz[j],stored_jac[j])
                pe,ps=self._locate(x[pending],element_hints=hints[pending],
                    local_hints=np.asarray(local_hints)[pending],trial_cache=trial_cache)
            else:
                pe,ps=self.locate(x[pending],element_hints=hints[pending],local_hints=np.asarray(local_hints)[pending])
            elements[pending]=pe;local[pending]=ps
        return elements,local

    def deposit(self,cloud,ids,load):
        rhs=np.zeros((5,self.n_dofs))
        for j,i in enumerate(ids):
            e=int(cloud.i_elm[i])-1
            _,_,_,b,_=self.point(e,cloud.st[i],np.zeros((1,4,1,self.n_nodes)))
            for v,node in enumerate(self.vertex[:,e]):
                for d in range(4):rhs[:,self.inverse[d,node]]+=load[j]*b[v,d]
        return rhs

    def deposit_batch(self,cloud,ids,load):
        e=cloud.i_elm[ids].astype(np.int64)-1
        basis=self._points_geometry(e,cloud.st[ids])[1]
        dofs=self.inverse[:,self.vertex[:,e].T].transpose(1,2,0).ravel()
        contributions=(load[:,:,None,None]*basis[:,None,:,:]).transpose(1,0,2,3).reshape(5,-1)
        rhs=np.zeros((5,self.n_dofs))
        for channel in range(5):np.add.at(rhs[channel],dofs,contributions[channel])
        return rhs

    def project(self,rhs):
        rhs=np.asarray(rhs)
        if rhs.ndim!=2 or rhs.shape[1]!=self.n_dofs or not np.isfinite(rhs).all():
            raise ValueError('finite unique-DOF source load required')
        if self.projection=='none':raise ValueError('formula-only geometry has no projection solver')
        vector=((self._equilibration[:,None]*self._factor.solve(self._equilibration[:,None]*rhs.T)).T if self.projection=='sparse' else np.linalg.solve(self.mass,rhs.T).T)
        if not np.isfinite(vector).all():raise ValueError('nonfinite source projection solution')
        return self.coefficients(vector)

    def integrate(self,values):
        values=np.asarray(values);out=np.zeros(len(values))
        for e,(ids,g) in enumerate(zip(self.vertex.T,self.gauss)):
            local=values[:,:,0,ids].transpose(0,2,1)
            v=np.einsum('qvd,fvd->qf',g.test_basis[:,:,:,0],local)
            out+=np.einsum('q,qf->f',g.weights*g.jacobian*g.coordinates[:,0],v)
        return out


class NativeMarkerSource:
    """Actual charge/deposit/push/project chain on full native fields.

    cuda_fp64_v1 is an explicit experimental diagnostic option, not qualified
    for training/production; its full injInner comparison failed element identity.

    Supported subset: Ne, non-CE, deterministic clouds, constant explicitly
    configured D_perp and no differential diffusive flux. Projection is an
    explicit unfiltered reference or inspected filtered n0 operator. Atomic
    provider returns SCD/ACD m3/s and per-ion radiation W, with version.
    """
    def __init__(self,mesh,*,atomic,atomic_version,potentials_ev,F0,diffusion_ju,time_sampling="endpoint_reference",field_static=None,execution_budget=None,spatial_backend='scalar',transport_policy=None,saved_neighbours=None):
        from .marker_state import MarkerExecutionBudget
        if execution_budget is not None and not isinstance(execution_budget,MarkerExecutionBudget):
            raise ValueError('explicit MarkerExecutionBudget required')
        self.execution_budget=execution_budget
        if spatial_backend not in ('scalar','batched_v1','cuda_fp64_v1'):raise ValueError('unsupported source spatial backend')
        self.spatial_backend=spatial_backend
        from .marker_transport import LocalSubcycling
        if transport_policy is not None and (not isinstance(transport_policy,LocalSubcycling) or spatial_backend!='batched_v1'):
            raise ValueError('local subcycling requires explicit policy and CPU batched native fields')
        self.transport_policy=transport_policy
        self._boundary_exit=None
        if transport_policy is not None and transport_policy.boundary_exit:
            from .marker_boundary_exit import NativeMarkerBoundaryExit
            self._boundary_exit=NativeMarkerBoundaryExit(mesh,saved_neighbours)
        elif saved_neighbours is not None:
            raise ValueError('saved neighbours supplied without configured boundary exit mode')
        if (not callable(atomic) or not atomic_version or not np.isfinite([F0,diffusion_ju]).all()
            or diffusion_ju<0 or mesh.units.gamma!=5/3):raise ValueError('explicit supported source configuration required')
        if hasattr(atomic,"version") and atomic.version!=atomic_version:
            raise ValueError("atomic provider version does not match source configuration")
        if time_sampling=="endpoint_reference" and field_static is None:field_static=False
        if time_sampling not in ("endpoint_reference","jorek_raw_deltas_v1") or type(field_static) is not bool:
            raise ValueError("unsupported explicit source time-sampling mode/static flag")
        self.time_sampling=time_sampling;self.field_static=field_static
        self.mesh=mesh;self.atomic=atomic;self.F0=float(F0);self.diffusion_ju=float(diffusion_ju)
        self.potentials_ev=np.array(potentials_ev,copy=True)
        if self.potentials_ev.shape!=(10,) or not np.isfinite(self.potentials_ev).all() or (self.potentials_ev<=0).any():
            raise ValueError('complete Ne ionization potentials required')
        self.configuration=dict(version='native-small-marker-source-v1',mesh=mesh.configuration,
            atomic=atomic_version,potentials_ev=self.potentials_ev.tolist(),F0=F0,diffusion_ju=diffusion_ju,
            approximation=('constant_Dperp_no_diff_flux_unfiltered_nonCE_atomic_provider' if mesh.projection_n0 is None
                           else 'constant_Dperp_no_diff_flux_inspected_n0_nonCE_atomic_provider'),differentiability='numpy_reference')

        if time_sampling=="jorek_raw_deltas_v1":
            self.configuration.update(version="native-small-marker-source-v2_rawDtime",
                time_sampling=time_sampling,field_static=field_static)
        if execution_budget is not None:
            self.configuration.update(version="native-marker-source-v3_budget",
                time_sampling=time_sampling,field_static=field_static,
                execution_budget=vars(execution_budget).copy())
        if spatial_backend!='scalar':self.configuration['spatial_backend']=spatial_backend
        if transport_policy is not None:self.configuration['transport_policy']=transport_policy.configuration
        if self._boundary_exit is not None:self.configuration['transport_boundary']=self._boundary_exit.configuration
        self._spatial=None
        if spatial_backend=='cuda_fp64_v1':
            from .native_cuda_spatial import NativeCudaSpatial
            self._spatial=NativeCudaSpatial(mesh,device='cuda')
            self.configuration['spatial_configuration']=deepcopy(self._spatial.configuration)

    def advance(self,state,target_si,previous,current,*,births=None,time_sampler=None):
        transport_policy=getattr(self,'transport_policy',None)
        if self.configuration.get('transport_policy') != (None if transport_policy is None else transport_policy.configuration):
            raise ValueError('source transport policy configuration changed')
        boundary=getattr(self,'_boundary_exit',None)
        if (bool(boundary is not None)!=bool(transport_policy is not None and transport_policy.boundary_exit)
            or self.configuration.get('transport_boundary')!=(None if boundary is None else boundary.configuration)):
            raise ValueError('source boundary exit configuration changed')
        if boundary is not None:boundary.validate(self.mesh)
        mesh=self.mesh;mesh.validate_fields(previous,9);mesh.validate_fields(current,9)
        mesh.validate_fields(state.feedback.values,5);mesh.validate_fields(state.feedback.deltas,5)
        state.cloud.validate()
        if (not np.isfinite(state.feedback.time_si) or abs(state.feedback.time_si-state.clock.sim_time_si)>1e-14):
            raise ValueError('marker feedback/clock event time mismatch')
        expected=np.sqrt(MU_ZERO*ATOMIC_MASS_UNIT*mesh.units.central_mass*mesh.units.n_norm_m3)
        if (state.feedback.layout!='native_full' or state.cloud.P_imp.shape[1]!=11
            or not np.isclose(mesh.units.t_norm_si,expected,rtol=2e-6,atol=0)):
            raise ValueError('explicit native Ne source state required')
        start,end=state.clock.field_start_si,state.clock.field_end_si
        if self.time_sampling=="jorek_raw_deltas_v1":
            from .native_source_time import NativeSourceTimeSampler
            if not isinstance(time_sampler,NativeSourceTimeSampler):
                raise ValueError("saved-history time sampler required")
            if (not np.array_equal(time_sampler.current,current) or time_sampler.static!=self.field_static
                or abs(time_sampler.time_prev_si-start)>1e-14 or abs(time_sampler.time_now_si-end)>1e-14):
                raise ValueError("time sampler is not bound to current field/clock/static mode")
        elif time_sampler is not None:
            raise ValueError("time sampler supplied to endpoint reference mode")
        if end<=start:raise ValueError('two available field endpoints required')
        last_sample_time=None
        cpu_fields=None
        cpu_feedback=None
        sampling=dict(batches=0,field_snapshots=0,
            field_value_channels=[4,6,7,8],field_gradient_channels=[0,1,7],
            feedback_value_channels=[3,4],feedback_gradient_channels=[])
        if self.spatial_backend=='batched_v1':
            # Owned for this event only; public entry already checked all five
            # channels, including channels not read by the transport formula.
            cpu_feedback=np.array(state.feedback.values,copy=True)
            cpu_feedback.flags.writeable=False
        def sample(t,cloud,ids,feedback):
            nonlocal last_sample_time,cpu_fields
            sampling['batches']+=1
            if self._spatial is not None:
                # Event-owned snapshots only: replace the single field slot at
                # each sampling time. Static rawD sampling uploads once/event.
                static=time_sampler is not None and time_sampler.static
                if last_sample_time is None or (not static and t!=last_sample_time):
                    fields=(time_sampler.fields_at(t) if time_sampler is not None else
                            previous+(current-previous)*((t-start)/(end-start)))
                    self._spatial.freeze_fields('field',fields)
                    last_sample_time=t
            elif self.spatial_backend=='batched_v1':
                static=time_sampler is not None and time_sampler.static
                if last_sample_time is None or (not static and t!=last_sample_time):
                    cpu_fields=(time_sampler.fields_at(t) if time_sampler is not None else
                                previous+(current-previous)*((t-start)/(end-start)))
                    mesh.validate_fields(cpu_fields,9)
                    cpu_fields.flags.writeable=False
                    last_sample_time=t
                    sampling['field_snapshots']+=1
                fields=cpu_fields
            else:
                fields=(time_sampler.fields_at(t) if time_sampler is not None else
                        previous+(current-previous)*((t-start)/(end-start)))
            velocities=[];densities=[];temperatures=[];transport_rate=None
            if self.spatial_backend!='scalar':
                e=cloud.i_elm[ids].astype(np.int64)-1
                if self._spatial is None:
                    v,g,p,rz=mesh._marker_points(e,cloud.st[ids],fields,cpu_feedback)
                else:
                    field_point,feedback_point=self._spatial.points_frozen_many(
                        ('field','feedback'),e,cloud.st[ids])
                    v,g,rz,_,_=field_point
                    p,_,_,_,_=feedback_point
                n=np.where(p[:,4]>=2e-4,p[:,4],1e-4+1e-4*np.exp(np.minimum((p[:,4]-2e-4)/1e-4,0.)))
                z=np.clip(p[:,3]/n,0,10)
                ni=np.maximum(v[:,7]*mesh.units.n_norm_m3*mesh.units.central_mass/20,0.)
                densities=np.maximum((v[:,4]-v[:,7])*mesh.units.n_norm_m3,1e16)+z*ni
                temperatures=np.maximum(v[:,8]/(MU_ZERO*mesh.units.n_norm_m3*1.3806488e-23),1e4)*1.3806488e-23/EL_CHG
                b=np.column_stack((g[:,0,1],-g[:,0,0],np.full(len(ids),self.F0)))/rz[:,0,None]
                velocities=np.column_stack((-g[:,1,1],g[:,1,0],np.zeros(len(ids))))*rz[:,0,None]/mesh.units.t_norm_si+v[:,6,None]/mesh.units.t_norm_si*b
                grad_ni=g[:,7]*mesh.units.n_norm_m3*mesh.units.central_mass/20
                velocities[:,:2]-=self.diffusion_ju/mesh.units.t_norm_si*grad_ni/np.maximum(ni,1e12)[:,None]
                if transport_policy is not None:
                    transport_rate=self.diffusion_ju/mesh.units.t_norm_si*np.sum(grad_ni**2,axis=1)/np.maximum(ni,1e12)**2
            for i in (ids if self.spatial_backend=='scalar' else ()):
                e=int(cloud.i_elm[i])-1
                v,g,rz,_,_=mesh.point(e,cloud.st[i],fields)
                p,_,_,_,_=mesh.point(e,cloud.st[i],feedback.values)
                # Literal corr_neg_dens in particle feedback path, not assembly's .5 floor.
                n=p[4] if p[4]>=2e-4 else 1e-4+1e-4*np.exp((p[4]-2e-4)/1e-4)
                z=np.clip(p[3]/n,0,10)
                ni=max(v[7]*mesh.units.n_norm_m3*mesh.units.central_mass/20,0.)
                ne=max((v[4]-v[7])*mesh.units.n_norm_m3,1e16)+z*ni
                te=max(v[8]/(MU_ZERO*mesh.units.n_norm_m3*1.3806488e-23),1e4)*1.3806488e-23/EL_CHG
                b=np.array([g[0,1],-g[0,0],self.F0])/rz[0]
                velocity=np.array([-g[1,1],g[1,0],0.])*rz[0]/mesh.units.t_norm_si+v[6]/mesh.units.t_norm_si*b
                grad_ni=g[7]*mesh.units.n_norm_m3*mesh.units.central_mass/20
                velocity[:2]-=self.diffusion_ju/mesh.units.t_norm_si*grad_ni/max(ni,1e12)
                velocities.append(velocity);densities.append(ne);temperatures.append(te)
            ne=np.array(densities);te=np.array(temperatures)
            scd,acd,rad=self.atomic(np.maximum(te,1.),ne)
            for a,shape in ((scd,(len(ids),10)),(acd,(len(ids),10)),(rad,(len(ids),11))):
                if np.shape(a)!=shape or not np.isfinite(a).all() or (np.asarray(a)<0).any():
                    raise ValueError('invalid atomic coefficient provider output/units')
            result=dict(velocity=np.array(velocities),ne_si=ne,scd=scd,acd=acd,
                radiation_w_per_atom=np.where((te>1)[:,None],rad,0.))
            if transport_policy is not None:result['transport_rate_si']=transport_rate
            return result
        from contextlib import nullcontext
        scope=(nullcontext() if self._spatial is None else self._spatial.event_scope(
            dict(clock=vars(state.clock),target_si=target_si,time_sampling=self.time_sampling,
                 field_static=self.field_static)))
        with scope:
            if self._spatial is not None:self._spatial.freeze_fields('feedback',state.feedback.values)
            locate=(mesh.locate if self.spatial_backend=='scalar' else
                    mesh.locate_batch if self._spatial is None else self._spatial.locate_batch)
            result,diagnostic=advance_window(state,target_si,sample=sample,locate=locate,
                deposit=mesh.deposit if self.spatial_backend=='scalar' else mesh.deposit_batch,
                project=mesh.project,potentials_ev=self.potentials_ev,t_norm=mesh.units.t_norm_si,
                central_density=mesh.units.n_norm_m3/1e20,births=births,grow=False,elementary_charge=EL_CHG,
                execution_budget=self.execution_budget,locate_with_hints=locate,transport_policy=transport_policy,
                boundary_exit=(None if boundary is None else
                    lambda cloud,ids,velocity,dt,previous_dt:boundary.exits(mesh,cloud,ids,velocity,dt,previous_dt)))
            if self.spatial_backend=='batched_v1':
                diagnostic['sampling']=sampling
            return result,diagnostic


class NativeNeonAtomicRates:
    """Charge-resolved ADAS chain, explicit PLT reference or PLT+PRB mode.

    ADF11 axes are log10(cm^-3), log10(eV), coefficients cm3/s or W cm3.
    This intentionally does not call the historical axis-fitting CE code.
    """
    def __init__(self,tables,*,radiation_mode="plt_reference",prb=None):
        if radiation_mode not in ("plt_reference","plt_prb"):
            raise ValueError("unsupported radiation mode")
        if (radiation_mode=="plt_prb") != (prb is not None):
            raise ValueError("explicit PRB required only for PLT+PRB mode")
        if tables.source!='adas' or tables.z_max!=10:
            raise ValueError('real complete Ne ADAS tables required; no surrogate')
        self.tables=deepcopy(tables)
        self.prb=deepcopy(prb)
        if prb is not None:
            if not isinstance(prb,(tuple,list)) or len(prb)!=3:
                raise ValueError("complete PRB ADF11 tuple required")
            n,t,c=map(np.asarray,prb)
            if (n.ndim!=1 or t.ndim!=1 or min(len(n),len(t))<2
                or c.shape!=(10,len(t),len(n)) or not all(np.isfinite(a).all() for a in (n,t,c))
                or not (np.diff(n)>0).all() or not (np.diff(t)>0).all()):
                raise ValueError("invalid complete PRB axes/charge rows")
        self.version='native-ne-ADAS-SCD-ACD-PLT-v1:'+configuration_digest(dict(
            scd=tables.scd,acd=tables.acd,plt=tables.plt,potentials=tables.ionization_potentials_ev))
        if prb is not None:
            self.version="native-ne-ADAS-SCD-ACD-PLT-PRB-v2:"+configuration_digest(dict(
                base=self.version,prb=prb,charge_rows="PLT0:9_PRB1:10",
                input_units="eV_m^-3",coefficient_units="log10_W_cm3",output_units="W_per_atom"))

    def __call__(self,te_ev,ne_si):
        from .native_atomic_interpolation import interpolate_charge_table
        ne=np.asarray(ne_si);te=np.asarray(te_ev)
        if te.shape!=ne.shape or te.ndim!=1 or not np.isfinite(te).all() or not np.isfinite(ne).all() or min(te.min(),ne.min())<=0:
            raise ValueError('positive finite atomic SI density/eV temperature required')
        logn=np.log10(ne*1e-6);logt=np.log10(te)
        def interp(table):
            return interpolate_charge_table(table,logn,logt)
        scd=interp(self.tables.scd)*1e-6;acd=interp(self.tables.acd)*1e-6
        rad=np.zeros((len(te),11));rad[:,:10]=interp(self.tables.plt)*(ne*1e-6)[:,None]
        if self.prb is not None:
            rad[:,1:]+=interp(self.prb)*(ne*1e-6)[:,None]
        return scd,acd,rad


from dataclasses import dataclass
import torch
from torch import nn
from .marker_feedback import ProjectionUnits,MarkerFeedbackParameters


@dataclass
class TorchConsumedLoads:
    direct: torch.Tensor         # native unique-DOF test-space vector; already dt integrated
    history: torch.Tensor        # native unique-DOF vector; delta_scale already applied
    alpha_e: torch.Tensor
    ne: torch.Tensor
    zeff: torch.Tensor
    pressure: torch.Tensor
    pressure_gradient: torch.Tensor
    electron_capacity: torch.Tensor  # raw rho+alpha*rho_imp; NOT its inverse
    mean_charge: torch.Tensor
    mean_square_charge: torch.Tensor
    mean_charge_gradient: torch.Tensor
    charge_history: torch.Tensor
    energy_rate: torch.Tensor
    projection_history_value: torch.Tensor


class TorchMarkerLoads(nn.Module):
    """Differentiable ordinary H1 formula and load consumer, no field response.

    Autograd differentiates this nonlinear *explicit* consumer; it must not be
    mistaken for the source's initialized-zero native matrix linearization.
    """
    def __init__(self,mesh,*,parameters,temperature_floor):
        super().__init__()
        units=mesh.units
        pu=ProjectionUnits(units.projection_convention,units.n_norm_m3,units.t_norm_si,units.energy_norm_j_m3)
        self.ratio=parameters.validate(pu)
        if not np.isfinite(temperature_floor) or temperature_floor<=0:
            raise ValueError('resolved positive temperature correction threshold required')
        self.units=units;self.parameters_physics=parameters;self.temperature_floor=temperature_floor
        self.n_dofs=mesh.n_dofs
        self.n_nodes=mesh.n_nodes;self.mesh_configuration=deepcopy(mesh.configuration)
        arrays=dict(inverse=mesh.inverse,node_positions=mesh.coordinates[:,0,0,:].T,vertex=mesh.vertex.T,basis=np.stack([g.test_basis[:,:,:,0] for g in mesh.gauss]),
            local_indices=mesh.local_indices,volume=np.stack([g.weights*g.jacobian*g.coordinates[:,0] for g in mesh.gauss]))
        # Precompute complete local physical gradient basis from independent native evaluator.
        gradients=[]
        for e,ids in enumerate(mesh.vertex.T):
            geo=NativeElementGeometry(mesh.coordinates[:,:,:,ids],mesh.size[:,:,e],3)
            g=evaluate_native_element(geo,np.eye(16).reshape(16,4,1,4),rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1))
            gradients.append(np.stack([g.d_x,g.d_y],axis=-1).reshape(16,4,4,2).transpose(0,2,1,3))
        arrays['gradient_basis']=np.stack(gradients)
        if mesh.global_basis is not None:arrays['global_basis']=np.stack(mesh.global_basis)
        for name,a in arrays.items():self.register_buffer(name,torch.from_numpy(np.array(a,copy=True)))

    @property
    def checkpoint_configuration(self):
        return dict(version='native-torch-marker-loads-v2_local_scatter',mesh=self.mesh_configuration,
            parameters=vars(self.parameters_physics),temperature_floor=self.temperature_floor,
            coverage={'electron_direct':'local_load','projection_history':'local_load',
                'electron_capacity':'formula_verified','pressure':'formula_verified',
                'resistivity':'unimplemented','ion_electron_exchange':'unimplemented'},
            linearization='autograd_explicit_not_JOREK_matrix')

    def evaluate(self,a,channels):
        if (not isinstance(a,torch.Tensor) or a.shape!=(channels,4,1,self.n_nodes)
            or a.dtype!=self.basis.dtype or a.device!=self.basis.device or not torch.isfinite(a).all()):
            raise ValueError('full finite native coefficients/history with matching dtype/device required')
        local=a[:,:,0,self.vertex].permute(2,3,1,0)
        return (torch.einsum('eqvd,evdf->eqf',self.basis,local),
            torch.einsum('eqvdk,evdf->eqfk',self.gradient_basis,local))

    def weak_load(self,values):
        local=torch.einsum('eqvd,eq,eq->evd',self.basis,self.volume,values)
        return values.new_zeros(self.n_dofs).index_add(0,self.local_indices.flatten(),local.flatten())

    @staticmethod
    def correction(a,low,tail):
        # Clamp exponent in inactive branch as torch.where evaluates both arms.
        return torch.where(a<low+tail,low+tail*torch.exp(torch.clamp((a-low-tail)/tail,max=0)),a)

    def forward(self,u,p,raw_deltas,*,time,dt_si):
        from .marker_state import FluidTimeCoefficients
        if (not isinstance(time,FluidTimeCoefficients) or not np.isfinite([dt_si,time.delta_scale,time.theta,time.zeta]).all()
            or dt_si<=0 or time.delta_scale<=0 or not (time.theta in (.5,1.))
            or not (time.zeta==0 or np.isclose(time.zeta,time.delta_scale/(1+time.delta_scale),rtol=1e-13,atol=0))):
            raise ValueError('supported explicit positive fluid time coefficients required')
        f,fg=self.evaluate(u,9);v,vg=self.evaluate(p,5);d,_=self.evaluate(raw_deltas,5)
        rho,ri,te=f[:,:,4],f[:,:,7],f[:,:,8]
        n=v[:,:,4].clamp_min(.5);q=v[:,:,3].clamp_min(0)/n;q2=v[:,:,2].clamp_min(0)/n
        dq=(d[:,:,3]-q*d[:,:,4])*time.delta_scale/n
        qgrad=(vg[:,:,3]-q[:,:,None]*vg[:,:,4])/n[:,:,None]
        qgrad=torch.where((q>self.parameters_physics.nuclear_charge)[:,:,None],0.,qgrad)
        z=q.clamp_max(self.parameters_physics.nuclear_charge)
        tc=self.correction(te,self.temperature_floor/2,self.temperature_floor/2)
        z=torch.where(tc/(EL_CHG*MU_ZERO*self.units.n_norm_m3)<.1,0.,z)
        alpha=self.ratio*z-1.;dalpha=self.ratio*dq
        rc=self.correction(rho,1e-12,1e-8);ric=self.correction(ri,1e-12,1e-8)
        ne=self.correction(rc+alpha*ric,1e-4,1e-4)
        zeff=((q2*n+(rc-ric).clamp_min(0))/ne).clamp_min(1.)
        capacity=rho+alpha*ri
        pressure=capacity*te
        grad=(fg[:,:,4]+alpha[:,:,None]*fg[:,:,7]+ri[:,:,None]*self.ratio*qgrad)*te[:,:,None]+capacity[:,:,None]*fg[:,:,8]
        energy=-(v[:,:,0]+v[:,:,1].clamp_min(0))*(dt_si/self.units.t_norm_si)
        hist=-dalpha*te*ri
        direct=self.weak_load(energy)
        history=self.weak_load(hist)
        out=TorchConsumedLoads(direct,history,alpha,ne,zeff,pressure,grad,capacity,z,q2,qgrad,dq,-(v[:,:,0]+v[:,:,1].clamp_min(0)),hist)
        if any(not torch.isfinite(a).all() for a in vars(out).values()):raise ValueError('nonfinite marker consumption')
        return out


@dataclass(frozen=True)
class NativeDensitySource:
    values: torch.Tensor        # [2,4,1,N], background/impurity MASS-density rates per JU
    units: str='model502_mass_density_per_JU'
    provenance: str='explicit_native_density_source'


class SmallNativeResponse(TorchMarkerLoads):
    """User-authorized frozen consistent scalar capacity, NOT nine-field solve.

    Electron direct/history loads use only 99 capacity. 95/98 and all transport,
    pressure, resistive and exchange operators are excluded. Independent shard
    rates drive mass rows5/8 using their consistent unit-capacity matrices.
    """
    def __init__(self,mesh,**kwargs):
        if mesh.projection!='dense':raise ValueError('frozen capacity reference requires small dense mesh')
        super().__init__(mesh,**kwargs)
        self.register_buffer('mass',torch.from_numpy(mesh.mass.copy()))

    @property
    def checkpoint_configuration(self):
        c=super().checkpoint_configuration
        c['response']='small_consistent_frozen99_and_density_v1'
        c['authorization']='user_explicit_approximation_after_JOREK_block_comparison'
        c['coverage'].update(electron_direct='approximate_response',projection_history='approximate_response',
            electron_capacity='used_as_coefficient',shard_density='approximate_response')
        c['omitted_blocks']=['95','98','all_transport','pressure_force','resistivity','ion_electron_exchange']
        return c

    def increment(self,u,p,raw_projection,shard,*,time,dt_si):
        loads=self(u,p,raw_projection,time=time,dt_si=dt_si)
        if (not isinstance(shard,NativeDensitySource) or shard.units!='model502_mass_density_per_JU'
            or not shard.provenance):raise ValueError('independent native shard density source with units/provenance required')
        rates,_=self.evaluate(shard.values,2)
        if (loads.electron_capacity<=0).any():raise ValueError('nonpositive raw electron capacity unsupported')
        matrix=torch.einsum('eqi,eqj,eq,eq->ij',self.global_basis,self.global_basis,self.volume,
            loads.electron_capacity)*(1+time.zeta)
        energy=torch.linalg.solve(matrix,loads.direct+loads.history)
        # Density injection is already a native *rate*; integrate once here.
        density_load=torch.einsum('eqd,eq,eqf->df',self.global_basis,self.volume,rates)*(dt_si/self.units.t_norm_si)
        density=torch.linalg.solve(self.mass*(1+time.zeta),density_load).T
        zero=u.new_zeros(self.mass.shape[0])
        rows=[zero,zero,zero,zero,density[0]+density[1],zero,zero,density[1],energy]
        vector=torch.stack(rows)
        return vector[:,self.inverse][:,:,None,:],loads


class CoupledNativeField(nn.Module):
    """Full native joint learning by default; explicit frozen99 reference alternative.

    No implicit upgrade of a field-only checkpoint; the complete declaration
    binds source physics and response semantics in the existing saver/driver.
    """
    def __init__(self,field,response,*,source_configuration,shard_configuration,mode="joint_learning",source_scales=None):
        super().__init__();self.field=field;self.response=response
        if field.units!=response.units or field.geometry.n_nodes!=response.n_nodes:
            raise ValueError('coupled source/field geometry or units mismatch')
        if not np.array_equal(field.geometry.pos,response.node_positions.cpu().numpy()):
            raise ValueError('coupled source/field geometry positions mismatch')
        if field.feature_path is not None:
            evaluator=field.feature_path.evaluator
            for name in ('vertex','basis','gradient_basis','volume'):
                if not torch.equal(getattr(evaluator,name),getattr(response,name)):
                    raise ValueError('coupled native feature geometry/basis mismatch: '+name)
        if not np.array_equal(field.update_mapping.inverse.cpu().numpy().reshape(4,-1),response.inverse.cpu().numpy()):
            raise ValueError('source and field native shared DOF mapping mismatch')
        if not source_configuration or not shard_configuration:raise ValueError('coupled physics versions required')
        self.source_configuration=deepcopy(source_configuration);self.shard_configuration=deepcopy(shard_configuration)
        if mode not in ('joint_learning','frozen99_reference'):
            raise ValueError('unsupported coupling mode')
        self.mode=mode
        if mode=='joint_learning':
            from .native_source_features import JointSourceEncoding,SourceFeatureScales
            if isinstance(response,SmallNativeResponse) or field.feature_path is None or not isinstance(source_scales,SourceFeatureScales):
                raise ValueError('joint learning requires formula-only consumer, full Gauss features and explicit source scales')
            self.source_encoding=JointSourceEncoding(response,field.feature_path.evaluator,scales=source_scales,hidden=field.encoder.out_features)
        else:
            if not isinstance(response,SmallNativeResponse) or source_scales is not None:
                raise ValueError('frozen99 reference requires response only; modes cannot be mixed')
            self.source_encoding=None

    @property
    def units(self):return self.field.units
    @property
    def geometry(self):return self.field.geometry
    @property
    def encoder(self):return self.field.encoder
    @property
    def temporal_core(self):return self.field.temporal_core
    @property
    def feature_path(self):return self.field.feature_path
    @property
    def checkpoint_configuration(self):
        c=deepcopy(self.field.checkpoint_configuration)
        c.update(coupled_version='native-source-field-v3_bound_events',coupling_mode=self.mode,
            feedback_policy='explicit_two_source_'+self.mode,
            source_encoding=None if self.source_encoding is None else self.source_encoding.checkpoint_configuration,
            response=self.response.checkpoint_configuration,source=self.source_configuration,shards=self.shard_configuration)
        return c
    def forward(self,u,raw_fluid_deltas,*,marker_feedback=None,projection_deltas=None,shard_source=None,source_time=None,field_event=None,**kwargs):
        from .marker_state import fluid_time_coefficients
        dt=kwargs['dt_si']
        if self.mode=='joint_learning':
            from .native_source_features import FieldEventContext
            if not isinstance(field_event,FieldEventContext):
                raise ValueError('trusted current field event context required')
            field_event.validate_source(source_time,dt_si=dt,units=self.units)
            # Exact no-op does not assemble a physical event. Still require valid
            # sources/metadata, using the preceding positive interval for formula validation.
            eval_dt=kwargs['previous_dt_si'] if dt==0 else dt
            time=fluid_time_coefficients(kwargs['scheme'],eval_dt,kwargs['previous_dt_si'])
            features,encoded=self.source_encoding(kwargs.get("observation_values",u),marker_feedback,projection_deltas,shard_source,
                source_time=source_time,time=time,dt_si=eval_dt)
            learned=self.field(u,raw_fluid_deltas,source_encoding=encoded,**kwargs)
            learned.source_features=features
            return learned
        if field_event is not None:raise ValueError('bound source event belongs to joint mode')
        if source_time is not None:raise ValueError('source_time conditioning belongs to joint mode only')
        learned=self.field(u,raw_fluid_deltas,**kwargs)
        if dt==0:return learned
        time=fluid_time_coefficients(kwargs['scheme'],dt,kwargs['previous_dt_si'])
        physical,_=self.response.increment(u,marker_feedback,projection_deltas,shard_source,time=time,dt_si=dt)
        out=self.field.update_mapping(u,raw_fluid_deltas,learned.increment+physical)
        out.memory=learned.memory
        return out
    def reference_solver(self,context):
        from .native_field_step import NativeReferenceSolver
        return NativeReferenceSolver(self,context)
    def coupled_inputs(self,shards,feedback,device):
        if feedback is None or feedback.layout!='native_full' or feedback.deltas is None:
            raise ValueError('complete explicit marker feedback/history required')
        if not isinstance(shards,dict) or 'native_density_source' not in shards:
            raise ValueError('missing independent shard density source')
        s=shards['native_density_source']
        if not isinstance(s,NativeDensitySource):raise ValueError('explicit native density source required')
        extra={}
        if self.mode=='joint_learning':
            from .native_source_features import SourceEventTime
            phase=('registered_restart_zero_projection' if feedback.history_provenance=='jorek_restart_zero_projection' else 'ordinary')
            extra['source_time']=SourceEventTime(feedback.time_si,feedback.gather_time_si,phase,units=self.units)
        return dict(**extra,marker_feedback=torch.as_tensor(feedback.values,device=device),
            projection_deltas=torch.as_tensor(feedback.deltas,device=device),
            shard_source=NativeDensitySource(s.values.to(device),s.units,s.provenance))


class SmallShardModel:
    """Pure Ne small-cloud Adapter: Parks(3), no rocket/nonlocal, resolved Gaussian.

    Radius erosion uses PREVIOUS ablation (literal update_spi ordering), then
    ablation is evaluated in the OLD field at advanced shard position. Gaussian
    density and one center marker/shard/window are explicit spatial reductions.
    """
    def __init__(self,mesh,*,width_m,material_density_m3,ablation_model):
        if mesh.projection_n0 is not None:
            raise ValueError("independent shard density requires its own unfiltered reference projector, not marker projection")
        if ablation_model!=3 or not np.isfinite([width_m,material_density_m3]).all() or min(width_m,material_density_m3)<=0:
            raise ValueError('supported explicit shard configuration required')
        self.mesh=mesh;self.width_m=width_m;self.material_density_m3=material_density_m3
        self.configuration=dict(version='small-native-Ne-shard-v1',mesh=mesh.configuration,
            width_m=width_m,material_density_m3=material_density_m3,ablation_model=3,
            approximations=['no_rocket','no_nonlocal','normalized_axisymmetric_Gaussian_density',
                'one_center_marker_per_shard_per_window','toroidal_density_average_1_over_2pi'])

    def validate(self,s):
        if not isinstance(s,dict):raise ValueError('complete small shard state required')
        n=len(s.get('radius',[]))
        for name,shape in (('positions',(n,3)),('velocities',(n,3)),('radius',(n,)),('ablation_atoms_s',(n,))):
            if name not in s or np.shape(s[name])!=shape or not np.isfinite(s[name]).all():raise ValueError('invalid shard '+name)
        if (s['radius']<0).any() or (s['ablation_atoms_s']<0).any():raise ValueError('negative shard physical state')

    def density_source(self,s):
        self.validate(s);mesh=self.mesh
        rates=np.zeros((len(mesh.gauss),16))
        for pos,r,abl in zip(s['positions'],s['radius'],s['ablation_atoms_s']):
            if r<=0 or abl==0:continue
            kernel=np.stack([np.exp(-np.sum((g.coordinates-pos[:2])**2,axis=1)/(2*self.width_m**2)) for g in mesh.gauss])
            vol=np.stack([g.weights*g.jacobian*g.coordinates[:,0] for g in mesh.gauss])
            norm=np.sum(kernel*vol)
            if norm<=0:raise ValueError('shard Gaussian has no supported mesh overlap')
            rates+=kernel/norm*(abl*mesh.units.t_norm_si/mesh.units.n_norm_m3*20/mesh.units.central_mass/(2*np.pi))
        rhs=mesh.weak_load(np.asarray(rates)[:,:,None])[0]
        imp=mesh.project(rhs[None])[0]
        return NativeDensitySource(torch.from_numpy(np.stack([np.zeros_like(imp),imp])),
            provenance='small_shard_atoms_s_to_axisymmetric_mass_rate_Gaussian_v1')

    def births(self,s,marker,target,previous,current,source):
        from .marker_state import MarkerCloud,particle_window_schedule
        self.validate(s);_,_,gather=particle_window_schedule(marker.clock,target)
        live=(s['radius']>0)&(s['ablation_atoms_s']>0)
        cloud=MarkerCloud.empty(10,int(live.sum()));cloud.x[:]=s['positions'][live]
        elm,st=self.mesh.locate(cloud.x);cloud.i_elm[:]=elm;cloud.st[:]=st
        cloud.weight[:]=s['ablation_atoms_s'][live]*gather
        start=marker.clock.particle_time_si;clock=marker.clock
        fields=previous+(current-previous)*((start-clock.field_start_si)/(clock.field_end_si-clock.field_start_si))
        for i in np.flatnonzero(cloud.alive):
            v,g,rz,_,_=self.mesh.point(int(elm[i])-1,st[i],fields)
            b=np.array([g[0,1],-g[0,0],source.F0])/rz[0]
            cloud.V_prev[i]=np.array([-g[1,1],g[1,0],0])*rz[0]/self.mesh.units.t_norm_si+v[6]/self.mesh.units.t_norm_si*b
        return cloud

    def update(self,s,old_fields,time,dt):
        from src.physics.spi_model import parks_ablation_neon
        self.validate(s);out=deepcopy(s);mesh=self.mesh
        out.pop('native_density_source',None)
        out['positions']=s['positions']+s['velocities']*dt
        for i in range(len(s['radius'])):
            r=s['radius'][i]
            if r<=0:out['ablation_atoms_s'][i]=0.;continue
            out['radius'][i]=max(0.,r-dt*s['ablation_atoms_s'][i]/(4*np.pi*r*r*self.material_density_m3))
            elm,st=mesh.locate(out['positions'][i:i+1])
            if elm[0]<=0 or out['radius'][i]<=0:out['ablation_atoms_s'][i]=0.;continue
            v,_,_,_,_=mesh.point(int(elm[0])-1,st[0],old_fields)
            te=max(v[8]/(EL_CHG*MU_ZERO*mesh.units.n_norm_m3),0.)
            ne=max((v[4]-v[7])*mesh.units.n_norm_m3,0.)
            args=[torch.tensor(a,dtype=torch.float64) for a in (out['radius'][i],te,ne,1.)]
            out['ablation_atoms_s'][i]=float(parks_ablation_neon(*args))
        self.validate(out);return out


def source_time_sampler(source,marker,current,*,fluid_deltas,completed_dt_si):
    """Bind saved history at source event, before next fluid control is consumed."""
    if getattr(source,'time_sampling','endpoint_reference')=='endpoint_reference':return None
    if getattr(source,'time_sampling',None)!='jorek_raw_deltas_v1':
        raise ValueError('unsupported source time-sampling mode')
    from .native_source_time import NativeSourceTimeSampler
    if fluid_deltas is None or completed_dt_si is None:
        raise ValueError('complete rawD and completed fluid step required for source sampling')
    return NativeSourceTimeSampler(current,fluid_deltas,time_prev_si=marker.clock.field_start_si,
        time_now_si=marker.clock.field_end_si,tstep_si=completed_dt_si,static=source.field_static)


def advance_source_event(candidate,source,shard_model,marker,target,previous,current,shards):
    """One saved-history sampler shared by births and the marker window."""
    from .native_source_production import NativeProductionShardModel
    sampler=source_time_sampler(source,marker,current,fluid_deltas=candidate.fluid_deltas,
        completed_dt_si=candidate.fluid_dt_previous_si)
    options={} if sampler is None else {'time_sampler':sampler}
    birth_options=options if isinstance(shard_model,NativeProductionShardModel) else {}
    births=shard_model.births(shards,marker,target,previous,current,source,**birth_options)
    return source.advance(marker,target,previous,current,births=births,**options)


def bind_coupled_equilibrium(candidate,shard_model,equilibrium_provider):
    """Version the external ES diagnostic; never infer missing physical state."""
    from .native_source_production import NativeProductionShardModel
    if not isinstance(shard_model,NativeProductionShardModel):
        if equilibrium_provider is not None:
            raise ValueError('equilibrium provider requires production shard diagnostics')
        return
    if (not callable(equilibrium_provider)
        or not isinstance(getattr(equilibrium_provider,'configuration',None),dict)
        or not equilibrium_provider.configuration):
        raise ValueError('explicit configured equilibrium provider required for production event')
    digest=configuration_digest(equilibrium_provider.configuration)
    teacher_configuration=shard_model.configuration.get('psi_teacher')
    if teacher_configuration is not None and configuration_digest(teacher_configuration)!=digest:
        raise ValueError('psi teacher/provider configuration mismatch')
    previous=candidate.shards.get('equilibrium_provider_digest')
    if previous is not None and previous!=digest:
        raise ValueError('equilibrium provider configuration changed across continuation')
    candidate.shards['equilibrium_provider_digest']=digest


def refresh_coupled_diagnostics(out,shard_model,equilibrium_provider):
    """Refresh on a private candidate inside the caller's full rollback scope."""
    from .native_source_production import NativeProductionShardModel
    bind_coupled_equilibrium(out,shard_model,equilibrium_provider)
    if isinstance(shard_model,NativeProductionShardModel):
        fields=shard_model.diagnostic_fields(out.fields_current,out.fluid_time_si)
        equilibrium=equilibrium_provider(fields.copy(),out.fluid_time_si)
        from .native_source_nonlocal import NativeFluxEquilibrium
        if (not isinstance(equilibrium,NativeFluxEquilibrium)
            or equilibrium.field_digest!=configuration_digest(fields)
            or equilibrium.time_si!=out.fluid_time_si):
            raise ValueError('equilibrium provider returned a different field/time event')
        out.shards=shard_model.refresh_diagnostics(out.shards,fields,out.marker.feedback,
            equilibrium,provenance='coupled_event:'+out.shards['equilibrium_provider_digest'])
        out.shards['diagnostic_field_binding']=dict(time_si=out.fluid_time_si,
            predicted_field_digest=configuration_digest(out.fields_current),
            diagnostic_field_digest=configuration_digest(fields),
            mode=shard_model.configuration.get('psi_feedback_mode','predicted_field'))
    return out


def advance_coupled_reference(state,*,model,source,shard_model,equilibrium_provider=None):
    """Existing event driver + concrete source/response/shard physics; atomic.

    NumPy inference only. Use CoupledNativeField.forward for Torch gradients;
    this source path and legacy ReferenceRollout are explicitly nondifferentiable.
    """
    import random
    from .marker_state import reference_particle_then_fluid,reference_fluid_event
    if (model.source_configuration!=source.configuration or model.shard_configuration!=shard_model.configuration):
        raise ValueError('coupled source/shard configuration mismatch')
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('small NumPy source/event reference supports CPU models only; Torch field seam is separate')
    snapshot={k:v.clone() for k,v in model.state_dict().items()}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    modes={n:m.training for n,m in model.named_modules()}
    try:
        candidate=deepcopy(state)
        bind_coupled_equilibrium(candidate,shard_model,equilibrium_provider)
        candidate.shards['native_density_source']=shard_model.density_source(candidate.shards)
        solver=model.reference_solver(candidate.native_context)
        if candidate.marker.clock.startup_pending:
            out=reference_fluid_event(candidate,solve_fluid=solver,update_shards=shard_model.update)
            diagnostics={'event':'registered_restart_zero_projection_startup'}
        else:
            def particle(marker,target,previous,current,shards):
                return advance_source_event(candidate,source,shard_model,marker,target,previous,current,shards)
            out,diagnostics=reference_particle_then_fluid(candidate,particle_step=particle,
                solve_fluid=solver,update_shards=shard_model.update)
        refresh_coupled_diagnostics(out,shard_model,equilibrium_provider)
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items()):
            raise ValueError('inference event mutated model parameters/buffers')
        return out,diagnostics
    except Exception:
        model.load_state_dict(snapshot)
        for n,m in model.named_modules():m.training=modes[n]
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        raise
