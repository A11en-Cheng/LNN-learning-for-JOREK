"""Frozen pre-accepted-trial-cache Newton algorithms (2026-10-09).

Only search control flow is frozen: geometry is evaluated through the original
full point evaluator, independently of optimized location helpers.
"""
import numpy as np

from src.models.native_coupling import NativeSourceMesh


class FrozenLocatorMesh(NativeSourceMesh):
    def _location_geometry(self, e, st):
        result = self.point(e, st, np.zeros((1, 4, 1, self.n_nodes)))
        return result[2], result[4]

    def _location_geometries(self, e, st):
        result = self.points(e, st, np.zeros((1, 4, 1, self.n_nodes)))
        return result[2], result[4]

    def locate(self,positions,*,element_hints=None,local_hints=None):
        positions=np.asarray(positions)
        if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
            raise ValueError('finite R/Z/phi positions required')
        if element_hints is not None:
            if (np.shape(element_hints)!=(len(positions),) or np.shape(local_hints)!=(len(positions),2)
                or not np.isfinite(local_hints).all()):
                raise ValueError('matching element/local hints required')
        elements=np.zeros(len(positions),dtype=np.int32);local=np.zeros((len(positions),2))
        for i,x in enumerate(positions):
            candidates=self._location_tree.query_ball_point(x[:2],self._location_radius)
            candidates=sorted(candidates,key=lambda e:np.linalg.norm(self._location_centers[e]-x[:2]))
            hint=-1 if element_hints is None else int(element_hints[i])-1
            if 0<=hint<len(self.gauss):candidates=[hint]+[e for e in candidates if e!=hint]
            for e in candidates:
                lower,upper=self._location_bounds[e]
                if np.any(x[:2]<lower-1e-9) or np.any(x[:2]>upper+1e-9):continue
                st=np.clip(local_hints[i],0,1).copy() if e==hint else np.array([.5,.5])
                for _ in range(25):
                    try:
                        rz,jac=self._location_geometry(e,st)
                        error=np.linalg.norm(rz-x[:2])
                        if error<1e-10:
                            elements[i]=e+1;local[i]=st;break
                        step=np.linalg.solve(jac,rz-x[:2])
                    except (ValueError,np.linalg.LinAlgError):break
                    accepted=False
                    for damping in (1.,.5,.25,.125,.0625,.03125):
                        trial=np.clip(st-damping*step,0,1)
                        try:trial_rz=self._location_geometry(e,trial)[0]
                        except ValueError:continue
                        if np.linalg.norm(trial_rz-x[:2])<error:
                            st=trial;accepted=True;break
                    if not accepted:break
                if elements[i]>0:break
        return elements,local

    def locate_batch(self,positions,*,element_hints=None,local_hints=None):
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
        for _ in range(25):
            if not len(active):break
            try:
                rz,jac=self._location_geometries(e[active],st[active])
                error=np.linalg.norm(rz-x[active,:2],axis=1)
                done=error<1e-10;ids=active[done]
                elements[ids]=e[ids]+1;local[ids]=st[ids]
                remain=~done;active=active[remain]
                if not len(active):break
                step=np.linalg.solve(jac[remain],(rz[remain]-x[active,:2])[...,None])[...,0]
                error=error[remain];accepted=np.zeros(len(active),bool)
                for damping in (1.,.5,.25,.125,.0625,.03125):
                    todo=np.flatnonzero(~accepted)
                    if not len(todo):break
                    ids=active[todo];trial=np.clip(st[ids]-damping*step[todo],0,1)
                    trial_rz=self._location_geometries(e[ids],trial)[0]
                    better=np.linalg.norm(trial_rz-x[ids,:2],axis=1)<error[todo]
                    st[ids[better]]=trial[better];accepted[todo[better]]=True
                active=active[accepted]
            except (ValueError,np.linalg.LinAlgError):
                break
        pending=np.flatnonzero(elements==0)
        if len(pending):
            pe,ps=self.locate(x[pending],element_hints=hints[pending],local_hints=np.asarray(local_hints)[pending])
            elements[pending]=pe;local[pending]=ps
        return elements,local
