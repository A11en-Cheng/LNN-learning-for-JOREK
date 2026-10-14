"""Opt-in terminal step on a saved exterior face of the current cubic cell.

This verifies the frozen variable-direction pusher trajectory, not the exact
transport ODE. Unknown neighbours, multi-cell exits, corners and tangencies
are NOT certified. Ordinary inverse-map failure is never evidence of death.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial

from .native_state import configuration_digest


def _cross(a, b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _st(face, u):
    return np.array(((u,0.),(1.,u),(1.-u,1.),(0.,1.-u))[face])


@dataclass(frozen=True)
class BoundaryExit:
    dt_si: float
    x: np.ndarray
    st: np.ndarray
    element: int
    edge: int
    residual_m: float


class NativeMarkerBoundaryExit:
    """One seam: actual topology + geometry -> optional verified terminal step."""
    tolerance_m = 5e-12

    def __init__(self, mesh, saved_neighbours):
        if saved_neighbours is None:
            raise ValueError('saved neighbours required for verified boundary exit')
        # Reuse the saved-table reciprocity/endpoint validation, NOT its Newton
        # result or lost flag. Negative entries are unknown, zero is exterior.
        from .native_nearby_locator import NativeNearbyLocator
        topology = NativeNearbyLocator(mesh,saved_neighbours)
        self.neighbours = topology.neighbours.copy()
        self.neighbours.flags.writeable = False
        self._neighbour_snapshot = self.neighbours.copy()
        self._neighbour_snapshot.flags.writeable = False
        self._snapshot = {k:np.array(getattr(mesh,k),copy=True)
            for k in ('coordinates','size','vertex','indices')}
        for a in self._snapshot.values(): a.flags.writeable = False
        self._geometry = mesh.configuration['geometry']
        self._topology = configuration_digest(np.asarray(saved_neighbours))
        self.edges = {}
        for e in np.flatnonzero(np.any(self.neighbours==0,axis=1)):
            curves = []
            for face in range(4):
                p0,j0 = mesh._location_geometry(int(e),_st(face,0.))
                p1,j1 = mesh._location_geometry(int(e),_st(face,1.))
                direction = np.array(((1.,0.),(0.,1.),(-1.,0.),(0.,-1.))[face])
                d0,d1 = j0@direction,j1@direction
                coefficients = np.stack((p0,d0,3*(p1-p0)-2*d0-d1,2*(p0-p1)+d0+d1))
                controls = np.stack((p0,p0+d0/3,p1-d1/3,p1))
                curves.append((coefficients,controls.min(0),controls.max(0)))
            self.edges[int(e)] = curves

    @property
    def configuration(self):
        return dict(version='saved-cubic-current-cell-exit-v2_side_witness',geometry=self._geometry,
            saved_neighbours=self._topology,tolerance_m=self.tolerance_m,
            trajectory='frozen_variable_interval_direction_quadratic_RZ',
            verification='first_root_transverse_and_resolved_inside_outside_same_cell_witness',
            terminal='negative_old_element_and_exact_face_st',
            unsupported='unknown_neighbour_corner_tangent_multicell_or_failed_locator',
            approximation='fractional_last_charge_and_old_position_load_not_JOREK_full_step_death')

    def validate(self, mesh):
        if (mesh.configuration['geometry']!=self._geometry or
            any(not np.array_equal(getattr(mesh,k),a) for k,a in self._snapshot.items())):
            raise ValueError('boundary exit geometry changed; rebuild provider')
        # Keep the original saved dtype in the fingerprint (no cast on restore).
        if not np.array_equal(self.neighbours, self._neighbour_snapshot):
            raise ValueError('boundary exit topology changed')

    def exits(self, mesh, cloud, ids, velocity, dt, previous_dt):
        result = {}
        for i in np.flatnonzero(np.any(self.neighbours[cloud.i_elm[ids].astype(np.int64)-1]==0,axis=1)):
            slot = int(ids[i]); e = int(cloud.i_elm[slot])-1
            event = self._exit(mesh,e,cloud.x[slot],cloud.st[slot],cloud.V_prev[slot],
                velocity[i],dt,float(previous_dt[i]))
            if event is not None: result[i] = event
        return result

    def _exit(self, mesh, e, x, st, previous, velocity, dt, previous_dt):
        if not np.isfinite(st).all() or ((st<0)|(st>1)).any():
            raise ValueError('boundary exit needs an inside current-cell coordinate')
        speed,old_speed = np.linalg.norm(velocity),np.linalg.norm(previous)
        if min(speed,old_speed,previous_dt,dt,x[0])<=0:
            raise ValueError('boundary exit needs a valid direction history')
        linear = dt*velocity[:2]
        quadratic = dt*dt*speed*(velocity[:2]/speed-previous[:2]/old_speed)/(2*previous_dt)
        # Quadratic Bezier hull bounds the entire proposed path, not just end.
        controls = np.stack((x[:2],x[:2]+linear/2,x[:2]+linear+quadratic))
        lo,hi = controls.min(0)-self.tolerance_m,controls.max(0)+self.tolerance_m
        curves = self.edges[e]
        if not any(np.all(hi>=lower)&np.all(lo<=upper) for face,(_,lower,upper) in enumerate(curves)
                   if self.neighbours[e,face]==0):
            return None
        # The ordinary global locator accepts a looser inverse-map residual.
        # Refine its hint ONLY for the certificate; do not replace sampled st,
        # field values, current x, or choose a different cell/locator here.
        inside=self._inverse(mesh,e,x[:2],st)
        if ((inside<0)|(inside>1)).any():
            raise ValueError('boundary exit current position/geometry mismatch or outside current cell')
        hits = []
        for face,(coefficients,lower,upper) in enumerate(curves):
            if not (np.all(hi>=lower) and np.all(lo<=upper)): continue
            hits.extend((tau,face,u,residual) for tau,u,residual in
                self._intersections(coefficients,x[:2],linear,quadratic))
        if not hits: return None
        hits.sort()
        tau,face,u,residual = hits[0]
        # First face must itself be known exterior; never skip an earlier
        # internal/unknown crossing to find a later apparent exterior root.
        if self.neighbours[e,face]!=0: return None
        if u<=1e-9 or u>=1-1e-9 or any(abs(a[0]-tau)<1e-9 and a[1]!=face for a in hits[1:]):
            raise ValueError('ambiguous corner boundary exit')
        if tau<=1e-12: raise ValueError('zero-duration boundary exit is unsupported')
        local = _st(face,u)
        _,jac = mesh._location_geometry(e,local)
        motion = np.linalg.solve(jac,linear+2*quadratic*tau)
        outward = (-motion[1],motion[0],motion[1],-motion[0])[face]
        if outward<=1e-12*np.linalg.norm(motion):
            raise ValueError('tangent/inward boundary exit is unsupported')
        # Repeated polynomial roots can split into a pair of tiny apparent
        # crossings in floating point. A positive derivative alone then gives
        # false absorption. Require spatially RESOLVED opposite sides of the
        # full cubic map, not another scalar tangent threshold.
        probe=min(1e-3,tau/2)
        later=[a[0]-tau for a in hits[1:] if a[0]-tau>1e-10]
        if later:probe=min(probe,min(later)/4)
        before=self._inverse(mesh,e,x[:2]+linear*(tau-probe)+quadratic*(tau-probe)**2,local)
        after=self._inverse(mesh,e,x[:2]+linear*(tau+probe)+quadratic*(tau+probe)**2,local)
        coordinate,outward_sign=((1,-1),(0,1),(1,1),(0,-1))[face]
        edge_value=(0.,1.,1.,0.)[face]
        resolution=4*self.tolerance_m*np.linalg.norm(np.linalg.inv(jac)[coordinate])
        if (outward_sign*(before[coordinate]-edge_value)>=-resolution
            or outward_sign*(after[coordinate]-edge_value)<=resolution):
            raise ValueError('unresolved/tangent boundary contact; no verified inside-outside crossing')
        from .marker_transport import push_variable_direction
        accepted = float(dt*tau)
        pushed = push_variable_direction(x[None],previous[None],velocity[None],accepted,
            np.array([previous_dt]))[0]
        edge_rz,_ = mesh._location_geometry(e,local)
        error = float(np.linalg.norm(pushed[:2]-edge_rz))
        if error>self.tolerance_m: raise ValueError('boundary intersection residual exceeds geometry tolerance')
        return BoundaryExit(accepted,pushed,local,-(e+1),face,max(residual,error))

    def _inverse(self, mesh, e, target, hint):
        st=hint.copy()
        for _ in range(8):
            rz,jac=mesh._location_geometry(e,st)
            if np.linalg.norm(rz-target)<=self.tolerance_m:return st
            st-=np.linalg.solve(jac,rz-target)
        rz,_=mesh._location_geometry(e,st)
        if np.linalg.norm(rz-target)>self.tolerance_m:
            raise ValueError('boundary exit current position/geometry mismatch')
        return st

    def _intersections(self, coefficients, origin, linear, quadratic):
        """Cubic curve / quadratic path roots, refined against BOTH coordinates.

        Eliminate tau: C*cross(B-origin,linear)+cross(B-origin,quadratic)^2=0,
        C=cross(linear,quadratic). Nearly straight paths use line seeds but are
        still solved and verified against the full unmodified quadratic path.
        """
        from scipy.optimize import root
        c=coefficients.copy();c[0]-=origin
        scale=max(float(np.linalg.norm(linear)+np.linalg.norm(quadratic)),
                  float(np.max(np.abs(c))),np.finfo(float).tiny)
        c/=scale; v=linear/scale; a=quadratic/scale
        direction=v if np.linalg.norm(v)>0 else a
        if np.linalg.norm(direction)==0:return []
        determinant=float(_cross(v,a))
        line=Polynomial(_cross(c,direction))
        curved=Polynomial(_cross(c,a))
        near_line=abs(determinant)<=1e-10*np.linalg.norm(v)*np.linalg.norm(a)
        polynomial=line if near_line else determinant*line+curved*curved
        if not np.any(polynomial.coef): return []  # coincident/tangent: no proof
        candidates=[]
        for r in polynomial.roots():
            if abs(r.imag)>1e-6 or not -1e-8<=r.real<=1+1e-8: continue
            u=float(r.real); delta=np.array([Polynomial(c[:,j])(u) for j in range(2)])
            if near_line:
                along=direction/np.linalg.norm(direction)
                # Solve the along-path coordinate without ignoring acceleration.
                q=Polynomial([-np.dot(delta,along),np.dot(v,along),np.dot(a,along)])
                times=[float(z.real) for z in q.roots() if abs(z.imag)<1e-6]
            else: times=[float(_cross(delta,a)/determinant)]
            for tau in times:
                if not -1e-7<=tau<=1+1e-7: continue
                def evaluate(y):
                    b=np.array([Polynomial(c[:,j])(y[0]) for j in range(2)])
                    return b-v*y[1]-a*y[1]**2
                def jacobian(y):
                    db=np.array([Polynomial(c[:,j]).deriv()(y[0]) for j in range(2)])
                    return np.column_stack((db,-v-2*a*y[1]))
                solved=root(evaluate,[u,tau],jac=jacobian,tol=1e-11)
                uu,tt=solved.x
                residual=float(np.linalg.norm(evaluate(solved.x))*scale)
                if residual<=self.tolerance_m and -1e-10<=uu<=1+1e-10 and -1e-12<=tt<=1+1e-12:
                    candidates.append((float(np.clip(tt,0.,1.)),float(np.clip(uu,0.,1.)),residual))
        return candidates
