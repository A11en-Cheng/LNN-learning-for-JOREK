"""Inspected pure-Ne Parks3/rocket6 dynamics, separate from birth/density.

Only nonrotating, nondrifting pure-Ne injectors. This is an update Interface,
not a complete production shard/source Adapter. Cached nonlocal tables and the
current OLD-field equilibrium are explicit diagnostic dependencies.
"""
from copy import deepcopy
import numpy as np
from .marker_feedback import MU_ZERO,EL_CHG
from .native_state import configuration_digest


class InspectedNeonShardDynamics:
    def __init__(self,mesh,*,material_density_m3,F0,R_geo,magnetic_reduction,
                 nonlocal_abl,ns_radius_min,ns_radius_ratio):
        if (not np.isfinite([material_density_m3,F0,R_geo,ns_radius_min,ns_radius_ratio]).all()
            or min(material_density_m3,R_geo,ns_radius_min,ns_radius_ratio)<=0
            or not isinstance(magnetic_reduction,bool) or not isinstance(nonlocal_abl,bool)):
            raise ValueError('explicit inspected shard configuration required')
        self.mesh=mesh;self.material_density_m3=float(material_density_m3)
        self.F0=float(F0);self.R_geo=float(R_geo)
        self.magnetic_reduction=magnetic_reduction;self.nonlocal_abl=nonlocal_abl
        self.configuration=dict(version=('inspected-Ne-Parks3-rocket6-update-v2_nonlocal' if nonlocal_abl
                                         else 'inspected-Ne-Parks3-rocket6-update-v1'),mesh=mesh.configuration,
            material_density_m3=material_density_m3,F0=F0,R_geo=R_geo,
            magnetic_reduction=magnetic_reduction,nonlocal_abl=nonlocal_abl,
            ns_radius_min=ns_radius_min,ns_radius_ratio=ns_radius_ratio,
            scope=('update_with_explicit_cached_nonlocal_diagnostic' if nonlocal_abl
                   else 'update_only_no_birth_no_density_no_nonlocal_table_reconstruction'),
            differentiability='numpy_stop_gradient',species='pure_Ne',drift=False,tor_rotation=False)

    def validate(self,s):
        if not isinstance(s,dict):raise ValueError('complete shard dictionary required')
        n=len(s.get('radius',[]))
        for name,shape in (('positions',(n,3)),('velocities',(n,3)),('radius',(n,)),
            ('ablation_atoms_s',(n,)),('injector_phi',(n,)),('species',(n,)),('ids',(n,))):
            if name not in s or np.shape(s[name])!=shape or not np.isfinite(s[name]).all():
                raise ValueError('complete finite shard '+name+' required')
        if (s.get('positions_coordinate_system')!='RZphi'
            or s.get('velocities_coordinate_system')!='injector_R_Z_RxZ'
            or (np.asarray(s['species'])!=1).any() or (s['positions'][:,0]<=0).any()
            or (s['radius']<0).any() or (s['ablation_atoms_s']<0).any()
            or len(np.unique(s['ids']))!=n):
            raise ValueError('pure Ne explicit identities and injector velocity coordinates required')

    @staticmethod
    def _ablation(radius,te_ev,ne_m3):
        return (27.0837*6.02214076e23/20.183*(radius*100/.2)**(4/3)
                *(ne_m3*1e-20)**(1/3)*(te_ev/2000)**(5/3))

    def update(self,s,old_fields,time,dt):
        self.validate(s);self.mesh.validate_fields(old_fields,9)
        if not np.isfinite([time,dt]).all() or dt<0:raise ValueError('finite SI update time/dt required')
        if self.nonlocal_abl:
            from .native_source_nonlocal import NativeNonlocalDiagnostic,NativeFluxEquilibrium
            if (not isinstance(s.get('nonlocal_diagnostic'),NativeNonlocalDiagnostic)
                or not isinstance(s.get('old_equilibrium'),NativeFluxEquilibrium)):
                raise ValueError('nonlocal spatial diagnostic required: cached table and current OLD equilibrium')
        if dt==0:return deepcopy(s)
        out=deepcopy(s);out.pop('native_density_source',None)
        delta=s['positions'][:,2]-s['injector_phi']
        radial=s['velocities'][:,0]*np.cos(delta)+s['velocities'][:,2]*np.sin(delta)
        angular=(s['velocities'][:,2]*np.cos(delta)-s['velocities'][:,0]*np.sin(delta))/s['positions'][:,0]
        out['positions'][:,0]+=radial*dt
        out['positions'][:,1]+=s['velocities'][:,1]*dt
        out['positions'][:,2]=np.mod(s['positions'][:,2]+angular*dt,2*np.pi)
        nonlocal_sample=None
        for i,r in enumerate(s['radius']):
            if r<=0:continue  # Fortran moves first, then skips dead shard state.
            out['radius'][i]=max(0.,r-dt*s['ablation_atoms_s'][i]/(4*np.pi*r*r*self.material_density_m3))
            elm,st=self.mesh.locate(out['positions'][i:i+1])
            if elm[0]<=0:
                out['ablation_atoms_s'][i]=0.;continue
            v,_,_,_,_=self.mesh.point(int(elm[0])-1,st[0],old_fields)
            # Literal use_marker Z_imp=0 placeholder in pellet_module Parks3.
            # This differs from the charge-aware nonlocal spatial diagnostic.
            n=max(v[4]*self.mesh.units.n_norm_m3,0.)
            nimp=max(v[7]*self.mesh.units.n_norm_m3,0.)
            ne=max(n-nimp,0.)
            te=max(v[8]/(EL_CHG*MU_ZERO*self.mesh.units.n_norm_m3),0.)
            radius=out['radius'][i]
            abl=self._ablation(radius,te,ne)
            if self.nonlocal_abl:
                # Lazy binding preserves zero-step/dead/outside-domain behavior.
                # The binding never escapes this update or survives to another.
                if nonlocal_sample is None:
                    nonlocal_sample=s['nonlocal_diagnostic']._bind_event_sampler(
                        equilibrium=s['old_equilibrium'],old_fields=old_fields,time_si=time,mesh=self.mesh)
                te_nl,ne_nl=nonlocal_sample(v[0],out['positions'][i,1])
                abl=max(abl,self._ablation(radius,te_nl,ne_nl))
            b0=abs(self.F0/self.R_geo)
            if self.magnetic_reduction and b0>2:abl*=(2/b0)**.843
            out['ablation_atoms_s'][i]=abl
            if radius>0:
                pressure=1e5*(te/2000)**1.26*(ne/1e20)**.4
                mass=4*np.pi/3*max(radius,1e-5)**3*20*1.660539040e-27*self.material_density_m3
                acceleration=pressure*np.pi*radius**2/mass
                # spi_delta_phi was computed BEFORE position update in Fortran.
                out['velocities'][i,0]+=acceleration*np.cos(delta[i])*dt
                out['velocities'][i,2]+=acceleration*np.sin(delta[i])*dt
        self.validate(out)
        return out
