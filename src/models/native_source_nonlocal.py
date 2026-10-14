"""Inspected model502/Ne marker nonlocal diagnostic, with explicit cache provenance.

No equilibrium finder: the caller supplies the actual ES diagnostic bound to its
field event. No cache refresh is inferred from a shard update or window split.
"""
from dataclasses import dataclass
import numpy as np
from .native_state import configuration_digest
from .marker_feedback import EL_CHG, MU_ZERO


@dataclass(frozen=True)
class NativeFluxEquilibrium:
    psi_axis: float
    psi_bnd: float
    lcfs_lost: bool
    xcase: str
    psi_xpoints: tuple
    z_xpoints: tuple
    time_si: float
    field_digest: str
    provenance: str

    @classmethod
    def bind(cls, fields, **kwargs):
        result = cls(field_digest=configuration_digest(fields), **kwargs)
        result.validate()
        return result

    def validate(self):
        if (self.xcase not in ('none','lower','upper','double','symmetric')
            or not isinstance(self.lcfs_lost, bool)
            or np.shape(self.psi_xpoints)!=(2,) or np.shape(self.z_xpoints)!=(2,)
            or not np.isfinite([self.psi_axis,self.psi_bnd,self.time_si,*self.psi_xpoints,*self.z_xpoints]).all()
            or not self.field_digest or not self.provenance):
            raise ValueError('explicit finite equilibrium diagnostic required')

    def normalized_flux(self, psi, z=None):
        self.validate()
        psi = np.asarray(psi)
        if self.lcfs_lost or abs(self.psi_bnd-self.psi_axis)<1e-6:
            return np.full_like(psi,1.01,dtype=float)
        result = (psi-self.psi_axis)/(self.psi_bnd-self.psi_axis)
        if z is not None and self.xcase!='none':
            z = np.asarray(z)
            xp = (np.asarray(self.psi_xpoints)-self.psi_axis)/(self.psi_bnd-self.psi_axis)
            if self.xcase!='upper':
                result = np.where((result<xp[0])&(z<self.z_xpoints[0]),2*xp[0]-result,result)
            if self.xcase!='lower':
                result = np.where((result<xp[1])&(z>self.z_xpoints[1]),2*xp[1]-result,result)
        return result


@dataclass(frozen=True)
class NativeNonlocalDiagnostic:
    psi: np.ndarray
    temperature_ev: np.ndarray
    electron_density_m3: np.ndarray
    shell_volume_m3: np.ndarray
    equilibrium: NativeFluxEquilibrium
    time_si: float
    field_digest: str
    feedback_digest: str
    feedback_time_si: float
    provenance: str
    mesh_digest: str
    version: str = 'inspected-model502-Ne-marker-nonlocal-v1'

    @classmethod
    def build(cls, mesh, fields, feedback, equilibrium, *, n_flux, n_table,
              feedback_time_si, provenance):
        mesh.validate_fields(fields,9); mesh.validate_fields(feedback,5)
        equilibrium.validate()
        if (type(n_flux) is not int or n_flux<2 or type(n_table) is not int or n_table<2
            or not provenance or not np.isfinite(feedback_time_si)
            or feedback_time_si>equilibrium.time_si
            or equilibrium.field_digest!=configuration_digest(fields)):
            raise ValueError('nonlocal diagnostic requires matching complete field/feedback/equilibrium event')
        volumes=np.zeros(n_flux); numbers=np.zeros(n_flux); energies=np.zeros(n_flux)
        ratio=mesh.units.central_mass/20*(2/2.01455279245)
        for ids,g in zip(mesh.vertex.T,mesh.gauss):
            basis=g.test_basis[:,:,:,0]
            u=np.einsum('qvd,fvd->qf',basis,fields[:,:,0,ids].transpose(0,2,1))
            p=np.einsum('qvd,fvd->qf',basis,feedback[:,:,0,ids].transpose(0,2,1))
            flux=equilibrium.normalized_flux(u[:,0],g.coordinates[:,1])
            inside=(flux>=0)&(flux<1)
            # corr_neg_dens(c5, (.1,.1), .001), then capped projected mean charge.
            density=np.where(p[:,4]<2e-4,1e-4+1e-4*np.exp(np.minimum((p[:,4]-2e-4)/1e-4,0)),p[:,4])
            charge=np.minimum(np.maximum(p[:,3],0)/density,10.)
            ne=u[:,4]+(ratio*charge-1)*u[:,7]  # raw r0/rimp0 in shell accumulation
            weight=g.weights*g.jacobian*g.coordinates[:,0]*2*np.pi
            shell=np.minimum(np.floor(flux[inside]*n_flux).astype(int),n_flux-1)
            np.add.at(volumes,shell,weight[inside])
            np.add.at(numbers,shell,(ne*weight)[inside])
            np.add.at(energies,shell,(ne*u[:,8]*weight)[inside])
        if (volumes<=0).any() or (numbers<=0).any():
            raise ValueError('nonlocal diagnostic has empty shell or nonpositive electron total')
        te=energies/numbers/(EL_CHG*MU_ZERO*mesh.units.n_norm_m3)
        ne=numbers/volumes*mesh.units.n_norm_m3
        coarse=equilibrium.psi_axis+np.arange(n_flux)/n_flux*(equilibrium.psi_bnd-equilibrium.psi_axis)
        psi=equilibrium.psi_axis+np.arange(n_table)/n_table*(equilibrium.psi_bnd-equilibrium.psi_axis)
        values=[]
        for x in psi:
            i=int(np.argmin(abs(coarse-x)))
            j=i+1 if abs(x-equilibrium.psi_axis)>=abs(coarse[i]-equilibrium.psi_axis) else i-1
            if j>=n_flux:j=n_flux-2
            values.append([(a[i]-a[j])/(coarse[i]-coarse[j])*(x-coarse[i])+a[i] for a in (te,ne)])
        values=np.asarray(values)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('nonlocal diagnostic produced invalid extrapolated thermodynamics')
        return cls(psi,values[:,0],values[:,1],volumes,equilibrium,equilibrium.time_si,
            equilibrium.field_digest,configuration_digest(feedback),float(feedback_time_si),provenance,configuration_digest(mesh.configuration))

    def sample(self, psi, z, *, equilibrium, old_fields, time_si, mesh):
        """Literal shard lookup in a cached table using the current OLD ES."""
        return self._bind_event_sampler(equilibrium=equilibrium,old_fields=old_fields,
                                        time_si=time_si,mesh=mesh)(psi,z)

    def _bind_event_sampler(self, *, equilibrium, old_fields, time_si, mesh):
        """Private synchronous-update scope; never persist/reuse across events.

        The shard updater owns immutable inputs for its loop. Bind their full
        provenance once, while standalone sample() still validates every call.
        This is not a cache of field hashes or a token accepting later inputs.
        """
        if (not isinstance(equilibrium,NativeFluxEquilibrium)
            or equilibrium.field_digest!=configuration_digest(old_fields)
            or equilibrium.time_si!=time_si or self.time_si>time_si
            or self.mesh_digest!=configuration_digest(mesh.configuration)):
            raise ValueError('nonlocal cache/current OLD-field diagnostic provenance mismatch')
        def sample_bound(psi,z):
            if equilibrium.normalized_flux(psi,z)>1:
                return 0.,0.
            i=int(np.argmin(abs(self.psi-psi)))
            if abs(psi-equilibrium.psi_axis)<abs(self.psi[i]-equilibrium.psi_axis):i=max(0,i-1)
            return float(self.temperature_ev[i]),float(self.electron_density_m3[i])
        return sample_bound
