"""Inspected pure-Ne circular SPI density kernel, independent of marker feedback.

Numerical source volumes are explicit cached diagnostics. H1 density uses the
declared discrete toroidal planes and an independent unfiltered representation.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import erf, sqrt, pi
import numpy as np
from .native_state import configuration_digest


@dataclass(frozen=True)
class PreparedDensityRates:
    """Owned read-only coefficients for one event, never a cross-event cache."""
    blocks: tuple
    deltaphi: float

    def rates_at(self, positions, *, toroidal_average=False):
        positions=np.asarray(positions)
        if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
            raise ValueError('finite RZphi points required')
        result=np.zeros(len(positions))
        for pos,width_squared,amplitude,average in self.blocks:
            for start in range(0,len(positions),32):
                stop=min(start+32,len(positions));points=positions[start:stop]
                pol=np.exp(-np.sum((points[None,:,:2]-pos[:,None,:2])**2,axis=2)/width_squared)
                if toroidal_average:
                    tor=average[:,None]
                else:
                    angle=np.abs((points[None,:,2]-pos[:,None,2]+pi)%(2*pi)-pi)
                    tor=np.exp(-(angle/self.deltaphi)**2)
                contribution=amplitude[:,None]*pol*tor
                # Identical shard ordering and previous-block initial value.
                result[start:stop]=np.cumsum(np.concatenate((result[None,start:stop],contribution),axis=0),axis=0)[-1]
        return result


class NativeShardDensityKernel:
    def __init__(self, mesh, *, ns_radius_min, ns_radius_ratio, ns_deltaphi,
                 spi_num_vol, ns_delta_minor_rad, injection_time_si, toroidal_angles):
        values=[ns_radius_min,ns_radius_ratio,ns_deltaphi,ns_delta_minor_rad,injection_time_si]
        if (not np.isfinite(values).all() or min(values[:3])<=0 or ns_delta_minor_rad!=0
            or not isinstance(spi_num_vol,bool)):
            raise ValueError('explicit circular Ne density configuration required; elongated source unsupported')
        self.mesh=mesh;self.minimum=float(ns_radius_min);self.ratio=float(ns_radius_ratio)
        self.deltaphi=float(ns_deltaphi);self.numerical=spi_num_vol
        self.injection_time_si=float(injection_time_si)
        self.angles=np.array(toroidal_angles,dtype=float,copy=True)
        if (self.angles.ndim!=1 or len(self.angles)<1 or not np.isfinite(self.angles).all()
            or (self.angles<0).any() or (self.angles>=2*pi).any()
            or len(np.unique(self.angles))!=len(self.angles)):
            raise ValueError('explicit distinct n_period=1 toroidal quadrature planes required')
        self.toroidal_integral=self.deltaphi*sqrt(pi)*erf(pi/self.deltaphi)
        self.configuration=dict(version='inspected-Ne-circular-SPI-density-v1',
            mesh=mesh.configuration,ns_radius_min=ns_radius_min,ns_radius_ratio=ns_radius_ratio,
            ns_deltaphi=ns_deltaphi,spi_num_vol=spi_num_vol,ns_delta_minor_rad=0.,
            injection_time_si=injection_time_si,toroidal_angles=self.angles.tolist(),
            representation='independent_unfiltered_H1_discrete_plane_average_nperiod1',
            units='model502_mass_density_per_JU',differentiability='numpy_stop_gradient')

    def validate(self,s):
        if not isinstance(s,dict):raise ValueError('complete density shard dictionary required')
        n=len(s.get('radius',[]))
        for name,shape in (('positions',(n,3)),('radius',(n,)),('ablation_atoms_s',(n,)),('species',(n,))):
            if name not in s or np.shape(s[name])!=shape or not np.isfinite(s[name]).all():
                raise ValueError('complete finite density shard '+name+' required')
        if (np.asarray(s['positions'])[:,0]<=0).any() or (np.asarray(s['radius'])<0).any() or (np.asarray(s['ablation_atoms_s'])<0).any() or (np.asarray(s['species'])!=1).any():
            raise ValueError('pure Ne circular source requires positive R and nonnegative radius/rate')

    def widths(self,s):
        self.validate(s)
        return np.maximum(np.asarray(s['radius'])*self.ratio,self.minimum)

    def _volumes(self,s,time_si):
        widths=self.widths(s)
        analytic=pi*s['positions'][:,0]*self.toroidal_integral*widths**2
        if not self.numerical:return analytic
        if ('source_volume_m3' not in s or np.shape(s['source_volume_m3'])!=np.shape(widths)
            or not np.isfinite(s['source_volume_m3']).all() or (s['source_volume_m3']<0).any()
            or not s.get('source_volume_provenance') or not np.isfinite(s.get('source_volume_time_si',np.nan))
            or s['source_volume_time_si']>time_si):
            raise ValueError('numerical source volume cache with diagnostic event provenance required')
        # Source literal uses analytical fallback for a present zero volume.
        return np.where(s['source_volume_m3']>0,s['source_volume_m3'],analytic)

    def rates_at(self,s,positions,*,time_si,toroidal_average=False):
        """Impurity mass-density rate per JU at RZphi points (no dt factor)."""
        self.validate(s);positions=np.asarray(positions)
        if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all() or not np.isfinite(time_si):
            raise ValueError('finite SI event time and RZphi points required')
        return self.prepare_rates(s,time_si=time_si).rates_at(positions,toroidal_average=toroidal_average)

    def prepare_rates(self,s,*,time_si):
        """Bind one immutable event snapshot; public point queries still validate.

        Pre-injection zero does not require volume diagnostics. Nothing is
        retained on the kernel, and subsequent events must prepare afresh.
        """
        self.validate(s)
        if not np.isfinite(time_si):raise ValueError('finite SI event time required')
        if time_si<=self.injection_time_si:return PreparedDensityRates((),self.deltaphi)
        volumes=self._volumes(s,time_si)
        scale=self.mesh.units.t_norm_si/self.mesh.units.n_norm_m3*20/self.mesh.units.central_mass
        widths=self.widths(s)
        live=np.flatnonzero((np.asarray(s['radius'])>0)&(np.asarray(s['ablation_atoms_s'])!=0))
        blocks=[]
        for begin in range(0,len(live),256):
            ids=live[begin:begin+256]
            pos=np.asarray(s['positions'])[ids]
            amplitude=np.asarray(s['ablation_atoms_s'])[ids]*scale/volumes[ids]
            angle=np.abs((self.angles[None,:]-pos[:,2,None]+pi)%(2*pi)-pi)
            average=np.mean(np.exp(-(angle/self.deltaphi)**2),axis=1)
            block=[]
            for value in (pos,widths[ids,None]**2,amplitude,average):
                owned=np.array(value,copy=True);owned.flags.writeable=False;block.append(owned)
            blocks.append(tuple(block))
        return PreparedDensityRates(tuple(blocks),float(self.deltaphi))

    def _plane_average(self,phi):
        angle=np.abs((self.angles-phi+pi)%(2*pi)-pi)
        return np.mean(np.exp(-(angle/self.deltaphi)**2))

    def refresh_volumes(self,s,*,time_si,provenance):
        """Explicit int3d diagnostic event; caller chooses its actual lifecycle."""
        self.validate(s)
        if not provenance or not np.isfinite(time_si):raise ValueError('source-volume diagnostic event required')
        volumes=np.zeros(len(s['radius']))
        widths=self.widths(s)
        toroidal=np.array([self._plane_average(pos[2]) for pos in s['positions']])
        for g in self.mesh.gauss:
            # Batch independent shards, not the element reduction. Preserve
            # each shard's Gauss/element accumulation and bound scratch memory.
            for start in range(0,len(widths),256):
                stop=min(start+256,len(widths))
                delta=g.coordinates[None,:,:]-s['positions'][start:stop,None,:2]
                pol=np.exp(-np.sum(delta**2,axis=2)/widths[start:stop,None]**2)
                volumes[start:stop]+=np.sum(pol*g.weights*g.jacobian*g.coordinates[:,0],axis=1)*2*pi*toroidal[start:stop]
        out=deepcopy(s)
        out.update(source_volume_m3=volumes,source_volume_time_si=float(time_si),source_volume_provenance=provenance)
        return out

    def density_source(self,s):
        """Independent unfiltered coefficients for the existing joint-field seam."""
        if self.mesh.projection_n0 is not None or self.mesh.projection=='none':
            raise ValueError('independent density requires an unfiltered projector, not marker projection')
        from .native_coupling import NativeDensitySource
        import torch
        time=s.get('time_si',np.nan)
        prepared=self.prepare_rates(s,time_si=time)
        rates=np.stack([prepared.rates_at(np.column_stack([g.coordinates,np.zeros(len(g.coordinates))]),
            toroidal_average=True) for g in self.mesh.gauss])
        imp=self.mesh.project(self.mesh.weak_load(rates[:,:,None]))[0]
        return NativeDensitySource(torch.from_numpy(np.stack([np.zeros_like(imp),imp])),
            provenance='inspected_Ne_density_kernel_independent_unfiltered_H1_'+configuration_digest(self.configuration))
