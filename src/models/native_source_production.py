"""Inspected Ne source/shard mechanisms behind the existing coupled-event seam.

The caller owns actual ES diagnostics and their event timing. A model checkpoint
keeps diagnostic caches in shards; the global NumPy birth stream is kept by the
existing full RNG checkpoint. No hidden mutable physics state lives here.
"""
from copy import deepcopy
import numpy as np
from .native_source_nonlocal import NativeNonlocalDiagnostic,NativeFluxEquilibrium


class NativeProductionShardModel:
    def __init__(self,dynamics,density,birth,*,n_flux,n_table):
        if (type(n_flux) is not int or n_flux<2 or type(n_table) is not int or n_table<2
            or birth.kernel.configuration!=density.configuration
            or dynamics.mesh.configuration['geometry']!=density.mesh.configuration['geometry']
            or dynamics.mesh.units!=density.mesh.units
            or dynamics.configuration['ns_radius_min']!=density.minimum
            or dynamics.configuration['ns_radius_ratio']!=density.ratio):
            raise ValueError('coherent explicit source/shard mechanisms required')
        self.dynamics=dynamics;self.density=density;self.birth=birth
        self.n_flux=n_flux;self.n_table=n_table
        self.configuration=dict(version='inspected-Ne-production-mechanisms-v1',
            dynamics=deepcopy(dynamics.configuration),density=deepcopy(density.configuration),
            birth=deepcopy(birth.configuration),n_flux=n_flux,n_nonlocal_array=n_table,
            equilibrium='required_external_actual_ES_not_inferred',diagnostics='explicit_postcommit_event',
            differentiability='numpy_stop_gradient',production_build='not_certified')

    def density_source(self,s):
        self.dynamics.validate(s)
        return self.density.density_source(s)

    def diagnostic_fields(self,fields,time_si):
        """Default autonomous input policy: diagnostics see the predicted fields."""
        self.dynamics.mesh.validate_fields(fields,9)
        return np.array(fields,copy=True)

    def births(self,s,marker,target,previous,current,source,*,time_sampler=None):
        self.dynamics.validate(s)
        return self.birth.births(s,marker,target,previous,current,source,time_sampler=time_sampler)

    def update(self,s,old_fields,time,dt):
        out=self.dynamics.update(s,old_fields,time,dt)
        if dt==0:return out
        out['time_si']=time+dt
        return out

    def refresh_diagnostics(self,s,fields,feedback,equilibrium,*,provenance):
        """After commit, use actual new U/ES and installed (possibly lagged) aux."""
        self.dynamics.validate(s)
        if (not isinstance(equilibrium,NativeFluxEquilibrium)
            or not np.isfinite(s.get('time_si',np.nan)) or s['time_si']!=equilibrium.time_si
            or feedback.layout!='native_full'):
            raise ValueError('diagnostics require committed shard/field event and installed native feedback')
        out=self.density.refresh_volumes(s,time_si=equilibrium.time_si,provenance=provenance)
        if self.dynamics.nonlocal_abl:
            out['nonlocal_diagnostic']=NativeNonlocalDiagnostic.build(self.dynamics.mesh,fields,feedback.values,
                equilibrium,n_flux=self.n_flux,n_table=self.n_table,feedback_time_si=feedback.time_si,provenance=provenance)
            out['old_equilibrium']=deepcopy(equilibrium)
        return out
