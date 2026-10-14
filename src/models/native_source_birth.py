"""Inspected SPI births with optional explicit n0 tabulated position experiment.

This stream is MT19937, not production PCG32/MPI/OpenMP replay. Seven uniforms
per pending slot per rejection round preserve the DEFAULT inspected protocol.
The opt-in table has a different, fingerprinted four-uniform position stream.
The event owner already checkpoints/rolls back np.random.get_state() in full.
"""
import numpy as np
from .marker_state import MarkerCloud,particle_window_schedule
from .native_birth_acceptance import NATIVE_BIRTH_ACCEPTANCE,native_birth_threshold


class NativeProductionBirths:
    def __init__(self,kernel,*,physical_particles,n_particle_slots,n_cpu,rank,R_geo,n_plane,max_proposals,
                 position_sampler=None):
        if (not np.isfinite([physical_particles,R_geo]).all() or min(physical_particles,R_geo)<=0
            or any(type(i) is not int or i<=0 for i in (n_particle_slots,n_cpu,n_plane,max_proposals))
            or type(rank) is not int or not 0<=rank<n_cpu):
            raise ValueError('explicit production birth normalization/topology/budget required')
        self.kernel=kernel;self.weight=physical_particles/n_particle_slots
        # Both 0.01 literals in Fortran are default real: round their product
        # before promotion by real*8 R_geo, then assign the final real*4 dv.
        centimeter_squared=float(np.float32(.01)*np.float32(.01))
        self.n_cpu=n_cpu;self.rank=rank;self.dv=float(np.float32(centimeter_squared*R_geo*(2*np.pi/n_plane)))
        self.max_proposals=max_proposals
        self.position_sampler=position_sampler
        self.configuration=dict(version='inspected-SPI-birth-distribution-v2',kernel=kernel.configuration,
            physical_particles=physical_particles,n_particle_slots=n_particle_slots,n_cpu=n_cpu,rank=rank,
            R_geo=R_geo,n_plane=n_plane,max_proposals=max_proposals,
            rng='numpy_global_MT19937_full_state',reproduction='distribution_only_not_production_PCG32',
            weight='float32_rate_times_actual_gather_div_count',velocity_time='sim_time_not_residual_start',
            acceptance=NATIVE_BIRTH_ACCEPTANCE)
        if position_sampler is not None:
            from .native_birth_positions import NativeTabulatedBirthSampler
            if not isinstance(position_sampler,NativeTabulatedBirthSampler):
                raise ValueError('supported explicit birth position sampler required')
            # The birth field sampler currently supports only H=1. The separate
            # position module exposes 3D geometry without claiming 3D coupling.
            if position_sampler.configuration['n_tor']!=1 or n_plane!=1:
                raise ValueError('coupled tabulated births currently require H=1 and n_plane=1')
            from copy import deepcopy
            self.configuration.update(version='explicit-tabulated-SPI-birth-v2',
                position_sampler=deepcopy(position_sampler.configuration),
                rng='numpy_global_MT19937_four_uniforms_per_position',
                reproduction='declared_native_gate_table_not_continuous_source_or_production_PCG32')

    def births(self,s,marker,target,previous,current,source,*,time_sampler=None):
        """Existing shard birth seam; consumes OLD shards before particle substeps."""
        if self.configuration.get('acceptance')!=NATIVE_BIRTH_ACCEPTANCE:
            raise ValueError('birth acceptance configuration requires explicit native-mass version')
        if self.position_sampler is not None:
            self.position_sampler.validate_configuration()
            if self.position_sampler.configuration!=self.configuration.get('position_sampler'):
                raise ValueError('birth position sampler differs from checkpoint configuration')
        elif 'position_sampler' in self.configuration:
            raise ValueError('bound birth position sampler missing')
        kernel=self.kernel;mesh=kernel.mesh
        kernel.validate(s);marker.cloud.validate()
        mesh.validate_fields(previous,9);mesh.validate_fields(current,9)
        if (mesh.configuration['geometry']!=source.mesh.configuration['geometry']
            or mesh.units!=source.mesh.units):
            raise ValueError('birth/source geometry/units mismatch')
        if marker.cloud.P_imp.shape[1]!=11:raise ValueError('complete Ne birth charge state required')
        _,_,gather=particle_window_schedule(marker.clock,target)
        if gather<=0:raise ValueError('positive actual particle gather required for births')
        rate=float(np.sum(s['ablation_atoms_s']*s['species']))/self.n_cpu
        count=int(rate*gather/self.weight)
        if rate*gather>1e14:count=max(count,1)
        free=np.count_nonzero(~marker.cloud.alive)
        if count>free:raise ValueError('marker slot capacity exhausted before birth sampling')
        if count>self.max_proposals:raise ValueError('birth proposal budget cannot accommodate requested count')
        cloud=MarkerCloud.empty(10,count);cloud.group=marker.cloud.group
        if count==0:return cloud
        domain=np.stack([mesh.coordinates[0,0,0],mesh.coordinates[1,0,0]],axis=1)
        lo=domain.min(0);hi=domain.max(0)
        widths=kernel.widths(s)
        lower=np.min(s['positions'][:,:2]-5*widths[:,None],axis=0)
        upper=np.max(s['positions'][:,:2]+5*widths[:,None],axis=0)
        # Source falls back to the full domain independently on each missing overlap.
        for d in range(2):
            if upper[d]>lo[d] and hi[d]>lower[d]:
                lo[d]=max(lo[d],lower[d]);hi[d]=min(hi[d],upper[d])
        if (hi<=lo).any() or lo[0]<0:raise ValueError('invalid cylindrical birth bounding box')
        if source.time_sampling=='jorek_raw_deltas_v1':
            from .native_source_time import NativeSourceTimeSampler
            if (not isinstance(time_sampler,NativeSourceTimeSampler)
                or not np.array_equal(time_sampler.current,current)
                or time_sampler.static!=source.field_static
                or time_sampler.time_now_si!=marker.clock.field_end_si
                or time_sampler.time_prev_si!=marker.clock.field_start_si):
                raise ValueError('birth sim-time velocity requires bound saved-history sampler')
            fields=time_sampler.fields_at(marker.clock.sim_time_si)
        else:
            clock=marker.clock
            if clock.field_end_si<=clock.field_start_si:raise ValueError('birth field endpoint clock required')
            fields=previous+(current-previous)*((clock.sim_time_si-clock.field_start_si)/(clock.field_end_si-clock.field_start_si))
        norm=rate*mesh.units.t_norm_si/mesh.units.n_norm_m3
        prepared=(kernel.prepare_rates(s,time_si=marker.clock.field_end_si)
                  if self.position_sampler is None else self.position_sampler.prepare(
                      kernel,s,time_si=marker.clock.field_end_si,bounds=np.stack([lo,hi]),dv=self.dv,norm=norm))
        pending=np.arange(count);used=0
        saved=np.random.get_state()
        try:
            while len(pending):
                rejected=[]
                if self.position_sampler is not None:
                    if used+len(pending)>self.max_proposals:
                        raise ValueError('birth mesh proposal budget exceeded')
                    positions=prepared.sample(len(pending))
                for offset,j in enumerate(pending):
                    if used>=self.max_proposals:raise ValueError('birth rejection proposal budget exceeded')
                    used+=1
                    if self.position_sampler is None:
                        ran=np.random.random(7)
                        x=np.array([np.sqrt(ran[0]*(hi[0]**2-lo[0]**2)+lo[0]**2),
                            lo[1]+ran[1]*(hi[1]-lo[1]),2*np.pi*ran[2]])
                    else:
                        x=positions[offset]
                    # Density rejection is independent of element location.
                    # Keep every seven-uniform draw and the same conjunction
                    # of acceptance predicates, but avoid Newton searches for
                    # proposals already rejected by the physical source kernel.
                    if self.position_sampler is None:
                        acceptance=float(native_birth_threshold(prepared.rates_at(x[None])[0],dv=self.dv,norm=norm))
                        if ran[3]>=acceptance:rejected.append(j);continue
                    elm,st=mesh.locate(x[None])
                    if elm[0]<=0:rejected.append(j);continue
                    cloud.x[j]=x;cloud.i_elm[j]=elm[0];cloud.st[j]=st[0]
                    v,g,rz,_,_=mesh.point(int(elm[0])-1,st[0],fields)
                    b=np.array([g[0,1],-g[0,0],source.F0])/rz[0]
                    cloud.V_prev[j]=np.array([-g[1,1],g[1,0],0])*rz[0]/mesh.units.t_norm_si+v[6]/mesh.units.t_norm_si*b
                pending=np.asarray(rejected,dtype=int)
            cloud.weight[:]=float(np.float32(rate*gather/count))
            cloud.validate()
        except BaseException:
            np.random.set_state(saved)
            raise
        return cloud
