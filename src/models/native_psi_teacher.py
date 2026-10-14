"""Recorded-psi intervention for diagnostic/shard validation, NOT autonomous AR.

The committed network state is never overwritten. Only the nonlocal/shard
diagnostic branch receives recorded psi; its other eight fields stay predicted.
Marker advection and birth velocity continue to use the committed network field.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from .native_state import configuration_digest
from .native_source_nonlocal import NativeFluxEquilibrium
from .native_source_production import NativeProductionShardModel


class RecordedPsiFeedback:
    mode='recorded_psi_diagnostic_intervention'

    def __init__(self,records,*,mesh,time_tolerance_si=1e-14):
        if not np.isfinite(time_tolerance_si) or not 0<=time_tolerance_si<=1e-14:
            raise ValueError('explicit narrow SI record matching tolerance required')
        self._records=tuple(deepcopy(tuple(records)))
        if not self._records:raise ValueError('explicit psi teacher records required')
        self.mesh=mesh;self.time_tolerance_si=float(time_tolerance_si)
        fingerprints=[]
        for r in self._records:
            if (np.shape(r.psi)!=(4,1,mesh.n_nodes) or not np.isfinite(r.psi).all()
                or not np.isfinite(r.time_si) or r.geometry_digest!=mesh.configuration['geometry']
                or r.units!=mesh.units or not r.provenance or not r.source_sha256
                or not isinstance(r.equilibrium,NativeFluxEquilibrium)
                or r.equilibrium.time_si!=r.time_si):
                raise ValueError('same-time complete psi/ES/units/geometry evidence required')
            r.equilibrium.validate()
            r.psi.flags.writeable=False
            fingerprints.append(configuration_digest(dict(psi=r.psi,equilibrium=vars(r.equilibrium),
                time=r.time_si,geometry=r.geometry_digest,units=vars(r.units),
                provenance=r.provenance,source_sha256=r.source_sha256)))
        times=sorted(r.time_si for r in self._records)
        if any(b-a<=2*self.time_tolerance_si for a,b in zip(times,times[1:])):
            raise ValueError('ambiguous duplicate psi teacher event records')
        self.configuration=dict(version='recorded-psi-diagnostic-v1',mode=self.mode,
            record_fingerprints=fingerprints,geometry=mesh.configuration['geometry'],
            units=vars(mesh.units),time_tolerance_si=self.time_tolerance_si,
            replaces='diagnostic_and_shard_branch_psi_only',
            does_not_replace='committed_nine_fields_rawD_marker_advection_birth_velocity',
            autonomous=False)

    def record(self,time_si):
        if not np.isfinite(time_si):raise ValueError('finite psi event time required')
        matches=[r for r in self._records if abs(r.time_si-time_si)<=self.time_tolerance_si]
        if len(matches)!=1:
            raise ValueError('missing same-time psi teacher record; no nearest/future interpolation')
        return deepcopy(matches[0])

    def diagnostic_fields(self,fields,time_si):
        self.mesh.validate_fields(fields,9)
        out=np.array(fields,copy=True);out[0]=self.record(time_si).psi
        return out

    def __call__(self,diagnostic_fields,time_si):
        record=self.record(time_si)
        self.mesh.validate_fields(diagnostic_fields,9)
        if not np.array_equal(diagnostic_fields[0],record.psi):
            raise ValueError('teacher ES requires the explicitly intervened psi branch')
        return replace(record.equilibrium,field_digest=configuration_digest(diagnostic_fields),
            time_si=float(time_si),provenance=self.mode+':'+record.provenance)


class NativePsiTeacherShardModel(NativeProductionShardModel):
    """Same production mechanisms, explicitly distinct diagnostic input policy."""
    def __init__(self,base,teacher):
        if not isinstance(base,NativeProductionShardModel) or not isinstance(teacher,RecordedPsiFeedback):
            raise ValueError('production shard model and explicit recorded psi feedback required')
        if (teacher.mesh.configuration['geometry']!=base.dynamics.mesh.configuration['geometry']
            or teacher.mesh.units!=base.dynamics.mesh.units):
            raise ValueError('psi teacher/shard mesh mismatch')
        super().__init__(base.dynamics,base.density,base.birth,n_flux=base.n_flux,n_table=base.n_table)
        self.teacher=teacher
        self.configuration.update(version='inspected-Ne-psi-teacher-shard-v1',
            psi_feedback_mode=teacher.mode,psi_teacher=deepcopy(teacher.configuration),autonomous=False)

    def diagnostic_fields(self,fields,time_si):
        return self.teacher.diagnostic_fields(fields,time_si)

    def update(self,s,old_fields,time,dt):
        # OLD here is the current committed network state; only its diagnostic
        # psi is replaced. Cached ES/table must match this same explicit branch.
        return super().update(s,self.diagnostic_fields(old_fields,time),time,dt)

    def refresh_diagnostics(self,s,fields,feedback,equilibrium,*,provenance):
        branch=self.diagnostic_fields(fields,equilibrium.time_si)
        out=super().refresh_diagnostics(s,branch,feedback,equilibrium,provenance=provenance)
        record=self.teacher.record(equilibrium.time_si)
        out['psi_teacher_event']=dict(mode=self.teacher.mode,time_si=equilibrium.time_si,
            record_time_si=record.time_si,source_sha256=record.source_sha256,
            psi_digest=configuration_digest(record.psi),provenance=record.provenance,
            diagnostic_field_digest=configuration_digest(branch),autonomous=False)
        return out
