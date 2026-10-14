"""One-shot native checkpoint initialization; never searches later snapshots.

A recorded checkpoint is at the after-fluid/before-next-particle event phase:
fluid/shards at T_k; particle event clock and previous feedback at T_{k-1}.
A continuous snapshot alone omits residual particle time, so requires the
saved MarkerClock and feedback deltas. Program restart instead creates zero
projection and explicitly queues the startup fluid event.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import h5py
from src.data.native_run import _read_native_shard
from src.data.output_contract import parse_macroscopic_times
from src.data.namelist import parse_namelist
from src.models.marker_state import MarkerCloud, MarkerClock, MarkerRuntime, ProjectionFeedback


@dataclass
class NativeMarkerCheckpoint:
    marker: MarkerRuntime
    fluid_values: np.ndarray       # full degrees/harmonics, not silently sliced
    fluid_deltas: np.ndarray
    previous_fluid_values: np.ndarray | None
    shards: dict[str, np.ndarray]
    fluid_time_jorek: float
    t_norm: float
    central_density: float
    provenance: dict


def _scalar(f, key):
    if key not in f or f[key].size != 1:
        raise ValueError(f"{f.filename}: missing/invalid scalar {key}")
    value = float(np.ravel(f[key][()])[0])
    if not np.isfinite(value):
        raise ValueError(f"{f.filename}: nonfinite {key}")
    return value


def _source(path):
    path = Path(path).resolve()
    # File contents can be large; only the namelist gets a complete digest.
    return {"path": str(path), "bytes": path.stat().st_size}


def read_marker_snapshot(path, *, group="001"):
    with h5py.File(path, "r") as f:
        time_si = _scalar(f, "time")
        key = f"groups/{group}"
        if key not in f:
            raise ValueError(f"missing marker group {group}")
        g = f[key]
        kind = np.ravel(g["type"][()])[0]
        if isinstance(kind, bytes): kind = kind.decode().strip()
        if kind != "particle_marker":
            raise ValueError("checkpoint group must contain particle_marker")
        mapping = {name: ("v" if name == "V_prev" else name) for name in MarkerCloud.array_names()}
        for name in mapping.values():
            if name not in g:
                raise ValueError(f"{path}: missing required marker field {name}")
        cloud = MarkerCloud(**{k: np.asarray(g[v][()]) for k,v in mapping.items()}, group=group)
        z = int(_scalar(g, "Z"))
        if cloud.P_imp.shape != (len(cloud.weight), z+1):
            raise ValueError("P_imp must contain every charge state")
        metadata = {"Z": z, "mass_amu": _scalar(g, "mass"),
                    "adas_suffix": str(np.ravel(g["adas_suffix"][()])[0])}
    cloud.validate()
    return cloud, time_si, metadata


def initialize_native_checkpoint(fluid_path, marker_path, *, event_index: int,
                                 times_path, namelist_path,
                                 boundary: str, clock: MarkerClock | None = None,
                                 feedback_path=None, feedback_deltas=None,
                                 previous_fluid_path=None,
                                 confirmed_no_live_markers: bool = False):
    """Validate INTERNAL times against event positions, not matching filenames.

    boundary is continuous or jorek_restart. For continuous recovery the saved
    clock, full previous feedback values/deltas and previous fluid frame are
    mandatory. HDF5 tstep is deliberately not treated as a schedule. A restart
    uses imported HDF5 tstep for particle startup only; control is supplied separately.
    """
    if boundary not in {"continuous", "jorek_restart"}:
        raise ValueError("boundary must be continuous or jorek_restart")
    table = parse_macroscopic_times(times_path)
    if event_index not in table or event_index-1 not in table:
        raise ValueError("checkpoint event times missing from macroscopic table")
    with h5py.File(fluid_path, "r") as f:
        tn, tf = _scalar(f, "t_norm"), _scalar(f, "t_now")
        density, mass = _scalar(f, "central_density"), _scalar(f, "central_mass")
        if min(tn, density, mass) <= 0:
            raise ValueError("checkpoint normalization must be positive")
        expected_tn = np.sqrt(4e-7*np.pi*mass*1.66053906660e-27*density*1e20)
        if not np.isclose(tn, expected_tn, rtol=2e-6, atol=0):
            raise ValueError("t_norm inconsistent with central_density/central_mass SI normalization")
        if _scalar(f,"index_now") != event_index:
            raise ValueError("fluid internal index disagrees with checkpoint event")
        values = np.asarray(f["values"][()])
        if values.ndim != 4 or values.shape[0] != 9 or not np.isfinite(values).all():
            raise ValueError("checkpoint requires full native nine-field values")
        if "deltas" not in f:
            raise ValueError("missing fluid node deltas; field endpoints cannot losslessly reconstruct nodal history")
        deltas = np.asarray(f["deltas"][()])
        if deltas.shape != values.shape or not np.isfinite(deltas).all():
            raise ValueError("fluid node deltas must be finite and match full native values")
        current_tstep = _scalar(f, "tstep")
    if abs(tf-table[event_index]) > 1e-7:
        raise ValueError("fluid internal time disagrees with checkpoint event")
    expected_particle_si = table[event_index-1]*tn
    params = parse_namelist(Path(namelist_path).read_text())
    if 'tstep_particles' not in params:
        raise ValueError('segment namelist missing required stepping parameter tstep_particles')
    nominal=float(params['tstep_particles'])
    if not np.isfinite(nominal) or nominal<=0 or current_tstep<=0:
        raise ValueError('invalid imported/particle step')
    if marker_path is None:
        if not confirmed_no_live_markers:
            raise ValueError("missing marker initial state; cannot infer or zero an active cloud")
        cloud, particle_si, marker_meta = MarkerCloud.empty(), expected_particle_si, {"confirmed_empty": True}
    else:
        cloud, particle_si, marker_meta = read_marker_snapshot(marker_path)
    if abs(particle_si-expected_particle_si) > max(tn*1e-7, 1e-14):
        raise ValueError("marker internal /time mismatches checkpoint particle event (filename ignored)")
    previous = None
    if previous_fluid_path is not None:
        with h5py.File(previous_fluid_path, "r") as f:
            if abs(_scalar(f,"t_now")*tn-particle_si) > 1e-14 or not np.isclose(_scalar(f,"t_norm"),tn,rtol=1e-12,atol=0):
                raise ValueError("previous field internal time/units mismatch")
            previous = np.asarray(f["values"][()])
            if previous.shape != values.shape:
                raise ValueError("field history shape mismatch")
    shape = (5, values.shape[1], values.shape[2], values.shape[3])
    if boundary == "continuous":
        if clock is None or feedback_path is None or feedback_deltas is None or previous is None:
            raise ValueError("continuous initialization requires saved clock, full feedback/deltas and previous field")
        if abs(clock.sim_time_si-particle_si) > 1e-14:
            raise ValueError("saved event clock mismatches marker snapshot")
        if clock.startup_pending or clock.particle_dt_si <= 0 or not 0 <= clock.step_rest_time < clock.particle_dt_si:
            raise ValueError("invalid continuous particle history")
        if abs(clock.field_start_si-particle_si) > 1e-14 or abs(clock.field_end_si-tf*tn)>1e-14:
            raise ValueError("saved field interpolation interval mismatch")
        with h5py.File(feedback_path, "r") as f:
            if abs(_scalar(f,"t_now")-particle_si)>1e-14:
                raise ValueError("previous projection internal time mismatches particle clock")
            feedback_values = np.asarray(f["values"][()])
        fd = np.asarray(feedback_deltas)
        if feedback_values.shape != shape or fd.shape != shape or not np.isfinite(fd).all() or not np.isfinite(feedback_values).all():
            raise ValueError("full feedback values/deltas shape mismatch")
        feedback = ProjectionFeedback(feedback_values, fd.copy(), particle_si,
            history_provenance='explicit_saved_continuous_projection_history')
    else:
        # marker_spi_source :82-86 adjusts once using imported HDF5 tstep,
        # before the timestepper starts its segment plan (these need not agree).
        n = int(current_tstep*tn/nominal)
        if n < 1:
            raise ValueError("imported fluid step shorter than nominal particle step")
        clock = MarkerClock(particle_si, 0., current_tstep*tn/n,
                            tf*tn, tf*tn, startup_pending=True)
        feedback = ProjectionFeedback(np.zeros(shape), np.zeros(shape), particle_si,
            history_provenance='jorek_restart_zero_projection')
    shards, _ = _read_native_shard(Path(fluid_path), event_index, side="checkpoint")
    provenance = {"fluid": _source(fluid_path), "marker": None if marker_path is None else _source(marker_path),
                  "namelist": _source(namelist_path), "namelist_sha256": hashlib.sha256(Path(namelist_path).read_bytes()).hexdigest(),
                  "times": _source(times_path), "event_index": event_index,
                  "event_phase": "after_fluid_before_particle", "boundary": boundary,
                  "particle_metadata": marker_meta, "t_norm": tn, "central_density": density,
                  "central_mass": mass, "tstep_h5": current_tstep, "fluid_time_jorek": tf,
                  "fluid_deltas_origin": "native_hdf5_node_deltas",
                  "time_evol_scheme": params.get("time_evol_scheme", "Crank-Nicholson"),
                  "time_evol_scheme_origin": "namelist" if "time_evol_scheme" in params else "jorek_preset_parameters",
                  "imported_step_origin": "native_hdf5_tstep",
                  "tstep_particles_nominal_si": nominal, "parameters": params,
                  "feedback": None if feedback_path is None else _source(feedback_path),
                  "previous_fluid": None if previous_fluid_path is None else _source(previous_fluid_path)}
    from copy import deepcopy
    runtime = MarkerRuntime(cloud, feedback, deepcopy(clock), provenance)
    return NativeMarkerCheckpoint(runtime, values, deltas, previous, shards, tf, tn, density, provenance)


def reduce_checkpoint_marker(native: MarkerRuntime, sampler, *, max_live_markers: int = 10000):
    """Explicit dof0/bilinear adapter. No cloud compression or sufficiency claim.

    Full slots, P_imp, v, identities are retained. Refuse large replay; a future
    calibrated reduction must supply its own registered approximation.
    """
    from copy import deepcopy
    import torch
    if native.cloud.alive.sum() > max_live_markers:
        raise ValueError("reference live-marker limit exceeded; calibrated reduction required")
    out = deepcopy(native)
    if out.feedback.layout == "native_full":
        out.feedback.values = out.feedback.values[:,0,0,:].T.copy()
        out.feedback.deltas = out.feedback.deltas[:,0,0,:].T.copy()
        out.feedback.layout = "node_dof0"
    ids = np.flatnonzero(out.cloud.alive)
    if len(ids):
        loc = sampler.locate(torch.as_tensor(out.cloud.x[ids,:2]).unsqueeze(0))
        if not bool(loc.inside.all()):
            raise ValueError("native live marker outside reduced bilinear mesh")
        out.cloud.i_elm[ids] = loc.element[0].numpy()+1
        out.cloud.st[ids] = np.stack([loc.s[0].numpy(),loc.t[0].numpy()],axis=1)
    out.provenance["representation"] = "all_slots_2d_bilinear_dof0_lumped_projection"
    return out


def reference_rollout_from_checkpoint(checkpoint: NativeMarkerCheckpoint, *,
                                      continuous_remaining_plan_si=None, native_context=None,
                                      control_plan_si=None, control_origin="external_control"):
    """Create the full reference event driver; no field/particle truth replay.

    Both restart and continuation require explicit next-step controls. HDF5
    tstep is imported history, not a plan. Offline record schedules must declare
    recorded_schedule_replay. Model checkpoints persist the plan and cursor.
    """
    from copy import deepcopy
    from src.models.marker_state import ReferenceRollout
    p = checkpoint.provenance
    restart = checkpoint.marker.clock.startup_pending
    if control_origin not in ('external_control','recorded_schedule_replay'):
        raise ValueError('unsupported fluid control origin')
    if control_plan_si is not None and continuous_remaining_plan_si is not None:
        raise ValueError('provide exactly one fluid control plan')
    selected=control_plan_si if control_plan_si is not None else continuous_remaining_plan_si
    if selected is None:
        raise ValueError('explicit next-step fluid control required; imported tstep is not a schedule')
    plan=tuple(selected)
    if not plan or not np.isfinite(plan).all() or min(plan)<=0:
        raise ValueError('fluid control plan must contain positive finite steps')
    if checkpoint.fluid_deltas is None:
        raise ValueError("missing fluid node deltas in checkpoint")
    rollout = ReferenceRollout(deepcopy(checkpoint.marker),checkpoint.fluid_values.copy(),
        None if checkpoint.previous_fluid_values is None else checkpoint.previous_fluid_values.copy(),
        deepcopy(checkpoint.shards), checkpoint.fluid_time_jorek*checkpoint.t_norm,
        p['event_index'],plan, fluid_dt_previous_si=(None if restart else
            p['tstep_h5']*checkpoint.t_norm),
        fluid_deltas=checkpoint.fluid_deltas.copy(),
        time_evol_scheme=p['time_evol_scheme'])
    rollout.marker.provenance["fluid_control_origin"]=control_origin
    rollout.validate_fluid_history()
    return rollout if native_context is None else rollout.with_native_context(native_context)
