"""Explicit marker reference state and small-component JOREK stepping.

SI particle time is distinct from the event clock. No file reads occur in
advance_window. Field/atomic/geometry/projection callbacks are dependencies,
not recorded truth. Full filtered Hermite FEM and stochastic birth sampling
are deliberately delegated; the SourceModule adapter is a registered 2D,
bilinear, lumped-projection approximation, not an exact JOREK solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import numpy as np
import resource
import time


@dataclass(frozen=True)
class MarkerExecutionBudget:
    """Opt-in all-slot resource policy; never changes particle physics.

    Callbacks must be particle-local in the supplied ids, with additive
    deposition. Chunk order is ascending slot order within each substep.
    RSS includes the whole process. Checks surround callbacks; an external
    process deadline is needed to interrupt a single slow native callback.
    """
    max_slots: int
    max_live: int
    chunk_size: int
    max_particle_steps: int
    max_seconds: float
    max_rss_bytes: int

    def __post_init__(self):
        for name in ('max_slots','max_live','chunk_size','max_particle_steps','max_rss_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError('positive explicit marker execution limits required')
        if not np.isfinite(self.max_seconds) or self.max_seconds<=0:
            raise ValueError('positive explicit marker execution time required')
        if self.max_live>self.max_slots or self.chunk_size>self.max_live:
            raise ValueError('marker chunk/live/slot limits are inconsistent')

    def check(self, started):
        if time.monotonic()-started>self.max_seconds:
            raise RuntimeError('marker execution time budget exceeded; candidate not committed')
        # Linux resource units are KiB; retained high-water includes imported Torch.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.max_rss_bytes:
            raise RuntimeError('marker execution RSS budget exceeded; candidate not committed')


@dataclass
class MarkerCloud:
    x: np.ndarray                 # slots x 3, R/Z/phi [m,m,rad]
    st: np.ndarray                # slots x 2 (native reference coordinates)
    i_elm: np.ndarray             # native 1-based, <=0 lost/free
    weight: np.ndarray            # atoms
    P_imp: np.ndarray             # slots x (Z+1), full probability vector
    V_prev: np.ndarray            # slots x 3 [m/s], HDF5 name v
    i_life: np.ndarray            # identity = (group, slot, lifetime)
    t_birth: np.ndarray           # native field, seconds; producer may leave zero
    group: str = "001"

    @classmethod
    def empty(cls, z: int = 10, capacity: int = 0):
        p = np.zeros((capacity, z + 1)); p[:, 0] = 1
        return cls(np.zeros((capacity, 3)), np.zeros((capacity, 2)),
                   np.zeros(capacity, dtype=np.int32), np.zeros(capacity), p,
                   np.zeros((capacity, 3)), np.zeros(capacity, dtype=np.int32),
                   np.zeros(capacity, dtype=np.float32))

    @property
    def alive(self):
        return self.i_elm > 0

    def validate(self):
        n = len(self.weight)
        for name, shape in {"x": (n, 3), "st": (n, 2), "V_prev": (n, 3),
                            "i_elm": (n,), "i_life": (n,), "t_birth": (n,)}.items():
            a = getattr(self, name)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"invalid marker {name}: expected {shape}")
        if self.P_imp.ndim != 2 or self.P_imp.shape[0] != n:
            raise ValueError("invalid full P_imp shape")
        p = self.P_imp[self.alive]
        if (not np.isfinite(self.weight).all() or (self.weight[self.alive] < 0).any()
                or not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p.sum(1), 1., rtol=1e-6, atol=1e-8)):
            raise ValueError("invalid live marker weights/P_imp normalization")

    def insert(self, born: MarkerCloud, *, grow: bool = False):
        """Reuse only i_elm<=0 slots; increment lifetime; never overwrite live.

        grow=True is the reduced model's explicit dynamic-capacity difference.
        JOREK instead fails if its fixed pool cannot accommodate the births.
        Native t_birth is preserved (initialise_particles_marker does not set it).
        """
        born.validate()
        if born.group != self.group or born.P_imp.shape[1] != self.P_imp.shape[1]:
            raise ValueError("birth group/charge species mismatch")
        ids = np.flatnonzero(born.alive)
        free = np.flatnonzero(~self.alive)
        shortage = len(ids) - len(free)
        if shortage > 0:
            if not grow:
                raise ValueError("marker slot capacity exhausted")
            extra = MarkerCloud.empty(self.P_imp.shape[1] - 1, shortage)
            for name in self.array_names():
                setattr(self, name, np.concatenate([getattr(self, name), getattr(extra, name)]))
            free = np.flatnonzero(~self.alive)
        target = free[:len(ids)]
        self.i_life[target] += 1
        for name in ("x", "st", "i_elm", "weight", "P_imp", "V_prev"):
            getattr(self, name)[target] = getattr(born, name)[ids]
        return target

    @staticmethod
    def array_names():
        return ("x", "st", "i_elm", "weight", "P_imp", "V_prev", "i_life", "t_birth")


@dataclass
class ProjectionFeedback:
    values: np.ndarray             # full native degrees, or explicit reduced view
    deltas: np.ndarray
    time_si: float
    layout: str = "native_full"
    history_provenance: str = "runtime_projection_increment"
    gather_time_si: float | None = None


@dataclass
class MarkerClock:
    sim_time_si: float
    step_rest_time: float
    particle_dt_si: float          # fixed adjusted particle substep, NOT fluid tstep
    field_start_si: float
    field_end_si: float
    window_index: int = 0
    startup_pending: bool = False

    @property
    def particle_time_si(self):
        return self.sim_time_si - self.step_rest_time


@dataclass
class MarkerRuntime:
    cloud: MarkerCloud
    feedback: ProjectionFeedback
    clock: MarkerClock
    provenance: dict = field(default_factory=dict)
    transport_history: object | None = None  # explicit opt-in numerical state, not native HDF5

    @classmethod
    def empty(cls, n_nodes: int, *, time_si: float, particle_dt_si: float,
              confirmed_no_live_markers: bool):
        if not confirmed_no_live_markers:
            raise ValueError("empty cloud requires confirmed absence of live markers")
        return cls(MarkerCloud.empty(),
                   ProjectionFeedback(np.zeros((n_nodes, 5)), np.zeros((n_nodes, 5)),
                                      time_si, "node_dof0"),
                   MarkerClock(time_si, 0., particle_dt_si, time_si, time_si),
                   {"initialization": "confirmed_empty", "representation": "reference_slots"})

    def jorek_restart(self):
        """Particle read overwrites event time; new projection and residual time=0.

        Startup event MUST be consumed by execute_startup before advancing.
        Model load and ordinary window advance never call this operation.
        """
        if getattr(self,'transport_history',None) is not None:
            raise ValueError('local transport history cannot be silently converted to a native JOREK restart')
        out = deepcopy(self)
        out.feedback.values.fill(0); out.feedback.deltas.fill(0)
        out.feedback.time_si = out.clock.sim_time_si
        out.feedback.history_provenance = 'jorek_restart_zero_projection'
        out.feedback.gather_time_si = None
        out.clock.step_rest_time = 0.
        out.clock.startup_pending = True
        out.provenance["boundary"] = "jorek_program_restart"
        return out


def execute_startup(state: MarkerRuntime, fluid_step):
    """Project zero RHS -> fluid solve/shard update, before first particle loop.

    fluid_step receives zero feedback and particle event time and must return
    the new field interpolation endpoints in SI seconds. This explicit hook
    keeps the fluid solver outside this component (no five-to-nine surrogate).
    """
    if not state.clock.startup_pending:
        raise ValueError("no JOREK startup event pending")
    out = deepcopy(state)
    start, end = fluid_step(out.feedback, out.clock.sim_time_si)
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("startup fluid step must return a positive field interval")
    out.clock.field_start_si, out.clock.field_end_si = float(start), float(end)
    out.clock.startup_pending = False
    return out


def charge_step(p, scd, acd, ne_si, dt):
    """marker_spi_source: explicit neighbor transitions, clamp, renormalize.

    Rates supplied in m^3/s; scd[:,z] is z->z+1, acd[:,z] is z+1->z.
    This is NOT relaxation to coronal equilibrium.
    """
    ion = p[:, :-1] * scd * ne_si[:, None] * dt
    rec = p[:, 1:] * acd * ne_si[:, None] * dt
    q = p.copy()
    q[:, :-1] += rec - ion
    q[:, 1:] += ion - rec
    q = np.maximum(q, 0.)
    return q / q.sum(1, keepdims=True)


def push_velocity(x, previous, velocity, dt):
    """Literal direction-based cylindrical JOREK pusher (:56-78).

    Zero speed has undefined divisions in upstream Fortran; fail explicitly
    rather than silently inventing a different integrator.
    """
    speed = np.linalg.norm(velocity, axis=1)
    old_speed = np.linalg.norm(previous, axis=1)
    if (speed == 0).any() or (old_speed == 0).any() or (x[:, 0] <= 0).any():
        raise ValueError("JOREK velocity pusher requires nonzero current/previous speed and R>0")
    delta = (1.5 * velocity / speed[:, None] - .5 * previous / old_speed[:, None]) * speed[:, None] * dt
    result = x.copy()
    result[:, :2] += delta[:, :2]
    result[:, 2] += delta[:, 2] / x[:, 0]
    return result


def particle_window_schedule(clock: MarkerClock, target_si: float):
    """Event time to actual particle gather time, retaining residual history."""
    start = clock.particle_time_si
    quotient = (target_si-start)/clock.particle_dt_si
    nearest = round(quotient)
    n = int(nearest if abs(quotient-nearest) < 1e-10 else quotient)
    return start, n, n*clock.particle_dt_si


def advance_window(state: MarkerRuntime, target_si: float, *, sample, locate,
                   deposit, project, potentials_ev, t_norm: float,
                   central_density: float, births: MarkerCloud | None = None,
                   grow: bool = False, max_particle_steps: int = 10000, pusher=push_velocity,
                   elementary_charge=1.602176634e-19, execution_budget: MarkerExecutionBudget | None = None,
                   locate_with_hints=None, transport_policy=None, boundary_exit=None):
    """Small reference component: birth -> substeps -> gather -> projection.

    sample(time, cloud, live_indices, previous_feedback) supplies velocity
    (including V_ext), ne_si, scd/acd [m^3/s], radiation_w_per_atom [N,Z+1].
    deposit uses OLD position/element before push; project solves its registered
    projection problem. Previous feedback is held fixed throughout the window.
    """
    started=time.monotonic()
    from .marker_transport import LocalSubcycling, transport_batches, push_variable_direction
    history=getattr(state,'transport_history',None)
    if transport_policy is None:
        if history is not None:raise ValueError('transport history requires its configured local subcycling mode')
    else:
        if not isinstance(transport_policy,LocalSubcycling) or history is None:
            raise ValueError('local subcycling requires explicit initialized transport history')
        from .marker_transport import TransportHistory
        if not isinstance(history,TransportHistory):raise ValueError('invalid marker transport history type')
        if pusher is not push_velocity:raise ValueError('local subcycling requires the registered direction pusher')
        history.validate(state.cloud,transport_policy)
    if (boundary_exit is not None) != bool(transport_policy is not None and transport_policy.boundary_exit):
        raise ValueError('configured verified boundary exit provider required')
    if execution_budget is not None:
        if not isinstance(execution_budget,MarkerExecutionBudget):
            raise ValueError('explicit MarkerExecutionBudget required')
        execution_budget.check(started)
        slots=len(state.cloud.weight)
        live=int(np.count_nonzero(state.cloud.alive))
        born_live=0 if births is None else int(np.count_nonzero(births.alive))
        required_slots=max(slots,live+born_live) if grow else slots
        if required_slots>execution_budget.max_slots or live+born_live>execution_budget.max_live:
            raise ValueError('marker execution slot/live budget exceeded')
        # One candidate copy plus per-slot diagnostics, slot ids and conservative
        # charge/sample scratch; callbacks can add their own bounded allocations.
        payload=sum(getattr(state.cloud,name).nbytes for name in MarkerCloud.array_names())
        if history is not None:
            payload+=sum(getattr(history,name).nbytes for name in ('previous_dt_si','level','lifetime'))
        required_bytes=2*payload+slots*48+execution_budget.chunk_size*1024
        if required_bytes>execution_budget.max_rss_bytes:
            raise ValueError('marker execution memory lower bound exceeds budget')
    c = state.clock
    if not np.isfinite([c.sim_time_si,c.step_rest_time,c.particle_dt_si,
                        c.field_start_si,c.field_end_si,t_norm,central_density,elementary_charge]).all() or min(t_norm,central_density,elementary_charge)<=0:
        raise ValueError("invalid marker clock/normalization")
    if c.startup_pending:
        raise ValueError("execute JOREK startup zero-projection/fluid event first")
    if not np.isfinite(target_si) or target_si <= c.sim_time_si:
        raise ValueError("particle target must advance event time")
    if c.particle_dt_si <= 0 or not 0 <= c.step_rest_time < c.particle_dt_si:
        raise ValueError("invalid particle step history")
    # Snap exact multiples only within 1e-10 substeps to avoid cancellation
    # of SI event timestamps; otherwise retain upstream Fortran INT behavior.
    start, n, gather_time = particle_window_schedule(c,target_si)
    if n < 1 or n > min(max_particle_steps, max_particle_steps if execution_budget is None else execution_budget.max_particle_steps):
        raise ValueError("unsupported particle substep count; no implicit million-particle replay")
    if start < c.field_start_si - c.step_rest_time - 1e-14 or start + (n-1)*c.particle_dt_si > c.field_end_si + 1e-14:
        raise ValueError("particle sampling times outside available field history")
    out = deepcopy(state)
    if births is not None:
        born_ids=out.cloud.insert(births, grow=grow)
        if transport_policy is not None:
            h=out.transport_history;extra=len(out.cloud.weight)-len(h.level)
            if extra:
                h.previous_dt_si=np.r_[h.previous_dt_si,np.full(extra,c.particle_dt_si)]
                h.level=np.r_[h.level,np.zeros(extra,np.uint8)]
                h.lifetime=np.r_[h.lifetime,out.cloud.i_life[-extra:]]
            h.previous_dt_si[born_ids]=c.particle_dt_si;h.level[born_ids]=0
            h.lifetime[born_ids]=out.cloud.i_life[born_ids]
    out.cloud.validate()
    if execution_budget is None and out.cloud.alive.sum() > 10000:
        raise ValueError("reference live-marker limit exceeded; calibrated reduction required")
    rhs = None
    energy = np.zeros(len(out.cloud.weight)); rad = energy.copy()
    moments = np.zeros((len(energy), 3))
    chunks=processed=refined=0
    boundary_exits=[]
    def bounded_sample(t,cloud,batch,feedback):
        if execution_budget is not None:execution_budget.check(started)
        if transport_policy is not None and t>c.field_end_si+1e-14:
            raise ValueError('local particle sampling time outside available field history')
        return sample(t,cloud,batch,feedback)
    def fixed_batches(ids,t):
        yield ids,c.particle_dt_si,bounded_sample(t,out.cloud,ids,state.feedback)
    for k in range(n):
        ids = np.flatnonzero(out.cloud.alive)
        if not len(ids):
            break
        width=len(ids) if execution_budget is None else execution_budget.chunk_size
        for offset in range(0,len(ids),width):
            selected=ids[offset:offset+width]
            batches=(fixed_batches(selected,start+k*c.particle_dt_si) if transport_policy is None else
                transport_batches(out.cloud,selected,time_si=start+k*c.particle_dt_si,
                    base_dt=c.particle_dt_si,feedback=state.feedback,sample=bounded_sample,
                    history=out.transport_history,policy=transport_policy,boundary_exit=boundary_exit))
            for batch,dt,a in batches:
                if execution_budget is not None: execution_budget.check(started)
                if transport_policy is not None and dt<c.particle_dt_si:
                    refined+=len(batch)
                    if refined>transport_policy.max_refined_particle_steps:
                        raise ValueError('local transport refined work budget exceeded; candidate not committed')
                p = out.cloud.P_imp[batch].copy()
                q = charge_step(p, a["scd"], a["acd"], a["ne_si"], dt)
                z = np.arange(p.shape[1])
                w = out.cloud.weight[batch]
                e = (q-p) @ np.r_[0., np.cumsum(potentials_ev)] * w * elementary_charge
                r = (p * a["radiation_w_per_atom"]).sum(1) * w * dt
                m2, m1, number = (p @ (z*z))*w*dt, (p @ z)*w*dt, w*dt
                load = np.stack([e*t_norm/(1./(4e-7*np.pi*(5./3.-1.))),
                                 r*t_norm/(1./(4e-7*np.pi*(5./3.-1.))),
                                 m2/(central_density*1e20), m1/(central_density*1e20),
                                 number/(central_density*1e20)], axis=1)
                contribution = deposit(out.cloud, batch, load)
                rhs = contribution if rhs is None else rhs + contribution
                energy[batch] += e; rad[batch] += r
                moments[batch] += np.stack([m2, m1, number], axis=1)
                out.cloud.P_imp[batch] = q
                velocity = a["velocity"]
                if transport_policy is None:
                    out.cloud.x[batch] = pusher(out.cloud.x[batch], out.cloud.V_prev[batch], velocity, dt)
                else:
                    out.cloud.x[batch] = push_variable_direction(out.cloud.x[batch],out.cloud.V_prev[batch],
                        velocity,dt,out.transport_history.previous_dt_si[batch])
                    out.transport_history.previous_dt_si[batch]=dt
                out.cloud.V_prev[batch] = velocity
                if execution_budget is not None: execution_budget.check(started)
                terminal=a.get('_boundary_exit')
                if terminal is not None:
                    event=terminal['event'];slot=int(batch[0])
                    out.cloud.x[slot]=event.x
                    elm,st=np.array([event.element]),event.st[None]
                    boundary_exits.append(dict(slot=slot,lifetime=int(out.cloud.i_life[slot]),
                        element=event.element,edge=event.edge,residual_m=event.residual_m,
                        time_si=terminal['time_si'],proposed_dt_si=terminal['proposed_dt_si'],
                        accepted_dt_si=dt,required_level=terminal['required_level'],indicator=terminal['indicator']))
                elif locate_with_hints is None:
                    elm, st = locate(out.cloud.x[batch])
                else:
                    elm, st = locate_with_hints(out.cloud.x[batch],
                        element_hints=out.cloud.i_elm[batch],local_hints=out.cloud.st[batch])
                if boundary_exit is not None and terminal is None and (np.asarray(elm)<=0).any():
                    raise ValueError('unverified marker loss; locator failure is not a certified boundary exit')
                out.cloud.i_elm[batch], out.cloud.st[batch] = elm, st
                chunks+=1;processed+=len(batch)
                if execution_budget is not None: execution_budget.check(started)
    if rhs is None:
        values = np.zeros_like(state.feedback.values)
    else:
        values = project(rhs / (n*c.particle_dt_si))
    if execution_budget is not None: execution_budget.check(started)
    if values.shape != state.feedback.values.shape or not np.isfinite(values).all():
        raise ValueError("projection must preserve registered feedback layout and be finite")
    out.feedback = ProjectionFeedback(values, values-state.feedback.values, target_si, state.feedback.layout,
        'runtime_new_minus_persisted_previous', n*c.particle_dt_si)
    out.clock.sim_time_si = target_si
    out.clock.step_rest_time = max(0., target_si - (start+n*c.particle_dt_si))
    out.clock.window_index += 1
    if transport_policy is not None:out.transport_history.validate(out.cloud,transport_policy)
    diagnostic={"energy": energy, "rad": rad, "moments": moments,
                 "gather_time_seconds": n*c.particle_dt_si, "n_steps": n,
                 "execution":dict(chunks=chunks,processed_particle_steps=processed,
                     refined_particle_steps=refined,
                     slots=len(out.cloud.weight),elapsed_seconds=time.monotonic()-started,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    if boundary_exit is not None:
        diagnostic['boundary_exits']=boundary_exits
        diagnostic['execution']['boundary_exits']=len(boundary_exits)
    return out,diagnostic


def _module_signature(module):
    import hashlib
    import pickle
    names = ("n_substeps", "field_path", "shard_layout", "use_memory",
             "use_correction", "use_marker_transport", "marker_config", "tables", "checkpoint_configuration")
    signature = {k: getattr(module,k) for k in names if hasattr(module,k)}
    if hasattr(module,"sampler"):
        signature["mesh"] = (module.sampler.pos.cpu().numpy(), module.sampler.vertex.cpu().numpy())
    return hashlib.sha256(pickle.dumps(signature,protocol=5)).hexdigest()


def save_model_rollout(path: str | Path, augmented_state, module):
    """Save trusted local inference state, weights and physics/config signature.

    No optimizer state is implied: this task does not implement training.
    Immutable mesh/tables/config must match on restore; mismatch fails instead
    of continuing a physically different model. No JOREK restart is attached.
    """
    import os
    import random
    import tempfile
    import torch
    if isinstance(augmented_state, ReferenceRollout):
        augmented_state.validate_fluid_history()
    marker=getattr(augmented_state,'marker',getattr(augmented_state,'marker_state',None))
    if isinstance(marker,MarkerRuntime):
        from .marker_transport import validate_transport_runtime
        validate_transport_runtime(marker)
    from .native_torch_rollout import TorchFieldState
    field_only = isinstance(augmented_state,TorchFieldState)
    if field_only: augmented_state.validate_for_model(module)
    context = getattr(augmented_state,'native_context',None)
    if hasattr(module,'checkpoint_configuration'):
        from .native_state import configuration_digest
        if not field_only and (context is None or configuration_digest(context.configuration['model']) != configuration_digest(module.checkpoint_configuration)):
            raise ValueError('model/rollout native configuration mismatch')
    record = {"schema": "marker-rollout-v2", "state": augmented_state,
              "module": module.state_dict(), "signature": _module_signature(module),
              "training": module.training,
              "modes": {name:m.training for name,m in module.named_modules()},
              "architecture": [(name,type(m).__module__,type(m).__qualname__,m.extra_repr())
                               for name,m in module.named_modules()],
              "rng": {'python':random.getstate(),'numpy':np.random.get_state(),
                      'torch':torch.get_rng_state(),
                      'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
              "native_fingerprint": None if context is None else context.fingerprint()}
    # Same-directory replacement: failed validation/serialization never replaces
    # an existing committed checkpoint. Only trusted local files are supported.
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(record,handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_model_rollout(path: str | Path, module, *, native_context=None):
    """Read only trusted, locally generated model rollout checkpoints."""
    import torch
    import random
    record = torch.load(path, weights_only=False, map_location="cpu")
    if record.get("schema") not in {"marker-rollout-v1","marker-rollout-v2"}:
        raise ValueError("unsupported model rollout checkpoint")
    if record['schema'] == 'marker-rollout-v2' and not {
            'state','module','signature','training','modes','architecture','rng','native_fingerprint'} <= record.keys():
        raise ValueError('incomplete model checkpoint state/configuration')
    if record.get("signature") != _module_signature(module):
        raise ValueError("model checkpoint physics/configuration mismatch")
    if isinstance(record["state"], ReferenceRollout):
        record["state"].validate_fluid_history()
    marker=getattr(record['state'],'marker',getattr(record['state'],'marker_state',None))
    if isinstance(marker,MarkerRuntime):
        from .marker_transport import validate_transport_runtime
        validate_transport_runtime(marker)
    from .native_torch_rollout import TorchFieldState
    if isinstance(record['state'],TorchFieldState): record['state'].validate_for_model(module)
    saved_context = getattr(record['state'],'native_context',None)
    if saved_context is not None:
        fingerprint = saved_context.fingerprint()
        if (native_context is None or fingerprint != native_context.fingerprint()
                or fingerprint != record.get('native_fingerprint')):
            raise ValueError('native checkpoint configuration mismatch or missing expected configuration')
    elif native_context is not None:
        raise ValueError('legacy checkpoint lacks required native configuration')
    modes = record.get('modes')
    architecture = [(name,type(m).__module__,type(m).__qualname__,m.extra_repr())
                    for name,m in module.named_modules()]
    if 'architecture' in record and record['architecture'] != architecture:
        raise ValueError('model architecture/configuration mismatch')
    if modes is not None and set(modes) != set(dict(module.named_modules())):
        raise ValueError('model configuration/module structure mismatch')
    if modes is not None and any(not isinstance(v,bool) for v in modes.values()):
        raise ValueError('invalid saved module modes')
    # load_state_dict can copy some tensors before discovering an error. Probe
    # on a private model and only then update the caller, with rollback support.
    probe = deepcopy(module)
    target_tensors = module.state_dict()
    for name,tensor in record['module'].items():
        if name in target_tensors and hasattr(tensor,'dtype') and tensor.dtype != target_tensors[name].dtype:
            raise ValueError('model checkpoint tensor dtype mismatch')
    probe.load_state_dict(record['module'])
    if hasattr(module,'checkpoint_configuration'):
        from .native_state import configuration_digest
        if configuration_digest(probe.checkpoint_configuration)!=configuration_digest(module.checkpoint_configuration):
            raise ValueError('checkpoint tensor/configuration mismatch in immutable scales or geometry')
    rng = record.get('rng')
    if rng is not None:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator().set_state(rng['torch'])
        if rng['cuda'] is not None and (not torch.cuda.is_available()
                or len(rng['cuda']) != torch.cuda.device_count()):
            raise ValueError('checkpoint CUDA RNG configuration mismatch')
    backup = deepcopy(module.state_dict())
    backup_modes = {name:m.training for name,m in module.named_modules()}
    backup_rng = {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                  'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    try:
        module.load_state_dict(record['module'])
        if modes is None:
            module.train(record['training'])
        else:
            for name,m in module.named_modules(): m.training = modes[name]
        if rng is not None:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None: torch.cuda.set_rng_state_all(rng['cuda'])
    except Exception:
        module.load_state_dict(backup)
        for name,m in module.named_modules(): m.training = backup_modes[name]
        random.setstate(backup_rng['python'])
        np.random.set_state(backup_rng['numpy'])
        torch.set_rng_state(backup_rng['torch'])
        if backup_rng['cuda'] is not None: torch.cuda.set_rng_state_all(backup_rng['cuda'])
        raise
    return record["state"]


@dataclass
class ReferenceRollout:
    """Event state for reference inference, independent of the reduced B1 path.

    Full fields/shards are opaque to this driver: supplied physics callbacks
    implement them. No five-to-nine mapping or formal training is implied.
    Plan cursor and both field endpoints survive a model checkpoint.
    """
    marker: MarkerRuntime
    fields_current: np.ndarray
    fields_previous: np.ndarray | None
    shards: object
    fluid_time_si: float
    fluid_index: int
    step_plan_si: tuple[float, ...]
    plan_cursor: int = 0
    fluid_dt_previous_si: float | None = None
    h_residual: object = None
    fluid_deltas: np.ndarray | None = None  # node%deltas, NOT marker feedback deltas
    time_evol_scheme: str = "Crank-Nicholson"  # JOREK preset_parameters default
    native_context: object = None  # explicit full-axis contract; legacy adapters remain available
    native_schema_versions: tuple[str,str] = ('native-field-v1','native-feedback-v1')
    field_memory: np.ndarray | None = None  # explicit field core memory; NOT h_residual

    def with_native_context(self, context):
        """Bind units/modes/configuration without changing the event driver."""
        from .native_state import NativeStateContext
        if not isinstance(context, NativeStateContext):
            raise ValueError('explicit NativeStateContext required')
        out = deepcopy(self)
        out.native_context = deepcopy(context)
        out.validate_fluid_history()
        return out

    def native_field_state(self):
        from .native_state import NativeFieldState
        if self.native_context is None:
            raise ValueError('missing native context')
        c = self.native_context
        return NativeFieldState(self.fields_current.copy(), None if self.fluid_deltas is None else self.fluid_deltas.copy(),
                                c.units, self.fluid_time_si, c.modes, self.fluid_index,
                                schema_version=self.native_schema_versions[0])

    def native_feedback_state(self):
        from .native_state import NativeFeedbackState
        if self.native_context is None:
            raise ValueError('missing native context')
        c, f = self.native_context, self.marker.feedback
        return NativeFeedbackState(f.values.copy(), None if f.deltas is None else f.deltas.copy(),
            c.units, f.time_si, c.modes, f.history_provenance, f.gather_time_si,
            schema_version=self.native_schema_versions[1])

    def validate_fluid_history(self):
        """Never infer nodal history from two rounded field endpoints."""
        from .marker_transport import validate_transport_runtime
        expected_transport=None
        if getattr(self,'native_context',None) is not None:
            expected_transport=self.native_context.configuration['model'].get('source',{}).get('transport_policy')
        validate_transport_runtime(self.marker,expected_configuration=expected_transport)
        if (getattr(self,'native_context',None) is not None and expected_transport is None
            and getattr(self.marker,'transport_history',None) is not None):
            raise ValueError('local transport history is absent from native model configuration')
        feedback = self.marker.feedback
        if feedback.layout == 'native_full':
            u = self.fields_current
            if (not isinstance(u,np.ndarray) or u.ndim != 4 or u.shape[:2] != (9,4)
                    or u.shape[2]<1 or u.shape[3]<1):
                raise ValueError('native_full requires all nine fields and four degrees')
            shape=(5,*u.shape[1:])
            if (not isinstance(feedback.values,np.ndarray) or feedback.values.shape != shape
                    or not np.isfinite(feedback.values).all()):
                raise ValueError('invalid full native projection values')
            if (not isinstance(feedback.deltas,np.ndarray) or feedback.deltas.shape != shape
                    or feedback.deltas.dtype != feedback.values.dtype or not np.isfinite(feedback.deltas).all()):
                raise ValueError('missing/invalid full native projection history')
        if getattr(self, 'native_context', None) is not None:
            if self.native_schema_versions != ('native-field-v1','native-feedback-v1'):
                raise ValueError('unsupported native state version')
            self.native_context.validate()
            temporal=self.native_context.configuration['model'].get('temporal_core')
            memory=getattr(self,'field_memory',None)
            if temporal is not None:
                if (not isinstance(memory,np.ndarray) or memory.shape!=(self.fields_current.shape[-1],temporal['hidden'])
                    or memory.dtype!=self.fields_current.dtype or not np.isfinite(memory).all()):
                    raise ValueError('missing/invalid explicit field temporal memory')
            elif memory is not None:
                raise ValueError('field memory supplied to memoryless model')
            scheme = self.native_context.configuration['numerics'].get('scheme')
            if scheme is not None and scheme != self.time_evol_scheme:
                raise ValueError('native time scheme/configuration mismatch')
            self.native_field_state().validate()
            f = self.native_feedback_state()
            if f.values.shape[1:] != self.fields_current.shape[1:] or self.marker.feedback.layout != 'native_full':
                raise ValueError('native feedback axes/layout mismatch')
            c = self.marker.clock
            if c.startup_pending and (f.values.any() or f.raw_deltas.any() or c.step_rest_time != 0):
                raise ValueError('JOREK startup requires zero projection/history and zero residual time')
            self.marker.cloud.validate()
            if (not np.isfinite([c.sim_time_si,c.step_rest_time,c.particle_dt_si,c.field_start_si,c.field_end_si]).all()
                    or c.particle_dt_si <= 0 or not 0 <= c.step_rest_time < c.particle_dt_si):
                raise ValueError('invalid native particle clock')
            tol = max(self.native_context.units.t_norm_si*1e-7,1e-14)
            if abs(f.time_si-c.sim_time_si)>tol or abs(c.field_end_si-self.fluid_time_si)>tol:
                raise ValueError('native feedback/field event time mismatch')
            if self.fields_previous is None:
                if not c.startup_pending:
                    raise ValueError('missing native previous field endpoint')
            elif (self.fields_previous.shape != self.fields_current.shape
                  or self.fields_previous.dtype != self.fields_current.dtype
                  or not np.isfinite(self.fields_previous).all()):
                raise ValueError('invalid native previous field endpoint')
            if (not c.startup_pending and (c.field_start_si>c.sim_time_si+tol
                    or c.sim_time_si>self.fluid_time_si+tol)):
                raise ValueError('native particle/field time ordering mismatch')
            if (not isinstance(self.plan_cursor,int) or not 0 <= self.plan_cursor <= len(self.step_plan_si)
                    or not self.step_plan_si or not np.isfinite(self.step_plan_si).all()
                    or min(self.step_plan_si)<=0):
                raise ValueError('invalid native step plan/cursor')
        history = getattr(self, "fluid_deltas", None)
        if history is None:
            raise ValueError("missing fluid node deltas; cannot losslessly infer them from field endpoints")
        if history.shape != self.fields_current.shape or not np.isfinite(history).all():
            raise ValueError("fluid node deltas must be finite and match the full field layout")
        if not np.isfinite(self.fields_current).all():
            raise ValueError("nonfinite current fluid fields")
        if not self.marker.clock.startup_pending:
            previous_dt = self.fluid_dt_previous_si
            if previous_dt is None or not np.isfinite(previous_dt) or previous_dt <= 0:
                raise ValueError("missing/invalid previous fluid timestep for continuous solve")


@dataclass(frozen=True)
class FluidTimeCoefficients:
    """Model502 assembly coefficients; nodal increments themselves stay raw.

    Interpolated fluid AND aux deltas are scaled by delta_scale at assembly.
    This value does not overwrite either persisted delta array.
    """
    theta: float
    zeta: float
    delta_scale: float


def fluid_time_coefficients(scheme: str, dt: float, previous_dt: float):
    from math import isfinite
    if not all(isfinite(x) and x > 0 for x in (dt, previous_dt)):
        raise ValueError("fluid timesteps must be positive and finite")
    schemes = {"implicit Euler": (1., 0.), "Crank-Nicholson": (.5, 0.),
               "Gears": (1., .5)}
    if scheme not in schemes:
        raise ValueError(f"unsupported JOREK time evolution scheme {scheme!r}")
    theta, base_zeta = schemes[scheme]
    return FluidTimeCoefficients(theta, base_zeta*2*dt/(dt+previous_dt), dt/previous_dt)


@dataclass(frozen=True)
class FluidSolveResult:
    """Candidate committed node values and exact post-update node%deltas.

    The backend owns update_values' nodal mapping, constraints/axis/frozen
    harmonics. Do not substitute values-old_values for fluid_deltas: rounded
    addition and preserved constrained/frozen history can make them differ.
    The driver commits both after the shard update. update_deltas subsequently
    packs this nodal history back into the solver vector in JOREK.
    """
    values: np.ndarray
    fluid_deltas: np.ndarray
    field_memory: np.ndarray | None = None


def reference_fluid_event(state: ReferenceRollout, *, solve_fluid, update_shards):
    """Source installed -> solve -> shard update in OLD field -> commit field.

    JOREK do_jorek_timestep :327-328 resets initial fluid time to sim%time;
    :392 solves, :418 updates shards, :424 commits field and advances time.
    solve_fluid(old_fields, shards, feedback, dt, dt_prev, *, fluid_deltas,
                time_coefficients) returns FluidSolveResult with post-update
    native nodal arrays. Field history and marker projection deltas are separate.
    update_shards(shards, OLD_fields, time_si, dt) returns updated shard state.
    """
    state.validate_fluid_history()
    if state.plan_cursor >= len(state.step_plan_si):
        raise ValueError("fluid step plan exhausted")
    out = deepcopy(state)
    dt = out.step_plan_si[out.plan_cursor]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("fluid step plan must be positive")
    startup = out.marker.clock.startup_pending
    time = out.marker.clock.sim_time_si if startup else out.fluid_time_si
    previous_dt = dt if startup else out.fluid_dt_previous_si
    coefficients = fluid_time_coefficients(out.time_evol_scheme, dt, previous_dt)
    old_fields = out.fields_current.copy()
    extra={}
    if getattr(solve_fluid,'supports_field_event',False):
        from .native_source_features import FieldEventContext
        extra['field_event']=FieldEventContext(out.fluid_time_si,time,
            'registered_restart' if startup else 'ordinary','ReferenceRollout_committed_clock',
            out.marker.feedback.gather_time_si)
    if getattr(solve_fluid,'supports_field_memory',False):
        extra['field_memory']=None if out.field_memory is None else out.field_memory.copy()
    elif getattr(out,'field_memory',None) is not None:
        raise ValueError('solver cannot consume explicit field memory')
    solved = solve_fluid(old_fields.copy(), out.shards, deepcopy(out.marker.feedback), dt, previous_dt,
                         fluid_deltas=out.fluid_deltas.copy(), time_coefficients=coefficients,**extra)
    if not isinstance(solved, FluidSolveResult):
        raise ValueError("fluid solver must return FluidSolveResult with explicit node deltas")
    for name, array in (("values", solved.values), ("fluid_deltas", solved.fluid_deltas)):
        if not isinstance(array, np.ndarray) or array.shape != old_fields.shape or not np.isfinite(array).all():
            raise ValueError(f"fluid solve result {name} must be finite and match the full field layout")
    out.shards = update_shards(out.shards, old_fields, time, dt)
    out.fields_previous = out.fields_current
    out.fields_current = solved.values.copy()
    out.fluid_deltas = solved.fluid_deltas.copy()
    out.field_memory = None if solved.field_memory is None else solved.field_memory.copy()
    out.fluid_time_si = time+dt
    out.fluid_index += 1
    out.fluid_dt_previous_si = dt
    out.plan_cursor += 1
    out.marker.clock.field_start_si = time
    out.marker.clock.field_end_si = out.fluid_time_si
    out.marker.clock.startup_pending = False
    out.validate_fluid_history()
    return out


def reference_particle_then_fluid(state: ReferenceRollout, *, particle_step,
                                  solve_fluid, update_shards):
    """Run one causal window with available fields, then next fluid event.

    particle_step(marker, target_si, field_previous, field_current, shards)
    returns (updated_marker, diagnostics) using only self-generated state.
    For startup, call reference_fluid_event FIRST: no particle step before the
    zero-projection fluid/shard event. Ordinary windows never call restart.
    """
    if state.marker.clock.startup_pending:
        raise ValueError("JOREK startup fluid event must precede particle sampling")
    if state.fields_previous is None:
        raise ValueError("particle window requires field interpolation history")
    out = deepcopy(state)
    out.marker, diagnostics = particle_step(out.marker, out.fluid_time_si,
                                           out.fields_previous, out.fields_current, out.shards)
    out = reference_fluid_event(out, solve_fluid=solve_fluid, update_shards=update_shards)
    return out, diagnostics
