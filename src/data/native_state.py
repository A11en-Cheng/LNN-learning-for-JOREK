"""Explicit full-axis one-file readers. No adjacent-file history inference."""
from pathlib import Path
import h5py
import numpy as np
from src.models.native_state import NativeFieldState, NativeFeedbackState, NativeUnits, NativeStateContext


def _scalar(f, key):
    if key not in f or f[key].size != 1:
        raise ValueError(f'missing native metadata {key}')
    v = float(np.ravel(f[key][()])[0])
    if not np.isfinite(v):
        raise ValueError(f'nonfinite native metadata {key}')
    return v


def read_native_field_state(path, *, units: NativeUnits, modes):
    """Field t_now is JOREK time; units and mode plan must be supplied."""
    units.validate()
    with h5py.File(path, 'r') as f:
        if _scalar(f,'jorek_model') != 502 or _scalar(f,'n_order') != 3 or _scalar(f,'n_tor') != len(modes):
            raise ValueError('unsupported native model/order/harmonic plan')
        actual = [_scalar(f,'t_norm'), _scalar(f,'central_density')*1e20, _scalar(f,'central_mass')]
        if not np.allclose(actual,[units.t_norm_si,units.n_norm_m3,units.central_mass],rtol=1e-12,atol=0):
            raise ValueError('native field units mismatch')
        if 'deltas' not in f:
            raise ValueError('missing raw fluid history; endpoints are insufficient')
        index = _scalar(f,'index_now')
        if index != int(index):
            raise ValueError('native internal index must be integer')
        return NativeFieldState(np.asarray(f['values'][()]), np.asarray(f['deltas'][()]),
            units, _scalar(f,'t_now')*units.t_norm_si, tuple(modes), int(index))


def read_native_feedback_state(path, *, units: NativeUnits, modes,
                               raw_deltas=None, history_provenance=None, gather_time_si=None):
    """Projection t_now is SI. Explicit saved history or file deltas required.

    This reader does not infer deltas from adjacent outputs. External history
    is a caller assertion with provenance, not evidence of lossless recovery.
    """
    units.validate()
    with h5py.File(path, 'r') as f:
        if raw_deltas is None:
            if 'deltas' not in f:
                raise ValueError('missing raw projection history; do not assume adjacent files suffice')
            raw_deltas = np.asarray(f['deltas'][()])
            history_provenance = str(Path(path).resolve()) + ':/deltas'
        elif not history_provenance:
            raise ValueError('external raw projection history requires provenance')
        return NativeFeedbackState(np.asarray(f['values'][()]), np.array(raw_deltas,copy=True),
            units, _scalar(f,'t_now'), tuple(modes), history_provenance, gather_time_si)


def read_native_context(path, *, units, modes, configuration):
    """Bind actual immutable mesh/BC content plus explicitly resolved config.

    Caller supplies physics/tables, numeric and model settings. File paths are
    not identities; hashes include geometry, signed sizes, topology and codes.
    """
    from copy import deepcopy
    read_native_field_state(path,units=units,modes=modes)  # validates layout/units
    config = deepcopy(configuration)
    with h5py.File(path,'r') as f:
        mesh_keys=('x','size','vertex','index','n_order','n_tor','n_plane','n_period','n_coord_tor')
        for k in (*mesh_keys,'boundary','freeboundary'):
            if k not in f: raise ValueError(f'missing native mesh/BC configuration {k}')
        config['mesh']={k:np.asarray(f[k][()]) for k in mesh_keys}
        for k in ('contain_node','neighbours'):
            if k in f: config['mesh'][k]=np.asarray(f[k][()])
        config['boundary']={'native':{k:np.asarray(f[k][()]) for k in ('boundary','freeboundary')},
                            'options':config.get('boundary',{})}
    return NativeStateContext(units,tuple(modes),config)


def audit_native_segment(run_path, indices, *, restart_events):
    """Read explicitly requested diagnostic frames; never a rollout input.

    Verify fluid internal step/time/step size against the macroscopic table;
    separately check projection SI time and sparse marker snapshot event time.
    No marker payload or future field is used to reconstruct inference history.
    """
    from .output_contract import parse_macroscopic_times
    run=Path(run_path)
    table=parse_macroscopic_times(run/'macroscopic_vars.dat')
    events=set(restart_events)
    rows=[]
    for index in indices:
        with h5py.File(run/f'jorek{index:06d}.h5','r') as f:
            internal=_scalar(f,'index_now')
            t,tn,dt=(_scalar(f,k) for k in ('t_now','t_norm','tstep'))
            shape=f['values'].shape
        if internal != index or index not in table or abs(t-table[index])>1e-7:
            raise ValueError('fluid internal step/time mismatch')
        if dt<=0 or tn<=0:
            raise ValueError('invalid fluid step/normalization')
        interval=None if index-1 not in table else t-table[index-1]
        restart=index in events
        if interval is not None:
            if restart:
                if abs(interval)>1e-7: raise ValueError('registered restart is not duplicate fluid time')
            elif interval<=0 or not np.isclose(interval,dt,rtol=1e-9,atol=1e-7):
                raise ValueError('fluid output interval is not one internal timestep; register specific segment')
        projection=run/f'projections{index:06d}.h5'
        with h5py.File(projection,'r') as f:
            projection_time=_scalar(f,'t_now')
            is_zero=not np.asarray(f['values'][()]).any() if index+1 in events else None
        if index+1 in events:
            phase='restart_zero_before_startup'
            if not is_zero or index-1 not in table or abs(projection_time-table[index-1]*tn)>1e-14:
                raise ValueError('registered startup zero projection/time mismatch')
        else:
            phase='aligned'
            if abs(projection_time-t*tn)>max(1e-14,tn*1e-7):
                raise ValueError('projection/fluid internal time mismatch')
        marker=run/f'part{index:05d}.h5'
        particle_time=None
        if marker.exists():
            with h5py.File(marker,'r') as f: particle_time=_scalar(f,'time')
            if index-1 not in table or abs(particle_time-table[index-1]*tn)>max(1e-14,tn*1e-7):
                raise ValueError('particle snapshot event time mismatch')
        rows.append(dict(index=index,internal_index=int(internal),fluid_time_jorek=t,
            t_norm_si=tn,fluid_tstep_jorek=dt,fluid_interval_jorek=interval,
            restart=restart,projection_time_si=projection_time,projection_phase=phase,
            marker_time_si=particle_time,marker_present=marker.exists(),field_shape=shape))
    return rows


def read_native_update_mapping(path, *, keep_n0_const, boundary_policy, jeccd=False):
    """One-shot update_values metadata; physical BC assembly is NOT provided.

    Current learning path must explicitly acknowledge unverified boundary
    dynamics. Requests for physical boundary enforcement are unsupported.
    """
    from src.models.native_field_step import NativeUpdateMapping
    if boundary_policy!='learned_unverified':
        raise ValueError('physical boundary policy is unsupported by native learning step')
    def logical(a):
        a=np.asarray(a)
        if a.dtype.kind=='b': return a
        text=np.char.strip(a.astype(str))
        if not np.isin(text,['F','T']).all():
            raise ValueError('invalid native logical metadata')
        return text=='T'
    with h5py.File(path,'r') as f:
        if _scalar(f,'jorek_model')!=502 or _scalar(f,'n_order')!=3 or _scalar(f,'n_tor')!=1:
            raise ValueError('native learning update supports model502 cubic H=1 only')
        for key in ('index','constrained','treat_axis'):
            if key not in f: raise ValueError(f'missing native update configuration {key}')
        axis=logical(f['treat_axis'][()])
        if axis.size!=1: raise ValueError('invalid native treat_axis flag')
        return NativeUpdateMapping(np.asarray(f['index'][()]),constrained=logical(f['constrained'][()]),
            treat_axis=bool(axis.ravel()[0]),keep_n0_const=keep_n0_const,jeccd=jeccd)
