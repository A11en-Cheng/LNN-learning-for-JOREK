"""Read-only, same-event psi teacher evidence; not an autonomous ES finder.

The saved equilibrium is bound ONLY to the original nine fields. Consumers
must explicitly identify and rebind any psi-only hybrid diagnostic branch.
No adjacent output is read, and no field/shard/marker state is installed here.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import h5py
import numpy as np
from .namelist import parse_namelist
from src.models.native_state import NativeUnits, configuration_digest
from src.models.native_source_nonlocal import NativeFluxEquilibrium


def _immutable(array):
    a=np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class NativePsiTeacherRecord:
    psi: np.ndarray
    equilibrium: NativeFluxEquilibrium
    time_si: float
    geometry_digest: str
    units: NativeUnits
    provenance: str
    source_sha256: str
    source_path: str
    field_digest: str
    step_index: int
    coordinates: np.ndarray
    size: np.ndarray
    vertex: np.ndarray             # zero-based, same as NativeSourceMesh
    indices: np.ndarray
    boundary: np.ndarray
    lcfs_min_distance_m: float
    lcfs_history_length: int
    namelist_path: str
    namelist_sha256: str


def read_native_psi_teacher(path, *, expected_time_si=None, namelist_path=None,
                            gamma=5/3, lcfs_restart_index=None,registered_restart_indices=()):
    """Return immutable original-HDF evidence at its own exact event time.

    Defaults to the explicitly named production ``jorek_namelist_spitest``;
    never falls back to the unrelated generic jorek_namelist or future files.
    ``gamma`` is the caller's explicit unit convention (default 5/3), not an
    assertion that a rounded log or an absent HDF value recorded it exactly.
    H1/model502/cubic only. Missing histories/configuration fail closed.
    LCFS is a restart-history reconstruction, NOT the continuously running
    JOREK ES flag. Default assumes restart at this record. Continuous readers
    should pass the fixed original ``lcfs_restart_index``. A false result over
    the whole available history also proves false for every restart prefix;
    a true result has no such general continuous-run interpretation.
    Explicit registered startup edges alone may have a backward roundoff up to
    the existing 1e-14 s event-clock tolerance. No times are modified or sorted;
    accepted discrepancies enter provenance. Ordinary edges keep the ULP guard.
    """
    path=Path(path).resolve()
    namelist_path=Path(namelist_path or path.parent/'jorek_namelist_spitest').resolve()
    params=parse_namelist(namelist_path.read_text())
    if 'xcase' not in params or 'r_geo' not in params or 'xpoint' not in params:
        raise ValueError('explicit production xcase/R_geo/xpoint configuration required')
    cases={1:'lower',2:'upper',3:'double',4:'symmetric'}
    if str(params['xpoint']).lower() not in ('.t.','.true.') or params['xcase'] not in cases:
        raise ValueError('supported explicit X-point case required')
    r_geo=float(params['r_geo'])
    if not np.isfinite(r_geo) or r_geo<=0:raise ValueError('positive R_geo required')
    source_sha256=_sha256(path)
    with h5py.File(path,'r') as f:
        def scalar(key):
            if key not in f or f[key].size!=1:raise ValueError('missing scalar '+key)
            value=float(f[key][()].ravel()[0])
            if not np.isfinite(value):raise ValueError('nonfinite scalar '+key)
            return value
        def array(key):
            if key not in f:raise ValueError('missing history/geometry '+key)
            a=f[key][...]
            if not np.isfinite(a).all():raise ValueError('nonfinite '+key)
            return a
        if any(scalar(k)!=v for k,v in [('jorek_model',502),('n_order',3),('n_tor',1),('n_plane',1),('n_period',1)]):
            raise ValueError('teacher requires complete model502 cubic H1 geometry')
        units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),gamma)
        units.validate()
        index=scalar('index_now')
        if index<1 or index!=int(index):raise ValueError('positive integer recorded index required')
        index=int(index);time_ju=scalar('t_now');time_si=time_ju*units.t_norm_si
        restart=index if lcfs_restart_index is None else lcfs_restart_index
        if type(restart) is not int or not 0<restart<=index:
            raise ValueError('explicit LCFS restart history boundary must be within this record')
        xtime=array('xtime')
        if xtime.shape!=(index,) or xtime[-1]!=time_ju:
            raise ValueError('saved ES history must end at this exact field event')
        # Real restart histories repeat a boundary time (v400 2701/2702,
        # inner 2703/2704). Equal times remain separate historical records.
        if any(type(i) is not int or i<1 for i in registered_restart_indices):
            raise ValueError('explicit integer registered startup indices required')
        delta=np.diff(xtime);bad=delta < -64*np.spacing(max(abs(xtime)))
        registered=np.isin(np.arange(2,index+1),registered_restart_indices)
        accepted=bad & registered & (np.abs(delta)*units.t_norm_si<=1e-14)
        if (bad & ~accepted).any() or (xtime<0).any():
            raise ValueError('saved ES history must be nonnegative and monotonic')
        if expected_time_si is not None and (not np.isfinite(expected_time_si)
                or not np.isclose(time_si,expected_time_si,rtol=0,atol=16*np.spacing(abs(time_si)))):
            raise ValueError('teacher record is not the requested same-time event')
        fields=array('values');x=array('x');size=array('size')
        vertex=array('vertex')-1;indices=array('index');boundary=array('boundary').ravel()
        n=fields.shape[-1]
        if (fields.shape!=(9,4,1,n) or x.shape!=(2,4,1,n) or vertex.ndim!=2
                or vertex.shape[0]!=4 or vertex.dtype.kind not in 'iu'
                or (vertex<0).any() or (vertex>=n).any()
                or size.shape!=(4,4,vertex.shape[1]) or indices.shape!=(4,n)
                or boundary.shape!=(n,)):
            raise ValueError('complete native fields and geometry required')
        histories={k:array(k) for k in ('R_axis_t','Z_axis_t','psi_axis_t','psi_bnd_t',
                                       'psi_xpoint_t','Z_xpoint_t')}
        for k,a in histories.items():
            if a.shape!=((2,index) if 'xpoint' in k else (index,)):
                raise ValueError('same-event history shape mismatch '+k)
    # communication/mod_import_restart:1542 sets index_start = saved index_now.
    # grids/mod_boundary:66 selects each side with two positive boundary codes.
    # equil_info:1282-1302 uses centers of those ELEMENTS, not edge midpoints.
    v=vertex.T;on_boundary=np.any((boundary[v]>0)&(boundary[np.roll(v,-1,axis=1)]>0),axis=1)
    if not on_boundary.any():raise ValueError('boundary elements required for LCFS history')
    # Exact cubic tensor basis at s=t=.5, in native vertex/degree order.
    one=np.array([[.5,.375],[.5,.375]])
    h=np.array([[one[a,c]*one[b,d] for c,d in ((0,0),(1,0),(0,1),(1,1))]
                for a,b in ((0,0),(1,0),(1,1),(0,1))])
    centers=np.einsum('vd,evd,evdc->ec',h,size[:,:,on_boundary].transpose(2,1,0),
                      x[:,:,0,:].transpose(2,1,0)[v[on_boundary]])
    axes=np.column_stack((histories['R_axis_t'][:restart],histories['Z_axis_t'][:restart]))
    # Bounded temporary storage: at most 256 history points x boundary elements.
    minimum=min(float(np.linalg.norm(centers[None]-a[:,None],axis=2).min())
                for a in np.array_split(axes,max(1,(restart+255)//256)))
    geometry_digest=configuration_digest(dict(x=x.astype(float),size=size.astype(float),vertex=vertex,indices=indices))
    field_digest=configuration_digest(fields)
    namelist_sha256=_sha256(namelist_path)
    provenance=(f'psi_teacher_record; original_nine_fields_evidence_only; path={path}; '
        f'sha256={source_sha256}; index={index}; ES_history_endpoint=xtime==t_now; '
        f'LCFS=restart_history_reconstruction; fixed_index_start={restart}; boundary_element_centers; '
        f'namelist={namelist_path}; namelist_sha256={namelist_sha256}; gamma_caller={gamma!r}')
    if accepted.any():
        provenance+=f'; registered_restart_time_roundoff={[(int(i+2),float(delta[i]*units.t_norm_si)) for i in np.flatnonzero(accepted)]}; tolerance_si=1e-14; timestamps_unchanged'
    eq=NativeFluxEquilibrium.bind(fields,psi_axis=float(histories['psi_axis_t'][-1]),
        psi_bnd=float(histories['psi_bnd_t'][-1]),lcfs_lost=bool(minimum<r_geo/30),
        xcase=cases[params['xcase']],psi_xpoints=tuple(histories['psi_xpoint_t'][:,-1]),
        z_xpoints=tuple(histories['Z_xpoint_t'][:,-1]),time_si=time_si,provenance=provenance)
    return NativePsiTeacherRecord(_immutable(fields[0]),eq,time_si,geometry_digest,units,provenance,
        source_sha256,str(path),field_digest,index,*[_immutable(a) for a in (x,size,vertex,indices,boundary)],
        minimum,restart,str(namelist_path),namelist_sha256)
