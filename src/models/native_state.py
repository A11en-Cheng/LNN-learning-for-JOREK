"""Versioned full native coefficients; no field evolution or reconstruction.

All times are explicit SI seconds; coefficient values retain JOREK units.
Node degree-zero observations are copies, never an inverse state representation.
"""
from dataclasses import dataclass, replace
from copy import deepcopy
import numpy as np


@dataclass(frozen=True)
class NativeUnits:
    t_norm_si: float
    n_norm_m3: float
    central_mass: float
    gamma: float
    field_convention: str = 'model502_native'
    projection_convention: str = 'jorek_marker_projection'

    def validate(self):
        if (self.field_convention != 'model502_native'
                or self.projection_convention != 'jorek_marker_projection'
                or not np.isfinite([self.t_norm_si, self.n_norm_m3, self.central_mass, self.gamma]).all()
                or min(self.t_norm_si, self.n_norm_m3, self.central_mass) <= 0 or self.gamma <= 1):
            raise ValueError('invalid explicit native units')

    @property
    def energy_norm_j_m3(self):
        return 1/(4e-7*np.pi*(self.gamma-1))


def validate_coefficients(values, history, channels, modes):
    if history is None:
        raise ValueError('missing raw native history; no lossless reconstruction assumed')
    if (not isinstance(values, np.ndarray) or values.ndim != 4
            or values.shape[:3] != (channels,4,len(modes)) or values.shape[3] < 1
            or values.dtype.kind != 'f' or not np.isfinite(values).all()):
        raise ValueError('requires full finite floating native [field,4,harmonic,node]')
    if (not isinstance(history, np.ndarray) or history.shape != values.shape
            or history.dtype != values.dtype or not np.isfinite(history).all()):
        raise ValueError('raw history must match full native axes and dtype')
    if (not modes or modes[0] != 0 or len(modes)%2 != 1
            or any(not isinstance(m, (int,np.integer)) for m in modes)
            or any(modes[i] <= 0 or modes[i] != modes[i+1] for i in range(1,len(modes),2))):
        raise ValueError('invalid explicit native harmonic modes')


class _NativeLayout:
    def node_dof0(self):
        """Explicit lossy degree0/harmonic0 observation [nodes,fields]."""
        return self.values[:,0,0,:].T.copy()

    def to_node_layout(self, *, history=False):
        """Lossless [node,degree,harmonic,field] Torch copy; dtype unchanged."""
        import torch
        a = self.raw_deltas if history else self.values
        return torch.from_numpy(a.transpose(3,1,2,0).copy())

    def from_node_layout(self, values, history):
        def native(a):
            if a is None:
                raise ValueError('missing raw native history')
            if hasattr(a, 'detach'):
                a = a.detach().cpu().numpy()
            a = np.asarray(a)
            if a.shape != (self.values.shape[3],4,len(self.modes),self.values.shape[0]):
                raise ValueError('node layout requires all native axes')
            return a.transpose(3,1,2,0).copy()
        return replace(self, values=native(values), raw_deltas=native(history))


@dataclass
class NativeFieldState(_NativeLayout):
    values: np.ndarray
    raw_deltas: np.ndarray
    units: NativeUnits
    time_si: float
    modes: tuple[int,...]
    step_index: int
    schema_version: str = 'native-field-v1'

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.schema_version != 'native-field-v1':
            raise ValueError('unsupported native field version')
        if not isinstance(self.units, NativeUnits):
            raise ValueError('explicit native units required')
        self.units.validate()
        validate_coefficients(self.values, self.raw_deltas, 9, self.modes)
        if not np.isfinite(self.time_si) or self.time_si < 0 or not isinstance(self.step_index, (int,np.integer)) or self.step_index < 0:
            raise ValueError('invalid native field time/index')


@dataclass
class NativeFeedbackState(_NativeLayout):
    values: np.ndarray
    raw_deltas: np.ndarray
    units: NativeUnits
    time_si: float                 # particle/projection event, NOT fluid time
    modes: tuple[int,...]
    history_provenance: str
    gather_time_si: float | None = None  # unknown from ordinary projection HDF5
    schema_version: str = 'native-feedback-v1'

    def __post_init__(self):
        self.validate()

    def to_projection_feedback(self):
        """Lossless adapter to the existing reference event driver."""
        from .marker_state import ProjectionFeedback
        self.validate()
        return ProjectionFeedback(self.values.copy(), self.raw_deltas.copy(), self.time_si,
            'native_full', self.history_provenance, self.gather_time_si)

    def validate(self):
        if self.schema_version != 'native-feedback-v1':
            raise ValueError('unsupported native feedback version')
        if not isinstance(self.units, NativeUnits):
            raise ValueError('explicit native units required')
        self.units.validate()
        validate_coefficients(self.values, self.raw_deltas, 5, self.modes)
        if not self.history_provenance:
            raise ValueError('raw projection history provenance required')
        if (not np.isfinite(self.time_si) or self.time_si < 0
                or (self.gather_time_si is not None and
                    (not np.isfinite(self.gather_time_si) or self.gather_time_si <= 0))):
            raise ValueError('invalid projection event/gather time')


def configuration_digest(value):
    """Canonical configuration identity, including exact array content/axes."""
    import hashlib
    import json
    def canonical(v):
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:
                raise ValueError('object arrays are not configuration')
            return {'dtype':v.dtype.str,'shape':list(v.shape),
                    'sha256':hashlib.sha256(v.tobytes()).hexdigest()}
        if isinstance(v,NativeUnits):
            return canonical(vars(v))
        if isinstance(v,dict):
            if not all(isinstance(k,str) for k in v):
                raise ValueError('configuration keys must be strings')
            return {k:canonical(v[k]) for k in sorted(v)}
        if isinstance(v,(tuple,list)):
            return [canonical(x) for x in v]
        if isinstance(v,np.generic):
            return v.item()
        return v
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


@dataclass
class NativeStateContext:
    units: NativeUnits
    modes: tuple[int,...]
    configuration: dict
    schema_version: str = 'native-context-v1'

    def __post_init__(self):
        self.configuration = deepcopy(self.configuration)
        self.validate()

    def validate(self):
        self.units.validate()
        if self.schema_version != 'native-context-v1':
            raise ValueError('unsupported native context version')
        for name in ('mesh','boundary','physics','numerics','model'):
            if not self.configuration.get(name):
                raise ValueError(f'missing native configuration {name}')
        configuration_digest((self.units,self.modes,self.configuration,self.schema_version))

    def fingerprint(self):
        self.validate()
        return configuration_digest((self.units,self.modes,self.configuration,self.schema_version))
