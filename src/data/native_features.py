"""Static geometry Adapter for compatible Torch features; no future fields."""
import h5py
import numpy as np
from src.models.native_features import TorchNativeGauss,select_feature_parameters


def read_native_feature_evaluator(path,*,units):
    units.validate()
    with h5py.File(path,'r') as f:
        def scalar(k):
            if k not in f or f[k].size!=1: raise ValueError('missing native feature geometry metadata')
            return float(np.ravel(f[k][()])[0])
        if (scalar('jorek_model')!=502 or scalar('n_order')!=3 or scalar('n_tor')!=1
            or scalar('n_coord_tor')!=1 or scalar('n_plane')!=1 or scalar('n_period')<1):
            raise ValueError('native features require model502 cubic H=1 one-plane axisymmetric geometry')
        if not np.allclose([scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass')],
            [units.t_norm_si,units.n_norm_m3,units.central_mass],rtol=1e-12,atol=0):
            raise ValueError('native feature geometry units mismatch')
        arrays=[np.asarray(f[k][()]) for k in ('x','size','vertex')]
    return TorchNativeGauss(arrays[0],arrays[1],arrays[2]-1,units=units)


def parameters_from_current_input(view,feature_path):
    """Explicit selection; excluded audit entries remain named in the result."""
    return select_feature_parameters(view.parameters,scenario_enabled=feature_path.scenario_enabled,
        physics_enabled=feature_path.physics_enabled)
