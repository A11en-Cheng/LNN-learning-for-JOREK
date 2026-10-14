"""One native mass-source acceptance contract for every SPI birth algorithm."""
import numpy as np

NATIVE_BIRTH_ACCEPTANCE='jorek_impurity_mass_source_real4_total_gate_v2'


def native_birth_threshold(source_mass,*,dv,norm):
    """JOREK f_source_imp: real*4(max(P*dv/P_norm,0)), before probability cap.

    P is already the impurity MASS-density rate. Do not convert it back to an
    atom-number rate. The real*4 overflow is valid saturation, not a NaN input.
    """
    source_mass=np.asarray(source_mass,dtype=float)
    if (not np.isfinite(source_mass).all() or not np.isfinite([dv,norm]).all()
        or min(dv,norm)<=0):
        raise ValueError('finite mass source and positive native birth dv/norm required')
    with np.errstate(over='ignore'):
        return np.maximum(source_mass*dv/norm,0.).astype(np.float32).astype(float)
