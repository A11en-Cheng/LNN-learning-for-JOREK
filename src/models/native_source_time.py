"""Inspected JOREK linear first-spatial-derivative time sampling.

mod_fields_linear.f90 do_interp_PRZ_1 / do_interp_PRZP_1. This does not
implement its distinct second-spatial-derivative branch or infer raw history.
Coefficients and their compatible spatial derivatives share the same linear
combination on fixed geometry. Times and tstep are SI seconds.
"""
import numpy as np


class NativeSourceTimeSampler:
    version='jorek_raw_deltas_v1'

    def __init__(self,current,raw_deltas,*,time_prev_si,time_now_si,tstep_si,static):
        current=np.asarray(current,dtype=float)
        raw=np.asarray(raw_deltas,dtype=float)
        if (current.ndim!=4 or current.shape[:2]!=(9,4) or min(current.shape[2:])<1
            or raw.shape!=current.shape or not np.isfinite(current).all() or not np.isfinite(raw).all()):
            raise ValueError('complete finite nine-field native coefficients/rawD required')
        if (not isinstance(static,(bool,np.bool_))
            or not np.isfinite([time_prev_si,time_now_si,tstep_si]).all()):
            raise ValueError('explicit finite SI clocks and boolean static required')
        self._current=current.copy();self._raw=raw.copy()
        self._current.flags.writeable=False;self._raw.flags.writeable=False
        self.time_prev_si=float(time_prev_si);self.time_now_si=float(time_now_si)
        self.tstep_si=float(tstep_si);self.static=bool(static)

    def _interval(self,time_si):
        if not np.isscalar(time_si) or not np.isfinite(time_si):
            raise ValueError('finite scalar sample time required')
        interval=self.time_now_si-self.time_prev_si
        return interval, self.tstep_si>0 and abs(interval)>1e-10 and not self.static

    @property
    def current(self):
        return self._current.copy()

    @property
    def raw_deltas(self):
        return self._raw.copy()

    def fields_at(self,time_si):
        """Full native U; no clamp, including residual before time_prev."""
        interval,interpolate=self._interval(time_si)
        if interpolate:
            return self._current-self._raw*((self.time_now_si-time_si)/interval)
        return self._current.copy()

    def derivative_at(self,time_si):
        """Native coefficients per second; static still has saved derivative."""
        interval,interpolate=self._interval(time_si)
        if self.tstep_si<=0:return np.zeros_like(self._raw)
        return self._raw/(interval if interpolate else self.tstep_si)
