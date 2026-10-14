"""Independent literals for inspected JOREK first-derivative field sampling."""
import numpy as np
import pytest


def test_raw_history_not_output_difference_and_residual_not_clamped():
    from src.models.native_source_time import NativeSourceTimeSampler
    current=np.full((9,4,1,2),10.)
    raw=np.full_like(current,4.)
    sampler=NativeSourceTimeSampler(current,raw,time_prev_si=2e-9,
        time_now_si=4e-9,tstep_si=3e-9,static=False)
    # Saved rawD=4; hypothetical preceding saved output=9 is irrelevant.
    # At midpoint the literal Fortran formula gives 8; residual t=1ns gives 4.
    np.testing.assert_array_equal(sampler.fields_at(3e-9),np.full_like(current,8.))
    np.testing.assert_allclose(sampler.fields_at(1e-9),4.,rtol=0,atol=2e-15)
    np.testing.assert_allclose(sampler.derivative_at(3e-9),2e9,rtol=2e-16,atol=0)
    current[:]=100.;raw[:]=100.
    np.testing.assert_array_equal(sampler.fields_at(3e-9),np.full_like(current,8.))


@pytest.mark.parametrize('interval,static,tstep,value,derivative',[
    (1e-10,False,2e-9,10.,2e9),
    (5e-11,False,2e-9,10.,2e9),
    (0.,False,2e-9,10.,2e9),
    (2e-9,True,2e-9,10.,2e9),
    (2e-9,False,0.,10.,0.),
    (2e-9,False,-2e-9,10.,0.),
    (-2e-9,False,2e-9,8.,-2e9),
])
def test_literal_short_static_and_nonpositive_step_branches(interval,static,tstep,value,derivative):
    from src.models.native_source_time import NativeSourceTimeSampler
    sampler=NativeSourceTimeSampler(np.full((9,4,1,1),10.),np.full((9,4,1,1),4.),
        time_prev_si=0.,time_now_si=interval,tstep_si=tstep,static=static)
    np.testing.assert_allclose(sampler.fields_at(interval/2),value,rtol=0,atol=2e-15)
    np.testing.assert_allclose(sampler.derivative_at(interval/2),derivative,rtol=2e-16,atol=0)


def test_complete_history_and_explicit_flags_required():
    from src.models.native_source_time import NativeSourceTimeSampler
    u=np.zeros((9,4,1,1));kw=dict(time_prev_si=0.,time_now_si=1e-9,tstep_si=1e-9,static=False)
    for raw in (None,np.zeros((9,1,1,1)),np.full_like(u,np.nan)):
        with pytest.raises(ValueError):NativeSourceTimeSampler(u,raw,**kw)
    with pytest.raises(ValueError):NativeSourceTimeSampler(u,u,**{**kw,'static':'false'})
    sampler=NativeSourceTimeSampler(u,u,**kw)
    with pytest.raises(ValueError):sampler.fields_at(np.nan)


def test_same_coefficients_define_analytic_sampled_value_and_compatible_gradient():
    from src.models.native_source_time import NativeSourceTimeSampler
    from tests.test_native_coupling import native_fixture
    from tests.test_native_gauss import native_coefficients
    mesh,_,_,_=native_fixture()
    # U=R+2Z, rawD=2R-Z. Midpoint U-.5*rawD=2.5Z exactly.
    u=native_coefficients(lambda s,t:2+s+2*t,lambda s,t:1.,lambda s,t:2.,lambda s,t:0.)
    d=native_coefficients(lambda s,t:4+2*s-t,lambda s,t:2.,lambda s,t:-1.,lambda s,t:0.)
    sampler=NativeSourceTimeSampler(np.repeat(u[None,:,None,:],9,axis=0),
        np.repeat(d[None,:,None,:],9,axis=0),time_prev_si=0.,time_now_si=2e-9,tstep_si=2e-9,static=False)
    values,grad,*_=mesh.point(0,[.3,.4],sampler.fields_at(1e-9))
    np.testing.assert_allclose(values,1.,rtol=0,atol=2e-15)
    np.testing.assert_allclose(grad,np.tile([0.,2.5],(9,1)),rtol=0,atol=3e-15)
