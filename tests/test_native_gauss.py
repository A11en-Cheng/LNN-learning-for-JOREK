"""Full native cubic basis and local two-term reference chain, not a solver."""
import numpy as np
import pytest
from pathlib import Path


def native_coefficients(value, ds, dt, dst):
    """Independent endpoint jet conversion, not the evaluation basis."""
    a = np.empty((4, 4))
    for v, (s, t) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        ss, tt = 1-2*s, 1-2*t
        a[:, v] = value(s, t), ss*ds(s, t)/3, tt*dt(s, t)/3, ss*tt*dst(s, t)/9
    return a


def rectangle_geometry():
    from src.models.native_gauss import NativeElementGeometry
    # Raw local HDF5 x layout [R/Z,degree,coord_harmonic,vertex].
    # R=2+s, Z=t; derivative-coefficients are derivative/3, right/top sign flips.
    x = np.zeros((2, 4, 1, 4))
    x[0, 0, 0] = [2, 3, 3, 2]
    x[1, 0, 0] = [0, 0, 1, 1]
    x[0, 1, 0] = [1/3, -1/3, -1/3, 1/3]
    x[1, 2, 0] = [1/3, 1/3, -1/3, -1/3]
    return NativeElementGeometry(x, np.ones((4, 4)), n_order=3)


def test_native_derivative_degree_is_bezier_scaled_and_not_dof0():
    from src.models.native_gauss import evaluate_native_element, GaussRule, ToroidalSampling
    values = np.zeros((1, 4, 1, 4))
    values[0, 1, 0, 0] = 1  # vertex1 degree2, not a standard Hermite derivative
    out = evaluate_native_element(rectangle_geometry(), values,
        rule=GaussRule(4), toroidal=ToroidalSampling((0,), 1, 1))
    s, t = out.s, out.t
    # Independent expanded polynomial for B0(s)*A0(t).
    f = (3*s-6*s*s+3*s**3)*(1-3*t*t+2*t**3)
    fs = (3-12*s+9*s*s)*(1-3*t*t+2*t**3)
    ft = (3*s-6*s*s+3*s**3)*(-6*t+6*t*t)
    np.testing.assert_allclose(out.values[:, 0], f, atol=2e-15)
    np.testing.assert_allclose(out.d_s[:, 0], fs, atol=2e-15)
    np.testing.assert_allclose(out.d_t[:, 0], ft, atol=2e-15)
    np.testing.assert_allclose(out.d_x[:, 0], fs, atol=2e-14)
    np.testing.assert_allclose(out.d_y[:, 0], ft, atol=2e-14)
    np.testing.assert_allclose(out.coordinates, np.column_stack([2+s, t]), atol=2e-15)


def test_complete_cubic_analytic_field_on_curved_signed_size_element():
    from src.models.native_gauss import evaluate_native_element, GaussRule, ToroidalSampling, NativeElementGeometry
    r = lambda s,t: 2+s+.1*s*s*t
    rs = lambda s,t: 1+.2*s*t
    rt = lambda s,t: .1*s*s
    rst = lambda s,t: .2*s
    z = lambda s,t: t+.2*s*t*t
    zs = lambda s,t: .2*t*t
    zt = lambda s,t: 1+.4*s*t
    zst = lambda s,t: .4*t
    f = lambda s,t: r(s,t)+2*z(s,t)+r(s,t)*z(s,t)
    fs = lambda s,t: (1+z(s,t))*rs(s,t)+(2+r(s,t))*zs(s,t)
    ft = lambda s,t: (1+z(s,t))*rt(s,t)+(2+r(s,t))*zt(s,t)
    fst = lambda s,t: (1+z(s,t))*rst(s,t)+zt(s,t)*rs(s,t)+(2+r(s,t))*zst(s,t)+rt(s,t)*zs(s,t)
    size = np.arange(1,17).reshape(4,4)/8
    size[1::2, ::2] *= -1  # ensure signed/nonunit native size used once
    coords = np.stack([native_coefficients(r,rs,rt,rst), native_coefficients(z,zs,zt,zst)])[:, :, None, :]/size[None, :, None, :]
    values = native_coefficients(f,fs,ft,fst)[None,:,None,:]/size[None,:,None,:]
    out = evaluate_native_element(NativeElementGeometry(coords, size, 3), values,
        rule=GaussRule(4), toroidal=ToroidalSampling((0,),1,1))
    s,t = out.s,out.t
    for actual, expected in ((out.values[:,0],f(s,t)), (out.d_s[:,0],fs(s,t)),
        (out.d_t[:,0],ft(s,t)), (out.d_x[:,0],1+z(s,t)), (out.d_y[:,0],2+r(s,t))):
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_all_fourier_coefficients_and_phi_derivatives_are_consumed():
    from src.models.native_gauss import evaluate_native_element, GaussRule, ToroidalSampling
    values = np.zeros((1,4,3,4))
    values[0,0,:,:] = np.array([1.,2.,3.])[:,None]
    out = evaluate_native_element(rectangle_geometry(), values,
        rule=GaussRule(4), toroidal=ToroidalSampling((0,2,2),8,2))
    p = out.phi
    np.testing.assert_allclose(out.values[:,0], 1+2*np.cos(2*p)+3*np.sin(2*p), atol=3e-15)
    np.testing.assert_allclose(out.d_phi[:,0], -4*np.sin(2*p)+6*np.cos(2*p), atol=3e-15)
    assert out.weights.sum() == pytest.approx(8)  # ordinary RHS is a plane SUM


def constant_native(fields, n_harmonics=1):
    a = np.zeros((len(fields),4,n_harmonics,4))
    a[:,0,0,:] = np.asarray(fields)[:,None]
    return a


def local_inputs():
    from tests.test_marker_feedback import fixture
    f, _, _, p = fixture()
    fluid = constant_native([0,0,0,0,10,1,0,2,4])
    projection = constant_native([2,3,20,6,2])
    deltas = constant_native([0,0,0,.8,.2])
    return fluid, projection, deltas, f.units, p


def test_local_two_loads_match_independent_polynomial_integrals():
    from src.models.native_gauss import (marker_electron_energy_local_loads, GaussRule,
                                         ToroidalSampling, FluidCorrectionParameters)
    from src.models.marker_state import fluid_time_coefficients
    u, p, d, units, parameters = local_inputs()
    out = marker_electron_energy_local_loads(rectangle_geometry(), u, p, d,
        rule=GaussRule(4), toroidal=ToroidalSampling((0,),1,1), units=units,
        parameters=parameters, correction=FluidCorrectionParameters(.01,.02),
        time=fluid_time_coefficients('Gears', .5, .25), dt_si=.5*units.t_norm_si)
    # Analytic ∫(2+s)*A_left=23/20, A_right=27/20; B_left=3/5,B_right=13/20.
    # In t: ∫A=1/2, ∫B=1/4. NO 2pi factor.
    integrals = np.array([[23/40,3/10,23/80,3/20],
                          [27/40,13/40,27/80,13/80],
                          [27/40,13/40,27/80,13/80],
                          [23/40,3/10,23/80,3/20]])[:,:,None]
    np.testing.assert_allclose(out.direct, -2.5*integrals, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(out.projection_history, -.16*integrals, rtol=2e-14, atol=2e-15)
    np.testing.assert_array_equal(out.rhs_indices[:,:,0], np.arange(16).reshape(4,4)*9+8)
    assert out.direct[:,0,0].sum() == pytest.approx(-6.25)
    np.testing.assert_allclose(out.feedback.raw_deltas[:,3], .8, atol=2e-15)


def test_native_hdf5_geometry_selects_full_degrees_and_correct_vertex_order(tmp_path):
    import h5py
    from src.data.native_element import read_native_element
    path = tmp_path/'native.h5'
    geo = rectangle_geometry()
    ids = np.array([5,2,7,1])  # one-based source vertex IDs
    coords = np.zeros((2,4,1,7))
    coords[:,:,:,ids-1] = geo.coordinates
    with h5py.File(path,'w') as f:
        f['x']=coords; f['size']=geo.size[:,:,None]; f['vertex']=ids[:,None]
        for k,v in dict(jorek_model=502,n_order=3,n_tor=1,n_coord_tor=1,n_plane=1,n_period=1).items():
            f[k]=[v]
    element = read_native_element(path, element_index=0)
    values = np.arange(9*4*7).reshape(9,4,1,7)
    np.testing.assert_array_equal(element.gather(values), values[:,:,:,ids-1])
    np.testing.assert_array_equal(element.geometry.coordinates, geo.coordinates)
    np.testing.assert_array_equal(element.node_indices, ids-1)
    np.testing.assert_array_equal(element.geometry.size, geo.size)


def test_corrections_are_after_full_evaluation_and_history_keeps_raw_fluid():
    from src.models.native_gauss import (marker_electron_energy_local_loads, GaussRule,
                                         ToroidalSampling, FluidCorrectionParameters)
    from src.models.marker_state import fluid_time_coefficients
    u,p,d,units,parameters = local_inputs()
    u[4,0,0,:] = -.1; u[7,0,0,:] = -.2; u[8,0,0,:] = -.4
    out = marker_electron_energy_local_loads(rectangle_geometry(),u,p,d,
        rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1),units=units,parameters=parameters,
        correction=FluidCorrectionParameters(.01,.02),time=fluid_time_coefficients('Gears',1.,.5),
        dt_si=units.t_norm_si)
    np.testing.assert_allclose(out.fluid.rho_corrected,1e-12,atol=1e-25)
    np.testing.assert_allclose(out.fluid.rho_imp_corrected,1e-12,atol=1e-25)
    np.testing.assert_allclose(out.fluid.te_corrected, .01+.01*np.exp(-42),atol=2e-16)
    np.testing.assert_allclose(out.fluid.rho_imp,-.2,atol=2e-16)
    # delta_alpha=.02, RAW Te=-.4, RAW rhoimp=-.2 -> history=-.0016.
    np.testing.assert_allclose(out.consumed.projection_history_jorek,-.0016,atol=2e-17)


@pytest.fixture(scope='module')
def fortran_reference(tmp_path_factory):
    import shutil
    import subprocess
    source = Path('/home/ac_desktop/JOREK_code/jorek/elements')
    if shutil.which('gfortran') is None or not (source/'mod_basisfunctions.f90').exists():
        pytest.skip('independent local JOREK Fortran basis/compiler unavailable')
    build = tmp_path_factory.mktemp('native-fortran')
    stub = build/'parameters.f90'
    stub.write_text('module mod_parameters\ninteger,parameter::n_order=3,n_degrees=4\nend module\n')
    binary = build/'reference'
    command = ['gfortran','-cpp','-ffree-line-length-none','-O0',str(stub),
               str(source/'gauss.f90'),str(source/'mod_basisfunctions.f90'),
               str(Path('tests/native_gauss_reference.f90').resolve()),'-o',str(binary)]
    subprocess.run(command,cwd=build,check=True,capture_output=True,text=True)
    def compare(geometry, u, p, d, step, ratio, mass):
        arrays = [geometry.size,geometry.coordinates[:,:,0,:],u[:,:,0,:],p[:,:,0,:],d[:,:,0,:]]
        stdin = f'{step:.17e} {ratio:.17e} {mass:.17e}\n'
        stdin += ''.join(' '.join(f'{x:.17e}' for x in a.ravel())+'\n' for a in arrays)
        run = subprocess.run([str(binary)],input=stdin,check=True,capture_output=True,text=True)
        lines = run.stdout.splitlines()
        gauss = np.array([np.fromstring(line,sep=' ') for line in lines[:-2]])
        loads = [np.fromstring(line,sep=' ').reshape(4,4,order='F')[:,:,None] for line in lines[-2:]]
        return gauss, loads
    return compare


REAL = Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')


@pytest.mark.skipif(not (REAL/'jorek004111.h5').exists(),reason='real phaseA checkpoint unavailable')
def test_real_phase_a_full_gauss_and_loads_match_unmodified_fortran_basis(fortran_reference):
    import h5py
    from src.data.native_element import read_native_element
    from src.models.native_gauss import marker_electron_energy_local_loads, GaussRule, FluidCorrectionParameters
    from src.models.marker_feedback import ProjectionUnits, MarkerFeedbackParameters
    from src.models.marker_state import fluid_time_coefficients
    with h5py.File(REAL/'jorek004111.h5') as f:
        values = f['values'][...]
        vertices = f['vertex'][...].astype(int)-1
        tn = float(f['t_norm'][0]); cd=float(f['central_density'][0]); cm=float(f['central_mass'][0])
        time = float(f['t_now'][0])*tn
    with h5py.File(REAL/'projections004111.h5') as f:
        projection=f['values'][...]
        assert float(f['t_now'][0]) == pytest.approx(time,abs=1e-14)
    with h5py.File(REAL/'projections004110.h5') as f:
        previous=f['values'][...]
        assert not previous.any()
        assert float(f['t_now'][0]) < time  # past-only reconstruction, no next frame
    deltas=projection-previous  # exact reconstruction from saved previous ZERO
    # Choose a current nonzero element, without any future data.
    signal=np.sum(abs(projection),axis=(0,1,2))
    index=int(np.argmax(signal[vertices].sum(axis=0)))
    element=read_native_element(REAL/'jorek004111.h5',element_index=index)
    u,p,d=[element.gather(a) for a in (values,projection,deltas)]
    units=ProjectionUnits('jorek_marker_projection',cd*1e20,tn,1/(4e-7*np.pi*(2/3)))
    parameters=MarkerFeedbackParameters(cd,cm,5/3,'Ne',10,'element_matrix',True,True,True)
    # Explicit component step .5 JU, ratio2, not an inferred future schedule.
    result=marker_electron_energy_local_loads(element.geometry,u,p,d,rule=GaussRule(4),
        toroidal=element.toroidal,units=units,parameters=parameters,
        correction=FluidCorrectionParameters(3.2085e-4,1e-5),
        time=fluid_time_coefficients('Gears',.5,.25),dt_si=.5*tn)
    expected, loads=fortran_reference(element.geometry,u,p,d,.5,2.,cm/20*(2/2.01455279245))
    g=result.gauss
    actual=np.column_stack([g.coordinates,g.metric.x_s,g.metric.y_s,g.metric.x_t,g.metric.y_t,
                            g.values,g.d_s,g.d_t,g.d_x,g.d_y,result.feedback.values,
                            result.feedback.d_s,result.feedback.d_t,result.feedback.raw_deltas])
    # Scale each physical field/derivative column independently, not by one huge energy channel.
    scale=np.maximum(np.max(abs(expected),axis=0),1e-13)
    assert np.max(abs(actual-expected)/scale) < 2e-11
    np.testing.assert_allclose(result.direct,loads[0],rtol=2e-11,atol=1e-18)
    np.testing.assert_allclose(result.projection_history,loads[1],rtol=2e-11,atol=1e-18)
    assert np.max(abs(result.direct)) > 0 and np.max(abs(result.projection_history)) > 0


@pytest.mark.skipif(not (REAL/'part04110.h5').exists(),reason='real restart checkpoint unavailable')
def test_registered_restart_zero_feedback_local_loads_preserve_fluid_history():
    import json
    import h5py
    from src.data.marker_checkpoint import initialize_native_checkpoint
    from src.data.native_element import read_native_element
    from src.models.native_gauss import marker_electron_energy_local_loads, GaussRule, FluidCorrectionParameters
    from src.models.marker_feedback import ProjectionUnits, MarkerFeedbackParameters
    from src.models.marker_state import fluid_time_coefficients
    registry=json.loads(Path('data/restart_events.json').read_text())
    assert 4111 in registry['runs'][REAL.name]['restart_events']
    cp=initialize_native_checkpoint(REAL/'jorek004110.h5',REAL/'part04110.h5',event_index=4110,
        times_path=REAL/'macroscopic_vars.dat',namelist_path=REAL/'jorek_namelist_spitest',boundary='jorek_restart')
    assert cp.marker.clock.startup_pending
    assert cp.fluid_deltas.any()
    with h5py.File(REAL/'projections004110.h5') as f:
        np.testing.assert_array_equal(cp.marker.feedback.values,f['values'][...])
        assert cp.marker.feedback.time_si == float(f['t_now'][0])
    element=read_native_element(REAL/'jorek004110.h5',element_index=100)
    units=ProjectionUnits('jorek_marker_projection',cp.central_density*1e20,cp.t_norm,1/(4e-7*np.pi*(2/3)))
    parameters=MarkerFeedbackParameters(cp.central_density,1.,5/3,'Ne',10,'element_matrix',True,True,True)
    old=cp.fluid_deltas.copy()
    out=marker_electron_energy_local_loads(element.geometry,element.gather(cp.fluid_values),
        element.gather(cp.marker.feedback.values),element.gather(cp.marker.feedback.deltas),
        rule=GaussRule(4),toroidal=element.toroidal,units=units,parameters=parameters,
        correction=FluidCorrectionParameters(3.2085e-4,1e-5),
        time=fluid_time_coefficients('Gears',.5,.5),dt_si=.5*cp.t_norm)
    np.testing.assert_array_equal(out.direct,np.zeros((4,4,1)))
    np.testing.assert_array_equal(out.projection_history,np.zeros((4,4,1)))
    np.testing.assert_array_equal(cp.fluid_deltas,old)  # not marker projection history


def test_finite_inputs_that_overflow_evaluation_fail_instead_of_returning_invalid_fields():
    from src.models.native_gauss import evaluate_native_element, GaussRule, ToroidalSampling
    geo=rectangle_geometry()
    geo.size[0,:]=8
    # Compensate geometry so it remains a valid rectangle; field values overflow.
    geo.coordinates[:,0,:,:] /= 8
    values=constant_native([1e308])
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError,match='nonfinite evaluated'):
            evaluate_native_element(geo,values,rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1))


@pytest.mark.parametrize('ratio', [.25,1.,2.,4.])
def test_local_history_scaling_is_once_and_signed_jacobian_is_preserved(ratio,monkeypatch):
    import h5py
    from src.models.native_gauss import marker_electron_energy_local_loads, GaussRule, ToroidalSampling, FluidCorrectionParameters
    from src.models.marker_state import fluid_time_coefficients
    u,p,d,units,parameters=local_inputs()
    geo=rectangle_geometry()
    geo.coordinates[1]*=-1  # J=-1; no abs(J), no extra axisymmetric 2pi
    before=[a.copy() for a in (u,p,d,geo.coordinates,geo.size)]
    def forbidden(*args,**kwargs):
        pytest.fail('pure local chain attempted a file read')
    monkeypatch.setattr(h5py,'File',forbidden)
    out=marker_electron_energy_local_loads(geo,u,p,d,rule=GaussRule(4),
        toroidal=ToroidalSampling((0,),1,1),units=units,parameters=parameters,
        correction=FluidCorrectionParameters(.01,.02),
        time=fluid_time_coefficients('Gears',.5,.5/ratio),dt_si=.5*units.t_norm_si)
    assert out.direct[:,0,0].sum() == pytest.approx(6.25)
    assert out.projection_history[:,0,0].sum() == pytest.approx(.2*ratio)
    for a,b in zip(before,(u,p,d,geo.coordinates,geo.size)):
        np.testing.assert_array_equal(a,b)


def test_fourier_local_test_functions_include_hz_once_and_sum_planes():
    from src.models.native_gauss import marker_electron_energy_local_loads, GaussRule, ToroidalSampling, FluidCorrectionParameters
    from src.models.marker_state import fluid_time_coefficients
    u,p,d,units,parameters=local_inputs()
    u=constant_native([0,0,0,0,10,1,0,2,4],3)
    p=constant_native([0,0,20,6,2],3)
    p[0,0,1,:]=2  # c1=2cos(2phi), signed and untruncated
    d=np.zeros_like(p)
    out=marker_electron_energy_local_loads(rectangle_geometry(),u,p,d,rule=GaussRule(4),
        toroidal=ToroidalSampling((0,2,2),8,2),units=units,parameters=parameters,
        correction=FluidCorrectionParameters(.01,.02),
        time=fluid_time_coefficients('Gears',.5,.5),dt_si=.5*units.t_norm_si)
    # Sum of cos^2 over eight planes is4; ∫R*Aleft*At=23/40.
    assert out.direct[0,0,1] == pytest.approx(-4*23/40)
    np.testing.assert_allclose(out.direct[:,:,[0,2]],0,atol=2e-14)
    np.testing.assert_array_equal(out.projection_history,np.zeros((4,4,3)))
    np.testing.assert_array_equal(out.rhs_indices[0,0,:],[24,25,26])


@pytest.mark.parametrize('bad',['dof0','harmonic','coordinate_modes','order','size','nan','rule','singular'])
def test_missing_or_unsupported_native_layouts_fail(bad):
    from dataclasses import replace
    from src.models.native_gauss import evaluate_native_element, GaussRule, ToroidalSampling
    geo=rectangle_geometry(); values=constant_native([1]); rule=GaussRule(4)
    if bad=='dof0': values=values[:,:1]
    if bad=='harmonic': values=np.zeros((1,4,2,4))
    if bad=='coordinate_modes': geo=replace(geo,coordinates=np.zeros((2,4,3,4)))
    if bad=='order': geo=replace(geo,n_order=5)
    if bad=='size': geo=replace(geo,size=None)
    if bad=='nan': values[0,0,0,0]=np.nan
    if bad=='rule': rule=GaussRule(6)
    if bad=='singular': geo.coordinates[1]=0
    with pytest.raises(ValueError):
        evaluate_native_element(geo,values,rule=rule,toroidal=ToroidalSampling((0,),1,1))


@pytest.mark.parametrize('bad',['delta','correction','channels','threshold'])
def test_local_chain_required_inputs_are_never_filled_with_zero(bad):
    from src.models.native_gauss import marker_electron_energy_local_loads, GaussRule, ToroidalSampling, FluidCorrectionParameters
    from src.models.marker_state import fluid_time_coefficients
    u,p,d,units,parameters=local_inputs()
    correction=FluidCorrectionParameters(.01,.02)
    if bad=='delta': d=None
    if bad=='correction': correction=None
    if bad=='channels': p=p[:2]
    if bad=='threshold': correction=FluidCorrectionParameters(0.,0.)
    with pytest.raises(ValueError):
        marker_electron_energy_local_loads(rectangle_geometry(),u,p,d,rule=GaussRule(4),
            toroidal=ToroidalSampling((0,),1,1),units=units,parameters=parameters,
            correction=correction,time=fluid_time_coefficients('Gears',.5,.5),dt_si=.5*units.t_norm_si)
