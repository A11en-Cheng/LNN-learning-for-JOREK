"""Production-distribution births at the actual source and RNG checkpoint seams."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture
from tests.test_native_production_density import source_fixture


def birth_fixture():
    from src.models.native_source_birth import NativeProductionBirths
    kernel,s=source_fixture();mesh,source,marker,u=native_fixture()
    # Domain and injection kernel share the same explicitly bound geometry.
    births=NativeProductionBirths(kernel,physical_particles=2e13,n_particle_slots=2,
        n_cpu=1,rank=0,R_geo=2.5,n_plane=1,max_proposals=100)
    s['ablation_atoms_s'][:]=2.5e22
    marker.clock.sim_time_si=1e-9;marker.clock.field_start_si=0.;marker.clock.field_end_si=1e-9
    marker.clock.residual_si=.5e-9
    marker.feedback.time_si=1e-9
    return births,s,source,marker,u


def test_birth_accepts_native_mass_source_gate_without_number_reconversion(tmp_path,monkeypatch):
    """Independent Fortran gate at one analytic source center."""
    import math,shutil,subprocess
    from pathlib import Path
    if not shutil.which('gfortran'):pytest.skip('independent Fortran oracle needs gfortran')
    program=tmp_path/'native-mass-birth-oracle'
    subprocess.run(['gfortran',str(Path(__file__).with_name('native_birth_reference.f90')),'-o',str(program)],check=True)
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    q=source.mesh.units.central_mass/20
    volume=math.pi*2.5*math.sqrt(math.pi)*math.erf(math.pi)*.05**2
    oracle=subprocess.run([str(program)],input=f'2.5 1 {1/(q*volume):.17g} 1\n',
        capture_output=True,text=True,check=True)
    _,native_gate=map(float,oracle.stdout.split())
    draw=native_gate*(1+q)/2
    assert native_gate*q<draw<native_gate<1
    births.max_proposals=2
    monkeypatch.setattr(np.random,'random',lambda n:np.array([.475,.5,0.,draw,.2,.3,.4]))
    cloud=births.births(s,marker,2e-9,u,u,source)
    assert len(cloud.weight)==2
    np.testing.assert_array_equal(cloud.P_imp[:,0],[1.,1.])
    assert not cloud.P_imp[:,1:].any()


def test_production_birth_count_weight_velocity_and_slot_lifetime(monkeypatch):
    births,s,source,marker,u=birth_fixture()
    marker.cloud.i_elm[:]=0;marker.cloud.i_life[:]=[8,3];marker.cloud.t_birth[:]=[7.,9.]
    previous=u.copy();previous[6]=0
    # External RNG oracle: cylindrical proposal yields exactly (R,Z,phi)=(2.5,.5,0).
    monkeypatch.setattr(np.random,'random',lambda n:np.array([.475,.5,0.,0.,.2,.3,.4]))
    cloud=births.births(s,marker,2e-9,previous,u,source)
    assert len(cloud.weight)==2
    np.testing.assert_allclose(cloud.x,[[2.5,.5,0.],[2.5,.5,0.]],atol=1e-15)
    np.testing.assert_array_equal(cloud.weight,np.array([1.25e13,1.25e13],dtype=np.float32).astype(float))
    # Velocity evaluated at sim_time=1ns, not particle_time=.5ns.
    assert cloud.V_prev[0,2]==pytest.approx(-.001/2.5/source.mesh.units.t_norm_si,rel=2e-15)
    # Source inserts before substeps, reuses slots, retains t_birth, increments lifetime once.
    out,_=source.advance(marker,2e-9,previous,u,births=cloud)
    np.testing.assert_array_equal(out.cloud.i_life,[9,4])
    np.testing.assert_array_equal(out.cloud.t_birth,[7.,9.])
    np.testing.assert_array_equal(marker.cloud.i_life,[8,3])


def test_birth_threshold_mpi_division_and_float32_weight(monkeypatch):
    from src.models.native_source_birth import NativeProductionBirths
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    births=NativeProductionBirths(births.kernel,physical_particles=1e22,n_particle_slots=2,
        n_cpu=2,rank=1,R_geo=2.5,n_plane=1,max_proposals=10)
    monkeypatch.setattr(np.random,'random',lambda n:np.array([.475,.5,0.,0.,.2,.3,.4]))
    s['ablation_atoms_s'][:]=2e23  # local rate 1e23 * gather1ns = 1e14, strict > threshold false.
    assert len(births.births(s,marker,2e-9,u,u,source).weight)==0
    s['ablation_atoms_s'][:]=2.02e23
    cloud=births.births(s,marker,2e-9,u,u,source)
    assert len(cloud.weight)==1
    assert cloud.weight[0]==float(np.float32(1.01e14))


def test_rejection_failure_and_missing_rawd_sampler_restore_full_numpy_rng():
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_coupling import NativeMarkerSource
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    limited=NativeProductionBirths(births.kernel,physical_particles=2e13,n_particle_slots=2,
        n_cpu=1,rank=0,R_geo=2.5,n_plane=1,max_proposals=2)
    np.random.seed(7351);before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='proposal budget'):
        limited.births(s,marker,2e-9,u,u,source)
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    assert not marker.cloud.alive.any()
    rawsource=NativeMarkerSource(source.mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=0.,
        time_sampling='jorek_raw_deltas_v1',field_static=True)
    with pytest.raises(ValueError,match='saved-history sampler'):
        births.births(s,marker,2e-9,u,u,rawsource)


def test_production_birth_stream_checkpoint_restores_all_mt19937_words_and_cursor():
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    # Broad width and explicit large dv isolate RNG recovery from rejection cost.
    s['radius'][:]=1/1.4
    from src.models.native_source_birth import NativeProductionBirths
    births=NativeProductionBirths(births.kernel,physical_particles=2e13,n_particle_slots=2,
        n_cpu=1,rank=0,R_geo=1e8,n_plane=1,max_proposals=20)
    np.random.seed(1729);saved=deepcopy(np.random.get_state())
    first=births.births(s,marker,2e-9,u,u,source);state_after=deepcopy(np.random.get_state())
    np.random.random(100)  # unrelated consumption cannot survive restoration.
    np.random.set_state(saved)
    second=births.births(s,marker,2e-9,u,u,source)
    for name in first.array_names():np.testing.assert_array_equal(getattr(first,name),getattr(second,name))
    actual=np.random.get_state()
    np.testing.assert_array_equal(actual[1],state_after[1]);assert actual[2:]==state_after[2:]


def test_rawd_birth_velocity_consumes_sim_time_sampler_and_not_endpoint_difference(monkeypatch):
    from src.models.native_coupling import NativeMarkerSource
    from src.models.native_source_time import NativeSourceTimeSampler
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    marker.clock.sim_time_si=.5e-9;marker.clock.residual_si=.2e-9
    marker.feedback.time_si=.5e-9
    source=NativeMarkerSource(source.mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=0.,
        time_sampling='jorek_raw_deltas_v1',field_static=False)
    raw=np.zeros_like(u);raw[6,0]=.0004
    sampler=NativeSourceTimeSampler(u,raw,time_prev_si=0.,time_now_si=1e-9,tstep_si=1e-9,static=False)
    previous=u.copy();previous[6]=0
    monkeypatch.setattr(np.random,'random',lambda n:np.array([.475,.5,0.,0.,.2,.3,.4]))
    cloud=births.births(s,marker,1.5e-9,previous,u,source,time_sampler=sampler)
    # Saved vpar=.001-.5*.0004=.0008; residual-time velocity would be .00072.
    assert cloud.V_prev[0,2]==pytest.approx(-.0008/2.5/source.mesh.units.t_norm_si,rel=2e-15)


def test_birth_unit_mismatch_fails_before_any_rng_consumption():
    from dataclasses import replace
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    source=deepcopy(source);source.mesh.units=replace(source.mesh.units,n_norm_m3=1e20)
    before=deepcopy(np.random.get_state())
    with pytest.raises(ValueError,match='geometry/units'):
        births.births(s,marker,2e-9,u,u,source)
    np.testing.assert_array_equal(np.random.get_state()[1],before[1]);assert np.random.get_state()[2:]==before[2:]


def test_rejection_prefilter_preserves_known_birth_stream_with_bounded_cost():
    """Versioned native-mass scalar stream; old number-gate results are invalid."""
    import time
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    births.max_proposals=100000
    start=time.perf_counter()
    for _ in range(8):
        np.random.seed(2917)
        cloud=births.births(s,marker,2e-9,u,u,source)
    seconds=time.perf_counter()-start
    np.testing.assert_array_equal(cloud.x,[[2.5280283400947785,.4969355326033923,5.845134906523379],
        [2.397521771355787,.46560221313850864,.028318653094736933]])
    np.testing.assert_array_equal(cloud.weight,[12500000047104.,12500000047104.])
    assert np.random.get_state()[2]==234
    from src.models.native_state import configuration_digest
    assert configuration_digest(np.random.get_state())=='187d3f940c6b3148c78721eca9c286ab3f4515b23dc4a5d5e233f60821b774ff'
    assert seconds<3, f'eight exact rejection streams took {seconds:.3f}s'


def test_birth_rejection_rounding_matches_independent_fortran_oracle(tmp_path,monkeypatch):
    import math,shutil,subprocess
    from pathlib import Path
    if not shutil.which('gfortran'):pytest.skip('independent Fortran oracle needs gfortran')
    program=tmp_path/'birth-oracle'
    subprocess.run(['gfortran',str(Path(__file__).with_name('native_birth_reference.f90')),'-o',str(program)],check=True)
    # Analytic center shape=1 and circular source volume; t_norm/n_norm cancels.
    volume=math.pi*2.5*math.sqrt(math.pi)*math.erf(math.pi)*.05**2
    oracle=subprocess.run([str(program)],input=f'2.5 1 {20/volume:.17g} 1\n',
        capture_output=True,text=True,check=True)
    _,acceptance=map(float,oracle.stdout.split())
    rejection=(acceptance+float(np.nextafter(np.float32(acceptance),np.float32(np.inf))))/2
    births,s,source,marker,u=birth_fixture();marker.cloud.i_elm[:]=0
    births.max_proposals=2
    monkeypatch.setattr(np.random,'random',lambda n:np.array([.475,.5,0.,rejection,.2,.3,.4]))
    with pytest.raises(ValueError,match='proposal budget'):
        births.births(s,marker,2e-9,u,u,source)
