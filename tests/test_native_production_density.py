"""Independent source kernel, not marker c5 or filtered projection."""
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture


def source_fixture(minimum=.05, numerical=False):
    from src.models.native_source_density import NativeShardDensityKernel
    mesh,_,_,_=native_fixture()
    kernel=NativeShardDensityKernel(mesh,ns_radius_min=minimum,ns_radius_ratio=1.4,
        ns_deltaphi=1.,spi_num_vol=numerical,ns_delta_minor_rad=0.,injection_time_si=0.,toroidal_angles=(0.,))
    shards=dict(positions=np.array([[2.5,.5,0.]]),radius=np.array([.001]),
        ablation_atoms_s=np.array([1e22]),species=np.array([1.]),time_si=1.)
    return kernel,shards


def test_density_width_and_wrapped_toroidal_kernel_have_independent_ratios():
    kernel,s=source_fixture()
    xyz=np.array([[2.5,.5,0.],[2.55,.5,0.],[2.5,.5,2*np.pi-.25]])
    rates=kernel.rates_at(s,xyz,time_si=1.)
    # Analytic source shape is exp(-d^2/w^2), not exp(-d^2/(2*w^2)).
    assert rates[1]/rates[0]==pytest.approx(.36787944117144233,rel=1e-14)
    assert rates[2]/rates[0]==pytest.approx(.9394130628134758,rel=1e-14)
    other,_=source_fixture(.01)
    assert other.rates_at(s,xyz[:1],time_si=1.)[0]/rates[0]==pytest.approx(25.)
    assert kernel.rates_at(s,xyz,time_si=-1.).sum()==0.


def test_density_actual_plane_numerical_volume_and_unfiltered_projection_preserve_atoms():
    kernel,s=source_fixture(numerical=True)
    # Width=1 makes fixed four-point Gauss quadrature well resolved.
    s['radius'][:]=1/1.4;s['positions'][0,2]=np.pi/2
    with pytest.raises(ValueError,match='volume cache'):
        kernel.density_source(s)
    cached=kernel.refresh_volumes(s,time_si=1.,provenance='explicit post-commit diagnostic')
    out=kernel.density_source(cached)
    total=kernel.mesh.integrate(out.values.numpy())[1]*2*np.pi
    # Number rate 1e22/s / 8e19/m3 times 20u / central_mass times t_norm.
    assert total/kernel.mesh.units.t_norm_si==pytest.approx(2500.,rel=2e-13)
    assert not out.values[0].any()
    assert 'source_volume_m3' not in s
    # Finite phi plane at zero, not the continuum average, enters the volume.
    at_center=kernel.rates_at(cached,np.array([[2.5,.5,0.],[2.5,.5,np.pi/2]]),time_si=1.)
    assert at_center[0]/at_center[1]==pytest.approx(.08480497247111378,rel=2e-14)


@pytest.mark.parametrize('name,index,minimum',[
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400',4112,.05),
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',2711,.01)])
def test_real_snapshot_single_shard_volume_diagnostic(name,index,minimum):
    """Bounded read-only geometry audit, not production-cloud qualification."""
    import h5py,json,time,resource
    from pathlib import Path
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_state import NativeUnits
    from src.models.native_source_density import NativeShardDensityKernel
    path=Path(__file__).resolve().parents[1]/'data/EHL-2'/name/f'jorek{index:06d}.h5'
    if not path.exists():pytest.skip('local phaseA snapshot unavailable')
    started=time.perf_counter()
    with h5py.File(path,'r') as f:
        scalar=lambda key:float(np.ravel(f[key][()])[0])
        assert scalar('n_plane')==scalar('n_tor')==scalar('n_period')==1
        units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
        mesh=NativeSourceMesh(f['x'][()],f['size'][()],f['vertex'][()]-1,f['index'][()],units=units,projection='none')
        kernel=NativeShardDensityKernel(mesh,ns_radius_min=minimum,ns_radius_ratio=1.4,
            ns_deltaphi=1e3,spi_num_vol=True,ns_delta_minor_rad=0.,injection_time_si=8380*units.t_norm_si,
            toroidal_angles=(0.,))
        j=int(np.argmax(f['spi_abl_arr'][()]));time_si=scalar('t_now')*units.t_norm_si
        s=dict(positions=np.array([[f['spi_R_arr'][j],f['spi_Z_arr'][j],f['spi_phi_arr'][j]]]),
            radius=np.array([f['spi_radius_arr'][j]]),ablation_atoms_s=np.array([f['spi_abl_arr'][j]]),
            species=np.array([f['spi_species_arr'][j]]),time_si=time_si)
        recorded=float(f['spi_vol_arr'][j])
    cached=kernel.refresh_volumes(s,time_si=time_si,provenance=str(path)+' standalone diagnostic')
    recomputed=float(cached['source_volume_m3'][0])
    # Recorded spi_vol is an independent JOREK diagnostic of this kernel.
    assert recomputed==pytest.approx(recorded,rel=3e-11,abs=1e-20)
    print(json.dumps(dict(run=name,index=index,shard_slot=j,ns_radius_min=minimum,
        recorded_volume=recorded,recomputed_volume=recomputed,relative_difference=recomputed/recorded-1,
        elapsed_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='single_shard_snapshot_volume_only_no_cloud_no_rollout')))


@pytest.mark.parametrize('suffix,index,minimum',[('',4112,.05),('_injInner',5682,.01)])
def test_real_all_shard_volume_refresh_matches_recorded_diagnostic_with_bounded_cost(suffix,index,minimum):
    """Public diagnostic seam, independent saved Fortran volumes; no cloud trim."""
    import h5py,time
    from pathlib import Path
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_source_density import NativeShardDensityKernel
    path=Path(__file__).resolve().parents[1]/'data/EHL-2'/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)/f'jorek{index:06d}.h5'
    if not path.exists():pytest.skip('local phaseA snapshot unavailable')
    record=read_native_psi_teacher(path)
    mesh=NativeSourceMesh(record.coordinates,record.size,record.vertex,record.indices,
        units=record.units,projection='none',max_elements=5000)
    kernel=NativeShardDensityKernel(mesh,ns_radius_min=minimum,ns_radius_ratio=1.4,ns_deltaphi=1e3,
        spi_num_vol=True,ns_delta_minor_rad=0.,injection_time_si=8380*record.units.t_norm_si,toroidal_angles=(0.,))
    with h5py.File(path,'r') as f:
        s=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
            radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],species=f['spi_species_arr'][...])
        expected=f['spi_vol_arr'][...]
    start=time.perf_counter()
    result=kernel.refresh_volumes(s,time_si=record.time_si,provenance='full recorded volume comparison')
    seconds=time.perf_counter()-start
    assert len(result['source_volume_m3'])==846
    np.testing.assert_allclose(result['source_volume_m3'],expected,rtol=3e-11,atol=1e-20)
    assert 'source_volume_m3' not in s
    # Deliberately loose budget: original scalar implementation measured 32 s.
    # This is a local integration resource check, not an acceleration claim.
    assert seconds<15, f'complete 846-shard diagnostic took {seconds:.3f}s'


def test_repeated_birth_density_queries_keep_superposition_and_bounded_cost():
    """Births query one proposal at a time; do not alter that RNG protocol."""
    import time
    kernel,one=source_fixture()
    point=np.array([[2.5,.5,0.]])
    single=kernel.rates_at(one,point,time_si=1.)
    many={k:np.repeat(v,846,axis=0) if isinstance(v,np.ndarray) else v for k,v in one.items()}
    started=time.perf_counter()
    for _ in range(512):
        rate=kernel.rates_at(many,point,time_si=1.)
    seconds=time.perf_counter()-started
    np.testing.assert_allclose(rate,846*single,rtol=3e-14,atol=0.)
    assert seconds<3, f'512 complete 846-shard birth queries took {seconds:.3f}s'
