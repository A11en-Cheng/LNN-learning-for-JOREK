"""Public predicted-field equilibrium provider seam; analytic cubic oracle."""
import numpy as np
import pytest


def analytic_case():
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_state import NativeUnits
    x=np.zeros((2,4,1,8));u=np.zeros((9,4,1,8))
    for e,(lo,hi) in enumerate(((-1.5,0.),(0.,1.5))):
        ids=slice(4*e,4*e+4)
        r=np.array([1.,3.,3.,1.]);z=np.array([lo,lo,hi,hi])
        rs=np.array([2.,-2.,-2.,2.])/3
        zt=np.array([hi-lo,hi-lo,lo-hi,lo-hi])/3
        x[0,0,0,ids]=r;x[1,0,0,ids]=z
        x[0,1,0,ids]=rs;x[1,2,0,ids]=zt
        # psi=(R-2)^2+Z^2-2/3 |Z|^3: axis (2,0), saddles (2,+/-1).
        u[0,0,0,ids]=(r-2)**2+z*z-2/3*abs(z)**3
        u[0,1,0,ids]=2*(r-2)*rs
        u[0,2,0,ids]=(2*z-2*z*abs(z))*zt
    mesh=NativeSourceMesh(x,np.ones((4,4,2)),np.arange(8).reshape(2,4).T,
        np.arange(1,33).reshape(4,8),units=NativeUnits(1.,1e20,1.,5/3),projection='none')
    return mesh,u


def provider(mesh,**overrides):
    from src.models.native_equilibrium import NativeEquilibriumProvider
    options=dict(axis_seed=(2.,.05),axis_search_radius=.4,z_xpoint_limits=(-.5,.5),
        R_geo=2.,restart_lcfs_lost=False,restart_history_provenance='analytic initial epoch')
    options.update(overrides)
    return NativeEquilibriumProvider(mesh,np.ones(mesh.n_nodes,int),**options)


def test_current_cubic_axis_and_two_saddles_define_equilibrium():
    mesh,u=analytic_case();p=provider(mesh);eq=p(u,2.)
    assert eq.psi_axis==pytest.approx(0.,abs=1e-12)
    assert eq.psi_bnd==pytest.approx(1/3,abs=1e-12)
    np.testing.assert_allclose(eq.psi_xpoints,[1/3,1/3],atol=1e-12)
    np.testing.assert_allclose(eq.z_xpoints,[-1.,1.],atol=1e-10)
    assert eq.time_si==2. and eq.xcase=='double' and not eq.lcfs_lost
    assert p.configuration['future_fields']=='inaccessible_no_reader'


def test_predicted_psi_perturbation_moves_saddles_without_any_future_file(monkeypatch):
    import builtins
    import h5py
    mesh,u=analytic_case();p=provider(mesh);before=p(u,2.)
    original=u.copy();configuration=p.configuration
    def forbidden(*args,**kwargs):
        raise AssertionError('equilibrium cannot read initial or future files')
    monkeypatch.setattr(builtins,'open',forbidden)
    monkeypatch.setattr(h5py,'File',forbidden)
    predicted=u.copy()
    z=mesh.coordinates[1,0,0];zt=mesh.coordinates[1,2,0]
    predicted[0,0,0]+=.1*z*z+.25
    predicted[0,2,0]+=.2*z*zt
    after=p(predicted,3.)
    assert after.psi_axis==pytest.approx(.25,abs=1e-12)
    assert after.psi_bnd==pytest.approx(.25+1.331/3,abs=1e-12)
    np.testing.assert_allclose(after.z_xpoints,[-1.1,1.1],atol=1e-10)
    assert after.field_digest!=before.field_digest and after.time_si==3.
    assert p(u,2.)==before and p.configuration==configuration
    np.testing.assert_array_equal(u,original)


def test_changed_geometry_cannot_reuse_bound_configuration():
    mesh,u=analytic_case();p=provider(mesh)
    mesh.coordinates[0,0,0]+=.01
    with pytest.raises(ValueError,match='geometry changed'):
        p(u,3.)


def test_maximum_axis_and_explicit_restart_lcfs_are_reproducible():
    from src.models.native_state import configuration_digest
    mesh,u=analytic_case();u[0]*=-1
    p=provider(mesh,restart_lcfs_lost=True)
    eq=p(u,9.)
    assert eq.psi_axis==pytest.approx(0.,abs=1e-12)
    assert eq.psi_bnd==pytest.approx(-1/3,abs=1e-12) and eq.lcfs_lost
    np.testing.assert_array_equal(eq.normalized_flux([0.,.1]),[1.01,1.01])
    assert provider(mesh,restart_lcfs_lost=True)(u,9.)==eq
    assert configuration_digest(p.configuration)!=configuration_digest(provider(mesh).configuration)
    config=p.configuration;config['restart_lcfs_lost']=False
    assert p(u,10.).lcfs_lost  # Returned configuration cannot mutate epoch state.


@pytest.mark.parametrize('case',['linear','no_saddles','nonfinite','harmonics','time'])
def test_missing_topology_or_invalid_prediction_fails_instead_of_reusing_es(case):
    mesh,u=analytic_case();p=provider(mesh,max_attempts=2);before=p(u,1.)
    broken=u.copy();time=2.
    if case=='linear':broken[0]=mesh.coordinates[0]
    elif case=='no_saddles':
        r=mesh.coordinates[0,0,0];z=mesh.coordinates[1,0,0]
        broken[0]=0.
        broken[0,0,0]=(r-2)**2+z*z
        broken[0,1,0]=2*(r-2)*mesh.coordinates[0,1,0]
        broken[0,2,0]=2*z*mesh.coordinates[1,2,0]
    elif case=='nonfinite':broken[0,0,0,0]=np.nan
    elif case=='harmonics':broken=np.repeat(broken,3,axis=2)
    else:time=np.nan
    with pytest.raises(ValueError):p(broken,time)
    assert p(u,1.)==before


@pytest.mark.parametrize('options',[dict(freeboundary=True),dict(xcase='lower'),
    dict(restart_lcfs_lost=None),dict(restart_history_provenance='')])
def test_unsupported_or_missing_initialization_is_explicit(options):
    mesh,_=analytic_case()
    with pytest.raises(ValueError):provider(mesh,**options)


def test_initial_record_factory_reads_only_named_initial_file(tmp_path,monkeypatch):
    import hashlib
    import h5py
    from types import SimpleNamespace
    from src.models.native_equilibrium import NativeEquilibriumProvider
    mesh,u=analytic_case();reference=provider(mesh)
    path=tmp_path/'initial.h5'
    with h5py.File(path,'w') as f:
        f['R_axis_t']=[2.,2.];f['Z_axis_t']=[0.,0.]
        f['index_now']=[2];f['t_now']=[2.];f['t_norm']=[1.]
        f['freeboundary']=[b'F']
    initial=SimpleNamespace(source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        namelist_sha256='0fca2b6a5d0cf94e92eb86a4ef11d961fba50a28d06520708e19dfec3d061811',
        boundary=np.ones(mesh.n_nodes,int),geometry_digest=reference.configuration['geometry'],
        equilibrium=reference(u,2.),provenance='initial record fixture',time_si=2.,step_index=2,
        namelist_path='/nonexistent/not_to_be_read',lcfs_history_length=1)
    real_open=h5py.File
    def initial_only(filename,*args,**kwargs):
        assert str(filename)==str(path)
        return real_open(filename,*args,**kwargs)
    monkeypatch.setattr(h5py,'File',initial_only)
    p=NativeEquilibriumProvider.from_initial_record(mesh,initial)
    assert p.configuration['axis_seed']==[2.,0.]
    assert p.configuration['axis_search_radius']==.5
    assert p.configuration['z_xpoint_limits']==[-1.,1.]
    assert p.configuration['R_geo']==1.11
    initial.namelist_sha256='unknown'
    with pytest.raises(ValueError,match='namelist'):
        NativeEquilibriumProvider.from_initial_record(mesh,initial)
