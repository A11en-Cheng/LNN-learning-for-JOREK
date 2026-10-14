"""Source diagnostic/update seams, with analytic shell and OLD-field oracles."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture
from tests.test_native_gauss import native_coefficients
from src.models.marker_feedback import EL_CHG, MU_ZERO


def test_nonlocal_source_uses_electron_weighted_temperature_and_full_geometry():
    from src.models.native_source_nonlocal import NativeFluxEquilibrium, NativeNonlocalDiagnostic
    mesh, _, _, u = native_fixture()
    u[:] = 0
    u[0, :, 0] = native_coefficients(lambda s,t:s, lambda s,t:1., lambda s,t:0., lambda s,t:0.)
    u[4, :, 0] = native_coefficients(lambda s,t:1+t, lambda s,t:0., lambda s,t:1., lambda s,t:0.)
    factor = 3600 * EL_CHG * MU_ZERO * mesh.units.n_norm_m3
    u[8, :, 0] = native_coefficients(lambda s,t:factor*t, lambda s,t:0., lambda s,t:factor, lambda s,t:0.)
    eq = NativeFluxEquilibrium.bind(u, time_si=1., psi_axis=0., psi_bnd=1.,
        lcfs_lost=False, xcase='none', psi_xpoints=(0.,0.), z_xpoints=(0.,0.), provenance='analytic rectangle')
    table = NativeNonlocalDiagnostic.build(mesh, u, np.zeros((5,4,1,4)), eq,
        n_flux=2, n_table=10, feedback_time_si=1., provenance='analytic diagnostic event')
    # ne=1+Z. Integral ne*T / integral ne = 3600*(1/2+1/3)/(1+1/2)=2000 eV.
    np.testing.assert_allclose(table.temperature_ev, 2000., rtol=2e-15)
    np.testing.assert_allclose(table.electron_density_m3, 1.2e20, rtol=2e-15)
    assert table.shell_volume_m3.sum() == pytest.approx(5*np.pi, rel=2e-15)
    assert table.field_digest == eq.field_digest and table.time_si == 1.


def test_nonlocal_shard_uses_cached_hot_shell_at_new_position_in_old_field():
    from src.models.native_source_nonlocal import NativeFluxEquilibrium, NativeNonlocalDiagnostic
    from tests.test_native_shard_dynamics import fixture
    dynamics,s,u=fixture(True)
    u[0,:,0]=native_coefficients(lambda a,b:a,lambda a,b:1.,lambda a,b:0.,lambda a,b:0.)
    hot=u.copy();hot[8]*=2
    def eq(fields,time):
        return NativeFluxEquilibrium.bind(fields,time_si=time,psi_axis=0.,psi_bnd=1.,lcfs_lost=False,
            xcase='none',psi_xpoints=(0.,0.),z_xpoints=(0.,0.),provenance='explicit ES event')
    table=NativeNonlocalDiagnostic.build(dynamics.mesh,hot,np.zeros((5,4,1,4)),eq(hot,0.),
        n_flux=2,n_table=10,feedback_time_si=0.,provenance='prior committed hot diagnostic')
    s['nonlocal_diagnostic']=table;s['old_equilibrium']=eq(u,1.)
    before=deepcopy(s)
    out=dynamics.update(s,u,1.,1e-6)
    # Cached 4000 eV doubles Te, giving Parks3 multiplier 2**(5/3).
    assert out['ablation_atoms_s'][0]==pytest.approx(2.56560525337068085e24,rel=4e-15)
    # Rocket pressure still uses local 2000 eV, not hot nonlocal Te.
    assert out['velocities'][0,0]-1e5==pytest.approx(.026259335139022505,rel=3e-10)
    np.testing.assert_array_equal(s['radius'],before['radius'])
    # The cache is retained, not silently refreshed in update.
    assert out['nonlocal_diagnostic'].time_si==0.


def test_missing_shell_or_current_equilibrium_is_not_local_fallback():
    from src.models.native_source_nonlocal import NativeFluxEquilibrium, NativeNonlocalDiagnostic
    mesh,_,_,u=native_fixture();u[0]=0
    eq=NativeFluxEquilibrium.bind(u,time_si=0.,psi_axis=0.,psi_bnd=1.,lcfs_lost=False,
        xcase='none',psi_xpoints=(0.,0.),z_xpoints=(0.,0.),provenance='one shell only')
    with pytest.raises(ValueError,match='empty shell'):
        NativeNonlocalDiagnostic.build(mesh,u,np.zeros((5,4,1,4)),eq,
            n_flux=2,n_table=10,feedback_time_si=0.,provenance='missing shell')


def test_postcommit_nonlocal_diagnostic_uses_installed_lagged_feedback_charge():
    from src.models.native_source_nonlocal import NativeFluxEquilibrium, NativeNonlocalDiagnostic
    mesh,_,_,u=native_fixture();u[:]=0
    u[0,:,0]=native_coefficients(lambda a,b:a,lambda a,b:1.,lambda a,b:0.,lambda a,b:0.)
    u[4,0]=2.;u[7,0]=1.;u[8,0]=2000*EL_CHG*MU_ZERO*mesh.units.n_norm_m3
    p=np.zeros((5,4,1,4));p[3,0]=2.;p[4,0]=.5  # mean charge4
    eq=NativeFluxEquilibrium.bind(u,time_si=2.,psi_axis=0.,psi_bnd=1.,lcfs_lost=False,
        xcase='none',psi_xpoints=(0.,0.),z_xpoints=(0.,0.),provenance='postcommit ES')
    table=NativeNonlocalDiagnostic.build(mesh,u,p,eq,n_flux=2,n_table=10,
        feedback_time_si=1.,provenance='installed feedback S1 with committed U2')
    # ne=2+(1/20*2/2.01455279245*4-1)*1, independent Decimal literal.
    np.testing.assert_allclose(table.electron_density_m3,9.588441867591028689e19,rtol=2e-15)
    assert table.feedback_time_si==1. and table.time_si==2.


def test_nonlocal_lookup_axisward_index_private_flux_and_old_field_binding():
    from dataclasses import replace
    from src.models.native_source_nonlocal import NativeFluxEquilibrium,NativeNonlocalDiagnostic
    mesh,_,_,u=native_fixture();u[:]=0.;u[4,0]=1.
    u[0,:,0]=native_coefficients(lambda a,b:1-a,lambda a,b:-1.,lambda a,b:0.,lambda a,b:0.)
    u[8,0]=2000*EL_CHG*MU_ZERO*mesh.units.n_norm_m3
    eq=NativeFluxEquilibrium.bind(u,time_si=1.,psi_axis=1.,psi_bnd=0.,lcfs_lost=False,
        xcase='double',psi_xpoints=(0.,0.),z_xpoints=(-.25,1.25),provenance='decreasing psi rectangle')
    table=NativeNonlocalDiagnostic.build(mesh,u,np.zeros((5,4,1,4)),eq,n_flux=2,n_table=10,
        feedback_time_si=0.,provenance='analytic decreasing cache')
    # Synthetic table entries are an independent oracle for the source's discrete
    # lookup: psi=.66 is nearer .7 than .6, then moves to the axisward/lower index.
    table=replace(table,temperature_ev=np.arange(10,dtype=float)*1000+1000)
    te,ne=table.sample(.66,.5,equilibrium=eq,old_fields=u,time_si=1.,mesh=mesh)
    assert te==4000. and ne==pytest.approx(8e19)
    assert table.sample(.8,-.5,equilibrium=eq,old_fields=u,time_si=1.,mesh=mesh)==(0.,0.)
    with pytest.raises(ValueError,match='OLD-field'):
        table.sample(.66,.5,equilibrium=eq,old_fields=u+1e-10,time_si=1.,mesh=mesh)
    assert eq.normalized_flux(.8,-.5)==pytest.approx(1.8)
    assert replace(eq,lcfs_lost=True).normalized_flux(.8,.5)==1.01
