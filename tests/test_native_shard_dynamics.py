"""Inspected pure-Ne rocket6 update, with independent force/erosion literals."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture


def fixture(nonlocal_abl=False):
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    mesh,_,_,u=native_fixture()
    # OLD uniform background: ne=1e20, Te=2000 eV, no impurity.
    u[:]=0.;u[4,0]=1.25;u[8,0]=2000*1.602176565e-19*1.2566370614359173e-6*8e19
    dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=4.3e28,F0=-1.,R_geo=3.,
        magnetic_reduction=False,nonlocal_abl=nonlocal_abl,ns_radius_min=.05,ns_radius_ratio=1.4)
    s=dict(positions=np.array([[2.25,.5,0.]]),velocities=np.array([[1e5,0.,0.]]),
        radius=np.array([.003]),ablation_atoms_s=np.array([4.863185427757e27]),
        injector_phi=np.array([0.]),species=np.ones(1),ids=np.array([17]),
        positions_coordinate_system='RZphi',velocities_coordinate_system='injector_R_Z_RxZ')
    return dynamics,s,u


def test_old_ablation_erosion_new_position_old_field_then_rocket6():
    dynamics,s,u=fixture();before=deepcopy(s)
    out=dynamics.update(s,u,0.,1e-6)
    np.testing.assert_allclose(out['positions'],[[2.35,.5,0.]],rtol=0,atol=5e-16)
    np.testing.assert_allclose(out['radius'],[.002],rtol=0,atol=1e-18)
    # P=100000 Pa, area=pi*(.002m)^2, mass=4*pi/3*(.002)^3*20*u*4.3e28.
    # Acceleration independently evaluated from these SI quantities.
    assert out['velocities'][0,0]-1e5==pytest.approx(.026259335139022505,rel=3e-10)
    # At radius .002m, ne=1e20, Te=2000: all Parks powers are exactly 1.
    assert out['ablation_atoms_s'][0]==pytest.approx(8.081150161106476e23,rel=3e-15)
    for name in ('positions','velocities','radius','ablation_atoms_s'):
        np.testing.assert_array_equal(s[name],before[name])
    assert out['ids'][0]==17


def test_nonlocal_requires_bound_spatial_diagnostic_and_failure_is_pure():
    dynamics,s,u=fixture(True);before=deepcopy(s)
    with pytest.raises(ValueError,match='nonlocal.*diagnostic'):
        dynamics.update(s,u,0.,1e-6)
    np.testing.assert_array_equal(s['positions'],before['positions'])
    np.testing.assert_array_equal(s['radius'],before['radius'])


def test_injector_velocity_rotation_uses_pre_move_angle():
    dynamics,s,u=fixture()
    s['positions'][0,2]=np.pi/2
    out=dynamics.update(s,u,0.,1e-6)
    # Old injector R velocity is locally toroidal here; explicit Euler uses old R.
    np.testing.assert_allclose(out['positions'][0],[2.25,.5,np.pi/2-1/22.5],rtol=0,atol=5e-16)
    assert out['velocities'][0,0]==1e5
    assert out['velocities'][0,2]==pytest.approx(.026259335139022505,rel=3e-15)


def test_mass_floor_does_not_replace_actual_force_area():
    dynamics,s,u=fixture();s['radius'][0]=5e-6;s['ablation_atoms_s'][0]=0.
    out=dynamics.update(s,u,0.,1e-6)
    # Area ratio=(5e-6/.002)^2, mass ratio=(1e-5/.002)^3: acceleration ratio=50.
    assert out['velocities'][0,0]-1e5==pytest.approx(1.3129667569511253,rel=1e-11)


def test_new_position_diagnostic_uses_old_field_spatial_values():
    from tests.test_native_gauss import native_coefficients
    dynamics,s,u=fixture()
    unit=2000*1.602176565e-19*1.2566370614359173e-6*8e19/2.35
    u[8,:,0,:]=native_coefficients(lambda a,b:unit*(2+a),lambda a,b:unit,
        lambda a,b:0.,lambda a,b:0.)
    out=dynamics.update(s,u,0.,1e-6)
    # New R=2.35 samples Te=2000; old R=2.25 would give a different ablation.
    assert out['ablation_atoms_s'][0]==pytest.approx(8.081150161106476e23,rel=3e-15)
