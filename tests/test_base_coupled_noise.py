"""Noise is an observation, with exact compatible gradients from native jets."""
import numpy as np
import torch
import pytest


def test_correlated_native_observation_recomputes_gradients_and_preserves_state_history_boundary():
    from src.training.base_coupled_noise import observe_native_noise
    from tests.test_native_gauss import rectangle_geometry,native_coefficients
    from src.models.native_features import TorchNativeGauss
    from tests.test_native_coupling import native_fixture
    units=native_fixture()[0].units;g=rectangle_geometry()
    ev=TorchNativeGauss(g.coordinates,g.size[:,:,None],np.arange(4)[:,None],units=units)
    u=torch.zeros((9,4,1,4),dtype=torch.float64);raw=torch.full_like(u,.17)
    basis=torch.zeros((1,9,4,1,4),dtype=u.dtype)
    # Smooth spatial correlation: delta psi=R-2=s, gradient [1,0].
    basis[0,0,:,0]=torch.tensor(native_coefficients(lambda s,t:s,lambda s,t:1.,lambda s,t:0.,lambda s,t:0.))
    config=dict(amplitude=.2,native_scales=torch.ones((9,4,1,1)),scale_provenance='fixture_explicit_native_units',
        correlation_provenance='one_rank_linear_R_field; fixed_coefficients_for_oracle',temporal_policy='explicit_history_correlated_draw')
    out=observe_native_noise(u,raw,evaluator=ev,units=units,basis=basis,coefficients=torch.tensor([2.]),
        constrained=torch.zeros((4,1,4),dtype=torch.bool),indices=np.arange(16).reshape(4,4),
        configuration=config,support_validator=lambda a:bool(torch.isfinite(a).all()),closure=lambda q:q.values.square())
    torch.testing.assert_close(out.gradients[:,:,0,0],torch.full((1,16),.4,dtype=u.dtype),rtol=0,atol=1e-14)
    torch.testing.assert_close(out.gradients[:,:,0,1],torch.zeros((1,16),dtype=u.dtype),rtol=0,atol=1e-14)
    assert torch.equal(out.raw_deltas,raw) and not u.any()
    torch.testing.assert_close(out.closure,out.gauss_values.square())
    # Constrain every coefficient: no perturbation; labels/committed state unchanged.
    blocked=observe_native_noise(u,raw,evaluator=ev,units=units,basis=basis,coefficients=torch.tensor([2.]),
        constrained=torch.ones((4,1,4),dtype=torch.bool),indices=np.arange(16).reshape(4,4),
        configuration=config,support_validator=lambda a:True,closure=lambda q:q.values.square())
    assert not blocked.values.any()


def test_noise_obeys_shared_boundary_aliases_and_refuses_unsupported_observations():
    from src.training.base_coupled_noise import observe_native_noise
    from tests.test_native_coupling import native_fixture
    from src.models.native_features import TorchNativeGauss
    mesh,_,_,u=native_fixture()
    ev=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    values=torch.tensor(u);raw=torch.zeros_like(values)
    basis=torch.ones((1,*values.shape),dtype=values.dtype)
    indices=np.arange(16).reshape(4,4);indices[0,1]=indices[0,0]
    constrained=torch.zeros((4,1,4),dtype=torch.bool);constrained[0,0,0]=True
    kwargs=dict(evaluator=ev,units=mesh.units,basis=basis,coefficients=torch.ones(1),
        constrained=constrained,indices=indices,closure=lambda q:q.values,
        configuration=dict(amplitude=.1,native_scales=np.ones((9,4,1,1)),scale_provenance='fixture_native_units',
            correlation_provenance='one_rank_spatially_constant',temporal_policy='explicit_draw'))
    out=observe_native_noise(values,raw,support_validator=lambda a:True,**kwargs)
    torch.testing.assert_close(out.values[:,0,0,:2],values[:,0,0,:2],rtol=0,atol=0)
    assert not torch.equal(out.values[:,0,0,2:],values[:,0,0,2:])
    with pytest.raises(ValueError,match='physical support'):
        observe_native_noise(values,raw,support_validator=lambda a:False,**kwargs)


def test_field_specific_constraints_preserve_other_fields_and_shared_aliases():
    from src.training.base_coupled_noise import observe_native_noise
    from tests.test_native_coupling import native_fixture
    from src.models.native_features import TorchNativeGauss
    mesh,_,_,u=native_fixture();values=torch.tensor(u);raw=torch.zeros_like(values)
    indices=np.arange(16).reshape(4,4);indices[0,1]=indices[0,0]
    constrained=torch.zeros_like(values,dtype=torch.bool);constrained[8,0,0,0]=True
    out=observe_native_noise(values,raw,evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units),
        units=mesh.units,basis=torch.ones((1,*values.shape)),coefficients=torch.ones(1),
        constrained=constrained,indices=indices,closure=lambda q:q.values,support_validator=lambda a:True,
        configuration=dict(amplitude=.1,native_scales=np.ones((9,4,1,1)),scale_provenance='explicit_fixture',
        correlation_provenance='constant_rank1',temporal_policy='explicit_draw'))
    torch.testing.assert_close(out.values[8,0,0,:2],values[8,0,0,:2],rtol=0,atol=0)
    torch.testing.assert_close(out.values[7,0,0,:2]-values[7,0,0,:2],torch.full((2,),.1,dtype=values.dtype))
