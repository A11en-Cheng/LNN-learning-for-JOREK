"""Projection/source advance compared to the retained dense reference."""
from copy import deepcopy
import numpy as np
from tests.test_native_coupling import native_fixture


def test_sparse_source_event_matches_dense_full_coefficients_and_history():
    from src.models.native_coupling import NativeSourceMesh
    mesh,source,state,u=native_fixture()
    sparse=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,
                            units=mesh.units,projection='sparse')
    alternate=deepcopy(source);alternate.mesh=sparse
    expected,_=source.advance(state,1e-9,u,u)
    actual,_=alternate.advance(state,1e-9,u,u)
    np.testing.assert_allclose(actual.feedback.values,expected.feedback.values,rtol=2e-12,atol=1e-20)
    np.testing.assert_allclose(actual.feedback.deltas,expected.feedback.deltas,rtol=2e-12,atol=1e-20)
    assert sparse.mass.nnz<=256 and sparse.global_basis is None
    np.testing.assert_allclose(sparse.integrate(actual.feedback.values)[4],1e12/8e19,rtol=2e-12)


def test_formula_only_coupled_field_needs_no_global_projection_matrix():
    import torch
    from src.models.native_coupling import NativeSourceMesh,TorchMarkerLoads,CoupledNativeField
    from src.models.native_source_features import SourceFeatureScales
    from tests.test_native_joint_source import joint_fixture
    model,mesh,_,_,u,d,kw=joint_fixture()
    local=NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,mesh.indices,units=mesh.units,projection='none')
    response=TorchMarkerLoads(local,parameters=model.response.parameters_physics,temperature_floor=1e-8)
    other=CoupledNativeField(model.field,response,source_configuration=model.source_configuration,
        shard_configuration=model.shard_configuration,source_scales=SourceFeatureScales.physical_reference())
    other.source_encoding.load_state_dict(model.source_encoding.state_dict())
    a=model(u,d,**kw);b=other(u,d,**kw)
    torch.testing.assert_close(a.increment,b.increment,rtol=1e-13,atol=1e-15)
    b.values.square().sum().backward()
    assert kw['marker_feedback'].grad.abs().sum()>0
    assert local.mass is None and local.global_basis is None


def test_shared_dof_sparse_projection_preserves_independent_rhs_channels():
    from src.models.native_coupling import NativeSourceMesh
    mesh,_,_,_=native_fixture()
    indices=mesh.indices.copy();indices[3,1]=indices[3,0]
    meshes=[NativeSourceMesh(mesh.coordinates,mesh.size,mesh.vertex,indices,units=mesh.units,projection=p)
            for p in ('dense','sparse')]
    rhs=np.random.default_rng(31).normal(size=(5,15))
    dense,sparse=[m.project(rhs) for m in meshes]
    np.testing.assert_allclose(sparse,dense,rtol=2e-11,atol=2e-10)
    np.testing.assert_array_equal(sparse[:,3,0,0],sparse[:,3,0,1])
    assert not np.array_equal(sparse[0],sparse[1])


def test_real_phase_a_shared_projection_constant_and_coordinate_fields():
    import h5py
    from pathlib import Path
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_state import NativeUnits
    for run,index in [('phaseA_dataset_1d22_ang15_massloss_neon_v400',4110),
                      ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',2703)]:
        with h5py.File(Path('data/EHL-2')/run/f'jorek{index:06d}.h5') as f:
            scalar=lambda k:float(np.asarray(f[k][()]).ravel()[0])
            units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
            mesh=NativeSourceMesh(f['x'][...],f['size'][...],f['vertex'][...]-1,f['index'][...],
                units=units,projection='sparse',max_elements=5000)
        assert mesh.n_dofs==16419 and mesh.mass.nnz<1000000
        rhs=mesh.weak_load(np.ones((len(mesh.gauss),16,1)))
        actual=mesh.project(rhs)
        expected=np.zeros_like(actual);expected[:,0]=1
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
        # R itself: physical derivatives identically [1,0], independent of map curvature.
        for e in range(0,len(mesh.gauss),101):
            v,g,rz,*_=mesh.point(e,np.array([.37,.61]),mesh.coordinates[:1])
            np.testing.assert_allclose(v,[rz[0]],rtol=0,atol=1e-14)
            np.testing.assert_allclose(g,[[1.,0.]],rtol=0,atol=1e-12)
