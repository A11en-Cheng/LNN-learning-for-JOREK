"""Model502 fixed Dirichlet update rows, distinct from particle projection BC."""
import numpy as np
import torch
from tests.test_native_field_step import make_step


def test_edge_direction_and_field_specific_dirichlet_rows_are_applied_to_updates():
    from src.models.native_field_boundary import NativeDirichletRows
    # bottom edge has nodes0,1 with code2; top is interior in this analytic fixture.
    bc=NativeDirichletRows(vertex=np.array([[0],[1],[2],[3]]),boundary=np.array([2,2,0,0]),
        indices=np.arange(1,17).reshape(4,4),freeboundary=False,RMP_on=False,no_mach1_bc=False,
        use_simple_bnd_types=False,bc_natural_open=True)
    u=torch.zeros((9,4,1,4),dtype=torch.float64)
    delta=torch.ones_like(u,requires_grad=True)
    out=bc(delta)
    # Bottom t constant: value + s derivative. model502 var_rhon=8
    # (mod_model_settings:39) includes rho_imp in apply_dirichlet_all.
    for f in range(9):
        assert not out[f,:2,0,:2].any()
        assert torch.all(out[f,2:,0,:2]==1)
    out.sum().backward()
    assert not delta.grad[0,:2,0,:2].any() and delta.grad[7].sum()>0
    assert bc.configuration['coverage']=='dirichlet_rows_only_open_boundary_unverified'


def test_joint_field_step_applies_rows_after_learning_without_clipping_other_degrees():
    from src.models.native_field_boundary import NativeDirichletRows
    from stencil_fixtures import structured_mesh
    model=make_step();vertex,_=structured_mesh(3,3)
    boundary=np.zeros(9,dtype=int);boundary[:3]=2
    rows=NativeDirichletRows(vertex=vertex,boundary=boundary,indices=model.update_mapping.indices.numpy(),
        freeboundary=False,RMP_on=False,no_mach1_bc=False,use_simple_bnd_types=False,bc_natural_open=True)
    model.update_mapping.boundary_rows=rows
    u=torch.zeros((9,4,1,9),dtype=torch.float64);d=torch.ones_like(u)*.17
    out=model(u,d,dt_si=.5,previous_dt_si=.5,scheme='Gears')
    assert torch.all(out.increment[rows.mask]==0)
    assert out.increment[~rows.mask].abs().sum()>0
    out.values.square().sum().backward()
    assert model.decoder.weight.grad.abs().sum()>0
    assert model.checkpoint_configuration['boundary_policy']=='dirichlet_subset_open_unverified'
