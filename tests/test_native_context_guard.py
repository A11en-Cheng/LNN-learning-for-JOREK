"""Public coupled solver context defenses; no optimizer or training."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_joint_source import joint_fixture
from src.models.native_state import NativeStateContext


def coupled_context(model, mesh, parameters):
    return NativeStateContext(mesh.units,(0,),dict(
        mesh={'x':mesh.coordinates.copy(),'size':mesh.size.copy(),
              'vertex':mesh.vertex.copy()+1,'index':model.field.update_mapping.indices.cpu().numpy().copy()},
        boundary={'policy':'learned_unverified'},
        physics={'feature_parameter_rows':list(parameters.rows)},
        numerics={'scheme':'Gears'},model=deepcopy(model.checkpoint_configuration)))


@pytest.mark.parametrize('damage',['position','size','indices','units','boundary','model','parameter_source'])
def test_coupled_public_solver_rejects_incompatible_context(damage):
    from dataclasses import replace
    model,mesh,_,_,_,_,kw=joint_fixture()
    context=coupled_context(model,mesh,kw['feature_parameters'])
    if damage=='position':context.configuration['mesh']['x'][0,0,0,0]+=.1
    if damage=='size':context.configuration['mesh']['size'][0,0,0]*=2
    if damage=='indices':context.configuration['mesh']['index'][0,0]+=100
    if damage=='units':context.units=replace(mesh.units,n_norm_m3=mesh.units.n_norm_m3*2)
    if damage=='boundary':context.configuration['boundary']['policy']='physical'
    if damage=='model':context.configuration['model']['hidden']+=1
    if damage=='parameter_source':context.configuration['physics']['feature_parameter_rows']=None
    with pytest.raises(ValueError):model.reference_solver(context)


def test_coupled_public_solver_accepts_full_context_without_mutating_it():
    model,mesh,_,_,_,_,kw=joint_fixture()
    context=coupled_context(model,mesh,kw['feature_parameters'])
    original=deepcopy(context.configuration)
    solver=model.reference_solver(context)
    from src.models.native_state import configuration_digest
    assert configuration_digest(original)==configuration_digest(context.configuration)
    assert solver.model is model

@pytest.mark.parametrize('damage',['missing','stale','future','restart','zero_probe'])
def test_joint_torch_entry_binds_source_to_field_event(damage):
    from src.models.native_source_features import FieldEventContext
    model,_,_,_,u,d,kw=joint_fixture()
    time=kw['source_time'].event_time_si
    if damage=='missing':kw.pop('field_event',None)
    else:
        phase={'restart':'registered_restart','zero_probe':'zero_time_probe'}.get(damage,'ordinary')
        offset={'stale':1e-8,'future':-1e-8}.get(damage,0.)
        kw['field_event']=FieldEventContext(time+offset,time+offset,phase,'explicit_test_state',kw['source_time'].gather_time_si)
    with pytest.raises(ValueError):model(u,d,**kw)


def test_solver_rejects_context_drift_before_evaluating_field():
    model,mesh,_,_,u,d,kw=joint_fixture()
    solver=model.reference_solver(coupled_context(model,mesh,kw['feature_parameters']))
    solver.context.configuration['boundary']['unreviewed_option']=True
    with pytest.raises(ValueError,match='context configuration changed'):
        solver(None,None,None,None,None,fluid_deltas=None,time_coefficients=None)


def test_ordinary_context_rejects_inconsistent_actual_gather():
    from dataclasses import replace
    model,_,_,_,u,d,kw=joint_fixture()
    kw['source_time']=replace(kw['source_time'],gather_time_si=kw['source_time'].gather_time_si*2)
    with pytest.raises(ValueError,match='gather'):model(u,d,**kw)
