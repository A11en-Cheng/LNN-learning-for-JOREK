from copy import deepcopy

import numpy as np
import pytest

from tests.test_native_production_birth import birth_fixture
from tools.benchmark_native_birth_generation import benchmark_case, legacy_positions, probabilities_on_support


def inputs():
    births,s,_,_,_=birth_fixture()
    return births.kernel,s,dict(time_si=1e-9,bounds=np.array([[2.25,.25],[2.75,.75]]),
                               dv=births.dv,norm=2.5e22*births.kernel.mesh.units.t_norm_si/births.kernel.mesh.units.n_norm_m3)


def test_position_only_legacy_oracle_retains_known_stream():
    kernel,s,options=inputs()
    np.random.seed(2917)
    points,proposals=legacy_positions(kernel,s,options,2)
    np.testing.assert_array_equal(points,[[2.5280283400947785,.4969355326033923,5.845134906523379],
        [2.397521771355787,.46560221313850864,.028318653094736933]])
    assert proposals>2 and np.random.get_state()[2]==234


def test_benchmark_reports_generation_scope_refinement_and_restores_rng(monkeypatch):
    kernel,s,options=inputs()
    monkeypatch.setattr('tools.benchmark_native_birth_generation.load_snapshot',
        lambda path:(kernel,s,options,2,dict(snapshot=str(path))))
    before=deepcopy(np.random.get_state());original=deepcopy(s)
    result=benchmark_case('fixture',repeats=1)
    assert len(result['iterations'])==2
    assert result['refinement']['cell_probability_relative_l2']<.01
    assert result['refinement']['refined_support_relative_l2']<.02
    assert result['production_qualified'] is False
    assert 'no_mesh_no_field' in result['scope']
    assert result['iterations'][1]['source_proposals']==2
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    for k,v in s.items():
        if isinstance(v,np.ndarray):np.testing.assert_array_equal(original[k],v)


def test_legacy_probe_has_explicit_budget():
    kernel,s,options=inputs()
    np.random.seed(2917)
    with pytest.raises(RuntimeError,match='budget'):
        legacy_positions(kernel,s,options,2,max_proposals=2)


def test_exact_bilinear_integral_preserves_probability_on_own_support():
    from tests.test_native_birth_positions import prepare
    _,table,_,_=prepare()
    np.testing.assert_allclose(probabilities_on_support(table,table.r_edges,table.z_edges),
                               table.probabilities[:,:,0],rtol=3e-15,atol=1e-18)
