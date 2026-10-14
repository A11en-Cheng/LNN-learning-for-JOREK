from copy import deepcopy

import numpy as np

from tests.test_validate_native_birth_distribution import inputs
from tools.validate_native_birth_refinement import audit_case


def test_refinement_uses_independent_truth_tracks_cost_and_seed_without_global_mutation(monkeypatch):
    kernel,shards,options=inputs()
    monkeypatch.setattr('tools.validate_native_birth_refinement.load_snapshot',
        lambda path:(kernel,shards,options,10,dict(snapshot=str(path))))
    rng=deepcopy(np.random.get_state())
    result=audit_case('fixture',resolutions=(8,64,128),bins=24,target=1e-4,seed=13)
    rows=result['resolution_sweep']
    assert rows[-1]['relative_l2']<rows[1]['relative_l2']<rows[0]['relative_l2']
    assert rows[-1]['empirical_target_met']
    assert rows[-1]['diagnostics']['owned_table_bytes']>rows[0]['diagnostics']['owned_table_bytes']
    assert all(r['build_seconds']>0 and r['same_seed_bitwise_positions'] for r in rows)
    assert result['projection_error_measured'] is False
    assert result['reference']['method']=='analytic_unsaturated_cylindrical_Gaussian_integrals'
    np.testing.assert_array_equal(rng[1],np.random.get_state()[1])
    assert rng[2:]==np.random.get_state()[2:]
