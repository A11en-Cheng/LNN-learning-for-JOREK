"""Deterministic nonlocal residual evidence, not a random birth accuracy gate."""
import numpy as np
import pytest


def test_rate_metrics_separate_zero_truth_and_release_weight_from_max_relative():
    from tools.diagnose_native_nonlocal_residual import rate_metrics
    result=rate_metrics(np.array([99.,20.,2.]),np.array([90.,10.,0.]),.25)
    assert result['max_relative_nonzero']==1.
    assert result['zero_truth_max_abs_atoms_s']==2.
    assert result['release_weighted_l1_relative']==pytest.approx(.21)
    assert result['signed_total_rate_relative']==pytest.approx(.21)
    assert result['reference_new_rate_integral_atoms']==25.
    assert result['predicted_new_rate_integral_atoms']==30.25
    assert result['affected_gt_1e6_relative_count']==2
    assert result['nonzero_truth_count']==2


@pytest.mark.parametrize('index',[5681,5682])
def test_real_full846_residual_is_removed_by_literal_fortran_cache_not_rng(index,tmp_path):
    """Public audit seam: frozen raw U/P/ES, future shard rate comparison only."""
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    run=root/'data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner'
    if not (run/f'jorek{index:06d}.h5').exists() or not shutil.which('gfortran'):
        pytest.skip('real raw HDF and independent Fortran compiler required')
    output=tmp_path/'residual.json'
    result=subprocess.run([sys.executable,str(root/'tools/diagnose_native_nonlocal_residual.py'),
        '--path',str(run/f'jorek{index:06d}.h5'),'--target',str(run/f'jorek{index+1:06d}.h5'),
        '--feedback-path',str(run/f'projections{index-1:06d}.h5'),
        '--fortran-oracle','--output',str(output)],capture_output=True,text=True,timeout=65)
    assert result.returncode==0,result.stderr
    evidence=json.loads(output.read_text())
    assert evidence['stop'] is None and evidence['returncode']==0,evidence['errors']
    residual=next(s for s in evidence['stages'] if s['stage']=='residual')
    assert all(residual['repeat_bitwise_equal'].values())
    assert residual['rng_used'] is False and residual['birth_executed'] is False
    assert residual['metrics']['shards']==846
    assert residual['metrics']['max_relative_nonzero']>1e-6  # Existing reconstruction repro.
    oracle=residual['probes']['independent_fortran_literal_cache']['metrics']
    assert oracle['max_relative_nonzero']<1e-6
    assert oracle['release_weighted_l1_relative']<1e-9
    assert oracle['affected_gt_1e6_relative_count']==0
