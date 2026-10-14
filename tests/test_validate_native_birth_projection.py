import numpy as np
from tools.validate_native_birth_projection import compact_metrics, ensemble_comparisons, gauss_values, main, metrics, rng_digest, saved_coefficients, write_json
from types import SimpleNamespace
import pytest


def test_zero_reference_is_explicitly_undefined():
    rows = metrics(np.zeros((5,3)),np.ones((5,3)),np.zeros(5),np.ones(5))
    assert all(r['relative_l2'] is None and r['integral_relative_error'] is None for r in rows)
    assert all(r['relative_l2_status']=='undefined_zero_reference' for r in rows)
    assert all(r['integral_absolute_error']==1 for r in rows)


def test_metrics_known_scale_and_constant_pearson():
    a = np.tile([1.,2.,3.],(5,1))
    rows = metrics(a,2*a,np.ones(5),2*np.ones(5))
    assert all(np.isclose(r['relative_l2'],1) and np.isclose(r['pearson'],1) for r in rows)
    assert all(r['integral_relative_error']==1 for r in rows)
    assert metrics(a,a*0,np.ones(5),np.zeros(5))[0]['pearson'] is None


def test_rng_digest_exact_replay_and_consumption():
    saved = np.random.get_state()
    try:
        np.random.seed(2917)
        before = rng_digest()
        np.random.random(5)
        after = rng_digest()
        np.random.seed(2917)
        assert rng_digest()==before and after!=before
    finally:
        np.random.set_state(saved)


def test_compact_summary_keeps_undefined_and_seed_count():
    channels = metrics(np.zeros((5,3)),np.ones((5,3)),np.zeros(5),np.ones(5))
    comparisons = [dict(other='new',scope='birth',representation='weak',seed=s,channels=channels) for s in (2917,2918)]
    result = compact_metrics(dict(anchor=4110,comparisons=comparisons,exact_replay={}))
    row = result['metrics'][0]
    assert row['seeds']==[2917,2918]
    assert row['relative_l2'] is None and row['relative_l2_undefined_count']==2
    assert row['integral_absolute_error']['mean']==1


def test_volume_weighted_metrics_use_quadrature_measure():
    a = np.tile([1.,2.],(5,1))
    b = np.tile([2.,2.],(5,1))
    rows = metrics(a,b,np.full(5,7.),np.full(5,8.),weights=np.array([1.,3.]))
    assert np.isclose(rows[0]['relative_l2'],1/np.sqrt(13))
    assert np.isclose(rows[0]['integral_relative_error'],1/7)


def test_gauss_evaluates_hermite_values_and_derivatives():
    basis = np.zeros((1,4,4,1))
    basis[0,0,0,0] = .5
    basis[0,1,2,0] = .25
    gauss = SimpleNamespace(test_basis=basis,weights=np.array([2.]),jacobian=np.array([3.]),coordinates=np.array([[4.,0.]]))
    mesh = SimpleNamespace(vertex=np.arange(4)[:,None],gauss=[gauss])
    coefficients = np.zeros((5,4,1,4))
    coefficients[:,0,0,0] = 2
    coefficients[:,2,0,1] = 4
    values,weights = gauss_values(mesh,coefficients)
    assert np.array_equal(values,np.full((5,1),2.))
    assert np.allclose(weights,[48*np.pi])


@pytest.mark.parametrize('seeds',['-1','4294967295','2917 2917'])
def test_invalid_seeds_fail_before_output_creation(tmp_path,monkeypatch,seeds):
    path = tmp_path/'must-not-exist'
    monkeypatch.setattr('sys.argv',['probe','--anchor','4110','--output-dir',str(path),'--seeds',*seeds.split()])
    with pytest.raises(SystemExit):main()
    assert not path.exists()


def test_birth_only_postprocess_archive(tmp_path):
    coefficients = np.zeros((5,4,1,2))
    np.savez(tmp_path/'new_2917_birth.npz',coeff=coefficients)
    assert np.array_equal(saved_coefficients(tmp_path,'new_2917','birth'),coefficients)


def test_ensemble_error_compares_mean_fields_not_mean_errors(tmp_path,monkeypatch):
    reference = np.tile([1.,2.],(5,1))
    comparisons = []
    for seed,offset in ((2917,1),(2918,-1)):
        np.savez(tmp_path/f'reference_{seed}.npz',**{f'reference_{seed}_whole_coeff':reference})
        np.savez(tmp_path/f'new_{seed}.npz',**{f'new_{seed}_whole_coeff':reference+offset})
        comparisons.append(dict(seed=seed,other='new',scope='whole',representation='coeff'))
    path = tmp_path/'summary.json'
    write_json(path,dict(initial={'cloud':'identical'},comparisons=comparisons))
    monkeypatch.setattr('tools.validate_native_birth_projection.gauss_values',lambda mesh,a:(a,np.ones(2)))
    result,skipped = ensemble_comparisons(None,[path])
    assert len(result)==1 and result[0]['ensemble_seeds']==[2917,2918]
    assert all(r['relative_l2']==0 for r in result[0]['channels'])
    assert skipped
