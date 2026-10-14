import numpy as np

from tools.analyze_native_ionization_projection import attribute_pair


def row(norm,error,integral):
    return dict(channel='ionization_energy',reference_norm=norm,relative_l2=error,
                reference_integral=integral,integral_relative_error=0.)


def test_additive_attribution_uses_gauss_norm_not_coefficient_norm():
    background=np.array([3.,4.]);a=np.array([1.,0.]);b=np.array([0.,1.])
    result=attribute_pair([row(10.,.1,20.)],[row(2.,.5,4.)],background+a,background+b,a,b)
    assert result['background_identity_residual_relative']==0.
    channel=result['channels'][0]
    assert channel['coefficient_identity_residual_relative']==0.
    assert channel['newborn_norm_fraction']==.2
    assert channel['newborn_integral_fraction']==.2
    assert channel['reference_existing_integral']==16.
    assert channel['additive_prediction_relative_l2']==.1


def test_zero_reference_and_zero_difference_are_undefined():
    a=np.zeros(2)
    result=attribute_pair([row(0.,None,0.)],[row(0.,None,0.)],a,a,a,a)
    assert result['background_identity_status']=='undefined_zero_difference'
    assert result['channels'][0]['newborn_norm_fraction'] is None
    assert result['channels'][0]['coefficient_identity_residual_relative'] is None
    assert result['channels'][0]['newborn_integral_fraction'] is None
    assert result['channels'][0]['additive_prediction_relative_l2'] is None
