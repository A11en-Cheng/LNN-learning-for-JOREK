"""No training: persistent clocks and noisy observations at actual field seam."""
import numpy as np
import torch
from copy import deepcopy
from tests.test_native_joint_source import joint_fixture


def test_physical_clock_origin_is_persistent_and_checkpoint_bound():
    from src.models.native_source_features import SourceFeatureScales,SourceEventTime
    from src.models.native_coupling import CoupledNativeField
    model,mesh,_,_,u,d,k=joint_fixture()
    scale=SourceFeatureScales.anchored_physical_reference(units=mesh.units,
        origin_si=.0035,gather_scale_si=1e-6,event_scale_si=1e-3,
        origin_provenance='declared_injection_onset_from_external_configuration')
    m=CoupledNativeField(model.field,model.response,source_configuration=model.source_configuration,
        shard_configuration=model.shard_configuration,source_scales=scale)
    time=deepcopy(k['source_time']);object.__setattr__(time,'event_time_si',.003514)
    from dataclasses import replace
    kw={**k,'source_time':time,'field_event':replace(k['field_event'],field_time_si=.003514,solve_time_si=.003514)}
    a=m(u,d,**kw)
    assert a.source_features[0,99].item()==pytest.approx(.014)
    assert m.checkpoint_configuration['source_encoding']['time_origin_si']==.0035
    b=deepcopy(m)(u,d,**kw)
    torch.testing.assert_close(a.values,b.values,rtol=0,atol=0)


def test_observation_changes_features_without_replacing_identity_or_raw_history():
    model,_,_,_,u,d,k=joint_fixture()
    observed=u.clone();observed[8,1,0,0]+=.01
    clean=model(u,d,**k)
    noisy=model(u,d,observation_values=observed,**k)
    assert not torch.equal(clean.source_features,noisy.source_features)
    assert not torch.equal(clean.increment,noisy.increment)
    with torch.no_grad():model.field.decoder.weight.zero_();model.field.decoder.bias.zero_()
    held=model(u,d,observation_values=observed,**k)
    torch.testing.assert_close(held.values,u,rtol=0,atol=0)
    assert not torch.equal(observed,u)
    # No-op must preserve the clean complete physical state even with noise.
    out=model(u,d,observation_values=observed,**{**k,'dt_si':0.,'field_event':
        __import__('dataclasses').replace(k['field_event'],phase='zero_time_probe')})
    torch.testing.assert_close(out.values,u,rtol=0,atol=0)
    torch.testing.assert_close(out.fluid_deltas,d,rtol=0,atol=0)


import pytest


def test_actual_adapter_noise_preserves_clean_source_state_and_rng_on_failure(tmp_path):
    from tests.test_base_coupled_training import event_fixture,sequence
    from src.training.base_coupled import TorchRecomputedSourceEvents,run_sequence,save_session,load_session
    from src.training.base_coupled_noise import NativeObservationNoise
    model,source,shard,runtime=event_fixture()
    basis=np.zeros((1,9,4,1,4));basis[0,8,1,0,0]=1.
    config=dict(amplitude=1e-3,native_scales=np.ones((9,4,1,1)),
        scale_provenance='fixture_explicit_native_units',correlation_provenance='fixed_rank1',
        temporal_policy='independent_observation_draw_each_field_event',support_provenance='finite_fixture_fields')
    noise=NativeObservationNoise(basis=basis,constrained=np.zeros((4,1,4),bool),
        indices=model.field.update_mapping.indices.numpy(),configuration=config,
        support_validator=lambda a:bool(torch.isfinite(a).all()))
    adapter=TorchRecomputedSourceEvents(model,source,shard,observation_noise=noise)
    first=run_sequence(sequence(),adapter,adapter.initialize(runtime),stop_cursor=2)
    save_session(tmp_path/'noise.pt',sequence(),adapter,first)
    tail=run_sequence(sequence(),adapter,session=first)
    fresh=TorchRecomputedSourceEvents(deepcopy(model),deepcopy(source),deepcopy(shard),observation_noise=deepcopy(noise))
    restored=load_session(tmp_path/'noise.pt',sequence(),fresh)
    resumed=run_sequence(sequence(),fresh,session=restored)
    torch.testing.assert_close(resumed.state.values,tail.state.values,rtol=0,atol=0)
    # Failure after the random observation draw must restore RNG and leave input clean.
    bad=deepcopy(shard)
    def fail(*args):raise RuntimeError('shard after noisy field failure')
    bad.update=fail
    broken=TorchRecomputedSourceEvents(model,source,bad,observation_noise=noise)
    state=broken.initialize(runtime);rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='shard after'):broken.advance(state,sequence().controls[0])
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    np.testing.assert_array_equal(runtime.fields_current,state.runtime.fields_current)


def test_corrupt_checkpoint_rng_rolls_back_model_and_global_rng(tmp_path):
    import random
    from tests.test_base_coupled_training import event_fixture,sequence
    from src.training.base_coupled import TorchRecomputedSourceEvents,run_sequence,save_session,load_session
    model,source,shard,state=event_fixture(n=2)
    adapter=TorchRecomputedSourceEvents(model,source,shard);seq=sequence(n=2,burnin=0)
    session=run_sequence(seq,adapter,adapter.initialize(state),stop_cursor=0)
    path=tmp_path/'state.pt';save_session(path,seq,adapter,session)
    payload=torch.load(path,weights_only=False);payload['rng']['python']=random.Random(12345).getstate();payload['rng']['torch']='corrupt';torch.save(payload,path)
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();py=random.getstate()
    with pytest.raises((TypeError,ValueError,RuntimeError)):load_session(path,seq,adapter)
    assert torch.equal(torch.get_rng_state(),rng) and random.getstate()==py
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
