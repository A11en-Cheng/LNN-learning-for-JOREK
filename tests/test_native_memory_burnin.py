import pytest
import torch
from tests.test_base_coupled_training import event_fixture,sequence
from src.training.base_coupled import TorchRecomputedSourceEvents
from tools.native_memory_burnin import make_prefix,forecast
from tools.diagnose_native_real_chain import _array_digest


def test_burnin_candidates_share_forecast_clock_preserve_complete_cache_and_gradients():
    model,source,shard,runtime=event_fixture(4)
    adapter=TorchRecomputedSourceEvents(model,source,shard)
    prefix=make_prefix(adapter,runtime,sequence().controls,2)
    before=_array_digest(prefix.states)
    for b in (0,1,2):
        predictions,proof=forecast(adapter,prefix,burnin=b,horizon=2,grad=True)
        assert proof['forecast_start_si']==pytest.approx(3e-9)
        assert proof['forecast_end_si']==pytest.approx(5e-9)
        assert proof['burnin_steps']==b and proof['full_prefix_support_steps']==2
        assert len(predictions)==2 and predictions[-1]['field']['values'].requires_grad
        assert _array_digest(prefix.states)==before
        predictions[-1]['field']['values'].square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
        model.zero_grad(set_to_none=True)


def test_burnin_rejects_stale_weight_cache():
    model,source,shard,runtime=event_fixture(4)
    adapter=TorchRecomputedSourceEvents(model,source,shard)
    prefix=make_prefix(adapter,runtime,sequence().controls,2)
    with torch.no_grad():next(model.parameters()).add_(.001)
    with pytest.raises(ValueError,match='stale'):
        forecast(adapter,prefix,burnin=1,horizon=2)
