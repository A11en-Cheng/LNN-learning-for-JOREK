"""Full real initialization must not open future fields in autonomous mode."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_autonomous_real_builder_needs_no_future_field_file():
    from tools.native_real_chain_builder import build_real
    from tools.diagnose_native_real_chain import Stages
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')
    controls=[dict(source='/nonexistent/future-field.h5',dt_si=2.0428876129759165e-8)]
    args=SimpleNamespace(chunk_size=256,wall_seconds=900.,memory_mib=8192,birth_max_proposals=100000)
    model,source,shard,runtime,provider=build_real(run,4110,controls,Stages(),args,
        feedback_mode='autonomous')
    assert 'psi_teacher' not in shard.configuration
    assert provider.configuration['future_fields']=='inaccessible_no_reader'
    predicted=runtime.fields_current.copy()
    predicted[0]*=1.01
    np.testing.assert_array_equal(shard.diagnostic_fields(predicted,runtime.fluid_time_si),predicted)
    before=provider(runtime.fields_current,runtime.fluid_time_si)
    after=provider(predicted,runtime.fluid_time_si)
    np.testing.assert_allclose(after.psi_axis,before.psi_axis*1.01,rtol=1e-9)
    assert model.shard_configuration==shard.configuration
    assert shard.birth.position_sampler.cells_per_width==64
    assert shard.birth.position_sampler.max_cells==16_000_000
