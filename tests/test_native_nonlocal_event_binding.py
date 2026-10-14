"""Exact digest-work regression at public shard.update; no real-cloud timing."""
from copy import deepcopy
import cProfile

import numpy as np
import pytest


def event_fixture(count=3):
    from tests.test_native_shard_dynamics import fixture
    from tests.test_native_gauss import native_coefficients
    from src.models.native_source_nonlocal import NativeFluxEquilibrium, NativeNonlocalDiagnostic
    dynamics, state, fields = fixture(True)
    fields[0, :, 0] = native_coefficients(lambda s,t:s, lambda s,t:1., lambda s,t:0., lambda s,t:0.)
    hot = fields.copy(); hot[8] *= 2
    def equilibrium(values, time):
        return NativeFluxEquilibrium.bind(values, time_si=time, psi_axis=0., psi_bnd=1.,
            lcfs_lost=False, xcase='none', psi_xpoints=(0.,0.), z_xpoints=(0.,0.),
            provenance='independent uniform hot-shell fixture')
    table = NativeNonlocalDiagnostic.build(dynamics.mesh, hot, np.zeros((5,4,1,4)),
        equilibrium(hot,0.), n_flux=2, n_table=10, feedback_time_si=0.,
        provenance='prior committed 4000eV cache; current local 2000eV')
    for name in ('positions','velocities','radius','ablation_atoms_s','injector_phi','species'):
        state[name] = np.repeat(state[name], count, axis=0)
    state['ids'] = np.arange(count)
    state['nonlocal_diagnostic'] = table
    state['old_equilibrium'] = equilibrium(fields,1.)
    return dynamics, state, fields


@pytest.mark.parametrize('count', [1, 3])
def test_update_hashes_old_field_and_mesh_once_per_event_not_per_shard(count):
    from src.models.native_state import configuration_digest
    dynamics,state,fields = event_fixture(count)
    profile = cProfile.Profile()
    out = profile.runcall(dynamics.update,state,fields,1.,1e-6)
    calls = sum(entry.callcount for entry in profile.getstats()
                if entry.code is configuration_digest.__code__)
    assert calls == 2, f'one field digest + one mesh digest per event; got {calls} for {count} shards'
    # Independent hot-shell Parks3 literal; local rocket remains at 2000eV.
    np.testing.assert_allclose(out['ablation_atoms_s'], 2.56560525337068085e24, rtol=4e-15)
    np.testing.assert_allclose(out['velocities'][:,0]-1e5, .026259335139022505, rtol=3e-10)


@pytest.mark.parametrize('seam', ['update','standalone_sample'])
@pytest.mark.parametrize('damage', ['old_field','mesh_configuration','future_table','wrong_equilibrium_time'])
def test_each_public_entry_rejects_changed_event_binding_without_state_or_rng_change(seam,damage):
    from dataclasses import replace
    from tools.diagnose_native_real_chain import _array_digest
    dynamics,state,fields = event_fixture()
    if damage == 'old_field': fields[8,0,0,0] += 1e-10
    elif damage == 'mesh_configuration': dynamics.mesh.configuration['event_binding_test'] = 'changed'
    elif damage == 'future_table': state['nonlocal_diagnostic']=replace(state['nonlocal_diagnostic'],time_si=2.)
    else: state['old_equilibrium']=replace(state['old_equilibrium'],time_si=2.)
    before = _array_digest((state,fields,np.random.get_state()))
    with pytest.raises(ValueError, match='OLD-field'):
        if seam == 'update': dynamics.update(state,fields,1.,1e-6)
        else:
            state['nonlocal_diagnostic'].sample(.35,.5,equilibrium=state['old_equilibrium'],
                old_fields=fields,time_si=1.,mesh=dynamics.mesh)
    assert _array_digest((state,fields,np.random.get_state())) == before


def test_update_preserves_input_rng_and_repeat_result_without_cross_event_reuse():
    from tools.diagnose_native_real_chain import _array_digest
    dynamics,state,fields = event_fixture()
    before = _array_digest((state,fields,np.random.get_state()))
    first = dynamics.update(state,fields,1.,1e-6)
    second = dynamics.update(state,fields,1.,1e-6)
    assert _array_digest(first) == _array_digest(second)
    assert _array_digest((state,fields,np.random.get_state())) == before
    assert first['nonlocal_diagnostic'].time_si == 0.
    # Captured before this optimization, including all arrays/cache metadata.
    assert _array_digest(first) == '3560c081c7e8d2e007b5c1a442740f0c14c773267407b8df590b1975caf1e7ce'
    changed = deepcopy(fields); changed[0,0,0,0] += 1e-10
    with pytest.raises(ValueError,match='OLD-field'):
        dynamics.update(state,changed,1.,1e-6)


def test_standalone_sample_revalidates_every_call_even_after_successful_update():
    from src.models.native_state import configuration_digest
    dynamics,state,fields=event_fixture()
    dynamics.update(state,fields,1.,1e-6)
    table=state['nonlocal_diagnostic']; eq=state['old_equilibrium']
    profile=cProfile.Profile()
    for _ in range(2):
        result=profile.runcall(table.sample,.35,.5,equilibrium=eq,old_fields=fields,time_si=1.,mesh=dynamics.mesh)
        assert result[0]==pytest.approx(4000.,rel=2e-15)
    calls=sum(entry.callcount for entry in profile.getstats() if entry.code is configuration_digest.__code__)
    assert calls==4  # Public sample never inherits update's local binding.
    dynamics.mesh.configuration['event_binding_test']='changed_after_success'
    with pytest.raises(ValueError,match='OLD-field'):
        table.sample(.35,.5,equilibrium=eq,old_fields=fields,time_si=1.,mesh=dynamics.mesh)
