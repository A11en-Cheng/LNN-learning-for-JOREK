"""Exact birth-stream and public density oracles for event-local preparation."""
import cProfile
import pstats

import numpy as np
import pytest

from tests.test_native_production_birth import birth_fixture


def test_complete_birth_stream_prepares_shards_once_not_per_rejection():
    from src.models.native_state import configuration_digest
    births, shards, source, marker, fields = birth_fixture()
    marker.cloud.i_elm[:] = 0
    births.max_proposals = 100000
    np.random.seed(2917)
    profiler = cProfile.Profile()
    cloud = profiler.runcall(births.births, shards, marker, 2e-9, fields, fields, source)
    # Native mass-gate v2 stream. The independent Fortran birth test proves
    # why the historical number-converted stream cannot remain the oracle.
    np.testing.assert_array_equal(cloud.x, [[2.5280283400947785, .4969355326033923, 5.845134906523379],
                                           [2.397521771355787, .46560221313850864, .028318653094736933]])
    np.testing.assert_array_equal(cloud.weight, [12500000047104., 12500000047104.])
    assert configuration_digest(np.random.get_state()) == '187d3f940c6b3148c78721eca9c286ab3f4515b23dc4a5d5e233f60821b774ff'
    calls = sum(v[1] for (file, line, name), v in pstats.Stats(profiler).stats.items()
                if file.endswith('native_source_density.py') and name == 'validate')
    assert calls <= 6, f'one birth event validated the same shards {calls} times'


@pytest.mark.parametrize('average', [False, True])
def test_prepared_density_matches_independent_scalar_shard_order_and_owns_inputs(average):
    from math import pi
    from tests.test_native_production_density import source_fixture
    kernel, _ = source_fixture(numerical=True)
    random = np.random.RandomState(1701)
    n = 846  # Crosses every 256-shard block and the 32-point tile boundary.
    s = dict(positions=np.column_stack([random.uniform(2.3, 2.7, n), random.uniform(.3, .7, n), random.uniform(0, 2*pi, n)]),
        radius=random.uniform(.01, .1, n), ablation_atoms_s=random.uniform(0, 1e23, n), species=np.ones(n),
        source_volume_m3=random.uniform(.001, .1, n), source_volume_time_si=1., source_volume_provenance='independent test volume')
    s['radius'][::19] = 0
    s['ablation_atoms_s'][::23] = 0
    s['source_volume_m3'][::17] = 0
    points = np.column_stack([random.uniform(2.3, 2.7, 65), random.uniform(.3, .7, 65), random.uniform(0, 2*pi, 65)])
    expected = np.zeros(len(points))
    scale = kernel.mesh.units.t_norm_si/kernel.mesh.units.n_norm_m3*20/kernel.mesh.units.central_mass
    for j in range(n):
        if s['radius'][j] == 0 or s['ablation_atoms_s'][j] == 0:
            continue
        pos = s['positions'][j]
        width = max(s['radius'][j]*kernel.ratio, kernel.minimum)
        volume = s['source_volume_m3'][j]
        if volume == 0:
            volume = pi*pos[0]*kernel.toroidal_integral*width**2
        pol = np.exp(-np.sum((points[:, :2]-pos[:2])**2, axis=1)/width**2)
        phi = kernel.angles if average else points[:, 2]
        tor = np.exp(-(np.abs((phi-pos[2]+pi) % (2*pi)-pi)/kernel.deltaphi)**2)
        if average:
            tor = np.mean(tor)
        expected += (s['ablation_atoms_s'][j]*scale/volume)*pol*tor
    prepared = kernel.prepare_rates(s, time_si=1.)
    np.testing.assert_array_equal(prepared.rates_at(points, toroidal_average=average), expected)
    np.testing.assert_array_equal(kernel.rates_at(s, points, time_si=1., toroidal_average=average), expected)
    for value in s.values():
        if isinstance(value, np.ndarray):
            value[:] = 0
    kernel.angles[:] = 2.; kernel.deltaphi = .3; kernel.minimum = .7; kernel.ratio = 3.
    np.testing.assert_array_equal(prepared.rates_at(points, toroidal_average=average), expected)
    with pytest.raises(ValueError):
        kernel.rates_at(s, points, time_si=1.)


def test_public_density_preserves_preinjection_zero_and_volume_input_guards():
    from tests.test_native_production_density import source_fixture
    kernel, s = source_fixture(numerical=True)
    points = np.array([[2.5, .5, 0.]])
    # Missing numerical volume is legal before/on injection, not afterwards.
    np.testing.assert_array_equal(kernel.rates_at(s, points, time_si=0.), [0.])
    np.testing.assert_array_equal(kernel.prepare_rates(s, time_si=-1.).rates_at(points), [0.])
    with pytest.raises(ValueError, match='volume cache'):
        kernel.rates_at(s, points, time_si=1.)
    s.update(source_volume_m3=np.array([1.]), source_volume_time_si=2., source_volume_provenance='future')
    with pytest.raises(ValueError, match='volume cache'):
        kernel.rates_at(s, points, time_si=1.)
    with pytest.raises(ValueError, match='finite'):
        kernel.rates_at(s, np.array([[np.nan, 0., 0.]]), time_si=0.)
