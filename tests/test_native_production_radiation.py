"""Source seam: charge-indexed PLT+PRB, no production-cloud certification."""
from copy import deepcopy
import numpy as np
import pytest
from src.models.native_coupling import NativeNeonAtomicRates, NativeMarkerSource
from src.models.source_module import load_neon_tables
from tests.test_native_coupling import native_fixture


def constant_tables():
    tables=deepcopy(load_neon_tables())
    axes=(np.array([10.,16.]),np.array([0.,5.]))
    def table(rows):
        return (*axes,np.broadcast_to(np.log10(rows)[:,None,None],(10,2,2)).copy())
    object.__setattr__(tables,'scd',table(np.full(10,1e-30)))
    object.__setattr__(tables,'acd',table(np.full(10,1e-30)))
    object.__setattr__(tables,'plt',table(np.arange(1.,11.)*1e-20))
    return tables,table(np.arange(11.,21.)*1e-20)


def test_plt_prb_charge_offsets_units_and_checkpoint_fingerprint():
    tables,prb=constant_tables()
    full=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=prb)
    old=NativeNeonAtomicRates(tables)
    _,_,rad=full(np.array([10.]),np.array([1e20]))
    # ne=1e14 cm^-3. q0 has PLT1 only; q1..9 sum PLT2..10+PRB11..19;
    # fully stripped q10 still radiates PRB20, but has no PLT row.
    expected=np.array([1.,13.,15.,17.,19.,21.,23.,25.,27.,29.,20.])*1e-6
    np.testing.assert_allclose(rad[0],expected,rtol=3e-14)
    assert old.version!=full.version
    changed=deepcopy(prb);changed[2][0]+=1
    assert NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=changed).version!=full.version
    with pytest.raises(ValueError,match='PRB'):NativeNeonAtomicRates(tables,radiation_mode='plt_prb')
    with pytest.raises(ValueError,match='mode'):NativeNeonAtomicRates(tables,radiation_mode='unknown',prb=prb)
    with pytest.raises(ValueError,match='PRB'):NativeNeonAtomicRates(tables,prb=prb)


def test_prb_reaches_marker_radiation_projection_with_old_charge_distribution():
    tables,prb=constant_tables()
    full=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=prb)
    mesh,_,state,u=native_fixture()
    # Fully stripped Ne; PRB nonzero, PLT-only identically zero for this state.
    state.cloud.P_imp[0]=0.;state.cloud.P_imp[0,10]=1.
    before=deepcopy(state)
    source=NativeMarkerSource(mesh,atomic=full,atomic_version=full.version,
        potentials_ev=np.full(10,20.),F0=-1.,diffusion_ju=0.)
    result,diagnostic=source.advance(state,1e-9,u,u)
    # Initial feedback mean-charge zero: ne=(rho-rho_imp)*8e19=4e19 m^-3.
    # 20e-20 W cm3 *4e13 cm^-3 *1e12 atoms *1e-9s = .008 J.
    assert diagnostic['rad'][0]==pytest.approx(.008,rel=3e-14)
    assert mesh.integrate(result.feedback.values)[1]>0
    np.testing.assert_array_equal(result.feedback.deltas,result.feedback.values-before.feedback.values)
    np.testing.assert_array_equal(state.cloud.P_imp,before.cloud.P_imp)
    np.testing.assert_array_equal(state.feedback.values,before.feedback.values)


def test_real_prb_table_knot_and_fully_stripped_radiation():
    from src.physics.adas_closure import parse_adf11, ADAS_DIR
    tables=load_neon_tables()
    prb=parse_adf11(ADAS_DIR/'prb96_ne.dat')
    provider=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=prb)
    i,j=5,8
    ne=10.**prb[0][i]*1e6;te=10.**prb[1][j]
    _,_,rad=provider(np.array([te]),np.array([ne]))
    # q10 has only the last PRB row. No implementation interpolation in oracle.
    expected=10.**prb[2][-1,j,i]*(ne*1e-6)
    assert expected>0
    assert rad[0,10]==pytest.approx(expected,rel=3e-14)
    assert np.isfinite(rad).all()
    malformed=deepcopy(prb);malformed[2][0,0,0]=np.nan
    with pytest.raises(ValueError,match='PRB'):
        NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=malformed)


def test_log_bilinear_prb_and_caller_tables_are_not_mutated():
    tables,prb=constant_tables()
    # log coefficient affine in both table axes; exact midpoint geometric mean.
    prb[2][9]=np.array([[-20.,-18.],[-16.,-14.]])
    provider=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=prb)
    prb[2][:]=100.  # provider owns a copy, not caller's mutable storage
    _,_,rad=provider(np.array([10.**2.5]),np.array([1e19]))
    assert rad[0,10]==pytest.approx(1e-17*1e13,rel=3e-14)


def test_source_rejects_stale_atomic_configuration_before_advancing():
    tables,prb=constant_tables()
    provider=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=prb)
    mesh,_,_,_=native_fixture()
    with pytest.raises(ValueError,match='atomic.*version'):
        NativeMarkerSource(mesh,atomic=provider,atomic_version=NativeNeonAtomicRates(tables).version,
            potentials_ev=np.full(10,20.),F0=-1.,diffusion_ju=0.)
