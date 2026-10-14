"""Teacher records through the public reader, never a predicted-field reset."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400'


@pytest.mark.parametrize('directory,index,axis,boundary,distance',[
    (RUN,4110,.5044675959845064,.11326835771124656,.7424498055841912),
    (Path(str(RUN)+'_injInner'),5680,.5037458959350467,.11256713195472477,.7358615545252232)])
def test_real_current_record_keeps_full_cubic_psi_saved_es_and_history_evidence(directory,index,axis,boundary,distance):
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.models.native_state import configuration_digest
    path=directory/f'jorek{index:06d}.h5'
    if not path.exists():pytest.skip('local target HDF unavailable')
    record=read_native_psi_teacher(path)
    with h5py.File(path,'r') as f:
        np.testing.assert_array_equal(record.psi,f['values'][0])
        assert record.equilibrium.field_digest==configuration_digest(f['values'][...])
    assert record.psi.shape==(4,1,4125)
    assert record.equilibrium.psi_axis==axis and record.equilibrium.psi_bnd==boundary
    assert record.equilibrium.xcase=='double'
    assert record.equilibrium.lcfs_lost is False
    assert record.lcfs_min_distance_m==pytest.approx(distance,abs=1e-14)
    assert record.lcfs_history_length==index
    assert record.equilibrium.time_si==record.time_si
    assert len(record.source_sha256)==len(record.geometry_digest)==64
    assert 'teacher' in record.provenance and 'original_nine_fields' in record.provenance
    with pytest.raises(FrozenInstanceError):record.time_si=0.
    with pytest.raises(ValueError):record.psi[0,0,0]=0.
    with pytest.raises(ValueError):record.psi.setflags(write=True)


def small_record(tmp_path, *, times=(1.,2.), axis_r=(2.505,2.25)):
    path=tmp_path/'jorek000002.h5'
    (tmp_path/'jorek_namelist_spitest').write_text('xpoint=.t.\nxcase=3\nR_geo=3.\n')
    with h5py.File(path,'w') as f:
        for k,v in dict(jorek_model=502,n_order=3,n_tor=1,n_plane=1,n_period=1,
                        t_norm=1e-6,central_density=.8,central_mass=1.,index_now=2,t_now=2.).items():f[k]=[v]
        x=np.zeros((2,4,1,4));x[:,0,0]=[[2.,3.,3.,2.],[0.,0.,1.,1.]]
        x[0,1,0]=np.array([1.,-1.,-1.,1.])/3
        x[1,2,0]=np.array([1.,1.,-1.,-1.])/3
        f['x']=x;f['size']=np.ones((4,4,1));f['vertex']=np.arange(1,5)[:,None]
        f['index']=np.arange(1,17).reshape(4,4);f['boundary']=np.ones(4,dtype=int)
        f['values']=np.zeros((9,4,1,4));f['xtime']=times
        f['R_axis_t']=axis_r;f['Z_axis_t']=[.5,.5]
        f['psi_axis_t']=[1.,1.];f['psi_bnd_t']=[0.,0.]
        f['psi_xpoint_t']=np.zeros((2,2));f['Z_xpoint_t']=[[-1.,-1.],[1.,1.]]
    return path


def test_reader_rejects_noncausal_diagnostic_history_even_when_endpoint_matches(tmp_path):
    from src.data.native_psi_teacher import read_native_psi_teacher
    with pytest.raises(ValueError,match='monotonic'):
        read_native_psi_teacher(small_record(tmp_path,times=(3.,2.)))


def test_only_explicit_registered_restart_accepts_subfemtosecond_time_roundoff(tmp_path):
    from src.data.native_psi_teacher import read_native_psi_teacher
    path=small_record(tmp_path,times=(2.+4e-10,2.))
    with pytest.raises(ValueError,match='monotonic'):read_native_psi_teacher(path)
    result=read_native_psi_teacher(path,registered_restart_indices=(2,))
    assert 'registered_restart_time_roundoff' in result.provenance
    with h5py.File(path,'r') as f:np.testing.assert_array_equal(f['xtime'][...],[2.+4e-10,2.])
    with h5py.File(path,'r+') as f:f['xtime'][0]=3.
    with pytest.raises(ValueError,match='monotonic'):
        read_native_psi_teacher(path,registered_restart_indices=(2,))


def test_real_inner_validation_restart_roundoff_is_preserved_not_retimed():
    from src.data.native_psi_teacher import read_native_psi_teacher
    path=Path(str(RUN)+'_injInner')/'jorek006441.h5'
    if not path.exists():pytest.skip('local target HDF unavailable')
    record=read_native_psi_teacher(path,lcfs_restart_index=6440,registered_restart_indices=(6441,))
    assert record.step_index==6441 and record.lcfs_history_length==6440
    assert 'registered_restart_time_roundoff' in record.provenance


def test_lcfs_history_is_rebuilt_not_default_false_and_uses_element_centers(tmp_path):
    from src.data.native_psi_teacher import read_native_psi_teacher
    record=read_native_psi_teacher(small_record(tmp_path))
    # Rectangle center (2.5,.5), historical axis (2.505,.5): .005 < 3/30.
    # Every vertex and edge midpoint is far away. Current axis alone is .25 away.
    assert record.lcfs_min_distance_m==pytest.approx(.005,abs=1e-15)
    assert record.equilibrium.lcfs_lost is True


def test_future_record_cannot_be_requested_as_current_and_missing_history_cannot_default(tmp_path):
    from src.data.native_psi_teacher import read_native_psi_teacher
    path=small_record(tmp_path)
    with pytest.raises(ValueError,match='same-time'):
        read_native_psi_teacher(path,expected_time_si=1e-6)
    assert read_native_psi_teacher(path,expected_time_si=2e-6).time_si==2e-6
    with h5py.File(path,'a') as f:del f['R_axis_t']
    with pytest.raises(ValueError,match='missing history'):
        read_native_psi_teacher(path)


def test_continuous_node_can_keep_fixed_restart_lcfs_history_boundary(tmp_path):
    from src.data.native_psi_teacher import read_native_psi_teacher
    path=small_record(tmp_path,axis_r=(2.25,2.505))
    record=read_native_psi_teacher(path,lcfs_restart_index=1)
    assert record.lcfs_history_length==1
    assert record.equilibrium.lcfs_lost is False
    assert record.equilibrium.psi_axis==1.
    assert 'restart_history_reconstruction' in record.provenance


@pytest.mark.parametrize('directory,index',[(RUN,4111),pytest.param(Path(str(RUN)+'_injInner'),5681,
    marks=pytest.mark.xfail(strict=True,reason='causal aux corrected; remaining real rate error 4.898e-4 exceeds 1e-6, not certified'))])
def test_registered_poststartup_cache_uses_original_zero_feedback_for_all846_rate(directory,index,tmp_path):
    """Real public CLI/source update; next record is an independent JOREK oracle."""
    import json,subprocess,sys
    if not (directory/f'jorek{index:06d}.h5').exists():pytest.skip('local target HDF unavailable')
    output=tmp_path/'causal.json'
    subprocess.run([sys.executable,str(ROOT/'tools/audit_native_psi_shards.py'),
        '--path',str(directory/f'jorek{index:06d}.h5'),'--target',str(directory/f'jorek{index+1:06d}.h5'),
        '--feedback-path',str(directory/f'projections{index-1:06d}.h5'),
        '--feedback-policy','registered-startup-zero','--shards-only','--output',str(output)],
        check=True,capture_output=True,text=True,timeout=65)
    result=json.loads(output.read_text())
    assert result['stop'] is None
    update=next(s for s in result['stages'] if s['stage']=='all_shard_production_update')
    assert update['shards']==846
    assert update['metrics']['ablation_atoms_s']['max_relative_nonzero']<1e-6


@pytest.mark.parametrize('field_index,feedback_index,message',[
    (4112,4110,'registered poststartup field index'),
    (4111,4111,'registered original zero projection')])
def test_startup_zero_policy_cannot_be_applied_to_ordinary_node_or_unregistered_aux(field_index,feedback_index,message,tmp_path):
    import json,subprocess,sys
    if not (RUN/f'jorek{field_index:06d}.h5').exists():pytest.skip('local target HDF unavailable')
    output=tmp_path/'rejected.json'
    subprocess.run([sys.executable,str(ROOT/'tools/audit_native_psi_shards.py'),
        '--path',str(RUN/f'jorek{field_index:06d}.h5'),'--target',str(RUN/f'jorek{field_index+1:06d}.h5'),
        '--feedback-path',str(RUN/f'projections{feedback_index:06d}.h5'),
        '--feedback-policy','registered-startup-zero','--shards-only','--output',str(output)],
        check=True,capture_output=True,text=True,timeout=65)
    result=json.loads(output.read_text())
    failures=[s for s in result['stages'] if s['stage']=='shard_failure']
    assert len(failures)==1 and message in failures[0]['message']
    assert not any(s['stage']=='all_shard_production_update' for s in result['stages'])
