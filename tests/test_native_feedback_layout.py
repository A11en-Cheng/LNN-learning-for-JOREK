"""Checkpoint/point-observation seam; affine oracle, no source advancement."""
from types import SimpleNamespace
import h5py
import numpy as np
import torch


def test_checkpoint_feedback_layout_probe_preserves_affine_values_and_canonical_observation(tmp_path):
    from tools.diagnose_native_feedback_layout import diagnose_feedback_layout
    from tools.diagnose_native_production_chain import build_fixture
    _,source,_,runtime,_=build_fixture(2)
    mesh=source.mesh
    runtime.marker.cloud.i_elm[:2]=1
    runtime.marker.cloud.st[:2]=[[.25,.4],[.7,.6]]
    runtime.marker.cloud.x[:2]=[[2.25,.4,0.],[2.7,.6,0.]]
    runtime.marker.feedback.values=np.stack([k*mesh.coordinates[0]+(k+1)*mesh.coordinates[1] for k in range(1,6)])
    checkpoint=tmp_path/'checkpoint.pt';geometry=tmp_path/'geometry.h5';output=tmp_path/'probe.json'
    torch.save({'session':SimpleNamespace(state=SimpleNamespace(runtime=runtime))},checkpoint)
    with h5py.File(geometry,'w') as f:
        for key,value in dict(x=mesh.coordinates,size=mesh.size,vertex=mesh.vertex+1,index=mesh.indices).items():f[key]=value
    result=diagnose_feedback_layout(checkpoint,geometry,output)
    assert result['sample_count']==2 and result['input_values_exact']
    assert result['layouts']['deepcopy']['strides']!=result['layouts']['saved_load']['strides']
    assert result['canonical']['values_exact'] and result['canonical']['gradients_exact']
    relative=result['comparison']['relative_metrics']['dE_ion']
    np.testing.assert_allclose(relative['values']['reference_l2'],np.hypot(3.05,3.9),rtol=1e-14)
    np.testing.assert_allclose(relative['gradient_R']['reference_l2'],np.sqrt(2),rtol=1e-14)
    np.testing.assert_allclose(relative['gradient_Z']['reference_l2'],np.sqrt(8),rtol=1e-14)
    for channel in result['channels']:
        for component in ('values','gradient_R','gradient_Z'):
            metric=result['canonical']['relative_metrics'][channel][component]
            assert metric['count']==2 and metric['relative_l2']==0
    for row,(r,z) in zip(result['points'],[(2.25,.4),(2.7,.6)]):
        np.testing.assert_allclose(row['deepcopy_values'],[k*r+(k+1)*z for k in range(1,6)],rtol=1e-14)
        np.testing.assert_allclose(row['deepcopy_gradients'],[[k,k+1] for k in range(1,6)],rtol=1e-14)
    assert output.is_file() and result['source_events_executed']==0
