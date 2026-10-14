"""Read a trusted local checkpoint and diagnose feedback layout at saved points.

No source event, sparse solve, model update or default-physics change. Geometry
is validated against the checkpoint, then only sampled elements are built.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from src.models.native_coupling import NativeSourceMesh
from tools.native_replay_metrics import relative_metrics

FEEDBACK_CHANNELS=('dE_ion','rad_sink','Z_eff','Z_imp','N_imp')
FIELD_CHANNELS=('psi','Phi','Z_j','omega','rho','T_i','v_par','rho_imp','T_e')


def _layout(a):
    return dict(shape=list(a.shape),strides=list(a.strides),dtype=str(a.dtype),
                C_contiguous=bool(a.flags.c_contiguous),F_contiguous=bool(a.flags.f_contiguous))


def _roundtrip(a):
    stream=io.BytesIO()
    torch.save(a,stream)
    stream.seek(0)
    return torch.load(stream,map_location='cpu',weights_only=False)


def _comparison(a,b,*,channels=FEEDBACK_CHANNELS):
    av,ag=a;bv,bg=b
    if len(channels)!=av.shape[1]:raise ValueError('explicit channel names must match observations')
    relative={name:dict(values=relative_metrics(av[:,i],bv[:,i]),
        gradient_R=relative_metrics(ag[:,i,0],bg[:,i,0]),
        gradient_Z=relative_metrics(ag[:,i,1],bg[:,i,1])) for i,name in enumerate(channels)}
    return dict(relative_metrics=relative,
        values_exact=bool(np.array_equal(av,bv)),gradients_exact=bool(np.array_equal(ag,bg)),
        changed_value_points=int(np.count_nonzero(np.any(av!=bv,axis=1))),
        changed_gradient_points=int(np.count_nonzero(np.any(ag!=bg,axis=(1,2)))),
        changed_values_by_channel=np.count_nonzero(av!=bv,axis=0).tolist(),
        value_max_abs_by_channel=np.max(np.abs(av-bv),axis=0).tolist(),
        gradient_max_abs_by_channel_direction=np.max(np.abs(ag-bg),axis=0).tolist())


def diagnose_feedback_layout(checkpoint_path,geometry_path,output_path,*,limit=64):
    """Public bounded checkpoint observation with an independent affine test."""
    started=time.perf_counter()
    checkpoint_path=Path(checkpoint_path).resolve();geometry_path=Path(geometry_path).resolve()
    output_path=Path(output_path).resolve()
    if output_path.exists():raise FileExistsError('new evidence output required: '+str(output_path))
    if type(limit) is not int or not 1<=limit<=64:raise ValueError('point budget must be 1..64')
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    state=payload['session'].state
    runtime=getattr(state,'runtime',state)
    del payload,state
    cloud=runtime.marker.cloud
    ids=np.flatnonzero(cloud.alive)[:limit]
    if not len(ids):raise ValueError('checkpoint has no live marker observations')
    geometry={}
    with h5py.File(geometry_path,'r') as f:
        for key in ('x','size','vertex','index'):geometry[key]=f[key][...]
    bound=runtime.native_context.configuration['mesh']
    if any(not np.array_equal(geometry[k],bound[k]) for k in geometry):
        raise ValueError('native geometry differs from checkpoint-bound geometry')
    elements=np.unique(cloud.i_elm[ids]-1)
    if (elements<0).any() or (elements>=geometry['vertex'].shape[1]).any():
        raise ValueError('saved live element outside native geometry')
    mesh=NativeSourceMesh(geometry['x'],geometry['size'][:,:,elements],geometry['vertex'][:,elements]-1,
        geometry['index'],units=runtime.native_context.units,projection='none',max_elements=64)
    feedback=runtime.marker.feedback.values
    mesh.validate_fields(feedback,5)
    _,first=np.unique(geometry['index'],return_index=True)
    vector=feedback.reshape(5,-1)[:,first]
    coefficients=mesh.coefficients(vector)
    if not np.array_equal(coefficients,feedback):
        raise ValueError('feedback is not exactly consistent across native shared DOFs')
    kept=deepcopy(coefficients);saved=_roundtrip(kept)
    canonical_kept=kept.copy(order='C');canonical_saved=saved.copy(order='C')
    def observe(values):
        samples=[mesh.point(int(np.searchsorted(elements,cloud.i_elm[i]-1)),cloud.st[i],values) for i in ids]
        return np.stack([s[0] for s in samples]),np.stack([s[1] for s in samples])
    direct=observe(kept);restored=observe(saved)
    canonical=_comparison(observe(canonical_kept),observe(canonical_saved))
    fields=deepcopy(runtime.fields_current)
    field_control=_comparison(observe(fields),observe(_roundtrip(fields)),channels=FIELD_CHANNELS)
    points=[dict(slot=int(i),element=int(cloud.i_elm[i]),st=cloud.st[i].tolist(),position_RZphi=cloud.x[i].tolist(),
        deepcopy_values=direct[0][j].tolist(),saved_load_values=restored[0][j].tolist(),
        deepcopy_gradients=direct[1][j].tolist(),saved_load_gradients=restored[1][j].tolist()) for j,i in enumerate(ids)]
    sources={}
    root=Path(__file__).resolve().parents[1]
    for path,lines in [('src/models/native_coupling.py','131-132 coefficients; 140-155 point; 283-303 source sampling'),
                       ('src/models/marker_state.py','340-345 projection installation'),
                       ('src/training/base_coupled.py','27-35 detach_state; 323-355 checkpoint save/load'),
                       ('tools/native_replay_metrics.py','20 relative_metrics; native units; no epsilon denominator')]:
        sources[path]=dict(lines=lines,sha256=hashlib.sha256((root/path).read_bytes()).hexdigest())
    result=dict(schema='native-feedback-layout-probe-v2_relative',checkpoint=str(checkpoint_path),geometry_file=str(geometry_path),
        geometry_matches_checkpoint=True,checkpoint_bytes=checkpoint_path.stat().st_size,
        sample_count=len(ids),sampled_elements=len(elements),marker_slots=len(cloud.weight),
        selection='first_up_to_64_live_slots_saved_positions_no_advancement',
        channels=list(FEEDBACK_CHANNELS),field_control_channels=list(FIELD_CHANNELS),gradient_directions=['R','Z'],
        metric_reference='deepcopy_of_native_coefficients_layout',metric_candidate='torch_saved_load',
        primary_metrics='per_channel_relative_l2_and_pointwise_relative_with_explicit_zero_reference_handling',
        input_values_exact=bool(np.array_equal(coefficients,kept) and np.array_equal(kept,saved)),
        layouts={name:_layout(a) for name,a in [('checkpoint',feedback),('coefficients',coefficients),
            ('deepcopy',kept),('saved_load',saved),('canonical_deepcopy',canonical_kept),('canonical_saved_load',canonical_saved)]},
        comparison=_comparison(direct,restored),canonical=canonical,current_field_control=field_control,points=points,
        source_references=sources,numpy_version=np.__version__,torch_version=str(torch.__version__),
        source_events_executed=0,optimizer_updates=0,default_physics_modified=False,
        interpretation='layout-only point-observation experiment; full-event causality not certified',
        elapsed_seconds=time.perf_counter()-started)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='trusted local session checkpoint')
    p.add_argument('--geometry',type=Path,required=True,help='native JOREK HDF geometry file')
    p.add_argument('--output',type=Path,required=True,help='new JSON path; existing files rejected')
    p.add_argument('--limit',type=int,default=64)
    a=p.parse_args();result=diagnose_feedback_layout(a.checkpoint,a.geometry,a.output,limit=a.limit)
    print(json.dumps({k:result[k] for k in ('sample_count','input_values_exact','comparison','canonical',
        'current_field_control','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
