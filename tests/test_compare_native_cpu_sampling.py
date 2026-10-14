"""Lightweight four-event capture comparison, never production execution."""
import json
from copy import deepcopy
import numpy as np
import pytest
import torch


def captures(root, *, changed=False):
    from tests.test_compare_native_spatial_rollouts import capture
    from tools.diagnose_native_real_chain import _array_digest
    root.mkdir(); chain=root/'chain'; chain.mkdir()
    binding=dict(initial_binding='same',initial_runtime_digest='runtime',initial_rng_digest='rng',
        initial_model_digest='weights',initial_model_components={'p':'weights','b':'buffer'},
        model_fingerprint='modelconfig',source_configuration={'backend':'CPU'},initial_spatial_mesh_binding='geometry')
    initial_rng=dict(python='fixture',numpy=np.array([3]),torch=torch.tensor([1],dtype=torch.uint8),cuda=None)
    binding['initial_rng_digest']=_array_digest(initial_rng)
    torch.save(dict(rng=initial_rng,initial_binding='same'),chain/'initial-rng.pt')
    outer=dict(success=True,mode='curriculum',run='run',anchor=4110,seed=1030,deterministic=True,
        device='fixture',torch_version='fixture',numpy_version='fixture',environment={'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        settings=dict(threads=2,chunk_size=2048,memory_mib=8192,wall_seconds=900,mode='curriculum'),
        restore=dict(actual_model_digest='weights',normalization_digest='norm'),selection={'burnin':0},
        coupling_implementation=dict(kind='reference' if not changed else 'candidate',sha256='different_allowed'))
    (root/'result.json').write_text(json.dumps(outer))
    summary=dict(binding,completed=True,success=True,model_unchanged=True,initial_runtime_unchanged=True,
        capture_endpoints=True,optimizer_updates=0,gradient_scope='torch_field_bptt_stop_gradient_numpy_source_shard',
        curriculum=dict(burnin_steps=0,forecast_steps=2,full_prefix_support_steps=2),
        loss=.2,loss_reports=[],parameter_gradients={},cross_step_gradients={},events=[])
    for event in range(1,5):
        arrays=capture(); arrays['fields_previous']=arrays['values'].copy()
        arrays['native_density_source']=np.zeros((2,4,1,2))
        for key in ('positions','velocities'):arrays['shards.'+key]=np.zeros((2,3))
        for key in ('ablation_atoms_s','species','injector_phi','source_volume_m3'):
            arrays['shards.'+key]=np.ones(2)
        if changed and event==4: arrays['values'][0,0,0,0]=2.
        rng=dict(python='fixture',numpy=np.array([3]),torch=torch.tensor([1],dtype=torch.uint8),cuda=None)
        clocks=dict(marker={'sim_time_si':event*.25},fluid={'time_si':event*.25},feedback={'time_si':event*.25})
        row=dict(event=event,segment='prefix' if event<=2 else 'forecast',phase='ordinary',time_si=event*.25,
            marker_slots=3,live_markers=2,initial_binding='same',model_digest='weights',state_digest='state'+str(event),
            rng_digest=_array_digest(rng),clocks=clocks,components={'rng':{k:_array_digest(v) for k,v in rng.items()}},
            numeric_capture_schema='native-full-slot-event-arrays-v1')
        row['components'].update(
            cloud={k[6:]:_array_digest(v) for k,v in arrays.items() if k.startswith('cloud.') and k!='cloud.group'},
            shards={k[7:]:_array_digest(v) for k,v in arrays.items() if k.startswith('shards.')},
            feedback={k:_array_digest(arrays['feedback.'+k]) for k in ('values','deltas')},
            torch_state={k:_array_digest(arrays[k]) for k in ('values','raw_deltas','memory')},
            fields={k:_array_digest(arrays[v]) for k,v in {'fields_current':'values',
                'fields_previous':'fields_previous','fluid_deltas':'raw_deltas','field_memory':'memory'}.items()})
        row['transient_density_digest']=_array_digest(arrays['native_density_source'])
        metadata=dict(schema='trusted-native-event-metadata-v1',state_digest=row['state_digest'],clocks=clocks,
            rng=rng,initial_binding='same',runtime={'clock':event*.25},marker={},feedback={},shards={})
        (chain/f'event-{event}.json').write_text(json.dumps(row))
        np.savez(chain/f'event-{event}-arrays.npz',**arrays)
        torch.save(metadata,chain/f'event-{event}-state.pt')
        summary['events'].append(row)
    (chain/'result.json').write_text(json.dumps(summary))


def test_compare_four_complete_captures_reports_relative_metrics_and_exactness(tmp_path):
    from tools.compare_native_cpu_sampling import compare_sampling
    a,b=tmp_path/'a',tmp_path/'b'; captures(a); captures(b,changed=True)
    result=compare_sampling(a,b,output=tmp_path/'comparison.json')
    assert result['success'] and result['binding_verified']
    assert len(result['events'])==4 and not result['all_arrays_exact']
    assert result['all_discrete_exact'] and not result['all_metadata_exact']  # changed field digest is metadata too
    metric=result['events'][3]['arrays']['values']
    assert metric['relative_l2']==pytest.approx(1/np.sqrt(72))
    assert metric['max_abs_difference_over_reference_max_abs_nonpointwise']==1.
    assert not metric['array_equal']
    assert result['events'][3]['detail']['fields']['values']['per_field']['0']['per_degree']['0']['relative_l2']==pytest.approx(1/np.sqrt(2))
    with pytest.raises(FileExistsError):compare_sampling(a,b,output=tmp_path/'comparison.json')


@pytest.mark.parametrize('damage',['weights','geometry','determinism','missing_event','nan','rng_pickle','initial_rng'])
def test_capture_comparison_fails_closed_and_preserves_failure_evidence(tmp_path,damage):
    from tools.compare_native_cpu_sampling import compare_sampling
    a,b=tmp_path/'a',tmp_path/'b';captures(a);captures(b)
    if damage in ('weights','geometry'):
        path=b/'chain/result.json';s=json.loads(path.read_text())
        if damage=='weights':s['initial_model_components']['b']='changed-buffer'
        else:s['initial_spatial_mesh_binding']='other-geometry'
        path.write_text(json.dumps(s))
    elif damage=='determinism':
        path=b/'result.json';s=json.loads(path.read_text());s['deterministic']=False
        path.write_text(json.dumps(s))
    elif damage=='missing_event':(b/'chain/event-3-arrays.npz').unlink()
    elif damage=='nan':
        path=b/'chain/event-2-arrays.npz'
        with np.load(path) as z:arrays={k:z[k] for k in z.files}
        arrays['cloud.x'][2,0]=np.nan  # Inactive slots cannot be ignored.
        np.savez(path,**arrays)
    elif damage=='rng_pickle':
        path=b/'chain/event-1-state.pt';p=torch.load(path,weights_only=False)
        p['rng']['python']='changed';torch.save(p,path)
    else:
        path=b/'chain/initial-rng.pt';p=torch.load(path,weights_only=False)
        p['rng']['python']='changed';torch.save(p,path)
    result=compare_sampling(a,b,output=tmp_path/'failure.json')
    assert not result['success']
    assert 'error' in json.loads((tmp_path/'failure.json').read_text())


def test_all_event_json_metadata_is_compared_not_only_clocks(tmp_path):
    from tools.compare_native_cpu_sampling import compare_sampling
    a,b=tmp_path/'a',tmp_path/'b';captures(a);captures(b)
    path=b/'chain/event-1.json';row=json.loads(path.read_text());row['unexpected_annotation']='changed'
    path.write_text(json.dumps(row))
    path=b/'chain/result.json';summary=json.loads(path.read_text());summary['events'][0]=row
    path.write_text(json.dumps(summary))
    result=compare_sampling(a,b,output=tmp_path/'metadata.json')
    assert result['success'] and not result['all_metadata_exact']
    assert '/unexpected_annotation' in result['events'][0]['event_metadata']['candidate_only_paths']


@pytest.mark.parametrize('damage',['both_missing_shard','both_bad_memory','both_bad_cloud','both_bad_fields','digest'])
def test_complete_schema_inventory_is_required_even_if_both_sides_match(tmp_path,damage):
    from tools.compare_native_cpu_sampling import compare_sampling
    a,b=tmp_path/'a',tmp_path/'b';captures(a);captures(b)
    for root in (a,b):
        p=root/'chain/event-1-arrays.npz'
        with np.load(p) as z:arrays={k:z[k] for k in z.files}
        if damage=='both_missing_shard':del arrays['shards.velocities']
        elif damage=='both_bad_memory':arrays['memory']=np.zeros((3,2))
        elif damage=='both_bad_cloud':arrays['cloud.P_imp']=np.zeros((3,10))
        elif damage=='both_bad_fields':arrays['values']=np.ones((9,4,2,2))
        else:arrays['feedback.deltas'][0,0,0,0]=7.
        np.savez(p,**arrays)
    result=compare_sampling(a,b,output=tmp_path/'schema.json')
    assert not result['success']
    assert result['failure_stage']=='event-1'
