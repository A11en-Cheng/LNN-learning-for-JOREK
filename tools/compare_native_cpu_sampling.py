"""Strict four-event CPU sampling A/B evidence, not a speedup assessment.

Inputs must be trusted local profile_native_coupled_backward artifacts:
event-state.pt metadata uses Python pickle. NPZ numerical arrays never use
pickle. Reads one paired event at a time, including every inactive slot.
"""
import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np
import torch

from tools.compare_native_spatial_rollouts import compare_event, _exact_tree
from tools.native_replay_metrics import relative_metrics
from tools.diagnose_native_real_chain import _array_digest

BINDING = ('initial_binding','initial_runtime_digest','initial_rng_digest','initial_model_digest',
    'initial_model_components','model_fingerprint','source_configuration','initial_spatial_mesh_binding','gradient_scope')
OUTER = ('mode','run','anchor','seed','deterministic','device','torch_version','numpy_version',
    'environment','restore','selection')
SETTINGS = ('threads','chunk_size','memory_mib','wall_seconds','mode')


def metadata_difference(a,b,path=''):
    """Exact typed metadata comparison, concise paths rather than huge values."""
    if _array_digest(a)==_array_digest(b):
        return []
    if type(a) is not type(b):
        return [path or '/']
    if is_dataclass(a):
        return [p for f in fields(a) for p in metadata_difference(getattr(a,f.name),getattr(b,f.name),path+'/'+f.name)]
    if isinstance(a,dict):
        out=[path+'/'+str(k) for k in a.keys()^b.keys()]
        return out+[p for k in a.keys()&b.keys() for p in metadata_difference(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(list,tuple)) and len(a)==len(b):
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in metadata_difference(x,y,path+'/'+str(i))]
    return [path or '/']


def _read(path):
    record=json.loads(path.read_text())
    json.dumps(record,allow_nan=False)
    return record


def validate_inventory(arrays,row):
    """Complete native neon capture, bound to independently recorded components."""
    from src.models.marker_state import MarkerCloud
    cloud=set(MarkerCloud.array_names())
    components=row['components']
    required_shards={'positions','velocities','radius','ablation_atoms_s','species',
                     'injector_phi','ids','source_volume_m3'}
    required_components={'fields','torch_state','cloud','feedback','shards','rng'}
    if not required_components<=components.keys():
        raise ValueError('missing full capture component inventory')
    if (set(components['cloud'])!=cloud or set(components['feedback'])!={'values','deltas'}
        or set(components['torch_state'])!={'values','raw_deltas','memory'}
        or set(components['fields'])!={'fields_current','fields_previous','fluid_deltas','field_memory'}
        or not required_shards<=components['shards'].keys()):
        raise ValueError('incomplete physical component inventory')
    expected={'values','raw_deltas','memory','fields_previous','native_density_source','cloud.group',
              'feedback.values','feedback.deltas'} | {'cloud.'+k for k in cloud} | {'shards.'+k for k in components['shards']}
    if set(arrays)!=expected:
        raise ValueError('capture key inventory mismatch: missing '+str(sorted(expected-arrays.keys()))+
                         '; extra '+str(sorted(arrays.keys()-expected)))
    shape=arrays['values'].shape
    if len(shape)!=4 or shape[:3]!=(9,4,1) or shape[-1]<1:
        raise ValueError('complete nine-field cubic n0 coefficient shape required')
    n=shape[-1]; slots=row['marker_slots']
    shapes={k:shape for k in ('values','raw_deltas','fields_previous')}
    shapes.update({'feedback.values':(5,4,1,n),'feedback.deltas':(5,4,1,n),
                   'native_density_source':(2,4,1,n),'cloud.group':()})
    shapes.update({'cloud.'+k:(slots,*tail) for k,tail in dict(x=(3,),st=(2,),i_elm=(),weight=(),
        P_imp=(11,),V_prev=(3,),i_life=(),t_birth=()).items()})
    if arrays['shards.ids'].ndim!=1:
        raise ValueError('one-dimensional shard identities required')
    shards=len(arrays['shards.ids'])
    shapes.update({'shards.'+k:(shards,3) if k in ('positions','velocities') else (shards,) for k in required_shards})
    if arrays['memory'].ndim!=2 or arrays['memory'].shape[0]!=n or arrays['memory'].shape[1]<1:
        raise ValueError('full-node temporal memory shape required')
    for key,want in shapes.items():
        if arrays[key].shape!=want:
            raise ValueError('capture shape mismatch: '+key)
    for key,value in arrays.items():
        if key=='cloud.group':
            valid=value.dtype.kind=='U'
        elif key in ('cloud.i_elm','cloud.i_life','shards.ids'):
            valid=value.dtype.kind in 'iu'
        else:
            valid=value.dtype.kind=='f' or (key.startswith('shards.') and value.dtype.kind in 'biu')
        if not valid:raise ValueError('capture dtype mismatch: '+key)
    digests={k:_array_digest(v) for k,v in arrays.items()}
    for group in ('cloud','shards','feedback'):
        for key,digest in components[group].items():
            if digests[group+'.'+key]!=digest:
                raise ValueError('capture component digest mismatch: '+group+'.'+key)
    for key,digest in components['torch_state'].items():
        if digests[key]!=digest:raise ValueError('Torch capture digest mismatch: '+key)
    for key,name in dict(fields_current='values',fields_previous='fields_previous',
                         fluid_deltas='raw_deltas',field_memory='memory').items():
        if digests[name]!=components['fields'][key]:
            raise ValueError('runtime mirror capture digest mismatch: '+key)
    if digests['native_density_source']!=row.get('transient_density_digest'):
        raise ValueError('transient density capture digest mismatch')


def compare_sampling(reference,candidate,*,output):
    """Comparison success means execution completed, not numerical equivalence."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    result=dict(success=False,binding_verified=False,events=[],speedup_claim=False,
        scope='four_complete_events_all_slots_same_trained_model_no_backend_whitelist',
        trusted_local_pickle=True)
    stage='binding'
    try:
        roots=[Path(reference),Path(candidate)]
        chains=[p/'chain' if (p/'chain').is_dir() else p for p in roots]
        outer=[_read(p.parent/'result.json') for p in chains]
        summaries=[_read(p/'result.json') for p in chains]
        for o,s in zip(outer,summaries):
            if (not o.get('success') or not s.get('completed') or not s.get('model_unchanged')
                or not s.get('initial_runtime_unchanged') or s.get('capture_endpoints') is not True
                or s.get('optimizer_updates')!=0):
                raise ValueError('successful frozen-weight full capture with no updates required')
            if len(s.get('events',[]))!=4 or any(k not in s or s[k] is None for k in BINDING):
                raise ValueError('four events and complete initial binding evidence required')
            if not s['initial_model_components']:
                raise ValueError('complete parameter/buffer component map required')
            if any(k not in o for k in OUTER) or any(k not in o.get('settings',{}) for k in SETTINGS):
                raise ValueError('execution context missing')
            if o['mode']!='curriculum' or s.get('curriculum',{}).get('burnin_steps')!=0:
                raise ValueError('current prefix2/B0/forecast2 curriculum required')
            if s['curriculum'].get('full_prefix_support_steps')!=2 or s['curriculum'].get('forecast_steps')!=2:
                raise ValueError('current curriculum event count mismatch')
        left=dict(binding={k:summaries[0][k] for k in BINDING},
            context={k:outer[0][k] for k in OUTER},settings={k:outer[0]['settings'][k] for k in SETTINGS})
        right=dict(binding={k:summaries[1][k] for k in BINDING},
            context={k:outer[1][k] for k in OUTER},settings={k:outer[1]['settings'][k] for k in SETTINGS})
        result['binding']=_exact_tree(left,right)
        result['coupling_implementations']=[o.get('coupling_implementation') for o in outer]
        if not result['binding']['equal']:
            raise ValueError('strict initial model/state/RNG/configuration or execution context mismatch')
        for p,summary in zip(chains,summaries):
            initial=torch.load(p/'initial-rng.pt',map_location='cpu',weights_only=False,mmap=True)
            if (initial.get('initial_binding')!=summary['initial_binding'] or
                _array_digest(initial.get('rng'))!=summary['initial_rng_digest']):
                raise ValueError('initial RNG artifact binding mismatch')
        result['binding_verified']=True
        stage='inventory'
        expected={f'event-{i}-arrays.npz' for i in range(1,5)}
        for p in chains:
            if {x.name for x in p.glob('event-*-arrays.npz')}!=expected:
                raise ValueError('exactly four complete numerical endpoints required')
            for i in range(1,5):
                if not (p/f'event-{i}.json').is_file() or not (p/f'event-{i}-state.pt').is_file():
                    raise ValueError('missing complete endpoint metadata')
        for event in range(1,5):
            stage=f'event-{event}'
            rows=[_read(p/f'event-{event}.json') for p in chains]
            metadata=[torch.load(p/f'event-{event}-state.pt',map_location='cpu',weights_only=False,mmap=True) for p in chains]
            for row,meta,summary in zip(rows,metadata,summaries):
                if row!=summary['events'][event-1]:
                    raise ValueError('event summary and standalone metadata disagree')
                if (row['event']!=event or row['segment']!=('prefix' if event<=2 else 'forecast')
                    or row['initial_binding']!=summary['initial_binding']
                    or row['model_digest']!=summary['initial_model_digest']
                    or row['numeric_capture_schema']!='native-full-slot-event-arrays-v1'):
                    raise ValueError('event/model/initial capture binding mismatch')
                if (meta.get('schema')!='trusted-native-event-metadata-v1'
                    or meta['initial_binding']!=row['initial_binding'] or meta['state_digest']!=row['state_digest']
                    or _array_digest(meta['rng'])!=row['rng_digest'] or meta['clocks']!=row['clocks']
                    or {k:_array_digest(v) for k,v in meta['rng'].items()}!=row['components']['rng']):
                    raise ValueError('trusted metadata/JSON/RNG integrity mismatch')
            if rows[0]['phase']!=rows[1]['phase']:
                raise ValueError('paired event phases differ')
            with np.load(chains[0]/f'event-{event}-arrays.npz',allow_pickle=False) as az, \
                 np.load(chains[1]/f'event-{event}-arrays.npz',allow_pickle=False) as bz:
                a={k:az[k] for k in az.files}; b={k:bz[k] for k in bz.files}
                for arrays,row in zip((a,b),rows):
                    validate_inventory(arrays,row)
                    if not {'fields_previous','native_density_source'}<=arrays.keys():
                        raise ValueError('full prior fields and transient density capture required')
                    if (len(arrays['cloud.x'])!=row['marker_slots'] or
                        np.count_nonzero(arrays['cloud.i_elm']>0)!=row['live_markers']):
                        raise ValueError('full-slot count/liveness metadata mismatch')
                detail=compare_event(a,b,reference_metadata=rows[0],candidate_metadata=rows[1])
                metrics={}; discrete={}
                for k in a:
                    if a[k].dtype!=b[k].dtype:
                        raise ValueError('capture dtype mismatch: '+k)
                    exact=bool(np.array_equal(a[k],b[k]))
                    if a[k].dtype.kind in 'biuU':
                        discrete[k]=dict(array_equal=exact,mismatch_count=int(np.count_nonzero(a[k]!=b[k])))
                    else:
                        metrics[k]=dict(relative_metrics(a[k],b[k]),array_equal=exact)
                paths=metadata_difference(metadata[0],metadata[1])
                event_metadata=_exact_tree(rows[0],rows[1])
                result['events'].append(dict(event=event,segment=rows[0]['segment'],detail=detail,
                    arrays=metrics,discrete=discrete,all_arrays_exact=detail['all_arrays_exact'],
                    all_discrete_exact=all(v['array_equal'] for v in discrete.values()),
                    metadata_exact=not paths and event_metadata['equal'],metadata_mismatch_paths=paths,
                    event_metadata=event_metadata,
                    clocks_exact=detail['clocks']['equal'],rng_exact=detail['rng']['equal']))
                del a,b
            del metadata
        result.update(success=True,all_arrays_exact=all(r['all_arrays_exact'] for r in result['events']),
            all_discrete_exact=all(r['all_discrete_exact'] for r in result['events']),
            all_metadata_exact=all(r['metadata_exact'] for r in result['events']),
            all_clocks_exact=all(r['clocks_exact'] for r in result['events']),
            all_rng_exact=all(r['rng_exact'] for r in result['events']))
        result['backward_diagnostics']={k:_exact_tree(summaries[0].get(k),summaries[1].get(k)) for k in
            ('loss','loss_reports','parameter_gradients','cross_step_gradients')}
        result['backward_scope']='loss_and_reported_gradient_statistics_only; full_gradient_arrays_not_captured'
        result['all_physical_evidence_exact']=all(result[k] for k in
            ('all_arrays_exact','all_discrete_exact','all_metadata_exact','all_clocks_exact','all_rng_exact'))
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),failure_stage=stage)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=compare_sampling(args.reference,args.candidate,output=args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('events','backward_diagnostics')},indent=2))
    return 0 if r['success'] and r['all_physical_evidence_exact'] else 1


if __name__=='__main__':raise SystemExit(main())
