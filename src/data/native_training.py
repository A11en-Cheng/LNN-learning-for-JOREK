"""Explicit-scope phaseA audit + causal banks + train-only statistics, no legacy manifest."""
from dataclasses import dataclass
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
from .native_window_io import read_segment,atomic_json
from .native_windows import build_windows,WindowConfig,SplitBlock,NativeScaler
from .native_features import read_native_feature_evaluator
from .native_parameters import parameter_variation
from src.models.native_features import NativeFeaturePath
from src.models.native_state import configuration_digest

SCOPE='field_only_no_feedback_consumer'


@dataclass
class TrainingData:
    banks: dict
    scaler: NativeScaler
    evaluator: object
    feature_statistics: dict
    coverage: dict
    spec_digest: str
    train_geometry_file: str


def prepare_training_data(spec,output_dir):
    if (spec.get('schema')!='native-curriculum-data-v1' or spec.get('horizons')!=[1,2,4]
        or not spec.get('training_run')):
        raise ValueError('explicit curriculum data schema/horizons/training run required')
    registry=json.loads(Path(spec['restart_registry']).read_text())
    records=[];audit=[];plan={}
    for segment in spec['segments']:
        if not segment.get('configuration_provenance') or not segment.get('schedule_provenance'):
            raise ValueError('explicit configuration and time schedule provenance required')
        run=Path(segment['run_path'])
        rs,rows=read_segment(run,range(segment['first'],segment['last']+1),
            restart_events=registry['runs'][run.name]['restart_events'],gamma=spec['gamma'],
            gamma_provenance=spec['gamma_provenance'],modes=tuple(spec['modes']))
        records.extend(rs);audit.extend(dict(run=run.name,**r) for r in rows)
        for p in segment['plan_jorek']:
            for i in range(p['first'],p['last']+1):
                key=(run.name,i)
                if key in plan:raise ValueError('overlapping explicit schedule')
                plan[key]=p['dt']*rs[0].state.units.t_norm_si
    blocks=[SplitBlock(**b) for b in spec['blocks']]
    banks={};exclusions={};unique={}
    for k in spec['horizons']:
        collection=build_windows(records,blocks,WindowConfig(spec['warmup'],k,spec['stride'],spec['isolation'],
            field_teacher_policy='paired_U_rawD',source_teacher_policy='none'),
            time_plan=plan,plan_provenance=spec['plan_provenance'],scheme=spec['scheme'])
        banks[k]=collection.windows;exclusions[str(k)]=collection.exclusions
        for w in collection.windows:
            for r in w.support_records:unique.setdefault(w.split,{})[r.run,r.state.step_index]=r
    if any(not any(w.split=='train' for w in ws) for ws in banks.values()):
        raise ValueError('each curriculum horizon requires training windows')
    identities={s:set(rows) for s,rows in unique.items()}
    for a,keys in identities.items():
        for b,other in identities.items():
            if a!=b and keys&other:raise ValueError('complete support crosses split')
    train=list(unique['train'].values())
    if any(r.run!=spec['training_run'] for r in train):raise ValueError('unexpected training run')
    scaler=NativeScaler.fit({'train':train})
    geometry_file=train[0].provenance
    ev=read_native_feature_evaluator(geometry_file,units=train[0].state.units)
    features=NativeFeaturePath.fit(ev,[w for ws in banks.values() for w in ws if w.split=='train'],
        scenario_enabled=True,physics_enabled=True)
    stats=features.statistics()
    status=Counter(p['status'] for r in records for p in r.parameters)
    coverage=dict(scope=SCOPE,schema='native-curriculum-coverage-v1',raw_records=len(records),
        spec_digest=configuration_digest(spec),record_audit=audit,exclusions=exclusions,
        window_counts={str(k):dict(Counter((w.split+':'+w.run) for w in ws)) for k,ws in banks.items()},
        split_unique_support={s:[list(k) for k in sorted(rows)] for s,rows in unique.items()},
        parameter_variation=parameter_variation({s:[r.parameters for r in rows.values()] for s,rows in unique.items()}),
        parameter_status_counts=dict(status),coupled_rollout_windows=0,
        feedback_missing_records=sum(r.feedback is None for r in records),
        gaps=['missing feedback remains None; no consumer','physical BC unverified',
            'node stencil/shared-index mean/positive feature pooling approximations',
            'constant or confounded parameters do not establish generalization'])
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'manifest.json',dict(scope=SCOPE,schema='native-curriculum-manifest-v1',spec=spec,
        spec_digest=configuration_digest(spec),banks={str(k):[w.manifest() for w in ws] for k,ws in banks.items()}))
    scaler.save(output/'normalizer.json')
    normalizer=json.loads((output/'normalizer.json').read_text());normalizer['scope']=SCOPE
    atomic_json(output/'normalizer.json',normalizer)
    atomic_json(output/'feature-statistics.json',dict(scope=SCOPE,**stats))
    atomic_json(output/'coverage.json',coverage)
    atomic_json(output/'parameter-routes.json',dict(scope=SCOPE,records=[
        dict(run=r.run,index=r.state.step_index,routes=list(r.parameters)) for r in records]))
    return TrainingData(banks,scaler,ev,stats,coverage,configuration_digest(spec),geometry_file)


def build_training_model(bundle,configuration):
    """One confirmed model configuration; fresh train-bound statistics, no solver search."""
    from .native_state import read_native_update_mapping
    from src.models.native_field_step import NativeFieldStep
    from src.models.native_temporal import CfCTimeCore
    from src.models.stencil_conv import build_fem_stencil_geometry
    if configuration.get('output_semantics','rate_memory_v2')!='rate_memory_v2':
        raise ValueError('incompatible field decoder semantics; rate_memory_v2 required')
    ev=deepcopy(bundle.evaluator);features=NativeFeaturePath(ev,deepcopy(bundle.feature_statistics))
    geom=build_fem_stencil_geometry(ev.vertex.numpy().T,ev.node_positions.numpy())
    mapping=read_native_update_mapping(bundle.train_geometry_file,keep_n0_const=False,boundary_policy='learned_unverified')
    s=bundle.scaler;c=configuration
    return NativeFieldStep(geom,mapping,units=ev.units,state_scale=s.value_scale[:,:,0,0],
        increment_scale=s.history_scale[:,:,0,0],center=s.value_center[:,:,0,0],
        hidden=c['hidden'],n_layers=c['n_layers'],n_iters=c['n_iters'],stencil_strength=c['stencil_strength'],
        feature_path=features,temporal_core=CfCTimeCore(c['hidden'],backbone_units=c['backbone_units'],time_scale_ju=c['time_scale_ju']))
