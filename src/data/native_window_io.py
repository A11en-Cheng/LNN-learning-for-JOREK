"""Native window file adapters and atomic reviewable JSON artifacts."""
from pathlib import Path
import json
import os
import tempfile


def atomic_json(path,payload):
    data=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    path=Path(path)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'w') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def read_segment(run_path, indices, *, restart_events, gamma, gamma_provenance, modes):
    """Read only explicit records; full-state reader + existing time audit.

    Missing ordinary projection history remains absent. No marker payload is
    read and no marker initialization/replay is claimed. Production namelist
    is bound to this declared fixed-configuration segment by caller contract.
    """
    import h5py
    import numpy as np
    from src.data.native_state import read_native_field_state,audit_native_segment
    from src.models.native_state import NativeUnits,configuration_digest
    from src.data.native_run import resolve_namelist_path
    from src.data.parameter_channel import build_condition_vector
    from src.data.native_parameters import route_parameters
    from src.data.native_windows import FieldRecord
    from src.data.namelist import parse_namelist
    if not gamma_provenance: raise ValueError('gamma source must be explicitly declared')
    run=Path(run_path); indices=tuple(indices)
    if not indices: raise ValueError('empty segment')
    rows=audit_native_segment(run,indices,restart_events=restart_events)
    nl=resolve_namelist_path(run)
    if nl is None: raise ValueError('production namelist source missing')
    text=nl.read_text(); parsed=parse_namelist(text)
    output=[]
    for i,row in zip(indices,rows):
        path=run/f'jorek{i:06d}.h5'
        with h5py.File(path) as f:
            scalar=lambda k:float(np.ravel(f[k][()])[0])
            units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),gamma)
            mesh={k:np.asarray(f[k][()]) for k in ('x','size','vertex','index','boundary','freeboundary','n_order','n_tor','n_plane','n_period')}
            payload=build_condition_vector(f,text)
            parameters=route_parameters(payload,source=str(path.resolve()),namelist_source=str(nl.resolve()))
            # The legacy chain pads partial arrays; never certify those defaults.
            for p in parameters:
                if '[' in p['name'] and p['provenance']['kind']=='namelist':
                    base=p['name'].split('[')[0].lower(); j=int(p['name'].split('[')[1][:-1])
                    raw=parsed.get(base)
                    if not isinstance(raw,list) or j>len(raw):
                        p['status']='unconfirmed_array_padding'; p['available_at']='unavailable_until_confirmed'
        state=read_native_field_state(path,units=units,modes=modes)
        with h5py.File(run/f'projections{i:06d}.h5') as f:
            row['projection_history_present']='deltas' in f
        feedback=None
        if row['projection_history_present'] and row['projection_phase']=='aligned':
            from src.data.native_state import read_native_feedback_state
            feedback=read_native_feedback_state(run/f'projections{i:06d}.h5',units=units,modes=modes)
        fixed=[(p['name'],p['value'],p['status']) for p in parameters if p['group']!='numeric_time']
        compatibility=configuration_digest(dict(mesh=mesh,units=units,modes=modes,parameters=fixed))
        # Source interval may be inferred only for ordinary aligned positive steps.
        interval=None
        if not row['restart'] and row['projection_phase']=='aligned' and row['fluid_interval_jorek'] is not None:
            interval=(state.time_si-row['fluid_interval_jorek']*units.t_norm_si,state.time_si)
        row['gamma_provenance']=gamma_provenance
        row['parameter_status_counts']=dict(__import__('collections').Counter(p['status'] for p in parameters))
        output.append(FieldRecord(run.name,state,row['fluid_tstep_jorek']*units.t_norm_si,
            compatibility,str(path.resolve()),row['restart'],feedback,interval,tuple(parameters),
            dict(kind='field_checkpoint_only',gamma_provenance=gamma_provenance,
                 marker_checkpoint_verified=False,feedback_history_available=feedback is not None)))
    return output,rows


def audit_spec(spec, output_dir):
    """Reproducible explicit-scope audit; never inherits old manifests/stats."""
    from collections import Counter
    from src.data.native_windows import WindowConfig,SplitBlock,build_windows,NativeScaler
    from src.data.native_parameters import parameter_variation
    from src.models.native_state import configuration_digest
    allowed={'schema','window','gamma','gamma_provenance','modes','scheme','plan_provenance',
             'restart_registry','segments','blocks'}
    if set(spec)!=allowed: raise ValueError('missing or unrecognized audit specification keys')
    if spec['schema']!='native-window-audit-spec-v1': raise ValueError('unsupported audit schema')
    config=WindowConfig(**spec['window']); blocks=[SplitBlock(**b) for b in spec['blocks']]
    registry=json.loads(Path(spec['restart_registry']).read_text())
    records=[]; rows=[]; plan={}
    for s in spec['segments']:
        if not s.get('configuration_provenance'): raise ValueError('fixed segment configuration provenance required')
        run=Path(s['run_path'])
        events=registry['runs'][run.name]['restart_events']
        rs,audit=read_segment(run,range(s['first'],s['last']+1),restart_events=events,
            gamma=spec['gamma'],gamma_provenance=spec['gamma_provenance'],modes=tuple(spec['modes']))
        for r in rs:
            r.initialization['configuration_provenance']=s['configuration_provenance']
        records.extend(rs); rows.extend(dict(run=run.name,**r) for r in audit)
        for p in s['plan_jorek']:
            for i in range(p['first'],p['last']+1):
                key=(run.name,i)
                if key in plan: raise ValueError('overlapping declared time plan')
                plan[key]=p['dt']*rs[0].state.units.t_norm_si
    collection=build_windows(records,blocks,config,time_plan=plan,
        plan_provenance=spec['plan_provenance'],scheme=spec['scheme'])
    split_records={}
    for w in collection.windows:
        split_records.setdefault(w.split,{})
        for r in w.support_records: split_records[w.split][r.run,r.state.step_index]=r
    scaler=NativeScaler.fit({s:list(rs.values()) for s,rs in split_records.items()})
    variations=parameter_variation({s:[r.parameters for r in rs.values()] for s,rs in split_records.items()})
    report=dict(schema='native-window-coverage-v2',spec_digest=configuration_digest(spec),
        scope=[dict(run_path=s['run_path'],first=s['first'],last=s['last']) for s in spec['segments']],
        window_counts=dict(Counter(w.split for w in collection.windows)),
        field_supervised_windows=sum(w.eligibility['field_supervised']['eligible'] for w in collection.windows),
        conditional_source_windows=sum(w.eligibility['conditional_source']['eligible'] for w in collection.windows),
        coupled_rollout_windows=sum(w.eligibility['self_generated_source']['eligible'] for w in collection.windows),
        qualification_reasons=dict(Counter(reason for w in collection.windows
            for q in w.eligibility.values() if not q['eligible'] for reason in q['reasons'])),
        exclusions=collection.exclusions,rejections=collection.rejections,raw_records=len(records),record_audit=rows,
        parameter_variation=variations,
        parameter_status_counts=dict(Counter(p['status'] for r in records for p in r.parameters)),
        gaps=['no verified coupled initialization/replay in dataset',
            'ordinary projection files lack saved raw deltas; lossless reconstruction remains evidence-dependent',
            'gather time absent; not inferred from fluid dt',
            'unresolved units/defaults/duplicate legacy parameters excluded from current inputs',
            'G/P and CfC are implemented; production source/BC and recoverable anchors remain uncertified'],
        conclusion='原生因果窗口与参数路由可用（已声明支持域）；不是C0闭环验证')
    # All computation/validation completes before publishing individual artifacts.
    # Each artifact is atomic; this is not a multi-file transaction protocol.
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    manifest=collection.manifest(); manifest['spec']=spec; manifest['spec_digest']=report['spec_digest']
    atomic_json(output/'manifest.json',manifest)
    scaler.save(output/'normalizer.json')
    atomic_json(output/'coverage.json',report)
    atomic_json(output/'parameter-routes.json',list(records[0].parameters))
    return report
