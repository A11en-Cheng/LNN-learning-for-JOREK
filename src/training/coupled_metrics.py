"""Real-node metrics and nine-field data loss. Pearson r is never R-squared."""
from dataclasses import dataclass
import math
import numpy as np
import torch

FIELDS=('psi','Phi','Z_j','omega','rho','T_i','v_par','rho_imp','T_e')
CHANNELS=('dE_ion','rad_sink','Z_eff','Z_imp','N_imp')
TERMS=('coefficients','raw_history','gauss_values','compatible_gradients')


def array(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)


def weighted_errors(prediction,target,*,weights=None,near_zero_rms=0.):
    p=np.asarray(array(prediction),dtype=float);t=np.asarray(array(target),dtype=float)
    if p.shape!=t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('matched finite prediction/target required; no dropped nonfinite samples')
    if not math.isfinite(near_zero_rms) or near_zero_rms<0:raise ValueError('explicit nonnegative native-unit threshold required')
    w=np.ones(p.shape) if weights is None else np.broadcast_to(np.asarray(array(weights),dtype=float),p.shape)
    if not np.isfinite(w).all() or (w<0).any():raise ValueError('finite nonnegative metric weights required')
    p=p.ravel();t=t.ravel();w=w.ravel();keep=w>0;p=p[keep];t=t[keep];w=w[keep]
    result=dict(status='empty',count=len(p),weight_sum=float(w.sum()),pearson_r=None,
        correlation_status='empty',relative_l2=None,relative_status='empty',mae=None,rmse=None)
    if not len(p):return result
    w=w/w.max();w=w/w.sum();diff=p-t
    if not np.isfinite(diff).all():raise FloatingPointError('unrepresentable absolute prediction error')
    # Rescale before squaring: no epsilon, and tiny nonzero targets stay nonzero.
    def norm(x):
        scale=float(np.max(np.abs(x)))
        return 0. if scale==0 else scale*float(np.sqrt(np.dot(w,(x/scale)**2)))
    ps=float(np.max(np.abs(p)));ts=float(np.max(np.abs(t)))
    pp=p if ps==0 else p/ps;tt0=t if ts==0 else t/ts
    pt=pp-float(np.dot(w,pp));tt=tt0-float(np.dot(w,tt0))
    pv=0. if np.all(p==p[0]) else float(np.dot(w,pt*pt))
    tv=0. if np.all(t==t[0]) else float(np.dot(w,tt*tt))
    ref=norm(t)
    r=None if pv==0 or tv==0 else float(np.clip(np.dot(w,pt*tt)/np.sqrt(pv*tv),-1.,1.))
    rms=norm(diff)
    return dict(result,status='defined',pearson_r=r,
        correlation_status='constant_prediction_and_target' if pv==tv==0 else
            ('constant_prediction' if pv==0 else ('constant_target' if tv==0 else 'defined')),
        relative_l2=None if ref<=near_zero_rms else rms/ref,
        relative_status='zero_reference' if ref==0 else ('near_zero_reference' if ref<=near_zero_rms else 'defined'),
        mae=float(np.dot(w,np.abs(diff))),rmse=rms)


def coefficient_weights(indices,n_nodes):
    """Each unique DOF per degree counts once; absent IDs means storage-node mean."""
    if indices is None:return np.ones((4,n_nodes))
    ids=np.asarray(indices)
    if ids.shape!=(4,n_nodes) or ids.dtype.kind not in 'iu':raise ValueError('native shared indices [4,N] required')
    out=np.empty_like(ids,dtype=float)
    for d,row in enumerate(ids):
        _,inverse,count=np.unique(row,return_inverse=True,return_counts=True)
        out[d]=1/count[inverse]
    return out


@dataclass
class NativeLoss:
    total: torch.Tensor
    unweighted: dict
    physical: dict
    weighted: dict
    per_field_contribution: torch.Tensor
    metadata: dict

    def report(self):
        to_list=lambda d:{k:v.detach().cpu().tolist() for k,v in d.items()}
        return dict(total=float(self.total.detach().cpu()),fields=list(FIELDS),
            unweighted=to_list(self.unweighted),native_unit_squared_mse=to_list(self.physical),
            weighted_contributions=to_list(self.weighted),
            per_field_total_contribution=self.per_field_contribution.detach().cpu().tolist(),**self.metadata)


def native_supervision(prediction,target,*,evaluator,scales,weights,region=None,indices=None):
    """Complete axes and compatible observations; labels never replace prediction."""
    if set(weights)!=set(TERMS) or any(not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('four explicit finite nonnegative data-loss weights required')
    if not scales.get('provenance'):raise ValueError('train-only or explicit fixture scale provenance required')
    if (prediction['units']!=target['units'] or abs(prediction['time_si']-target['time_si'])>1e-14):
        raise ValueError('loss target time/units mismatch')
    p=prediction['values']
    if not isinstance(p,torch.Tensor) or p.ndim!=4 or p.shape[:3]!=(9,4,1):
        raise ValueError('Torch full native nine fields required')
    tensor=lambda x:torch.as_tensor(x,dtype=p.dtype,device=p.device)
    t=tensor(target['values']).detach();d=prediction['raw_deltas'];td=tensor(target['raw_deltas']).detach()
    if any(x.shape!=p.shape or not torch.isfinite(x).all() for x in (p,t,d,td)):
        raise ValueError('complete finite values and independently saved rawD labels required')
    s={k:tensor(scales[k]) for k in TERMS}
    expected={'coefficients':(9,4,1,1),'raw_history':(9,4,1,1),'gauss_values':(9,),'compatible_gradients':(9,2)}
    if any(s[k].shape!=expected[k] or not torch.isfinite(s[k]).all() or (s[k]<=0).any() for k in TERMS):
        raise ValueError('positive full-field/degree/direction scales required')
    gauss=evaluator(p,units=prediction['units'])
    with torch.no_grad():truth=evaluator(t,units=target['units'])
    volume=evaluator.volume
    if region is not None:
        mask=torch.as_tensor(region,dtype=torch.bool,device=p.device)
        if mask.shape!=volume.shape:raise ValueError('explicit Gauss region shape mismatch')
        volume=volume*mask
    if not torch.isfinite(volume).all() or (volume<0).any() or volume.sum()<=0:
        raise ValueError('positive nonempty physical-volume region required for loss')
    nw=tensor(coefficient_weights(indices,p.shape[-1]))
    delta={'coefficients':p-t,'raw_history':d-td,
        'gauss_values':gauss.values-truth.values,'compatible_gradients':gauss.gradients-truth.gradients}
    def reduce(k,x):
        if k in ('coefficients','raw_history'):
            return (x.mean(2)*nw[None]).sum(-1)/nw.sum(-1)[None]
        extra=volume[...,None] if k=='gauss_values' else volume[...,None,None]
        return (x*extra).sum((0,1))/volume.sum()
    physical={k:reduce(k,x.square()) for k,x in delta.items()}
    normalized={k:reduce(k,(x/s[k]).square()) for k,x in delta.items()}
    # Each entry is its actual contribution to total, including field/group means.
    weighted={k:v*weights[k]/v.numel() for k,v in normalized.items()}
    per_field=sum(v.reshape(9,-1).sum(1) for v in weighted.values())
    total=per_field.sum()
    if not torch.isfinite(total):raise FloatingPointError('nonfinite data loss')
    return NativeLoss(total,normalized,physical,weighted,per_field,
        dict(scale_provenance=scales['provenance'],spatial_region='all' if region is None else 'explicit_Gauss_mask',
            volume_weighting='R_absJ_Gauss',coefficient_weighting='storage_node_mean' if indices is None else 'unique_DOF_per_degree',
            time_aggregation='one_real_node',loss_weights=dict(weights),no_pde_residual=True))


def shard_position_metrics(prediction,target):
    """Explicit Cartesian meters and persistent IDs, including unmatched identities."""
    def read(s):
        if not {'ids','positions_m','alive'}<=set(s):raise ValueError('fixed shard IDs, Cartesian meters and liveness required')
        ids=list(s['ids']);pos=np.asarray(array(s['positions_m']),dtype=float);alive=np.asarray(s['alive'],dtype=bool)
        if len(set(ids))!=len(ids) or pos.shape!=(len(ids),3) or alive.shape!=(len(ids),) or not np.isfinite(pos).all():
            raise ValueError('invalid fixed-identity shard positions')
        return {identity:(pos[i],bool(alive[i]),i) for i,identity in enumerate(ids)}
    p=read(prediction);t=read(target);matched=sorted(set(p)&set(t))
    pp=np.array([p[i][0] for i in matched]).reshape(-1,3);tt=np.array([t[i][0] for i in matched]).reshape(-1,3)
    delta=pp-tt
    result=dict(status='matched_subset' if matched else 'empty_matched_subset',matched_ids=matched,
        prediction_count=len(p),target_count=len(t),matched_count=len(matched),
        unmatched_prediction=sorted(set(p)-set(t)),unmatched_target=sorted(set(t)-set(p)),
        liveness_mismatch=[i for i in matched if p[i][1]!=t[i][1]],
        coordinates={name:weighted_errors(pp[:,j],tt[:,j]) for j,name in enumerate(('x_m','y_m','z_m'))},
        matched_position_rmse_m=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))) if matched else None,
        position_error_scope='all_matched_identities_including_liveness_mismatches; unmatched separately counted',
        displacement_rmse_m=None,
        trajectory_samples=[dict(id=i,prediction=p[i][0].tolist(),target=t[i][0].tolist()) for i in matched])
    if 'initial_positions_m' in prediction and 'initial_positions_m' in target:
        pi=np.asarray(array(prediction['initial_positions_m']));ti=np.asarray(array(target['initial_positions_m']))
        if pi.shape!=(len(p),3) or ti.shape!=(len(t),3) or not np.isfinite(pi).all() or not np.isfinite(ti).all():
            raise ValueError('complete initial identity positions required for displacement')
        dp=np.array([pp[j]-pi[p[i][2]] for j,i in enumerate(matched)]).reshape(-1,3)
        dt=np.array([tt[j]-ti[t[i][2]] for j,i in enumerate(matched)]).reshape(-1,3)
        result['displacement_rmse_m']=float(np.sqrt(np.mean(np.sum((dp-dt)**2,axis=1)))) if matched else None
    return result


def coupled_node_metrics(prediction,target,*,evaluator,normalization_source,region_name,
                         region=None,indices=None,near_zero_rms=None):
    """One real node, separate dual sources, full field degree/Gauss/gradient axes."""
    if not normalization_source or not region_name:raise ValueError('metric normalization and spatial region provenance required')
    thresholds={} if near_zero_rms is None else near_zero_rms
    volume=array(evaluator.volume)
    if region is not None:
        mask=np.asarray(region,dtype=bool)
        if mask.shape!=volume.shape:raise ValueError('metric region shape mismatch')
        volume=volume*mask
    fields={'status':'missing'};like=evaluator.basis
    tensor=lambda a:torch.as_tensor(a,dtype=like.dtype,device=like.device)
    def gauss(a):
        with torch.no_grad():return evaluator(tensor(a),units=evaluator.units)
    def errors(p,t,w,name):return weighted_errors(p,t,weights=w,near_zero_rms=thresholds.get(name,0.))
    if 'field' in prediction and 'field' in target:
        p=prediction['field'];t=target['field']
        if p['units']!=t['units'] or abs(p['time_si']-t['time_si'])>1e-14:raise ValueError('metric node units/time mismatch')
        pu=array(p['values']);tu=array(t['values']);pg=gauss(pu);tg=gauss(tu)
        nw=coefficient_weights(indices,pu.shape[-1])
        fields=dict(coefficients={},raw_history={},gauss_values={},compatible_gradients={})
        for f,name in enumerate(FIELDS):
            fields['coefficients'][name]=[errors(pu[f,d],tu[f,d],nw[d][None],name) for d in range(4)]
            if p.get('raw_deltas') is not None and t.get('raw_deltas') is not None:
                fields['raw_history'][name]=[errors(array(p['raw_deltas'])[f,d],array(t['raw_deltas'])[f,d],nw[d][None],name) for d in range(4)]
            else:fields['raw_history'][name]={'status':'missing'}
            fields['gauss_values'][name]=errors(pg.values[:,:,f],tg.values[:,:,f],volume,name)
            fields['compatible_gradients'][name]={direction:errors(pg.gradients[:,:,f,j],tg.gradients[:,:,f,j],volume,name+'_per_m')
                for j,direction in enumerate(('dR','dZ'))}
    def source_metrics(key,names):
        if key not in prediction or key not in target:return {'status':'missing'}
        clock_key='marker_time_si' if key=='marker_projection' else 'shard_density_time_si'
        if clock_key not in prediction or clock_key not in target or abs(prediction[clock_key]-target[clock_key])>1e-14:
            raise ValueError('source metric requires matched object event times')
        p=array(prediction[key]);t=array(target[key])
        if p.shape!=t.shape or p.shape[:3]!=(len(names),4,1):raise ValueError('full native source axes required')
        a=like.new_zeros((9,*p.shape[1:]));b=torch.zeros_like(a)
        a[:len(names)]=tensor(p);b[:len(names)]=tensor(t)
        ag=gauss(a);bg=gauss(b);nw=coefficient_weights(indices,p.shape[-1])
        return {name:dict(gauss_values=errors(ag.values[:,:,f],bg.values[:,:,f],volume,name),
            native_degree=[errors(p[f,d],t[f,d],nw[d][None],name) for d in range(4)]) for f,name in enumerate(names)}
    if 'shard_positions' in prediction and 'shard_positions' in target:
        if ('shard_time_si' not in prediction or 'shard_time_si' not in target
            or not all(math.isfinite(o['shard_time_si']) for o in (prediction,target))
            or abs(prediction['shard_time_si']-target['shard_time_si'])>1e-14):
            raise ValueError('shard metrics require matched object times')
    shards=shard_position_metrics(prediction['shard_positions'],target['shard_positions']) if (
        'shard_positions' in prediction and 'shard_positions' in target) else {'status':'missing'}
    return dict(fields=fields,marker_projection=source_metrics('marker_projection',CHANNELS),
        shard_density=source_metrics('shard_density',('main_mass_density_rate','impurity_mass_density_rate')),
        shard_positions=shards,normalization_source=normalization_source,spatial_region=region_name,
        volume_weighting='R_absJ_Gauss',coefficient_region='full_native_support',
        coefficient_weighting='storage_node_mean' if indices is None else 'unique_DOF_per_degree',
        units=dict(coefficients='JOREK_native_units_per_field_and_degree',gauss_values='JOREK_native_field_units',
            compatible_gradients='JOREK_native_field_units_per_m',marker_projection='recorded_projection_channel_units',
            shard_density='model502_mass_density_per_JU',shard_positions='m'),
        correlation_definition='weighted_Pearson_r_not_R_squared',time_si=prediction.get('field',{}).get('time_si'))


def recorded_supervision(prediction,node,mask,*,evaluator,scales,weights,normalization_source,
                         region_name,region=None,indices=None):
    """Unified real-node entry: data loss and all requested object metrics."""
    target={k:v for k,v in node.targets.items() if mask.get(k,False)}
    if 'marker_projection' in target:
        target['marker_time_si']=node.object_times_si['marker_projection']
    if 'shard_density' in target:
        target['shard_density_time_si']=node.object_times_si['shard_density']
    if 'shard_positions' in target:
        if 'shard_positions' not in node.object_times_si:raise ValueError('shard label object time required')
        target['shard_time_si']=node.object_times_si['shard_positions']
    loss=native_supervision(prediction['field'],target['field'],evaluator=evaluator,scales=scales,
        weights=weights,region=region,indices=indices) if 'field' in target else None
    metrics=coupled_node_metrics(prediction,target,evaluator=evaluator,normalization_source=normalization_source,
        region_name=region_name,region=region,indices=indices)
    return dict(loss=loss,metrics=metrics)


class MetricGroups:
    required=('run','task','burnin_time_si','burnin_record_count','prediction_time_si','dt_si','dt_ratio','teacher_policy')

    def __init__(self):self.rows={};self.trajectories={}

    def add(self,group,metrics,*,trajectory_id=None):
        if set(group)!=set(self.required):raise ValueError('complete metric grouping keys required')
        key=tuple(group[k] for k in self.required)
        self.rows.setdefault(key,[]).append(metrics)
        if trajectory_id is not None:
            trajectory_key=(group['run'],group['task'],group['burnin_time_si'],group['burnin_record_count'],group['teacher_policy'],trajectory_id)
            self.trajectories.setdefault(trajectory_key,[]).append(metrics)

    def trajectory_report(self):
        """Across real times of one explicitly identified trajectory, not windows pooled by accident."""
        result=[]
        for key,rows in self.trajectories.items():
            samples={}
            for row in rows:
                time=row['time_si']
                if time is None:raise ValueError('trajectory metrics require actual physical time')
                for sample in row['shard_positions'].get('trajectory_samples',[]):
                    track=samples.setdefault(sample['id'],{})
                    if time in track and track[time]!=sample:raise ValueError('conflicting predictions at same trajectory time')
                    track[time]=sample
            tracks={identity:dict(times_si=sorted(track),
                coordinates={direction:weighted_errors([track[t]['prediction'][j] for t in sorted(track)],
                    [track[t]['target'][j] for t in sorted(track)]) for j,direction in enumerate(('x_m','y_m','z_m'))})
                for identity,track in samples.items()}
            result.append(dict(run=key[0],task=key[1],burnin_time_si=key[2],burnin_record_count=key[3],
                teacher_policy=key[4],trajectory_id=key[5],shards=tracks,
                time_aggregation='equal_real_times_per_fixed_identity',node_count=len(rows)))
        return result

    def report(self):
        result=[]
        metric_names={'pearson_r','relative_l2','mae','rmse','matched_position_rmse_m','displacement_rmse_m'}
        for key,rows in self.rows.items():
            flattened={}
            def walk(value,path=()):
                if isinstance(value,dict):
                    for k,v in value.items():
                        if k in metric_names:flattened.setdefault('/'.join((*path,k)),[]).append(v)
                        else:walk(v,(*path,k))
                elif isinstance(value,list):
                    for i,v in enumerate(value):walk(v,(*path,str(i)))
            for row in rows:
                walk(row)
            means={path:dict(mean=float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else None,
                defined_count=sum(v is not None for v in vs),undefined_count=sum(v is None for v in vs)) for path,vs in flattened.items()}
            result.append(dict(group=dict(zip(self.required,key)),node_count=len(rows),nodes=rows,
                means=means,time_aggregation='equal_record_nodes',
                caveat='mean_node_Pearson_is_not_pooled_Pearson; no implicit physical_time_weighting'))
        return result
