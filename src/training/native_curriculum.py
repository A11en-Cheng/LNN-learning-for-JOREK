"""Budgeted field-only trajectory supervision. No source consumer or PDE residual."""
from dataclasses import dataclass,asdict
import math
import torch

SCOPE='field_only_no_feedback_consumer'


@dataclass(frozen=True)
class LossWeights:
    coefficients: float
    raw_history: float
    gauss_values: float
    compatible_gradients: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x<0 for x in asdict(self).values()) or not sum(asdict(self).values()):
            raise ValueError('explicit finite nonnegative loss weights required')


@dataclass
class Supervision:
    total: torch.Tensor
    terms: dict
    weights: LossWeights
    groups: dict
    physical_mse: dict

    def metrics(self):
        scalar=lambda a:float(a.detach().cpu())
        unweighted={k:scalar(v) for k,v in self.terms.items()}
        return dict(scope=SCOPE,total=scalar(self.total),unweighted=unweighted,
            weighted={k:v*getattr(self.weights,k) for k,v in unweighted.items()},
            normalized_group_mse={k:v.detach().cpu().tolist() for k,v in self.groups.items()},
            native_unit_squared_mse={k:v.detach().cpu().tolist() for k,v in self.physical_mse.items()})


def field_supervision(model,prediction,target,weights):
    """True rawD labels, compatible B/G on actual predicted/recorded coefficients."""
    target.validate();prediction.validate_for_model(model)
    if (prediction.units!=target.units or target.modes!=(0,)
        or prediction.step_index!=target.step_index
        or abs(prediction.time_si-target.time_si)>max(1e-14,target.units.t_norm_si*1e-7)
        or model.feature_path is None):
        raise ValueError('supervision state/time/feature contract mismatch')
    u=torch.as_tensor(target.values,dtype=prediction.values.dtype,device=prediction.values.device)
    d=torch.as_tensor(target.raw_deltas,dtype=u.dtype,device=u.device)
    if u.shape!=prediction.values.shape: raise ValueError('target full axes mismatch')
    ev=model.feature_path.evaluator
    predicted=ev(prediction.values,units=prediction.units)
    with torch.no_grad(): truth=ev(u,units=target.units)
    du=prediction.values-u;dd=prediction.raw_deltas-d
    dv=predicted.values-truth.values;dg=predicted.gradients-truth.gradients
    volume=ev.volume;norm=volume.sum()
    groups=dict(coefficients=(du/model.state_scale).square().mean((2,3)),
        raw_history=(dd/model.increment_scale).square().mean((2,3)),
        gauss_values=((dv/model.feature_path.value_scale).square()*volume[...,None]).sum((0,1))/norm,
        compatible_gradients=((dg/model.feature_path.gradient_scale).square()*volume[...,None,None]).sum((0,1))/norm)
    physical=dict(coefficients=du.square().mean((2,3)),raw_history=dd.square().mean((2,3)),
        gauss_values=(dv.square()*volume[...,None]).sum((0,1))/norm,
        compatible_gradients=(dg.square()*volume[...,None,None]).sum((0,1))/norm)
    terms={k:v.mean() for k,v in groups.items()}
    total=sum(v*getattr(weights,k) for k,v in terms.items())
    if not torch.isfinite(total): raise FloatingPointError('nonfinite supervision loss')
    return Supervision(total,terms,weights,groups,physical)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    updates: int
    horizon: int
    teacher_probability: float
    detach_before: tuple=()

    def __post_init__(self):
        if (not self.name or type(self.updates)!=int or self.updates<1 or self.horizon not in (1,2,4)
            or not math.isfinite(self.teacher_probability) or not 0<=self.teacher_probability<=1
            or len(set(self.detach_before))!=len(self.detach_before)
            or any(type(i)!=int or not 0<=i<self.horizon for i in self.detach_before)):
            raise ValueError('invalid explicit curriculum stage')


class NativeCurriculumTrainer:
    """Optimizer-boundary Interface; batch=one complete existing causal window.

    Separate private sampler/mask RNG. No hidden field state continuation across
    independent sampled windows: each performs its declared causal warm-up.
    """
    def __init__(self,model,window_banks,*,weights,stages,seed,learning_rate,lr_decay,max_grad_norm,experiment_configuration):
        from src.models.native_state import configuration_digest
        self.model=model;self.window_banks=window_banks;self.weights=weights;self.stages=tuple(stages)
        if (not self.stages or sum(s.updates for s in self.stages)>200
            or any(s.horizon not in window_banks or not window_banks[s.horizon] for s in self.stages)
            or any(w.split!='train' or w.config.source_teacher_policy!='none' or w.config.horizon!=k
                for k,ws in window_banks.items() for w in ws)):
            raise ValueError('train-only banks/field-only stages and <=200 budget required')
        if (not math.isfinite(learning_rate) or learning_rate<=0 or not 0<lr_decay<=1
            or not math.isfinite(max_grad_norm) or max_grad_norm<=0):
            raise ValueError('invalid optimizer configuration')
        if experiment_configuration.get('scope')!=SCOPE: raise ValueError('explicit field-only scope required')
        self.learning_rate=learning_rate;self.lr_decay=lr_decay;self.max_grad_norm=max_grad_norm
        # Bind labels too: modified arrays with unchanged provenance must not resume silently.
        import hashlib
        unique={(w.run,r.state.step_index):r.state for ws in window_banks.values() for w in ws for r in w.support_records}
        data_hash=hashlib.sha256()
        for key,state in sorted(unique.items()):
            data_hash.update(str(key).encode());data_hash.update(state.values.tobytes());data_hash.update(state.raw_deltas.tobytes())
        self.configuration=dict(scope=SCOPE,version='native-curriculum-v1',model=model.checkpoint_configuration,
            weights=asdict(weights),stages=[asdict(s) for s in stages],seed=seed,
            optimizer={'kind':'AdamW','lr':learning_rate,'weight_decay':0.},
            scheduler={'kind':'ExponentialLR','gamma':lr_decay},max_grad_norm=max_grad_norm,
            experiment=experiment_configuration,data_sha256=data_hash.hexdigest(),
            bank_manifest=configuration_digest({str(k):[w.manifest() for w in ws] for k,ws in window_banks.items()}))
        self.configuration_digest=configuration_digest(self.configuration)
        self.generator=torch.Generator().manual_seed(seed)
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.)
        self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=lr_decay)
        self.update_count=0;self.history=[];self.last_state=None;self.in_update=False

    @property
    def stage_position(self):
        cursor=self.update_count
        for i,s in enumerate(self.stages):
            if cursor<s.updates:return i,cursor
            cursor-=s.updates
        return len(self.stages),0

    def advance(self):
        from src.models.native_torch_rollout import rollout_field_window
        stage_index,position=self.stage_position
        if stage_index==len(self.stages):raise ValueError('optimizer update budget exhausted')
        if self.in_update:raise ValueError('update already active')
        stage=self.stages[stage_index];bank=self.window_banks[stage.horizon]
        index=0 if self.update_count==0 else int(torch.randint(len(bank),(1,),generator=self.generator))
        mask=tuple(bool(x<stage.teacher_probability) for x in torch.rand(stage.horizon,generator=self.generator))
        window=bank[index]
        self.in_update=True
        try:
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            result=rollout_field_window(self.model,window,teacher_mask=mask,detach_before=stage.detach_before)
            losses=[field_supervision(self.model,s,t,self.weights) for s,t in zip(result.states,window.targets)]
            total=torch.stack([l.total for l in losses]).mean()
            total.backward()
            params=list(self.model.parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('missing/nonfinite parameter gradient; update refused')
            norm=torch.nn.utils.clip_grad_norm_(params,self.max_grad_norm,error_if_nonfinite=True)
            before=[p.detach().clone() for p in params]
            lr=self.optimizer.param_groups[0]['lr']
            self.optimizer.step();self.scheduler.step()
            if any(not torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite optimizer result')
            change=sum((p.detach()-old).square().sum() for p,old in zip(params,before)).sqrt()
            self.update_count+=1
            self.last_state=result.states[-1].truncated()
            record=dict(scope=SCOPE,update=self.update_count,stage=stage.name,stage_index=stage_index,
                stage_position=position,window_index=index,run=window.run,prediction_index=window.prediction_index,
                horizon=stage.horizon,teacher_mask=list(mask),source_policy='none',
                detach_before=list(stage.detach_before),loss=float(total.detach()),learning_rate=lr,
                gradient_norm=float(norm),parameter_change_l2=float(change),
                losses=[l.metrics() for l in losses],
                decoder_field_degree_gradient_coverage=(self.model.decoder.weight.grad.reshape(9,4,-1).abs().sum(-1)>0).tolist())
            self.history.append(record)
            return record
        finally:self.in_update=False

    def save(self,path):
        """Trusted-local complete training checkpoint, only at optimizer boundaries."""
        from copy import deepcopy
        from pathlib import Path
        import os,random,tempfile
        import numpy as np
        from src.models.native_state import configuration_digest
        if self.in_update:raise ValueError('checkpoint requires optimizer boundary')
        if configuration_digest(self.model.checkpoint_configuration)!=configuration_digest(self.configuration['model']):
            raise ValueError('model configuration changed')
        if self.last_state is not None:self.last_state.validate_for_model(self.model)
        record=dict(schema='native-field-training-v1',scope=SCOPE,configuration=self.configuration,
            configuration_digest=self.configuration_digest,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
            update_count=self.update_count,stage_position=self.stage_position,history=self.history,
            last_state=self.last_state,training=self.model.training,
            rng=dict(sampler=self.generator.get_state(),python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None))
        # Same-directory replacement, including failed serialization protection.
        path=Path(path);fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
        try:
            with os.fdopen(fd,'wb') as f:
                torch.save(record,f);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)

    def load(self,path):
        """Validate on private model/optimizer/RNG before committing anything."""
        from copy import deepcopy
        import random,json
        import numpy as np
        from src.models.native_state import configuration_digest
        if self.in_update:raise ValueError('restore requires optimizer boundary')
        r=torch.load(path,weights_only=False,map_location='cpu')
        required={'schema','scope','configuration','configuration_digest','model','optimizer','scheduler',
            'update_count','stage_position','history','last_state','training','rng'}
        if not isinstance(r,dict) or set(r)!=required or r['schema']!='native-field-training-v1' or r['scope']!=SCOPE:
            raise ValueError('invalid training checkpoint schema')
        if (r['configuration_digest']!=self.configuration_digest
            or configuration_digest(r['configuration'])!=self.configuration_digest):
            raise ValueError('training configuration mismatch')
        count=r['update_count']
        if type(count)!=int or not 0<=count<=sum(s.updates for s in self.stages) or type(r['training'])!=bool:
            raise ValueError('invalid training checkpoint position')
        pos=count
        expected=(len(self.stages),0)
        for i,s in enumerate(self.stages):
            if pos<s.updates:expected=(i,pos);break
            pos-=s.updates
        if tuple(r['stage_position'])!=expected or len(r['history'])!=count:
            raise ValueError('invalid curriculum/history position')
        if any(v.get('update')!=i+1 or v.get('scope')!=SCOPE for i,v in enumerate(r['history'])):
            raise ValueError('invalid training history')
        try:json.dumps(r['history'],allow_nan=False)
        except (TypeError,ValueError) as e:raise ValueError('corrupt training history') from e
        model=deepcopy(self.model)
        target=model.state_dict()
        if set(r['model'])!=set(target):raise ValueError('invalid model tensors')
        for name,tensor in r['model'].items():
            if tensor.dtype!=target[name].dtype or tensor.shape!=target[name].shape or not torch.isfinite(tensor).all():
                raise ValueError('invalid model tensor shape/dtype/finiteness')
        model.load_state_dict(r['model'])
        if configuration_digest(model.checkpoint_configuration)!=configuration_digest(self.configuration['model']):
            raise ValueError('corrupt model statistics/configuration')
        if count:
            if r['last_state'] is None:raise ValueError('missing complete boundary field state')
            r['last_state'].validate_for_model(model)
        elif r['last_state'] is not None:raise ValueError('unexpected boundary field state')
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.learning_rate,weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=self.lr_decay)
        expected_groups=deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(r['optimizer']);scheduler.load_state_dict(r['scheduler'])
        if len(optimizer.param_groups)!=len(expected_groups):raise ValueError('corrupt optimizer groups')
        for actual,expected_group in zip(optimizer.param_groups,expected_groups):
            if (set(actual)!=set(expected_group) or any(actual[k]!=expected_group[k]
                for k in actual if k not in ('params','lr'))):
                raise ValueError('corrupt optimizer policy')
        if (scheduler.gamma!=self.lr_decay or scheduler.base_lrs!=[self.learning_rate]
            or scheduler._step_count!=count+1):
            raise ValueError('corrupt scheduler policy')
        if scheduler.last_epoch!=count or not math.isclose(optimizer.param_groups[0]['lr'],
            self.learning_rate*self.lr_decay**count,rel_tol=1e-12):
            raise ValueError('corrupt scheduler/optimizer clock')
        for p in model.parameters():
            st=optimizer.state.get(p)
            if count==0:
                if st:raise ValueError('unexpected optimizer history')
                continue
            if not st or set(st)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('missing Adam state')
            for k in ('exp_avg','exp_avg_sq'):
                if st[k].shape!=p.shape or st[k].dtype!=p.dtype or not torch.isfinite(st[k]).all():
                    raise ValueError('corrupt optimizer tensor')
            if (st['exp_avg_sq']<0).any() or float(st['step'])!=count:raise ValueError('corrupt Adam history')
        rng=r['rng']
        generator=torch.Generator();generator.set_state(rng['sampler'])
        random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
        torch.Generator().set_state(rng['torch'])
        if rng['cuda'] is not None and (not torch.cuda.is_available() or len(rng['cuda'])!=torch.cuda.device_count()):
            raise ValueError('unsupported CUDA RNG restore')
        backup=(deepcopy(self.model.state_dict()),deepcopy(self.optimizer.state_dict()),
            deepcopy(self.scheduler.state_dict()),self.model.training,self.generator.get_state(),
            self.update_count,self.history,self.last_state,
            random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        try:
            self.model.load_state_dict(r['model']);self.optimizer.load_state_dict(r['optimizer'])
            self.scheduler.load_state_dict(r['scheduler']);self.model.train(r['training'])
            self.generator.set_state(rng['sampler']);self.update_count=count
            self.history=deepcopy(r['history']);self.last_state=r['last_state']
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None:torch.cuda.set_rng_state_all(rng['cuda'])
        except Exception:
            self.model.load_state_dict(backup[0]);self.optimizer.load_state_dict(backup[1])
            self.scheduler.load_state_dict(backup[2]);self.model.train(backup[3])
            self.generator.set_state(backup[4]);self.update_count,self.history,self.last_state=backup[5:8]
            random.setstate(backup[8]);np.random.set_state(backup[9]);torch.set_rng_state(backup[10])
            if backup[11] is not None:torch.cuda.set_rng_state_all(backup[11])
            raise
