"""Explicit node memory Adapter to ncps' original default CfC, no physical guarantees."""
import hashlib
import inspect
import math
from importlib.metadata import version
import torch
from torch import nn
from ncps.torch import CfCCell


class CfCTimeCore(nn.Module):
    """Interface: (node features, explicit memory, positive JU interval) -> output,memory."""
    def __init__(self,hidden,*,backbone_units=32,time_scale_ju=1.):
        super().__init__()
        if (type(hidden)!=int or hidden<1 or type(backbone_units)!=int or backbone_units<1
                or not math.isfinite(time_scale_ju) or time_scale_ju<=0):
            raise ValueError('invalid CfC configuration')
        if version('ncps')!='1.0.1': raise ValueError('validated ncps 1.0.1 required')
        self.hidden=hidden; self.backbone_units=backbone_units; self.time_scale_ju=float(time_scale_ju)
        self.cell=CfCCell(hidden,hidden,mode='default',backbone_activation='lecun_tanh',
            backbone_units=backbone_units,backbone_layers=1,backbone_dropout=0.).double()
        self.source_hash=hashlib.sha256(inspect.getsource(CfCCell).encode()).hexdigest()

    @property
    def checkpoint_configuration(self):
        return dict(adapter='native-cfc-v1',library='ncps',library_version=version('ncps'),
            source_sha256=self.source_hash,mode='default',backbone_activation='lecun_tanh',
            backbone_units=self.backbone_units,backbone_layers=1,backbone_dropout=0.,
            mixed_memory=False,hidden=self.hidden,memory_layout='node_hidden',
            time_scale_ju=self.time_scale_ju)

    def validate_memory(self,memory,like):
        if (not isinstance(memory,torch.Tensor) or memory.shape!=(len(like),self.hidden)
                or memory.dtype!=like.dtype or memory.device!=like.device or not torch.isfinite(memory).all()):
            raise ValueError('explicit finite CfC memory [N,hidden] matching dtype/device required')

    def forward(self,features,memory,*,dt_ju):
        if (not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=self.hidden
                or features.dtype!=self.cell.ff1.weight.dtype or features.device!=self.cell.ff1.weight.device
                or not torch.isfinite(features).all()):
            raise ValueError('invalid CfC features')
        self.validate_memory(memory,features)
        if not math.isfinite(dt_ju) or dt_ju<=0: raise ValueError('CfC interval must be positive')
        output,new_memory=self.cell(features,memory,dt_ju/self.time_scale_ju)
        if not torch.isfinite(output).all(): raise ValueError('nonfinite CfC output')
        return output,new_memory


class TimeConsistentResidualAdapter(nn.Module):
    """Outer learned rate/memory contract; NOT the official CfC formula.

    The decoder emits dimensionless coefficient rates per reference interval.
    Native rate is increment_scale / reference_time_ju (per JU).
    Only these learned rates are integrated here, never physical weak loads.
    """
    def __init__(self,reference_time_ju=1.):
        super().__init__()
        if not math.isfinite(reference_time_ju) or reference_time_ju<=0:
            raise ValueError('positive finite residual reference time required')
        self.reference_time_ju=float(reference_time_ju)

    @property
    def checkpoint_configuration(self):
        return dict(adapter='native-rate-memory-v2',decoder='dimensionless_rate',
            reference_time_ju=self.reference_time_ju,field_factor='dt_ju/reference_time_ju',
            memory_factor='-expm1(-dt_ju/reference_time_ju)',
            scale_source='train_unique_support_raw_history_scale',
            memory_layout='node_hidden',official_cfc_unchanged=True)

    def memory_update(self,memory,candidate,*,dt_ju):
        gate=-math.expm1(-dt_ju/self.reference_time_ju)
        return memory+gate*(candidate-memory)

    def integrate_rate(self,rate,increment_scale,*,dt_ju):
        return rate*increment_scale*(dt_ju/self.reference_time_ju)
