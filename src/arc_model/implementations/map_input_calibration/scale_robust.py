"""Raw-input normalization x angular-readout factorial; no auxiliary detector.

GroupNorm/LayerNorm remove running batch statistics. A cosine readout controls
feature-norm effects on logits. Neither technique is proposed as a new invention.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .deep_network import DeepCodedPhaseMap


class CosineLinear(nn.Linear):
    """One signed cosine direction, positive learned scale and a global bias.

    sigmoid(s * <h/||h||, w/||w||> + b), s=1+softplus(scale_raw).
    This is binary BCE classification, not a face-identification margin loss.
    """
    def __init__(self, previous, initial_scale=8.):
        super().__init__(previous.in_features, previous.out_features, bias=previous.bias is not None)
        with torch.no_grad():
            self.weight.copy_(previous.weight)
            if self.bias is not None:
                self.bias.copy_(previous.bias)
        self.scale_raw=nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale-1.))))

    def forward(self, x):
        scale=1+F.softplus(self.scale_raw)
        return scale*F.linear(F.normalize(x,p=2,dim=-1,eps=1e-6),
                              F.normalize(self.weight,p=2,dim=-1,eps=1e-6),None)+self.bias


class ScaleRobustPhaseMap(DeepCodedPhaseMap):
    def __init__(self, normalization='batch', head='linear'):
        super().__init__('raw')
        if normalization not in ('batch','sample') or head not in ('linear','cosine'):
            raise ValueError('Unknown factorial variant')
        self.normalization=normalization;self.head_type=head
        if normalization=='sample':
            for name in ('n1','n2','n3','ms_norm','n4','readout_norm'):
                previous=getattr(self,name)
                replacement=nn.LayerNorm(previous.num_features) if name=='readout_norm' else nn.GroupNorm(4,previous.num_features)
                with torch.no_grad():
                    replacement.weight.copy_(previous.weight);replacement.bias.copy_(previous.bias)
                setattr(self,name,replacement)
        if head=='cosine':
            self.readout=CosineLinear(self.readout)
