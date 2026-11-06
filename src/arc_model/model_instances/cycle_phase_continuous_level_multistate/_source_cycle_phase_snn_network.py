"""Cycle-budgeted direct classifier with two isolated information-preservation changes.

Analog phase encoding and analog readout surround a trainable spiking temporal
layer. This is an ANN--SNN hybrid, not a completely spike-only implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        x,=ctx.saved_tensors
        return gradient/(1+5*x.abs()).square()


class ChannelAffine(nn.Module):
    """Learned affine transform, without per-cycle centering or scaling."""
    def __init__(self, channels):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))

    def forward(self,x):
        shape=(1,-1,1) if x.ndim==3 else (1,-1)
        return x*self.weight.view(shape)+self.bias.view(shape)


class PhaseEncoder(nn.Module):
    def __init__(self, preserve_levels=False):
        super().__init__()
        norm=lambda c: ChannelAffine(c) if preserve_levels else nn.GroupNorm(4,c)
        self.stem=nn.Sequential(nn.Conv1d(5,24,1,bias=False),norm(24),nn.GELU())
        self.scales=nn.ModuleList([nn.Conv1d(24,24,k,padding=k//2,padding_mode='circular',groups=24,bias=False) for k in (3,7,15)])
        self.fuse=nn.Conv1d(72,32,1,bias=False);self.skip=nn.Conv1d(24,32,1,bias=False)
        self.norm=norm(32)

    def forward(self,x):
        h=self.stem(x)
        return F.avg_pool1d(F.gelu(self.norm(self.fuse(torch.cat([m(h) for m in self.scales],1))+self.skip(h))),4)


class CyclePhaseSNN(nn.Module):
    def __init__(self,spiking=True,preserve_levels=False,multistate=False):
        super().__init__()
        self.spiking=spiking;self.multistate=multistate;self.phase=PhaseEncoder(preserve_levels);self.input_map=nn.Linear(32,32)
        self.leak_logit=nn.Parameter(torch.full((32,),1.38629436112))
        self.threshold_raw=nn.Parameter(torch.full((32,),.54132485461))
        if multistate:
            weights=torch.zeros(9,32);weights[-1]=1.
            self.temporal_weights=nn.Parameter(weights)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,32),ChannelAffine(32) if preserve_levels else nn.LayerNorm(32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))

    def forward(self,x,mask,return_activity=False):
        batch,steps,channels,bins=x.shape
        if channels!=5 or bins!=64:raise ValueError('Expected [batch,real cycles,5,64]')
        h=self.phase(x.reshape(batch*steps,channels,bins)).reshape(batch,steps,32,16)
        h=h.permute(0,1,3,2)  # [batch, actual cycle, phase position, feature]
        leak=self.leak_logit.sigmoid()[None,None,:]
        threshold=(F.softplus(self.threshold_raw)+.05)[None,None,:]
        potential=torch.zeros_like(h[:,0]);trace=torch.zeros_like(potential);activity=[];states=[]
        for step in range(steps):
            valid=mask[:,step,None,None]
            potential=(leak*potential+self.input_map(h[:,step]))*valid
            activation=Spike.apply(potential-threshold) if self.spiking else torch.sigmoid(3*(potential-threshold))
            activation=activation*valid
            potential=(potential-threshold*activation.detach())*valid
            trace=(.8*trace+.2*activation)*valid
            activity.append(activation.detach().mean())
            if self.multistate:states.append(potential/threshold+trace)
        if self.multistate:
            if steps!=9:raise ValueError('Finite nine-cycle readout required')
            representation=(torch.stack(states,1)*self.temporal_weights[None,:,None,:]).sum(1).transpose(1,2)
        else:representation=(potential/threshold+trace).transpose(1,2)
        logit=self.head(representation).flatten()
        if return_activity:return logit,torch.stack(activity).mean()
        return logit
