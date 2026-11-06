"""Continuous-valued compact CNN followed by16 LIF units over physical cycles."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):ctx.save_for_backward(x);return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors;return g/(1+5*x.abs()).square()


class CompactHybrid(nn.Module):
    spiking=True
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv1d(5,16,5,padding=2,padding_mode='circular',bias=False);self.n1=nn.BatchNorm1d(16)
        self.dw1=nn.Conv1d(16,16,7,padding=3,padding_mode='circular',groups=16,bias=False)
        self.pw1=nn.Conv1d(16,24,1,bias=False);self.n2=nn.BatchNorm1d(24)
        self.dw2=nn.Conv1d(24,24,5,padding=2,padding_mode='circular',groups=24,bias=False)
        self.pw2=nn.Conv1d(24,16,1,bias=False);self.n3=nn.BatchNorm1d(16)
        self.input_map=nn.Linear(128,16,bias=False);self.n4=nn.BatchNorm1d(16)
        self.leak_logit=nn.Parameter(torch.full((16,),math.log(4)))
        self.threshold_raw=nn.Parameter(torch.full((16,),math.log(math.expm1(.45))))
        self.readout=nn.Linear(16,1)
    def forward(self,x,mask,return_activity=False):
        b,k,_,p=x.shape
        if (k,p)!=(9,64):raise ValueError('Nine physical cycles and64 bins required')
        z=x[:,:,:5]*mask[:,:,None,None]
        z=F.gelu(self.n1(self.c1(z.reshape(b*k,5,64))))
        z=F.avg_pool1d(F.gelu(self.n2(self.pw1(self.dw1(z)))),4)
        z=F.avg_pool1d(F.gelu(self.n3(self.pw2(self.dw2(z)))),2)
        current=self.n4(self.input_map(z.reshape(b*k,128))).reshape(b,k,16)
        leak=self.leak_logit.sigmoid();threshold=F.softplus(self.threshold_raw)+.05
        potential=torch.zeros_like(current[:,0]);trace=torch.zeros_like(potential);events=[]
        for j in range(9):
            valid=mask[:,j,None];u=(potential*leak+current[:,j])*valid
            s=Spike.apply(u-threshold)*valid
            potential=(u-threshold*s.detach())*valid;trace=(.8*trace+.2*s)*valid;events.append(s)
        logits=self.readout(potential/threshold+trace).flatten()
        activity=torch.stack(events,1).sum()/(mask.sum().clamp_min(1)*16)
        return (logits,activity) if return_activity else logits
