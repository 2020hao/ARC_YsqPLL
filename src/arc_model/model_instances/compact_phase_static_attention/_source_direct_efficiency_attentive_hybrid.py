"""Independent six-parameter local pooling candidates; no teacher or score fusion."""
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


class AttentiveHybrid(nn.Module):
    spiking=True
    def __init__(self,pooling="content"):
        super().__init__()
        if pooling not in ("content","static"):raise ValueError("Unknown pooling")
        self.pooling=pooling
        self.c1=nn.Conv1d(5,16,5,padding=2,padding_mode='circular',bias=False);self.n1=nn.BatchNorm1d(16)
        self.dw1=nn.Conv1d(16,16,7,padding=3,padding_mode='circular',groups=16,bias=False)
        self.pw1=nn.Conv1d(16,24,1,bias=False);self.n2=nn.BatchNorm1d(24)
        self.dw2=nn.Conv1d(24,24,5,padding=2,padding_mode='circular',groups=24,bias=False)
        self.pw2=nn.Conv1d(24,16,1,bias=False);self.n3=nn.BatchNorm1d(16)
        self.input_map=nn.Linear(128,16,bias=False);self.n4=nn.BatchNorm1d(16)
        self.leak_logit=nn.Parameter(torch.full((16,),math.log(4)))
        self.threshold_raw=nn.Parameter(torch.full((16,),math.log(math.expm1(.45))))
        self.readout=nn.Linear(16,1)
        # Append after common layers: their seeded initialization stays unchanged.
        if pooling=='content':
            self.pool_score=nn.Conv1d(2,1,3,padding=1,padding_mode='circular',bias=False)
        else:
            theta=(torch.arange(64,dtype=torch.float32)+.5)*(2*math.pi/64)
            basis=torch.stack([f(h*theta) for h in (1,2,3) for f in (torch.sin,torch.cos)])[None]
            self.register_buffer('position_basis',basis)
            self.pool_score=nn.Conv1d(6,1,1,bias=False)
        nn.init.zeros_(self.pool_score.weight)
    def attentive_pool(self,z,return_weights=False):
        if self.pooling=='content':
            summary=torch.cat([z.mean(1,keepdim=True),z.amax(1,keepdim=True)],1)
        else:summary=self.position_basis.expand(z.shape[0],-1,-1)
        score=self.pool_score(summary).reshape(z.shape[0],1,16,4)
        weights=score.softmax(-1)
        result=(z.reshape(z.shape[0],24,16,4)*weights).sum(-1)
        return (result,weights) if return_weights else result

    def forward(self,x,mask,return_activity=False):
        b,k,_,p=x.shape
        if (k,p)!=(9,64):raise ValueError('Nine physical cycles and64 bins required')
        z=x[:,:,:5]*mask[:,:,None,None]
        z=F.gelu(self.n1(self.c1(z.reshape(b*k,5,64))))
        z=self.attentive_pool(F.gelu(self.n2(self.pw1(self.dw1(z)))))
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
