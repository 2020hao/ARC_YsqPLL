"""Three weighted spike-to-spike stages; one direct per-cycle classifier.

Training uses dense FP32 simulation. Sparse operation counts describe a possible
sparsity-aware execution, not actual skipped GPU operations or measured energy.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ._source_phase_event_snn_neurons import LIFScan


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        x,=ctx.saved_tensors
        return gradient/(1+5*x.abs()).square()


class EventEncoder(nn.Module):
    def __init__(self,encoding='ttfs',steps=4,deadzone=.25,saturation=3.):
        super().__init__();self.encoding=encoding;self.steps=steps;self.deadzone=deadzone;self.saturation=saturation
    def forward(self,x,mask):
        values=torch.cat([x.clamp_min(0),(-x).clamp_min(0)],dim=2)
        amplitude=((values-self.deadzone)/(self.saturation-self.deadzone)).clamp(0,1)
        step=torch.arange(self.steps,device=x.device)[None,None,:,None,None]
        if self.encoding=='ttfs':
            latency=torch.round((1-amplitude)*(self.steps-1)).long()
            out=(step==latency[:,:,None])&(values[:,:,None]>=self.deadzone)
            out=out.to(x.dtype)
        elif self.encoding=='rate':
            out=(torch.floor((step+1)*amplitude[:,:,None])-torch.floor(step*amplitude[:,:,None])).to(x.dtype)
        elif self.encoding=='direct':out=amplitude[:,:,None].expand(-1,-1,self.steps,-1,-1)
        else:raise ValueError('Unknown event encoding')
        return out*mask[:,:,None,None,None]


class LIFSequence(nn.Module):
    def __init__(self,channels,spiking=True):
        super().__init__();self.spiking=spiking
        self.leak_logit=nn.Parameter(torch.full((channels,),math.log(4)))
        self.threshold_raw=nn.Parameter(torch.full((channels,),math.log(math.expm1(.45))))
    def forward(self,current,valid):
        leak=self.leak_logit.sigmoid().view(1,-1,1)
        threshold=(F.softplus(self.threshold_raw)+.05).view(1,-1,1)
        data=current.unsqueeze(-1) if current.ndim==3 else current
        result=LIFScan.apply(data,valid,leak,threshold,self.spiking)
        return result.squeeze(-1) if current.ndim==3 else result


class PhaseEventSNN(nn.Module):
    def __init__(self,encoding='ttfs',spiking=True):
        super().__init__();self.spiking=spiking;self.encoding=encoding;self.steps=4
        self.encoder=EventEncoder(encoding)
        self.conv1=nn.Conv1d(16,16,5,padding=2,padding_mode='circular',bias=False)
        self.bn1=nn.BatchNorm1d(16,eps=1e-4,momentum=.05);self.lif1=LIFSequence(16,spiking)
        self.conv2=nn.Conv1d(16,24,3,padding=1,padding_mode='circular',bias=False)
        self.bn2=nn.BatchNorm1d(24,eps=1e-4,momentum=.05);self.lif2=LIFSequence(24,spiking)
        self.fc=nn.Linear(384,16,bias=False);self.bn3=nn.BatchNorm1d(16,eps=1e-4,momentum=.05)
        self.lif3=LIFSequence(16,spiking);self.readout=nn.Linear(16,1)

    def forward(self,x,mask,return_activity=False,return_details=False):
        b,k,c,p=x.shape
        if (k,c,p)!=(9,8,64):raise ValueError('Expected nine cycles, eight channels and64phase bins')
        encoded=self.encoder(x,mask);t=k*self.steps;valid=mask.repeat_interleave(self.steps,dim=1)
        e=encoded.reshape(b,t,16,64)
        current1=self.bn1(self.conv1(e.reshape(b*t,16,64))).reshape(b,t,16,64)
        s1=self.lif1(current1,valid)
        pooled1=F.max_pool1d(s1.reshape(b*t,16,64),2)
        current2=self.bn2(self.conv2(pooled1)).reshape(b,t,24,32)
        s2=self.lif2(current2,valid)
        pooled2=F.max_pool1d(s2.reshape(b*t,24,32),2).reshape(b*t,384)
        current3=self.bn3(self.fc(pooled2)).reshape(b,t,16)
        s3=self.lif3(current3,valid)
        logit=self.readout(s3[:,-self.steps:]).mean(1).flatten()
        denom=valid.sum().clamp_min(1)
        rates=[s.sum()/(denom*(s[0,0].numel())) for s in (s1,s2,s3)]
        activity=torch.stack(rates).mean()
        if return_details:
            return logit,dict(encoded=e,layer1=s1,pool1=pooled1.reshape(b,t,16,32),layer2=s2,
                pool2=pooled2.reshape(b,t,384),layer3=s3,valid=valid,rates=torch.stack(rates),activity=activity)
        if return_activity:return logit,activity
        return logit
