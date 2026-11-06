"""Original direct CNN--LIF backbone with5/7/9 active raw-valued channels."""
import torch
from torch import nn
from torch.nn import functional as F
from ._source_map_input_calibration_deep_network import DeepCodedPhaseMap, Spike

ACTIVE={'base':(0,1,2,3,4),'mid':(0,1,2,3,4,5,6),
        'competition':(0,1,2,3,4,7,8),'both':tuple(range(9))}


class MidbandPhaseMap(DeepCodedPhaseMap):
    def __init__(self,arm='base'):
        super().__init__('raw')
        self.arm=arm;self.active=ACTIVE[arm]
        if len(self.active)>5:
            previous=self.c1
            self.c1=nn.Conv1d(len(self.active),16,5,padding=2,padding_mode='circular',bias=False)
            # Every arm starts with exactly the same original-model function.
            with torch.no_grad():self.c1.weight.zero_();self.c1.weight[:,:5].copy_(previous.weight)

    def forward(self,x,mask,return_activity=False,return_details=False):
        b,k,_,p=x.shape
        if (k,p)!=(9,64):raise ValueError('Nine physical cycles and64 bins required')
        z=(x[:,:,self.active,:]*mask[:,:,None,None]).reshape(b*k,len(self.active),64)
        z=F.gelu(self.n1(self.c1(z)))
        z=F.avg_pool1d(F.gelu(self.n2(self.pw1(self.dw1(z)))),4)
        z=F.avg_pool1d(F.gelu(self.n3(self.pw2(self.dw2(z)))),2)
        z=z+F.gelu(self.ms_norm(self.ms_fuse(torch.cat([conv(z) for conv in self.multiscale],1))))
        current=self.n4(self.input_map(z)).reshape(b,k,16,8)
        leak=self.leak_logit.sigmoid()[None,:,None];threshold=(F.softplus(self.threshold_raw)+.05)[None,:,None]
        potential=torch.zeros_like(current[:,0]);trace=torch.zeros_like(potential);events=[]
        for j in range(9):
            valid=mask[:,j,None,None];u=(potential*leak+current[:,j])*valid
            s=Spike.apply(u-threshold)*valid;potential=(u-threshold*s.detach())*valid
            trace=(.8*trace+.2*s)*valid;events.append(s)
        representation=(potential/threshold+trace).flatten(1)
        logits=self.readout(F.gelu(self.readout_norm(self.readout_hidden(representation)))).flatten()
        activity=torch.stack(events,1).sum()/(mask.sum().clamp_min(1)*128)
        if return_details:return logits,dict(representation=representation,membrane=potential,trace=trace,spikes=torch.stack(events,1),activity=activity)
        return (logits,activity) if return_activity else logits
