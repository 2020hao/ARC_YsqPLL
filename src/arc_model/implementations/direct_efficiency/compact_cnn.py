"""Continuous-valued phase-bin CNN; all nine physical cycles enter readout."""
import torch
from torch import nn
from torch.nn import functional as F


class CompactCNN(nn.Module):
    spiking=False
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv1d(5,16,5,padding=2,padding_mode='circular',bias=False);self.n1=nn.BatchNorm1d(16)
        self.dw1=nn.Conv1d(16,16,7,padding=3,padding_mode='circular',groups=16,bias=False)
        self.pw1=nn.Conv1d(16,24,1,bias=False);self.n2=nn.BatchNorm1d(24)
        self.dw2=nn.Conv1d(24,24,5,padding=2,padding_mode='circular',groups=24,bias=False)
        self.pw2=nn.Conv1d(24,16,1,bias=False);self.n3=nn.BatchNorm1d(16)
        self.temporal_dw=nn.Conv1d(128,128,3,groups=128,bias=False);self.nt=nn.BatchNorm1d(128)
        self.temporal_pw=nn.Conv1d(128,16,1,bias=False);self.n4=nn.BatchNorm1d(16)
        self.history_weights=nn.Parameter(torch.zeros(9));self.readout=nn.Linear(16,1)
    def forward(self,x,mask,return_activity=False):
        b,k,_,p=x.shape
        if (k,p)!=(9,64):raise ValueError('Nine physical cycles and64 bins required')
        z=x[:,:,:5]*mask[:,:,None,None]
        z=F.gelu(self.n1(self.c1(z.reshape(b*k,5,64))))
        z=F.avg_pool1d(F.gelu(self.n2(self.pw1(self.dw1(z)))),4)
        z=F.avg_pool1d(F.gelu(self.n3(self.pw2(self.dw2(z)))),2)
        z=z.reshape(b,k,128)*mask[:,:,None]
        z=F.gelu(self.nt(self.temporal_dw(F.pad(z.transpose(1,2),(2,0)))))
        z=F.gelu(self.n4(self.temporal_pw(z)))
        weight=self.history_weights[None].expand(b,-1).masked_fill(mask==0,-1e9).softmax(1)
        logits=self.readout((z*weight[:,None,:]).sum(2)).flatten()
        return (logits,logits.new_zeros(())) if return_activity else logits
