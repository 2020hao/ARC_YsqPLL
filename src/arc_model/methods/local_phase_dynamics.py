"""Small phase-local dynamics branches and train-subset group-balanced loss."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from .protected_phase_branch import ProtectedBranch


def phase_shifts(dataset, record, cycles):
    return np.array([int.from_bytes(hashlib.sha256(f'LB1|{dataset}|{record}|{k}'.encode()).digest()[:4],'little')%20 for k in cycles])


def difference_maps(distributions, records, shifts=None):
    p=np.asarray(distributions,float)
    if shifts is not None:
        index=(np.arange(20)[None,:]-np.asarray(shifts)[:,None])%20
        p=np.take_along_axis(p,index[:,None,:],axis=2)
    out=np.zeros((len(p),8,20),dtype='float32');available=np.zeros((len(p),4),bool)
    lookup={(str(r['record_id']),int(r['cycle'])):i for i,r in enumerate(records)}
    for i,r in enumerate(records):
        for j,lag in enumerate([1,2,4,8]):
            k=lookup.get((str(r['record_id']),int(r['cycle'])-lag))
            if k is not None:
                out[i,j*2:j*2+2]=p[i]-p[k];available[i,j]=True
    return out,available


class LocalPhaseBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth=nn.Conv1d(8,8,3,padding=1,padding_mode='circular',groups=8)
        self.point=nn.Conv1d(8,8,1);self.output=nn.Linear(8,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        x=torch.relu(self.depth(x));x=torch.relu(self.point(x)).mean(-1)
        return 4*torch.tanh(self.output(x).flatten()/4)


def group_class_weights(labels,groups,ids):
    weights=np.zeros(len(labels),dtype='float32');unique=np.unique(groups[ids])
    for group in unique:
        for c in [0,1]:
            j=ids[(groups[ids]==group)&(labels[ids]==c)]
            if not len(j):raise ValueError('Every training condition must contain both classes')
            weights[j]=len(ids)/(2*len(unique)*len(j))
    return weights


def normalize_maps(x,ids):
    # Sharing scale over phase preserves common-origin circular invariance.
    mu=x[ids].mean((0,2),keepdims=True);sd=np.maximum(x[ids].std((0,2),keepdims=True),1e-6)
    return np.clip((x-mu)/sd,-8,8).astype('float32'),dict(mean=mu,std=sd)


def make_model(kind):
    model=ProtectedBranch('mlp') if kind=='mlp' else LocalPhaseBranch()
    assert sum(p.numel() for p in model.parameters())==113
    return model


def fit_local_branch(inputs,labels,groups,ids,offset,valid,kind,balanced,seed):
    torch.manual_seed(seed);model=make_model(kind).to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.002)
    x=torch.tensor(inputs,device='cuda:0');y=torch.tensor(labels,dtype=torch.float32,device='cuda:0')
    base=torch.tensor(offset,dtype=torch.float32,device='cuda:0');gate=torch.tensor(valid,dtype=torch.float32,device='cuda:0')
    w=group_class_weights(labels,groups,ids) if balanced else np.ones(len(labels),dtype='float32')
    weights=torch.tensor(w,device='cuda:0');ix=torch.tensor(ids,device='cuda:0')
    rng=torch.Generator(device='cuda:0').manual_seed(seed+100)
    for _ in range(80):
        for batch in ix[torch.randperm(len(ix),generator=rng,device='cuda:0')].split(256):
            optimizer.zero_grad(set_to_none=True)
            correction=model(x[batch])*gate[batch]
            loss=(nn.functional.binary_cross_entropy_with_logits(base[batch]+correction,y[batch],reduction='none')*weights[batch]).mean()+2e-4*model.output.weight.abs().mean()
            loss.backward();optimizer.step()
    model.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(inputs),1024):output.append((model(x[start:start+1024])*gate[start:start+1024]).cpu().numpy())
    return np.concatenate(output),{k:v.detach().cpu() for k,v in model.state_dict().items()}
