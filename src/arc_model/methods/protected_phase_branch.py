"""Independent residual branch; the frozen baseline is an immutable offset."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA


class ProtectedBranch(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture == 'linear':
            self.hidden = nn.Identity();self.output = nn.Linear(12, 1)
        elif architecture == 'mlp':
            self.hidden = nn.Sequential(nn.Linear(12, 8), nn.ReLU());self.output = nn.Linear(8, 1)
        else:raise ValueError(architecture)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return 4*torch.tanh(self.output(self.hidden(x)).flatten()/4)


def branch_inputs(arm, phase, randomized, amplitude, dynamic, frozen_keep, train):
    """All controls have 12 inputs; PCA/statistics only use new-branch train IDs."""
    masks = phase[:, 48:52].copy()
    info = {}
    if arm in ['P', 'Q'] or arm.startswith('P_no'):
        values = phase[:, 40:48].copy()
        if arm == 'Q':values[:]=0
        if arm.startswith('P_no'):
            h=int(arm.removeprefix('P_no'));j=[3,5,7,9].index(h)
            values[:,2*j:2*j+2]=0;masks[:,j]=0
    elif arm == 'R':values=randomized[:,40:48].copy()
    elif arm == 'A':
        a=amplitude[:,36:40];values=np.c_[a,a*a]
    elif arm == 'D':
        d=dynamic[:,frozen_keep]
        mu=d[train].mean(0);sd=np.maximum(d[train].std(0),1e-6)
        z=(d-mu)/sd;pca=PCA(n_components=8,random_state=0).fit(z[train]);values=pca.transform(z)
        info=dict(dynamic_keep=np.asarray(frozen_keep),dynamic_mean=mu,dynamic_std=sd,
                  pca_mean=pca.mean_,pca_components=pca.components_)
    else:raise ValueError(arm)
    x=np.c_[values,masks].astype('float32')
    mu=x[train].mean(0);sd=np.maximum(x[train].std(0),1e-6)
    z=np.clip((x-mu)/sd,-8,8).astype('float32')
    return z,dict(mean=mu,std=sd,**info)


def fit_branch(inputs, labels, train, offset, architecture, seed):
    torch.manual_seed(seed)
    model=ProtectedBranch(architecture).to('cuda:0')
    count=sum(p.numel() for p in model.parameters())
    assert count == {'linear':13,'mlp':113}[architecture]
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.002)
    x=torch.tensor(inputs,device='cuda:0');y=torch.tensor(labels,dtype=torch.float32,device='cuda:0')
    base=torch.tensor(offset,dtype=torch.float32,device='cuda:0')
    ids=torch.tensor(train,device='cuda:0');gen=torch.Generator(device='cuda:0').manual_seed(seed+100)
    for _ in range(80):
        for batch in ids[torch.randperm(len(ids),device='cuda:0',generator=gen)].split(256):
            opt.zero_grad(set_to_none=True)
            correction=model(x[batch])
            loss=nn.functional.binary_cross_entropy_with_logits(base[batch]+correction,y[batch])+2e-4*model.output.weight.abs().mean()
            loss.backward();opt.step()
    model.eval()
    with torch.no_grad():delta=model(x).cpu().numpy()
    return delta,{k:v.detach().cpu() for k,v in model.state_dict().items()},count
