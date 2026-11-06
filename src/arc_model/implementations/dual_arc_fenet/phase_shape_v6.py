"""Independent targeted phase/shape descriptors and additive correction head.

Frozen parents remain unchanged. Phase-bin features, not microsecond event times.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch

FEATURE_NAMES=('low_shift_abs','low_shape_MAD','low_half_L1','high_shift_abs','high_shape_MAD','high_half_L1')

def rotations(dataset,metadata,kind):
    if kind not in ('record','cycle'):raise ValueError('Unknown rotation control')
    result=[]
    for row in metadata:
        key=f"phase-shape-v6|{dataset}|{kind}|{row['record_id']}"
        if kind=='cycle':key+=f"|{row['anchor_index']}"
        v=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        result.append(v%20 if kind=='cycle' else 1+v%19)
    return np.asarray(result,dtype=np.int64)


def six_features(energy,metadata,sample_rate,shifts=None):
    e=np.asarray(energy,dtype=np.float64)
    if e.shape!=(len(metadata),2,20) or not np.isfinite(e).all() or (e<0).any():raise ValueError('Native-integrated nonnegative [N,2,20] energies required')
    if shifts is not None:
        shifts=np.asarray(shifts)
        if shifts.shape!=(len(e),) or shifts.dtype.kind not in 'iu':raise ValueError('Aligned integer rotations required')
        e=np.take_along_axis(e,(np.arange(20)[None,None,:]-shifts[:,None,None])%20,axis=2)
    n=len(e);tot=e.sum(-1);p=np.divide(e,tot[...,None],out=np.zeros_like(e),where=tot[...,None]>1e-20)
    values=np.zeros((n,6),dtype=np.float32);valid=np.zeros((n,6),dtype=bool);prev=np.full(n,-1,dtype=np.int64)
    lookup={(str(r['record_id']),int(r['anchor_index'])):i for i,r in enumerate(metadata)}
    if len(lookup)!=n:raise ValueError('Unique record/anchor IDs required')
    tolerance=1.5/float(sample_rate)
    for i,row in enumerate(metadata):
        j=lookup.get((str(row['record_id']),int(row['anchor_index'])-1))
        if j is None:continue
        old=metadata[j]
        same=all(str(old[k])==str(row[k]) for k in ('kind','segment','epoch'))
        continuous=abs(float(old['end_seconds'])-float(row['start_seconds']))<=tolerance
        causal=float(old['available_seconds'])<=float(row['anchor_seconds'])+1e-9
        if same and continuous and causal:prev[i]=j
    ids=np.flatnonzero(prev>=0);ties=np.zeros((n,2),dtype=np.int64);best=np.full((n,2),-1,dtype=np.int64)
    if len(ids):
        a=p[ids];b=p[prev[ids]]
        costs=np.stack([np.abs(a-np.roll(b,s,axis=-1)).mean(-1) for s in range(20)],axis=-1)
        minimum=costs.min(-1);optimal=costs<=minimum[...,None]+1e-12
        magnitude=np.minimum(np.arange(20),20-np.arange(20))/10.
        selected=np.argmin(np.where(optimal,magnitude[None,None,:],np.inf),axis=-1)
        count=optimal.sum(-1);ties[ids]=count;best[ids]=selected
        for band in range(2):
            pair=(tot[ids,band]>1e-20)&(tot[prev[ids],band]>1e-20)
            unique=pair&(count[:,band]==1)
            values[ids,band*3]=np.where(unique,magnitude[selected[:,band]],0.)
            values[ids,band*3+1]=np.where(pair,minimum[:,band],0.)
            valid[ids,band*3]=unique;valid[ids,band*3+1]=pair
    for band in range(2):
        left=e[:,band,:10];right=e[:,band,10:];ls=left.sum(-1);rs=right.sum(-1)
        ok=(ls>1e-20)&(rs>1e-20)
        a=np.divide(left,ls[:,None],out=np.zeros_like(left),where=ls[:,None]>1e-20)
        b=np.divide(right,rs[:,None],out=np.zeros_like(right),where=rs[:,None]>1e-20)
        values[:,band*3+2]=np.where(ok,.5*np.abs(a-b).sum(-1),0.)
        valid[:,band*3+2]=ok
    assert np.isfinite(values).all() and (values>=0).all() and (values[:,[0,2,3,5]]<=1+1e-6).all()
    return values,valid,dict(previous_index=prev,optimum_count=ties,selected_roll=best)


def transform(state,dynamic,extra,extra_valid):
    x=np.asarray(dynamic,dtype=np.float32);v=np.asarray(extra,dtype=np.float32);mask=np.asarray(extra_valid,dtype=bool)
    if x.ndim!=2 or x.shape[1]!=36 or v.shape!=(len(x),6) or mask.shape!=v.shape:raise ValueError('Aligned 32 statistics/4 controls and6 descriptors/masks required')
    if not np.isfinite(x).all() or not np.isfinite(v).all() or not np.isin(x[:,32:],[0,1]).all():raise ValueError('Finite features and binary history controls required')
    valid=np.repeat(x[:,32:],8,axis=1)!=0
    u=np.ascontiguousarray(np.where(valid,(x[:,:32]-state['mu'])/state['sd'],0.),dtype=np.float32)
    active=mask&state['active'][None,:]
    w=np.ascontiguousarray(np.where(active,(v-state['extra_mu'])/state['extra_sd'],0.),dtype=np.float32)
    return u,w,x[:,32:].mean(1)


def score(state,dynamic,extra,extra_valid,raw,enabled=True):
    raw=np.asarray(raw,dtype=np.float64)
    if not enabled:return raw.copy()
    u,v,h=transform(state,dynamic,extra,extra_valid)
    calibrated=float(state['calibration_slope'])*raw+float(state['calibration_intercept'])
    return raw+h*(calibrated-raw+u@state['coef'])+v@state['extra_coef']


def fit(dynamic,extra,extra_valid,labels,raw,indices,active,*,calibration_slope,calibration_intercept,seed=0,epochs=80,lr=.002,weight_decay=.002,l1_mean_penalty=.0002,batch_size=256):
    x=np.asarray(dynamic,dtype=np.float32);v=np.asarray(extra,dtype=np.float32);mask=np.asarray(extra_valid,dtype=bool)
    active=np.asarray(active,dtype=bool);ids=np.asarray(indices,dtype=np.int64);y=np.asarray(labels);raw=np.asarray(raw,dtype=np.float64)
    if active.shape!=(6,) or len(np.unique(ids))!=len(ids) or not len(ids):raise ValueError('Active6 and unique nonempty training IDs required')
    valid=np.repeat(x[:,32:],8,axis=1);count=valid[ids].sum(0);safe=np.maximum(count,1.)
    mu=np.where(valid[ids]!=0,x[ids,:32],0.).sum(0)/safe
    variance=np.where(valid[ids]!=0,(x[ids,:32]-mu)**2,0.).sum(0)/safe
    mu=np.where(count>0,mu,0.);sd=np.where(count>0,np.maximum(np.sqrt(variance),1e-6),1.)
    mask=mask&active[None,:];n=mask[ids].sum(0);safe=np.maximum(n,1)
    vm=np.where(mask[ids],v[ids],0.).sum(0)/safe
    vs=np.sqrt(np.where(mask[ids],(v[ids]-vm)**2,0.).sum(0)/safe)
    vm=np.where(n>0,vm,0.);vs=np.where(n>0,np.maximum(vs,1e-6),1.)
    state=dict(mu=np.asarray(mu,dtype=np.float32),sd=np.asarray(sd,dtype=np.float32),valid_fit_counts=count.astype(np.int64),
        extra_mu=vm.astype(np.float32),extra_sd=vs.astype(np.float32),extra_fit_counts=n,active=active,fit_indices=ids,
        calibration_slope=np.asarray(calibration_slope),calibration_intercept=np.asarray(calibration_intercept))
    u,w,h=transform(state,x,v,extra_valid)
    torch.manual_seed(seed+501);base=torch.nn.Linear(32,1,bias=False);new=torch.nn.Linear(6,1,bias=False)
    torch.nn.init.zeros_(base.weight);torch.nn.init.zeros_(new.weight)
    opt=torch.optim.AdamW(list(base.parameters())+list(new.parameters()),lr=lr,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed+601)
    uu=torch.from_numpy(u);ww=torch.from_numpy(w);hh=torch.tensor(h,dtype=torch.float32);rr=torch.tensor(raw,dtype=torch.float32)
    cc=torch.tensor(float(calibration_slope)*raw+float(calibration_intercept),dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32);ii=torch.from_numpy(ids)
    losses=[]
    for epoch in range(epochs):
        total=0.
        for j in ii[torch.randperm(len(ii),generator=generator)].split(batch_size):
            opt.zero_grad(set_to_none=True)
            margin=rr[j]+hh[j]*(cc[j]-rr[j]+base(uu[j]).flatten())+new(ww[j]).flatten()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(margin,yy[j])+l1_mean_penalty*(base.weight.abs().mean()+new.weight.abs().mean())
            loss.backward();opt.step();total+=float(loss.detach())*len(j)
        losses.append(total/len(ids))
    state.update(coef=base.weight.detach().numpy().ravel(),extra_coef=new.weight.detach().numpy().ravel(),train_loss=np.asarray(losses))
    return {k:np.array(v,copy=True,order='C') for k,v in state.items()}


def save(state,path):np.savez(path,**state)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
