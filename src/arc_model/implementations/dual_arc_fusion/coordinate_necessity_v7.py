"""Single-factor gate comparison with unchanged availability and training.

Independent implementation for this model package. Not a default or a verified
performance improvement. Keep all baseline networks and historical v2 states.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import expit
from sklearn.decomposition import PCA


def validate_input(dynamic):
    x=np.asarray(dynamic,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=36 or not np.isfinite(x).all():
        raise ValueError('Finite 32 dynamic statistics plus four availability masks required')
    masks=x[:,32:]
    if not np.isin(masks,[0.,1.]).all():raise ValueError('Binary historical availability required')
    return x[:,:32],np.repeat(masks,8,axis=1),masks


def transform(state,dynamic):
    values,valid,masks=validate_input(dynamic)
    z=(values-state['mu'])/state['sd']
    if bool(state['availability_aware']):
        z=np.where(valid!=0,z,0.)
        coverage=masks.mean(1)
    else:coverage=np.ones(len(z),dtype=np.float32)
    return np.ascontiguousarray(z,dtype=np.float32),coverage


def quality(state,z,coverage,reference_quality=None):
    kind=str(state.get('gate_kind',np.asarray('pca_distance')))
    if kind=='no_distance':return coverage.copy()
    if kind=='reference_quality':
        q=np.asarray(reference_quality,dtype=np.float32)
        if q.shape!=(len(z),) or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
            raise ValueError('Aligned finite causal reference quality in [0,1] required')
        return coverage*q
    if kind!='pca_distance':raise ValueError('Unknown gate')
    projected=z@state['pca_components'].T-state['pca_projected_mean']
    distance=np.sqrt(np.mean(projected*projected,axis=1))
    threshold=float(state['gate_threshold']);softness=float(state['softness'])
    return coverage*expit(-(distance-threshold)/(softness*threshold+1e-8))


def score(state,dynamic,raw,*,enabled=True,reference_quality=None):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim!=1:raise ValueError('One raw margin per observation required')
    strength=float(state['strength'])
    if not enabled or strength==0:return raw.copy()
    z,coverage=transform(state,dynamic)
    if len(z)!=len(raw):raise ValueError('Features and margins must align')
    q=quality(state,z,coverage,reference_quality)
    delta=z@state['coef']
    calibrated=float(state['calibration_slope'])*raw+float(state['calibration_intercept'])
    return raw+strength*q*(calibrated-raw+delta)


def fit(dynamic,labels,raw,indices,*,calibration_slope,calibration_intercept,
        seed=0,strength=1.,quantile=.95,softness=.15,availability_aware=True,
        consistent_objective=True,epochs=80,lr=.002,weight_decay=.002,
        l1_mean_penalty=.0002,batch_size=256,gate_kind="pca_distance",reference_quality=None):
    """Fit one predeclared strength/gate candidate on explicit training IDs.

    A/B/C model selection stays outside this function. Caller must supply honest
    OOF/raw margins and calibration fitted without this candidate's validation
    groups. No validation/test label is read during fitting.
    """
    values,valid,_=validate_input(dynamic);y=np.asarray(labels);raw=np.asarray(raw,dtype=np.float64)
    ids=np.asarray(indices,dtype=np.int64)
    if ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Unique nonempty training IDs required')
    if ids.min()<0 or ids.max()>=len(values):raise ValueError('Training index outside feature array')
    if y.shape!=(len(values),) or raw.shape!=y.shape or not np.isfinite(raw).all():raise ValueError('Labels/raw margins must align')
    if not np.isin(y[ids],[0,1]).all() or set(y[ids])!={0,1}:raise ValueError('Both training classes required')
    if not 0<=strength<=1 or not 0<quantile<1 or softness<=0:raise ValueError('Invalid bounded fusion settings')
    if not np.isfinite(calibration_slope) or calibration_slope<=0 or not np.isfinite(calibration_intercept):raise ValueError('Finite positive-slope calibration required')
    if availability_aware:
        count=valid[ids].sum(0);safe=np.maximum(count,1.)
        mu=(np.where(valid[ids]!=0,values[ids],0.)).sum(0)/safe
        variance=np.where(valid[ids]!=0,(values[ids]-mu)**2,0.).sum(0)/safe
        mu=np.where(count>0,mu,0.);sd=np.where(count>0,np.maximum(np.sqrt(variance),1e-6),1.)
    else:
        count=np.full(32,len(ids));mu=values[ids].mean(0);sd=np.maximum(values[ids].std(0),1e-6)
    state=dict(mu=np.asarray(mu,dtype=np.float32),sd=np.asarray(sd,dtype=np.float32),
        valid_fit_counts=np.asarray(count,dtype=np.int64),fit_indices=ids.copy(),
        calibration_slope=np.asarray(calibration_slope),calibration_intercept=np.asarray(calibration_intercept),
        strength=np.asarray(strength),softness=np.asarray(softness),quantile=np.asarray(quantile),
        availability_aware=np.asarray(availability_aware),consistent_objective=np.asarray(consistent_objective),
        prototype_only=np.asarray(True),gate_kind=np.asarray(gate_kind))
    z,coverage=transform(state,dynamic);normal=ids[y[ids]==0]
    if len(normal)<2:raise ValueError('At least two normal training observations required for PCA')
    if gate_kind=="pca_distance":
        pca=PCA(n_components=min(6,32,len(normal)-1),random_state=seed).fit(z[normal])
        components=np.array(pca.components_,dtype=np.float32,order='C',copy=True)
        projected_mean=np.asarray(pca.mean_@components.T,dtype=np.float32)
        dist=np.sqrt(np.mean((z@components.T-projected_mean)**2,axis=1))
        threshold=max(float(np.quantile(dist[normal],quantile)),1e-6)
        state.update(pca_components=components,pca_projected_mean=projected_mean,
                     PCA_fit_indices=normal.copy(),gate_threshold=np.asarray(threshold))
    q=quality(state,z,coverage,reference_quality)
    torch.manual_seed(seed+501);head=torch.nn.Linear(32,1,bias=False)
    torch.nn.init.zeros_(head.weight)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed+601)
    xx=torch.from_numpy(z);yy=torch.tensor(y,dtype=torch.float32)
    rr=torch.tensor(raw,dtype=torch.float32);qq=torch.tensor(q,dtype=torch.float32)
    cc=torch.tensor(float(calibration_slope)*raw+float(calibration_intercept),dtype=torch.float32)
    ii=torch.from_numpy(ids)
    losses=[]
    for _ in range(epochs):
        total=0.
        for j in ii[torch.randperm(len(ii),generator=generator)].split(batch_size):
            optimizer.zero_grad(set_to_none=True);delta=head(xx[j]).flatten()
            margin=rr[j]+float(strength)*qq[j]*(cc[j]-rr[j]+delta) if consistent_objective else cc[j]+delta
            loss=torch.nn.functional.binary_cross_entropy_with_logits(margin,yy[j])+l1_mean_penalty*head.weight.abs().mean()
            loss.backward();optimizer.step();total+=float(loss.detach())*len(j)
        losses.append(total/len(ids))
    state['coef']=np.array(head.weight.detach().numpy().ravel(),dtype=np.float32,order='C',copy=True)
    state['train_loss']=np.asarray(losses,dtype=np.float64)
    # Canonicalize storage before any inference or serialization.
    return {k:np.array(v,copy=True,order='C') for k,v in state.items()}


def save(state,path):np.savez(path,**state)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def paired_dynamics(probabilities, previous, mode="ordered"):
    """32 lag/band descriptors; availability is not a learned feature.

    `registered` quotients relative circular displacement before measuring shape.
    It is deliberately a phase-position-free alternative, not a PLL improvement.
    """
    p=np.asarray(probabilities,dtype=np.float64)
    previous=np.asarray(previous,dtype=np.int64)
    if p.ndim!=3 or p.shape[1:]!=(2,20) or previous.shape!=(len(p),4):
        raise ValueError('Two bands,20 bins,and4 strictly causal previous indices required')
    if not np.isfinite(p).all() or (p<0).any() or (previous>=len(p)).any():
        raise ValueError('Invalid distributions or history')
    if mode not in ('ordered','sorted','registered'):raise ValueError('Unknown shape coordinate')
    if mode=='sorted':p=np.sort(p,axis=-1)
    x=np.zeros((len(p),36),dtype=np.float64)
    for k in range(4):
        ids=np.flatnonzero(previous[:,k]>=0);j=previous[ids,k]
        if (j>=ids).any():raise ValueError('History must precede current cycle')
        u=p[ids];v=p[j]
        if mode=='registered' and len(ids):
            costs=np.stack([np.abs(u-np.roll(v,s,axis=-1)).mean(-1) for s in range(20)],axis=-1)
            shifts=costs.argmin(-1)
            v=np.take_along_axis(v,(np.arange(20)[None,None,:]-shifts[:,:,None])%20,axis=-1)
        mid=.5*(u+v)
        stats=np.stack([(u*v).sum(-1)/(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)+1e-12),
             np.abs(u-v).mean(-1),
             .5*np.sum(u*np.log((u+1e-12)/(mid+1e-12)),axis=-1)+.5*np.sum(v*np.log((v+1e-12)/(mid+1e-12)),axis=-1),
             (u.max(-1)-v.max(-1))/(u.max(-1)+v.max(-1)+1e-12)],axis=-1)
        x[ids,k*8:(k+1)*8]=stats.reshape(-1,8);x[ids,32+k]=1.
    return x
