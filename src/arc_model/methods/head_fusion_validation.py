"""Train-subset residual heads and pure replayable confidence/fusion policies."""
import numpy as np,torch
from sklearn.decomposition import PCA
from scipy.special import expit
from .local_phase_dynamics import group_class_weights

def fit_residual(x,y,g,offset,ids,keep,seed,balanced=False,bias=False,bounded=False,score=False):
    values=np.repeat(offset[:,None],len(keep),axis=1) if score else x[:,keep]
    training_values=values[ids] if score else x[ids][:,keep]
    mu=training_values.mean(0);sd=np.maximum(training_values.std(0),1e-6);z=(values-mu)/sd
    torch.manual_seed(seed+501);model=torch.nn.Linear(len(keep),1,bias=bias).to('cuda:0');torch.nn.init.zeros_(model.weight)
    if bias:torch.nn.init.zeros_(model.bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.002);gen=torch.Generator(device='cuda:0').manual_seed(seed+601)
    xx=torch.tensor(z,dtype=torch.float32,device='cuda:0');b=torch.tensor(offset,dtype=torch.float32,device='cuda:0');yy=torch.tensor(y,dtype=torch.float32,device='cuda:0');ix=torch.tensor(ids,device='cuda:0')
    w=torch.tensor(group_class_weights(y,g,ids),device='cuda:0') if balanced else None
    for _ in range(80):
        for j in ix[torch.randperm(len(ix),generator=gen,device='cuda:0')].split(256):
            optimizer.zero_grad(set_to_none=True);raw=model(xx[j]).flatten();delta=4*torch.tanh(raw/4) if bounded else raw
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(b[j]+delta,yy[j],reduction='none')*w[j]).mean() if balanced else torch.nn.functional.binary_cross_entropy_with_logits(b[j]+delta,yy[j])
            (loss+2e-4*model.weight.abs().mean()).backward();optimizer.step()
    normal=ids[y[ids]==0];pca=PCA(n_components=min(6,len(keep),len(normal)-1),random_state=seed).fit(z[normal]);coef=model.weight.detach().cpu().numpy().flatten();intercept=float(model.bias.detach().cpu().item()) if bias else 0.
    state=dict(keep=np.asarray(keep),mu=mu,sd=sd,coef=coef,bias=intercept,bounded=bounded,score=score,fit_ids=np.asarray(ids),pca_components=pca.components_,pca_mean=pca.mean_,balanced=balanced)
    del xx,b,yy,ix,w,model,optimizer;torch.cuda.empty_cache();return state

def replay(state,x,offset):
    values=np.repeat(offset[:,None],len(state['keep']),axis=1) if bool(state.get('score',False)) else x[:,state['keep']]
    z=(values-state['mu'])/state['sd'];raw=z.astype('float32')@state['coef']+float(state.get('bias',0.));delta=4*np.tanh(raw/4) if bool(state.get('bounded',False)) else raw
    u=(z-state['pca_mean'])@state['pca_components'].T;return delta,np.sqrt(np.mean(u*u,axis=1))

def fuse(offset,delta,distance,normal_distance,alpha,quantile,gated):
    threshold=max(float(np.quantile(normal_distance,quantile)),1e-6)
    gate=expit(-(distance-threshold)/(.15*threshold+1e-8)) if gated else np.ones(len(delta))
    return expit(offset+alpha*gate*delta),gate
