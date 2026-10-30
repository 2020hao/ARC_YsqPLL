"""Independent phase-dynamic correction v2, exploratory candidate.

Keeps the historical CPU80-epoch zero-initialized linear fitting recipe, and
adds an entire-path bounded blend whose zero-strength state is the raw parent.
No imports from another model package and no changes to historical modules.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import expit
from sklearn.decomposition import PCA
from .network import Model as Original

ALPHAS=(0.,.1,.25,.5,1.,1.5,2.)
LAMBDAS=(0.,.1,.25,.5,.75,1.)
QUANTILES=(.75,.9,.95,.975)

def fit(
    x,
    labels,
    offset,
    indices,
    columns,
    seed,
    *,
    device="cpu",
    epochs=80,
    lr=0.002,
    weight_decay=0.002,
    batch_size=256,
    l1_mean_penalty=0.0002,
):
    """Train a zero-initialized linear correction against fixed OOF logits.

    Preserve historical float32 indexing/reduction order for numerical replay.
    All fitted statistics and PCA use only the supplied training indices.
    """
    mu = x[indices][:, columns].mean(0)
    sd = np.maximum(x[indices][:, columns].std(0), 1e-6)
    z = (x[:, columns] - mu) / sd
    torch.manual_seed(seed + 501)
    model = torch.nn.Linear(len(columns), 1, bias=False).to(device)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device).manual_seed(seed + 601)
    xx = torch.tensor(z, dtype=torch.float32, device=device)
    base = torch.tensor(offset, dtype=torch.float32, device=device)
    yy = torch.tensor(labels, dtype=torch.float32, device=device)
    ids = torch.tensor(indices, device=device)
    for _ in range(epochs):
        for j in ids[torch.randperm(len(ids), generator=generator, device=device)].split(
            batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            delta = model(xx[j]).flatten()
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(base[j] + delta, yy[j])
                + l1_mean_penalty * model.weight.abs().mean()
            )
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        delta = model(xx).flatten().cpu().numpy()
        coef = model.weight.flatten().cpu().numpy()
    normal = indices[labels[indices] == 0]
    pca = PCA(n_components=min(6, len(columns), len(normal) - 1), random_state=seed).fit(z[normal])
    scores = pca.transform(z)
    distance = np.sqrt(np.mean(scores * scores, axis=1))
    state = dict(
        keep=np.asarray(columns),
        mu=mu,
        sd=sd,
        coef=coef,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        fit_indices=indices,
    )
    return delta, distance, state


def replay(state, x):
    z = (x[:, state["keep"]] - state["mu"]) / state["sd"]
    delta = z.astype("float32") @ state["coef"]
    # Match sklearn PCA.transform's uncentred-input reduction order.
    score = z @ state["pca_components"].T - state["pca_mean"] @ state["pca_components"].T
    return delta, np.sqrt(np.mean(score * score, axis=1))



def condition_metrics(y,p,g,ids):
    values=[]
    for group in np.unique(g[ids]):
        ix=ids[g[ids]==group];yy=y[ix];pp=p[ix]
        if set(np.unique(yy)) != {0,1}:
            raise ValueError("Every condition requires both classes")
        recall=float(pp[yy==1].mean());fpr=float(pp[yy==0].mean())
        values.append((.5*(recall+1-fpr),recall,fpr))
    return tuple(np.mean(values,axis=0))


def select_strength(y,g,raw,calibrated,delta,distance,fit_indices,validation_indices,*,fusion,recall_constraint):
    normal=fit_indices[y[fit_indices]==0]
    if not len(normal):raise ValueError("Normal training examples required")
    raw_p=raw>=0
    raw_metrics=condition_metrics(y,raw_p,g,validation_indices)
    if fusion=='old_residual':
        base=calibrated>=0
        baseline_score=condition_metrics(y,base,g,validation_indices)[0]
        ni=validation_indices[y[validation_indices]==0]
        baseline_fpr=float(base[ni].mean())
        best=((baseline_score,-baseline_fpr,0.,-1.),(0.,.9))
        grid=ALPHAS
    elif fusion=='whole_path':
        best=((raw_metrics[0],-raw_metrics[2],0.,-1.),(0.,.9))
        grid=LAMBDAS
    else:raise ValueError("Unknown fusion")
    trace=[]
    for strength in grid:
        for quantile in QUANTILES:
            threshold=max(float(np.quantile(distance[normal],quantile)),1e-6)
            gate=expit(-(distance-threshold)/(.15*threshold+1e-8))
            if fusion=='old_residual':
                margin=calibrated+strength*gate*delta
            else:
                margin=raw+strength*gate*(calibrated-raw+delta)
            p=margin>=0;bacc,recall,fpr=condition_metrics(y,p,g,validation_indices)
            if fusion=='old_residual':
                pooled_fpr=float(p[ni].mean());eligible=pooled_fpr<=baseline_fpr+.0200001
                rank=(bacc,-pooled_fpr,-strength,-quantile)
            else:
                eligible=(fpr<=raw_metrics[2]+.0200001) and (not recall_constraint or recall>=raw_metrics[1]-.0100001)
                rank=(bacc,-fpr,-strength,-quantile)
            trace.append(dict(strength=float(strength),quantile=quantile,BAcc=float(bacc),Recall=float(recall),FPR=float(fpr),eligible=bool(eligible),
                              raw_BAcc=float(raw_metrics[0]),raw_Recall=float(raw_metrics[1]),raw_FPR=float(raw_metrics[2])))
            if eligible and rank>best[0]:best=(rank,(float(strength),quantile))
    return best[1],trace


class Correction:
    def __init__(self,state,*,baseline_id,score_kind):
        self.state={k:np.asarray(v).copy() for k,v in state.items()}
        if str(self.state.get('baseline_id',''))!=baseline_id:raise ValueError("Wrong parent baseline")
        if str(self.state.get('score_kind',''))!=score_kind:raise ValueError("Wrong margin convention")
        self.dimension=int(self.state['dimension'])
        if self.dimension not in (4,32,36,40):raise ValueError("Unsupported feature dimension")
        if len(self.state['coef'])!=self.dimension:raise ValueError("Head dimension mismatch")
        if float(self.state['calibration_slope'])<=0:raise ValueError("Positive score calibration required")
        if float(self.state['gate_threshold'])<=0:raise ValueError("Positive training-only gate required")

    def score(self,raw_margin,features,*,enabled=True):
        raw=np.asarray(raw_margin)
        if not enabled:return raw.copy()
        x=np.asarray(features)
        if x.shape!=(len(raw),self.dimension) or not np.isfinite(x).all():raise ValueError("Invalid finite feature matrix")
        strength=float(self.state['strength'])
        fusion=str(self.state['fusion'])
        # This exact early return is part of the whole-path identity contract.
        if fusion=='whole_path' and strength==0:return raw.copy()
        delta,distance=replay(self.state,x)
        threshold=float(self.state['gate_threshold'])
        gate=expit(-(distance-threshold)/(.15*threshold+1e-8))
        calibrated=float(self.state['calibration_slope'])*raw+float(self.state['calibration_intercept'])
        if fusion=='old_residual':return calibrated+strength*gate*delta
        if fusion=='whole_path':return raw+strength*gate*(calibrated-raw+delta)
        raise ValueError("Unknown fusion mode")

    def save(self,path):np.savez(path,**self.state)


class Model(Original):
    """Local independent HYDRA parent with optional phase-dynamic v2 correction."""
    def attach_correction(self,state):
        self.correction=Correction(state,baseline_id='B_HYDRA_FD_fixed20_v1',score_kind='calibrated_svm_logit')
    def decision_function(self,features,*,dynamic=None,enabled=False):
        raw=super().decision_function(features)
        if not enabled:return raw
        if not hasattr(self,'correction'):raise ValueError("Correction not fitted")
        return self.correction.score(raw,dynamic)
