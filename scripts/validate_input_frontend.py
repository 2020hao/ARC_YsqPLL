"""Bounded dual-dataset HYDRA pilot; four arms, fixed five folds and one seed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import _bootstrap as boot
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from arc_model.data.hydra_loader import HydraLoader
from arc_model.models.standalone.hydra import SparseScaler, transform_numpy, make_classifier
from arc_model.methods.phase_representation.input_frontend import pack_current_energy

ROOT = boot.ROOT
OUT = ROOT / 'outputs/input_frontend_pilot_20261029'
SOURCE = ROOT / 'outputs/rcr_three_backbones_20261028/data'
DATASETS = ('continuous5m', 'zenodo5501527')
ARMS = ('fixed_wave', 'rcr_wave', 'fixed_packed', 'rcr_packed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(status='frozen_before_fits', datasets=DATASETS, arms=ARMS,
                folds=5, seeds=[0], models=['HYDRA'], fits=40,
                epochs=70, batch=256, lr=.01, weight_decay=.0001,
                optimizer='AdamW', loss='unweighted cross entropy', threshold=.5,
                scope='exploratory single-seed input replacement; no new network branch',
                comparisons=['rcr_wave-fixed_wave','fixed_packed-fixed_wave',
                             'rcr_packed-rcr_wave','rcr_packed-fixed_packed'],
                common_mask='guard40 intersection; same sample IDs/folds; grouped by condition/source',
                dataset_note='continuous 5m plus public C3, not old cropped data',
                limitations=['different cycle supports: full-front-end comparison',
                             'packed block boundaries and LF decimation change representation',
                             'single seed; historical test exposure; no confirmation claim'],
                gpu_physical=[1,2,3])
    inputs = []
    for ds in DATASETS:
        a, b = (HydraLoader(SOURCE/ds/q) for q in ('fixed20','recovery'))
        assert np.array_equal(a.table.sample_id, b.table.sample_id)
        assert np.array_equal(a.labels,b.labels)
        mask = a.table.guard40.to_numpy(bool) & b.table.guard40.to_numpy(bool)
        reviewed = ROOT/'outputs/rcr_fundamental_consensus_20261028/detector/data'/ds/'recovery/cycles.csv'
        review_table = pd.read_csv(reviewed)
        assert np.array_equal(a.table.sample_id, review_table.sample_id)
        assert np.array_equal(a.labels, review_table.label)
        mask &= review_table.guard40.to_numpy(bool)
        inputs.append(dict(path=str(reviewed),sha256=sha(reviewed),role='latest reviewed common support mask'))
        ids = np.flatnonzero(mask)
        table = a.table.iloc[ids].copy()
        table.to_csv(OUT/f'{ds}_metadata.csv',index=False)
        for fe in ('fixed20','recovery'):
            for name in ('wave.npy','energy.npy','cycles.csv','splits.json'):
                path=SOURCE/ds/fe/name
                inputs.append(dict(path=str(path),sha256=sha(path)))
        np.save(OUT/f'{ds}_indices.npy',ids)
        print('COHORT',ds,len(ids),table.condition_key.nunique(),flush=True)
    spec['inputs']=inputs
    spec['code_sha256']=sha(__file__)
    spec['frontend_sha256']=sha(ROOT/'src/arc_model/methods/phase_representation/input_frontend.py')
    (OUT/'protocol.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))


def fit_job(ds, arm, fold):
    dest=OUT/'runs'/ds/arm/f'f{fold}'
    if (dest/'COMPLETE.json').exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    loader=HydraLoader(SOURCE/ds/('fixed20' if arm.startswith('fixed') else 'recovery'))
    keep=np.load(OUT/f'{ds}_indices.npy')
    tr,te=loader.indices(fold)
    tr=np.intersect1d(tr,keep);te=np.intersect1d(te,keep)
    cache=OUT/'features'/ds/(arm+'.npy');cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        x=np.asarray(loader.wave)
        if arm.endswith('packed'):
            duration=(loader.table.end_seconds-loader.table.start_seconds).to_numpy()
            x=pack_current_energy(x,np.load(loader.root/'energy.npy'),duration)
        z=transform_numpy(x,seed=0,device='cuda',batch_size=128)
        np.save(cache,z)
    z=torch.from_numpy(np.load(cache))
    scaler=SparseScaler();scaler.fit(z[tr]);x=scaler.transform(z).to('cuda')
    labels=torch.as_tensor(loader.labels,device='cuda')
    torch.manual_seed(fold);model=make_classifier(x.shape[1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001)
    g=torch.Generator(device='cuda').manual_seed(90000+fold)
    tri=torch.as_tensor(tr,device='cuda')
    for _ in range(70):
        order=tri[torch.randperm(len(tri),generator=g,device='cuda')]
        for ix in order.split(256):
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[ix]),labels[ix])
            loss.backward();opt.step()
    with torch.no_grad():
        scores=torch.softmax(model(x[te]),1)[:,1].cpu().numpy()
    np.savez(dest/'predictions.npz',indices=te,labels=loader.labels[te],scores=scores)
    np.savez(dest/'fit_state.npz',train=tr,test=te,mu=scaler.mu.numpy(),
             sigma=scaler.sigma.numpy(),epsilon=scaler.epsilon.numpy())
    torch.save(model.state_dict(),dest/'model.pt')
    receipt=dict(status='complete',seconds=time.perf_counter()-start,train=len(tr),test=len(te),
                 parameters=sum(p.numel() for p in model.parameters()),
                 gpu=os.environ['CUDA_VISIBLE_DEVICES'],source_features_sha256=sha(cache),
                 protocol_sha256=sha(OUT/'protocol.json'),
                 predictions_sha256=sha(dest/'predictions.npz'))
    (dest/'COMPLETE.json').write_text(json.dumps(receipt,indent=2))
    print('DONE',ds,arm,fold,round(receipt['seconds'],2),flush=True)


def summarize():
    rows=[];deltas=[]
    for ds in DATASETS:
        tab=pd.read_csv(OUT/f'{ds}_metadata.csv');keep=np.load(OUT/f'{ds}_indices.npy')
        conditions={};all_scores={}
        for arm in ARMS:
            parts=[]
            for f in range(5):
                path=OUT/'runs'/ds/arm/f'f{f}'
                r=json.loads((path/'COMPLETE.json').read_text())
                assert r['predictions_sha256']==sha(path/'predictions.npz')
                z=np.load(path/'predictions.npz');t=pd.DataFrame(dict(index=z['indices'],label=z['labels'],score=z['scores']))
                parts.append(t)
            t=pd.concat(parts).sort_values('index');assert np.array_equal(t['index'],keep)
            y=t.label.to_numpy();s=t.score.to_numpy();p=s>=.5
            assert np.array_equal(y,tab.label.to_numpy())
            per=[]
            for _,g in tab.groupby('condition_key',sort=True):
                j=tab.index.get_indexer(g.index);yy=y[j];pp=p[j]
                assert set(yy)=={0,1}
                per.append(.5*(pp[yy==1].mean()+(~pp[yy==0]).mean()))
            conditions[arm]=np.array(per);all_scores[arm]=s
            rows.append(dict(dataset=ds,arm=arm,N=len(y),conditions=len(per),
                             BAcc=np.mean(per),Recall=recall_score(y,p),FPR=p[y==0].mean(),
                             Precision=precision_score(y,p,zero_division=0),F1=f1_score(y,p),
                             PR_AUC=average_precision_score(y,s)))
        for candidate,baseline in [('rcr_wave','fixed_wave'),('fixed_packed','fixed_wave'),
                                   ('rcr_packed','rcr_wave'),('rcr_packed','fixed_packed')]:
            v=conditions[candidate]-conditions[baseline]
            rng=np.random.default_rng(20261029)
            draws=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)
            lo,hi=np.quantile(draws,[.025,.975])
            deltas.append(dict(dataset=ds,candidate=candidate,baseline=baseline,delta_pp=v.mean()*100,
                               lo_pp=lo*100,hi_pp=hi*100,positive_groups=int((v>0).sum()),groups=len(v)))
        np.savez(OUT/f'{ds}_oof.npz',indices=keep,**all_scores)
    pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
    pd.DataFrame(deltas).to_csv(OUT/'paired_deltas.csv',index=False)
    (OUT/'RESULTS_COMPLETE.json').write_text(json.dumps(dict(status='complete',fits=40,
        note='Single-seed exploratory pilot. Bootstrap intervals are descriptive, unadjusted; not independent confirmation.')))
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(deltas).to_string(index=False))


def run_study(suite, *, execute=False):
    assert suite['specialized_study']=='input_frontend_pilot_20261029'
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if not execute:
        for ds in DATASETS:
            for fe in ('fixed20','recovery'):HydraLoader(SOURCE/ds/fe)
        print('PREFLIGHT_READY: both datasets; 40 fits; GPU1,2,3 only')
        return
    raise ValueError('Freeze with --prepare, then launch three partition workers explicitly')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--partition',type=int)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.summarize:summarize()
    else:
        assert a.partition in (0,1,2) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.08)
        jobs=[(ds,arm,f) for ds in DATASETS for arm in ARMS for f in range(5)]
        # Assign whole (dataset,arm) blocks to a worker: one GPU feature transform per arm.
        for i,(ds,arm,f) in enumerate(jobs):
            if (i//5)%3==a.partition:fit_job(ds,arm,f)
