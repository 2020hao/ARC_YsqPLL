"""Fix the current-only Fusion training cohort and freeze three local comparators.

This never promotes current-only Fusion to the voltage/differential author method.
Unchanged source folds can be reused only after exact train-index equality.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
import _bootstrap as boot
import complete_paper_baselines as base
from arc_model.models.standalone.arc_fusion import FusionNet
from arc_model.evaluation.cycle_metrics import cycle_metrics

ROOT=boot.ROOT
OUT=ROOT/'outputs/grouped_baseline_standard_20261029'
OLD=ROOT/'outputs/rcr_three_backbones_20261028/runs'
DATASETS=base.DATASETS


def location(ds,fold,seed):
    return OUT/'runs'/ds/f'f{fold}_s{seed}'


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True,exist_ok=True);jobs=[];identities=[]
    for ds in DATASETS:
        for n in ['temporal.npy','spectrum.npy','cycles.csv']:
            p=base.FIXED/ds/'fixed20'/n;identities.append(dict(path=str(p),sha256=base.sha(p)))
        for p in [base.COHORT/f'{ds}_indices.npy',base.CURRENT/'data'/ds/'recovery/splits.json']:
            identities.append(dict(path=str(p),sha256=base.sha(p)))
        for seed in [0,1,2]:
            for fold in range(5):
                _,tr,te=base.indices(ds,fold);src=OLD/ds/'fixed20/ArcFusionNet'/f'f{fold}_s{seed}'
                old=np.load(src/'fit_indices.npz')['train'];reuse=np.array_equal(old,tr)
                jobs.append(dict(dataset=ds,fold=fold,seed=seed,reuse=reuse))
                if reuse:
                    for n in ['outer/M0/model.pt','outer/M0/predictions.npz','outer/M0/COMPLETE.json','outer/preprocessing.npz']:
                        p=src/n;identities.append(dict(path=str(p),sha256=base.sha(p)))
    # Order the long self-collected refits across all three GPUs.
    jobs.sort(key=lambda j:(not j['reuse'],j['dataset']=='continuous5m',j['seed'],j['fold']))
    code=[Path(__file__),ROOT/'src/arc_model/models/standalone/arc_fusion.py',ROOT/'src/arc_model/evaluation/cycle_metrics.py']
    spec=dict(status='frozen_before_refits',datasets=DATASETS,models=['ArcFENet_reconstructed','HYDRA_paper_FD','ArcFusionNet_current'],
        jobs=jobs,new_Fusion_fits=sum(not j['reuse'] for j in jobs),reused_Fusion_fits=sum(j['reuse'] for j in jobs),
        cohorts=dict(continuous5m=38374,zenodo5501527=1361),grouped_folds=5,
        training=dict(epochs=70,batch=64,optimizer='AdamW',lr=.0001,min_lr=.000001,weight_decay=.01,
            scheduler='cosine local realization of author adaptive rate; weight decay undocumented in paper'),
        preprocessing=dict(input='fixed20ms current-only250points + first/second gradient and RF top32 FFT',
            RF='200trees,max_depth8,min_samples_leaf8,class_weight balanced,seed1000*s+fold; local undisclosed choices',
            fitting='RF selection and all means/std only current outer training rows'),
        architecture='independent ArcFusionNet-current; extra=None; CNN/Transformer/cross-attention retained; no voltage or synthetic adaptation claimed',
        execution=dict(physical_gpu=[1,2,3],model='BF16 autocast; FP32 parameters/loss/softmax',
            matmul_tf32=False,cudnn_tf32=False,cudnn_deterministic=True,
            inputs='GPU resident identical values; no epochs/batch/architecture reduction'),
        other_baselines='reuse completed paper_baseline_completion_20261029 FENet30/HYDRA10; exact current reviewed index protocol',
        ArcFENet_status='local reconstruction, limited numerical VMD convergence; sensitivity audit separate, no automatic input changes',
        scope='local implementation benchmark; not three complete author-source reproductions; previously exposed data exploratory',
        identities=identities,code=[dict(path=str(p.relative_to(ROOT)),sha256=base.sha(p)) for p in code])
    base.save(OUT/'protocol.json',spec);print('FROZEN',spec['new_Fusion_fits'],spec['reused_Fusion_fits'],flush=True)


def train_job(j):
    ds,fold,seed=(j[k] for k in ['dataset','fold','seed']);dest=location(ds,fold,seed)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);tick=time.perf_counter();table,tr,te=base.indices(ds,fold)
    np.savez(dest/'indices.npz',train=tr,test=te);src=OLD/ds/'fixed20/ArcFusionNet'/f'f{fold}_s{seed}'
    y=table.label.to_numpy();runseed=1000*seed+fold
    if j['reuse']:
        old=np.load(src/'fit_indices.npz')['train'];assert np.array_equal(old,tr)
        state=np.load(src/'outer/preprocessing.npz');assert np.array_equal(state['fit_indices'],tr)
        z=np.load(src/'outer/M0/predictions.npz');lookup={int(v):i for i,v in enumerate(z['indices'])};ii=np.array([lookup[int(v)] for v in te])
        assert np.array_equal(z['labels'][ii],y[te])
        shutil.copyfile(src/'outer/M0/model.pt',dest/'model.pt');shutil.copyfile(src/'outer/preprocessing.npz',dest/'preprocessing.npz')
        probability=z['probability'][ii];history=[]
    else:
        d=base.FIXED/ds/'fixed20';temporal=np.load(d/'temporal.npy');spectrum=np.load(d/'spectrum.npy')
        rf=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=8,class_weight='balanced',random_state=runseed,n_jobs=2)
        rf.fit(spectrum[tr],y[tr]);bins=np.sort(np.argsort(rf.feature_importances_)[-32:]);a=spectrum[:,bins]
        mu_t=temporal[tr].mean((0,3),keepdims=True);sd_t=np.maximum(temporal[tr].std((0,3),keepdims=True),1e-6)
        mu_s=a[tr].mean(0);sd_s=np.maximum(a[tr].std(0),1e-6)
        np.savez(dest/'preprocessing.npz',fit_indices=tr,harmonic_indices=bins,rf_importance=rf.feature_importances_,
            temporal_mean=mu_t,temporal_std=sd_t,spectral_mean=mu_s,spectral_std=sd_s)
        x=torch.tensor((temporal-mu_t)/sd_t,device='cuda');s=torch.tensor(((a-mu_s)/sd_s)[:,None],device='cuda')
        labels=torch.tensor(y,device='cuda');tri=torch.tensor(tr,device='cuda')
        torch.manual_seed(runseed);net=FusionNet(extra=0).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,70,eta_min=1e-6)
        generator=torch.Generator(device='cuda').manual_seed(runseed+90000);history=[]
        for epoch in range(70):
            net.train();order=tri[torch.randperm(len(tri),generator=generator,device='cuda')];total=torch.zeros((),device='cuda')
            for ids in order.split(64):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logit=net(x[ids],s[ids])
                loss=torch.nn.functional.cross_entropy(logit.float(),labels[ids]);loss.backward();optimizer.step();total+=loss.detach()*len(ids)
            value=float(total)/len(tr);assert np.isfinite(value)
            history.append(dict(epoch=epoch+1,loss=value,lr=optimizer.param_groups[0]['lr']));scheduler.step()
            base.save(dest/'progress.json',dict(**j,epoch=epoch+1,epochs=70,seconds=time.perf_counter()-tick,loss=value))
            if epoch==0 or (epoch+1)%10==0:print('EPOCH',ds,fold,seed,epoch+1,round(time.perf_counter()-tick,1),flush=True)
        net.eval();parts=[]
        with torch.no_grad():
            for ids in torch.tensor(te,device='cuda').split(64):
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x[ids],s[ids])
                parts.append(logits.float().softmax(-1)[:,1].cpu().numpy())
        probability=np.concatenate(parts);torch.save(net.state_dict(),dest/'model.pt')
        del net,optimizer,scheduler,x,s,labels,tri;torch.cuda.empty_cache()
    np.savez(dest/'predictions.npz',indices=te,labels=y[te],probability=probability)
    base.save(dest/'history.json',history)
    base.save(dest/'COMPLETE.json',dict(status='complete',**j,runseed=runseed,train=len(tr),test=len(te),
        seconds=time.perf_counter()-tick,gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],
        protocol_sha256=base.sha(OUT/'protocol.json'),model_sha256=base.sha(dest/'model.pt'),
        preprocessing_sha256=base.sha(dest/'preprocessing.npz'),predictions_sha256=base.sha(dest/'predictions.npz')))
    print('DONE',j,round(time.perf_counter()-tick,1),flush=True)


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Prepare/freeze, then explicit workers; GPU1/2/3 only')
    print('PREFLIGHT: current-grouped Fusion standard,24refits+6index-exact reuses; paper original boundary disclosed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--partition',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.prepare:prepare()
    else:
        assert a.partition in [0,1,2] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
        torch.cuda.set_per_process_memory_fraction(.4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
        spec=json.loads((OUT/'protocol.json').read_text())
        for i,j in enumerate(spec['jobs']):
            if i%3==a.partition:train_job(j)
