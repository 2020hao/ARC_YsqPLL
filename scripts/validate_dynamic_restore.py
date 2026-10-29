"""Restore C0 independently, then test fixed universal36 and dynamic32 controls."""

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
from scipy.special import expit
from arc_model.data.hydra_loader import HydraLoader
from arc_model.methods import dynamic_correction as method

ROOT=boot.ROOT
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
COHORT=ROOT/'outputs/shared_eight_study_20261029'
OUT=ROOT/'outputs/dynamic_restore_20261029'
DATASETS=('continuous5m','zenodo5501527')
MODELS=('ArcFENet','HYDRA','ArcFusionNet')
SEEDS=(0,1,2)
ARMS=('legacy_refit','uniform36','dynamic32')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(doc,ensure_ascii=False,indent=2));t.replace(path)


def logit(p):
    # Exact historical clipping (not a new calibration transform).
    return np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))


def verified_bundle(ds,model,fold,seed):
    q=SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s{seed}'
    receipt=json.loads((q/'COMPLETE.json').read_text())
    assert receipt['prediction_sha256']==sha(q/'predictions.npz')
    assert receipt['head_sha256']==sha(q/'head.npz')
    state=np.load(q/'fit_indices.npz');z=np.load(q/'predictions.npz');cross=np.load(q/'crossfit_scores.npz')
    tr,te=state['train'],state['test'];assert np.array_equal(cross['indices'],tr) and np.array_equal(z['indices'],te)
    n=len(pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv'))
    offset=np.zeros(n);offset[tr]=cross['base_logit'];offset[te]=logit(z['M0'])
    rebuilt=np.zeros(n);counts=np.zeros(n,int)
    for inner in range(3):
        p=q/f'inner{inner}'
        v=np.load(p/'M0/predictions.npz');i=np.load(p/'fit_indices.npz');train,validation=i['train'],v['indices']
        assert not np.intersect1d(train,validation).size and not np.intersect1d(train,te).size and not np.intersect1d(validation,te).size
        assert np.isin(np.r_[train,validation],tr).all() and np.array_equal(validation,i['validation'])
        rebuilt[validation]=logit(v['probability']);counts[validation]+=1
    assert (counts[tr]==1).all() and (counts[te]==0).all()
    np.testing.assert_array_equal(rebuilt[tr],cross['base_logit'])
    return q,state,z,offset


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already exists')
    OUT.mkdir(parents=True,exist_ok=True);identity=[]
    frozen=json.loads((SOURCE/'frozen_inputs.json').read_text())
    for ds in DATASETS:
        d=HydraLoader(SOURCE/'data'/ds/'recovery');keep=np.load(COHORT/f'{ds}_indices.npy')
        np.save(OUT/f'{ds}_indices.npy',keep);d.table.iloc[keep].to_csv(OUT/f'{ds}_metadata.csv',index=False)
        x=np.load(d.root/'dynamic.npy');assert x.shape==(len(d.table),36)
        assert np.isfinite(x).all()
        for name in ['dynamic.npy','energy.npy','cycles.csv','splits.json']:
            p=d.root/name;value=sha(p)
            if str(p) in frozen:assert value==frozen[str(p)]
            identity.append(dict(path=str(p),sha256=value))
        for seed in SEEDS:
            for fold in range(5):
                selected=[]
                for model in MODELS:
                    q,ids,z,base=verified_bundle(ds,model,fold,seed)
                    tr,te=d.indices(fold)
                    assert np.array_equal(tr,ids['train']) and np.array_equal(te,ids['test'])
                    fit,cal=method.calibration_split(d.table.condition_key.to_numpy(),tr,fold)
                    assert np.array_equal(fit,ids['head_fit']) and np.array_equal(cal,ids['head_calibration'])
                    for column in ['condition_key','source_group','record_id']:
                        assert not set(d.table.iloc[tr][column])&set(d.table.iloc[te][column])
                    selected.append(np.load(q/'head.npz')['keep'])
                    for fn in ['head.npz','predictions.npz','crossfit_scores.npz','fit_indices.npz','selection.json']:
                        p=q/fn;identity.append(dict(path=str(p),sha256=sha(p)))
                assert all(np.array_equal(selected[0],v) for v in selected[1:])
        print('VERIFIED',ds,len(keep),flush=True)
    code=[Path(__file__),ROOT/'src/arc_model/methods/dynamic_correction.py',ROOT/'src/arc_model/methods/cycle_dynamics.py']
    spec=dict(status='frozen_before_new_head_fits',datasets=DATASETS,models=MODELS,seeds=SEEDS,folds=5,
        arms=ARMS,reference=['M0','legacy_saved','legacy_replay'],head_bundles=90,new_fits=540,
        features=list(method.feature_names()),variants=dict(legacy_refit='historical training-only selector max24',
            uniform36='same fixed36 for all models; no feature selection',dynamic32='same fixed32 dynamics, no availability masks'),
        history='2native-rateHFbands x20phasebins; lags1,2,4,8; 4stats perband perlag plus4availability masks',
        training='historical zero-initialized linear residual,80epochs AdamW lr.002 wd.002 batch256 plus2e-4meanL1; initial fit/calibration then outer-train refit',
        calibration='training-only alpha[0,.1,.25,.5,1,1.5,2] and normal PCA-distance q[.75,.9,.95,.975]; calFPR<=baseFPR+2pp',
        statistics='seed-average condition deltas; cluster bootstrap5000/signflip20000; 12primary contrasts uniform36-M0 and uniform36-legacy_saved Holm; dynamic32 comparisons separate12secondary',
        scope='restore current continuous RCR+C0, not old cropped data; cached original backbone and honest3innerOOF reused; no backbone retraining; fixed-source exploratory validation',
        interpretation='replay/refit reproducibility is not novel-method or independent-data confirmation; insignificant loss is not equivalence',
        precision='FP32 residuals, matmulTF32 disabled; prior references retained',gpu_physical=[1,2,3],
        inputs=identity,code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in code])
    save(OUT/'protocol.json',spec)


def run_job(ds,model,fold,seed):
    dest=OUT/'runs'/ds/model/f'f{fold}_s{seed}'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);tick=time.perf_counter()
    tab=pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv');y=tab.label.to_numpy();groups=tab.condition_key.to_numpy()
    x=np.load(SOURCE/'data'/ds/'recovery/dynamic.npy')
    source,ids,saved,base=verified_bundle(ds,model,fold,seed)
    tr,te=ids['train'],ids['test'];fit,cal=method.calibration_split(groups,tr,fold)
    teval=np.intersect1d(te,np.load(OUT/f'{ds}_indices.npy'))
    state=np.load(source/'head.npz');selection=json.loads((source/'selection.json').read_text())
    delta,distance=method.replay(state,x)
    replay=expit(base[te]+method.weights(distance,tr[y[tr]==0],selection['alpha'],selection['quantile'])[te]*delta[te])
    replay_error=float(abs(replay-saved['C0']).max())
    if replay_error>5e-5:raise ValueError(f'Legacy replay mismatch {replay_error}')
    lookup={int(v):j for j,v in enumerate(te)};view=np.array([lookup[int(j)] for j in teval])
    predictions=dict(indices=teval,labels=y[teval],M0=saved['M0'][view],legacy_saved=saved['C0'][view],legacy_replay=replay[view])
    diagnostics=dict(legacy_replay_max_error=replay_error,source=str(source))
    for arm in ARMS:
        columns=method.select_columns(x,fit,y,groups) if arm=='legacy_refit' else np.arange(36 if arm=='uniform36' else 32)
        de,od,first=method.fit(x,y,base,fit,columns,seed*100+fold,device='cuda:0')
        (alpha,quantile),trace=method.calibrate(y,groups,base,de,od,fit,cal)
        np.savez(dest/(arm+'_initial.npz'),**first)
        columns=method.select_columns(x,tr,y,groups) if arm=='legacy_refit' else columns
        de,od,final=method.fit(x,y,base,tr,columns,seed*100+fold+31,device='cuda:0')
        p=expit(base[teval]+method.weights(od,tr[y[tr]==0],alpha,quantile)[teval]*de[teval])
        predictions[arm]=p;np.savez(dest/(arm+'_head.npz'),**final)
        save(dest/(arm+'_selection.json'),dict(alpha=alpha,quantile=quantile,trace=trace))
        if arm=='legacy_refit':
            diagnostics.update(legacy_refit_max_error=float(abs(p-saved['C0'][view]).max()),
                selected_columns_equal=bool(np.array_equal(final['keep'],state['keep'])),
                final_coefficient_max_error=float(abs(final['coef']-state['coef']).max()) if np.array_equal(final['keep'],state['keep']) else None,
                selected_alpha_equal=alpha==selection['alpha'],selected_quantile_equal=quantile==selection['quantile'])
    np.savez(dest/'predictions.npz',**predictions);np.savez(dest/'indices.npz',train=tr,test=teval,head_fit=fit,head_calibration=cal)
    save(dest/'diagnostics.json',diagnostics)
    save(dest/'COMPLETE.json',dict(status='complete',dataset=ds,model=model,fold=fold,seed=seed,
        protocol_sha256=sha(OUT/'protocol.json'),predictions_sha256=sha(dest/'predictions.npz'),
        seconds=time.perf_counter()-tick,gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],new_fits=6))
    print('DONE',ds,model,fold,seed,round(time.perf_counter()-tick,2),diagnostics['legacy_refit_max_error'],flush=True)


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Prepare/freeze, then launch three workers explicitly')
    for ds in DATASETS:HydraLoader(SOURCE/'data'/ds/'recovery')
    print('PREFLIGHT_READY: restore C0 and fixed universal dynamics, both datasets, GPU1/2/3')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--partition',type=int);a=p.parse_args()
    if a.prepare:prepare()
    else:
        assert a.partition in [0,1,2] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1);torch.backends.cuda.matmul.allow_tf32=False
        jobs=[(ds,model,fold,seed) for seed in SEEDS for fold in range(5) for ds in DATASETS for model in MODELS]
        for i,job in enumerate(jobs):
            if i%3==a.partition:run_job(*job)
