"""HYDRA published alternating-window/5-holdout protocol; no added research module.

Current reviewed raw-derived cohorts and the author's released MAT are separate
paths. No window-to-MAT mapping is assumed. Original grouped tests remain intact.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import _bootstrap as boot
import complete_paper_baselines as baseline
from arc_model.evaluation.cycle_metrics import cycle_metrics
from arc_model.models.standalone.hydra import SparseScaler
from arc_model.models.standalone.hydra_paper import fit_l1_hinge

ROOT=boot.ROOT
OUT=ROOT/'outputs/hydra_alternating_protocol_20261029'
MAT=Path('/data/hzy/Document/arc_selfsync_research/code_availability_audit/HYDRA_author/Datasets/dataset_2classes_20ms_50kHz.mat')
DATASETS=('continuous5m','zenodo5501527','author_mat')


def alternating_membership(table,origins):
    """Parity of original continuous 20ms ordinal, before guard/label exclusions."""
    origin=np.array([origins[int(v)] for v in table.record_id])
    value=(table.start_seconds.to_numpy()-origin)/.02
    if np.max(abs(value-np.rint(value)))>1e-6:
        raise ValueError('Window is not on the original nominal20ms grid')
    ordinal=np.rint(value).astype(int)
    if (ordinal<0).any() or table.assign(ordinal=ordinal).duplicated(['record_id','ordinal']).any():
        raise ValueError('Negative/duplicate original window ordinal')
    return ordinal,ordinal%2==0


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError(OUT)
    OUT.mkdir(exist_ok=True,parents=True);identities=[];inventory=[]
    for ds in DATASETS[:2]:
        table=pd.read_csv(baseline.FIXED/ds/'fixed20/cycles.csv')
        keep=np.load(baseline.COHORT/f'{ds}_indices.npy')
        tab=table.iloc[keep].copy();origins={}
        for rid in tab.record_id.unique():
            rec=json.loads((baseline.RECORDS/ds/f'{rid:04}'/'COMPLETE.json').read_text())
            origin=0.
            if ds=='zenodo5501527':
                with open(rec['source_file']) as f:origin=float(f.readline().split()[0])
            origins[int(rid)]=origin
        ordinal,is_train=alternating_membership(tab,origins)
        tr,te=keep[is_train],keep[~is_train]
        assert not np.intersect1d(tr,te).size
        assert set(table.iloc[tr].record_id)==set(table.iloc[te].record_id)
        assert set(table.iloc[tr].condition_key)==set(table.iloc[te].condition_key)
        y=table.label.to_numpy()
        p=OUT/'data'/ds;p.mkdir(parents=True)
        np.savez(p/'partition.npz',train_pool=tr,test=te,labels=y,reviewed_indices=keep,ordinal=ordinal)
        tab['original_window_ordinal']=ordinal;tab['partition']=np.where(is_train,'train_pool','test')
        tab.to_csv(p/'partition_metadata.csv',index=False)
        for q in [p/'partition.npz',p/'partition_metadata.csv',baseline.OUT/'data'/ds/'author_hydra.npy',
                  baseline.FIXED/ds/'fixed20/cycles.csv',baseline.COHORT/f'{ds}_indices.npy',
                  baseline.OUT/f'{ds}_HYDRA_paper_s0_oof.npz']:
            identities.append(dict(path=str(q),sha256=baseline.sha(q)))
        inventory.append(dict(dataset=ds,reviewed_N=len(keep),train_pool=len(tr),test=len(te),
            train_pool_normal=int((y[tr]==0).sum()),train_pool_arc=int((y[tr]==1).sum()),
            test_normal=int((y[te]==0).sum()),test_arc=int((y[te]==1).sum()),
            shared_records=table.iloc[te].record_id.nunique(),shared_conditions=table.iloc[te].condition_key.nunique()))
    raw=loadmat(MAT);p=OUT/'data/author_mat';p.mkdir(parents=True)
    # Unmodified author transform on CPU, as in the previous author-data reproduction.
    sys.path.insert(0,str(ROOT/'third_party/hydra'))
    from hydra_class import Hydra
    transform=Hydra(1000,k=8,g=8,kernel_length=5,stride=1,divisor=1,seed=0)
    with torch.no_grad():
        a=transform.batch(torch.from_numpy(raw['yTrain'][:,None,:]),128).numpy()
        b=transform.batch(torch.from_numpy(raw['yTest'][:,None,:]),128).numpy()
    np.save(p/'features.npy',np.r_[a,b])
    y=np.r_[raw['labelsTrain'].ravel(),raw['labelsTest'].ravel()].astype(int)
    assert set(y)=={0,1};tr=np.arange(len(a));te=np.arange(len(a),len(a)+len(b))
    np.savez(p/'partition.npz',train_pool=tr,test=te,labels=y)
    for q in [MAT,p/'features.npy',p/'partition.npz',ROOT/'third_party/hydra/hydra_class.py']:
        identities.append(dict(path=str(q),sha256=baseline.sha(q)))
    inventory.append(dict(dataset='author_mat',reviewed_N=len(y),train_pool=len(tr),test=len(te),
        train_pool_normal=int((y[tr]==0).sum()),train_pool_arc=int((y[tr]==1).sum()),
        test_normal=int((y[te]==0).sum()),test_arc=int((y[te]==1).sum()),
        shared_records=None,shared_conditions=None))
    pd.DataFrame(inventory).to_csv(OUT/'cohort_counts.csv',index=False)
    protocol=dict(status='frozen_before_fits',datasets=DATASETS,repeats=[1,2,3,4,5],fits=15,
        partition='original20ms ordinal even=train_pool, odd=test; do not renumber after exclusions; author MAT released train/test used directly',
        training='each repeat stratified random80% of train_pool; discarded20% not used for scaler/training/tuning; fixed test',
        hyperparameters=dict(t_ms=20,fs_hz=50000,k=8,g=8,m=5,stride=1,divisor=1,hydra_seed=0,lambda_l1=.001),
        optimizer='same mean-hinge plus L1 objective, unpenalized intercept; exact HiGHS optimal solution required;900s timeout',
        label_semantics='NOARC=0 ARC=1, verified original Table2; current reviewed intervals unchanged',
        comparisons='current grouped HYDRA OOF overall and restricted to identical alternating-test rows; author MAT separate reference',
        precision='CPU float32 author transform/scaler, float64 HiGHS; current features reused after identity checks',
        device='CPU LP is the classifier; no neural training; physicalGPU0 excluded, workers expose1/2/3 only',
        scope='within-record window recognition, not unseen-appliance evaluation; 5holdouts share the same test samples',
        differences=['MATLAB cvpartition replaced by sklearn stratification','HiGHS instead of SpaRSA',
            'current raw-derived labels/filter/guard/range differ from author MAT; original grid parity fixes only the split',
            'no assumed per-window mapping between MAT and raw DAT; no dataset pooling or unreviewed label regeneration'],
        identities=identities,code=[dict(path=str(q.relative_to(ROOT)),sha256=baseline.sha(q)) for q in
            [Path(__file__),ROOT/'src/arc_model/models/standalone/hydra_paper.py',ROOT/'src/arc_model/evaluation/cycle_metrics.py']])
    baseline.save(OUT/'protocol.json',protocol)
    print(pd.DataFrame(inventory).to_string(index=False),flush=True)


def feature_path(ds):
    return OUT/'data/author_mat/features.npy' if ds=='author_mat' else baseline.OUT/'data'/ds/'author_hydra.npy'


def run_job(ds, repeat):
    q=OUT/'runs'/ds/f'repeat{repeat}'
    if q.exists():raise FileExistsError(q)
    q.mkdir(parents=True);start=time.perf_counter()
    z=np.load(OUT/'data'/ds/'partition.npz');tr_pool,te,y=z['train_pool'],z['test'],z['labels']
    tr,discard=train_test_split(tr_pool,test_size=.2,stratify=y[tr_pool],random_state=repeat)
    assert not np.intersect1d(tr,te).size and not np.intersect1d(discard,te).size
    features=torch.from_numpy(np.load(feature_path(ds)))
    sc=SparseScaler();x=sc.fit_transform(features[tr]).numpy();xt=sc.transform(features[te]).numpy()
    w,b,diagnostics=fit_l1_hinge(x,y[tr],lambda_l1=.001)
    score=xt@w+b;prediction=score>=0
    np.savez(q/'state.npz',weight=w,intercept=b,mu=sc.mu.numpy(),sigma=sc.sigma.numpy(),epsilon=sc.epsilon.numpy())
    np.savez(q/'predictions.npz',indices=te,labels=y[te],score=score,prediction=prediction)
    np.savez(q/'indices.npz',train=tr,discarded=discard,test=te)
    baseline.save(q/'solver.json',diagnostics)
    baseline.save(q/'COMPLETE.json',dict(status='complete',dataset=ds,repeat=repeat,
        train=len(tr),discard=len(discard),test=len(te),seconds=time.perf_counter()-start,
        protocol_sha256=baseline.sha(OUT/'protocol.json'),state_sha256=baseline.sha(q/'state.npz'),
        predictions_sha256=baseline.sha(q/'predictions.npz'),device='CPU HiGHS',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print('DONE',ds,repeat,len(tr),len(te),round(time.perf_counter()-start,2),
        'Accuracy',float((prediction==y[te]).mean()),flush=True)


def summarize():
    spec=json.loads((OUT/'protocol.json').read_text());rows=[];references=[];audits=[];devices=[]
    for a in spec['identities']:assert baseline.sha(a['path'])==a['sha256']
    for a in spec['code']:assert baseline.sha(ROOT/a['path'])==a['sha256']
    for ds in DATASETS:
        source=np.load(OUT/'data'/ds/'partition.npz');pool,te,y=source['train_pool'],source['test'],source['labels']
        table=pd.read_csv(baseline.FIXED/ds/'fixed20/cycles.csv') if ds!='author_mat' else None
        conditions=table.iloc[te].condition_key.to_numpy() if table is not None else np.full(len(te),'pooled_only')
        features=torch.from_numpy(np.load(feature_path(ds)))
        for repeat in spec['repeats']:
            q=OUT/'runs'/ds/f'repeat{repeat}';z=np.load(q/'predictions.npz');state=np.load(q/'state.npz');i=np.load(q/'indices.npz')
            done=json.loads((q/'COMPLETE.json').read_text());tr,unused=train_test_split(pool,test_size=.2,stratify=y[pool],random_state=repeat)
            assert np.array_equal(tr,i['train']) and np.array_equal(unused,i['discarded']) and np.array_equal(te,i['test'])
            assert np.array_equal(te,z['indices']) and np.array_equal(y[te],z['labels'])
            assert done['protocol_sha256']==baseline.sha(OUT/'protocol.json')
            assert done['state_sha256']==baseline.sha(q/'state.npz') and done['predictions_sha256']==baseline.sha(q/'predictions.npz')
            assert done['cuda_visible_devices'] in ['1','2','3']
            sc=SparseScaler();sc.fit(features[tr])
            for name in ['mu','sigma','epsilon']:np.testing.assert_array_equal(state[name],getattr(sc,name).numpy())
            rebuilt=sc.transform(features[te]).numpy()@state['weight']+state['intercept'];error=float(abs(rebuilt-z['score']).max())
            assert error<1e-7 and np.array_equal(rebuilt>=0,z['prediction'])
            diag=json.loads((q/'solver.json').read_text());assert diag['max_constraint_violation']<1e-6
            m=cycle_metrics(y[te],z['score'],conditions,threshold=0.)
            if ds=='author_mat':m['BAcc_condition']=None;m['groups']=None
            rows.append(dict(dataset=ds,repeat=repeat,**m));audits.append(dict(dataset=ds,repeat=repeat,replay_error=error,seconds=done['seconds']))
            if table is not None:
                for appliance,g in table.iloc[te].assign(score=z['score']).groupby('appliance'):
                    mm=cycle_metrics(g.label.to_numpy(),g.score.to_numpy(),g.condition_key.to_numpy(),threshold=0)
                    devices.append(dict(dataset=ds,repeat=repeat,appliance=appliance,**mm))
        if table is not None:
            old=np.load(baseline.OUT/f'{ds}_HYDRA_paper_s0_oof.npz')
            lookup={int(v):j for j,v in enumerate(old['indices'])};view=np.array([lookup[int(v)] for v in te])
            assert np.array_equal(old['labels'][view],y[te])
            refs=[('grouped_full_reviewed',old['labels'],old['score'],table.iloc[old['indices']].condition_key.to_numpy()),
                  ('grouped_same_alternating_test',y[te],old['score'][view],conditions)]
            for name,yy,score,group in refs:references.append(dict(dataset=ds,reference=name,**cycle_metrics(yy,score,group,threshold=0)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'repeat_metrics.csv',index=False)
    columns=['Accuracy','BAcc_condition','BAcc_pooled','Precision','Recall','Specificity','F1','FPR','FNR','ROC_AUC','AP','MCC']
    means=df.groupby('dataset',sort=False)[columns].mean().reset_index();means.to_csv(OUT/'main_results.csv',index=False)
    df.groupby('dataset',sort=False)[columns].std().reset_index().to_csv(OUT/'repeat_sd.csv',index=False)
    pd.DataFrame(references).to_csv(OUT/'grouped_references.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'fit_audit.csv',index=False);pd.DataFrame(devices).to_csv(OUT/'appliance_metrics.csv',index=False)
    baseline.save(OUT/'AUDIT_PASS.json',dict(status='pass',fits=len(rows),max_replay_error=max(x['replay_error'] for x in audits),
        original_ordinal_parity=True,train_test_windows_disjoint=True,train_test_records_intentionally_shared=True,
        train_only_scaler=True,holdout_unused=True,unchanged_labels_and_inputs=True,gpu0_used=False,
        current_cohorts_and_author_MAT_separate=True))
    baseline.save(OUT/'RESULTS_COMPLETE.json',dict(status='complete',fits=len(rows),repeats=5,
        cohorts=DATASETS,author_target_accuracy=.996252,test_cohorts=3,
        note='each cohort reuses one fixed test set; author MAT and public raw derive from the same published source, not independent datasets'))
    print(means.to_string(index=False));print(pd.DataFrame(references).to_string(index=False))


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Prepare/freeze then explicit workers; no grouped benchmark overwritten')
    print('PREFLIGHT: both reviewed datasets alternate20ms; separate author MAT; 15 L1-SVM fits')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--partition',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.prepare:prepare()
    elif a.summarize:summarize()
    else:
        assert a.partition in [0,1,2] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
        jobs=[(ds,r) for ds in DATASETS for r in range(1,6)]
        for i,job in enumerate(jobs):
            if i%3==a.partition:run_job(*job)
