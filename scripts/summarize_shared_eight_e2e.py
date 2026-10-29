"""Metrics and paired inference for the from-scratch common-eight experiment."""

import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import _bootstrap as boot

OUT=boot.ROOT/'outputs/shared_eight_e2e_20261029'
OLD=boot.ROOT/'outputs/shared_eight_study_20261029'
ARMS=('original','fixed8','phase8')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def holm(table,indices):
    ordered=table.loc[indices,'p'].sort_values().index
    values=np.maximum.accumulate([min(1.,table.loc[ix,'p']*(len(ordered)-i)) for i,ix in enumerate(ordered)])
    table.loc[ordered,'p_holm']=values


def group_bacc(tab,scores):
    vals=[]
    for name,g in tab.groupby('condition_key',sort=True):
        y=g.label.to_numpy();p=scores[g.index]>=.5
        if set(y)!={0,1}:continue
        vals.append((name,.5*(p[y==1].mean()+(~p[y==0]).mean())))
    return vals


def main(partial=False):
    spec=json.loads((OUT/'protocol.json').read_text());rows=[];deltas=[];groups=[];cost=[];strata=[];receipts=[]
    if not partial:
        for r in spec['input_hashes']:assert sha(r['path'])==r['sha256']
        for r in spec['code_hashes']:assert sha(boot.ROOT/r['path'])==r['sha256']
    completed=0
    for ds in spec['datasets']:
        tab=pd.read_csv(OUT/f'{ds}_metadata.csv');keep=np.load(OUT/f'{ds}_indices.npy')
        full=pd.read_csv(boot.ROOT/'outputs/rcr_fundamental_consensus_20261028/detector/data'/ds/'recovery/cycles.csv')
        splits=json.loads((boot.ROOT/'outputs/rcr_fundamental_consensus_20261028/detector/data'/ds/'recovery/splits.json').read_text())
        for model in spec['models']:
            paths=[OUT/'runs'/ds/model/f'f{f}'/a/'COMPLETE.json' for f in range(5) for a in ARMS]
            if not all(p.exists() for p in paths):
                if partial:continue
                raise FileNotFoundError(f'Incomplete {ds}/{model}')
            condition={}
            for arm in ARMS:
                pieces=[]
                for fold in range(5):
                    q=OUT/'runs'/ds/model/f'f{fold}'/arm;r=json.loads((q/'COMPLETE.json').read_text())
                    assert r['protocol_sha256']==sha(OUT/'protocol.json')
                    assert r['checkpoint_sha256']==sha(q/'model.pt') and r['prediction_sha256']==sha(q/'predictions.npz')
                    assert r['gpu_physical'] in ['1','2','3']
                    z=np.load(q/'predictions.npz');ix=np.load(q/'indices.npz');tr,te=ix['train'],ix['test']
                    assert np.array_equal(tr,splits[fold]['train']) and np.array_equal(te,np.intersect1d(splits[fold]['test'],keep))
                    for col in ['condition_key','source_group','record_id']:
                        assert not set(full.iloc[tr][col])&set(full.iloc[te][col])
                    assert np.array_equal(z['indices'],te) and np.array_equal(z['labels'],full.label.to_numpy()[te])
                    assert r['settings']==spec['settings'][model] and len(r['history'])==r['settings']['epochs']
                    if arm!='original':
                        prep=np.load(q/'preprocessing.npz');f=np.load(OUT/f'{ds}_{"fixed" if arm=="fixed8" else "phase"}8.npy')
                        np.testing.assert_array_equal(prep['feature_mean'],f[tr].mean(0))
                        np.testing.assert_array_equal(prep['feature_std'],np.maximum(f[tr].std(0),1e-6))
                    diag=json.loads((q/'training_diagnostics.json').read_text())
                    if model!='HYDRA':
                        keys=[k for k in diag['parameter_change_norm'] if 'head' not in k and not k.startswith('extra.')]
                        assert any(diag['parameter_change_norm'][k]>0 for k in keys)
                        assert any(diag['first_batch_gradient_norm'].get(k,0)>0 for k in keys)
                    pieces.append(pd.DataFrame(dict(index=te,labels=z['labels'],scores=z['scores'])))
                    completed+=1
                    cost.append(dict(dataset=ds,model=model,fold=fold,arm=arm,seconds=r['training_seconds'],
                                     parameters=r['trainable_parameters'],peak_bytes=r['peak_allocated_bytes']))
                    receipts.append(dict(dataset=ds,model=model,fold=fold,arm=arm,
                                         core_initial_hash=r['common_initial_parameter_hash'],gpu=r['gpu_physical']))
                t=pd.concat(pieces).sort_values('index');assert np.array_equal(t['index'],keep)
                y=t.labels.to_numpy();s=t.scores.to_numpy();p=s>=.5
                vals=group_bacc(tab,s);assert len(vals)==tab.condition_key.nunique()
                condition[arm]=np.array([v for _,v in vals])
                rows.append(dict(dataset=ds,model=model,arm=arm,N=len(y),groups=len(vals),BAcc=condition[arm].mean(),
                    Recall=recall_score(y,p),FPR=p[y==0].mean(),Precision=precision_score(y,p,zero_division=0),
                    F1=f1_score(y,p),AP=average_precision_score(y,s),TP=int((p&(y==1)).sum()),FP=int((p&(y==0)).sum()),
                    TN=int((~p&(y==0)).sum()),FN=int((~p&(y==1)).sum())))
                for name,value in vals:groups.append(dict(dataset=ds,model=model,arm=arm,condition=name,BAcc=value))
                for tag,mask in [('valid_phase',tab.kind.eq('pll').to_numpy()),('fallback',~tab.kind.eq('pll').to_numpy())]:
                    if mask.any():
                        sub=tab.loc[mask].reset_index(drop=True);v=group_bacc(sub,s[mask])
                        strata.append(dict(dataset=ds,model=model,arm=arm,stratum=tag,N=int(mask.sum()),
                                           binary_groups=len(v),BAcc=np.mean([n for _,n in v]) if v else None))
                np.savez(OUT/f'{ds}_{model}_{arm}_oof.npz',indices=keep,labels=y,scores=s)
            for a,b in [('phase8','original'),('phase8','fixed8'),('fixed8','original')]:
                v=condition[a]-condition[b];rng=np.random.default_rng(20261029)
                draws=v[rng.integers(len(v),size=(5000,len(v)))].mean(1);lo,hi=np.quantile(draws,[.025,.975])
                flips=v[None]*(rng.integers(0,2,size=(20000,len(v)))*2-1)
                p=(1+(np.abs(flips.mean(1))>=abs(v.mean())-1e-15).sum())/20001
                deltas.append(dict(dataset=ds,model=model,candidate=a,baseline=b,primary=a=='phase8',delta_pp=v.mean()*100,
                                   lo_pp=lo*100,hi_pp=hi*100,p=float(p),positive_groups=int((v>0).sum()),groups=len(v)))
    result=pd.DataFrame(rows);paired=pd.DataFrame(deltas)
    if result.empty:print('No complete dataset/model cell yet');return
    paired['p_holm']=np.nan
    if not partial:
        assert completed==90
        for is_primary in [True,False]:holm(paired,paired.index[paired.primary==is_primary])
        receipt=pd.DataFrame(receipts)
        assert receipt.groupby(['dataset','model','fold']).core_initial_hash.nunique().eq(1).all()
        # The same feature means/stds must be used across the three models.
        for ds in spec['datasets']:
            for fold in range(5):
                for arm in ['fixed8','phase8']:
                    states=[np.load(OUT/'runs'/ds/m/f'f{fold}'/arm/'preprocessing.npz') for m in spec['models']]
                    for st in states[1:]:
                        for name in ['feature_mean','feature_std']:np.testing.assert_array_equal(st[name],states[0][name])
        receipt.to_csv(OUT/'fit_identity.csv',index=False)
    prefix='stage_' if partial else ''
    result.to_csv(OUT/(prefix+'results.csv'),index=False)
    paired.to_csv(OUT/(prefix+'paired_deltas.csv'),index=False)
    pd.DataFrame(groups).to_csv(OUT/(prefix+'condition_results.csv'),index=False)
    pd.DataFrame(cost).to_csv(OUT/(prefix+'cost.csv'),index=False)
    pd.DataFrame(strata).to_csv(OUT/(prefix+'phase_validity.csv'),index=False)
    if not partial:
        old=pd.read_csv(OLD/'paired_deltas.csv');comparisons=[]
        for x in paired[paired.primary].itertuples():
            match=old[(old.dataset==x.dataset)&(old.model==x.model)&(old.candidate==x.candidate)&(old.baseline==x.baseline)]
            if len(match):comparisons.append(dict(dataset=x.dataset,model=x.model,comparison=x.candidate+'-'+x.baseline,
                e2e_delta_pp=x.delta_pp,frozen_delta_pp=float(match.delta_pp.iloc[0]),
                change_pp=x.delta_pp-float(match.delta_pp.iloc[0]),
                interpretation='cross-study descriptive difference, not isolated freezing effect'))
        pd.DataFrame(comparisons).to_csv(OUT/'frozen_comparison.csv',index=False)
        (OUT/'METRICS_AUDIT_PASS.json').write_text(json.dumps(dict(status='pass',fits=90,
            input_hashes=len(spec['input_hashes']),code_hashes=len(spec['code_hashes']),
            initialization_matched=True,neural_encoder_updates_verified=True,train_only_stats=True,
            same_eight_across_models=True,OOF_unique=True,gpu0_used=False,
            pending='separate full checkpoint forward replay audit'),indent=2))
    print(result[['dataset','model','arm','BAcc','Recall','FPR']].to_string(index=False))
    print(paired.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
