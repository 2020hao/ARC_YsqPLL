"""Audit all restored residuals and compare universal dynamics with saved C0."""

import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
import _bootstrap as boot
from arc_model.methods import dynamic_correction as method

ROOT=boot.ROOT
OUT=ROOT/'outputs/dynamic_restore_20261029'
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
ARMS=('M0','legacy_saved','legacy_replay','legacy_refit','uniform36','dynamic32')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    spec=json.loads((OUT/'protocol.json').read_text())
    for r in spec['inputs']:assert sha(r['path'])==r['sha256']
    for r in spec['code']:assert sha(ROOT/r['path'])==r['sha256']
    metrics=[];groups=[];audits=[];cost=[];decisions=[]
    for ds in spec['datasets']:
        tab=pd.read_csv(OUT/f'{ds}_metadata.csv');keep=np.load(OUT/f'{ds}_indices.npy')
        full=pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv');x=np.load(SOURCE/'data'/ds/'recovery/dynamic.npy')
        splits=json.loads((SOURCE/'data'/ds/'recovery/splits.json').read_text())
        for model in spec['models']:
            for seed in spec['seeds']:
                parts=[]
                for fold in range(5):
                    q=OUT/'runs'/ds/model/f'f{fold}_s{seed}';src=SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s{seed}'
                    done=json.loads((q/'COMPLETE.json').read_text());diag=json.loads((q/'diagnostics.json').read_text())
                    assert done['protocol_sha256']==sha(OUT/'protocol.json') and done['predictions_sha256']==sha(q/'predictions.npz')
                    assert done['gpu_physical'] in ['1','2','3']
                    z=np.load(q/'predictions.npz');ids=np.load(q/'indices.npz');tr,te=ids['train'],ids['test']
                    assert np.array_equal(tr,splits[fold]['train']) and np.array_equal(te,np.intersect1d(splits[fold]['test'],keep))
                    assert np.array_equal(te,z['indices']) and np.array_equal(z['labels'],full.label.to_numpy()[te])
                    for col in ['condition_key','source_group','record_id']:assert not set(full.iloc[tr][col])&set(full.iloc[te][col])
                    fit,cal=method.calibration_split(full.condition_key.to_numpy(),tr,fold)
                    assert np.array_equal(fit,ids['head_fit']) and np.array_equal(cal,ids['head_calibration'])
                    base=np.zeros(len(full));cross=np.load(src/'crossfit_scores.npz');base[tr]=cross['base_logit']
                    old=np.load(src/'predictions.npz');base[old['indices']]=np.log(np.clip(old['M0'],1e-7,1-1e-7)/np.clip(1-old['M0'],1e-7,1))
                    worst=0.
                    for arm in spec['arms']:
                        final=np.load(q/(arm+'_head.npz'));initial=np.load(q/(arm+'_initial.npz'))
                        assert np.array_equal(final['fit_indices'],tr) and np.array_equal(initial['fit_indices'],fit)
                        for state,allowed in [(initial,fit),(final,tr)]:
                            np.testing.assert_array_equal(state['mu'],x[allowed][:,state['keep']].mean(0))
                            np.testing.assert_array_equal(state['sd'],np.maximum(x[allowed][:,state['keep']].std(0),1e-6))
                        if arm=='uniform36':np.testing.assert_array_equal(final['keep'],np.arange(36))
                        if arm=='dynamic32':np.testing.assert_array_equal(final['keep'],np.arange(32))
                        delta,distance=method.replay(final,x);selection=json.loads((q/(arm+'_selection.json')).read_text())
                        p=expit(base[te]+method.weights(distance,tr[full.label.to_numpy()[tr]==0],selection['alpha'],selection['quantile'])[te]*delta[te])
                        error=float(abs(p-z[arm]).max());assert error<5e-5
                        worst=max(worst,error)
                        decisions.append(dict(dataset=ds,model=model,fold=fold,seed=seed,arm=arm,alpha=selection['alpha'],quantile=selection['quantile'],columns=len(final['keep'])))
                    audits.append(dict(dataset=ds,model=model,fold=fold,seed=seed,
                        head_replay_error=worst,legacy_replay_error=diag['legacy_replay_max_error'],legacy_refit_error=diag['legacy_refit_max_error'],
                        coefficient_error=diag['final_coefficient_max_error'],same_selection=diag['selected_columns_equal'],
                        same_alpha=diag['selected_alpha_equal'],same_quantile=diag['selected_quantile_equal']))
                    pieces=dict(index=te,labels=z['labels'])
                    for arm in ARMS:pieces[arm]=z[arm]
                    parts.append(pd.DataFrame(pieces));cost.append(dict(dataset=ds,model=model,fold=fold,seed=seed,seconds=done['seconds'],new_fits=done['new_fits']))
                t=pd.concat(parts).sort_values('index');assert np.array_equal(t['index'],keep)
                y=t.labels.to_numpy();assert np.array_equal(y,tab.label.to_numpy())
                for arm in ARMS:
                    s=t[arm].to_numpy();p=s>=.5;vals=[]
                    for condition,g in tab.groupby('condition_key',sort=True):
                        yy=y[g.index];pp=p[g.index];assert set(yy)=={0,1}
                        bacc=.5*(pp[yy==1].mean()+(~pp[yy==0]).mean());vals.append(bacc)
                        groups.append(dict(dataset=ds,model=model,seed=seed,arm=arm,condition=condition,BAcc=bacc))
                    metrics.append(dict(dataset=ds,model=model,seed=seed,arm=arm,N=len(y),groups=len(vals),BAcc=np.mean(vals),
                        Recall=recall_score(y,p),FPR=p[y==0].mean(),Precision=precision_score(y,p,zero_division=0),F1=f1_score(y,p),AP=average_precision_score(y,s),
                        TP=int((p&(y==1)).sum()),FP=int((p&(y==0)).sum()),TN=int((~p&(y==0)).sum()),FN=int((~p&(y==1)).sum())))
                np.savez(OUT/f'{ds}_{model}_seed{seed}_oof.npz',indices=keep,labels=y,**{a:t[a].to_numpy() for a in ARMS})
    frame=pd.DataFrame(metrics);frame.to_csv(OUT/'seed_metrics.csv',index=False)
    summary=frame.groupby(['dataset','model','arm'],sort=False).agg(BAcc=('BAcc','mean'),seed_sd=('BAcc','std'),Recall=('Recall','mean'),FPR=('FPR','mean'),Precision=('Precision','mean'),F1=('F1','mean'),AP=('AP','mean'),N=('N','first'),groups=('groups','first')).reset_index()
    summary.to_csv(OUT/'results.csv',index=False)
    group=pd.DataFrame(groups);group.to_csv(OUT/'condition_metrics.csv',index=False)
    averaged=group.groupby(['dataset','model','arm','condition'],sort=True).BAcc.mean().reset_index();paired=[]
    for ds in spec['datasets']:
        for model in spec['models']:
            t=averaged[(averaged.dataset==ds)&(averaged.model==model)].pivot(index='condition',columns='arm',values='BAcc')
            for candidate,baseline,primary in [('uniform36','M0',True),('uniform36','legacy_saved',True),('dynamic32','uniform36',False),('dynamic32','legacy_saved',False)]:
                v=(t[candidate]-t[baseline]).to_numpy();rng=np.random.default_rng(20261029)
                boot=v[rng.integers(len(v),size=(5000,len(v)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
                flip=v[None]*(rng.integers(0,2,size=(20000,len(v)))*2-1)
                p=(1+(abs(flip.mean(1))>=abs(v.mean())-1e-15).sum())/20001
                paired.append(dict(dataset=ds,model=model,candidate=candidate,baseline=baseline,primary=primary,
                    delta_pp=v.mean()*100,lo_pp=lo*100,hi_pp=hi*100,p=float(p),positive_groups=int((v>0).sum()),groups=len(v)))
    paired=pd.DataFrame(paired);paired['p_holm']=np.nan
    for primary in [True,False]:
        ids=paired.index[paired.primary==primary];ordered=paired.loc[ids,'p'].sort_values().index
        paired.loc[ordered,'p_holm']=np.maximum.accumulate([min(1.,paired.loc[ix,'p']*(len(ids)-j)) for j,ix in enumerate(ordered)])
    paired.to_csv(OUT/'paired_deltas.csv',index=False)
    audit=pd.DataFrame(audits);audit.to_csv(OUT/'audit.csv',index=False)
    pd.DataFrame(cost).to_csv(OUT/'cost.csv',index=False);pd.DataFrame(decisions).to_csv(OUT/'calibration_choices.csv',index=False)
    assert len(audit)==90 and audit.same_selection.all()
    receipt=dict(status='pass',bundles=90,new_residual_fits=540,source_hashes=len(spec['inputs']),code_hashes=len(spec['code']),
        max_head_replay_error=float(audit.head_replay_error.max()),max_legacy_replay_error=float(audit.legacy_replay_error.max()),
        max_legacy_refit_error=float(audit.legacy_refit_error.max()),all_same_selection=bool(audit.same_selection.all()),
        all_same_alpha=bool(audit.same_alpha.all()),all_same_quantile=bool(audit.same_quantile.all()),
        same_fixed36_or32_for_all_models=True,train_only_stats=True,OOF_unique=True,gpu0_used=False,
        scope='existing frozen encoders and honest OOF reused; not new independent acquisition or end-to-end backbone validation')
    (OUT/'AUDIT_PASS.json').write_text(json.dumps(receipt,indent=2))
    (OUT/'RESULTS_COMPLETE.json').write_text(json.dumps(dict(status='complete',bundles=90,new_fits=540,seeds=[0,1,2],datasets=2,models=3,rows=len(summary)),indent=2))
    print(summary[['dataset','model','arm','BAcc','seed_sd','Recall','FPR']].to_string(index=False));print(paired.to_string(index=False));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
