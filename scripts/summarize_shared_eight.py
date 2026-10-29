"""Replay all fitted heads, audit grouping, and summarize the common-eight study."""

import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/shared_eight_study_20261029'
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
ARMS=('original','refit','fixed8','phase8','relative2','position6','shift8')
PAIRS=(('phase8','original'),('phase8','refit'),('phase8','fixed8'),
       ('phase8','relative2'),('phase8','position6'),('phase8','shift8'))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def metrics(y,s):
    p=s>=.5
    return dict(Recall=recall_score(y,p,zero_division=0),FPR=float(p[y==0].mean()) if (y==0).any() else None,
                Precision=precision_score(y,p,zero_division=0),F1=f1_score(y,p,zero_division=0),
                AP=average_precision_score(y,s),TP=int((p&(y==1)).sum()),FN=int((~p&(y==1)).sum()),
                TN=int((~p&(y==0)).sum()),FP=int((p&(y==0)).sum()))


def group_values(tab,s):
    rows=[]
    for name,g in tab.groupby('condition_key',sort=True):
        y=g.label.to_numpy();p=s[g.index]>=.5
        if set(y)!={0,1}:continue
        rows.append((name,.5*(p[y==1].mean()+(~p[y==0]).mean())))
    return rows


def main():
    spec=json.loads((OUT/'protocol.json').read_text())
    for r in spec['source_inputs']:assert sha(r['path'])==r['sha256'],r['path']
    for r in spec['source_code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    results=[];paired=[];groups=[];audit=[];strata=[];cost=[]
    for ds in spec['datasets']:
        tab=pd.read_csv(OUT/f'{ds}_metadata.csv');keep=np.load(OUT/f'{ds}_indices.npy')
        full=pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv')
        for model in spec['models']:
            prediction={a:np.full(len(full),np.nan) for a in ARMS};counts=np.zeros(len(full),int)
            for fold in range(5):
                q=OUT/'runs'/ds/model/f'f{fold}'
                done=json.loads((q/'COMPLETE.json').read_text())
                assert done['protocol_sha256']==sha(OUT/'protocol.json')
                assert done['predictions_sha256']==sha(q/'predictions.npz')
                assert done['heads_sha256']==sha(q/'heads.npz')
                assert done['shared_features']==spec['features'] and done['gpu_physical'] in ['1','2','3']
                z=np.load(q/'predictions.npz');v=np.load(q/'indices.npz');state=np.load(q/'heads.npz')
                tr,te=v['train'],v['test']
                assert np.array_equal(z['indices'],te) and not np.intersect1d(tr,te).size
                sp=json.loads((SOURCE/'data'/ds/'recovery/splits.json').read_text())[fold]
                assert np.array_equal(tr,np.intersect1d(sp['train'],keep))
                assert np.array_equal(te,np.intersect1d(sp['test'],keep))
                for col in ['condition_key','source_group','record_id']:
                    assert not set(full.iloc[tr][col])&set(full.iloc[te][col])
                x={}
                for name in ['fixed','phase','shift']:
                    f=np.load(OUT/f'{ds}_{name}8.npy')
                    np.testing.assert_array_equal(f[tr].mean(0),state[name+'_mean'])
                    np.testing.assert_array_equal(np.maximum(f[tr].std(0),1e-6),state[name+'_std'])
                    x[name]=((f[te]-state[name+'_mean'])/state[name+'_std']).astype('float32')
                relative=x['phase'].copy();relative[:,2:]=0
                position=x['phase'].copy();position[:,:2]=0
                feature_arms=[np.zeros_like(x['phase']),x['fixed'],x['phase'],relative,position,x['shift']]
                h=np.load(q/'embedding.npy',mmap_mode='r')[te]
                worst=0.
                for k,(arm,f) in enumerate(zip(ARMS[1:],feature_arms)):
                    logits=np.concatenate([h,f],axis=1)@state['weight'][k].T+state['bias'][k]
                    prob=expit(logits[:,0]) if logits.shape[1]==1 else softmax(logits,axis=1)[:,1]
                    err=float(abs(prob-z[arm]).max());assert err<3e-5,(ds,model,fold,arm,err)
                    worst=max(worst,err)
                assert np.array_equal(z['labels'],full.label.to_numpy()[te])
                for a in ARMS:prediction[a][te]=z[a]
                counts[te]+=1
                audit.append(dict(dataset=ds,model=model,fold=fold,train=len(tr),test=len(te),
                                  max_head_replay_error=worst,heads_sha256=sha(q/'heads.npz'),
                                  original_replay_error=json.loads((q/'extraction.json').read_text())['max_original_probability_error']))
                cost.append(dict(dataset=ds,model=model,fold=fold,
                                 total_seconds=done['total_seconds'],head_training_seconds=done['head_training_seconds'],
                                 new_head_fits=done['new_head_fits'],extra_effective_parameters=done['additional_effective_parameters']))
            assert (counts[keep]==1).all() and counts.sum()==len(keep)
            condition={}
            for arm in ARMS:
                s=prediction[arm][keep];assert np.isfinite(s).all()
                vals=group_values(tab,s);assert len(vals)==tab.condition_key.nunique()
                condition[arm]=np.array([v for _,v in vals])
                row=dict(dataset=ds,model=model,arm=arm,N=len(tab),groups=len(vals),BAcc=condition[arm].mean(),**metrics(tab.label.to_numpy(),s))
                results.append(row)
                for name,value in vals:groups.append(dict(dataset=ds,model=model,arm=arm,condition=name,BAcc=value))
                for stratum,mask in [('valid_phase',tab.kind.eq('pll').to_numpy()),('fallback',~tab.kind.eq('pll').to_numpy())]:
                    if mask.any():
                        t=tab.loc[mask].reset_index(drop=True);ss=s[mask];vv=group_values(t,ss)
                        strata.append(dict(dataset=ds,model=model,arm=arm,stratum=stratum,N=int(mask.sum()),
                                           complete_binary_groups=len(vv),BAcc=np.mean([v for _,v in vv]) if vv else None,**metrics(t.label.to_numpy(),ss)))
            for candidate,baseline in PAIRS:
                d=condition[candidate]-condition[baseline];rng=np.random.default_rng(20261029)
                boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                flips=(rng.integers(0,2,size=(20000,len(d)))*2-1)*d[None]
                p=(1+(np.abs(flips.mean(1))>=abs(d.mean())-1e-15).sum())/20001
                paired.append(dict(dataset=ds,model=model,candidate=candidate,baseline=baseline,
                                   primary=baseline in ['original','refit','fixed8'],delta_pp=d.mean()*100,
                                   lo_pp=lo*100,hi_pp=hi*100,p=float(p),positive_groups=int((d>0).sum()),groups=len(d)))
            np.savez(OUT/f'{ds}_{model}_oof.npz',indices=keep,**{k:v[keep] for k,v in prediction.items()})
    p=pd.DataFrame(paired);p['p_holm']=np.nan
    for primary in [True,False]:
        ids=p.index[p.primary==primary];ordered=p.loc[ids,'p'].sort_values().index
        vals=np.maximum.accumulate([min(1.,p.loc[ix,'p']*(len(ids)-j)) for j,ix in enumerate(ordered)])
        p.loc[ordered,'p_holm']=vals
    pd.DataFrame(results).to_csv(OUT/'results.csv',index=False)
    p.to_csv(OUT/'paired_deltas.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'condition_results.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'phase_validity_results.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'audit.csv',index=False)
    pd.DataFrame(cost).to_csv(OUT/'cost.csv',index=False)
    (OUT/'AUDIT_PASS.json').write_text(json.dumps(dict(status='pass',frozen_encoders=30,
        new_head_fits=180,source_hashes=len(spec['source_inputs']),source_code_hashes=len(spec['source_code']),
        max_head_replay_error=max(r['max_head_replay_error'] for r in audit),
        checks=['all source and code hashes','group disjointness','same8 for all','train-only feature standardization',
                'exact OOF coverage','180 fitted head forward replays','saved original checkpoint parity','GPU0 excluded']),indent=2))
    (OUT/'RESULTS_COMPLETE.json').write_text(json.dumps(dict(status='complete',result_rows=len(results),paired_rows=len(p),
        scope='single pretrained seed; frozen encoders; exploratory cluster inference; not end-to-end confirmation'),indent=2))
    print(pd.DataFrame(results)[['dataset','model','arm','BAcc','Recall','FPR']].to_string(index=False))
    print(p.to_string(index=False))


if __name__=='__main__':main()
