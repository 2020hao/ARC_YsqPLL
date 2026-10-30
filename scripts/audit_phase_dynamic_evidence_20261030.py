"""Read-only independent audit of predefined phase/dynamic evidence experiments.

No backbone/head is trained. Feature definitions, past-only masks, partitions,
train-only calibration/statistics and serialized prediction arithmetic are
recomputed independently of the candidate's fit/replay/selection functions.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
import yaml
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'
OLD=ROOT/'outputs/locked_routes_20261030/dual'
CFG=ROOT/'configs/modules/phase_dynamic_evidence_20261030.yaml'
MODELS={'ArcFENet':'dual_arc_fenet','HYDRA':'dual_hydra','ArcFusionNet':'dual_arc_fusion'}
SOURCE_RAW=ROOT/'outputs/continuous_pll_validation_20261027_v2/records'
ARMS={
 'A1_rcr_masks4_oldfusion':('masks4','old_residual',False),
 'A2_rcr_stats32_oldfusion':('stats32','old_residual',False),
 'B1_rcr36_wholeblend':('rcr36','whole_path',False),
 'B2_rcr36_wholeblend_recall':('rcr36','whole_path',True),
 'P1_rcr40_phase':('rcr40','whole_path',True),
 'P2_fixed40_phase':('fixed40','whole_path',True),
 'P3_rcr40_shift':('shift40','whole_path',True),
 'P4_rcr40_phasefree':('free40','whole_path',True),
}
REFERENCES={'R0_parent':'baseline','R1_calibration':'calibration_only','R2_rcr36_old':'dual_rcr','R3_fixed36_old':'dual_fixed20'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)


def probabilities(energy):
    a=np.asarray(energy,np.float64)
    assert a.ndim==3 and a.shape[1:]==(2,20) and np.isfinite(a).all() and np.all(a>=0)
    total=a.sum(2,keepdims=True)
    return np.divide(a,total,out=np.zeros_like(a),where=total>0)


def position_features(p):
    out=np.empty((len(p),4),np.float32)
    for band in (0,1):
        out[:,2*band]=p[:,band,19]+p[:,band,0]
        out[:,2*band+1]=p[:,band,9]+p[:,band,10]
    return out


def phasefree_features(p):
    out=np.empty((len(p),4),np.float32)
    for band in (0,1):
        out[:,2*band]=p[:,band].max(1)
        out[:,2*band+1]=(p[:,band]**2).sum(1)
    return out


def key(row,fs):
    return (str(row.kind),int(row.segment),int(row.epoch),round(row.start_seconds*fs),round(row.end_seconds*fs))


def history_masks(ds,table,fs,dynamic,frontend):
    checked=0;past_pairs=0
    for rid,g in table.groupby('record_id',sort=False):
        ledger=pd.read_csv(SOURCE_RAW/ds/f'{rid:04}'/(frontend+'_ledger.csv'))
        lookup={};cycle=-1;last_token=None
        for i,row in enumerate(ledger.itertuples()):
            token=(str(row.kind),int(row.segment),int(row.epoch))
            cycle=cycle+1 if token==last_token else 0
            last_token=token
            lookup[key(row,fs)]=(i,cycle)
        for row in g.itertuples():
            i,cycle=lookup[key(row,fs)]
            expected=np.asarray([float(cycle>=lag) for lag in (1,2,4,8)],np.float32)
            np.testing.assert_array_equal(dynamic[row.Index,32:],expected)
            assert max(row.end_seconds,row.available_seconds)<=row.anchor_seconds+1e-9
            for lag,available in zip((1,2,4,8),expected):
                if available:
                    prior=ledger.iloc[i-lag]
                    assert prior.start_seconds<row.start_seconds
                    assert prior.end_seconds<=row.start_seconds+1e-8
                    assert (prior.kind,int(prior.segment),int(prior.epoch))==(row.kind,int(row.segment),int(row.epoch))
                    past_pairs+=1
            checked+=1
    return dict(rows=checked,strictly_past_history_pairs=past_pairs,frontend=frontend)


def feature_audit():
    cfg=yaml.safe_load(CFG.read_text());audit=json.loads((OUT/'FEATURE_AUDIT.json').read_text())
    assert len(cfg['arms'])==12 and sum('reuse' in a for a in cfg['arms'].values())==4
    assert len(ARMS)==8 and len(set(a[0] for a in ARMS.values()))==7
    assert 2*3*5*8==240 and 2*3*5*7*2==420
    assert cfg['protocol']['new_backbone_fits']==0
    for row in audit['identities']:assert digest(row['path'])==row['sha256'],row['path']
    data=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text());reports=[]
    for ds,c in data.items():
        r=ROOT/c['prepared_roots']['current_self_sync'];f=ROOT/c['prepared_roots']['fixed20'];t=pd.read_csv(r/'cycles.csv');u=pd.read_csv(f/'cycles.csv');fs=c['native_sample_rate_hz']
        for col in ['sample_id','record_id','condition_key','source_group','label','anchor_index']:np.testing.assert_array_equal(t[col],u[col])
        pr=probabilities(np.load(r/'energy.npy'));pf=probabilities(np.load(f/'energy.npy'));keep=np.load(ROOT/c['reviewed_indices']);valid=t.kind.eq('pll').to_numpy()
        shifts=[]
        for row in t.itertuples():
            token=(f'phase_dynamic_evidence_20261030|{row.record_id}|{row.kind}|{row.segment}|{row.epoch}|{round(row.start_seconds*fs)}|{round(row.end_seconds*fs)}')
            shifts.append(int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'little')%20)
        shifts=np.asarray(shifts,np.int64);np.testing.assert_array_equal(shifts,np.load(OUT/'features'/ds/'circular_shifts.npy'))
        rolled=np.stack([np.roll(row,int(shift),axis=-1) if ok else row for row,shift,ok in zip(pr,shifts,valid)])
        np.testing.assert_array_equal(np.sort(rolled,axis=-1),np.sort(pr,axis=-1))
        np.testing.assert_array_equal(rolled[~valid],pr[~valid])
        expected={'rcr_phase4':position_features(pr),'fixed_phase4':position_features(pf),'rcr_shift4':position_features(rolled),'rcr_phasefree4':phasefree_features(pr)}
        for name,x in expected.items():np.testing.assert_array_equal(x,np.load(OUT/'features'/ds/(name+'.npy')))
        np.testing.assert_allclose(phasefree_features(rolled),phasefree_features(pr),rtol=1e-6,atol=1e-7)
        rd=np.load(r/'dynamic.npy');fd=np.load(f/'dynamic.npy')
        mask_reports=[history_masks(ds,t,fs,rd,'recovery'),history_masks(ds,u,fs,fd,'fixed20')]
        for name,x in [('rcr',rd),('fixed',fd)]:
            assert x.shape==(len(t),36) and np.isfinite(x).all()
            assert np.isin(x[:,32:],[0.,1.]).all()
        reports.append(dict(dataset=ds,reviewed=int(len(keep)),valid_reference=int(valid[keep].sum()),fallback=int((~valid[keep]).sum()),new_feature_rebuild_max_error=0.,masks=mask_reports))
    zero=np.zeros((2,2,20));np.testing.assert_array_equal(position_features(probabilities(zero)),np.zeros((2,4)))
    return dict(status='pass',matrix_arms=12,reused_arms=4,new_arms=8,new_outer_arm_bundles=240,distinct_feature_set_folds=210,matrix_head_fits=420,additional_parity_head_fit=1,new_backbone_fits=0,datasets=reports,
                interpretation=['P1vsP3 destroys only appended4 positions; inherited36 remain unchanged','P1vsP2 compares complete coordinate/input-support workflows with equal information/dimension','P4 has same40-dimensional capacity but replaces4 positional values by4 phase-invariant values','Whole-path limits apply to mean condition-weighted training-internal FPR/Recall, not each-condition or outer-test guarantees','B1vsR2 changes calibration fallback and metric weighting in addition to fusion; B2vsB1 isolates recall guard'])


def load_features(ds,dc):
    r=ROOT/dc['prepared_roots']['current_self_sync'];f=ROOT/dc['prepared_roots']['fixed20'];rcr=np.load(r/'dynamic.npy');fixed=np.load(f/'dynamic.npy');d=OUT/'features'/ds
    return dict(masks4=np.ascontiguousarray(rcr[:,32:]),stats32=np.ascontiguousarray(rcr[:,:32]),rcr36=rcr,
                rcr40=np.c_[rcr,np.load(d/'rcr_phase4.npy')],fixed40=np.c_[fixed,np.load(d/'fixed_phase4.npy')],
                shift40=np.c_[rcr,np.load(d/'rcr_shift4.npy')],free40=np.c_[rcr,np.load(d/'rcr_phasefree4.npy')])


def disjoint(table,a,b):
    assert not np.intersect1d(a,b).size
    for col in ['condition_key','record_id','source_group']:assert not set(table.iloc[a][col])&set(table.iloc[b][col])


def inner_source(ds,model,fold,inner):
    if model=='ArcFusionNet':return ROOT/f'outputs/fusion_qualification_20261030/runs/inner_{ds}_f{fold}_i{inner}'
    return OLD/'inner'/ds/model/f'f{fold}_i{inner}'


def calibration(m,y,g):
    weights=np.zeros(len(y),float);unique=np.unique(g)
    for group in unique:
        for label in (0,1):
            mask=(g==group)&(y==label);assert mask.any();weights[mask]=1./(2*len(unique)*mask.sum())
    def fun(theta):
        slope=np.exp(theta[0]);z=slope*m+theta[1]
        value=weights@(np.logaddexp(0,z)-y*z)+.001*(theta@theta)
        residual=weights*(expit(z)-y)
        return float(value),np.asarray([residual@(slope*m),residual.sum()])+.002*theta
    opt=minimize(fun,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-8,8),(-20,20)],options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    assert opt.success,opt.message
    return float(np.exp(opt.x[0])),float(opt.x[1]),float(opt.fun)


def fitted_stats(state,x,y,indices,seed):
    np.testing.assert_array_equal(state['fit_indices'],indices)
    np.testing.assert_array_equal(state['keep'],np.arange(x.shape[1]))
    mu=x[indices][:,state['keep']].mean(0);sd=np.maximum(x[indices][:,state['keep']].std(0),1e-6)
    np.testing.assert_array_equal(state['mu'],mu);np.testing.assert_array_equal(state['sd'],sd)
    z=(x[:,state['keep']]-mu)/sd;normal=indices[y[indices]==0]
    pca=PCA(n_components=min(6,x.shape[1],len(normal)-1),random_state=seed).fit(z[normal])
    np.testing.assert_array_equal(state['pca_mean'],pca.mean_)
    np.testing.assert_allclose(state['pca_components'],pca.components_,rtol=0,atol=1e-6)
    return z,pca


def independent_score(state,x,raw):
    if str(state['fusion'])=='whole_path' and float(state['strength'])==0.:return raw.copy()
    z=(x[:,state['keep']]-state['mu'])/state['sd'];delta=z.astype('float32')@state['coef'];projection=z@state['pca_components'].T-state['pca_mean']@state['pca_components'].T;distance=np.sqrt(np.mean(projection**2,axis=1));h=float(state['gate_threshold']);gate=expit(-(distance-h)/(.15*h+1e-8));calibrated=float(state['calibration_slope'])*raw+float(state['calibration_intercept'])
    if str(state['fusion'])=='old_residual':return calibrated+float(state['strength'])*gate*delta
    return raw+float(state['strength'])*gate*(calibrated-raw+delta)


def verify_trace(trace,state):
    assert len(trace)==(28 if str(state['fusion'])=='old_residual' else 24)
    fusion=str(state['fusion']);recall=bool(state['recall_constraint']);best=None
    for r in trace:
        strength=float(r['strength']);q=float(r['quantile']);assert strength in ([0.,.1,.25,.5,1.,1.5,2.] if fusion=='old_residual' else [0.,.1,.25,.5,.75,1.]);assert q in [.75,.9,.95,.975]
        if fusion=='whole_path':
            allowed=(r['FPR']<=r['raw_FPR']+.0200001 and (not recall or r['Recall']>=r['raw_Recall']-.0100001));assert r['eligible']==allowed
            rank=(r['BAcc'],-r['FPR'],-strength,-q)
            if allowed and (best is None or rank>best[0]):best=(rank,(strength,q))
    if fusion=='whole_path':assert (float(state['strength']),float(state['quantile']))==best[1]


def full_audit(preflight):
    spec=json.loads((OUT/'protocol.json').read_text());cfg=yaml.safe_load(CFG.read_text())
    for row in spec['code']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
    for row in spec['identities']:assert digest(row['path'])==row['sha256'],row['path']
    assert len(spec['jobs'])==30 and len(spec['arms'])==12 and spec['head_fits']==420
    data=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text());reports=[];stats_count=0;predictions=0;calibrations=0;worst_error=0.
    for ds in cfg['datasets']:
        dc=data[ds];t=pd.read_csv(ROOT/dc['prepared_roots']['fixed20']/'cycles.csv');y=t.label.to_numpy();groups=t.condition_key.to_numpy();keep=np.load(ROOT/dc['reviewed_indices']);splits=json.loads((ROOT/dc['split_file']).read_text());xx=load_features(ds,dc)
        coverage={arm:np.zeros(len(t),np.int64) for arm in cfg['arms']}
        for model in MODELS:
            for f in range(5):
                d=OUT/'runs'/ds/model/f'f{f}';done=json.loads((d/'COMPLETE.json').read_text());old=OLD/'runs'/ds/model/f'f{f}';index=np.load(d/'indices.npz');tr=index['outer_train'];te=index['outer_test'];a=index['selection_backbone_train'];b=index['selection_fit'];c=index['selection_validation']
                assert np.array_equal(np.sort(tr),np.setdiff1d(keep,te));assert np.array_equal(tr,np.asarray(splits[f]['train']));assert np.array_equal(te,np.intersect1d(splits[f]['test'],keep));disjoint(t,tr,te)
                for l,r in [(a,b),(a,c),(b,c)]:disjoint(t,l,r)
                assert np.array_equal(np.sort(np.r_[a,b,c]),np.sort(tr))
                margins=np.zeros(len(t));cover=np.zeros(len(t),np.int64)
                for inner in range(2):
                    p=inner_source(ds,model,f,inner);source_receipt=json.loads((p/'COMPLETE.json').read_text())
                    weight=p/('model.npz' if model=='HYDRA' else 'model.pt');assert digest(weight)==source_receipt['model_sha256']
                    index_sha=source_receipt.get('index_sha256');prediction_sha=source_receipt.get('prediction_sha256')
                    if index_sha is not None:assert digest(p/'indices.npz')==index_sha
                    if prediction_sha is not None:assert digest(p/'validation_predictions.npz')==prediction_sha
                    else:assert digest(p/'validation_predictions.npz')==source_receipt['prediction_hashes']['validation_predictions.npz']
                    z=np.load(p/'validation_predictions.npz');ix=np.load(p/'indices.npz');fit=ix['train'];val=ix['validation'];disjoint(t,fit,val);assert np.array_equal(np.sort(np.r_[fit,val]),np.sort(tr));assert np.array_equal(z['indices'],val);np.testing.assert_array_equal(z['labels'],y[val]);margins[val]=z['margin'];cover[val]+=1
                    if model=='ArcFusionNet':
                        assert digest(p/'preprocessing.npz')==source_receipt['preprocessing_sha256'];np.testing.assert_array_equal(np.load(p/'preprocessing.npz')['fit_indices'],fit)
                    if inner==0:raw_b=np.zeros(len(t));raw_b[val]=z['margin'];np.testing.assert_array_equal(fit,a);assert np.array_equal(np.sort(val),np.sort(np.r_[b,c]))
                assert np.all(cover[tr]==1) and cover.sum()==len(tr)
                finalcal=json.loads((old/'calibration.json').read_text());initialcal=json.loads((old/'selection_calibration.json').read_text());np.testing.assert_array_equal(finalcal['fit_indices'],tr)
                for ids,scores,expected in [(b,raw_b,initialcal),(tr,margins,finalcal)]:
                    slope,intercept,objective=calibration(np.asarray(scores[ids],float),np.asarray(y[ids],float),groups[ids]);np.testing.assert_allclose([slope,intercept,objective],[expected['slope'],expected['intercept'],expected['objective']],rtol=0,atol=1e-10);calibrations+=1
                fitted={}
                for feature,x in xx.items():
                    for suffix,ids,seed in [('initial',b,f),('final',tr,f+31)]:
                        state=dict(np.load(d/'fits'/(feature+'_'+suffix+'.npz')));z,pca=fitted_stats(state,x,y,ids,seed);stats_count+=1
                        if suffix=='final':fitted[feature]=(state,z,pca)
                raw=np.load(old/'baseline.npz')['margin']
                for arm,(feature,fusion,recall) in ARMS.items():
                    state=dict(np.load(d/(arm+'_state.npz')));x=xx[feature];final,z,pca=fitted[feature]
                    for key in ['mu','sd','coef','pca_mean','pca_components','fit_indices','keep']:np.testing.assert_array_equal(state[key],final[key])
                    for key,expected in [('calibration_fit_indices',tr),('selection_fit_indices',b),('selection_validation_indices',c),('selection_backbone_fit_indices',a)]:np.testing.assert_array_equal(state[key],expected)
                    np.testing.assert_allclose([state['calibration_slope'],state['calibration_intercept']],[finalcal['slope'],finalcal['intercept']],rtol=0,atol=0)
                    assert str(state['fusion'])==fusion and bool(state['recall_constraint'])==recall
                    distances=np.sqrt((pca.transform(z)**2).mean(1));normal=tr[y[tr]==0];expected_h=max(float(np.quantile(distances[normal],float(state['quantile']))),1e-6);np.testing.assert_allclose(float(state['gate_threshold']),expected_h,rtol=0,atol=1e-5)
                    trace=json.loads((d/(arm+'_selection.json')).read_text());verify_trace(trace,state)
                    p=np.load(d/(arm+'.npz'));np.testing.assert_array_equal(p['indices'],te);np.testing.assert_array_equal(p['y'],y[te]);scores=independent_score(state,x[te],raw);err=float(abs(scores-p['margin']).max());assert err==0.,(ds,model,f,arm,err);np.testing.assert_array_equal(expit(scores),p['probability']);worst_error=max(worst_error,err);predictions+=len(te);coverage[arm][te]+=1
                    # Execute the public CPU state API only; no GPU or fitting.
                    if f==0:
                        module=importlib.import_module('arc_model.implementations.'+MODELS[model]+'.phase_dynamic_v2');cls=module.Correction;obj=cls(state,baseline_id=str(state['baseline_id']),score_kind=str(state['score_kind']));np.testing.assert_array_equal(obj.score(raw,x[te],enabled=False),raw)
                        if fusion=='whole_path':
                            zero=dict(state);zero['strength']=np.array(0.);obj=cls(zero,baseline_id=str(state['baseline_id']),score_kind=str(state['score_kind']));np.testing.assert_array_equal(obj.score(raw,x[te]),raw)
                for arm,origin in REFERENCES.items():
                    source=np.load(old/(origin+'.npz'));p=np.load(d/(arm+'.npz'));np.testing.assert_array_equal(source['indices'],te);np.testing.assert_array_equal(source['margin'],p['margin']);coverage[arm][te]+=1
                for path,h in done['prediction_sha256'].items():assert digest(d/path)==h
                assert done['head_fits']==14 and done['gpu0_used'] is False
                reports.append(dict(dataset=ds,model=model,fold=f,stats_checked=14,calibrations_checked=2,new_arms_checked=8,group_isolation=True,honest_OOF_unique_rows=len(tr),max_prediction_error=0.))
                print('PHASE_INDEPENDENT_JOB_PASS',ds,model,f,flush=True)
            # coverage is summed across models; by the end each model has one
            # prediction per reviewed cycle, and none outside the reviewed cohort.
        for arm,count in coverage.items():assert np.all(count[keep]==3) and count.sum()==3*len(keep),(ds,arm)
    assert stats_count==420 and calibrations==60 and len(reports)==30
    result=dict(status='pass',preflight=preflight,outer_jobs=30,new_outer_arm_bundles=240,inner_backbone_weight_prediction_identities_verified=60,linear_head_fit_stats_verified=stats_count,additional_matrix_head_refits=0,calibration_reoptimizations=calibrations,new_arm_heldout_rows_replayed=predictions,max_prediction_error=worst_error,OOF_and_train_only_state_checks=reports,training_or_GPU_use_in_this_audit=False,new_backbone_fits=0,scope='seed0 exposed-cohort exploratory study; CPU inference/statistical checks only',statistical_significance_review='not performed here; primary30 Holm family must be reported by the main summarizer')
    write(OUT/'independent_audit.json',result)
    print('PHASE_INDEPENDENT_AUDIT_PASS',stats_count,calibrations,predictions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Read-only audit is CPU-only; set CUDA_VISIBLE_DEVICES empty')
    p=feature_audit();write(OUT/'independent_preflight.json',p)
    if args.preflight:print(json.dumps(p,ensure_ascii=False,indent=2))
    else:full_audit(p)
