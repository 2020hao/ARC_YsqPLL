"""Locked-parent, honest grouped-OOF dual-route screen (seed0, both datasets).

The frozen parent networks stay unchanged. Three candidate packages are exercised
with independent model code, calibration-only and fixed20 same-information controls.
No label/feature tuning uses outer test observations.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
import _bootstrap as boot
from qualify_fusion_current_20261030 import fit_calibration

ROOT=boot.ROOT
OUT=ROOT/'outputs/locked_routes_20261030/dual'
DATASETS=('continuous5m','zenodo5501527')
MODELS={'ArcFENet':('dual_arc_fenet','ArcFENet_reconstructed'),
        'HYDRA':('dual_hydra','HYDRA_paper'),
        'ArcFusionNet':('dual_arc_fusion','ArcFusionNet_current')}
CACHE={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)


def package(model,component):
    return importlib.import_module('arc_model.implementations.'+MODELS[model][0]+'.'+component)


def dataset(ds):
    if ds not in CACHE:
        config=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[ds]
        base=ROOT/config['prepared_roots']['fixed20'];rcr=ROOT/config['prepared_roots']['current_self_sync']
        table=pd.read_csv(base/'cycles.csv');other=pd.read_csv(rcr/'cycles.csv')
        for col in ['sample_id','record_id','label','condition_key','anchor_index']:
            assert np.array_equal(table[col],other[col]),col
        keep=np.load(ROOT/config['reviewed_indices']);splits=json.loads((ROOT/config['split_file']).read_text())
        for fold,s in enumerate(splits):
            train=np.asarray(s['train']);test=np.intersect1d(s['test'],keep)
            assert np.array_equal(np.sort(train),np.setdiff1d(keep,test))
            for k in ['record_id','source_group','condition_key']:
                assert not set(table.iloc[train][k]) & set(table.iloc[test][k]),(fold,k)
        CACHE[ds]=dict(config=config,base=base,rcr=rcr,table=table,keep=keep,splits=splits,
                       y=table.label.to_numpy(),groups=table.condition_key.to_numpy())
    return CACHE[ds]


def entries(ds,model,fold):
    rows=json.loads((ROOT/'outputs/grouped_baseline_standard_20261029/baseline_registry.json').read_text())['entries']
    return next(e for e in rows if e['dataset']==ds and e['model']==MODELS[model][1] and e['seed']==0 and e['fold']==fold)


def inner_path(ds,model,fold,inner):
    if model=='ArcFusionNet':
        return ROOT/f'outputs/fusion_qualification_20261030/runs/inner_{ds}_f{fold}_i{inner}'
    return OUT/'inner'/ds/model/f'f{fold}_i{inner}'


def inner_indices(ds,fold,inner):
    p=ROOT/f'outputs/fusion_qualification_20261030/data/{ds}/outer{fold}_inner{inner}.npz'
    z=np.load(p)
    return z['train'],z['validation']


def prepare():
    from check_baseline_release import verify
    verify()
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    jobs=[];identities=[];cohorts=[]
    for ds in DATASETS:
        d=dataset(ds);t=d['table'];cohorts.append(dict(dataset=ds,N=len(d['keep']),groups=int(t.iloc[d['keep']].condition_key.nunique())))
        for f in range(5):
            outer_train=np.asarray(d['splits'][f]['train']);seen=[]
            for i in range(2):
                tr,va=inner_indices(ds,f,i);assert not np.intersect1d(tr,va).size
                assert np.array_equal(np.sort(np.r_[tr,va]),np.sort(outer_train))
                for field in ['condition_key','record_id','source_group']:assert not set(t.iloc[tr][field])&set(t.iloc[va][field])
                seen.extend(va)
                for model in MODELS:
                    if model!='ArcFusionNet':jobs.append(dict(dataset=ds,model=model,fold=f,inner=i))
                    else:
                        p=inner_path(ds,model,f,i);receipt=json.loads((p/'COMPLETE.json').read_text())
                        assert receipt['model_sha256']==sha(p/'model.pt')
                        assert receipt['preprocessing_sha256']==sha(p/'preprocessing.npz')
                        identities.append(dict(path=str(p/'COMPLETE.json'),sha256=sha(p/'COMPLETE.json')))
            assert np.array_equal(np.sort(seen),np.sort(outer_train))
        for p in [ROOT/d['config']['reviewed_indices'],ROOT/d['config']['split_file'],d['base']/'cycles.csv',d['base']/'dynamic.npy',d['rcr']/'dynamic.npy']:
            identities.append(dict(path=str(p),sha256=sha(p)))
    code=[Path(__file__)]
    for model in MODELS:
        code.extend(sorted((ROOT/'src/arc_model/implementations'/MODELS[model][0]).glob('*.py')))
    save(OUT/'protocol.json',dict(status='frozen_before_fits',cohorts=cohorts,jobs=jobs,seed=0,outer_folds=5,
        arms=['baseline','calibration_only','dual_fixed20','dual_rcr'],
        question='Extra causal history versus phase-coordinate organization on frozen grouped parents',
        feature_definition='same36 dimensions:2native-HF-bands x4lags x4statistics +4causal availability masks; no supervised feature selection',
        tuning='inner0 backbone A; its unseen D grouped into B(head/calibration fit) and C(strength selection); refit from honest2fold outertrainOOF',
        calibration='positive affine margin, condition/class-balanced logistic objective+.001L2, separate matched calibration-only control',
        scope='single-seed exploratory screen on previously exposed grouped data; does not replace3seed sealed baseline',
        physical_gpu=[1],never_gpu0=True,reused_fusion_inner_fits=20,new_fenet_inner_fits=20,new_hydra_inner_fits=20,
        identities=identities,code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in code]))
    print('FROZEN',len(jobs),'new inner fits',flush=True)


def check():
    spec=json.loads((OUT/'protocol.json').read_text())
    for r in spec['code']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    return spec


def gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only physical GPU1 assigned to this runner')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True


def infer_neural(model,net,x,s,ids):
    margins=[];prob=[]
    with torch.no_grad():
        for ix in torch.tensor(ids,device='cuda').split(128 if model=='ArcFENet' else 64):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model=='ArcFusionNet'):
                z=net(x[ix],None if s is None else s[ix])
            z=z.float();margins.append((z[:,1]-z[:,0]).cpu().numpy());prob.append(z.softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(margins),np.concatenate(prob)


def train_inner(job):
    ds,model,f,i=job['dataset'],job['model'],job['fold'],job['inner']
    q=inner_path(ds,model,f,i);q.mkdir(parents=True,exist_ok=False)
    d=dataset(ds);tr,va=inner_indices(ds,f,i);seed=720000+f*10+i;start=time.perf_counter()
    np.savez(q/'indices.npz',train=tr,validation=va)
    if model=='HYDRA':
        x=np.load(ROOT/d['config']['author_hydra_features'],mmap_mode='r')
        net=package(model,'network').Model();diagnostic=net.fit(x,d['y'],tr,lambda_l1=.001)
        margin=net.decision_function(x[va]);prob=expit(margin)
        np.savez(q/'model.npz',**net.state);save(q/'solver.json',diagnostic)
        model_path=q/'model.npz'
    else:
        host=np.load(d['base']/'rgb.npy',mmap_mode='r')
        for a in range(0,len(host),128):assert np.isfinite(host[a:a+128]).all()
        x=torch.tensor(np.array(host),device='cuda');labels=torch.tensor(d['y'],device='cuda')
        torch.manual_seed(seed);net=package(model,'network').Model(extra=0).cuda()
        optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5)
        generator=torch.Generator().manual_seed(seed+10000);tri=torch.tensor(tr,device='cuda');history=[]
        for epoch in range(20):
            net.train();order=tri[torch.randperm(len(tr),generator=generator).to('cuda')];total=torch.zeros((),device='cuda')
            for ix in order.split(128):
                optimizer.zero_grad(set_to_none=True);z=net(x[ix]);loss=torch.nn.functional.cross_entropy(z,labels[ix]);loss.backward();optimizer.step();total+=loss.detach()*len(ix)
            value=float(total)/len(tr);assert np.isfinite(value);history.append(value)
            save(q/'progress.json',dict(epoch=epoch+1,epochs=20,seconds=time.perf_counter()-start,job=job,gpu=1))
        net.eval();margin,prob=infer_neural(model,net,x,None,va);torch.save(net.state_dict(),q/'model.pt');save(q/'history.json',history);model_path=q/'model.pt'
        del optimizer,x,labels,tri;torch.cuda.empty_cache()
    np.savez(q/'validation_predictions.npz',indices=va,labels=d['y'][va],margin=margin,probability=prob)
    save(q/'COMPLETE.json',dict(status='complete',job=job,runseed=seed,seconds=time.perf_counter()-start,
        model_sha256=sha(model_path),index_sha256=sha(q/'indices.npz'),prediction_sha256=sha(q/'validation_predictions.npz'),
        protocol_sha256=sha(OUT/'protocol.json'),gpu=1 if model=='ArcFENet' else None))
    print('INNER_DONE',job,round(time.perf_counter()-start,2),flush=True)
    del net;gc.collect()
    if model!='HYDRA':torch.cuda.empty_cache()


def inner_worker(model):
    torch.set_num_threads(2)
    if model=='ArcFENet':gpu()
    spec=check()
    for j in spec['jobs']:
        if j['model']!=model:continue
        q=inner_path(j['dataset'],model,j['fold'],j['inner'])
        if (q/'COMPLETE.json').exists():continue
        if q.exists():raise RuntimeError('Incomplete existing fit needs review: '+str(q))
        try:train_inner(j)
        except Exception as e:
            save(q/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


def oof(ds,model,fold):
    d=dataset(ds);tr=np.asarray(d['splits'][fold]['train']);margin=np.zeros(len(d['table']),float);cover=np.zeros(len(d['table']),int);receipts=[]
    for inner in range(2):
        p=inner_path(ds,model,fold,inner);r=json.loads((p/'COMPLETE.json').read_text());z=np.load(p/'validation_predictions.npz');ix=np.load(p/'indices.npz')
        train,va=ix['train'],ix['validation'];assert np.array_equal(z['indices'],va)
        if model=='ArcFusionNet':assert r['model_sha256']==sha(p/'model.pt')
        else:assert r['prediction_sha256']==sha(p/'validation_predictions.npz')
        assert not set(d['groups'][train])&set(d['groups'][va])
        cover[va]+=1;margin[va]=z['margin'];receipts.append(dict(train=train,validation=va,path=str(p)))
    assert (cover[tr]==1).all() and cover.sum()==len(tr)
    return margin,receipts


def outer(ds,model,fold):
    d=dataset(ds);e=entries(ds,model,fold);q=Path(e['folder']);p=np.load(q/'predictions.npz');ids=p['indices'];ix=np.load(q/'indices.npz')
    assert np.array_equal(np.sort(ix['train']),np.sort(d['splits'][fold]['train']))
    assert sha(e['weights_path'])==e['weights_sha256']
    if model=='HYDRA':
        net=package(model,'model').Model.load(e['weights_path']);host=np.load(ROOT/d['config']['author_hydra_features'],mmap_mode='r');features=np.asarray(host[ids]);margin=net.decision_function(features);prob=expit(margin);error=float(np.max(abs(margin-p['score'])))
        data=(features,None,None)
    elif model=='ArcFusionNet':
        state=np.load(q/'preprocessing.npz');temporal=np.load(d['base']/'temporal.npy');spectral=np.load(d['base']/'spectrum.npy')
        # The frozen BF16 output depends on its original full test batching order.
        done=json.loads((q/'COMPLETE.json').read_text());full=ids
        if done.get('reuse'):
            old=ROOT/f'outputs/rcr_three_backbones_20261028/runs/{ds}/fixed20/ArcFusionNet/f{fold}_s0/outer/M0/predictions.npz';full=np.load(old)['indices']
        x=torch.tensor((temporal-state['temporal_mean'])/state['temporal_std'],device='cuda');sp=spectral[:,state['harmonic_indices']];s=torch.tensor(((sp-state['spectral_mean'])/state['spectral_std'])[:,None],device='cuda')
        net=package(model,'model').Model(extra=0).cuda();net.load_state_dict(torch.load(e['weights_path'],map_location='cpu',weights_only=True));net.eval();m,pp=infer_neural(model,net,x,s,full);lookup={int(v):i for i,v in enumerate(full)};pick=np.array([lookup[int(v)] for v in ids]);margin,prob=m[pick],pp[pick];error=float(abs(prob-p['probability']).max());data=(x,s,full)
    else:
        host=np.load(d['base']/'rgb.npy',mmap_mode='r');x=torch.tensor(np.array(host),device='cuda');net=package(model,'model').Model(extra=0).cuda();net.load_state_dict(torch.load(e['weights_path'],map_location='cpu',weights_only=True));net.eval();margin,prob=infer_neural(model,net,x,None,ids);error=float(abs(prob-p['score']).max());data=(x,None,ids)
    assert error<5e-5,(ds,model,fold,error)
    raw=p['score'] if model!='ArcFusionNet' else p['probability'];threshold=0 if model=='HYDRA' else .5
    assert np.array_equal(margin>=0,raw>=threshold)
    return e,ids,margin,prob,net,data,error


def selection(d,receipts,fold):
    a=receipts[0]['train'];heldout=receipts[0]['validation'];g=d['groups'];unique=sorted(set(g[heldout]));assert len(unique)>=3
    # Stable group split: no class/score information and no outer-test groups.
    ranked=sorted(unique,key=lambda key:hashlib.sha256(f'locked-dual|{fold}|{key}'.encode()).hexdigest())
    selected=set(ranked[::3]);c=heldout[np.isin(g[heldout],list(selected))];b=heldout[~np.isin(g[heldout],list(selected))]
    return a,b,c


def write_prediction(path,ids,y,margin,fold):
    np.savez(path,indices=ids,y=y[ids],margin=margin,probability=expit(margin),fold=np.full(len(ids),fold))


def heads(model):
    torch.set_num_threads(2)
    if model!='HYDRA':gpu()
    check()
    for ds in DATASETS:
        d=dataset(ds);y,g=d['y'],d['groups'];xx={key:np.load((d['base'] if key=='dual_fixed20' else d['rcr'])/'dynamic.npy') for key in ['dual_fixed20','dual_rcr']}
        for fold in range(5):
            dest=OUT/'runs'/ds/model/f'f{fold}'
            if (dest/'COMPLETE.json').exists():continue
            dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter();tr=np.asarray(d['splits'][fold]['train']);z,receipts=oof(ds,model,fold)
            a,b,c=selection(d,receipts,fold)
            heldout_pred=np.load(inner_path(ds,model,fold,0)/'validation_predictions.npz');selection_raw=np.zeros(len(y));selection_raw[heldout_pred['indices']]=heldout_pred['margin']
            selection_cal=fit_calibration(selection_raw[b],y[b],g[b]);final_cal=fit_calibration(z[tr],y[tr],g[tr]);final_cal['fit_indices']=tr
            selection_offset=selection_cal['slope']*selection_raw+selection_cal['intercept'];offset=final_cal['slope']*z+final_cal['intercept']
            e,te,base,prob,net,data,error=outer(ds,model,fold)
            calibrated=final_cal['slope']*base+final_cal['intercept']
            write_prediction(dest/'baseline.npz',te,y,base,fold);write_prediction(dest/'calibration_only.npz',te,y,calibrated,fold)
            np.savez(dest/'training_indices.npz',outer_train=tr,outer_test=te,selection_base_train=a,selection_head_fit=b,selection_validation=c)
            save(dest/'calibration.json',{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in final_cal.items()});save(dest/'selection_calibration.json',selection_cal)
            replay=[]
            for arm,x in xx.items():
                state,trace=package(model,'training').fit_from_nested_oof(x,y,offset,g,tr,receipts,
                    selection_offset=selection_offset,selection_fit=b,selection_validation=c,selection_base_train=a,
                    baseline_id=e['baseline_id'],score_kind='calibrated_svm_logit' if model=='HYDRA' else 'logit_raw',seed=fold,device='cpu')
                calibration=dict(slope=final_cal['slope'],intercept=final_cal['intercept'],fit_indices=tr)
                net.attach_correction(state,calibration=calibration)
                predicted=calibrated+net.correction.delta(x[te])
                # Replay full candidate Model (including exact off path), not only correction arithmetic.
                if model=='HYDRA':
                    full=net.decision_function(data[0],dynamic=x[te],enabled=True);off=net.decision_function(data[0],enabled=False)
                else:
                    full_ids=data[2];pieces=[];offpieces=[]
                    with torch.no_grad():
                        for ix in torch.tensor(full_ids,device='cuda').split(128 if model=='ArcFENet' else 64):
                            dyn=x[ix.cpu().numpy()]
                            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model=='ArcFusionNet'):
                                enh=net(data[0][ix],None if data[1] is None else data[1][ix],dynamic=dyn,enabled=True)
                                original=net(data[0][ix],None if data[1] is None else data[1][ix],enabled=False)
                            pieces.append((enh[:,1].float()-enh[:,0].float()).cpu().numpy());offpieces.append((original[:,1].float()-original[:,0].float()).cpu().numpy())
                    full=np.concatenate(pieces);off=np.concatenate(offpieces)
                    if len(full_ids)!=len(te):
                        look={int(v):i for i,v in enumerate(full_ids)};pick=np.array([look[int(v)] for v in te]);full,off=full[pick],off[pick]
                replay_error=float(abs(full-predicted).max());off_error=float(abs(off-base).max())
                assert replay_error<2e-4 and off_error==0,(replay_error,off_error)
                assert np.array_equal(full>=0,predicted>=0)
                write_prediction(dest/(arm+'.npz'),te,y,full,fold);net.save_correction(dest/(arm+'_state.npz'));save(dest/(arm+'_selection.json'),trace)
                replay.append(dict(arm=arm,alpha=float(state['alpha']),replay_margin_error=replay_error,off_margin_error=off_error,decision_changes=0))
            save(dest/'COMPLETE.json',dict(status='complete',model=model,dataset=ds,fold=fold,seed=0,parent_baseline_id=e['baseline_id'],parent_weights_sha256=e['weights_sha256'],parent_prediction_error=error,
                protocol_sha256=sha(OUT/'protocol.json'),replay=replay,seconds=time.perf_counter()-start,
                prediction_sha256={p.name:sha(p) for p in dest.glob('*.npz')},honest_inner_coverage=len(tr),selection_groups=dict(base=len(set(g[a])),head_fit=len(set(g[b])),head_validation=len(set(g[c])))))
            print('HEAD_DONE',ds,model,fold,[(r['arm'],r['alpha']) for r in replay],round(time.perf_counter()-start,1),flush=True)
            del net,data;gc.collect()
            if model!='HYDRA':torch.cuda.empty_cache()


def run_study(suite,execute=False):
    if set(suite['datasets'])!={'new5m','zenodo5501527'}:raise ValueError('Both datasets separately')
    if execute:raise ValueError('Use documented --prepare, --inner and --heads stages for resource assignment')
    print('Locked-parent dual screen: calibration-only, fixed20 dynamics, original RCR dynamics; seed0,5fold')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--inner',choices=['ArcFENet','HYDRA']);p.add_argument('--heads',choices=list(MODELS));args=p.parse_args()
    if args.prepare:prepare()
    elif args.inner:inner_worker(args.inner)
    elif args.heads:heads(args.heads)
