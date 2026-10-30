"""Predefined feature/fusion/phase evidence screen; no new backbone training.

Three independent candidate implementations, both datasets, unchanged outer/inner
partitions, seed0. All head fitting/strength selection uses honest inner data.
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
from scipy.special import expit
import yaml
import _bootstrap as boot
from validate_locked_dual_20261030 import dataset,entries,oof,MODELS,DATASETS

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'
OLD=ROOT/'outputs/locked_routes_20261030/dual'
CONFIG=ROOT/'configs/modules/phase_dynamic_evidence_20261030.yaml'
NEW_ARMS={
 'A1_rcr_masks4_oldfusion':('masks4','old_residual',False),
 'A2_rcr_stats32_oldfusion':('stats32','old_residual',False),
 'B1_rcr36_wholeblend':('rcr36','whole_path',False),
 'B2_rcr36_wholeblend_recall':('rcr36','whole_path',True),
 'P1_rcr40_phase':('rcr40','whole_path',True),
 'P2_fixed40_phase':('fixed40','whole_path',True),
 'P3_rcr40_shift':('shift40','whole_path',True),
 'P4_rcr40_phasefree':('free40','whole_path',True),
}
REFERENCE_ARMS={'R0_parent':'baseline','R1_calibration':'calibration_only','R2_rcr36_old':'dual_rcr','R3_fixed36_old':'dual_fixed20'}
PHASE_NAMES={'rcr40':'rcr_phase4.npy','fixed40':'fixed_phase4.npy','shift40':'rcr_shift4.npy','free40':'rcr_phasefree4.npy'}
CACHE={}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)


def method(model):return importlib.import_module('arc_model.implementations.'+MODELS[model][0]+'.phase_dynamic_v2')


def features(ds):
 if ds not in CACHE:
  d=dataset(ds);rcr=np.load(d['rcr']/'dynamic.npy');fixed=np.load(d['base']/'dynamic.npy')
  new={k:np.load(OUT/'features'/ds/name) for k,name in PHASE_NAMES.items()}
  for k,v in new.items():assert v.shape==(len(d['table']),4) and v.dtype==np.float32 and np.isfinite(v).all(),k
  CACHE[ds]=dict(masks4=np.ascontiguousarray(rcr[:,32:]),stats32=np.ascontiguousarray(rcr[:,:32]),rcr36=rcr,
   rcr40=np.concatenate([rcr,new['rcr40']],1),fixed40=np.concatenate([fixed,new['fixed40']],1),
   shift40=np.concatenate([rcr,new['shift40']],1),free40=np.concatenate([rcr,new['free40']],1))
 return CACHE[ds]


def prepare():
 from check_baseline_release import verify
 verify()
 if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
 if not CONFIG.exists():raise FileNotFoundError(CONFIG)
 identities=[dict(path=str(CONFIG),sha256=sha(CONFIG)),dict(path=str(OLD/'COMPLETE.json'),sha256=sha(OLD/'COMPLETE.json'))]
 jobs=[]
 for ds in DATASETS:
  d=dataset(ds);xx=features(ds)
  assert np.array_equal(xx['rcr40'][:,:36],xx['rcr36']) and np.array_equal(xx['shift40'][:,:36],xx['rcr36'])
  for name in PHASE_NAMES.values():
   p=OUT/'features'/ds/name;identities.append(dict(path=str(p),sha256=sha(p)))
  for model in MODELS:
   for fold in range(5):
    p=OLD/'runs'/ds/model/f'f{fold}'
    for name in ['COMPLETE.json','training_indices.npz','calibration.json','selection_calibration.json']:
     identities.append(dict(path=str(p/name),sha256=sha(p/name)))
    jobs.append(dict(dataset=ds,model=model,fold=fold))
 code=[Path(__file__)] + [ROOT/'src/arc_model/implementations'/MODELS[m][0]/'phase_dynamic_v2.py' for m in MODELS]
 save(OUT/'protocol.json',dict(status='frozen_before_new_head_fits',jobs=jobs,arms=list(REFERENCE_ARMS)+list(NEW_ARMS),new_arms=NEW_ARMS,
  feature_sets={k:int(v.shape[1]) for k,v in features(DATASETS[0]).items()},head_fits=420,new_backbone_fits=0,
  seed=0,outer_folds=5,physical_gpu_allowed=[1,2,3],training_device='cpu',gpu0_used=False,
  fitting='unchanged CPU80epoch AdamW lr.002/wd.002/BS256/L1mean.0002,zero-init no-bias linear; normal-only PCA<=6',
  selection='old independent A/B/C group split; same B-only calibration and outertrainOOF calibration; no outer test choice',
  bounded_fusion='z + lambda*q*((a-1)*z+b+r); lambda0 returns raw parent exactly, including no calibration',
  primary='P1_rcr40_phase predetermined; never selected from outer test scores',
  exploratory=True,identities=identities,code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in code]))
 print('FROZEN',len(jobs),'outer jobs,420 head fits,240 new arm outputs',flush=True)


def check():
 p=json.loads((OUT/'protocol.json').read_text())
 for row in p['code']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 for row in p['identities']:assert sha(Path(row['path']))==row['sha256'],row['path']
 return p


def old_context(ds,model,fold):
 d=dataset(ds);p=OLD/'runs'/ds/model/f'f{fold}';idx=np.load(p/'training_indices.npz')
 tr,te=idx['outer_train'],idx['outer_test'];a,b,c=idx['selection_base_train'],idx['selection_head_fit'],idx['selection_validation'];g=d['groups']
 for u,v in [(tr,te),(a,b),(a,c),(b,c)]:assert not np.intersect1d(u,v).size and not set(g[u])&set(g[v])
 z,receipts=oof(ds,model,fold);cal=json.loads((p/'calibration.json').read_text());scal=json.loads((p/'selection_calibration.json').read_text())
 assert np.array_equal(cal['fit_indices'],tr) and cal['slope']>0 and scal['slope']>0
 first=Path(receipts[0]['path']);v=np.load(first/'validation_predictions.npz');raw_select=np.zeros(len(d['table']));raw_select[v['indices']]=v['margin']
 assert np.array_equal(np.sort(v['indices']),np.sort(np.r_[b,c])) and np.array_equal(receipts[0]['train'],a)
 selection_offset=scal['slope']*raw_select+scal['intercept'];offset=cal['slope']*z+cal['intercept']
 base=np.load(p/'baseline.npz');assert np.array_equal(base['indices'],te)
 return d,p,tr,te,a,b,c,z,raw_select,offset,selection_offset,cal,scal,base['margin']


def parity():
 torch.set_num_threads(2)
 ds,model,fold='continuous5m','ArcFENet',0
 d,p,tr,te,a,b,c,z,raw_select,offset,selection_offset,cal,scal,base=old_context(ds,model,fold)
 x=np.load(d['rcr']/'dynamic.npy');m=method(model)
 delta,distance,state=m.fit(x,d['y'],offset,tr,np.arange(36),fold+31,device='cpu')
 old=dict(np.load(p/'dual_rcr_state.npz'));errors={}
 for key in ['keep','mu','sd','coef','pca_components','pca_mean','fit_indices']:
  error=float(np.max(abs(state[key]-old[key])));errors[key]=error;assert error==0,(key,error)
 old_delta,old_distance=importlib.import_module('arc_model.implementations.dual_arc_fenet.dynamic_correction').replay(old,x)
 delta_error=float(abs(delta-old_delta).max());distance_error=float(abs(distance-old_distance).max());assert delta_error<1e-5 and distance_error<1e-5
 save(OUT/'parity.json',dict(status='pass',dataset=ds,model=model,fold=fold,state_errors=errors,delta_replay_error=delta_error,distance_replay_error=distance_error,
  note='One full36D final head refit; exact learned parameters/PCA/statistics, all-row replay float reduction tolerance1e-5'))
 print('PARITY_PASS',errors,delta_error,distance_error,flush=True)


def prediction(path,ids,y,margin,fold):
 np.savez(path,indices=ids,y=y[ids],margin=margin,probability=expit(margin),fold=np.full(len(ids),fold))


def one(job):
 ds,model,fold=job['dataset'],job['model'],job['fold'];dest=OUT/'runs'/ds/model/f'f{fold}';dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 d,p,tr,te,a,b,c,z,raw_select,offset,selection_offset,cal,scal,base=old_context(ds,model,fold)
 g,y=d['groups'],d['y'];xx=features(ds);m=method(model);entry=entries(ds,model,fold);fits={};fit_audit=[]
 np.savez(dest/'indices.npz',outer_train=tr,outer_test=te,selection_backbone_train=a,selection_fit=b,selection_validation=c)
 for key,x in xx.items():
  initial_delta,initial_dist,initial=m.fit(x,y,selection_offset,b,np.arange(x.shape[1]),fold,device='cpu')
  final_delta,final_dist,final=m.fit(x,y,offset,tr,np.arange(x.shape[1]),fold+31,device='cpu')
  folder=dest/'fits';folder.mkdir(exist_ok=True);np.savez(folder/(key+'_initial.npz'),**initial);np.savez(folder/(key+'_final.npz'),**final)
  fits[key]=(initial_delta,initial_dist,final_dist,final)
  fit_audit.append(dict(feature=key,dimension=x.shape[1],selection_fit_rows=len(b),final_fit_rows=len(tr),initial_sha256=sha(folder/(key+'_initial.npz')),final_sha256=sha(folder/(key+'_final.npz'))))
 for arm,(key,fusion,recall) in NEW_ARMS.items():
  x=xx[key];initial_delta,initial_dist,final_dist,final=fits[key]
  (strength,quantile),trace=m.select_strength(y,g,raw_select,selection_offset,initial_delta,initial_dist,b,c,fusion=fusion,recall_constraint=recall)
  normal=tr[y[tr]==0];threshold=max(float(np.quantile(final_dist[normal],quantile)),1e-6)
  state={k:np.asarray(v).copy() for k,v in final.items()};state.update(dimension=np.asarray(x.shape[1]),baseline_id=np.asarray(entry['baseline_id']),score_kind=np.asarray('calibrated_svm_logit' if model=='HYDRA' else 'logit_raw'),
   calibration_slope=np.asarray(cal['slope']),calibration_intercept=np.asarray(cal['intercept']),calibration_fit_indices=tr,
   gate_threshold=np.asarray(threshold),strength=np.asarray(strength),quantile=np.asarray(quantile),fusion=np.asarray(fusion),recall_constraint=np.asarray(recall),
   selection_fit_indices=b,selection_validation_indices=c,selection_backbone_fit_indices=a,feature=np.asarray(key),arm=np.asarray(arm),
   protocol_sha256=np.asarray(sha(OUT/'protocol.json')),source_hash=np.asarray(sha(p/'COMPLETE.json')))
  correction=m.Correction(state,baseline_id=entry['baseline_id'],score_kind=str(state['score_kind']));margin=correction.score(base,x[te]);off=correction.score(base,x[te],enabled=False);assert np.array_equal(off,base)
  if fusion=='whole_path':
   zero=dict(state);zero['strength']=np.asarray(0.);identity=m.Correction(zero,baseline_id=entry['baseline_id'],score_kind=str(state['score_kind'])).score(base,x[te]);assert np.array_equal(identity,base)
  correction.save(dest/(arm+'_state.npz'));saved=m.Correction(dict(np.load(dest/(arm+'_state.npz'))),baseline_id=entry['baseline_id'],score_kind=str(state['score_kind']))
  replay=saved.score(base,x[te]);assert np.array_equal(margin,replay)
  prediction(dest/(arm+'.npz'),te,y,margin,fold);save(dest/(arm+'_selection.json'),trace)
 for arm,old_arm in REFERENCE_ARMS.items():
  old=np.load(p/(old_arm+'.npz'));assert np.array_equal(old['indices'],te);prediction(dest/(arm+'.npz'),te,y,old['margin'],fold)
 save(dest/'COMPLETE.json',dict(status='complete',job=job,seconds=time.perf_counter()-start,head_fits=14,baseline_id=entry['baseline_id'],parent_weights_sha256=entry['weights_sha256'],
  protocol_sha256=sha(OUT/'protocol.json'),source_hash=sha(p/'COMPLETE.json'),fit_audit=fit_audit,
  prediction_sha256={q.name:sha(q) for q in dest.glob('*.npz')},selection_group_isolation=True,outer_group_isolation=True,honest_inner_coverage=len(tr),
  off_margin_error=0.,serialized_margin_error=0.,whole_path_lambda0_margin_error=0.,training_device='cpu',gpu0_used=False))
 print('DONE',ds,model,fold,round(time.perf_counter()-start,2),flush=True);gc.collect()


def worker():
 if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Head-only worker must have CUDA_VISIBLE_DEVICES empty')
 torch.set_num_threads(2);spec=check();(OUT/'claims').mkdir(exist_ok=True)
 if json.loads((OUT/'parity.json').read_text())['status']!='pass':raise ValueError('36D parity first')
 for job in spec['jobs']:
  ds,model,fold=job['dataset'],job['model'],job['fold'];dest=OUT/'runs'/ds/model/f'f{fold}';claim=OUT/'claims'/f'{ds}_{model}_f{fold}.json'
  if (dest/'COMPLETE.json').exists():continue
  try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
  except FileExistsError:continue
  with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),job=job,start=time.time()),f)
  try:one(job)
  except Exception as e:
   save(OUT/'failures'/f'{ds}_{model}_f{fold}.json',dict(error=repr(e),traceback=traceback.format_exc(),job=job));raise
 print('WORKER_COMPLETE',os.getpid(),flush=True)



def real_replay():
 from validate_locked_dual_20261030 import outer
 physical=os.environ.get('CUDA_VISIBLE_DEVICES')
 if physical not in ['1','2','3']:raise ValueError('Only physical GPU1/2/3')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
 spec=check();rows=[]
 for job in spec['jobs']:
  ds,model,fold=job['dataset'],job['model'],job['fold'];folder=OUT/'runs'/ds/model/f'f{fold}'
  if not (folder/'COMPLETE.json').exists():raise ValueError('All head jobs must complete before real replay')
  m=method(model);entry,te,base,prob,oldnet,data,error=outer(ds,model,fold)
  if model=='HYDRA':net=m.Model(state=oldnet.state)
  else:
   net=m.Model(extra=0).cuda();net.load_state_dict(oldnet.state_dict(),strict=True);net.eval()
  del oldnet
  for arm,(key,fusion,recall) in NEW_ARMS.items():
   state=dict(np.load(folder/(arm+'_state.npz')));net.attach_correction(state);x=features(ds)[key]
   if model=='HYDRA':
    actual=net.decision_function(data[0],dynamic=x[te],enabled=True);off=net.decision_function(data[0],enabled=False)
   else:
    parts=[];offs=[];full=data[2]
    with torch.no_grad():
     for ix in torch.tensor(full,device='cuda').split(128 if model=='ArcFENet' else 64):
      dyn=x[ix.cpu().numpy()]
      with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model=='ArcFusionNet'):
       enhanced=net(data[0][ix],None if data[1] is None else data[1][ix],dynamic=dyn,enabled=True)
       raw=net(data[0][ix],None if data[1] is None else data[1][ix],enabled=False)
      parts.append((enhanced[:,1].float()-enhanced[:,0].float()).cpu().numpy());offs.append((raw[:,1].float()-raw[:,0].float()).cpu().numpy())
    actual=np.concatenate(parts);off=np.concatenate(offs)
    if len(full)!=len(te):
     lookup={int(v):i for i,v in enumerate(full)};pick=np.array([lookup[int(v)] for v in te]);actual,off=actual[pick],off[pick]
   saved=np.load(folder/(arm+'.npz'))['margin'];err=float(abs(actual-saved).max());offerr=float(abs(off-base).max());changes=int(((actual>=0)!=(saved>=0)).sum())
   assert err<2e-4 and offerr==0 and changes==0,(ds,model,fold,arm,err,offerr,changes)
   rows.append(dict(dataset=ds,model=model,fold=fold,arm=arm,margin_error=err,off_margin_error=offerr,decision_changes=changes,physical_gpu=int(physical) if model!='HYDRA' else None))
  print('REAL_REPLAY',ds,model,fold,flush=True);del net,data;gc.collect();torch.cuda.empty_cache()
 pd.DataFrame(rows).to_csv(OUT/'real_model_replay.csv',index=False)
 save(OUT/'REAL_MODEL_REPLAY.json',dict(status='pass',model_states=len(rows),max_margin_error=max(v['margin_error'] for v in rows),max_off_margin_error=max(v['off_margin_error'] for v in rows),decision_changes=0,physical_gpu=[int(physical)],gpu0_used=False))


def run_study(suite,execute=False):
 if set(suite['datasets'])!={'continuous5m','zenodo5501527'}:raise ValueError('Both continuous datasets separately')
 if execute:raise ValueError('Use --parity, --prepare then three explicit CPU --worker processes')
 print('12 predefined arms,420 linear head fits; no new backbone training; parent state frozen')


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--parity',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
 if args.parity:parity()
 elif args.prepare:prepare()
 elif args.worker:worker()
 elif args.replay:real_replay()
