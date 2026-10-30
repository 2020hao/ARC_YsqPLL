"""Fusion-current capability and honest inner-group calibration qualification.

No frozen baseline file is changed. Same-record tests are explicitly diagnostic,
not unknown-appliance evidence or a complete author-method reproduction.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import yaml
import _bootstrap as boot
from arc_model.implementations.baseline_arc_fusion.network import Model
from arc_model.evaluation.cycle_metrics import cycle_metrics

ROOT=boot.ROOT
OUT=ROOT/'outputs/fusion_qualification_20261030'
CONFIG=ROOT/'configs/modules/fusion_qualification_20261030.yaml'
DATASETS=('continuous5m','zenodo5501527')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def config():return yaml.safe_load(CONFIG.read_text())


def dataset(ds):
 c=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[ds]
 d=ROOT/c['prepared_roots']['fixed20'];t=pd.read_csv(d/'cycles.csv');keep=np.load(ROOT/c['reviewed_indices']);splits=json.loads((ROOT/c['split_file']).read_text())
 return c,d,t,keep,splits


def prepare():
 if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
 OUT.mkdir(parents=True,exist_ok=True);jobs=[];identities=[];counts=[]
 for ds in DATASETS:
  c,d,t,keep,splits=dataset(ds);y=t.label.to_numpy();folder=OUT/'data'/ds;folder.mkdir(parents=True)
  stratum=t.iloc[keep].record_id.astype(str)+'|'+t.iloc[keep].label.astype(str)
  if stratum.value_counts().min()<10:
   stratum=t.iloc[keep].condition_key.astype(str)+'|'+t.iloc[keep].label.astype(str);rule='condition_key x label fallback because record/class count<10'
  else:rule='record_id x label; all strata>=10'
  trainval,test=train_test_split(keep,test_size=.1,random_state=20261030,stratify=stratum.to_numpy())
  lookup=dict(zip(keep,stratum));s=np.array([lookup[int(i)] for i in trainval])
  train,val=train_test_split(trainval,test_size=2/9,random_state=20261031,stratify=s)
  assert not np.intersect1d(train,test).size and not np.intersect1d(train,val).size and not np.intersect1d(val,test).size
  assert np.array_equal(np.sort(np.r_[train,val,test]),np.sort(keep))
  assert set(t.iloc[train].record_id)==set(t.iloc[test].record_id)==set(t.iloc[val].record_id)
  np.savez(folder/'capability_split.npz',train=train,validation=val,test=test,reviewed=keep)
  counts.append(dict(dataset=ds,protocol='same_record_capability',N=len(keep),train=len(train),validation=len(val),test=len(test),stratification=rule,shared_records=t.iloc[test].record_id.nunique()))
  for seed in [0,1,2]:jobs.append(dict(id=f'capability_{ds}_s{seed}',kind='capability',dataset=ds,seed=seed,indices=str(folder/'capability_split.npz')))
  for f,spl in enumerate(splits):
   tr=np.asarray(spl['train'],dtype=int);te=np.intersect1d(spl['test'],keep)
   assert np.array_equal(np.sort(tr),np.setdiff1d(keep,te))
   groups=t.iloc[tr].condition_key.to_numpy();sg=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261030+f)
   for inner,(a,b) in enumerate(sg.split(np.zeros(len(tr)),y[tr],groups)):
    fit,validation=tr[a],tr[b]
    assert not set(t.iloc[fit].condition_key)&set(t.iloc[validation].condition_key)
    assert set(np.unique(y[fit]))==set(np.unique(y[validation]))=={0,1}
    p=folder/f'outer{f}_inner{inner}.npz';np.savez(p,train=fit,validation=validation,outer_train=tr,outer_test=te)
    jobs.append(dict(id=f'inner_{ds}_f{f}_i{inner}',kind='inner',dataset=ds,outer_fold=f,inner_fold=inner,seed=0,indices=str(p)))
  paths=[d/'temporal.npy',d/'spectrum.npy',d/'cycles.csv',ROOT/c['reviewed_indices'],ROOT/c['split_file']]
  for p in paths:identities.append(dict(path=str(p),sha256=sha(p)))
  for p in folder.glob('*.npz'):identities.append(dict(path=str(p),sha256=sha(p)))
 # Capability first; small public jobs complete quickly. All jobs are immutable.
 jobs.sort(key=lambda j:(j['kind']!='capability',j['dataset']=='continuous5m',j.get('seed',0),j.get('outer_fold',0),j.get('inner_fold',0)))
 paths=[Path(__file__),CONFIG,ROOT/'src/arc_model/implementations/baseline_arc_fusion/network.py',ROOT/'src/arc_model/evaluation/cycle_metrics.py']
 protocol=dict(status='frozen_before_new_fits',date='2026-10-30',datasets=DATASETS,jobs=jobs,training=config()['training'],
  identities=identities,code=[dict(path=str(p),sha256=sha(p)) for p in paths],gpu_allowed=[1,2,3],gpu_reserved=[0],
  capability='same-record70/20/10 fixed stratified split, three initialization seeds; no validation/test model selection; not exact author data/protocol',
  calibration='seed0 only; two group-disjoint inner OOF fits per outer fold; positive-slope affine raw-logit calibration fit using outer-training OOF only; fixed penalties',
  unchanged='current38374/1361 cohort, labels, outer splits, all original frozen weights/metrics',
  missing_original='current-only architecture adaptation; no supply voltage/healthy differential/synthetic-target method',
  inference='BF16 network, FP32 loss/softmax and raw class-logit difference; probability clipping is not used to reconstruct logits')
 save(OUT/'protocol.json',protocol);pd.DataFrame(counts).to_csv(OUT/'split_counts.csv',index=False)
 print('FROZEN',len(jobs),'fits',flush=True)


def setup_gpu():
 physical=os.environ.get('CUDA_VISIBLE_DEVICES')
 if physical not in ['1','2','3']:raise ValueError('Expose exactly one allowed physicalGPU1/2/3')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.30)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
 return physical


def check_protocol():
 p=json.loads((OUT/'protocol.json').read_text())
 for r in p['code']:assert sha(r['path'])==r['sha256'],r['path']
 return p


def preprocess(t,sp,y,train,runseed):
 rf=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=8,class_weight='balanced',random_state=runseed,n_jobs=2)
 rf.fit(sp[train],y[train]);bins=np.sort(np.argsort(rf.feature_importances_)[-32:]);a=sp[:,bins]
 state=dict(fit_indices=train,harmonic_indices=bins,rf_importance=rf.feature_importances_,temporal_mean=t[train].mean((0,3),keepdims=True),temporal_std=np.maximum(t[train].std((0,3),keepdims=True),1e-6),spectral_mean=a[train].mean(0),spectral_std=np.maximum(a[train].std(0),1e-6))
 return state


def tensors(t,sp,state):
 x=torch.tensor((t-state['temporal_mean'])/state['temporal_std'],device='cuda')
 a=sp[:,state['harmonic_indices']];s=torch.tensor(((a-state['spectral_mean'])/state['spectral_std'])[:,None],device='cuda')
 return x,s


def infer(net,x,s,ids):
 margin=[];prob=[]
 with torch.no_grad():
  for ix in torch.tensor(ids,device='cuda').split(64):
   with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x[ix],s[ix])
   z=z.float();margin.append((z[:,1]-z[:,0]).cpu().numpy());prob.append(z.softmax(-1)[:,1].cpu().numpy())
 return np.concatenate(margin),np.concatenate(prob)


def train_job(j,physical):
 q=OUT/'runs'/j['id'];q.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cfg=config()['training']
 c,d,table,keep,_=dataset(j['dataset']);inds=np.load(j['indices']);tr=inds['train'];va=inds['validation'];te=inds['test'] if j['kind']=='capability' else None
 y=table.label.to_numpy();temporal=np.load(d/'temporal.npy');spectrum=np.load(d/'spectrum.npy')
 if j['kind']=='inner':
  for key in ['record_id','source_group','condition_key']:assert not set(table.iloc[tr][key])&set(table.iloc[va][key])
  assert not np.intersect1d(np.r_[tr,va],inds['outer_test']).size
 runseed=310000+j['seed']*1000+(j.get('outer_fold',0))*10+j.get('inner_fold',0)
 state=preprocess(temporal,spectrum,y,tr,runseed);np.savez(q/'preprocessing.npz',**state)
 np.savez(q/'indices.npz',train=tr,validation=va,**({'test':te} if te is not None else {}))
 x,s=tensors(temporal,spectrum,state);labels=torch.tensor(y,device='cuda');tri=torch.tensor(tr,device='cuda')
 torch.manual_seed(runseed);net=Model(extra=0).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['min_lr'])
 rng=torch.Generator(device='cuda').manual_seed(runseed+90000);history=[]
 for epoch in range(cfg['epochs']):
  net.train();order=tri[torch.randperm(len(tri),device='cuda',generator=rng)];total=torch.zeros((),device='cuda')
  for ix in order.split(cfg['batch']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x[ix],s[ix])
   loss=torch.nn.functional.cross_entropy(z.float(),labels[ix]);loss.backward();opt.step();total+=loss.detach()*len(ix)
  v=float(total)/len(tr);assert np.isfinite(v);history.append(dict(epoch=epoch+1,loss=v,lr=opt.param_groups[0]['lr']));sch.step()
  save(q/'progress.json',dict(job=j['id'],epoch=epoch+1,epochs=cfg['epochs'],loss=v,seconds=time.perf_counter()-start,gpu_physical=physical))
  if epoch==0 or (epoch+1)%10==0:print('EPOCH',j['id'],epoch+1,round(time.perf_counter()-start,1),flush=True)
 net.eval();torch.save(net.state_dict(),q/'model.pt')
 for part,ids in [('train',tr),('validation',va)]+([('test',te)] if te is not None else []):
  margin,prob=infer(net,x,s,ids);np.savez(q/f'{part}_predictions.npz',indices=ids,labels=y[ids],margin=margin,probability=prob)
 save(q/'history.json',history)
 save(q/'COMPLETE.json',dict(status='complete',job=j,runseed=runseed,epochs=len(history),gpu_physical=physical,seconds=time.perf_counter()-start,protocol_sha256=sha(OUT/'protocol.json'),model_sha256=sha(q/'model.pt'),preprocessing_sha256=sha(q/'preprocessing.npz'),index_sha256=sha(q/'indices.npz'),prediction_hashes={p.name:sha(p) for p in q.glob('*_predictions.npz')}))
 print('DONE',j['id'],round(time.perf_counter()-start,2),flush=True)
 del net,opt,sch,x,s,labels,tri;gc.collect();torch.cuda.empty_cache()


def worker():
 physical=setup_gpu();spec=check_protocol();(OUT/'claims').mkdir(exist_ok=True)
 for j in spec['jobs']:
  q=OUT/'runs'/j['id'];claim=OUT/'claims'/(j['id']+'.json')
  if (q/'COMPLETE.json').exists():continue
  try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
  except FileExistsError:continue
  with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),gpu=physical,job=j['id'],started=time.time()),f)
  try:train_job(j,physical)
  except Exception as e:
   save(OUT/'failures'/(j['id']+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),gpu=physical));raise
 print('WORKER_FINISHED',physical,flush=True)


def group_weights(y,g):
 w=np.zeros(len(y),float)
 for key in np.unique(g):
  for cls in [0,1]:
   mask=(g==key)&(y==cls)
   if not mask.any():raise ValueError('Every training condition must contain both classes')
   w[mask]=1/(2*len(np.unique(g))*mask.sum())
 return w


def fit_calibration(margin,y,groups):
 m=np.asarray(margin,dtype=float);y=np.asarray(y,dtype=float);w=group_weights(y,groups)
 def objective(theta):
  a=np.exp(theta[0]);z=a*m+theta[1];loss=float(w@(np.logaddexp(0,z)-y*z))+.001*float(theta@theta)
  residual=w*(expit(z)-y);grad=np.array([residual@(a*m),residual.sum()])+.002*theta
  return loss,grad
 r=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-8,8),(-20,20)],options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
 if not r.success:raise RuntimeError(str(r.message))
 return dict(slope=float(np.exp(r.x[0])),intercept=float(r.x[1]),objective=float(r.fun),status=str(r.message),regularization=.001)


def outer_logits(ds,fold,physical):
 c,d,table,keep,splits=dataset(ds);reg=json.loads((ROOT/'outputs/grouped_baseline_standard_20261029/baseline_registry.json').read_text())['entries']
 e=next(v for v in reg if v['dataset']==ds and v['model']=='ArcFusionNet_current' and v['fold']==fold and v['seed']==0)
 assert sha(e['weights_path'])==e['weights_sha256'];q=Path(e['folder']);state=np.load(q/'preprocessing.npz');p=np.load(q/'predictions.npz');ids=p['indices'];full=ids
 receipt=json.loads((q/'COMPLETE.json').read_text())
 if receipt['reuse']:
  old=ROOT/f'outputs/rcr_three_backbones_20261028/runs/{ds}/fixed20/ArcFusionNet/f{fold}_s0/outer/M0/predictions.npz';full=np.load(old)['indices']
 x,s=tensors(np.load(d/'temporal.npy'),np.load(d/'spectrum.npy'),state);net=Model(extra=0).cuda();net.load_state_dict(torch.load(e['weights_path'],map_location='cpu',weights_only=True),strict=True);net.eval()
 margin,prob=infer(net,x,s,full);lookup={int(v):i for i,v in enumerate(full)};ix=np.array([lookup[int(v)] for v in ids]);margin=margin[ix];prob=prob[ix]
 err=float(abs(prob-p['probability']).max());changes=int(((margin>=0)!=(p['probability']>=.5)).sum());assert err<5e-5 and changes==0,(err,changes)
 del net,x,s;gc.collect();torch.cuda.empty_cache()
 return ids,margin,prob,dict(max_probability_error=err,decision_changes=changes,parent_baseline_id=e['baseline_id'],parent_weights_sha256=e['weights_sha256'],gpu_physical=physical)


def calibrate_all():
 physical=setup_gpu();spec=check_protocol()
 for ds in DATASETS:
  c,d,table,keep,splits=dataset(ds);y=table.label.to_numpy();groups=table.condition_key.to_numpy()
  for fold,spl in enumerate(splits):
   out=OUT/'calibration'/ds/f'f{fold}';
   if (out/'COMPLETE.json').exists():continue
   matches=[j for j in spec['jobs'] if j['kind']=='inner' and j['dataset']==ds and j['outer_fold']==fold]
   if not all((OUT/'runs'/j['id']/'COMPLETE.json').exists() for j in matches):raise ValueError('Complete all inner jobs before calibration')
   tr=np.asarray(spl['train']);cover=np.zeros(len(table),int);oof=np.full(len(table),np.nan)
   for j in matches:
    q=OUT/'runs'/j['id'];r=json.loads((q/'COMPLETE.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json')
    z=np.load(q/'validation_predictions.npz');ids=np.load(q/'indices.npz');fit,va=ids['train'],ids['validation']
    assert not set(groups[fit])&set(groups[va]);assert np.array_equal(z['indices'],va)
    cover[va]+=1;oof[va]=z['margin']
   assert (cover[tr]==1).all() and cover.sum()==len(tr)
   params=fit_calibration(oof[tr],y[tr],groups[tr]);te,margin,prob,audit=outer_logits(ds,fold,physical)
   out.mkdir(parents=True,exist_ok=False);np.savez(out/'predictions.npz',indices=te,labels=y[te],raw_margin=margin,raw_probability=prob,calibrated_margin=params['slope']*margin+params['intercept'])
   np.savez(out/'calibration_training.npz',indices=tr,labels=y[tr],inner_oof_margin=oof[tr]);save(out/'calibrator.json',params);save(out/'COMPLETE.json',dict(status='complete',dataset=ds,fold=fold,**audit,parameters=params,protocol_sha256=sha(OUT/'protocol.json')))
   print('CALIBRATED',ds,fold,params,flush=True)


def summarize():
 spec=check_protocol();rows=[];capability=[];group_rows=[];audit=[]
 for ds in DATASETS:
  _,_,table,keep,_=dataset(ds);g=table.condition_key.to_numpy();y=table.label.to_numpy()
  for seed in [0,1,2]:
   q=OUT/'runs'/f'capability_{ds}_s{seed}';done=json.loads((q/'COMPLETE.json').read_text());assert done['epochs']==70
   for part in ['train','validation','test']:
    z=np.load(q/f'{part}_predictions.npz');assert sha(q/f'{part}_predictions.npz')==done['prediction_hashes'][f'{part}_predictions.npz']
    ids=z['indices'];m=cycle_metrics(z['labels'],z['margin'],g[ids],0);capability.append(dict(dataset=ds,seed=seed,partition=part,**m))
  for arm,key in [('raw','raw_margin'),('calibration_only','calibrated_margin')]:
   score=np.full(len(table),np.nan);coverage=np.zeros(len(table),int)
   for fold in range(5):
    p=OUT/'calibration'/ds/f'f{fold}';done=json.loads((p/'COMPLETE.json').read_text());z=np.load(p/'predictions.npz');ids=z['indices'];score[ids]=z[key];coverage[ids]+=1
    if arm=='raw':audit.append(dict(dataset=ds,fold=fold,**{k:done[k] for k in ['max_probability_error','decision_changes']}))
   assert (coverage[keep]==1).all() and coverage.sum()==len(keep)
   np.savez(OUT/f'{ds}_{arm}_oof.npz',indices=keep,labels=y[keep],scores=score[keep])
   rows.append(dict(dataset=ds,arm=arm,seed=0,**cycle_metrics(y[keep],score[keep],g[keep],0)))
   for group in np.unique(g[keep]):
    ix=keep[g[keep]==group];m=cycle_metrics(y[ix],score[ix],g[ix],0);group_rows.append(dict(dataset=ds,arm=arm,condition=group,**m))
 pd.DataFrame(rows).to_csv(OUT/'calibration_results.csv',index=False);pd.DataFrame(capability).to_csv(OUT/'capability_seed_results.csv',index=False);groups=pd.DataFrame(group_rows);groups.to_csv(OUT/'calibration_group_results.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'outer_replay_audit.csv',index=False)
 cap=pd.DataFrame(capability);fields=['Accuracy','BAcc_condition','Precision','Recall','FPR','F1','ROC_AUC','AP']
 cap.groupby(['dataset','partition'])[fields].mean().to_csv(OUT/'capability_means.csv');cap.groupby(['dataset','partition'])[fields].std().to_csv(OUT/'capability_seed_sd.csv')
 intervals=[]
 for ds in DATASETS:
  a=groups[(groups.dataset==ds)&(groups.arm=='raw')].set_index('condition');b=groups[(groups.dataset==ds)&(groups.arm=='calibration_only')].set_index('condition');delta=(b.BAcc_condition-a.BAcc_condition).to_numpy();rng=np.random.default_rng(20261030);values=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(1);lo,hi=np.quantile(values,[.025,.975]);intervals.append(dict(dataset=ds,delta=float(delta.mean()),lower=float(lo),upper=float(hi),seeds=1,unit='condition',interpretation='exploratory descriptive paired interval'))
 pd.DataFrame(intervals).to_csv(OUT/'calibration_paired_intervals.csv',index=False)
 save(OUT/'RESULTS_COMPLETE.json',dict(status='complete',capability_fits=6,inner_group_fits=20,outer_calibrations=10,physical_gpu0_used=False,outer_decision_replay_changes=sum(r['decision_changes'] for r in audit),scope='current-only capability and single-seed honest calibration diagnostics; no full author method claim; no frozen metric overwrite'))
 report='Fusion-current基线能力核验与校准诊断\n\n原网络与训练预算保持不变；同源窗口70/20/10是能力核验，不是未见电器泛化或原论文完整复现。分组校准单seed0，采用每外折两内折组隔离OOF训练两参数正斜率仿射校准，不使用外测试调阈值。\n\n同源窗口测试（三种子均值）\n'+cap[cap.partition=='test'].groupby('dataset')[fields].mean().to_string()+'\n\n当前整组隔离原始/校准指标（seed0）\n'+pd.DataFrame(rows).to_string(index=False)+'\n\n条件成对描述区间\n'+pd.DataFrame(intervals).to_string(index=False)+'\n\n校准保留各外折内排序，改变跨折分数尺度及决策边界；AUC/AP需按其评价范围解释。正向校准不足以证明原架构已等价于供电电压对齐/健康差分/合成适配完整原方法。所有指标均为周波/20ms窗口级，不推导报警时延。\n'
 (OUT/'完整结果与解释.txt').write_text(report);print(report,flush=True)


def run_study(suite,*,execute=False):
 if set(suite['datasets'])!={'new5m','zenodo5501527'}:raise ValueError('Both datasets required')
 if execute:raise ValueError('Freeze preparation then use explicit GPU workers')
 print('PREFLIGHT Fusion-current6capability+20inner-group fits; no automatic training')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--calibrate',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker:worker()
 elif a.calibrate:calibrate_all()
 elif a.summarize:summarize()
 else:p.error('Select an action')
