"""Hierarchical physical-cycle states on the complete462 and public data.

No raw labels are regenerated and no baseline package is edited. Model selection
does not inspect outer-test checkpoints or optimize the decision threshold.
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_hierarchy_20261105'
CFG=ROOT/'configs/modules/phase_hierarchy_20261105.yaml'
PYTHON='/data/hzy/conda/envs/NILMFormer/bin/python'
SELF=ROOT/'outputs/self462_1m_baselines_20261105'
PUBLIC=ROOT/'outputs/map_input_calibration_20261105/data/zenodo5501527'

def make_model(arm):
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 from arc_model.implementations.phase_hierarchy_net import PhaseHierarchyNet
 if arm in ('maxpool_dynamic','gain_phase'):return DynamicStateNet('maxpool_dynamic')
 return PhaseHierarchyNet('small' if arm=='hierarchy_small' else 'wide')

def sha(path):
 import hashlib
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**22),b''):h.update(b)
 return h.hexdigest()

def save(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 tmp.replace(p)

def host_data(ds):
 if ds=='self462':
  table=pd.read_csv(SELF/'cycles.csv');keep=np.arange(len(table))
  x=np.load(SELF/'cnn_snn_2m_control.npy',mmap_mode='r')
  mask=np.ones((len(table),9),dtype=np.float32)
 else:
  table=pd.read_csv(PUBLIC/'cycles.csv');keep=np.load(PUBLIC/'keep.npy')
  x=np.load(PUBLIC/'realtime_raw.npy',mmap_mode='r')[:,:,:5,:]
  mask=np.load(PUBLIC/'realtime_mask.npy')
 return table,keep,x,mask

def roles(ds,fold):
 if ds=='self462':
  table=pd.read_csv(SELF/'cycles.csv')
  return np.flatnonzero(table.fold.to_numpy()!=fold),np.flatnonzero(table.fold.to_numpy()==fold)
 r=np.load(PUBLIC/f'roles_f{fold}.npz')
 return r['outer_train'],r['test']

def validate_roles(ds):
 table,keep,_,_=host_data(ds);seen=np.zeros(len(table),int)
 for fold in range(4 if ds=='self462' else 5):
  tr,te=roles(ds,fold)
  assert not set(tr)&set(te)
  for col in ('condition_key','record_id','source_group'):
   if col in table:assert not set(table.iloc[tr][col])&set(table.iloc[te][col]),(ds,fold,col)
  seen[te]+=1
  assert set(table.iloc[tr].label)=={0,1}
 assert (seen[keep]==1).all() and (seen[np.setdiff1d(np.arange(len(table)),keep)]==0).all()
 return dict(windows=len(keep),groups=table.iloc[keep].condition_key.nunique(),records=table.iloc[keep].record_id.nunique(),normal=int((table.iloc[keep].label==0).sum()),arc=int((table.iloc[keep].label==1).sum()))

def prepare(cfg):
 import torch
 from arc_model.implementations.dynamic_state_net import native_dynamic_summary
 torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
 identities=[];cohorts={}
 for ds in cfg['datasets']:
  table,keep,x,mask=host_data(ds);cohorts[ds]=validate_roles(ds)
  path=OUT/f'{ds}_dynamic36.npy'
  prior=ROOT/'outputs/dynamic_state_20261105'/path.name
  if not path.exists() and prior.exists():path.symlink_to(prior)
  if not path.exists():
   values=np.zeros((len(table),36),dtype=np.float32)
   for start in range(0,len(table),1024):
    end=min(len(table),start+1024)
    with torch.inference_mode():
     raw=torch.from_numpy(np.array(x[start:end],copy=True))
     values[start:end]=native_dynamic_summary(raw,torch.from_numpy(np.array(mask[start:end],copy=True))).cpu().numpy()
   assert np.isfinite(values).all()
   np.save(path,values)
  for p in ([SELF/'cycles.csv',SELF/'cnn_snn_2m_control.npy',SELF/'splits.json'] if ds=='self462' else [PUBLIC/'cycles.csv',PUBLIC/'realtime_raw.npy',PUBLIC/'realtime_mask.npy',PUBLIC/'keep.npy']+[PUBLIC/f'roles_f{f}.npz' for f in range(5)]):
   identities.append(dict(path=str(p),sha256=sha(p)))
  identities.append(dict(path=str(path),sha256=sha(path)))
 jobs=[]
 # Start with all public jobs: diagnostic feedback arrives without touching any selection rule.
 for ds in ('zenodo5501527','self462'):
  for seed in cfg['seeds']:
   for fold in range(cfg['outer_folds'][ds]):
    for arm in cfg['arms']:jobs.append(dict(dataset=ds,fold=fold,seed=seed,arm=arm))
 for p in [CFG,Path(__file__),ROOT/'scripts/run_experiment_suite.py',ROOT/'src/arc_model/implementations/phase_hierarchy_net/network.py',ROOT/'src/arc_model/implementations/phase_hierarchy_net/augmentation.py',ROOT/'src/arc_model/implementations/dynamic_state_net/network.py',ROOT/'src/arc_model/implementations/dynamic_state_net/summary.py',ROOT/'src/arc_model/implementations/dynamic_state_net/__init__.py',ROOT/'src/arc_model/implementations/map_input_calibration/deep_network.py',ROOT/'src/arc_model/implementations/phase_event_snn/data.py']:
  identities.append(dict(path=str(p),sha256=sha(p)))
 sealed=[ROOT/'src/arc_model/implementations'/n for n in ('baseline_arc_fenet','baseline_hydra','baseline_arc_fusion')]
 baseline_hashes={str(p):sha(p) for d in sealed for p in d.rglob('*.py')}
 protocol=dict(status='frozen_before_new_fits',config=cfg,jobs=jobs,cohorts=cohorts,identities=identities,baseline_sources=baseline_hashes,
  exploration='Existing outer folds have been exposed during previous development; this is a versioned exploratory comparison, not independent confirmation.',
  references=[dict(title='SpikingResformer',url='https://openaccess.thecvf.com/content/CVPR2024/html/Shi_SpikingResformer_Bridging_ResNet_and_Vision_Transformer_in_Spiking_Neural_Networks_CVPR_2024_paper.html',used_for='Motivation for preserving local peaks and multistage information; no architecture performance transferred'),dict(title='QKFormer',url='https://proceedings.neurips.cc/paper_files/paper/2024/hash/179f5dcdeedc149443ebd3ba70811dbd-Abstract-Conference.html',used_for='Hierarchy and information preservation motivation; attention not imported')])
 if (OUT/'protocol.json').exists():
  old=json.loads((OUT/'protocol.json').read_text())
  assert old['config']==cfg
  for row in old['identities']:assert sha(row['path'])==row['sha256'],row['path']
  return old
 save(OUT/'protocol.json',protocol);return protocol

def destination(j):return OUT/'runs'/j['dataset']/j['arm']/f'f{j["fold"]}_s{j["seed"]}'

def source(j):
 if j['arm']!='maxpool_dynamic':return None
 return ROOT/'outputs/dynamic_state_20261105/runs'/j['dataset']/'maxpool_dynamic'/f'f{j["fold"]}_s{j["seed"]}'

def batches(ids,size):
 parts=list(ids.split(size))
 if len(parts)>1 and len(parts[-1])==1:parts[-2]=__import__('torch').cat(parts[-2:]);parts.pop()
 return parts

def run_job(j,cfg):
 import torch
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 from arc_model.implementations.phase_event_snn.data import training_statistics,condition_class_weights
 from arc_model.evaluation.cycle_metrics import cycle_metrics
 d=destination(j);d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return
 start=time.perf_counter();table,keep,x,mask=host_data(j['dataset']);tr,te=roles(j['dataset'],j['fold'])
 y=table.label.to_numpy();groups=table.condition_key.to_numpy();mean,std=training_statistics(x,tr)
 dsumm=np.load(OUT/f'{j["dataset"]}_dynamic36.npy',mmap_mode='r')
 summary_mean=np.zeros(36,np.float32);summary_std=np.ones(36,np.float32)
 # Missing history is omitted from moment estimation and represented explicitly.
 for lag_i in range(4):
  valid=tr[dsumm[tr,32+lag_i]>.5]
  if len(valid):
   cols=slice(8*lag_i,8*(lag_i+1));summary_mean[cols]=dsumm[valid,cols].mean(0,dtype=np.float64);summary_std[cols]=np.maximum(dsumm[valid,cols].std(0,dtype=np.float64),1e-6)
 run_seed=1000*j['seed']+j['fold'];torch.manual_seed(run_seed);torch.cuda.manual_seed_all(run_seed)
 net=make_model(j['arm']).cuda();net.set_statistics(mean,std,summary_mean,summary_std)
 xp=torch.as_tensor((x-mean[None,None,:,None])/std[None,None,:,None],device='cuda')
 mp=torch.as_tensor(mask,device='cuda');summ=torch.as_tensor(np.array(dsumm,copy=True),device='cuda')
 target=torch.as_tensor(y,dtype=torch.float32,device='cuda')
 weights=torch.as_tensor(condition_class_weights(y,groups,tr),device='cuda')
 src=source(j);reused=False;losses=[]
 if src is not None and (src/'COMPLETE.json').exists():
  receipt=json.loads((src/'COMPLETE.json').read_text())
  for name,digest in receipt['hashes'].items():assert sha(src/name)==digest,(src,name)
  old_ids=np.load(src/'indices.npz');np.testing.assert_array_equal(tr,old_ids['train']);np.testing.assert_array_equal(te,old_ids['test'])
  old=np.load(src/'preprocessing.npz');np.testing.assert_array_equal(mean,old['mean'][:5]);np.testing.assert_array_equal(std,old['std'][:5])
  result=net.load_state_dict(torch.load(src/'model.pt',map_location='cpu',weights_only=True),strict=False)
  assert not result.unexpected_keys
  assert all(k in ('input_mean','input_std','dynamic_mean','dynamic_std') for k in result.missing_keys),result.missing_keys
  net.set_statistics(mean,std,summary_mean,summary_std);reused=True
  save(d/'REUSE.json',dict(source=str(src),receipt=receipt))
 else:
  tc=cfg['training'];opt=torch.optim.AdamW(net.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'],eta_min=tc['min_learning_rate'])
  ids=torch.as_tensor(tr,device='cuda');gen=torch.Generator(device='cuda').manual_seed(run_seed+90000)
  auggen=torch.Generator(device='cuda').manual_seed(run_seed+191105)
  augmean=torch.as_tensor(mean,device='cuda');augstd=torch.as_tensor(std,device='cuda')
  for epoch in range(tc['epochs']):
   net.train();total=torch.zeros((),device='cuda')
   for ix in batches(ids[torch.randperm(len(ids),device='cuda',generator=gen)],tc['batch_size']):
    opt.zero_grad(set_to_none=True)
    batch_x=xp[ix]
    if j['arm']=='gain_phase':
     from arc_model.implementations.phase_hierarchy_net.augmentation import augment_standardized
     batch_x=augment_standardized(batch_x,augmean,augstd,auggen)
    logits,activity=net(batch_x,mp[ix],return_activity=True,summary=summ[ix])
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target[ix],reduction='none')*weights[ix]).mean()+tc['spike_activity_penalty']*activity
    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),tc['gradient_clip']);opt.step();total+=loss.detach()*len(ix)
   scheduler.step();value=float(total)/len(tr);assert np.isfinite(value);losses.append(value)
   save(d/'progress.json',dict(job=j,epoch=epoch+1,epochs=tc['epochs'],loss=value,elapsed_seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
  del opt,scheduler
 net.eval();pred=[]
 with torch.inference_mode():
  for ix in torch.as_tensor(te,device='cuda').split(512):pred.append(net(xp[ix],mp[ix],summary=summ[ix]).sigmoid().cpu().numpy())
 prob=np.concatenate(pred)
 if reused:
  z=np.load(src/'predictions.npz');oldprob=z['probability'] if 'probability' in z else z['score']
  np.testing.assert_allclose(prob,oldprob,atol=1e-6,rtol=0)
  replay_error=float(np.max(np.abs(prob-oldprob)))
 else:replay_error=None
 torch.save(net.state_dict(),d/'model.pt');np.savez(d/'predictions.npz',indices=te,labels=y[te],probability=prob)
 np.savez(d/'indices.npz',train=tr,test=te);np.savez(d/'preprocessing.npz',fit_indices=tr,mean=mean,std=std,summary_mean=summary_mean,summary_std=summary_std)
 save(d/'metrics.json',cycle_metrics(y[te],prob,groups[te]));save(d/'training_loss.json',losses)
 files=['model.pt','predictions.npz','indices.npz','preprocessing.npz','metrics.json','training_loss.json']
 save(d/'COMPLETE.json',dict(status='complete',job=j,reused=reused,parameters=sum(p.numel() for p in net.parameters()),threshold=.5,
  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.perf_counter()-start,reuse_prediction_max_abs_error=replay_error,protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(d/n) for n in files}))
 print('COMPLETE',j,'seconds',round(time.perf_counter()-start,2),flush=True)
 del net,xp,mp,summ,target,weights,dsumm;gc.collect();torch.cuda.empty_cache()

def worker(gpu):
 import torch
 assert gpu in (1,2,3) and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 protocol=json.loads((OUT/'protocol.json').read_text());cfg=protocol['config']
 for j in protocol['jobs']:
  d=destination(j);d.mkdir(parents=True,exist_ok=True)
  if (d/'COMPLETE.json').exists():continue
  with (d/'claim.lock').open('w') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (d/'COMPLETE.json').exists():continue
   if (d/'FAILED.json').exists():raise RuntimeError('Investigate preserved failure before retry: '+str(d))
   try:run_job(j,cfg)
   except Exception as exc:save(d/'FAILED.json',dict(job=j,error=repr(exc)));raise

def audit_replay(gpu):
 import torch
 from arc_model.implementations.dynamic_state_net import DynamicStateNet,native_dynamic_summary
 assert gpu in (1,2,3) and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 jobs=json.loads((OUT/'protocol.json').read_text())['jobs'];rows=[]
 for idx,j in enumerate(jobs):
  if idx%3!=gpu-1:continue
  d=destination(j);receipt=json.loads((d/'COMPLETE.json').read_text())
  for n,h in receipt['hashes'].items():assert sha(d/n)==h
  table,keep,x,mask=host_data(j['dataset']);tr,te=roles(j['dataset'],j['fold']);ids=np.load(d/'indices.npz')
  np.testing.assert_array_equal(ids['train'],tr);np.testing.assert_array_equal(ids['test'],te)
  stats=np.load(d/'preprocessing.npz');np.testing.assert_array_equal(stats['fit_indices'],tr)
  net=make_model(j['arm']).cuda();net.load_state_dict(torch.load(d/'model.pt',map_location='cpu',weights_only=True));net.eval()
  summary=np.load(OUT/f'{j["dataset"]}_dynamic36.npy',mmap_mode='r');pred=[]
  # Every saved prediction is replayed, not just selected samples.
  with torch.inference_mode():
   for batch in np.array_split(te,max(1,int(np.ceil(len(te)/512)))):
    xp=torch.as_tensor((x[batch]-stats['mean'][None,None,:,None])/stats['std'][None,None,:,None],device='cuda')
    mp=torch.as_tensor(mask[batch],device='cuda');ss=torch.as_tensor(np.array(summary[batch],copy=True),device='cuda')
    pred.append(net(xp,mp,summary=ss).sigmoid().cpu().numpy())
   probe=te[np.linspace(0,len(te)-1,min(32,len(te))).astype(int)]
   raw=torch.as_tensor(np.array(x[probe],copy=True),device='cuda');mp=torch.as_tensor(mask[probe],device='cuda')
   fresh=native_dynamic_summary(raw,mp).cpu().numpy();np.testing.assert_allclose(fresh,summary[probe],atol=2e-6,rtol=2e-5)
  p=np.concatenate(pred);old=np.load(d/'predictions.npz')['probability'];np.testing.assert_allclose(p,old,atol=2e-6,rtol=0)
  assert np.array_equal(p>=.5,old>=.5)
  rows.append(dict(**j,prediction_max_abs_error=float(abs(p-old).max()),decisions_equal=True,summary_max_abs_error=float(abs(fresh-summary[probe]).max())))
  del net;torch.cuda.empty_cache()
 pd.DataFrame(rows).to_csv(OUT/f'audit_gpu{gpu}.csv',index=False)

def collect(cfg):
 from arc_model.evaluation.cycle_metrics import cycle_metrics
 from scipy.stats import binomtest
 seedrows=[];conditionrows=[];foldrows=[]
 for ds in cfg['datasets']:
  table,keep,_,_=host_data(ds);y=table.label.to_numpy();g=table.condition_key.to_numpy()
  load_col='load' if 'load' in table else 'target_appliance' if 'target_appliance' in table else None
  for arm in cfg['arms']:
   for seed in cfg['seeds']:
    p=np.full(len(table),np.nan);seen=np.zeros(len(table),int)
    for fold in range(cfg['outer_folds'][ds]):
     j=dict(dataset=ds,fold=fold,seed=seed,arm=arm);d=destination(j);assert (d/'COMPLETE.json').exists()
     z=np.load(d/'predictions.npz');ids=z['indices'];p[ids]=z['probability'];seen[ids]+=1
     foldrows.append(dict(**j,**cycle_metrics(y[ids],p[ids],g[ids])))
    assert (seen[keep]==1).all() and np.isfinite(p[keep]).all()
    seedrows.append(dict(dataset=ds,arm=arm,seed=seed,**cycle_metrics(y[keep],p[keep],g[keep])))
    np.savez(OUT/f'{ds}_{arm}_s{seed}_OOF.npz',indices=keep,labels=y[keep],probability=p[keep])
    for group in np.unique(g[keep]):
     ids=keep[g[keep]==group];rr=dict(dataset=ds,arm=arm,seed=seed,condition=group,**cycle_metrics(y[ids],p[ids],g[ids]))
     if load_col:rr['load']=str(table.iloc[ids[0]][load_col])
     conditionrows.append(rr)
 seeds=pd.DataFrame(seedrows);conds=pd.DataFrame(conditionrows)
 metrics=[k for k in cycle_metrics([0,1],[.1,.9],['g','g']) if k not in ('N','groups')]
 summary=seeds.groupby(['dataset','arm'])[metrics].agg(['mean','std']);summary.columns=['_'.join(k) for k in summary.columns]
 summary.reset_index().to_csv(OUT/'main_results.csv',index=False);seeds.to_csv(OUT/'seed_metrics.csv',index=False)
 conds.to_csv(OUT/'per_condition_seed.csv',index=False);pd.DataFrame(foldrows).to_csv(OUT/'per_fold.csv',index=False)
 pairs=[];rng=np.random.default_rng(31105)
 for ds in cfg['datasets']:
  wide=conds[conds.dataset==ds].groupby(['condition','arm']).BAcc_condition.mean().unstack('arm')
  for arm,ref in cfg['evaluation']['primary_pairs']:
   diff=(wide[arm]-wide[ref]).to_numpy();n=len(diff);bs=diff[rng.integers(0,n,size=(10000,n))].mean(1)
   nonzero=diff[diff!=0];pvalue=binomtest(int((nonzero>0).sum()),len(nonzero),.5).pvalue if len(nonzero) else 1.
   pairs.append(dict(dataset=ds,candidate=arm,comparator=ref,delta_pp=float(diff.mean()*100),CI_low_pp=float(np.quantile(bs,.025)*100),CI_high_pp=float(np.quantile(bs,.975)*100),p=pvalue,positive_groups=int((diff>0).sum()),negative_groups=int((diff<0).sum()),groups=n))
 order=sorted(range(len(pairs)),key=lambda i:pairs[i]['p']);prev=0.
 for rank,i in enumerate(order):prev=max(prev,min(1.,pairs[i]['p']*(len(order)-rank)));pairs[i]['p_holm']=prev
 pd.DataFrame(pairs).to_csv(OUT/'paired_effects.csv',index=False)
 if 'load' in conds:
  # Per-load metrics are recomputed from pooled windows, rather than averaging condition F1.
  rows=[]
  for ds in cfg['datasets']:
   table,keep,_,_=host_data(ds);lc='load' if 'load' in table else 'target_appliance' if 'target_appliance' in table else None
   if not lc:continue
   for arm in cfg['arms']:
    for seed in cfg['seeds']:
     z=np.load(OUT/f'{ds}_{arm}_s{seed}_OOF.npz');ids=z['indices'];prob=z['probability']
     for load in table.iloc[ids][lc].unique():
      choose=table.iloc[ids][lc].to_numpy()==load;ix=ids[choose]
      rows.append(dict(dataset=ds,arm=arm,seed=seed,load=load,**cycle_metrics(table.iloc[ix].label.to_numpy(),prob[choose],table.iloc[ix].condition_key.to_numpy())))
  pd.DataFrame(rows).to_csv(OUT/'per_load_seed.csv',index=False)
 costs=[]
 import torch
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 for arm in cfg['arms']:
  net=make_model(arm);total=[0];hooks=[]
  def hook(layer,args,result):
   if isinstance(layer,torch.nn.Conv1d):total[0]+=result.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]
   elif isinstance(layer,torch.nn.Linear):total[0]+=result.numel()*layer.in_features
  for layer in net.modules():
   if isinstance(layer,(torch.nn.Conv1d,torch.nn.Linear)):hooks.append(layer.register_forward_hook(hook))
  net.eval()
  with torch.inference_mode():net(torch.zeros(1,9,5,64),torch.ones(1,9),summary=torch.zeros(1,36))
  if arm in ('gain_phase','maxpool_dynamic'):
   # The split F.linear calls preserve zero-initialization parity but bypass
   # the nn.Linear module hook; count their complete164x32 matrix explicitly.
   total[0]+=net.readout_hidden.in_features*net.readout_hidden.out_features
  for h in hooks:h.remove()
  updates={'hierarchy_small':18432,'hierarchy_wide':26496}.get(arm,1152)
  costs.append(dict(arm=arm,parameters=sum(p.numel() for p in net.parameters()),matrix_MAC=total[0],nominal_matrix_FLOPs=2*total[0],LIF_scalar_updates=updates,summary_scalar_ops='separate; not counted as matrix MAC',frontend='unchanged cached native RCR/HF; not included'))
 pd.DataFrame(costs).to_csv(OUT/'model_cost.csv',index=False)
 audits=pd.concat([pd.read_csv(OUT/f'audit_gpu{g}.csv') for g in (1,2,3)],ignore_index=True)
 assert len(audits)==len(json.loads((OUT/'protocol.json').read_text())['jobs'])
 audits.to_csv(OUT/'replay_audit.csv',index=False)
 protocol=json.loads((OUT/'protocol.json').read_text())
 for path,digest in protocol['baseline_sources'].items():assert sha(path)==digest
 receipts=[json.loads(destination(j).joinpath('COMPLETE.json').read_text()) for j in protocol['jobs']]
 save(OUT/'FINAL_CHECK.json',dict(status='passed',model_states=len(receipts),new_fits=sum(not r['reused'] for r in receipts),reused_fits=sum(r['reused'] for r in receipts),
  datasets=protocol['cohorts'],physical_gpus=[1,2,3],GPU0_used=False,baseline_sources_unmodified=True,threshold=.5,
  prediction_max_abs_error=float(audits.prediction_max_abs_error.max()),summary_max_abs_error=float(audits.summary_max_abs_error.max()),all_replay_decisions_equal=bool(audits.decisions_equal.all()),independent_SOTA_claim=False))
 print(summary.to_string(),flush=True)

def run_study(cfg,execute=False):
 assert cfg['datasets']==['self462','zenodo5501527'] and cfg['execution']['physical_gpus']==[1,2,3]
 if not execute:
  print('Four prespecified architectures, native2MHz self462 and native1MHz public; 4/5group folds,3seeds; GPU0 forbidden.');return
 prepare(cfg)
 env0=os.environ.copy();env0['CUDA_VISIBLE_DEVICES']=''
 started=time.perf_counter();logs=[];children=[]
 for gpu in (1,2,3):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
  log=open(OUT/f'gpu{gpu}.log','a');logs.append(log)
  children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(c.poll() is None for c in children):
  complete=len(list((OUT/'runs').glob('*/*/*/COMPLETE.json')))
  save(OUT/'live_status.json',dict(status='running',completed=complete,target=len(json.loads((OUT/'protocol.json').read_text())['jobs']),elapsed_seconds=time.perf_counter()-started,pids=[c.pid for c in children],returncodes=[c.poll() for c in children]))
  if time.perf_counter()-started>cfg['execution']['timeout_seconds']:
   for c in children:
    if c.poll() is None:c.terminate()
   raise TimeoutError('Prespecified study timeout; completed receipts preserved')
  time.sleep(10)
 for log in logs:log.close()
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 save(OUT/'TRAINING_COMPLETE.json',dict(status='complete',seconds=time.perf_counter()-started))
 # Replaying all saved predictions is a distinct validation stage.
 children=[];logs=[]
 for gpu in (1,2,3):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
  log=open(OUT/f'audit_gpu{gpu}.log','a');logs.append(log)
  children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--audit',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(c.poll() is None for c in children):time.sleep(5)
 for log in logs:log.close()
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 collect(cfg);save(OUT/'live_status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--audit',type=int);p.add_argument('--collect',action='store_true');a=p.parse_args()
 if a.worker is not None:worker(a.worker)
 elif a.audit is not None:audit_replay(a.audit)
 elif a.collect:
  import yaml
  collect(yaml.safe_load(CFG.read_text()))
