"""Group-OOF calibration folded into the sole direct classification layer."""
import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import _bootstrap as boot
from run_dynamic_state_20261105 import host_data,roles,sha,save,batches

ROOT=boot.ROOT
OUT=ROOT/'outputs/folded_calibration_20261105'
PARENT=ROOT/'outputs/phase_hierarchy_20261105'
DYNAMIC=ROOT/'outputs/dynamic_state_20261105'
CFG=ROOT/'configs/modules/folded_calibration_20261105.yaml'
PYTHON='/data/hzy/conda/envs/NILMFormer/bin/python'

def destination(j):return OUT/'runs'/j['dataset']/f'f{j["fold"]}_s{j["seed"]}'
def parent_folder(j):return PARENT/'runs'/j['dataset']/'gain_phase'/f'f{j["fold"]}_s{j["seed"]}'

def inner_roles(j):
 table,_,_,_=host_data(j['dataset']);train,test=roles(j['dataset'],j['fold']);g=table.condition_key.to_numpy()
 keys=np.unique(g[train]);order=np.random.default_rng(205110+j['fold']).permutation(keys)
 output=[]
 for i in range(2):
  selected=set(order[i::2]);val=train[np.isin(g[train],list(selected))];tr=np.setdiff1d(train,val)
  assert not set(g[tr])&set(g[val]) and not set(train)&set(test)
  output.append((tr,val))
 np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in output])),train)
 return output

def freeze(cfg):
 OUT.mkdir(parents=True,exist_ok=True)
 assert json.loads((PARENT/'FINAL_CHECK.json').read_text())['status']=='passed'
 for row in json.loads((PARENT/'protocol.json').read_text())['identities']:
  if Path(row['path']).suffix in ('.npy','.npz','.csv'):
   assert sha(row['path'])==row['sha256'],row['path']
 jobs=[dict(dataset=ds,fold=f,seed=s) for ds in ('zenodo5501527','self462') for s in cfg['seeds'] for f in range(cfg['outer_folds'][ds])]
 inputs=[CFG,Path(__file__),ROOT/'scripts/run_dynamic_state_20261105.py',ROOT/'scripts/run_experiment_suite.py',PARENT/'FINAL_CHECK.json',PARENT/'protocol.json']
 for d in ('folded_calibration','dynamic_state_net','phase_hierarchy_net'):
  inputs+=list((ROOT/'src/arc_model/implementations'/d).glob('*.py'))
 for ds in cfg['datasets']:
  table,keep,x,mask=host_data(ds)
  assert len(keep)==(125051 if ds=='self462' else 1357)
  for f in range(cfg['outer_folds'][ds]):inner_roles(dict(dataset=ds,fold=f))
 for j in jobs:
  d=parent_folder(j)
  inputs += [d/n for n in ['COMPLETE.json','model.pt','preprocessing.npz','indices.npz','predictions.npz']]
 identities=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(inputs)]
 protocol=dict(status='frozen_before_inner_fits',config=cfg,jobs=jobs,identities=identities,inner_fits=len(jobs)*2,outer_models_reused=len(jobs),
  rule='Calibration is fitted only to group-out-of-fold logits within each outer training set; folded into the original final layer; fixed0.5 decisions.',
  selection='No outer-test threshold or coefficient selection. All three arms and seeds reported.',
  exploration='Third prespecified development round; previous group folds have been exposed. Not an independent SOTA confirmation.')
 if (OUT/'protocol.json').exists():
  old=json.loads((OUT/'protocol.json').read_text());assert old['config']==cfg
  for r in old['identities']:assert sha(r['path'])==r['sha256'],r['path']
  return old
 save(OUT/'protocol.json',protocol)
 snapshots=[]
 for row in identities:
  p=Path(row['path'])
  if p.suffix in ('.py','.yaml'):
   target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
   snapshots.append(dict(**row,snapshot=str(target)))
 save(OUT/'SOURCE_SNAPSHOT.json',snapshots)
 return protocol

def summary_statistics(summary,tr):
 mean=np.zeros(36,np.float32);std=np.ones(36,np.float32)
 for lag in range(4):
  valid=tr[summary[tr,32+lag]>.5];cols=slice(8*lag,8*(lag+1))
  if len(valid):mean[cols]=summary[valid,cols].mean(0,dtype=np.float64);std[cols]=np.maximum(summary[valid,cols].std(0,dtype=np.float64),1e-6)
 return mean,std

def inner_fit(j,index,tr,val,cfg):
 import torch
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 from arc_model.implementations.phase_hierarchy_net.augmentation import augment_standardized
 from arc_model.implementations.phase_event_snn.data import training_statistics,condition_class_weights
 d=destination(j)/f'inner{index}';d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():
  receipt=json.loads((d/'COMPLETE.json').read_text())
  for n,h in receipt['hashes'].items():assert sha(d/n)==h
  z=np.load(d/'predictions.npz');np.testing.assert_array_equal(z['indices'],val);return z['logit']
 table,_,x,mask=host_data(j['dataset']);y=table.label.to_numpy();g=table.condition_key.to_numpy();summary=np.load(DYNAMIC/f'{j["dataset"]}_dynamic36.npy',mmap_mode='r')
 mean,std=training_statistics(x,tr);sm,ss=summary_statistics(summary,tr)
 seed=100000+1000*j['seed']+10*j['fold']+index
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 net=DynamicStateNet('maxpool_dynamic').cuda();net.set_statistics(mean,std,sm,ss)
 xp=torch.as_tensor((x-mean[None,None,:,None])/std[None,None,:,None],device='cuda');mp=torch.as_tensor(mask,device='cuda');sp=torch.as_tensor(np.array(summary,copy=True),device='cuda')
 target=torch.as_tensor(y,dtype=torch.float32,device='cuda');weight=torch.as_tensor(condition_class_weights(y,g,tr),device='cuda');ids=torch.as_tensor(tr,device='cuda')
 tc=cfg['training'];opt=torch.optim.AdamW(net.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'],eta_min=tc['min_learning_rate'])
 gen=torch.Generator(device='cuda').manual_seed(seed+90000);auggen=torch.Generator(device='cuda').manual_seed(seed+191105);m=torch.as_tensor(mean,device='cuda');s=torch.as_tensor(std,device='cuda');losses=[];start=time.perf_counter()
 for epoch in range(tc['epochs']):
  net.train();total=torch.zeros((),device='cuda')
  for batch in batches(ids[torch.randperm(len(ids),device='cuda',generator=gen)],512):
   opt.zero_grad(set_to_none=True);xx=augment_standardized(xp[batch],m,s,auggen)
   logits,activity=net(xx,mp[batch],return_activity=True,summary=sp[batch])
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target[batch],reduction='none')*weight[batch]).mean()+.001*activity
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();total+=loss.detach()*len(batch)
  scheduler.step();value=float(total)/len(tr);assert np.isfinite(value);losses.append(value)
  save(d/'progress.json',dict(epoch=epoch+1,epochs=tc['epochs'],loss=value,seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 net.eval();logits=[]
 with torch.inference_mode():
  for batch in torch.as_tensor(val,device='cuda').split(512):logits.append(net(xp[batch],mp[batch],summary=sp[batch]).cpu().numpy())
 result=np.concatenate(logits);torch.save(net.state_dict(),d/'model.pt');np.savez(d/'predictions.npz',indices=val,labels=y[val],logit=result)
 np.savez(d/'indices.npz',train=tr,test=val);np.savez(d/'preprocessing.npz',fit_indices=tr,mean=mean,std=std,summary_mean=sm,summary_std=ss);save(d/'training_loss.json',losses)
 files=['model.pt','predictions.npz','indices.npz','preprocessing.npz','training_loss.json']
 save(d/'COMPLETE.json',dict(status='complete',job=j,inner_fold=index,seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],hashes={n:sha(d/n) for n in files}))
 del net,xp,mp,sp,target,weight,opt,scheduler;gc.collect();torch.cuda.empty_cache();return result

def run_job(j,cfg):
 import torch
 from scipy.special import expit
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 from arc_model.implementations.folded_calibration import fit_bias,fit_affine,fold_to_readout
 from arc_model.evaluation.cycle_metrics import cycle_metrics
 d=destination(j);d.mkdir(parents=True,exist_ok=True);table,keep,x,mask=host_data(j['dataset']);train,test=roles(j['dataset'],j['fold']);oof=np.full(len(table),np.nan)
 for index,(tr,val) in enumerate(inner_roles(j)):oof[val]=inner_fit(j,index,tr,val,cfg)
 assert np.isfinite(oof[train]).all() and not np.isfinite(oof[test]).any()
 np.savez(d/'inner_OOF.npz',indices=train,labels=table.iloc[train].label.to_numpy(),logit=oof[train])
 fits={'folded_bias':fit_bias(oof[train],table.iloc[train].label.to_numpy(),table.iloc[train].condition_key.to_numpy()),'folded_affine':fit_affine(oof[train],table.iloc[train].label.to_numpy(),table.iloc[train].condition_key.to_numpy())}
 assert all(v['success'] for v in fits.values()),fits
 src=parent_folder(j);receipt=json.loads((src/'COMPLETE.json').read_text())
 for n,h in receipt['hashes'].items():assert sha(src/n)==h
 np.testing.assert_array_equal(np.load(src/'indices.npz')['train'],train);np.testing.assert_array_equal(np.load(src/'indices.npz')['test'],test)
 net=DynamicStateNet('maxpool_dynamic').cuda();net.load_state_dict(torch.load(src/'model.pt',map_location='cpu',weights_only=True));net.eval()
 stats=np.load(src/'preprocessing.npz');xp=torch.as_tensor((x[test]-stats['mean'][None,None,:,None])/stats['std'][None,None,:,None],device='cuda');mp=torch.as_tensor(mask[test],device='cuda');sp=torch.as_tensor(np.array(np.load(DYNAMIC/f'{j["dataset"]}_dynamic36.npy',mmap_mode='r')[test],copy=True),device='cuda')
 arms={'uncalibrated':{'scale':1.,'bias':0.},**fits};results={};fold_error=0.
 with torch.inference_mode():
  raw=np.concatenate([net(xp[ix],mp[ix],summary=sp[ix]).cpu().numpy() for ix in torch.arange(len(test),device='cuda').split(512)])
  old=np.load(src/'predictions.npz')['probability'];np.testing.assert_allclose(expit(raw),old,atol=1e-6,rtol=0)
  for arm,fit in arms.items():
   current=fold_to_readout(net,fit['scale'],fit['bias'])
   logit=np.concatenate([current(xp[ix],mp[ix],summary=sp[ix]).cpu().numpy() for ix in torch.arange(len(test),device='cuda').split(512)])
   expected=fit['scale']*raw+fit['bias'];fold_error=max(fold_error,float(np.abs(logit-expected).max()))
   np.testing.assert_allclose(logit,expected,atol=5e-6,rtol=2e-6)
   prob=expit(logit);np.savez(d/f'{arm}_predictions.npz',indices=test,labels=table.iloc[test].label.to_numpy(),probability=prob,logit=logit)
   torch.save(current.state_dict(),d/f'{arm}_model.pt');results[arm]=cycle_metrics(table.iloc[test].label.to_numpy(),prob,table.iloc[test].condition_key.to_numpy())
 save(d/'calibration.json',fits);save(d/'metrics.json',results)
 files=['inner_OOF.npz','calibration.json','metrics.json']+[f'{a}_{kind}' for a in arms for kind in ('predictions.npz','model.pt')]
 save(d/'COMPLETE.json',dict(status='complete',job=j,parameters=8249,extra_inference_parameters=0,threshold=.5,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],folded_logit_max_abs_error=fold_error,source_outer=str(src),hashes={n:sha(d/n) for n in files}))
 del net,xp,mp,sp;gc.collect();torch.cuda.empty_cache();print('COMPLETE',j,'calibration',[(k,round(v['scale'],3),round(v['bias'],3)) for k,v in fits.items()],flush=True)

def worker(gpu):
 import torch
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu) and gpu in (1,2,3)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.45);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 p=json.loads((OUT/'protocol.json').read_text())
 for j in p['jobs']:
  d=destination(j);d.mkdir(parents=True,exist_ok=True)
  with (d/'claim.lock').open('w') as h:
   try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (d/'COMPLETE.json').exists():continue
   if (d/'FAILED.json').exists():raise RuntimeError('Preserved failure needs inspection: '+str(d))
   try:run_job(j,p['config'])
   except Exception as e:save(d/'FAILED.json',dict(job=j,error=repr(e)));raise

def audit_replay(gpu):
 import torch
 from scipy.special import expit
 from arc_model.implementations.dynamic_state_net import DynamicStateNet
 assert gpu in (1,2,3) and os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 protocol=json.loads((OUT/'protocol.json').read_text());rows=[]
 for k,j in enumerate(protocol['jobs']):
  if k%3!=gpu-1:continue
  d=destination(j);table,keep,x,mask=host_data(j['dataset']);summary=np.load(DYNAMIC/f'{j["dataset"]}_dynamic36.npy',mmap_mode='r')
  sets=[(d/f'inner{i}','model.pt','predictions.npz') for i in (0,1)]+[(d,f'{arm}_model.pt',f'{arm}_predictions.npz') for arm in protocol['config']['arms']]
  for folder,state_name,pred_name in sets:
   saved=np.load(folder/pred_name);ids=saved['indices'];stats=np.load(folder/'preprocessing.npz') if folder!=d else np.load(parent_folder(j)/'preprocessing.npz')
   net=DynamicStateNet('maxpool_dynamic').cuda();net.load_state_dict(torch.load(folder/state_name,map_location='cpu',weights_only=True));net.eval();values=[]
   with torch.inference_mode():
    for batch in np.array_split(ids,max(1,int(np.ceil(len(ids)/512)))):
     xp=torch.as_tensor((x[batch]-stats['mean'][None,None,:,None])/stats['std'][None,None,:,None],device='cuda')
     mp=torch.as_tensor(mask[batch],device='cuda');sp=torch.as_tensor(np.array(summary[batch],copy=True),device='cuda')
     values.append(net(xp,mp,summary=sp).cpu().numpy())
   logits=np.concatenate(values)
   np.testing.assert_allclose(logits,saved['logit'],atol=4e-6,rtol=1e-6)
   if 'probability' in saved:
    prob=expit(logits);np.testing.assert_allclose(prob,saved['probability'],atol=2e-6,rtol=0)
    assert np.array_equal(prob>=.5,saved['probability']>=.5)
   rows.append(dict(**j,model=str(folder/state_name),prediction_logit_max_abs_error=float(abs(logits-saved['logit']).max()),decisions_equal=True))
   del net;torch.cuda.empty_cache()
 pd.DataFrame(rows).to_csv(OUT/f'audit_gpu{gpu}.csv',index=False)

def collect(cfg):
 from arc_model.evaluation.cycle_metrics import cycle_metrics
 from scipy.stats import binomtest
 rows=[];groups=[];calibrations=[]
 for ds in cfg['datasets']:
  table,keep,_,_=host_data(ds);y=table.label.to_numpy();g=table.condition_key.to_numpy()
  for arm in cfg['arms']:
   for seed in cfg['seeds']:
    p=np.full(len(table),np.nan);seen=np.zeros(len(table),int)
    for f in range(cfg['outer_folds'][ds]):
     j=dict(dataset=ds,fold=f,seed=seed);d=destination(j);r=json.loads((d/'COMPLETE.json').read_text())
     for n,h in r['hashes'].items():assert sha(d/n)==h
     tr,te=roles(ds,f);inner=np.load(d/'inner_OOF.npz');np.testing.assert_array_equal(inner['indices'],tr)
     for ii,(it,iv) in enumerate(inner_roles(j)):
      saved=np.load(d/f'inner{ii}/indices.npz');np.testing.assert_array_equal(saved['train'],it);np.testing.assert_array_equal(saved['test'],iv)
      assert not set(table.iloc[it].condition_key)&set(table.iloc[iv].condition_key)
     z=np.load(d/f'{arm}_predictions.npz');ids=z['indices'];np.testing.assert_array_equal(ids,te);p[ids]=z['probability'];seen[ids]+=1
     if arm!='uncalibrated':calibrations.append(dict(**j,arm=arm,**json.loads((d/'calibration.json').read_text())[arm]))
    assert (seen[keep]==1).all()
    rows.append(dict(dataset=ds,arm=arm,seed=seed,**cycle_metrics(y[keep],p[keep],g[keep])))
    np.savez(OUT/f'{ds}_{arm}_s{seed}_OOF.npz',indices=keep,labels=y[keep],probability=p[keep])
    for key in np.unique(g[keep]):
     ix=keep[g[keep]==key];groups.append(dict(dataset=ds,arm=arm,seed=seed,condition=key,**cycle_metrics(y[ix],p[ix],g[ix])))
 frame=pd.DataFrame(rows);conditions=pd.DataFrame(groups)
 metrics=[n for n in cycle_metrics([0,1],[.1,.9],['g','g']) if n not in ('N','groups')]
 summary=frame.groupby(['dataset','arm'])[metrics].agg(['mean','std']);summary.columns=['_'.join(k) for k in summary.columns]
 frame.to_csv(OUT/'seed_metrics.csv',index=False);summary.reset_index().to_csv(OUT/'main_results.csv',index=False);conditions.to_csv(OUT/'per_condition_seed.csv',index=False)
 pd.DataFrame(calibrations).to_csv(OUT/'calibration_coefficients.csv',index=False)
 effects=[];rng=np.random.default_rng(51103)
 for ds in cfg['datasets']:
  wide=conditions[conditions.dataset==ds].groupby(['condition','arm']).BAcc_condition.mean().unstack('arm')
  for cand,ref in cfg['evaluation']['primary_pairs']:
   diff=(wide[cand]-wide[ref]).to_numpy();n=len(diff);draw=diff[rng.integers(n,size=(10000,n))].mean(1);nz=diff[diff!=0]
   pv=binomtest(int((nz>0).sum()),len(nz),.5).pvalue if len(nz) else 1.
   effects.append(dict(dataset=ds,candidate=cand,comparator=ref,delta_pp=float(diff.mean()*100),CI_low_pp=float(np.quantile(draw,.025)*100),CI_high_pp=float(np.quantile(draw,.975)*100),p=pv,positive_groups=int((diff>0).sum()),negative_groups=int((diff<0).sum())))
 prev=0.
 for rank,i in enumerate(sorted(range(len(effects)),key=lambda i:effects[i]['p'])):prev=max(prev,min(1.,effects[i]['p']*(len(effects)-rank)));effects[i]['p_holm']=prev
 pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False)
 protocol=json.loads((OUT/'protocol.json').read_text());snap=json.loads((OUT/'SOURCE_SNAPSHOT.json').read_text())
 for r in snap:assert sha(r['snapshot'])==r['sha256']
 audits=pd.concat([pd.read_csv(OUT/f'audit_gpu{gpu}.csv') for gpu in (1,2,3)],ignore_index=True)
 assert len(audits)==5*len(protocol['jobs']) and audits.decisions_equal.all()
 audits.to_csv(OUT/'replay_audit.csv',index=False)
 save(OUT/'FINAL_CHECK.json',dict(status='passed',outer_models_reused=len(protocol['jobs']),inner_models_trained=2*len(protocol['jobs']),three_output_versions=3,zero_added_inference_parameters=True,physical_gpus=[1,2,3],GPU0_used=False,group_OOF_calibration=True,outer_test_fit=False,threshold=.5,model_prediction_replays=len(audits),replay_logit_max_abs_error=float(audits.prediction_logit_max_abs_error.max()),all_replay_decisions_equal=True,independent_SOTA_claim=False))
 print(summary.to_string(),flush=True)

def run_study(cfg,execute=False):
 assert cfg['datasets']==['self462','zenodo5501527'] and cfg['execution']['physical_gpus']==[1,2,3]
 if not execute:print('2inner group fits x27outer models; bias/affine folded into sole final layer; GPU0 forbidden.');return
 p=freeze(cfg);children=[];logs=[];start=time.perf_counter()
 for gpu in (1,2,3):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/f'gpu{gpu}.log','a');logs.append(log)
  children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(c.poll() is None for c in children):
  save(OUT/'live_status.json',dict(status='running',completed=len(list((OUT/'runs').glob('*/*/COMPLETE.json'))),target=len(p['jobs']),elapsed_seconds=time.perf_counter()-start,pids=[c.pid for c in children],returncodes=[c.poll() for c in children]))
  if time.perf_counter()-start>cfg['execution']['timeout_seconds']:
   for c in children:
    if c.poll() is None:c.terminate()
   raise TimeoutError('Prespecified calibration budget exceeded')
  time.sleep(10)
 for l in logs:l.close()
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 children=[];logs=[]
 for gpu in (1,2,3):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/f'audit_gpu{gpu}.log','a');logs.append(log)
  children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--audit',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(c.poll() is None for c in children):time.sleep(5)
 for l in logs:l.close()
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 collect(cfg);save(OUT/'live_status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--audit',type=int);p.add_argument('--collect',action='store_true');a=p.parse_args()
 if a.worker is not None:worker(a.worker)
 elif a.audit is not None:audit_replay(a.audit)
 elif a.collect:
  import yaml
  collect(yaml.safe_load(CFG.read_text()))
