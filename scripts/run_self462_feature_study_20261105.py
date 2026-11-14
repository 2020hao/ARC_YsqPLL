"""Grouped discovery and compact direct CNN--SNN validation on reviewed self462."""
import os,json,time,csv,hashlib,argparse,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
import _bootstrap as boot
ROOT=boot.ROOT;OUT=ROOT/'outputs/self462_feature_study_20261105';CONFIG=ROOT/'configs/modules/self462_feature_study_20261105.yaml'
PYTHON=__import__("sys").executable
REFS=['fixed','sogi','rcr'];CANDIDATES={'relative_distribution':[17,18,19,20],'absolute_distribution_delta':[31,32,33,34],'burst_concentration':[15,16],'band_allocation':[13]}
ARMS=['fixed_base','rcr_base','fixed_selected','rcr_selected','rcr_capacity','rcr_dropfirst','rcr_dropsecond']
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aucs(y,x):
 n=int(y.sum());m=len(y)-n
 assert n and m
 ranks=rankdata(x,axis=0);return (ranks[y==1].sum(0)-n*(n+1)/2)/(n*m)
def groupauc(y,x,g):
 keys=np.unique(g);return keys,np.array([aucs(y[g==k],x[g==k]) for k in keys])
def weights(y,g):
 w=np.zeros(len(y))
 for k in np.unique(g):
  for lab in (0,1):
   ids=(g==k)&(y==lab);assert ids.any();w[ids]=1/ids.sum()
 return w/w.mean()
def metric(y,p,g):
 pred=p>=.5;tp=int(sum(pred&(y==1)));tn=int(sum(~pred&(y==0)));fp=int(sum(pred&(y==0)));fn=int(sum(~pred&(y==1)))
 ba=[]
 for k in np.unique(g):
  ids=g==k;yy=y[ids];pp=pred[ids];ba.append(.5*(pp[yy==1].mean()+(~pp[yy==0]).mean()))
 return dict(BAcc=float(np.mean(ba)),Acc=float(np.mean(pred==y)),Recall=tp/max(tp+fn,1),FPR=fp/max(fp+tn,1),Precision=tp/max(tp+fp,1),F1=2*tp/max(2*tp+fp+fn,1),AP=float(average_precision_score(y,p)),ROC_AUC=float(roc_auc_score(y,p)),TP=tp,TN=tn,FP=fp,FN=fn)
def prepare():
 if (OUT/'DATA_COMPLETE.json').exists():return
 mf=pd.read_csv(OUT/'manifest.csv');rng=np.random.default_rng(20261105);foldmap={}
 for load,df in mf.groupby('target_load'):
  keys=sorted(df.physical_condition_key.unique());rng.shuffle(keys)
  foldmap.update({k:i%4 for i,k in enumerate(keys)})
 assert len(foldmap)==156
 save(OUT/'condition_folds.json',foldmap)
 rows=[];features={r:[] for r in REFS};blocks={r:[] for r in REFS};maps={r:[] for r in REFS};masks=[];native={r:[] for r in REFS}
 for j,row in mf.iterrows():
  path=OUT/'data/records'/row.id;zs={r:np.load(path/(r+'.npz')) for r in REFS}
  for r in REFS:
   z=zs[r];assert np.array_equal(z['anchor'],zs['fixed']['anchor']) and np.array_equal(z['label'],zs['fixed']['label'])
   for name in ['features','base','maps']:assert np.isfinite(z[name]).all(),(row.id,r,name)
   assert ((z['support'][:,2]<=z['anchor']+1e-9)&(z['support'][:,1]<=z['anchor']+1e-9)).all()
   features[r].append(z['features']);bb=z['base'][z['hist']].copy();mm=z['maps'][z['hist']].copy()
   # The first recurrent step cannot consult a tenth physical cycle.
   bb[:,0,3:5]=0;mm[:,0,1]=0
   blocks[r].append(bb);maps[r].append(mm);native[r].append(z['native'])
  for a,y in zip(zs['fixed']['anchor'],zs['fixed']['label']):
   rows.append(dict(record_id=row.id,anchor_seconds=a,label=int(y),condition_key=row.physical_condition_key,load=row.target_load,fold=foldmap[row.physical_condition_key]))
  if (j+1)%50==0:print('ASSEMBLED',j+1,flush=True)
 table=pd.DataFrame(rows);table.to_csv(OUT/'cycles.csv',index=False)
 for r in REFS:
  np.save(OUT/(r+'_features.npy'),np.concatenate(features[r]));np.save(OUT/(r+'_base.npy'),np.concatenate(blocks[r]));np.save(OUT/(r+'_maps.npy'),np.concatenate(maps[r]));np.save(OUT/(r+'_native.npy'),np.concatenate(native[r]))
  del features[r],blocks[r],maps[r]
 save(OUT/'DATA_COMPLETE.json',dict(records=len(mf),cycles=len(table),normal=int(sum(table.label==0)),arc=int(sum(table.label==1)),conditions=int(table.condition_key.nunique()),common_anchors=True,guard40_and_history9_same_label=True,fold_counts=table.groupby('fold').size().to_dict(),manifest_sha256=sha(OUT/'manifest.csv')))

def screen():
 if (OUT/'SCREEN_COMPLETE.json').exists():return
 t=pd.read_csv(OUT/'cycles.csv');y=t.label.to_numpy();g=t.condition_key.to_numpy();load=t.load.to_numpy();fold=t.fold.to_numpy();names=json.loads((OUT/'data/feature_names.json').read_text())
 keys,raw=groupauc(y,np.load(OUT/'rcr_features.npy'),g);app=np.array([load[np.flatnonzero(g==k)[0]] for k in keys]);raws={};directions={};allrows=[];selected={};diagnostic=[];probe=[]
 for r in REFS:
  x=np.load(OUT/(r+'_features.npy'));kk,aa=groupauc(y,x,g);assert np.array_equal(kk,keys);raws[r]=aa;orient=np.zeros((len(keys),40));directions[r]=np.zeros((4,40))
  for f in range(4):
   train=fold!=f;test=~train;trcond=np.array([bool(train[np.flatnonzero(g==k)[0]]) for k in keys]);direction=np.where(aa[trcond].mean(0)>=.5,1,-1);directions[r][f]=direction
   orient[~trcond]=.5+(aa[~trcond]-.5)*direction
   # A simple fixed-budget grouped classifier is a probe, not a neural-network result.
   mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-6);z=(x-mean)/std
   clf=LogisticRegression(C=1.,max_iter=2000,solver='lbfgs');clf.fit(z[train],y[train],sample_weight=weights(y[train],g[train]));pp=clf.predict_proba(z[test])[:,1]
   probe.append(dict(reference=r,fold=f,**metric(y[test],pp,g[test])))
  np.save(OUT/(r+'_oriented_auc.npy'),orient)
  for k in range(40):
   rr=dict(reference=r,feature=names[k],mean_oriented_AUC=float(orient[:,k].mean()),condition_fraction_above_chance=float((orient[:,k]>.5).mean()),raw_mean_AUC=float(aa[:,k].mean()))
   for a in np.unique(app):rr[a]=float(orient[app==a,k].mean())
   rr['stable_candidate']=min(rr[a] for a in np.unique(app))>=.6 and rr['condition_fraction_above_chance']>=.7
   allrows.append(rr)
  for i,key in enumerate(keys):
   for k,name in enumerate(names):diagnostic.append(dict(reference=r,condition=key,load=app[i],feature=name,raw_AUC=float(aa[i,k]),train_oriented_AUC=float(orient[i,k])))
 # Selection uses outer training only, RCR channel families with prespecified map definitions.
 for f in range(4):
  trcond=np.array([fold[np.flatnonzero(g==k)[0]]!=f for k in keys]);aa=raws['rcr'][trcond];apps=app[trcond];scores=[]
  for fam,ids in CANDIDATES.items():
   choices=[]
   for k in ids:
    direction=1 if aa[:,k].mean()>=.5 else -1;o=.5+(aa[:,k]-.5)*direction
    perload={str(a):float(o[apps==a].mean()) for a in np.unique(apps)};minimum=min(perload.values());agreement=float((o>.5).mean())
    choices.append(dict(feature=names[k],minimum_load_AUC=minimum,direction_agreement=agreement,score=float(o.mean()),passed=minimum>=.6 and agreement>=.7,per_load=perload))
   passed=[c for c in choices if c['passed']];best=max(passed or choices,key=lambda c:c['minimum_load_AUC'])
   scores.append(dict(family=fam,passed=bool(passed),best=best))
  passed=[s for s in scores if s['passed']];chosen=max(passed,key=lambda s:s['best']['minimum_load_AUC'])['family'] if passed else None
  selected[str(f)]=dict(family=chosen,training_candidates=scores)
 pd.DataFrame(allrows).to_csv(OUT/'feature_stability.csv',index=False);pd.DataFrame(diagnostic).to_csv(OUT/'feature_per_condition.csv',index=False);pd.DataFrame(probe).to_csv(OUT/'linear_probe_metrics.csv',index=False)
 save(OUT/'training_only_selections.json',selected)
 # Cluster resampling and sign flips are by condition, never individual cycles.
 rng=np.random.default_rng(20261105);rngidx=rng.integers(0,len(keys),size=(5000,len(keys)));signs=rng.choice([-1,1],size=(10000,len(keys)));effects=[]
 fx=np.load(OUT/'fixed_oriented_auc.npy')
 for r in ['sogi','rcr']:
  delta=np.load(OUT/(r+'_oriented_auc.npy'))-fx;boots=delta[rngidx].mean(1);perm=signs@delta/len(keys)
  for k,name in enumerate(names):
   mean=float(delta[:,k].mean());ci=np.quantile(boots[:,k],[.025,.975]);pv=(1+sum(abs(perm[:,k])>=abs(mean)-1e-12))/(len(signs)+1)
   effects.append(dict(reference=r,feature=name,delta_AUC=mean,CI_low=float(ci[0]),CI_high=float(ci[1]),p=float(pv)))
 order=np.argsort([r['p'] for r in effects]);last=0
 for rank,i in enumerate(order):last=max(last,min(1,effects[i]['p']*(len(order)-rank)));effects[i]['Holm_p']=last
 pd.DataFrame(effects).to_csv(OUT/'phase_effects.csv',index=False)
 save(OUT/'SCREEN_COMPLETE.json',dict(status='complete',features=40,coordinates=3,outer_folds=4,selection='training_only_each_outer_fold',bootstrap_clusters='physical_condition',comparisons=80))

def inp(ref,arm,fold):
 base=np.load(OUT/(ref+'_base.npy'),mmap_mode='r');mp=np.load(OUT/(ref+'_maps.npy'),mmap_mode='r');sel=json.loads((OUT/'training_only_selections.json').read_text())[str(fold)]['family']
 if arm.endswith('base'):return np.asarray(base)
 if arm.endswith('capacity'):add=base[:,:,1:3].copy()
 elif sel is None:add=np.zeros(base.shape[:2]+(2,64),'float32')
 else:add=mp[:,:,list(CANDIDATES).index(sel)].copy()
 if arm.endswith('dropfirst'):add[:,:,0]=0
 if arm.endswith('dropsecond'):add[:,:,1]=0
 return np.concatenate([base,add],axis=2)

def job(arm,fold,cfg):
 import torch
 from arc_model.implementations.map_input_calibration.midband_network import MidbandPhaseMap
 from arc_model.implementations.phase_event_snn.data import condition_class_weights,training_statistics
 ref='fixed' if arm.startswith('fixed') else 'rcr';dest=OUT/'runs'/arm/f'f{fold}_s0';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 t=pd.read_csv(OUT/'cycles.csv');y=t.label.to_numpy();g=t.condition_key.to_numpy();train=np.flatnonzero(t.fold.to_numpy()!=fold);test=np.flatnonzero(t.fold.to_numpy()==fold)
 assert not set(g[train])&set(g[test]);assert not set(t.record_id.iloc[train])&set(t.record_id.iloc[test])
 x=inp(ref,arm,fold);mean,std=training_statistics(x,train);xp=torch.as_tensor((x-mean[None,None,:,None])/std[None,None,:,None],device='cuda');del x
 mask=torch.ones((len(y),9),device='cuda');target=torch.as_tensor(y,dtype=torch.float32,device='cuda');w=torch.as_tensor(condition_class_weights(y,g,train),device='cuda');tr=torch.as_tensor(train,device='cuda')
 seed=fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=MidbandPhaseMap('base' if arm.endswith('base') else 'mid').cuda();tt=cfg['training'];opt=torch.optim.AdamW(net.parameters(),lr=tt['learning_rate'],weight_decay=tt['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tt['epochs'],eta_min=1e-5);gen=torch.Generator(device='cuda').manual_seed(seed+90000);losses=[];start=time.time()
 for ep in range(tt['epochs']):
  net.train();total=torch.zeros((),device='cuda');parts=list(tr[torch.randperm(len(tr),device='cuda',generator=gen)].split(tt['batch_size']))
  if len(parts[-1])==1:parts[-2]=torch.cat(parts[-2:]);parts.pop()
  for ix in parts:
   opt.zero_grad(set_to_none=True);logit,activity=net(xp[ix],mask[ix],return_activity=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,target[ix],reduction='none')*w[ix]).mean()+tt['spike_activity_penalty']*activity;loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();total+=loss.detach()*len(ix)
  scheduler.step();value=float(total)/len(train);assert np.isfinite(value);losses.append(value);save(dest/'progress.json',dict(epoch=ep+1,epochs=tt['epochs'],loss=value,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 net.eval();pred=[]
 with torch.inference_mode():
  for ids in torch.as_tensor(test,device='cuda').split(512):pred.append(net(xp[ids],mask[ids]).sigmoid().cpu().numpy())
 pp=np.concatenate(pred);np.savez(dest/'predictions.npz',indices=test,label=y[test],probability=pp);np.savez(dest/'preprocessing.npz',train=train,test=test,mean=mean,std=std);torch.save(net.state_dict(),dest/'model.pt');save(dest/'loss.json',losses);save(dest/'metrics.json',metric(y[test],pp,g[test]));save(dest/'COMPLETE.json',dict(status='complete',arm=arm,fold=fold,seed=0,parameters=sum(p.numel() for p in net.parameters()),seconds=time.time()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],hashes={n:sha(dest/n) for n in ['model.pt','predictions.npz','preprocessing.npz','loss.json','metrics.json']}));print('MODEL_COMPLETE',arm,fold,flush=True)
 del net,xp,opt,mask,target,w;torch.cuda.empty_cache()

def worker(part):
 import yaml,torch
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(part+1)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 cfg=yaml.safe_load(CONFIG.read_text());jobs=[(a,f) for f in range(4) for a in ARMS]
 for i,(a,f) in enumerate(jobs):
  if i%3==part:job(a,f,cfg)

def collect():
 t=pd.read_csv(OUT/'cycles.csv');y=t.label.to_numpy();g=t.condition_key.to_numpy();summaries=[];pergroup=[];perload=[];preds={}
 for arm in ARMS:
  pp=np.full(len(t),np.nan);seen=np.zeros(len(t),int)
  for f in range(4):
   dest=OUT/'runs'/arm/f'f{f}_s0';c=json.loads((dest/'COMPLETE.json').read_text())
   for n,h in c['hashes'].items():assert sha(dest/n)==h
   z=np.load(dest/'predictions.npz');ids=z['indices'];assert (t.fold.iloc[ids]==f).all();np.testing.assert_array_equal(y[ids],z['label']);pp[ids]=z['probability'];seen[ids]+=1
  assert (seen==1).all() and np.isfinite(pp).all();preds[arm]=pp;np.save(OUT/(arm+'_OOF.npy'),pp)
  summaries.append(dict(arm=arm,seed=0,**metric(y,pp,g)))
  for key in np.unique(g):
   ix=g==key;pergroup.append(dict(arm=arm,condition=key,**metric(y[ix],pp[ix],g[ix])))
  for load in t.load.unique():
   ix=t.load.eq(load).to_numpy();perload.append(dict(arm=arm,load=load,**metric(y[ix],pp[ix],g[ix])))
 pd.DataFrame(summaries).to_csv(OUT/'CNN_SNN_metrics.csv',index=False);pd.DataFrame(pergroup).to_csv(OUT/'CNN_SNN_per_condition.csv',index=False);pd.DataFrame(perload).to_csv(OUT/'CNN_SNN_per_load.csv',index=False)
 rng=np.random.default_rng(20261105);pg=pd.DataFrame(pergroup);keys=sorted(set(g));effects=[]
 pairs=[('rcr_selected','rcr_base'),('fixed_selected','fixed_base'),('rcr_selected','rcr_capacity'),('rcr_selected','fixed_selected'),('rcr_selected','rcr_dropfirst'),('rcr_selected','rcr_dropsecond')]
 for a,b in pairs:
  da=pg[pg.arm.eq(a)].set_index('condition').loc[keys].BAcc.to_numpy()-pg[pg.arm.eq(b)].set_index('condition').loc[keys].BAcc.to_numpy();boot=da[rng.integers(0,len(da),(5000,len(da)))].mean(1);ci=np.quantile(boot,[.025,.975]);perm=(rng.choice([-1,1],(10000,len(da)))*da).mean(1);pv=(1+sum(abs(perm)>=abs(da.mean())-1e-12))/10001
  effects.append(dict(candidate=a,control=b,delta_BAcc_pp=float(da.mean()*100),CI_low_pp=float(ci[0]*100),CI_high_pp=float(ci[1]*100),p=float(pv)))
 order=np.argsort([r['p'] for r in effects]);last=0
 for rank,i in enumerate(order):last=max(last,min(1,effects[i]['p']*(len(order)-rank)));effects[i]['Holm_p']=last
 pd.DataFrame(effects).to_csv(OUT/'CNN_SNN_paired_effects.csv',index=False)
 save(OUT/'COMPLETE.json',dict(status='complete',fits=28,seeds=[0],outer_folds=4,feature_selection='outer_training_only',common_dataset='self462',warning='single-seed exploratory validation, no public dataset validation; historical source overlap not independent external evidence'))

def run_study(cfg,execute=False):
 OUT.mkdir(exist_ok=True)
 assert sha(cfg['manifest'])==cfg['manifest_sha256'];assert sha(OUT/'manifest.csv')==cfg['manifest_sha256']
 if not execute:print('Ready for extraction, grouped screening and28 controlled CNN-SNN fits');return
 prepare();screen()
 if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',dict(config=cfg,selection_sha256=sha(OUT/'training_only_selections.json'),frozen_before_training=True,identities=[dict(path=str(p),sha256=sha(p)) for p in [CONFIG,Path(__file__),ROOT/'scripts/self462_extract_20261105.py',OUT/'manifest.csv',OUT/'condition_folds.json']]))
 children=[];logs=[]
 for part in range(3):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(part+1);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/f'gpu{part+1}.log','a');logs.append(log);children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--worker',str(part)],env=env,stdout=log,stderr=subprocess.STDOUT))
 while any(c.poll() is None for c in children):
  n=len(list((OUT/'runs').glob('*/*/COMPLETE.json')));save(OUT/'live_status.json',dict(completed=n,total=28,pids=[c.pid for c in children],updated=time.time()));time.sleep(15)
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 for log in logs:log.close()
 collect()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--stage',choices=['prepare','screen','collect']);a=p.parse_args()
 if a.worker is not None:worker(a.worker)
 elif a.stage:globals()[a.stage]()
