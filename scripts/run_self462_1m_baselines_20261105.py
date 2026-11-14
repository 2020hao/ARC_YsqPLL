"""1MHz study with three source-sealed baseline implementations, no research heads."""
import os,time,json,argparse,subprocess,fcntl
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
import _bootstrap as boot
from prepare_self462_1m_baselines_20261105 import OUT,PARENT,ORIGINAL,prepare
from run_self462_feature_study_20261105 import sha,save
ROOT=boot.ROOT;CONFIG=ROOT/'configs/modules/self462_1m_baselines_20261105.yaml';PYTHON=__import__("sys").executable
MODELS=['cnn_snn_1m','cnn_snn_2m_control','ArcFENet_Baseline','HYDRA_Baseline','ArcFusionNet_Current_Baseline']

def metrics(y,s,g,threshold=.5):
 pred=s>=threshold;tp=int(sum(pred&(y==1)));tn=int(sum(~pred&(y==0)));fp=int(sum(pred&(y==0)));fn=int(sum(~pred&(y==1)));ba=[]
 for k in np.unique(g):
  ix=g==k;yy=y[ix];pp=pred[ix];ba.append(.5*(pp[yy==1].mean()+(~pp[yy==0]).mean()))
 den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
 return dict(BAcc=float(np.mean(ba)),Acc=float(np.mean(pred==y)),Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),FPR=fp/max(fp+tn,1),Specificity=tn/max(tn+fp,1),F1=2*tp/max(2*tp+fp+fn,1),AP=float(average_precision_score(y,s)),ROC_AUC=float(roc_auc_score(y,s)),MCC=(tp*tn-fp*fn)/den if den else 0.,TP=tp,TN=tn,FP=fp,FN=fn)

def data():return pd.read_csv(OUT/'cycles.csv')
def indices(f):
 t=data();return t,np.flatnonzero(t.fold.to_numpy()!=f),np.flatnonzero(t.fold.to_numpy()==f)
def folder(m,f):return OUT/'runs'/m/f'f{f}_s0'
def package(m):
 if m=='ArcFENet_Baseline':
  from arc_model.implementations.baseline_arc_fenet.configuration import load_config
  from arc_model.implementations.baseline_arc_fenet.network import Model
  from arc_model.implementations.baseline_arc_fenet.data_loader import DataLoader
  from arc_model.implementations.baseline_arc_fenet.preprocessing import fit_state,apply_state
 elif m=='ArcFusionNet_Current_Baseline':
  from arc_model.implementations.baseline_arc_fusion.configuration import load_config
  from arc_model.implementations.baseline_arc_fusion.network import Model
  from arc_model.implementations.baseline_arc_fusion.data_loader import DataLoader
  from arc_model.implementations.baseline_arc_fusion.preprocessing import fit_state,apply_state
 else:raise ValueError(m)
 return load_config(),Model,DataLoader,fit_state,apply_state

def finish(d,m,f,tr,te,s,history,extra):
 t=data();y=t.label.to_numpy();g=t.condition_key.to_numpy();threshold=0. if m=='HYDRA_Baseline' else .5
 np.savez(d/'predictions.npz',indices=te,labels=y[te],score=s,prediction=s>=threshold);np.savez(d/'indices.npz',train=tr,test=te);save(d/'history.json',history);save(d/'metrics.json',metrics(y[te],s,g[te],threshold))
 files=[p for p in d.iterdir() if p.name in ['model.pt','model.npz','preprocessing.npz','predictions.npz','indices.npz','history.json','metrics.json','resolved_model.yaml','solver.json']]
 save(d/'COMPLETE.json',dict(status='complete',model=m,fold=f,seed=0,threshold=threshold,**extra,hashes={p.name:sha(p) for p in files}));print('DONE',m,f,flush=True)

def cnn(m,f,cfg):
 import torch
 from arc_model.implementations.map_input_calibration.midband_network import MidbandPhaseMap
 from arc_model.implementations.phase_event_snn.data import training_statistics,condition_class_weights
 d=folder(m,f);d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return
 t,tr,te=indices(f);x=np.load(OUT/(m+'.npy'),mmap_mode='r');mean,std=training_statistics(x,tr);xp=torch.as_tensor((x-mean[None,None,:,None])/std[None,None,:,None],device='cuda');mask=torch.ones((len(t),9),device='cuda');target=torch.as_tensor(t.label.to_numpy(),dtype=torch.float32,device='cuda');w=torch.as_tensor(condition_class_weights(t.label.to_numpy(),t.condition_key.to_numpy(),tr),device='cuda');ids=torch.as_tensor(tr,device='cuda');torch.manual_seed(f);torch.cuda.manual_seed_all(f);net=MidbandPhaseMap('base').cuda();cc=cfg['cnn_snn'];opt=torch.optim.AdamW(net.parameters(),lr=cc['learning_rate'],weight_decay=cc['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cc['epochs'],eta_min=1e-5);gen=torch.Generator(device='cuda').manual_seed(f+90000);history=[];tick=time.time()
 for ep in range(cc['epochs']):
  net.train();parts=list(ids[torch.randperm(len(ids),device='cuda',generator=gen)].split(cc['batch_size']))
  if len(parts[-1])==1:parts[-2]=torch.cat(parts[-2:]);parts.pop()
  total=torch.zeros((),device='cuda')
  for ix in parts:
   opt.zero_grad(set_to_none=True);z,activity=net(xp[ix],mask[ix],return_activity=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,target[ix],reduction='none')*w[ix]).mean()+cc['spike_activity_penalty']*activity;loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();total+=loss.detach()*len(ix)
  schedule.step();value=float(total)/len(tr);history.append(value);assert np.isfinite(value);save(d/'progress.json',dict(epoch=ep+1,epochs=cc['epochs'],seconds=time.time()-tick,loss=value))
 net.eval();s=[]
 with torch.inference_mode():
  for ix in torch.as_tensor(te,device='cuda').split(512):s.append(net(xp[ix],mask[ix]).sigmoid().cpu().numpy())
 torch.save(net.state_dict(),d/'model.pt');np.savez(d/'preprocessing.npz',fit_indices=tr,mean=mean,std=std);finish(d,m,f,tr,te,np.concatenate(s),history,dict(parameters=sum(p.numel() for p in net.parameters()),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.time()-tick));del net,opt,xp,mask,target,w;torch.cuda.empty_cache()

def hydra_features():
 import torch
 from arc_model.implementations.baseline_hydra.features import paper_transform
 from arc_model.implementations.baseline_hydra.configuration import load_config
 from arc_model.implementations.baseline_hydra.data_loader import DataLoader
 if (OUT/'HYDRA_FEATURES_COMPLETE.json').exists():return
 cfg=load_config();loader=DataLoader(OUT,reviewed_indices=OUT/'reviewed.npy');x=np.asarray(loader.wave);tick=time.time();z=paper_transform(x,device='cuda',**dict(k=cfg['model_kwargs']['k'],g=cfg['model_kwargs']['g'],kernel_length=cfg['model_kwargs']['kernel_length'],stride=cfg['model_kwargs']['stride'],seed=cfg['model_kwargs']['transform_seed']))
 cpu=paper_transform(x[np.linspace(0,len(x)-1,8).astype(int)],device='cpu');np.testing.assert_allclose(z[np.linspace(0,len(x)-1,8).astype(int)],cpu,rtol=2e-5,atol=.02);assert z.shape==(len(x),1024) and np.isfinite(z).all();np.save(OUT/'hydra_features.npy',z);save(OUT/'HYDRA_FEATURES_COMPLETE.json',dict(status='complete',shape=list(z.shape),cpu_gpu_max_error=float(abs(z[np.linspace(0,len(x)-1,8).astype(int)]-cpu).max()),seconds=time.time()-tick));print('HYDRA_FEATURES_READY',flush=True)

def hydra_fit(f):
 from arc_model.implementations.baseline_hydra.network import Model
 from arc_model.implementations.baseline_hydra.configuration import load_config
 from arc_model.implementations.baseline_hydra.data_loader import DataLoader
 d=folder('HYDRA_Baseline',f);d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return
 import yaml
 cfg=load_config();(d/'resolved_model.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));loader=DataLoader(OUT,reviewed_indices=OUT/'reviewed.npy');tr,te=loader.indices(f);x=np.load(OUT/'hydra_features.npy');net=Model(**cfg['model_kwargs']);save(d/'progress.json',dict(stage='HiGHS exact frozen L1-hinge fitting',time_limit_seconds=900,started=time.time(),pid=os.getpid()));tick=time.time();diag=net.fit(x,loader.labels,tr,lambda_l1=cfg['model_training_param']['lambda_l1']);np.savez(d/'model.npz',**net.state);save(d/'solver.json',diag);s=net.decision_function(x[te]);finish(d,'HYDRA_Baseline',f,tr,te,s,[],dict(parameters=1025,fixed_kernel_coefficients=2560,physical_gpu=None,CPU_solver=True,seconds=time.time()-tick))

def rf_states():
 for f in range(4):
  d=OUT/'rf';d.mkdir(exist_ok=True);dest=d/f'f{f}.npz'
  if dest.exists():continue
  _,_,Loader,fit,_=package('ArcFusionNet_Current_Baseline');loader=Loader(OUT,reviewed_indices=OUT/'reviewed.npy');tr,te=loader.indices(f);state=fit(loader,tr,f);np.savez(dest,**state);print('RF_STATE',f,flush=True)
 save(OUT/'RF_COMPLETE.json',dict(status='complete',folds=4,fit='outer_training_only'))

def baseline(m,f):
 import torch,yaml
 cfg,Model,Loader,fit,apply=package(m);d=folder(m,f);d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return
 (d/'resolved_model.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));loader=Loader(OUT,reviewed_indices=OUT/'reviewed.npy');tr,te=loader.indices(f);state=dict(np.load(OUT/'rf'/f'f{f}.npz')) if m=='ArcFusionNet_Current_Baseline' else fit(loader,tr,f);np.testing.assert_array_equal(state['fit_indices'],tr);host,sp=apply(loader,state)
 # GPU residency changes no values, batch size, optimizer, precision or architecture.
 x=torch.as_tensor(np.asarray(host),device='cuda');s=None if sp is None else torch.as_tensor(np.asarray(sp),device='cuda');target=torch.as_tensor(loader.labels,device='cuda');torch.manual_seed(f);torch.cuda.manual_seed_all(f);net=Model(**cfg['model_kwargs']).cuda();assert net.extra is None;tc=cfg['model_training_param'];Opt=torch.optim.Adam if tc['optimizer']=='Adam' else torch.optim.AdamW;opt=Opt(net.parameters(),lr=tc['lr'],weight_decay=tc['wd']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'],eta_min=tc['min_lr']) if tc['scheduler']=='cosine' else None;gen=torch.Generator().manual_seed(f+10000);ids=torch.as_tensor(tr,device='cuda');history=[];tick=time.time();amp=cfg['precision']['autocast']=='bfloat16'
 for ep in range(tc['epochs']):
  net.train();order=ids[torch.randperm(len(tr),generator=gen).to('cuda')];total=torch.zeros((),device='cuda')
  for ix in order.split(tc['batch_size']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=net(x[ix],None if s is None else s[ix])
   loss=torch.nn.functional.cross_entropy(z.float(),target[ix]);loss.backward();opt.step();total+=loss.detach()*len(ix)
  if schedule:schedule.step()
  value=float(total)/len(tr);assert np.isfinite(value);history.append(value);save(d/'progress.json',dict(epoch=ep+1,epochs=tc['epochs'],seconds=time.time()-tick,loss=value,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 net.eval();score=[]
 # Frozen portable baseline API scores in FP32 after BF16 training for Fusion.
 with torch.inference_mode():
  for ix in torch.as_tensor(te,device='cuda').split(tc['batch_size']):score.append(net(x[ix],None if s is None else s[ix]).softmax(-1)[:,1].cpu().numpy())
 torch.save(net.state_dict(),d/'model.pt');np.savez(d/'preprocessing.npz',**state);finish(d,m,f,tr,te,np.concatenate(score),history,dict(parameters=sum(p.numel() for p in net.parameters()),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.time()-tick,baseline_source_unmodified=True,train_precision=cfg['precision']['autocast'],inference_precision='FP32 portable frozen API'));del net,opt,x,s,target,host,sp,loader;torch.cuda.empty_cache()

def locked_baseline_queue(model):
 for f in range(4):
  d=folder(model,f);d.mkdir(parents=True,exist_ok=True)
  with (d/'claim.lock').open('w') as h:
   try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (d/'COMPLETE.json').exists():continue
   if (d/'FAILED.json').exists():raise RuntimeError('Prior failure needs inspection: '+str(d))
   try:baseline(model,f)
   except Exception as e:save(d/'FAILED.json',dict(error=repr(e)));raise

def gpu_worker(part):
 import torch,yaml
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(part+2);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75 if part==1 else .55);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;cfg=yaml.safe_load(CONFIG.read_text())
 if part==0:
  hydra_features()
  for f in range(4):
   for m in ['cnn_snn_1m','cnn_snn_2m_control']:cnn(m,f,cfg)
 else:
  from fenet_self462_1m_20261105 import prepare_rgb
  prepare_rgb();locked_baseline_queue('ArcFENet_Baseline')
 while not (OUT/'RF_COMPLETE.json').exists():time.sleep(5)
 locked_baseline_queue('ArcFusionNet_Current_Baseline')

def cpu_worker(part):
 while not (OUT/'HYDRA_FEATURES_COMPLETE.json').exists():time.sleep(5)
 for f in range(part,4,2):hydra_fit(f)

def freeze(cfg):
 from check_baseline_release import verify
 verify();inputs=[CONFIG,Path(__file__),ROOT/'scripts/prepare_self462_1m_baselines_20261105.py',ROOT/'scripts/prepare_self462_1m_signals_20261105.py',ROOT/'scripts/fenet_self462_1m_20261105.py',ROOT/'scripts/fenet_vmd_graph_20261030.py',ROOT/'releases/baselines_v1_20261030/SEALED.json',ROOT/'configs/models.yaml',OUT/'cycles.csv',OUT/'splits.json',OUT/'reviewed.npy',OUT/'DATA_COMPLETE.json',OUT/'signals/AA_spec.json']
 for m in ['ArcFENet_Baseline','ArcFusionNet_Current_Baseline']:package(m)
 from arc_model.implementations.baseline_hydra.configuration import load_config
 load_config()
 inputs += [OUT/(a+'.npy') for a in ['cnn_snn_1m','cnn_snn_2m_control','wave','fenet_wave','temporal','spectrum']]
 if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',dict(status='frozen_before_fits',config=cfg,jobs=20,baselines='source sealed,no auxiliary research components',identities=[dict(path=str(p),sha256=sha(p)) for p in inputs]))
 else:
  for p in json.loads((OUT/'protocol.json').read_text())['identities']:assert sha(p['path'])==p['sha256']

def collect():
 t=data();y=t.label.to_numpy();g=t.condition_key.to_numpy();rows=[];conds=[];apps=[];folds=[]
 for m in MODELS:
  s=np.full(len(t),np.nan);seen=np.zeros(len(t),int);threshold=0 if m=='HYDRA_Baseline' else .5
  for f in range(4):
   d=folder(m,f);c=json.loads((d/'COMPLETE.json').read_text())
   for n,h in c['hashes'].items():assert sha(d/n)==h
   z=np.load(d/'predictions.npz');ix=z['indices'];assert (t.fold.iloc[ix]==f).all();np.testing.assert_array_equal(z['labels'],y[ix]);s[ix]=z['score'];seen[ix]+=1;folds.append(dict(model=m,fold=f,**metrics(y[ix],s[ix],g[ix],threshold)))
  assert (seen==1).all() and np.isfinite(s).all();np.save(OUT/(m+'_OOF.npy'),s);rows.append(dict(model=m,threshold=threshold,parameters=c['parameters'],history_cycles=9 if m.startswith('cnn_') else 1,**metrics(y,s,g,threshold)))
  for k in np.unique(g):
   ix=g==k;conds.append(dict(model=m,condition=k,**metrics(y[ix],s[ix],g[ix],threshold)))
  for a in t.load.unique():
   ix=t.load.eq(a).to_numpy();apps.append(dict(model=m,load=a,**metrics(y[ix],s[ix],g[ix],threshold)))
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(conds).to_csv(OUT/'per_condition.csv',index=False);pd.DataFrame(apps).to_csv(OUT/'per_load.csv',index=False);pd.DataFrame(folds).to_csv(OUT/'per_fold.csv',index=False)
 rng=np.random.default_rng(20261105);pg=pd.DataFrame(conds);keys=sorted(set(g));effects=[]
 for b in ['cnn_snn_2m_control','ArcFENet_Baseline','HYDRA_Baseline','ArcFusionNet_Current_Baseline']:
  delta=pg[pg.model.eq('cnn_snn_1m')].set_index('condition').loc[keys].BAcc.to_numpy()-pg[pg.model.eq(b)].set_index('condition').loc[keys].BAcc.to_numpy();boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1);ci=np.quantile(boot,[.025,.975]);perm=(rng.choice([-1,1],(10000,len(delta)))*delta).mean(1);p=(1+sum(abs(perm)>=abs(delta.mean())-1e-12))/10001;effects.append(dict(candidate='cnn_snn_1m',control=b,delta_BAcc_pp=float(delta.mean()*100),CI_low_pp=float(ci[0]*100),CI_high_pp=float(ci[1]*100),p=float(p)))
 order=np.argsort([r['p'] for r in effects]);last=0
 for rank,i in enumerate(order):last=max(last,min(1,effects[i]['p']*(len(order)-rank)));effects[i]['Holm_p']=last
 pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False);save(OUT/'TRAINING_COMPLETE.json',dict(status='complete',fits=20,folds=4,seeds=[0],physical_gpus=[2,3]));print(pd.DataFrame(rows).to_string(index=False),flush=True)

def run_study(cfg,execute=False):
 if not execute:print('20fits:1M CNN-SNN, matched2M control,3frozen baselines;4folds');return
 assert sha(OUT/'manifest.csv')==cfg['manifest_sha256'];prepare(cfg);freeze(cfg);children=[];logs=[]
 for part in [0,1]:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(part+2);env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/f'gpu{part+2}.log','a');logs.append(log);children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--gpu-worker',str(part)],env=env,stdout=log,stderr=subprocess.STDOUT))
 for part in [0,1]:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/f'cpu{part}.log','a');logs.append(log);children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--cpu-worker',str(part)],env=env,stdout=log,stderr=subprocess.STDOUT))
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';log=open(OUT/'rf.log','a');logs.append(log);children.append(subprocess.Popen([PYTHON,str(Path(__file__)),'--rf'],env=env,stdout=log,stderr=subprocess.STDOUT));tick=time.time()
 while any(c.poll() is None for c in children):
  save(OUT/'live_status.json',dict(status='running',completed=len(list((OUT/'runs').glob('*/*/COMPLETE.json'))),target=20,elapsed_seconds=time.time()-tick,pids=[c.pid for c in children],returncodes=[c.poll() for c in children]));time.sleep(15)
 for l in logs:l.close()
 assert all(c.returncode==0 for c in children),[c.returncode for c in children]
 collect();save(OUT/'live_status.json',dict(status='training_complete',completed=20,target=20,elapsed_seconds=time.time()-tick))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu-worker',type=int);p.add_argument('--cpu-worker',type=int);p.add_argument('--rf',action='store_true');p.add_argument('--collect',action='store_true');a=p.parse_args()
 if a.gpu_worker is not None:gpu_worker(a.gpu_worker)
 elif a.cpu_worker is not None:cpu_worker(a.cpu_worker)
 elif a.rf:rf_states()
 elif a.collect:collect()
