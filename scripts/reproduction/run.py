from pathlib import Path
import os,sys,json,time,fcntl,traceback,importlib,subprocess
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'outputs/current_correction_20261109'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SELF=ROOT/'outputs/self462_1m_baselines_20261105'
PUB=ROOT/'outputs/map_input_calibration_20261105/data/zenodo5501527'
MODELS={'ArcFENet':'baseline_arc_fenet','HYDRA':'baseline_hydra','Fusion-current':'baseline_arc_fusion'};OLDN={'ArcFENet':'ArcFENet_Baseline','HYDRA':'HYDRA_Baseline','Fusion-current':'ArcFusionNet_Current_Baseline'}
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp'+str(os.getpid()));q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def context(ds):
 folder=SELF if ds=='self462' else PUB;t=pd.read_csv(folder/'cycles.csv');keep=np.arange(len(t)) if ds=='self462' else np.load(PUB/'keep.npy');return folder,t,keep

def prepare(datasets=("self462","zenodo5501527")):
 import torch
 from sklearn.model_selection import GroupKFold
 from arc_model.modules.rcr_dynamic_branch_1mhz.module import RCRDynamicBranch1MHz
 torch.set_num_threads(4);jobs=[]
 for ds,nf in [('self462',4),('zenodo5501527',5)]:
  if ds not in datasets:continue
  folder,t,keep=context(ds);dest=O/'data'/ds;dest.mkdir(parents=True,exist_ok=True);t.to_csv(dest/'cycles.csv',index=False);np.save(dest/'reviewed.npy',keep);splits=[]
  for name in ('rgb.npy','temporal.npy','spectrum.npy','hydra_features.npy'):
   asset=folder/name
   if not asset.is_file():raise FileNotFoundError(asset)
   link=dest/name
   if not link.exists():link.symlink_to(asset)
  for f in range(nf):
   if ds=='self462':tr=np.flatnonzero(t.fold.to_numpy()!=f);te=np.flatnonzero(t.fold.to_numpy()==f)
   else:r=np.load(PUB/f'roles_f{f}.npz');tr=r['outer_train'];te=r['test']
   assert np.array_equal(np.sort(tr),np.setdiff1d(keep,te));splits.append(dict(train=tr.tolist(),test=te.tolist()))
   for col in ['record_id','condition_key','source_group']:assert not set(t.iloc[tr][col])&set(t.iloc[te][col])
   inn=list(GroupKFold(2).split(tr,t.iloc[tr].label,t.iloc[tr].condition_key))
   for model in MODELS:
    for part in [-1,0,1]:
     fit,val=(tr,te) if part==-1 else (tr[inn[part][0]],tr[inn[part][1]])
     for col in ['record_id','condition_key','source_group']:assert not set(t.iloc[fit][col])&set(t.iloc[val][col])
     d=O/'runs'/ds/model/f'f{f}'/('outer' if part==-1 else f'inner{part}');d.mkdir(parents=True,exist_ok=True);np.savez(d/'indices.npz',train=fit,test=val)
     jobs.append(dict(dataset=ds,model=model,fold=f,part=part,folder=str(d)))
  save(dest/'splits.json',splits)
  if not (dest/'dynamic.npy').exists():
   xx=np.load(folder/('cnn_snn_1m.npy' if ds=='self462' else 'realtime_raw.npy'),mmap_mode='r');mm=np.ones((len(t),9),np.float32) if ds=='self462' else np.load(folder/'realtime_mask.npy');out=np.zeros((len(t),36),np.float32)
   for start in range(0,len(t),1024):out[start:start+1024]=RCRDynamicBranch1MHz.transform(np.array(xx[start:start+1024,:,:5],copy=True),mm[start:start+1024])
   np.save(dest/'dynamic.npy',out)
 save(O/'protocol.json',dict(selected_datasets=list(datasets),datasets={'self462':125051,'zenodo5501527':1357},seed=0,outer_folds=[4,5],inner_group_folds=2,arms=['baseline','calibration_only','RCR_dynamic_no_gate','RCR_dynamic_soft_gate'],strength=1,gate_quantile=.95,gate_softness=.15,feature_count=36,head_epochs=80,head_hyperparameters='existing shared module defaults; no outer-test selection',backbone='source-sealed, same inputs/budget; self outer scores reused exactly; inner fits and public outer fits use current cohort only',scope='group-exposed development validation, no new independent confirmation',gpu_authorization='physical GPUs 1,2,3 only',jobs=jobs))
 print('PREPARED',len(jobs),flush=True)

def cnn(j):
 import torch
 torch.set_num_threads(3)
 ds,model,f,part=j['dataset'],j['model'],j['fold'],j['part'];d=Path(j['folder']);src=O/'data'/ds;pkg='arc_model.implementations.'+MODELS[model]
 cfg=importlib.import_module(pkg+'.configuration').load_config();Model=importlib.import_module(pkg+'.network').Model;Loader=importlib.import_module(pkg+'.data_loader').DataLoader;prep=importlib.import_module(pkg+'.preprocessing')
 loader=Loader(src,reviewed_indices=src/'reviewed.npy');ids=np.load(d/'indices.npz');tr,te=ids['train'],ids['test'];seed=f if part==-1 else f+100*(part+1)
 state=prep.fit_state(loader,tr,seed);host,sp=prep.apply_state(loader,state)
 x=torch.as_tensor(np.asarray(host),device='cuda');s=None if sp is None else torch.as_tensor(np.asarray(sp),device='cuda');y=torch.as_tensor(loader.labels,device='cuda');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=Model(**cfg['model_kwargs']).cuda();tc=cfg['model_training_param'];opt=(torch.optim.Adam if tc['optimizer']=='Adam' else torch.optim.AdamW)(net.parameters(),lr=tc['lr'],weight_decay=tc['wd']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'],eta_min=tc['min_lr']) if tc['scheduler']=='cosine' else None;gen=torch.Generator().manual_seed(seed+10000);ti=torch.as_tensor(tr,device='cuda');start=time.time();hist=[]
 for ep in range(tc['epochs']):
  net.train();order=ti[torch.randperm(len(tr),generator=gen).to('cuda')];total=torch.zeros((),device='cuda')
  for ix in order.split(tc['batch_size']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']['autocast']=='bfloat16'):z=net(x[ix],None if s is None else s[ix])
   loss=torch.nn.functional.cross_entropy(z.float(),y[ix]);loss.backward();opt.step();total+=loss.detach()*len(ix)
  if sch:sch.step()
  value=float(total)/len(tr);assert np.isfinite(value);hist.append(value);save(d/'progress.json',dict(epoch=ep+1,epochs=tc['epochs'],elapsed=time.time()-start,gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 net.eval();scores=[]
 with torch.inference_mode():
  for ix in torch.as_tensor(te,device='cuda').split(tc['batch_size']):
   z=net(x[ix],None if s is None else s[ix]);scores.append((z[:,1]-z[:,0]).cpu().numpy())
 torch.save(net.state_dict(),d/'model.pt');np.savez(d/'preprocessing.npz',**state);np.savez(d/'predictions.npz',indices=te,margin=np.concatenate(scores));save(d/'loss.json',hist);save(d/'COMPLETE.json',dict(status='complete',seconds=time.time()-start,parameters=sum(p.numel()for p in net.parameters()),config=cfg));del x,s,net,opt;torch.cuda.empty_cache()

def hydra(j):
 import torch
 torch.set_num_threads(2)
 from arc_model.implementations.baseline_hydra.network import Model
 from arc_model.implementations.baseline_hydra import features
 if j['dataset']=='self462':
  from dual_ds_hydra_self462_1m_20261105 import solve
  features.linprog=solve
 d=Path(j['folder']);src=O/'data'/j['dataset'];x=np.load(src/'hydra_features.npy',mmap_mode='r');y=pd.read_csv(src/'cycles.csv').label.to_numpy();ids=np.load(d/'indices.npz');net=Model();diag=net.fit(x,y,ids['train'],lambda_l1=.001);np.savez(d/'model.npz',**net.state);np.savez(d/'predictions.npz',indices=ids['test'],margin=net.decision_function(x[ids['test']]));save(d/'COMPLETE.json',dict(status='complete',solver=diag))

def reuse(j):
 from scipy.special import logit
 d=Path(j['folder']);src=SELF/'runs'/OLDN[j['model']]/f'f{j["fold"]}_s0';old=np.load(src/'indices.npz');new=np.load(d/'indices.npz')
 for k in ['train','test']:np.testing.assert_array_equal(old[k],new[k])
 p=np.load(src/'predictions.npz');np.testing.assert_array_equal(p['indices'],new['test']);s=p['score'];margin=s if j['model']=='HYDRA' else logit(np.clip(s.astype(float),1e-12,1-1e-12));np.savez(d/'predictions.npz',indices=p['indices'],margin=margin);save(d/'COMPLETE.json',dict(status='reused_exact_outer_predictions',source=str(src),score_conversion='logit of saved probability (clipped only at 1e-12 endpoints)' if j['model']!='HYDRA' else 'unchanged SVM margin'))

def worker(kind):
 if kind=='gpu' and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1','2','3'):raise RuntimeError('Only physical GPUs 1,2,3 authorized; GPU0 prohibited')
 jobs=json.loads((O/'protocol.json').read_text())['jobs']
 # Complete public first for quick full audit, then current full self cohort.
 jobs.sort(key=lambda j:(j['dataset']=='self462',j['fold'],j['part'],j['model']))
 for j in jobs:
  if (j['model']=='HYDRA')!=(kind=='cpu'):continue
  d=Path(j['folder'])
  with (d/'claim.lock').open('w') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if (d/'COMPLETE.json').exists():continue
   if (d/'FAILED.json').exists():continue
   try:
    print('START',j,flush=True)
    if j['dataset']=='self462' and j['part']==-1:reuse(j)
    elif kind=='cpu':hydra(j)
    else:cnn(j)
    print('DONE',j,flush=True)
   except Exception:save(d/'FAILED.json',dict(traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)

if __name__=='__main__':
 if len(sys.argv)!=2 or sys.argv[1] not in ('prepare','gpu','cpu'):
  raise SystemExit('Usage: run.py {prepare,gpu,cpu}')
 if sys.argv[1]=='prepare':prepare()
 else:worker(sys.argv[1])
