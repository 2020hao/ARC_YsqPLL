"""Locked-parent single input frontend: direct hot-swap and budget-matched retrain.

Only seed0 is screened. Full grouped folds and fixed baseline budgets are retained.
The model-native interface never gains a branch or features. Raw-rate reference
caches are bound to original source identities; baseline release is read-only.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import gc
import fcntl
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
from scipy.signal import firwin, upfirdn
import yaml
import _bootstrap as boot
from arc_model.evaluation.cycle_metrics import cycle_metrics

ROOT=boot.ROOT
OUT=ROOT/'outputs/locked_routes_20261030/plugin'
PREP=ROOT/'outputs/rcr_three_backbones_20261028/data'
RAW=ROOT/'outputs/continuous_pll_validation_20261027_v2/records'
PAPER=ROOT/'outputs/paper_baseline_completion_20261029/data'
DS=('continuous5m','zenodo5501527')
PACKAGES={'ArcFENet':'plugin_arc_fenet','HYDRA':'plugin_hydra','ArcFusionNet':'plugin_arc_fusion'}
IDS={'ArcFENet':'B_FENET_fixed20_v1','HYDRA':'B_HYDRA_FD_fixed20_v1','ArcFusionNet':'B_FUSION_CURRENT_fixed20_v2'}
CONFIG_KEYS={'ArcFENet':'ArcFENet_Plugin','HYDRA':'HYDRA_Plugin','ArcFusionNet':'ArcFusionNet_Current_Plugin'}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)


def config(model):return yaml.safe_load((ROOT/'configs/models.yaml').read_text())[CONFIG_KEYS[model]]


def package(model,part):return importlib.import_module('arc_model.implementations.'+PACKAGES[model]+'.'+part)


def cohort(ds):
 d=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[ds]
 loader=package('ArcFENet','data_loader').DataLoader(PREP/ds/'fixed20',split_file=ROOT/d['split_file'],reviewed_indices=ROOT/d['reviewed_indices'])
 return loader.table,loader.reviewed,[loader.indices(f) for f in range(5)]


def prepare_record(job):
 ds,rid=job;dest=OUT/'data'/ds/'records'/f'{rid:04}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 src=RAW/ds/f'{rid:04}';meta=json.loads((src/'COMPLETE.json').read_text());fs=int(meta['fs']);source=Path(meta['source_file']);st=source.stat();assert st.st_size==meta['source_size'] and st.st_mtime_ns==meta['source_mtime_ns']
 prior={int(r['record_id']):r for r in json.loads((PAPER/ds/'record_receipts.json').read_text())}[rid]
 rawsha=sha(src/'raw_current.npy');assert rawsha==prior['raw_channel_cache_sha256']
 raw=np.load(src/'raw_current.npy',mmap_mode='r');low=upfirdn(firwin(801 if fs==2000000 else 401,20000,fs=fs,window=('kaiser',7.5)),np.abs(raw),down=fs//50000)
 t0=0.
 if ds=='zenodo5501527':
  with source.open() as f:t0=float(f.readline().split()[0])
 clock=pd.read_csv(src/'recovery_clock.csv');parts={int(k):(g.estimate_time_s.to_numpy(),g.phase_rad.to_numpy()) for k,g in clock[clock.phase_valid].groupby('reference_segment')}
 tab=pd.read_csv(PREP/ds/'recovery/cycles.csv');tab=tab[tab.record_id==rid];fixed=pd.read_csv(PREP/ds/'fixed20/cycles.csv').loc[tab.index];expected=np.load(PAPER/ds/'rectified50k.npy',mmap_mode='r');aligned=[];maxerr=0.;max_required=-np.inf;low_grid=np.arange(len(low))
 for i,q in tab.iterrows():
  a=round((q.start_seconds-t0)*fs)/fs;b=round((q.end_seconds-t0)*fs)/fs
  if q.kind=='pll':
   tx,px=parts[int(q.segment)];grid=np.interp(np.linspace(np.interp(a,tx,px),np.interp(b,tx,px),1001),px,tx)
  else:grid=np.linspace(a,b,1001)
  p=grid[:-1]*50000;assert np.min(p)>=0 and np.max(p)<len(low)
  # Linear interpolation only consumes filtered samples available at the anchor.
  required=np.ceil(p.max())/50000+t0;assert required<=q.anchor_seconds+1e-9
  max_required=max(max_required,required-q.anchor_seconds);aligned.append(np.interp(p,low_grid,low))
  f=fixed.loc[i];positions=(np.linspace(f.start_seconds,f.end_seconds,1000,endpoint=False)-t0)*50000
  z=np.asarray(np.interp(positions,low_grid,low),np.float32)
  err=float(abs(z-expected[i]).max());maxerr=max(err,maxerr)
 assert maxerr==0.,(ds,rid,maxerr)
 waves=np.asarray(aligned,np.float32);np.save(dest/'aligned_rectified50k.npy',waves);np.save(dest/'indices.npy',tab.index.to_numpy())
 save(dest/'COMPLETE.json',dict(dataset=ds,record_id=rid,status='complete',native_fs=fs,original_source=str(source),original_source_sha256=meta['source_sha256'],raw_channel_cache_sha256=rawsha,prior_full_source_identity_receipt=str(src/'COMPLETE.json'),original_size_mtime_checked=True,fresh_full_original_file_hash=False,rectify_before_FIR_and_downsample=True,fixed20_rebuild_max_error=maxerr,causal_max_required_minus_anchor_s=float(max_required),clock_sha256=sha(src/'recovery_clock.csv'),metadata_sha256=sha(src/'recovery.json'),waves_sha256=sha(dest/'aligned_rectified50k.npy')))
 print('PLUGIN_HYDRA_RAW',ds,rid,flush=True)


def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():raise FileExistsError('Already prepared; no overwrite')
 jobs=[];identities=[]
 for ds in DS:
  table,keep,folds=cohort(ds)
  for model in PACKAGES:
   loader=package(model,'data_loader').DataLoader(PREP/ds/'fixed20',split_file=ROOT/yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[ds]['split_file'],reviewed_indices=ROOT/'outputs/dynamic_restore_20261029'/f'{ds}_indices.npy')
   for f,(tr,te) in enumerate(folds):
    a,b=loader.indices(f);assert np.array_equal(a,tr) and np.array_equal(b,te)
    jobs.append(dict(dataset=ds,model=model,fold=f,seed=0,id=f'{ds}_{model}_f{f}'))
  for p in [PREP/ds/'fixed20/cycles.csv',PREP/ds/'recovery/cycles.csv',ROOT/'outputs/dynamic_restore_20261029'/f'{ds}_indices.npy']:
   identities.append(dict(path=str(p),sha256=sha(p)))
 code=[Path(__file__)]+list((ROOT/'src/arc_model/implementations').glob('plugin_*/*.py'))
 reg=json.loads((ROOT/'outputs/grouped_baseline_standard_20261029/baseline_registry.json').read_text())
 parents=[e for e in reg['entries'] if e['seed']==0];assert len(parents)==30
 for e in parents:assert sha(e['weights_path'])==e['weights_sha256']
 protocol=dict(status='frozen_before_new_fits',date='2026-10-30',scope='seed0 exploratory screen,5 outer grouped folds, both datasets; no new independent holdout',jobs=jobs,parents=parents,identities=identities,code=[dict(path=str(p),sha256=sha(p)) for p in code],configs={m:config(m) for m in PACKAGES},gpu_allowed=[2,3],gpu_reserved=[0,1],arms=['single_direct','single_retrained'],direct='Frozen parent weights/preprocessor; switch model-native input only; exact off parity required',retrained='New weights same initialization seed, epochs,batch,optimizer,precision and architecture; input switch at train and test; disabling retains new weights and cannot recover frozen parent performance',availability='kind==pll uses RCR aligned input; all other windows use same fixed20 input; cohort unchanged',HYDRA='Native abs before causal FIR20kHz and50k downsample, then original RCR signed-current coordinate; fixed20 raw rebuild compared exactly; 1024FD author transform; no signed448 cache',raw_input='Hash-bound full-channel cache from native1/2MHz originals; no fresh full original-file hash is claimed',fixed_model_input={'ArcFENet':'5000@250k,VMD500,MBWVD100x100RGB','HYDRA':'1000@50k,rectified,1024FD','ArcFusionNet':'250@12.5k,current+d1+d2; trainRF32 FFT; current-only'},no_extra_feature_branch=True)
 jobs.sort(key=lambda q:(q['dataset']=='continuous5m',q['model']!='HYDRA',q['model']=='ArcFusionNet',q['fold']))
 protocol['jobs']=jobs;save(OUT/'protocol.json',protocol)
 tasks=[(ds,int(rid)) for ds in DS for rid in pd.read_csv(PREP/ds/'recovery/cycles.csv').record_id.unique()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for _ in pool.map(prepare_record,tasks):pass
 for ds in DS:
  t=pd.read_csv(PREP/ds/'recovery/cycles.csv');x=np.zeros((len(t),1000),np.float32)
  for p in sorted((OUT/'data'/ds/'records').glob('*/COMPLETE.json')):
   d=p.parent;x[np.load(d/'indices.npy')]=np.load(d/'aligned_rectified50k.npy')
  assert np.isfinite(x).all();np.save(OUT/'data'/ds/'aligned_rectified50k.npy',x)
 save(OUT/'PREPARED.json',dict(status='ready',raw_records=len(tasks),fixed20_parity='exact across all native raw rebuilt author-HYDRA windows'))
 print('PLUGIN_PREPARED',len(tasks),flush=True)


def features(ds):
 p=OUT/'data'/ds/'aligned_author_hydra.npy'
 with (OUT/'data'/ds/'features.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not p.exists():
   waves=np.load(OUT/'data'/ds/'aligned_rectified50k.npy');m=package('HYDRA','model').Model(**config('HYDRA')['model_kwargs']);x=m.extract(waves,'cuda');assert x.shape==(len(waves),1024)
   cpu=m.extract(waves[:8],'cpu');np.testing.assert_allclose(x[:8],cpu,rtol=2e-5,atol=.01)
   tmp=p.with_suffix('.tmp.npy');np.save(tmp,x);tmp.replace(p);save(p.with_suffix('.json'),dict(status='complete',wave_sha256=sha(OUT/'data'/ds/'aligned_rectified50k.npy'),features_sha256=sha(p),cpu_gpu_max_error=float(abs(x[:8]-cpu).max()),shape=list(x.shape)))
 return np.load(p)


def state_inputs(model,ds,train,seed,state=None):
 d=PREP/ds;valid=pd.read_csv(d/'recovery/cycles.csv').kind.eq('pll').to_numpy()
 if model=='ArcFENet':return np.load(d/'fixed20/rgb.npy',mmap_mode='r'),None,np.load(d/'recovery/rgb.npy',mmap_mode='r'),None,valid,dict(fit_indices=train)
 if model=='HYDRA':return np.load(PAPER/ds/'author_hydra.npy'),None,features(ds),None,valid,None
 fixed_t=np.load(d/'fixed20/temporal.npy');fixed_s=np.load(d/'fixed20/spectrum.npy');align_t=np.load(d/'recovery/temporal.npy');align_s=np.load(d/'recovery/spectrum.npy')
 if state is None:
  class Loader:pass
  l=Loader();l.temporal=np.where(valid[:,None,None,None],align_t,fixed_t);l.spectrum=np.where(valid[:,None],align_s,fixed_s);l.labels=pd.read_csv(d/'fixed20/cycles.csv').label.to_numpy();state=package(model,'preprocessing').fit_state(l,train,seed)
 def apply(t,s):
  a=(t-state['temporal_mean'])/state['temporal_std'];b=s[:,state['harmonic_indices']];return a,((b-state['spectral_mean'])/state['spectral_std'])[:,None]
 x,s=apply(fixed_t,fixed_s);a,b=apply(align_t,align_s);return x,s,a,b,valid,state


def gpu_tensors(x,s,a,b,valid):
 # FE two full arrays fit the32GB cards; copies avoid readonly tensor aliases.
 return tuple(torch.tensor(np.asarray(v),device='cuda') if v is not None else None for v in [x,s,a,b,valid])


def infer(net,inputs,ids,model,enabled):
 x,s,a,b,v=inputs;prob=[];margin=[];cfg=config(model);batch=cfg['model_training_param']['batch_size']
 with torch.no_grad():
  chunks=np.array_split(ids,int(np.ceil(len(ids)/batch))) if model=='ArcFENet' else [q.cpu().numpy() for q in torch.tensor(ids,device='cuda').split(batch)]
  for chunk in chunks:
   ix=torch.tensor(chunk,device='cuda')
   aligned={'temporal':a[ix],'spectrum':None if b is None else b[ix]}
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']['autocast']=='bfloat16'):
    z=net(x[ix],None if s is None else s[ix],aligned=aligned,enabled=enabled,available=v[ix])
   z=z.float();prob.append(z.softmax(-1)[:,1].cpu().numpy());margin.append((z[:,1]-z[:,0]).cpu().numpy())
 return np.concatenate(margin),np.concatenate(prob)


def parent_infer(net,inputs,ids,model,parent,enabled):
 full=ids
 if model=='ArcFusionNet' and json.loads((Path(parent['folder'])/'COMPLETE.json').read_text()).get('reuse'):
  p=ROOT/'outputs/rcr_three_backbones_20261028/runs'/parent['dataset']/'fixed20/ArcFusionNet'/f"f{parent['fold']}_s0/outer/M0/predictions.npz"
  full=np.load(p)['indices']
 margin,prob=infer(net,inputs,full,model,enabled)
 lookup={int(k):i for i,k in enumerate(full)};view=np.asarray([lookup[int(k)] for k in ids]);return margin[view],prob[view]


def predict_save(p,ids,y,margin,prob,fold):np.savez(p,indices=ids,y=y[ids],margin=margin,probability=prob,fold=np.full(len(ids),fold))


def train_nn(model,ds,tr,te,fold,dest):
 cfg=config(model);raw=cfg['model_training_param'];runseed=fold;x,s,a,b,v,state=state_inputs(model,ds,tr,runseed)
 inputs=gpu_tensors(x,s,a,b,v);del x,s,a,b,v;gc.collect();x,s,a,b,v=inputs
 y=pd.read_csv(PREP/ds/'fixed20/cycles.csv').label.to_numpy();labels=torch.tensor(y,device='cuda');tri=torch.tensor(tr,device='cuda')
 torch.manual_seed(runseed);net=package(model,'model').Model(**cfg['model_kwargs']).cuda();opt=(torch.optim.Adam if raw['optimizer']=='Adam' else torch.optim.AdamW)(net.parameters(),lr=raw['lr'],weight_decay=raw['wd']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,raw['epochs'],eta_min=raw['min_lr']) if raw['scheduler']=='cosine' else None
 rng=torch.Generator(device='cuda' if model=='ArcFusionNet' else 'cpu').manual_seed(runseed+(90000 if model=='ArcFusionNet' else 10000));hist=[];tick=time.perf_counter()
 for epoch in range(raw['epochs']):
  net.train()
  order=tri[torch.randperm(len(tri),device='cuda',generator=rng)] if model=='ArcFusionNet' else torch.tensor(tr[torch.randperm(len(tr),generator=rng).numpy()],device='cuda')
  total=torch.zeros((),device='cuda')
  for ix in order.split(raw['batch_size']):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['precision']['autocast']=='bfloat16'):
    z=net(x[ix],None if s is None else s[ix],aligned={'temporal':a[ix],'spectrum':None if b is None else b[ix]},enabled=True,available=v[ix])
   loss=torch.nn.functional.cross_entropy(z.float(),labels[ix]);loss.backward();opt.step();total+=loss.detach()*len(ix)
  value=float(total)/len(tr);assert np.isfinite(value);hist.append(dict(epoch=epoch+1,loss=value,lr=opt.param_groups[0]['lr']));
  if sch:sch.step()
  save(dest/'progress.json',dict(epoch=epoch+1,epochs=raw['epochs'],seconds=time.perf_counter()-tick,loss=value,gpu_physical=os.environ['CUDA_VISIBLE_DEVICES']))
  if epoch==0 or (epoch+1)%10==0:print('PLUGIN_EPOCH',ds,model,fold,epoch+1,round(time.perf_counter()-tick,1),flush=True)
 net.eval();margin,prob=infer(net,inputs,te,model,True);predict_save(dest/'single_retrained.npz',te,y,margin,prob,fold);off_margin,off_prob=infer(net,inputs,te,model,False);predict_save(dest/'retrained_switch_off_diagnostic.npz',te,y,off_margin,off_prob,fold)
 torch.save(net.state_dict(),dest/'model.pt');np.savez(dest/'preprocessing.npz',**state);save(dest/'history.json',hist)
 result=dict(training_seconds=time.perf_counter()-tick,trainable_parameters=sum(p.numel() for p in net.parameters()),epoch_budget=raw['epochs'],architecture_unchanged=True,retrained_off_is_not_frozen_parent=True)
 del net,opt,sch,inputs,x,s,a,b,v,labels,tri;gc.collect();torch.cuda.empty_cache();return result


def run_job(j):
 model,ds,fold=j['model'],j['dataset'],j['fold'];dest=OUT/'runs'/ds/model/f'f{fold}';dest.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();table,keep,folds=cohort(ds);tr,te=folds[fold];y=table.label.to_numpy();np.savez(dest/'indices.npz',train=tr,test=te)
 spec=json.loads((OUT/'protocol.json').read_text());parent=next(e for e in spec['parents'] if e['dataset']==ds and e['fold']==fold and e['baseline_id']==IDS[model]);assert sha(parent['weights_path'])==parent['weights_sha256'];parent_path=Path(parent['folder']);old=np.load(parent_path/'predictions.npz');assert np.array_equal(te,old['indices']);expected=old['score'] if 'score' in old else old['probability'];assert np.array_equal(y[te],old['labels']);parent_indices=np.load(parent_path/'indices.npz');assert np.array_equal(parent_indices['train'],tr)
 if model=='HYDRA':
  fixed,_,align,_,valid,_=state_inputs(model,ds,tr,fold);net=package(model,'model').Model.load(parent['weights_path']);off=net.decision_function(fixed[te],aligned=align[te],enabled=False,available=valid[te]);np.testing.assert_array_equal(off,expected)
  margin=net.decision_function(fixed[te],aligned=align[te],enabled=True,available=valid[te]);again=net.decision_function(fixed[te],enabled=False);np.testing.assert_array_equal(again,off)
  predict_save(dest/'baseline.npz',te,y,off,1/(1+np.exp(-np.clip(off,-700,700))),fold);predict_save(dest/'single_direct.npz',te,y,margin,1/(1+np.exp(-np.clip(margin,-700,700))),fold)
  effective=np.where(valid[:,None],align,fixed);new=package(model,'model').Model(**config(model)['model_kwargs']);diag=new.fit(effective,y,tr,lambda_l1=.001);margin=new.decision_function(fixed[te],aligned=align[te],enabled=True,available=valid[te]);predict_save(dest/'single_retrained.npz',te,y,margin,1/(1+np.exp(-np.clip(margin,-700,700))),fold)
  np.savez(dest/'model.npz',**new.state);save(dest/'solver.json',diag);result=dict(optimizer='HiGHS mean hinge+.001L1',trainable_parameters=int(new.feature_dim+1),architecture_unchanged=True)
 else:
  state=dict(np.load(parent_path/'preprocessing.npz')) if model=='ArcFusionNet' else None
  x,s,a,b,v,_=state_inputs(model,ds,tr,fold,state);inputs=gpu_tensors(x,s,a,b,v);del x,s,a,b,v;gc.collect();net=package(model,'model').Model(**config(model)['model_kwargs']).cuda();loaded=torch.load(parent['weights_path'],map_location='cpu',weights_only=True);net.load_state_dict(loaded);net.eval();off_margin,off=parent_infer(net,inputs,te,model,parent,False);np.testing.assert_array_equal(off,expected)
  margin,prob=parent_infer(net,inputs,te,model,parent,True);again_margin,again=parent_infer(net,inputs,te,model,parent,False);np.testing.assert_array_equal(again,off)
  predict_save(dest/'baseline.npz',te,y,off_margin,off,fold);predict_save(dest/'single_direct.npz',te,y,margin,prob,fold);del net,inputs;gc.collect();torch.cuda.empty_cache();result=train_nn(model,ds,tr,te,fold,dest)
 save(dest/'COMPLETE.json',dict(status='complete',job=j,parent_baseline_id=IDS[model],parent_weights_sha256=parent['weights_sha256'],protocol_sha256=sha(OUT/'protocol.json'),off_parity_max_error=0.,switch_off_after_on_parity=0.,train=len(tr),test=len(te),gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.perf_counter()-tick,valid_pll_test=int(pd.read_csv(PREP/ds/'recovery/cycles.csv').iloc[te].kind.eq('pll').sum()),predictions={p.name:sha(p) for p in dest.glob('*.npz')},**result))
 print('PLUGIN_DONE',j['id'],round(time.perf_counter()-tick,1),flush=True)


def worker():
 physical=os.environ.get('CUDA_VISIBLE_DEVICES');assert physical in ['2','3'];torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.84);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
 spec=json.loads((OUT/'protocol.json').read_text());assert (OUT/'PREPARED.json').exists()
 for r in spec['code']:assert sha(r['path'])==r['sha256'],r['path']
 (OUT/'claims').mkdir(exist_ok=True)
 for j in spec['jobs']:
  claim=OUT/'claims'/(j['id']+'.json')
  if (OUT/'runs'/j['dataset']/j['model']/f"f{j['fold']}"/'COMPLETE.json').exists():continue
  try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
  except FileExistsError:continue
  with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),gpu=physical,job=j['id'],started=time.time()),f)
  try:run_job(j)
  except Exception as e:save(OUT/'failures'/(j['id']+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),gpu=physical));raise
 print('PLUGIN_WORKER_FINISHED',physical,flush=True)


def run_study(suite,*,execute=False):
 if not execute:return dict(study='locked_routes_20261030_plugin',status='protocol_and_explicit_workers_required')
 raise ValueError('Use --prepare followed by physicalGPU2/3 --worker to preserve source freeze and budget')


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
 if args.prepare:prepare()
 elif args.worker:worker()
 else:parser.error('Select --prepare or --worker')
