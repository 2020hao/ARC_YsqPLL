"""Exact500-step frozen-FENet preprocessing, CUDA graph execution only."""
import os,time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from matplotlib import colormaps
import _bootstrap as boot
from fenet_vmd_graph_20261030 import VMDGraph
from arc_model.implementations.baseline_arc_fenet.mbwvd_torch import transform as frozen_gpu
from arc_model.implementations.baseline_arc_fenet.mbwvd import mbwvd as frozen_cpu
from run_self462_feature_study_20261105 import sha,save
ROOT=boot.ROOT;OUT=ROOT/'outputs/self462_1m_baselines_20261105'

def prepare_rgb():
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ['1','2','3'];torch.set_num_threads(2)
 if (OUT/'FENET_RGB_COMPLETE.json').exists():return
 waves=np.load(OUT/'fenet_wave.npy',mmap_mode='r');n=len(waves);dest=OUT/'rgb.npy';rgb=np.lib.format.open_memmap(dest,mode='r+' if dest.exists() else 'w+',dtype='float32',shape=(n,3,100,100))
 graph=VMDGraph(batch=64,max_iter=500,alpha=2000.,tol=1e-7,steps=25)
 if not (OUT/'fenet_acceleration_parity.json').exists():
  ix=np.linspace(0,n-1,64).astype(int);x=np.asarray(waves[ix]);a,ca,ia,da=graph.transform(x);b,cb,ib,db=frozen_gpu(x,max_iter=500,tol=1e-7,alpha=2000)
  err=float(abs(a-b).max());np.testing.assert_allclose(a,b,rtol=2e-5,atol=1e-10);np.testing.assert_array_equal(ia,ib);np.testing.assert_array_equal(da,db);np.testing.assert_allclose(ca,cb,rtol=1e-10,atol=1e-7)
  cpu=[]
  for j in [0,17,39,63]:
   im,c,it,done=frozen_cpu(x[j]);cpu.append(dict(index=int(ix[j]),image_max_error=float(abs(im-b[j]).max()),center_max_error=float(abs(c-cb[j]).max()),iteration_cpu=int(it),iteration_gpu=int(ib[j]),converged_cpu=bool(done),converged_gpu=bool(db[j])))
   np.testing.assert_allclose(im,b[j],rtol=5e-5,atol=1e-9)
  save(OUT/'fenet_acceleration_parity.json',dict(status='passed',GPU_same_cap500_max_error=err,CPU_samples=cpu,graph_only_no_numerical_repair=True,original_frozen_gpu_sha256=sha(ROOT/'src/arc_model/implementations/baseline_arc_fenet/mbwvd_torch.py')))
 steps=(n+63)//64;donefile=OUT/'fenet_rgb_done.npy';done=np.load(donefile) if donefile.exists() else np.zeros(steps,bool);diagfile=OUT/'fenet_vmd_diagnostics.npz'
 if diagfile.exists():
  z=np.load(diagfile);centers=z['centers'];iterations=z['iterations'];converged=z['converged']
 else:centers=np.zeros((n,3));iterations=np.zeros(n,int);converged=np.zeros(n,bool)
 tick=time.time()
 for k in range(steps):
  if done[k]:continue
  a,b=k*64,min(n,(k+1)*64);im,c,it,ok=graph.transform(np.asarray(waves[a:b]));images=[]
  for image in im:
   lo,hi=float(image.min()),float(image.max());images.append(colormaps['jet']((image-lo)/max(hi-lo,1e-20))[...,:3].transpose(2,0,1))
  rgb[a:b]=np.asarray(images,dtype='float32');centers[a:b]=c;iterations[a:b]=it;converged[a:b]=ok;done[k]=True
  if k%20==0 or k+1==steps:
   rgb.flush();np.save(donefile,done);np.savez(diagfile,centers=centers,iterations=iterations,converged=converged);save(OUT/'fenet_rgb_progress.json',dict(completed_windows=int(min(n,(k+1)*64)),total=n,seconds=time.time()-tick,nonconvergence_fraction=float((~converged[:b]).mean()),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']));print('RGB',b,n,round(time.time()-tick,1),flush=True)
 assert done.all() and np.isfinite(rgb).all();save(OUT/'FENET_RGB_COMPLETE.json',dict(status='complete',windows=n,max_iter=500,converged=int(converged.sum()),nonconverged=int((~converged).sum()),nonconvergence_fraction=float((~converged).mean()),seconds=time.time()-tick,rgb_sha256=sha(dest),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
