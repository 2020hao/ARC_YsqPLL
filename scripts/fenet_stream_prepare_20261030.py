"""Pipeline already-qualified VMD while original files are being read; no equation changes."""
from pathlib import Path
import argparse,fcntl,os,time,json
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
import _bootstrap
from qualify_fenet_20261030 import ROOT,OUT,DS,save,sha


def worker():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2']
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    from fenet_vmd_graph_20261030 import VMDGraph
    assert json.loads((OUT/'GRAPH_PARITY_2000.json').read_text())['status']=='pass'
    p=OUT/'stream_execution_protocol.json'
    if not p.exists():
        with (OUT/'stream_protocol.lock').open('w') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            if not p.exists():save(p,dict(status='frozen_before_streaming',script_sha256=sha(Path(__file__)),graph_sha256=sha(ROOT/'scripts/fenet_vmd_graph_20261030.py'),preparation_protocol_sha256=sha(OUT/'preparation_protocol.json'),pipeline='same per-record native250k signal, same2000step FP64 VMD/WVD andRGB; stream after each source is complete instead of waiting for all151sources'))
    spec=json.loads((OUT/'preparation_protocol.json').read_text());engine=VMDGraph(batch=64,max_iter=2000);last=-1
    while True:
        completed=0
        for j in spec['jobs']:
            q=OUT/'records'/j['dataset']/f"{j['record_id']:04}"
            if (q/'RGB_COMPLETE.json').exists():completed+=1;continue
            if not (q/'NATIVE_COMPLETE.json').exists():continue
            with (q/'rgb_claim.lock').open('w') as handle:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (q/'RGB_COMPLETE.json').exists():completed+=1;continue
                start=time.perf_counter();native=np.load(q/'native.npy');doc=json.loads((q/'NATIVE_COMPLETE.json').read_text());assert sha(q/'native.npy')==doc['native_sha256']
                rgb=[];centers=[];iters=[];converged=[]
                for a in range(0,len(native),64):
                    image,c,it,v=engine.transform(native[a:a+64]);lo=image.min((1,2),keepdims=True);hi=image.max((1,2),keepdims=True)
                    rgb.append(colormaps['jet']((image-lo)/np.maximum(hi-lo,1e-20))[...,:3].transpose(0,3,1,2).astype('float32'));centers.append(c);iters.append(it);converged.append(v)
                np.save(q/'repaired_rgb.npy',np.concatenate(rgb));np.savez(q/'repaired_diagnostics.npz',centers=np.concatenate(centers),iterations=np.concatenate(iters),converged=np.concatenate(converged))
                save(q/'RGB_COMPLETE.json',dict(status='complete',**j,rows=len(native),native_sha256=doc['native_sha256'],rgb_sha256=sha(q/'repaired_rgb.npy'),stream_protocol_sha256=sha(p),seconds=time.perf_counter()-start,gpu_physical=os.environ['CUDA_VISIBLE_DEVICES']))
                completed+=1;print('STREAM_RGB',j,round(time.perf_counter()-start,3),flush=True)
        if completed!=last:save(OUT/f"stream_gpu{os.environ['CUDA_VISIBLE_DEVICES']}_progress.json",dict(completed=completed,total=len(spec['jobs'])));last=completed
        if completed==len(spec['jobs']):break
        time.sleep(2)


def assemble():
    counts=[]
    for ds in DS:
        tab=pd.read_csv(ROOT/f'outputs/rcr_three_backbones_20261028/data/{ds}/fixed20/cycles.csv');n=len(tab);dest=OUT/'data'/ds/'repaired_rgb.npy'
        rgb=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(n,3,100,100));cover=np.zeros(n,int);centers=np.zeros((n,3));iterations=np.zeros(n,int);conv=np.zeros(n,bool)
        for rid in tab.record_id.unique():
            q=OUT/'records'/ds/f'{int(rid):04}';m=json.loads((q/'RGB_COMPLETE.json').read_text());assert sha(q/'repaired_rgb.npy')==m['rgb_sha256']
            ix=np.load(q/'indices.npy');rgb[ix]=np.load(q/'repaired_rgb.npy');cover[ix]+=1;z=np.load(q/'repaired_diagnostics.npz');centers[ix]=z['centers'];iterations[ix]=z['iterations'];conv[ix]=z['converged']
        assert (cover==1).all();rgb.flush();del rgb
        np.savez(dest.parent/'repaired_vmd_diagnostics.npz',centers=centers,iterations=iterations,converged=conv)
        save(dest.parent/'IMAGES_COMPLETE.json',dict(status='complete',rows=n,rgb_sha256=sha(dest),fresh_original_files=int(tab.record_id.nunique()),original_native_rates_hz=[2000000 if ds=='continuous5m' else 1000000],preparation_protocol_sha256=sha(OUT/'preparation_protocol.json'),stream_protocol_sha256=sha(OUT/'stream_execution_protocol.json'),no_windows_removed_for_nonconvergence=True))
        print('IMAGES_READY',ds,n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--assemble',action='store_true');a=p.parse_args()
    if a.worker:worker()
    elif a.assemble:assemble()
