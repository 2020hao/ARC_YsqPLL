"""Rebuild the versioned FENet frontend from original native current recordings."""
from pathlib import Path
import argparse,fcntl,json,os,time
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from scipy.signal import firwin,upfirdn
import _bootstrap as boot
from qualify_fenet_20261030 import ROOT,OUT,DS,sha,save
from arc_model.implementations.baseline_arc_fenet.raw_reader import read_current

SOURCE=ROOT/'outputs/continuous_pll_validation_20261027_v2/records'
OLD=ROOT/'outputs/rcr_three_backbones_20261028/data'


def freeze():
    p=OUT/'preparation_protocol.json'
    if p.exists():raise FileExistsError(p)
    jobs=[]
    for ds in DS:
        t=pd.read_csv(OLD/ds/'fixed20/cycles.csv')
        for rid in t.record_id.unique():jobs.append(dict(dataset=ds,record_id=int(rid)))
    save(p,dict(status='frozen_before_native_rebuild',jobs=jobs,original_sources_read=True,raw_reference='NET9770_114CH0 native2MHz and publicC3 native1MHz, C2 annotation-only',
        frontend=dict(taps={'1000000':401,'2000000':801},cutoff_hz=118500,kaiser_beta=7.5,output_fs=250000,signed=True,causal=True),
        numerical=dict(VMD_K=3,initial_centers_hz=[50,22000,101000],alpha=2000,tau=0,tol=1e-7,max_iter=2000,FP64=True),
        rationale='Original sensitive band extends112kHz. New adapter preserves<=112kHz and suppresses>=125kHz; maxiter2000 already assessed on98training-only cases. No new test-score selection.',
        graph='same registered equations, done-state freezing and counter; GPU graph is an execution change only; CPU parity required before transformations',
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__),ROOT/'scripts/fenet_vmd_graph_20261030.py',ROOT/'src/arc_model/implementations/baseline_arc_fenet/mbwvd.py']],
        old_classification_protocol_sha256=sha(OUT/'protocol.json')))
    print('FROZEN_NATIVE',len(jobs),flush=True)


def rebuild_native(part):
    spec=json.loads((OUT/'preparation_protocol.json').read_text())
    for i,job in enumerate(spec['jobs']):
        if i%2!=part:continue
        ds,rid=job['dataset'],job['record_id'];q=OUT/'records'/ds/f'{rid:04}';q.mkdir(parents=True,exist_ok=True)
        if (q/'NATIVE_COMPLETE.json').exists():continue
        tick=time.perf_counter();doc=json.loads((SOURCE/ds/f'{rid:04}'/'COMPLETE.json').read_text());p=Path(doc['source_file']);st=p.stat()
        assert st.st_size==doc['source_size'] and st.st_mtime_ns==doc['source_mtime_ns']
        actual_sha=sha(p);assert actual_sha==doc['source_sha256']
        recording=read_current(p,int(doc['fs']));x=recording.current;fs=recording.sample_rate_hz
        taps=401 if fs==1000000 else 801;h=firwin(taps,118500,fs=fs,window=('kaiser',7.5))
        # Truncate convolution tail: no fabricated future observations.
        low=upfirdn(h,x,down=fs//250000)[:(len(x)+fs//250000-1)//(fs//250000)]
        tab=pd.read_csv(OLD/ds/'fixed20/cycles.csv');g=tab[tab.record_id==rid];waves=[]
        for _,r in g.iterrows():
            assert abs(r.end_seconds-r.start_seconds-.02)<1e-8
            positions=(np.linspace(r.start_seconds,r.end_seconds,5000,endpoint=False)-recording.time_origin_s)*250000
            assert positions.min()>=0 and positions.max()<=len(low)-1
            assert recording.time_origin_s+np.ceil(positions.max())/250000<=r.anchor_seconds+1e-8
            waves.append(np.interp(positions,np.arange(len(low)),low).astype('float32'))
        np.save(q/'native.npy',np.asarray(waves));np.save(q/'indices.npy',g.index.to_numpy())
        save(q/'NATIVE_COMPLETE.json',dict(status='complete',**job,original_source=str(p),source_sha256=actual_sha,source_full_hash_rechecked=True,
            fs=fs,samples=len(x),time_origin=recording.time_origin_s,rows=len(g),native_sha256=sha(q/'native.npy'),protocol_sha256=sha(OUT/'preparation_protocol.json'),seconds=time.perf_counter()-tick))
        print('NATIVE',ds,rid,len(g),round(time.perf_counter()-tick,2),flush=True)


def assemble_native():
    for ds in DS:
        tab=pd.read_csv(OLD/ds/'fixed20/cycles.csv');dest=OUT/'data'/ds/'repaired_native.npy'
        a=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(tab),5000));coverage=np.zeros(len(tab),int)
        for rid in tab.record_id.unique():
            q=OUT/'records'/ds/f'{int(rid):04}';done=json.loads((q/'NATIVE_COMPLETE.json').read_text());assert sha(q/'native.npy')==done['native_sha256']
            ix=np.load(q/'indices.npy');a[ix]=np.load(q/'native.npy');coverage[ix]+=1
        assert (coverage==1).all();a.flush();del a
        save(dest.parent/'NATIVE_COMPLETE.json',dict(status='complete',native_sha256=sha(dest),rows=len(tab),fresh_original_files=int(tab.record_id.nunique()),protocol_sha256=sha(OUT/'preparation_protocol.json')))
        print('ASSEMBLED',ds,len(tab),flush=True)


def graph_smoke():
    from fenet_vmd_graph_20261030 import VMDGraph
    from arc_model.implementations.baseline_arc_fenet import mbwvd as cpu
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    checks=[]
    # Only original fold0 training rows are used for numerical qualification.
    case=[]
    for ds in DS:
        t=pd.read_csv(OLD/ds/'fixed20/cycles.csv');s=json.loads((ROOT/f'outputs/rcr_fundamental_consensus_20261028/detector/data/{ds}/recovery/splits.json').read_text())[0]
        for label in [0,1]:case.append((ds,int(next(i for i in s['train'] if t.iloc[i].label==label))))
    data=np.stack([np.load(OLD/ds/'fixed20/native.npy',mmap_mode='r')[i] for ds,i in case])
    engine=VMDGraph(batch=4,max_iter=500,steps=25);tick=time.perf_counter();maps,c,it,done=engine.transform(data);seconds=time.perf_counter()-tick
    for j,(ds,index) in enumerate(case):
        m,cc,ii,v=cpu.mbwvd(data[j]);np.testing.assert_allclose(maps[j],m,atol=1e-7,rtol=2e-4);np.testing.assert_allclose(c[j],cc,rtol=1e-6,atol=.001)
        assert int(it[j])==ii and bool(done[j])==v
        checks.append(dict(dataset=ds,index=index,max_map_error=float(abs(maps[j]-m).max()),max_center_error=float(abs(c[j]-cc).max()),iterations=ii))
    del engine;torch.cuda.empty_cache()
    fs=250000;t=np.arange(5000)/fs;wave=np.sin(2*np.pi*50*t)+.05*np.sin(2*np.pi*22000*t)+.01*np.sin(2*np.pi*101000*t)
    engine=VMDGraph(batch=4,max_iter=2000,steps=25);tick=time.perf_counter();maps,centers,it,done=engine.transform(np.stack([wave]*4));fulltime=time.perf_counter()-tick
    _,cc,ii,v=cpu.vmd_high(wave,max_iter=2000);np.testing.assert_allclose(centers[0],cc,rtol=1e-6,atol=.001)
    assert abs(centers[0,1]-22000)<25 and abs(centers[0,2]-101000)<25
    save(OUT/'GRAPH_PARITY.json',dict(status='pass',case_selection='4 fold0 training rows, no test-based selection',checks=checks,real_four_map_seconds=seconds,synthetic_seconds=fulltime,synthetic_centers=centers[0].tolist(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],graph_code_sha256=sha(ROOT/'scripts/fenet_vmd_graph_20261030.py')))
    print('GRAPH_PARITY_PASS',seconds,fulltime,checks,flush=True)
    del engine;torch.cuda.empty_cache()
    import functools
    original=cpu.vmd_high
    cpu.vmd_high=functools.partial(original,max_iter=2000)
    extended=[];real=[];cases=[]
    for ds in DS:
        d=OLD/ds/'fixed20';table=pd.read_csv(d/'cycles.csv');diag=np.load(d/'vmd_diagnostics.npz');raw=np.load(d/'native.npy',mmap_mode='r')
        tr=json.loads((ROOT/f'outputs/rcr_fundamental_consensus_20261028/detector/data/{ds}/recovery/splits.json').read_text())[0]['train']
        for label in [0,1]:
            i=next(i for i in tr if table.iloc[i].label==label and not diag['converged'][i]);cases.append((ds,i));real.append(raw[i])
    engine=VMDGraph(batch=4,max_iter=2000);maps,centers,iterations,converged=engine.transform(np.asarray(real))
    try:
        for j,(ds,i) in enumerate(cases):
            m,c,it,v=cpu.mbwvd(real[j]);np.testing.assert_allclose(maps[j],m,atol=1e-7,rtol=2e-4);np.testing.assert_allclose(centers[j],c,atol=.001,rtol=1e-6)
            assert int(iterations[j])==it and bool(converged[j])==v
            extended.append(dict(dataset=ds,index=i,iterations=it,converged=bool(v),max_map_error=float(abs(maps[j]-m).max())))
    finally:cpu.vmd_high=original
    save(OUT/'GRAPH_PARITY_2000.json',dict(status='pass',training_only=True,checks=extended))
    print('GRAPH_PARITY_2000_PASS',flush=True)


def transform(part):
    from fenet_vmd_graph_20261030 import VMDGraph
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu==str(part+1)
    parity=json.loads((OUT/'GRAPH_PARITY.json').read_text());assert parity['status']=='pass' and parity['graph_code_sha256']==sha(ROOT/'scripts/fenet_vmd_graph_20261030.py')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.32);engine=VMDGraph(batch=64,max_iter=2000,steps=25)
    count=0
    for ds in DS:
        native=np.load(OUT/'data'/ds/'repaired_native.npy',mmap_mode='r');n=len(native)
        for start in range(0,n,64):
            if (start//64)%2!=part:continue
            q=OUT/'blocks'/ds/f'{start:06}';q.mkdir(parents=True,exist_ok=True)
            if (q/'COMPLETE.json').exists():continue
            tick=time.perf_counter();a,c,it,done=engine.transform(np.array(native[start:start+64]))
            lo=a.min((1,2),keepdims=True);hi=a.max((1,2),keepdims=True);scaled=(a-lo)/np.maximum(hi-lo,1e-20)
            rgb=colormaps['jet'](scaled)[...,:3].transpose(0,3,1,2).astype('float32')
            np.save(q/'rgb.npy',rgb);np.savez(q/'diagnostics.npz',centers=c,iterations=it,converged=done)
            save(q/'COMPLETE.json',dict(status='complete',start=start,stop=start+len(a),seconds=time.perf_counter()-tick,rgb_sha256=sha(q/'rgb.npy'),graph_code_sha256=parity['graph_code_sha256']))
            count+=1
            if count%10==0:print('BLOCK',ds,start,'/',n,round(time.perf_counter()-tick,3),'seconds',flush=True)


def assemble_rgb():
    for ds in DS:
        native=np.load(OUT/'data'/ds/'repaired_native.npy',mmap_mode='r');n=len(native);dest=OUT/'data'/ds/'repaired_rgb.npy'
        rgb=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(n,3,100,100));centers=np.zeros((n,3));iterations=np.zeros(n,int);converged=np.zeros(n,bool)
        for start in range(0,n,64):
            q=OUT/'blocks'/ds/f'{start:06}';m=json.loads((q/'COMPLETE.json').read_text());assert sha(q/'rgb.npy')==m['rgb_sha256']
            stop=m['stop'];rgb[start:stop]=np.load(q/'rgb.npy');z=np.load(q/'diagnostics.npz');centers[start:stop]=z['centers'];iterations[start:stop]=z['iterations'];converged[start:stop]=z['converged']
        rgb.flush();del rgb
        np.savez(dest.parent/'repaired_vmd_diagnostics.npz',centers=centers,iterations=iterations,converged=converged)
        save(dest.parent/'IMAGES_COMPLETE.json',dict(status='complete',rgb_sha256=sha(dest),rows=n,protocol_sha256=sha(OUT/'preparation_protocol.json'),GRAPH_PARITY_sha256=sha(OUT/'GRAPH_PARITY.json'),numerical_nonconvergence_is_retained=True))
        print('IMAGES_READY',ds,n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--native',type=int,choices=[0,1]);p.add_argument('--assemble-native',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--transform',type=int,choices=[0,1]);p.add_argument('--assemble-rgb',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.native is not None:rebuild_native(a.native)
    elif a.assemble_native:assemble_native()
    elif a.smoke:graph_smoke()
    elif a.transform is not None:transform(a.transform)
    elif a.assemble_rgb:assemble_rgb()
