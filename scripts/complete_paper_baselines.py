"""Fixed nominal inputs, no research enhancement: reconstructed FENet and paper HYDRA.

Uses frozen current reviewed row IDs and condition-disjoint splits. Never changes
historical weights. Full ArcFusionNet requires unavailable grid voltage and is
not relabelled from the current-only implementation.
"""
import argparse
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import _bootstrap as boot
import numpy as np
import pandas as pd
import torch
from scipy.signal import firwin, upfirdn
from arc_model.models.standalone.arc_fenet import FENet
from arc_model.models.standalone.hydra import SparseScaler
from arc_model.models.standalone.hydra_paper import paper_transform, fit_l1_hinge

ROOT=boot.ROOT
OUT=ROOT/'outputs/paper_baseline_completion_20261029'
CURRENT=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
FIXED=ROOT/'outputs/rcr_three_backbones_20261028/data'
RECORDS=ROOT/'outputs/continuous_pll_validation_20261027_v2/records'
COHORT=ROOT/'outputs/dynamic_restore_20261029'
DATASETS=('continuous5m','zenodo5501527')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(p, v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)


def indices(ds, fold):
    splits=json.loads((CURRENT/'data'/ds/'recovery/splits.json').read_text())
    keep=np.load(COHORT/f'{ds}_indices.npy')
    tr=np.array(splits[fold]['train']);te=np.intersect1d(splits[fold]['test'],keep)
    assert np.isin(tr,keep).all()
    tab=pd.read_csv(FIXED/ds/'fixed20/cycles.csv')
    for col in ['record_id','source_group','condition_key']:
        assert not set(tab.iloc[tr][col])&set(tab.iloc[te][col])
    return tab,tr,te


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True,exist_ok=True);identity=[]
    for ds in DATASETS:
        d=OUT/'data'/ds;d.mkdir(parents=True)
        tab=pd.read_csv(FIXED/ds/'fixed20/cycles.csv')
        current=pd.read_csv(CURRENT/'data'/ds/'recovery/cycles.csv')
        assert np.array_equal(tab.sample_id,current.sample_id) and np.array_equal(tab.label,current.label)
        waves=np.zeros((len(tab),1000),np.float32);receipts=[]
        for rid, g in tab.groupby('record_id',sort=True):
            folder=RECORDS/ds/f'{rid:04}'
            doc=json.loads((folder/'COMPLETE.json').read_text());source=Path(doc['source_file'])
            st=source.stat();assert st.st_size==doc['source_size'] and st.st_mtime_ns==doc['source_mtime_ns']
            raw=np.load(folder/'raw_current.npy',mmap_mode='r');fs=int(doc['fs'])
            # Published order: absolute value of the continuous signal, then windows.
            # Causal anti-alias filtering is the declared local rate adapter. The
            # author's original resampling filter is not specified in the release.
            low=upfirdn(firwin(801 if fs==2000000 else 401,20000,fs=fs,window=('kaiser',7.5)),
                np.abs(raw),down=fs//50000)
            origin=0.
            if ds=='zenodo5501527':
                with source.open() as f:origin=float(f.readline().split()[0])
            for i,r in g.iterrows():
                positions=(np.linspace(r.start_seconds,r.end_seconds,1000,endpoint=False)-origin)*50000
                assert abs((r.end_seconds-r.start_seconds)-.02)<1e-8
                assert positions.min()>=0 and positions.max()<len(low)
                waves[i]=np.interp(positions,np.arange(len(low)),low)
            receipts.append(dict(record_id=int(rid),source=str(source),native_fs=fs,
                original_source_sha256=doc['source_sha256'],source_size=st.st_size,source_mtime_ns=st.st_mtime_ns,
                raw_channel_cache_sha256=sha(folder/'raw_current.npy'),
                check='original size/mtime plus full channel-cache hash; not a fresh full original-file hash'))
        assert np.isfinite(waves).all();np.save(d/'rectified50k.npy',waves)
        save(d/'record_receipts.json',receipts)
        for p in [d/'rectified50k.npy',d/'record_receipts.json',FIXED/ds/'fixed20/rgb.npy',
                  FIXED/ds/'fixed20/cycles.csv',CURRENT/'data'/ds/'recovery/splits.json',COHORT/f'{ds}_indices.npy']:
            identity.append(dict(path=str(p),sha256=sha(p)))
        print('PREPARED',ds,len(tab),flush=True)
    jobs=[dict(dataset=ds,model='HYDRA_paper',fold=f,seed=0) for ds in DATASETS[::-1] for f in range(5)]
    jobs += [dict(dataset=ds,model='ArcFENet_reconstructed',fold=f,seed=s)
        for s in [0,1,2] for ds in DATASETS[::-1] for f in range(5)]
    spec=dict(status='frozen_before_new_fits',datasets=DATASETS,jobs=jobs,gpu_physical=[1,2,3],
        input_sha256=identity,code_sha256=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
            [Path(__file__),ROOT/'src/arc_model/models/standalone/hydra_paper.py',ROOT/'src/arc_model/models/standalone/arc_fenet.py']],
        cohort='current common reviewed guard40 anchors:38374/1361; exact current outer training indices',
        HYDRA=dict(t_ms=20,fs_hz=50000,k=8,g=8,kernel_length=5,stride=1,divisor=1,transform_seed=0,
            lambda_l1=.001,transform_features=1024,loss='mean hinge plus lambda L1, free intercept',
            solver='HiGHS LP,900s limit; failure remains failure; no fallback classifier',
            differences=['group-disjoint 5fold rather than alternating windows and stratified holdouts',
                'published best lambda fixed, not retuned on current tests',
                'local causal20kHz anti-alias; author resampling kernel undisclosed',
                'HiGHS instead of MATLAB SpaRSA; no research PLL, C0 or added HF/history']),
        ArcFENet=dict(seeds=[0,1,2],epochs=20,batch=128,lr=.001,weight_decay=1e-5,optimizer='Adam',
            input='fixed20ms native-rate causal250k, VMD/MBWVD/RGB100x100; native HFDC/DDC preserved',
            precision='FP32 TF32 off; cudnn deterministic',
            differences=['paper reconstruction, not located author code','VMD and fusion unspecified choices remain disclosed in fidelity audit',
                'causal112.5kHz adapter attenuates upper edge; this is inherited declared reconstruction, not an exact signal-processing clone']),
        ArcFusionNet='full original unavailable: no synchronized supply voltage/healthy differential baseline; current-only control separate',
        scope='exploratory existing-data baseline completion; no independent confirmation; no test-based arm selection')
    save(OUT/'protocol.json',spec)


def feature_cache(ds):
    d=OUT/'data'/ds;p=d/'author_hydra.npy'
    with (d/'feature.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not p.exists():
            x=np.load(d/'rectified50k.npy');z=paper_transform(x,'cuda:0');assert z.shape==(len(x),1024)
            # CPU comparison is independent of the device-aware reduction.
            from arc_model.models.standalone.hydra import Hydra
            with torch.no_grad():expected=Hydra(1000,k=8,g=8,kernel_length=5,stride=1,divisor=1,seed=0)(torch.tensor(x[:8,None])).numpy()
            cpu=paper_transform(x[:8],'cpu');np.testing.assert_array_equal(cpu,expected)
            np.testing.assert_allclose(z[:8],expected,rtol=2e-5,atol=.01)
            temp=p.with_suffix('.tmp.npy');np.save(temp,z);temp.replace(p)
    return np.load(p)


def job_run(job):
    ds,model,fold,seed=(job[k] for k in ['dataset','model','fold','seed'])
    dest=OUT/'runs'/ds/model/f'f{fold}_s{seed}'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);start=time.perf_counter();tab,tr,te=indices(ds,fold)
    y=tab.label.to_numpy();np.savez(dest/'indices.npz',train=tr,test=te)
    if model=='HYDRA_paper':
        z=torch.from_numpy(feature_cache(ds));sc=SparseScaler();x=sc.fit_transform(z[tr]).numpy()
        xt=sc.transform(z[te]).numpy();w,b,diag=fit_l1_hinge(x,y[tr])
        score=xt@w+b
        np.savez(dest/'model.npz',weight=w,intercept=b,mu=sc.mu.numpy(),sigma=sc.sigma.numpy(),epsilon=sc.epsilon.numpy())
        save(dest/'solver.json',diag)
    else:
        x=np.load(FIXED/ds/'fixed20/rgb.npy',mmap_mode='r')
        runseed=seed*100+fold;torch.manual_seed(runseed);net=FENet(extra=0).cuda()
        optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5)
        order_rng=torch.Generator().manual_seed(runseed+10000);hist=[]
        for ep in range(20):
            net.train();order=tr[torch.randperm(len(tr),generator=order_rng).numpy()];losses=[]
            for start_id in range(0,len(order),128):
                ids=order[start_id:start_id+128]
                a=torch.tensor(np.array(x[ids]),device='cuda');target=torch.tensor(y[ids],device='cuda')
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(a),target)
                loss.backward();optimizer.step();losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)));save(dest/'progress.json',dict(epoch=ep+1,epochs=20,seconds=time.perf_counter()-start))
        net.eval();parts=[]
        with torch.no_grad():
            for ids in np.array_split(te,np.ceil(len(te)/128).astype(int)):
                parts.append(net(torch.tensor(np.array(x[ids]),device='cuda')).softmax(-1)[:,1].cpu().numpy())
        score=np.concatenate(parts);torch.save(net.state_dict(),dest/'model.pt');save(dest/'loss.json',hist)
        del net,optimizer,x;gc.collect();torch.cuda.empty_cache()
    pred=score>= (0 if model=='HYDRA_paper' else .5)
    np.savez(dest/'predictions.npz',indices=te,labels=y[te],score=score,prediction=pred)
    save(dest/'COMPLETE.json',dict(status='complete',job=job,seconds=time.perf_counter()-start,
        gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],protocol_sha256=sha(OUT/'protocol.json'),
        predictions_sha256=sha(dest/'predictions.npz'),train=len(tr),test=len(te)))
    print('DONE',job,round(time.perf_counter()-start,1),flush=True)


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Prepare/freeze first; explicit workers use physicalGPU1/2/3')
    print('PREFLIGHT: paper HYDRA10fits; reconstructed FENet30fits; both current datasets')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--partition',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    if a.prepare:prepare()
    else:
        assert a.partition in [0,1,2] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
        torch.cuda.set_per_process_memory_fraction(.4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
        spec=json.loads((OUT/'protocol.json').read_text())
        for i,j in enumerate(spec['jobs']):
            if i%3==a.partition:job_run(j)
