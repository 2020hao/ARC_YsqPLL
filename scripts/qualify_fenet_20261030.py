"""Versioned ArcFENet qualification: unchanged net, explicit numerical frontend repair."""
from pathlib import Path
import argparse,hashlib,json,os,time,fcntl
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
import _bootstrap as boot
from arc_model.implementations.baseline_arc_fenet.network import Model
from arc_model.evaluation.cycle_metrics import cycle_metrics
from check_baseline_release import verify

ROOT=boot.ROOT;OUT=ROOT/'outputs/fenet_qualification_20261030';DS=['continuous5m','zenodo5501527']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Study already frozen')
    verify();OUT.mkdir(parents=True,exist_ok=True);conf=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text());identities=[];jobs=[];inventory=[]
    for ds in DS:
        c=conf[ds];d=ROOT/c['prepared_roots']['fixed20'];tab=pd.read_csv(d/'cycles.csv');keep=np.load(ROOT/c['reviewed_indices'])
        strat=tab.record_id.astype(str)+'::'+tab.label.astype(str)
        assert strat.iloc[keep].value_counts().min()>=10
        a,te=train_test_split(keep,test_size=.1,stratify=strat.iloc[keep],random_state=20261030)
        tr,val=train_test_split(a,test_size=2/9,stratify=strat.iloc[a],random_state=20261031)
        for v in [tr,val,te]:assert set(tab.iloc[v].record_id)==set(tab.iloc[keep].record_id)
        assert not set(tr)&set(te) and not set(val)&set(te) and not set(tr)&set(val)
        p=OUT/'data'/ds;p.mkdir(parents=True);np.savez(p/'capability_indices.npz',train=tr,validation=val,test=te)
        inventory.append(dict(dataset=ds,N=len(keep),train=len(tr),validation=len(val),test=len(te)))
        for path in [d/'cycles.csv',d/'rgb.npy',ROOT/c['reviewed_indices'],ROOT/c['split_file'],p/'capability_indices.npz']:
            identities.append(dict(path=str(path),sha256=sha(path)))
        for version in ['frozen_v1','repaired_v2']:
            for seed in [0,1,2]:jobs.append(dict(dataset=ds,scope='same_record_capability',version=version,seed=seed,fold=-1))
        for seed in [0,1,2]:
            for fold in range(5):jobs.append(dict(dataset=ds,scope='grouped',version='repaired_v2',seed=seed,fold=fold))
    cfg=yaml.safe_load((ROOT/'configs/models.yaml').read_text())['ArcFENet_Baseline']
    save(OUT/'protocol.json',dict(status='frozen_before_fits',cohorts=inventory,jobs=jobs,model_config=cfg,
        capability='record x class stratified70/20/10; same source recordings deliberately shared; fixed split across seeds; not exact original author data/split reproduction',
        grouped='existing47condition/13record outer5fold, same reviewed38374/1361 cohort; no new split',
        repair='causal native1/2MHz FIR same401/801 taps beta7.5 cutoff118500Hz; model250k5kpoints; VMD2000steps/tol1e-7/alpha2000, same K3 and50/22k/101k centers; original HFDC/DDC/network/20epoch budget unchanged',
        fixed_before_test=True,no_research_RCR_C0_or_added_history=True,selection='numerical signal-fidelity repair fixed before new test outputs, not selected by classification score',
        identities=identities,code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__),ROOT/'scripts/fenet_vmd_graph_20261030.py']],
        baseline_source_immutable=True,physical_gpu=[1,2],never_gpu0=True,
        scope='current-data paper reconstruction qualification; cannot assert original author-data metric reproduction'))
    pd.DataFrame(inventory).to_csv(OUT/'cohort_counts.csv',index=False);print('FROZEN',inventory,len(jobs),flush=True)

def location(job):return OUT/'runs'/job['dataset']/job['scope']/job['version']/f"f{job['fold']}_s{job['seed']}"

def indices(job):
    c=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[job['dataset']]
    if job['scope']=='same_record_capability':
        z=np.load(OUT/'data'/job['dataset']/'capability_indices.npz');return z['train'],z['test'],z['validation']
    keep=np.load(ROOT/c['reviewed_indices']);s=json.loads((ROOT/c['split_file']).read_text())[job['fold']]
    tr=np.asarray(s['train']);te=np.intersect1d(s['test'],keep)
    assert np.array_equal(np.sort(tr),np.setdiff1d(keep,te));return tr,te,None

def fit(job):
    q=location(job);q.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cfg=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())[job['dataset']]
    data=ROOT/cfg['prepared_roots']['fixed20'];tab=pd.read_csv(data/'cycles.csv');tr,te,val=indices(job)
    if job['scope']=='grouped':
        for key in ['record_id','source_group','condition_key']:assert not set(tab.iloc[tr][key])&set(tab.iloc[te][key])
    p=data/'rgb.npy' if job['version']=='frozen_v1' else OUT/'data'/job['dataset']/'repaired_rgb.npy'
    if not p.exists():raise FileNotFoundError('Repaired frontend not ready: '+str(p))
    host=np.load(p,mmap_mode='r')
    for scan_start in range(0,len(host),128):
        if not np.isfinite(host[scan_start:scan_start+128]).all():raise ValueError('Nonfinite model input')
    x=torch.tensor(np.array(host),device='cuda');y=torch.tensor(tab.label.to_numpy(),device='cuda')
    runseed=job['seed']*100+max(job['fold'],0);torch.manual_seed(runseed);net=Model(extra=0).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5);rng=torch.Generator().manual_seed(runseed+10000)
    histories=[];tri=torch.tensor(tr,device='cuda')
    for epoch in range(20):
        net.train();order=tri[torch.randperm(len(tr),generator=rng).to('cuda')];total=torch.zeros((),device='cuda')
        for ids in order.split(128):
            opt.zero_grad(set_to_none=True);logit=net(x[ids]);loss=torch.nn.functional.cross_entropy(logit,y[ids]);loss.backward();opt.step();total+=loss.detach()*len(ids)
        histories.append(dict(epoch=epoch+1,loss=float(total)/len(tr)))
        save(q/'progress.json',dict(epoch=epoch+1,epochs=20,seconds=time.perf_counter()-start,job=job))
    net.eval()
    def predict(ids):
        pieces=[]
        with torch.no_grad():
            for ix in torch.tensor(ids,device='cuda').split(128):pieces.append(net(x[ix]).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(pieces)
    scores=predict(te);train_score=predict(tr)
    np.savez(q/'predictions.npz',indices=te,labels=tab.label.to_numpy()[te],score=scores)
    np.savez(q/'indices.npz',train=tr,test=te,validation=np.asarray([] if val is None else val,dtype=int))
    if val is not None:np.savez(q/'validation.npz',indices=val,score=predict(val),labels=tab.label.to_numpy()[val])
    torch.save(net.state_dict(),q/'model.pt');save(q/'history.json',histories)
    m=cycle_metrics(tab.label.to_numpy()[te],scores,tab.condition_key.to_numpy()[te]);tm=cycle_metrics(tab.label.to_numpy()[tr],train_score,tab.condition_key.to_numpy()[tr])
    save(q/'metrics.json',m);save(q/'train_metrics.json',tm)
    save(q/'COMPLETE.json',dict(status='complete',job=job,seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],protocol_sha256=sha(OUT/'protocol.json'),execution_code_sha256=sha(Path(__file__)),execution_revision_sha256=sha(OUT/'execution_revision_1.json') if (OUT/'execution_revision_1.json').exists() else None,input_sha256=sha(p),model_sha256=sha(q/'model.pt'),predictions_sha256=sha(q/'predictions.npz')))
    print('DONE',job,'BAcc',m['BAcc_condition'],'Accuracy',m['Accuracy'],'seconds',time.perf_counter()-start,flush=True)
    del x,y,net,opt;torch.cuda.empty_cache()

def worker(stage):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ['1','2']
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    spec=json.loads((OUT/'protocol.json').read_text());jobs=spec['jobs']
    for job in jobs:
        if stage=='old_capability' and job['version']!='frozen_v1':continue
        if stage=='repaired' and job['version']!='repaired_v2':continue
        q=location(job);q.mkdir(parents=True,exist_ok=True)
        with (q/'claim.lock').open('w') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (q/'COMPLETE.json').exists():continue
            if (q/'FAILED.json').exists():raise RuntimeError('Prior failed fit requires review: '+str(q))
            try:fit(job)
            except Exception as exc:
                save(q/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise


def run_study(suite,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Freeze first, then explicitGPU1/2 workers')
    print('Dual dataset FENet qualification; old baselines unchanged; separate capability and grouped reports')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',choices=['old_capability','repaired']);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
