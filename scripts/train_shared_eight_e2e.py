"""From-scratch shared-eight experiment: original/fixed8/phase8, both datasets."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import _bootstrap as boot
import numpy as np
import pandas as pd
import torch
from torch import nn
from arc_model.data.arc_fenet_loader import ArcFENetLoader
from arc_model.data.arc_fusion_loader import ArcFusionLoader
from arc_model.data.hydra_loader import HydraLoader
from arc_model.models.standalone.arc_fenet import FENet
from arc_model.models.standalone.arc_fusion import FusionNet
from arc_model.models.standalone.hydra import SparseScaler
from arc_model.methods.phase_representation.shared_features import shared_eight, NAMES

ROOT=boot.ROOT
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
FEATURES=ROOT/'outputs/shared_eight_study_20261029'
OUT=ROOT/'outputs/shared_eight_e2e_20261029'
DATASETS=('continuous5m','zenodo5501527')
MODELS=('ArcFENet','HYDRA','ArcFusionNet')
ARMS=('original','fixed8','phase8')
SETTINGS={
    'ArcFENet':dict(epochs=20,lr=.001,batch=128,optimizer='Adam',weight_decay=1e-5),
    'HYDRA':dict(epochs=70,lr=.01,batch=256,optimizer='AdamW',weight_decay=1e-4),
    'ArcFusionNet':dict(epochs=70,lr=.0001,batch=64,optimizer='AdamW_cosine',weight_decay=.01),
}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(doc,ensure_ascii=False,indent=2));temp.replace(path)


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol exists; do not overwrite')
    OUT.mkdir(parents=True,exist_ok=True);identity=[]
    old=json.loads((FEATURES/'protocol.json').read_text())
    assert tuple(old['features'])==NAMES
    for row in old['source_code']:
        if row['path'].endswith('shared_features.py'):assert sha(ROOT/row['path'])==row['sha256']
    historical=json.loads((SOURCE/'frozen_inputs.json').read_text())
    for ds in DATASETS:
        d=HydraLoader(SOURCE/'data'/ds/'recovery');keep=np.load(FEATURES/f'{ds}_indices.npy')
        np.save(OUT/f'{ds}_indices.npy',keep)
        d.table.iloc[keep].to_csv(OUT/f'{ds}_metadata.csv',index=False)
        for key in ['fixed','phase']:
            a=np.load(FEATURES/f'{ds}_{key}8.npy');assert a.shape==(len(d.table),8)
            # Recompute all columns using their actual source arrays before freezing.
            path=(ROOT/'outputs/rcr_three_backbones_20261028/data'/ds/'fixed20') if key=='fixed' else d.root
            wave=np.load(path/'wave.npy',mmap_mode='r');energy=np.load(path/'energy.npy',mmap_mode='r')
            tab=pd.read_csv(path/'cycles.csv');assert np.array_equal(tab.sample_id,d.table.sample_id)
            np.testing.assert_array_equal(a,shared_eight(wave,energy,(tab.end_seconds-tab.start_seconds).to_numpy()))
            np.save(OUT/f'{ds}_{key}8.npy',a)
            for name in ['wave.npy','energy.npy','cycles.csv']:
                p=path/name;identity.append(dict(path=str(p),sha256=sha(p)))
        for name in ['rgb.npy','temporal.npy','spectrum.npy','hydra_seed0.npy','cycles.csv','splits.json']:
            p=d.root/name;value=sha(p)
            if str(p) in historical:assert historical[str(p)]==value
            identity.append(dict(path=str(p),sha256=value))
        for fold in range(5):
            tr,te=d.indices(fold)
            assert np.array_equal(tr,np.intersect1d(tr,keep))
            assert not set(d.table.iloc[tr].condition_key)&set(d.table.iloc[te].condition_key)
            for m in MODELS:
                p=SOURCE/'runs'/ds/'recovery'/m/f'f{fold}_s0/outer/preprocessing.npz'
                state=np.load(p);assert np.array_equal(state['fit_indices'],tr)
                identity.append(dict(path=str(p),sha256=sha(p)))
        print('COHORT',ds,len(keep),flush=True)
    code=[Path(__file__),ROOT/'src/arc_model/methods/phase_representation/shared_features.py']
    code+=list((ROOT/'src/arc_model/models/standalone').glob('*.py'))
    code+=list((ROOT/'src/arc_model/data').glob('*_loader.py'))
    spec=dict(status='frozen_before_training',datasets=DATASETS,models=MODELS,arms=ARMS,features=NAMES,
        settings=SETTINGS,folds=5,seeds=[0],fits=90,initialization='from scratch; same original parameters per fold across arms; feature weights zero',
        backbone_inputs='RCR for every arm; only added feature coordinates fixed20 vs RCR differ',
        preprocessing='same original train-only preprocessing states reused after exact train-index and identity checks',
        feature_preprocessing='each coordinate standardized using current outer train only; no selection, clipping, gate or threshold tuning',
        attachment='existing independent neural models extra linear is algebraically classifier concatenation; no MLP or attention added',
        hydra='author fixed random transform remains fixed; all original trainable classifier parameters fitted from scratch',
        neural='all encoder/classifier parameters trainable; no pretrained weights loaded',
        precision={'ArcFENet':'FP32','HYDRA':'FP32','ArcFusionNet':'BF16 autocast; FP32 parameters and losses'},
        primary_pairs=['phase8-original','phase8-fixed8'],secondary_pairs=['fixed8-original'],
        statistics='condition bootstrap5000; signflip20000; Holm12primary, separate6secondary; one seed exploratory',
        limitations=['no learned PLL or differentiable VMD','single seed and prior outer-test exposure',
                     'previous frozen-head study has different initialization/budget; cross-study difference is not a pure freezing effect'],
        gpu_physical=[1,2,3],input_hashes=identity,code_hashes=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in code])
    save(OUT/'protocol.json',spec)
    # Distribute long self-collected FusionNet jobs evenly across the three GPUs.
    jobs=[]
    for ds in ['zenodo5501527','continuous5m']:
        for model in ['HYDRA','ArcFENet','ArcFusionNet']:
            for fold in range(5):
                for arm in ARMS:jobs.append(dict(dataset=ds,model=model,fold=fold,arm=arm))
    save(OUT/'jobs.json',jobs)


def make_model(model,arm,seed,width=448):
    torch.manual_seed(seed)
    if model=='HYDRA':
        base=nn.Linear(width,1)
        if arm=='original':return base
        full=nn.Linear(width+8,1)
        with torch.no_grad():
            full.weight.zero_();full.weight[:,:width].copy_(base.weight);full.bias.copy_(base.bias)
        return full
    network=(FENet if model=='ArcFENet' else FusionNet)(extra=0 if arm=='original' else 8)
    if network.extra is not None:nn.init.zeros_(network.extra.weight)
    return network


def input_tensors(ds,model,fold,arm,dest):
    folder=SOURCE/'data'/ds/'recovery'
    loader=(ArcFENetLoader if model=='ArcFENet' else ArcFusionLoader if model=='ArcFusionNet' else HydraLoader)(folder)
    tr,te=loader.indices(fold);keep=np.load(OUT/f'{ds}_indices.npy');te=np.intersect1d(te,keep)
    state=np.load(SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s0/outer/preprocessing.npz')
    assert np.array_equal(state['fit_indices'],tr)
    scaler={}
    if model=='HYDRA':
        z=torch.from_numpy(np.load(folder/'hydra_seed0.npy'))
        sc=SparseScaler();sc.fit(z[tr]);x=sc.transform(z).cuda();s=None
        for name in ['mu','sigma','epsilon']:
            np.testing.assert_array_equal(getattr(sc,name).numpy(),state['hydra_'+name])
            scaler['hydra_'+name]=getattr(sc,name).numpy()
    elif model=='ArcFENet':
        x=torch.tensor(np.array(loader.rgb,copy=True),device='cuda');s=None
    else:
        # Values selected by the audited original train-only RF32 state.
        b=loader.inputs(np.arange(len(loader.table)),selected_bins=state['harmonic_indices'])
        x=torch.from_numpy(((b['temporal'].numpy()-state['temporal_mean'])/state['temporal_std']).astype('float32')).cuda()
        s=torch.from_numpy(((b['spectrum'].numpy()-state['spectral_mean'][None,None])/state['spectral_std'][None,None]).astype('float32')).cuda()
        for name in ['harmonic_indices','spectral_mean','spectral_std','temporal_mean','temporal_std']:scaler[name]=state[name]
    features=None
    if arm!='original':
        a=np.load(OUT/f'{ds}_{"fixed" if arm=="fixed8" else "phase"}8.npy')
        mu=a[tr].mean(0);sd=np.maximum(a[tr].std(0),1e-6)
        features=torch.from_numpy(((a-mu)/sd).astype('float32')).cuda()
        scaler.update(feature_mean=mu,feature_std=sd)
        if model=='HYDRA':x=torch.cat([x,features],1)
    np.savez(dest/'preprocessing.npz',fit_indices=tr,**scaler)
    np.savez(dest/'indices.npz',train=tr,test=te)
    return loader,x,s,features,tr,te


def train_job(job):
    ds,model,fold,arm=(job[k] for k in ['dataset','model','fold','arm'])
    dest=OUT/'runs'/ds/model/f'f{fold}'/arm
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    tick=time.perf_counter();loader,x,s,features,tr,te=input_tensors(ds,model,fold,arm,dest)
    seed=fold;network=make_model(model,arm,seed).cuda()
    initial={k:v.detach().cpu().clone() for k,v in network.named_parameters()}
    core={k:v for k,v in initial.items() if not k.startswith('extra.')}
    if model=='HYDRA':core={'weight':initial['weight'][:,:448],'bias':initial['bias']}
    initial_hash=hashlib.sha256(b''.join(core[k].numpy().tobytes() for k in sorted(core))).hexdigest()
    # Reset stochastic training state after model construction so added columns
    # do not alter dropout RNG or train-batch order relative to the original.
    torch.manual_seed(seed+100000);generator=torch.Generator(device='cuda').manual_seed(seed+90000)
    st=SETTINGS[model];kind=torch.optim.Adam if st['optimizer']=='Adam' else torch.optim.AdamW
    optimizer=kind(network.parameters(),lr=st['lr'],weight_decay=st['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,st['epochs'],eta_min=1e-6) if model=='ArcFusionNet' else None
    labels=torch.as_tensor(loader.labels,device='cuda');tri=torch.as_tensor(tr,device='cuda')
    history=[];gradient={};bf=model=='ArcFusionNet'
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    def forward(ids):
        return network(x[ids]) if model=='HYDRA' else network(x[ids],None if s is None else s[ids],None if features is None else features[ids])
    for epoch in range(st['epochs']):
        network.train();order=tri[torch.randperm(len(tri),generator=generator,device='cuda')]
        total=torch.zeros((),device='cuda')
        for j,ids in enumerate(order.split(st['batch'])):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf):
                logits=forward(ids)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float()[:,0],labels[ids].float()) if model=='HYDRA' else torch.nn.functional.cross_entropy(logits.float(),labels[ids])
            loss.backward()
            if epoch==0 and j==0:
                for name,p in network.named_parameters():
                    if p.grad is not None:gradient[name]=float(p.grad.detach().norm())
            optimizer.step();total+=loss.detach()*len(ids)
        value=float(total)/len(tr)
        if not np.isfinite(value):raise ValueError('Nonfinite loss')
        history.append(dict(epoch=epoch+1,loss=value,lr=optimizer.param_groups[0]['lr']))
        if scheduler:scheduler.step()
        save(dest/'progress.json',dict(epoch=epoch+1,epochs=st['epochs'],elapsed_seconds=time.perf_counter()-start,loss=value))
        if epoch==0 or (epoch+1)%10==0:print('EPOCH',ds,model,fold,arm,epoch+1,round(value,6),round(time.perf_counter()-start,1),flush=True)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-start
    network.eval();prob=[]
    with torch.no_grad():
        for ids in torch.as_tensor(te,device='cuda').split(st['batch']):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf):logits=forward(ids)
            prob.append((logits.float()[:,0].sigmoid() if model=='HYDRA' else logits.float().softmax(-1)[:,1]).cpu().numpy())
    prob=np.concatenate(prob)
    torch.save(network.state_dict(),dest/'model.pt')
    np.savez(dest/'predictions.npz',indices=te,labels=loader.labels[te],scores=prob)
    changes={k:float((v.detach().cpu()-initial[k]).norm()) for k,v in network.named_parameters()}
    backbone_names=[k for k in changes if ('head' not in k and not k.startswith('extra.'))] if model!='HYDRA' else []
    if backbone_names:assert any(changes[k]>0 for k in backbone_names) and any(gradient.get(k,0)>0 for k in backbone_names)
    save(dest/'training_diagnostics.json',dict(first_batch_gradient_norm=gradient,parameter_change_norm=changes))
    save(dest/'COMPLETE.json',dict(status='complete',job=job,protocol_sha256=sha(OUT/'protocol.json'),
        encoder_frozen=False if model!='HYDRA' else 'fixed random transform by design',initialization='from_scratch',
        common_initial_parameter_hash=initial_hash,settings=st,training_seconds=training_seconds,
        total_seconds=time.perf_counter()-tick,train=len(tr),test=len(te),
        trainable_parameters=sum(p.numel() for p in network.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),history=history,
        checkpoint_sha256=sha(dest/'model.pt'),prediction_sha256=sha(dest/'predictions.npz'),
        gpu_physical=os.environ['CUDA_VISIBLE_DEVICES']))
    print('DONE',job,round(time.perf_counter()-tick,2),flush=True)
    del network,optimizer,x,s,features;gc.collect();torch.cuda.empty_cache()


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Freeze --prepare, then launch three explicit workers')
    for ds in DATASETS:
        for cls in [ArcFENetLoader,ArcFusionLoader,HydraLoader]:cls(SOURCE/'data'/ds/'recovery')
    print('PREFLIGHT_READY: same8, end-to-end90 fits, datasets separate, physicalGPU1/2/3')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--partition',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    else:
        assert args.partition in (0,1,2) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.partition+1)
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
        torch.backends.cuda.matmul.allow_tf32=False
        jobs=json.loads((OUT/'jobs.json').read_text())
        for i,job in enumerate(jobs):
            if i%3==args.partition:train_job(job)
