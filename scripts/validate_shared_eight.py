"""Same eight features for three independent models; bounded frozen-encoder study."""

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
from arc_model.methods.phase_representation.shared_features import shared_eight, NAMES

ROOT = boot.ROOT
SOURCE = ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
FIXED = ROOT/'outputs/rcr_three_backbones_20261028/data'
OUT = ROOT/'outputs/shared_eight_study_20261029'
DATASETS = ('continuous5m', 'zenodo5501527')
MODELS = ('ArcFENet', 'HYDRA', 'ArcFusionNet')
ARMS = ('refit', 'fixed8', 'phase8', 'relative2', 'position6', 'shift8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Protocol already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs=[]
    for ds in DATASETS:
        phase=HydraLoader(SOURCE/'data'/ds/'recovery')
        fixed=HydraLoader(FIXED/ds/'fixed20')
        assert np.array_equal(phase.table.sample_id,fixed.table.sample_id)
        assert np.array_equal(phase.labels,fixed.labels)
        keep=np.flatnonzero(phase.table.guard40.to_numpy(bool) & fixed.table.guard40.to_numpy(bool))
        np.save(OUT/f'{ds}_indices.npy',keep)
        phase.table.iloc[keep].to_csv(OUT/f'{ds}_metadata.csv',index=False)
        # Deterministic, nontrivial shifts; 180 degrees excluded for pi-periodic moments.
        choices=np.r_[1:10,11:20]
        rng=np.random.default_rng(20261029)
        shifts=rng.choice(choices,len(phase.table))
        for tag,loader in [('fixed',fixed),('phase',phase)]:
            duration=(loader.table.end_seconds-loader.table.start_seconds).to_numpy()
            e=np.load(loader.root/'energy.npy',mmap_mode='r')
            f=shared_eight(loader.wave,e,duration)
            np.save(OUT/f'{ds}_{tag}8.npy',f)
            if tag=='phase':
                np.save(OUT/f'{ds}_shift8.npy',shared_eight(loader.wave,e,duration,shifts=shifts))
                np.save(OUT/f'{ds}_shifts.npy',shifts)
            for name in ['wave.npy','energy.npy','cycles.csv','splits.json']:
                path=loader.root/name;inputs.append(dict(path=str(path),sha256=sha(path)))
        for model in MODELS:
            for fold in range(5):
                stage=SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s0'/'outer'
                receipt=json.loads((stage/'M0/COMPLETE.json').read_text())
                assert receipt['checkpoint_sha256']==sha(stage/'M0/model.pt')
                for path in [stage/'M0/model.pt',stage/'preprocessing.npz',stage/'M0/predictions.npz']:
                    inputs.append(dict(path=str(path),sha256=sha(path)))
                prep=np.load(stage/'preprocessing.npz')
                tr,te=phase.indices(fold)
                assert np.array_equal(prep['fit_indices'],tr)
                assert not set(phase.table.iloc[tr].condition_key)&set(phase.table.iloc[te].condition_key)
        print('COHORT',ds,len(keep),phase.table.iloc[keep].condition_key.nunique(),flush=True)
    source_files=[Path(__file__),ROOT/'src/arc_model/methods/phase_representation/shared_features.py']
    source_files+=list((ROOT/'src/arc_model/models/standalone').glob('*.py'))
    source_files+=list((ROOT/'src/arc_model/data').glob('*_loader.py'))
    spec=dict(status='frozen_before_fits',datasets=DATASETS,models=MODELS,features=NAMES,
              arms=['original']+list(ARMS),new_trainable_networks='existing linear heads only; no MLP/gate',
              backbone_seed=0,head_shuffle_seed='fold+90000',folds=5,
              training=dict(epochs=70,batch=512,optimizer='AdamW',lr=.001,weight_decay=.0001,
                            loss='original-output binary BCE or two-class CE, no class weighting',
                            scheduler=None,selection=None,threshold=.5),
              common_backbone_input='RCR inputs fixed across every arm; fixed20 changes ONLY added feature coordinate/support',
              original='saved encoder and original linear head; FP32 head replay for common numerical basis',
              primary_pairs=['phase8-original','phase8-refit','phase8-fixed8'],
              secondary_pairs=['phase8-relative2','phase8-position6','phase8-shift8'],
              inference='condition-level paired bootstrap5000 and sign-flip20000; Holm18primary tests',
              source_inputs=inputs,source_code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in source_files],
              scope='single pretrained encoder seed, frozen-feature screening, no end-to-end or universal-positive claim',
              exclusions='guard40 common support; no test-dependent deletion/selection',gpu_physical=[1,2,3])
    save(OUT/'protocol.json',spec)


def extract(ds, model, fold, dest):
    stage=SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s0'/'outer'
    folder=SOURCE/'data'/ds/'recovery'
    state=np.load(stage/'preprocessing.npz')
    weights=torch.load(stage/'M0/model.pt',map_location='cpu',weights_only=True)
    weights={k.removeprefix('base.'):v for k,v in weights.items()}
    if model=='HYDRA':
        loader=HydraLoader(folder)
        f=torch.from_numpy(np.load(folder/'hydra_seed0.npy'))
        v=f.clamp(0).sqrt()
        h=(((v-torch.from_numpy(state['hydra_mu']))*(v!=0))/torch.from_numpy(state['hydra_sigma'])).numpy()
        head=nn.Linear(h.shape[1],1);head.load_state_dict(weights,strict=True)
        with torch.no_grad(): original=head(torch.from_numpy(h)).sigmoid()[:,0].numpy()
    else:
        loader=ArcFENetLoader(folder) if model=='ArcFENet' else ArcFusionLoader(folder)
        network=(FENet() if model=='ArcFENet' else FusionNet()).cuda().eval()
        network.load_state_dict(weights,strict=True)
        head=network.backbone.head if model=='ArcFENet' else network.head
        captured=[]
        hook=head.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().float().cpu().numpy()))
        reference=[]
        with torch.no_grad():
            for start in range(0,len(loader.table),128 if model=='ArcFENet' else 64):
                ids=np.arange(start,min(start+(128 if model=='ArcFENet' else 64),len(loader.table)))
                if model=='ArcFENet':
                    x=loader.inputs(ids)['image'].cuda();s=None
                else:
                    b=loader.inputs(ids,selected_bins=state['harmonic_indices'])
                    x=((b['temporal'].numpy()-state['temporal_mean'])/state['temporal_std'])
                    s=(b['spectrum'].numpy()-state['spectral_mean'][None,None])/state['spectral_std'][None,None]
                    x=torch.from_numpy(x.astype('float32')).cuda();s=torch.from_numpy(s.astype('float32')).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model=='ArcFusionNet'):
                    z=network(x,s)
                reference.extend(z.float().softmax(-1)[:,1].cpu().numpy())
        hook.remove();h=np.concatenate(captured)
        cpu_head=nn.Linear(head.in_features,2);cpu_head.load_state_dict({k:v.cpu() for k,v in head.state_dict().items()})
        head=cpu_head
        with torch.no_grad(): original=head(torch.from_numpy(h)).softmax(-1)[:,1].numpy()
        np.save(dest/'original_native_precision.npy',np.asarray(reference))
        del network;torch.cuda.empty_cache()
    tr,te=loader.indices(fold)
    assert np.array_equal(state['fit_indices'],tr)
    old=np.load(stage/'M0/predictions.npz')
    assert np.array_equal(old['indices'],te)
    err=float(abs(original[te]-old['probability']).max())
    # FP32 replay of a BF16-trained final linear map may differ slightly.
    if err > .015:raise ValueError(f'Original checkpoint replay mismatch {ds}/{model}/{fold}: {err}')
    np.save(dest/'embedding.npy',h.astype('float32'))
    np.save(dest/'original.npy',original.astype('float32'))
    torch.save(head.state_dict(),dest/'initial_head.pt')
    save(dest/'extraction.json',dict(max_original_probability_error=err,head_shape=list(head.weight.shape),
                                   stage=str(stage),checkpoint_sha256=sha(stage/'M0/model.pt')))
    return loader,h,head,original


def train_job(ds,model,fold):
    dest=OUT/'runs'/ds/model/f'f{fold}'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    tick=time.perf_counter();loader,h,head,original=extract(ds,model,fold,dest)
    keep=np.load(OUT/f'{ds}_indices.npy');tr,te=loader.indices(fold)
    tr=np.intersect1d(tr,keep);te=np.intersect1d(te,keep)
    np.savez(dest/'indices.npz',train=tr,test=te)
    features={};scalers={}
    for name in ['fixed','phase','shift']:
        f=np.load(OUT/f'{ds}_{name}8.npy')
        mu=f[tr].mean(0);sd=np.maximum(f[tr].std(0),1e-6)
        features[name]=((f-mu)/sd).astype('float32')
        scalers[name+'_mean']=mu;scalers[name+'_std']=sd
    zeros=np.zeros_like(features['phase'])
    relative=features['phase'].copy();relative[:,2:]=0
    position=features['phase'].copy();position[:,:2]=0
    xs=[zeros,features['fixed'],features['phase'],relative,position,features['shift']]
    # A single batched optimizer implements six independent linear classifiers.
    # Loss is SUM over arms, so each arm receives its own unscaled loss gradient.
    x=torch.from_numpy(np.stack([np.concatenate([h,f],axis=1) for f in xs])).cuda()
    w0=head.weight.detach().cpu();b0=head.bias.detach().cpu();outputs=w0.shape[0]
    w=nn.Parameter(torch.cat([w0,torch.zeros(outputs,8)],1)[None].repeat(len(ARMS),1,1).cuda())
    b=nn.Parameter(b0[None].repeat(len(ARMS),1).cuda())
    labels=torch.from_numpy(loader.labels).cuda();tri=torch.from_numpy(tr).cuda()
    optimizer=torch.optim.AdamW([w,b],lr=.001,weight_decay=.0001)
    generator=torch.Generator(device='cuda').manual_seed(fold+90000)
    history=[];train_start=time.perf_counter()
    for epoch in range(70):
        order=tri[torch.randperm(len(tri),generator=generator,device='cuda')]
        totals=torch.zeros(len(ARMS),device='cuda')
        for ix in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            logits=torch.einsum('and,aod->ano',x[:,ix],w)+b[:,None]
            if outputs==1:
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:,0],labels[ix].float()[None].expand(len(ARMS),-1),reduction='none').mean(1)
            else:
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),labels[ix][None].expand(len(ARMS),-1).reshape(-1),reduction='none').reshape(len(ARMS),-1).mean(1)
            loss.sum().backward();optimizer.step();totals+=loss.detach()*len(ix)
        history.append((totals/len(tr)).cpu().tolist())
        if epoch in [0,34,69]:
            save(dest/'progress.json',dict(epoch=epoch+1,epochs=70,loss=history[-1]))
    with torch.no_grad():
        logits=torch.einsum('and,aod->ano',x[:,te],w)+b[:,None]
        prob=(logits[:,:,0].sigmoid() if outputs==1 else logits.softmax(-1)[:,:,1]).cpu().numpy()
    np.savez(dest/'predictions.npz',indices=te,labels=loader.labels[te],original=original[te],**dict(zip(ARMS,prob)))
    np.savez(dest/'heads.npz',weight=w.detach().cpu().numpy(),bias=b.detach().cpu().numpy(),**scalers)
    save(dest/'COMPLETE.json',dict(status='complete',dataset=ds,model=model,fold=fold,epochs=70,
         frozen_backbone=True,encoder_seed=0,new_head_fits=len(ARMS),head_training_seconds=time.perf_counter()-train_start,
         total_seconds=time.perf_counter()-tick,train=len(tr),test=len(te),
         shared_features=list(NAMES),additional_effective_parameters=8*outputs,head_arms=list(ARMS),
         protocol_sha256=sha(OUT/'protocol.json'),predictions_sha256=sha(dest/'predictions.npz'),
         heads_sha256=sha(dest/'heads.npz'),gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],history=history))
    print('DONE',ds,model,fold,round(time.perf_counter()-tick,2),flush=True)
    del x,w,b,optimizer;gc.collect();torch.cuda.empty_cache()


def run_study(suite,*,execute=False):
    assert set(suite['datasets'])=={'new5m','zenodo5501527'}
    if execute:raise ValueError('Use --prepare, then three explicit workers after protocol freeze')
    for ds in DATASETS:
        ArcFENetLoader(SOURCE/'data'/ds/'recovery')
        ArcFusionLoader(SOURCE/'data'/ds/'recovery')
        HydraLoader(SOURCE/'data'/ds/'recovery')
    print('PREFLIGHT_READY: same8, both datasets, three models, GPUs1/2/3')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--partition',type=int)
    args=p.parse_args()
    if args.prepare:prepare()
    else:
        assert args.partition in (0,1,2) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.partition+1)
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.18)
        torch.backends.cuda.matmul.allow_tf32=False
        model=MODELS[args.partition]
        for ds in DATASETS:
            for fold in range(5):train_job(ds,model,fold)
