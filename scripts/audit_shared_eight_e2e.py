"""Independent full held-out checkpoint replay, using each saved preprocessing state."""

import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import _bootstrap as boot
from arc_model.data.arc_fenet_loader import ArcFENetLoader
from arc_model.data.arc_fusion_loader import ArcFusionLoader
from arc_model.data.hydra_loader import HydraLoader
from arc_model.models.standalone.arc_fenet import FENet
from arc_model.models.standalone.arc_fusion import FusionNet

ROOT=boot.ROOT
OUT=ROOT/'outputs/shared_eight_e2e_20261029'
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector/data'
MODELS=('ArcFENet','HYDRA','ArcFusionNet')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def audit(model,ds,fold,arm):
    q=OUT/'runs'/ds/model/f'f{fold}'/arm
    if not (q/'COMPLETE.json').exists():raise FileNotFoundError(q)
    done=json.loads((q/'COMPLETE.json').read_text());state=np.load(q/'preprocessing.npz')
    z=np.load(q/'predictions.npz');indices=z['indices']
    loader=(ArcFENetLoader if model=='ArcFENet' else ArcFusionLoader if model=='ArcFusionNet' else HydraLoader)(SOURCE/ds/'recovery')
    feature=None
    if arm!='original':
        f=np.load(OUT/f'{ds}_{"fixed" if arm=="fixed8" else "phase"}8.npy')
        feature=((f-state['feature_mean'])/state['feature_std']).astype('float32')
    if model=='HYDRA':
        network=nn.Linear(448+(0 if arm=='original' else 8),1)
        raw=np.load(loader.root/'hydra_seed0.npy')
        v=np.sqrt(np.maximum(raw,0))
        xx=((v-state['hydra_mu'])*(v!=0))/state['hydra_sigma']
        if feature is not None:xx=np.concatenate([xx,feature],1)
    else:network=(FENet if model=='ArcFENet' else FusionNet)(extra=0 if arm=='original' else 8)
    network.load_state_dict(torch.load(q/'model.pt',map_location='cpu',weights_only=True),strict=True)
    network.cuda().eval();prob=[]
    with torch.no_grad():
        for start in range(0,len(indices),done['settings']['batch']):
            ids=indices[start:start+done['settings']['batch']]
            if model=='HYDRA':x=torch.from_numpy(xx[ids]).cuda();s=None
            elif model=='ArcFENet':x=loader.inputs(ids)['image'].cuda();s=None
            else:
                b=loader.inputs(ids,selected_bins=state['harmonic_indices'])
                x=torch.from_numpy(((b['temporal'].numpy()-state['temporal_mean'])/state['temporal_std']).astype('float32')).cuda()
                s=torch.from_numpy(((b['spectrum'].numpy()-state['spectral_mean'][None,None])/state['spectral_std'][None,None]).astype('float32')).cuda()
            f=None if feature is None else torch.from_numpy(feature[ids]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model=='ArcFusionNet'):
                logits=network(x) if model=='HYDRA' else network(x,s,f)
            prob.extend((logits.float()[:,0].sigmoid() if model=='HYDRA' else logits.float().softmax(-1)[:,1]).cpu().numpy())
    p=np.asarray(prob);error=float(abs(p-z['scores']).max());changed=int(((p>=.5)!=(z['scores']>=.5)).sum())
    assert error<5e-5 and changed==0,(ds,model,fold,arm,error,changed)
    assert sha(q/'model.pt')==done['checkpoint_sha256'] and sha(q/'predictions.npz')==done['prediction_sha256']
    return dict(dataset=ds,model=model,fold=fold,arm=arm,windows=len(indices),max_error=error,changed_decisions=changed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',type=int,required=True);a=p.parse_args()
    assert a.partition in [0,1,2] and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.partition+1)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for ds in ['continuous5m','zenodo5501527']:
        for fold in range(5):
            for arm in ['original','fixed8','phase8']:rows.append(audit(MODELS[a.partition],ds,fold,arm))
    pd.DataFrame(rows).to_csv(OUT/f'checkpoint_replay_gpu{a.partition+1}.csv',index=False)
    (OUT/f'CHECKPOINT_AUDIT_GPU{a.partition+1}_PASS.json').write_text(json.dumps(dict(status='pass',fits=len(rows),max_error=max(v['max_error'] for v in rows)),indent=2))
    print('PASS',MODELS[a.partition],len(rows))
