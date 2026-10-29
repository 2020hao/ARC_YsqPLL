"""Replay every ready Fusion comparator; incremental files avoid repeated checks."""
import argparse
import json
import os
import numpy as np
import torch
import complete_paper_baselines as base
import freeze_grouped_baselines as study
from arc_model.models.standalone.arc_fusion import FusionNet


def portable_state(q,reused):
    weights=torch.load(q/'model.pt',map_location='cpu',weights_only=True)
    if reused:
        assert all(k.startswith('base.') for k in weights)
        weights={k.removeprefix('base.'):v for k,v in weights.items()}
    assert not any(k.startswith('base.') or k.startswith('extra.') for k in weights)
    dest=q/'model_portable.pt'
    if not dest.exists():torch.save(weights,dest)
    return weights,base.sha(dest)


def main(partition):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(partition+1)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    spec=json.loads((study.OUT/'protocol.json').read_text());rows=[]
    for i,j in enumerate(spec['jobs']):
        if i%3!=partition:continue
        ds,fold,seed=(j[k] for k in ['dataset','fold','seed']);q=study.location(ds,fold,seed)
        if not (q/'COMPLETE.json').exists():continue
        dest=q/'FORWARD_AUDIT.json';h=base.sha(q/'model.pt')
        if dest.exists():
            r=json.loads(dest.read_text());assert r['model_sha256']==h
            _,portable_sha=portable_state(q,j['reuse']);r['portable_model_sha256']=portable_sha
            base.save(dest,r);rows.append(r);continue
        state=np.load(q/'preprocessing.npz');expected=np.load(q/'predictions.npz')
        if j['reuse']:
            source=study.OLD/ds/'fixed20/ArcFusionNet'/f'f{fold}_s{seed}/outer/M0/predictions.npz'
            reference=np.load(source);ids=reference['indices'];target=reference['probability']
        else:ids=expected['indices'];target=expected['probability']
        d=base.FIXED/ds/'fixed20';temp=np.load(d/'temporal.npy',mmap_mode='r');sp=np.load(d/'spectrum.npy',mmap_mode='r')
        net=FusionNet(extra=0).cuda();weights,portable_sha=portable_state(q,j['reuse'])
        # The historical training wrapper used base.* around the same network.
        # Strip only this verified namespace; never drop unmatched model tensors.
        net.load_state_dict(weights,strict=True);net.eval()
        assert net.extra is None
        parts=[]
        with torch.no_grad():
            for a in range(0,len(ids),64):
                ix=ids[a:a+64]
                x=torch.tensor((temp[ix]-state['temporal_mean'])/state['temporal_std'],device='cuda')
                s=torch.tensor(((sp[ix][:,state['harmonic_indices']]-state['spectral_mean'])/state['spectral_std'])[:,None],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x,s)
                parts.append(z.float().softmax(-1)[:,1].cpu().numpy())
        value=np.concatenate(parts);err=float(abs(value-target).max());changes=int(((value>=.5)!=(target>=.5)).sum())
        assert err<5e-5 and changes==0,(ds,fold,seed,err,changes)
        r=dict(**j,model_sha256=h,max_probability_error=err,decision_changes=changes,
            replayed=len(ids),gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'],portable_model_sha256=portable_sha)
        base.save(dest,r);rows.append(r);del net;torch.cuda.empty_cache()
    print('PASS_READY',partition+1,len(rows),max([r['max_probability_error'] for r in rows],default=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',type=int,required=True);a=p.parse_args();main(a.partition)
