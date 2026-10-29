"""Reload every saved reconstructed FENet and replay its entire held-out fold."""
import argparse
import os
import json
import numpy as np
import torch
import complete_paper_baselines as s
from arc_model.models.standalone.arc_fenet import FENet


def main(partition):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(partition+1)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    jobs=[j for j in json.loads((s.OUT/'protocol.json').read_text())['jobs'] if j['model']=='ArcFENet_reconstructed']
    rows=[]
    for i,j in enumerate(jobs):
        if i%3!=partition:continue
        q=s.OUT/'runs'/j['dataset']/j['model']/f'f{j["fold"]}_s{j["seed"]}'
        z=np.load(q/'predictions.npz');x=np.load(s.FIXED/j['dataset']/'fixed20/rgb.npy',mmap_mode='r')
        net=FENet(extra=0).cuda();net.load_state_dict(torch.load(q/'model.pt',map_location='cpu',weights_only=True));net.eval()
        parts=[]
        with torch.no_grad():
            # Batch construction matches the original inference, not the training.
            for ids in np.array_split(z['indices'],np.ceil(len(z['indices'])/128).astype(int)):
                parts.append(net(torch.tensor(np.array(x[ids]),device='cuda')).softmax(-1)[:,1].cpu().numpy())
        rebuilt=np.concatenate(parts);err=float(abs(rebuilt-z['score']).max())
        assert err<2e-5 and np.array_equal(rebuilt>=.5,z['prediction'])
        rows.append(dict(**j,max_probability_error=err,decision_changes=0,test_windows=len(rebuilt),
            checkpoint_sha256=s.sha(q/'model.pt')))
        del net,x;torch.cuda.empty_cache()
    s.save(s.OUT/f'forward_audit_gpu{partition+1}.json',rows)
    print('PASS',partition+1,len(rows),max(r['max_probability_error'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',type=int,required=True);a=p.parse_args();main(a.partition)
