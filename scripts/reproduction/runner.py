"""Execute one manuscript stage with the caller's Python and explicit GPUs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))


def call(script, *args, gpu=None, audit_all=False):
    if gpu is None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    else:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if audit_all:
        env['ARC_AUDIT_ALL']='1'
    else:
        env.pop('ARC_AUDIT_ALL',None)
    subprocess.run([sys.executable,str(script),*map(str,args)],cwd=ROOT,env=env,check=True)


def network(stage, gpus):
    from importlib import import_module
    module = import_module(f'run_{stage}_20261105')
    config = yaml.safe_load(module.CFG.read_text())
    module.freeze(config) if stage == 'folded_calibration' else module.prepare(config)
    with ThreadPoolExecutor(len(gpus)) as pool:
        jobs = [pool.submit(call, Path(module.__file__), '--worker', gpu, gpu=gpu) for gpu in gpus]
        for job in jobs:
            job.result()
    if len(gpus)==3:
        with ThreadPoolExecutor(3) as pool:
            jobs = [pool.submit(call, Path(module.__file__), '--audit', gpu, gpu=gpu) for gpu in gpus]
            for job in jobs:
                job.result()
    else:
        call(Path(module.__file__), '--audit', gpus[0], gpu=gpus[0], audit_all=True)
        # The existing collector reads three audit CSVs.
        import pandas as pd
        frame = pd.read_csv(module.OUT/f'audit_gpu{gpus[0]}.csv')
        for missing in {1,2,3}-{gpus[0]}:
            frame.iloc[:0].to_csv(module.OUT/f'audit_gpu{missing}.csv',index=False)
    module.collect(config)


def visual_features(folder, device, *, public):
    import numpy as np
    from matplotlib import colormaps
    import torch
    from arc_model.implementations.baseline_hydra.features import paper_transform
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if public:
        from arc_model.representations.mbwvd_torch import transform
    else:
        from fenet_vmd_graph_20261030 import VMDGraph
        transform = VMDGraph(batch=64,max_iter=500,alpha=2000.,tol=1e-7,steps=25).transform
    w = np.load(folder/'wave.npy',mmap_mode='r')
    source = np.load(folder/'fenet_wave.npy',mmap_mode='r')
    tmp = folder/'rgb.tmp.npy'
    dest = folder/'rgb.npy'
    if not dest.exists():
        images = np.lib.format.open_memmap(tmp, mode='w+',dtype='float32',shape=(len(w),3,100,100))
        for a in range(0,len(w),64):
            z,*_ = transform(np.array(source[a:a+64],copy=True))
            if public:
                lo=z.min((1,2),keepdims=True);hi=z.max((1,2),keepdims=True)
                rgb=colormaps['jet']((z-lo)/np.maximum(hi-lo,1e-20))[...,:3].transpose(0,3,1,2)
            else:
                rgb=[]
                for image in z:
                    lo,hi=float(image.min()),float(image.max())
                    rgb.append(colormaps['jet']((image-lo)/max(hi-lo,1e-20))[...,:3].transpose(2,0,1))
            images[a:a+len(z)]=rgb
        images.flush()
        del images
        tmp.replace(dest)
    if not (folder/'hydra_features.npy').exists():
        tmp = folder/'hydra_features.tmp.npy'
        np.save(tmp,paper_transform(w,device=device))
        tmp.replace(folder/'hydra_features.npy')
    torch.cuda.empty_cache()


def self_baselines():
    import torch
    from check_baseline_release import verify
    verify()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import run_self462_1m_baselines_20261105 as refs
    from dual_ds_hydra_self462_1m_20261105 import solve
    from arc_model.implementations.baseline_hydra import features
    config=yaml.safe_load(refs.CONFIG.read_text())
    refs.rf_states()
    from active_lp_hydra_self462_1m_20261105 import solve as working_set_solve
    for fold in range(4):
        for model in refs.MODELS:
            if model=='HYDRA_Baseline':
                features.linprog = working_set_solve if fold==2 else solve
                refs.hydra_fit(fold)
            elif model.startswith('cnn_snn'):
                refs.cnn(model,fold,config)
            else:
                refs.baseline(model,fold)


def correction(gpus, dataset):
    import run
    run.prepare() if dataset=='both' else run.prepare((dataset,))
    with ThreadPoolExecutor(len(gpus)+1) as pool:
        jobs=[pool.submit(call,HERE/'run.py','cpu')]
        jobs.extend(pool.submit(call,HERE/'run.py','gpu',gpu=gpu) for gpu in gpus)
        for job in jobs:job.result()
    # Fail promptly after a failed fit instead of waiting for a missing receipt.
    import json
    for job in json.loads((run.O/'protocol.json').read_text())['jobs']:
        if not (Path(job['folder'])/'COMPLETE.json').exists():
            raise RuntimeError('Missing completed fit: '+job['folder'])
    call(HERE/'heads.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','self-baselines','public-reference','dynamic_state','phase_hierarchy','folded_calibration','correction'])
    parser.add_argument('--gpus',nargs='+',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--dataset',choices=('both','self462','zenodo5501527'),default='both')
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus):parser.error('Specify each GPU once.')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpus[0])
    if args.stage=='features':
        import run
        if args.dataset!='zenodo5501527':visual_features(run.SELF,'cuda',public=False)
        if args.dataset!='self462':visual_features(run.PUB,'cuda',public=True)
    elif args.stage=='self-baselines':
        self_baselines()
    elif args.stage=='public-reference':
        from run_raw_bn_stability_20261105 import rebuild_public
        rebuild_public()
    elif args.stage=='correction':
        correction(args.gpus,args.dataset)
    else:
        network(args.stage,args.gpus)


if __name__=='__main__':
    import _bootstrap  # noqa: F401
    main()
