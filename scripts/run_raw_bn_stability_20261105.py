"""Rebuild the published public raw-encoding reference fits."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import _bootstrap as boot
from run_dynamic_state_20261105 import sha, save
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap
from arc_model.implementations.phase_event_snn.data import training_statistics, condition_class_weights

ROOT = boot.ROOT
OUT = ROOT/'outputs/raw_bn_stability_20261105'
CONFIG = ROOT/'configs/modules/raw_bn_stability_20261105.yaml'


def destination(j):
    return OUT/'runs'/j['dataset']/j['reference']/f'f{j["fold"]}_s{j["seed"]}'


def batches(indices, size=512):
    chunks = list(indices.split(size))
    if len(chunks)>1 and len(chunks[-1]) == 1:
        chunks[-2] = torch.cat([chunks[-2], chunks[-1]]); chunks.pop()
    return chunks


def predict(net, xp, mask, indices):
    net.eval()
    with torch.inference_mode():
        return np.concatenate([net(xp[b], mask[b]).float().sigmoid().cpu().numpy()
                               for b in torch.as_tensor(indices, device='cuda').split(512)])


def fit(j, cfg, table, x, mask, train, test, d):
    original = d/'original'; original.mkdir(exist_ok=True)
    if (original/'COMPLETE.json').exists():
        return
    start = time.perf_counter()
    mean, std = training_statistics(x, train)
    xp = torch.as_tensor((x-mean[None,None,:,None])/std[None,None,:,None], device='cuda')
    y = table.label.to_numpy(); weights = condition_class_weights(y, table.condition_key.to_numpy(), train)
    target = torch.as_tensor(y, dtype=torch.float32, device='cuda')
    weight = torch.as_tensor(weights, device='cuda'); tr = torch.as_tensor(train, device='cuda')
    run_seed = 1000*j['seed']+j['fold']; torch.manual_seed(run_seed); torch.cuda.manual_seed_all(run_seed)
    net = DeepCodedPhaseMap('raw').cuda(); t = cfg['training']
    opt = torch.optim.AdamW(net.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t['epochs'], eta_min=t['min_learning_rate'])
    gen = torch.Generator(device='cuda').manual_seed(run_seed+90000); losses=[]
    for epoch in range(t['epochs']):
        net.train(); total = torch.zeros((), device='cuda')
        for ix in batches(tr[torch.randperm(len(tr), device='cuda', generator=gen)], t['batch_size']):
            opt.zero_grad(set_to_none=True); logit, activity = net(xp[ix], mask[ix], return_activity=True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), target[ix], reduction='none')*weight[ix]).mean()+t['spike_activity_penalty']*activity
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), t['gradient_clip']); opt.step()
            total += loss.detach()*len(ix)
        scheduler.step(); value = float(total)/len(train); assert np.isfinite(value); losses.append(value)
        save(d/'progress.json', dict(job=j, epoch=epoch+1, epochs=t['epochs'], loss=value, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    p = predict(net, xp, mask, test)
    torch.save(net.state_dict(), original/'model.pt')
    np.savez(original/'predictions.npz', indices=test, labels=y[test], probability=p)
    np.savez(original/'preprocessing.npz', fit_indices=train, mean=mean, std=std)
    np.savez(original/'indices.npz', train=train, test=test)
    save(original/'training_loss.json', losses)
    status='new_fixed_raw_fit'; del xp, net, opt, target, weight
    save(original/'COMPLETE.json', dict(status=status, seconds=time.perf_counter()-start,
         hashes={n:sha(original/n) for n in ('model.pt','preprocessing.npz','indices.npz','predictions.npz','training_loss.json')}))


def rebuild_public():
    """The public base model uses the identical raw-code and training recipe."""
    import yaml
    from run_dynamic_state_20261105 import host_data, roles
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = yaml.safe_load(CONFIG.read_text())
    dataset = 'zenodo5501527'
    table, _, _, masks = host_data(dataset)
    # The historical baseline consumed all eight channels, three of them zero.
    x = np.load(ROOT/'outputs/map_input_calibration_20261105/data/zenodo5501527/realtime_raw.npy', mmap_mode='r')
    mask = torch.as_tensor(masks, device='cuda')
    for fold in range(5):
        train, test = roles(dataset, fold)
        for seed in range(3):
            job = dict(dataset=dataset, reference='rcr', fold=fold, seed=seed, reuse_raw_outer=False)
            d = destination(job)
            d.mkdir(parents=True, exist_ok=True)
            fit(job, config, table, x, mask, train, test, d)
