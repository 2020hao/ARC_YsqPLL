"""Read-only full held-out checkpoint replay for the locked single-input study."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
import _bootstrap
import validate_locked_plugin_20261030 as v


def audit():
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2','3']
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
 spec=json.loads((v.OUT/'protocol.json').read_text());results=[]
 versions_path=v.OUT/'EXECUTION_SOURCE_VERSIONS.json'
 if versions_path.exists():versions={q['job']:q for q in json.loads(versions_path.read_text())['jobs']}
 else:
  source=Path(v.__file__).resolve();fingerprint=next(r['sha256'] for r in spec['code'] if Path(r['path']).resolve()==source)
  assert v.sha(source)==fingerprint
  versions={j['id']:dict(job=j['id'],version='current',source_path=str(source),sha256=fingerprint) for j in spec['jobs']}
 for j in spec['jobs']:
  model,ds,f=j['model'],j['dataset'],j['fold'];d=v.OUT/'runs'/ds/model/f'f{f}';receipt=json.loads((d/'COMPLETE.json').read_text());table,keep,folds=v.cohort(ds);tr,te=folds[f];y=table.label.to_numpy();saved=np.load(d/'indices.npz');assert np.array_equal(saved['train'],tr) and np.array_equal(saved['test'],te)
  for col in ['condition_key','source_group','record_id']:assert not set(table.iloc[tr][col])&set(table.iloc[te][col])
  for p,s in receipt['predictions'].items():assert v.sha(d/p)==s
  parent=next(e for e in spec['parents'] if e['dataset']==ds and e['fold']==f and e['baseline_id']==v.IDS[model]);assert v.sha(parent['weights_path'])==parent['weights_sha256']
  maxerr=0.;replayed=0
  for arm in ['single_direct','single_retrained']:
   if model=='HYDRA':
    x,_,a,_,avail,_=v.state_inputs(model,ds,tr,f)
    net=v.package(model,'model').Model.load(parent['weights_path'] if arm=='single_direct' else d/'model.npz')
    margin=net.decision_function(x[te],aligned=a[te],enabled=True,available=avail[te]);prob=1/(1+np.exp(-np.clip(margin,-700,700)))
    off=net.decision_function(x[te],enabled=False)
   else:
    direct_reused_batches=arm=='single_direct' and versions[j['id']]['version'] not in ['v0','v1']
    pp=Path(parent['folder'])/'preprocessing.npz' if arm=='single_direct' else d/'preprocessing.npz'
    state=dict(np.load(pp)) if model=='ArcFusionNet' else None
    x,s,a,b,avail,_=v.state_inputs(model,ds,tr,f,state);inp=v.gpu_tensors(x,s,a,b,avail);del x,s,a,b,avail;gc.collect()
    net=v.package(model,'model').Model(**v.config(model)['model_kwargs']).cuda();net.load_state_dict(torch.load(parent['weights_path'] if arm=='single_direct' else d/'model.pt',map_location='cpu',weights_only=True));net.eval();margin,prob=(v.parent_infer(net,inp,te,model,parent,True) if direct_reused_batches else v.infer(net,inp,te,model,True));offmargin,off=(v.parent_infer(net,inp,te,model,parent,False) if direct_reused_batches else v.infer(net,inp,te,model,False));off2margin,off2=(v.parent_infer(net,inp,te,model,parent,False) if direct_reused_batches else v.infer(net,inp,te,model,False));np.testing.assert_array_equal(off,off2)
   z=np.load(d/(arm+'.npz'));assert np.array_equal(z['indices'],te) and np.array_equal(z['y'],y[te]);np.testing.assert_array_equal(z['margin'],margin);np.testing.assert_array_equal(z['probability'],prob);maxerr=max(maxerr,float(abs(z['probability']-prob).max()));replayed+=len(te)
   if arm=='single_direct':np.testing.assert_array_equal(off,np.load(d/'baseline.npz')['margin' if model=='HYDRA' else 'probability'])
   del net
   if model!='HYDRA':del inp;gc.collect();torch.cuda.empty_cache()
  source=versions[j['id']];assert v.sha(source['source_path'])==source['sha256'];results.append(dict(**j,rows_replayed=replayed,max_error=maxerr,off_exact=True,train_groups_disjoint=True,actual_execution_source=source))
  print('PLUGIN_AUDIT',j['id'],replayed,maxerr,flush=True)
 out=dict(status='pass',jobs=len(results),heldout_rows_replayed=sum(q['rows_replayed'] for q in results),max_error=max(q['max_error'] for q in results),results=results,seed_scope='seed0',baseline_untouched=True,gpu_physical=os.environ['CUDA_VISIBLE_DEVICES'])
 v.save(v.OUT/'AUDIT.json',out)
 print('PLUGIN_AUDIT_PASS',out['jobs'],out['heldout_rows_replayed'],flush=True)


if __name__=='__main__':audit()
