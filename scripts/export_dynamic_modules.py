"""Export and verify inference-only C0 restoration bundles; no new fitting."""

import json
import numpy as np
import pandas as pd
import _bootstrap as boot
from arc_model.methods.dynamic_correction import replay
from arc_model.methods.dynamic_score_module import DynamicScoreModule

ROOT=boot.ROOT
OUT=ROOT/'outputs/dynamic_restore_20261029'
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'


def main():
    assert json.loads((OUT/'AUDIT_PASS.json').read_text())['status']=='pass'
    rows=[]
    for ds in ['continuous5m','zenodo5501527']:
        tab=pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv');x=np.load(SOURCE/'data'/ds/'recovery/dynamic.npy')
        for model in ['ArcFENet','HYDRA','ArcFusionNet']:
            for seed in range(3):
                for fold in range(5):
                    q=OUT/'runs'/ds/model/f'f{fold}_s{seed}';z=np.load(q/'predictions.npz');ix=np.load(q/'indices.npz');tr,te=ix['train'],ix['test']
                    old=np.load(SOURCE/'runs'/ds/'recovery'/model/f'f{fold}_s{seed}/predictions.npz')
                    look={int(v):i for i,v in enumerate(old['indices'])};prob=old['M0'][[look[int(v)] for v in te]]
                    base=np.log(np.clip(prob,1e-7,1-1e-7)/np.clip(1-prob,1e-7,1)).astype('float64')
                    for arm in ['legacy_refit','uniform36','dynamic32']:
                        state={k:v for k,v in dict(np.load(q/(arm+'_head.npz'))).items() if k!='fit_indices'};sel=json.loads((q/(arm+'_selection.json')).read_text())
                        _,distance=replay(state,x);threshold=max(float(np.quantile(distance[tr[tab.label.to_numpy()[tr]==0]],sel['quantile'])),1e-6)
                        state.update(alpha=np.asarray(sel['alpha']),gate_threshold=np.asarray(threshold))
                        path=q/(arm+'_module.npz');np.savez(path,**state)
                        module=DynamicScoreModule.load(path);p=module.predict(base,x[te]);error=float(abs(p-z[arm]).max())
                        assert error<5e-5
                        np.testing.assert_allclose(module.predict(base,enabled=False),1/(1+np.exp(-base)),rtol=1e-12,atol=1e-12)
                        rows.append(dict(dataset=ds,model=model,seed=seed,fold=fold,arm=arm,error=error,path=str(path)))
    pd.DataFrame(rows).to_csv(OUT/'inference_export_audit.csv',index=False)
    (OUT/'INFERENCE_EXPORT_PASS.json').write_text(json.dumps(dict(status='pass',modules=len(rows),max_error=max(r['error'] for r in rows),training_samples_needed_at_inference=False),indent=2))
    print('PASS exported inference modules',len(rows))


if __name__=='__main__':main()
