"""Verify complete baseline fits and recompute grouped cycle-level metrics."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import complete_paper_baselines as study


def metrics(ds, model, seed, tab, keep, score, pred):
    y=tab.iloc[keep].label.to_numpy();groups=tab.iloc[keep].condition_key.to_numpy()
    per=[]
    for g in sorted(set(groups)):
        mask=groups==g;yy=y[mask];pp=pred[mask]
        assert set(yy)=={0,1}
        per.append(dict(dataset=ds,model=model,seed=seed,condition=g,
            BAcc=.5*(pp[yy==1].mean()+(~pp[yy==0]).mean())))
    row=dict(dataset=ds,model=model,seed=seed,N=len(y),groups=len(per),BAcc=np.mean([a['BAcc'] for a in per]),
        Recall=recall_score(y,pred),FPR=pred[y==0].mean(),Precision=precision_score(y,pred,zero_division=0),
        F1=f1_score(y,pred),AP=average_precision_score(y,score),TP=int((pred&(y==1)).sum()),
        FP=int((pred&(y==0)).sum()),TN=int((~pred&(y==0)).sum()),FN=int((~pred&(y==1)).sum()))
    return row,per


def main():
    out=study.OUT;spec=json.loads((out/'protocol.json').read_text());audits=[];rows=[];groups=[]
    for receipt in spec['input_sha256']:
        assert study.sha(receipt['path'])==receipt['sha256']
    for receipt in spec['code_sha256']:
        assert study.sha(study.ROOT/receipt['path'])==receipt['sha256']
    for ds in study.DATASETS:
        tab=pd.read_csv(study.FIXED/ds/'fixed20/cycles.csv');keep=np.load(study.COHORT/f'{ds}_indices.npy')
        for model,seeds in [('ArcFENet_reconstructed',[0,1,2]),('HYDRA_paper',[0])]:
            for seed in seeds:
                score=np.zeros(len(tab));pred=np.zeros(len(tab),bool);counts=np.zeros(len(tab),int)
                for fold in range(5):
                    q=out/'runs'/ds/model/f'f{fold}_s{seed}';z=np.load(q/'predictions.npz');i=np.load(q/'indices.npz')
                    done=json.loads((q/'COMPLETE.json').read_text());_,tr,te=study.indices(ds,fold)
                    assert np.array_equal(i['train'],tr) and np.array_equal(i['test'],te)
                    assert np.array_equal(z['indices'],te) and np.array_equal(z['labels'],tab.label.to_numpy()[te])
                    assert done['protocol_sha256']==study.sha(out/'protocol.json')
                    assert done['predictions_sha256']==study.sha(q/'predictions.npz')
                    assert done['gpu_physical'] in ['1','2','3']
                    error=None
                    if model=='HYDRA_paper':
                        from arc_model.models.standalone.hydra import SparseScaler
                        import torch
                        state=np.load(q/'model.npz');feat=torch.from_numpy(np.load(out/'data'/ds/'author_hydra.npy'))
                        sc=SparseScaler();sc.fit(feat[tr])
                        for name in ['mu','sigma','epsilon']:np.testing.assert_array_equal(state[name],getattr(sc,name).numpy())
                        rebuilt=sc.transform(feat[te]).numpy()@state['weight']+state['intercept']
                        error=float(abs(rebuilt-z['score']).max());assert error<1e-7
                        diag=json.loads((q/'solver.json').read_text());assert diag['max_constraint_violation']<1e-6
                    score[te]=z['score'];pred[te]=z['prediction'];counts[te]+=1
                    audits.append(dict(dataset=ds,model=model,fold=fold,seed=seed,
                        seconds=done['seconds'],replay_error=error,
                        checkpoint_sha256=study.sha(q/('model.npz' if model=='HYDRA_paper' else 'model.pt'))))
                assert counts.sum()==len(keep) and (counts[keep]==1).all()
                row,per=metrics(ds,model,seed,tab,keep,score[keep],pred[keep]);rows.append(row);groups+=per
                np.savez(out/f'{ds}_{model}_s{seed}_oof.npz',indices=keep,labels=tab.label.to_numpy()[keep],score=score[keep],prediction=pred[keep])
    frame=pd.DataFrame(rows);frame.to_csv(out/'seed_metrics.csv',index=False)
    summary=frame.groupby(['dataset','model'],sort=False).agg(BAcc=('BAcc','mean'),seed_sd=('BAcc','std'),
        seeds=('seed','nunique'),Recall=('Recall','mean'),FPR=('FPR','mean'),Precision=('Precision','mean'),
        F1=('F1','mean'),AP=('AP','mean'),N=('N','first'),groups=('groups','first')).reset_index()
    summary.to_csv(out/'results.csv',index=False);pd.DataFrame(groups).to_csv(out/'condition_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'fit_audit.csv',index=False)
    # Historical fixed20 current-only Fusion remains a labelled adaptation, with
    # its original training-index discrepancy explicitly counted.
    controls=[];discrepancies=[]
    for ds in study.DATASETS:
        tab=pd.read_csv(study.FIXED/ds/'fixed20/cycles.csv');keep=np.load(study.COHORT/f'{ds}_indices.npy')
        for seed in [0,1,2]:
            scores=np.zeros(len(tab));counts=np.zeros(len(tab),int)
            for fold in range(5):
                folder=study.ROOT/'outputs/rcr_three_backbones_20261028/runs'/ds/'fixed20/ArcFusionNet'/f'f{fold}_s{seed}'
                q=folder/'outer/M0';z=np.load(q/'predictions.npz');_,tr,te=study.indices(ds,fold)
                done=json.loads((q/'COMPLETE.json').read_text());assert done['predictions_sha256']==study.sha(q/'predictions.npz')
                old=np.load(folder/'fit_indices.npz')['train']
                discrepancies.append(dict(dataset=ds,seed=seed,fold=fold,
                    extra_training_windows=len(np.setdiff1d(old,tr)),missing_training_windows=len(np.setdiff1d(tr,old))))
                for col in ['record_id','condition_key','source_group']:
                    assert not set(tab.iloc[old][col])&set(tab.iloc[te][col])
                assert np.array_equal(z['labels'],tab.label.to_numpy()[z['indices']])
                scores[z['indices']]=z['probability'];counts[z['indices']]+=1
            assert (counts[keep]==1).all()
            row,_=metrics(ds,'ArcFusionNet_current_fixed20_historical',seed,tab,keep,scores[keep],scores[keep]>=.5)
            controls.append(row)
    pd.DataFrame(controls).to_csv(out/'current_only_control_seed_metrics.csv',index=False)
    pd.DataFrame(discrepancies).to_csv(out/'current_only_control_index_differences.csv',index=False)
    study.save(out/'AUDIT_PASS.json',dict(status='pass',new_fits=len(audits),OOF_unique=True,
        group_record_source_disjoint=True,train_only_scaler=True,primal_solver_certified=True,
        neural_forward_replay='separate audit required',gpu0_used=False,
        ArcFusionNet_full_original='unavailable; current-only historical control explicitly separate'))
    study.save(out/'RESULTS_COMPLETE.json',dict(status='complete',new_fits=len(audits),datasets=2,
        paper_config_HYDRA_fits=10,reconstructed_ArcFENet_fits=30,ArcFusionNet_full_original='unavailable_inputs'))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
