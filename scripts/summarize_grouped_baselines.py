"""Audit the common grouped baseline registry and report all cycle metrics."""
import json
import numpy as np
import pandas as pd
import complete_paper_baselines as base
import freeze_grouped_baselines as study
from arc_model.evaluation.cycle_metrics import cycle_metrics


def main():
    out=study.OUT;spec=json.loads((out/'protocol.json').read_text())
    for r in spec['identities']:assert base.sha(r['path'])==r['sha256']
    for r in spec['code']:assert base.sha(base.ROOT/r['path'])==r['sha256']
    for r in json.loads((base.OUT/'protocol.json').read_text())['input_sha256']:
        assert base.sha(r['path'])==r['sha256']
    old_audit=pd.read_csv(base.OUT/'fit_audit.csv')
    rows=[];conditions=[];registry=[]
    names={'ArcFENet_reconstructed':'B_FENET_fixed20_v1',
        'HYDRA_paper':'B_HYDRA_FD_fixed20_v1','ArcFusionNet_current':'B_FUSION_CURRENT_fixed20_v2'}
    for ds in study.DATASETS:
        table=pd.read_csv(base.FIXED/ds/'fixed20/cycles.csv');keep=np.load(base.COHORT/f'{ds}_indices.npy')
        for model,seeds in [('ArcFENet_reconstructed',[0,1,2]),('HYDRA_paper',[0]),('ArcFusionNet_current',[0,1,2])]:
            for seed in seeds:
                scores=np.zeros(len(table));pred=np.zeros(len(table),bool);coverage=np.zeros(len(table),int)
                for fold in range(5):
                    _,tr,te=base.indices(ds,fold)
                    q=study.location(ds,fold,seed) if model=='ArcFusionNet_current' else base.OUT/'runs'/ds/model/f'f{fold}_s{seed}'
                    ids=np.load(q/'indices.npz');z=np.load(q/'predictions.npz');receipt=json.loads((q/'COMPLETE.json').read_text())
                    assert np.array_equal(ids['train'],tr) and np.array_equal(ids['test'],te)
                    assert np.array_equal(z['indices'],te) and np.array_equal(z['labels'],table.label.to_numpy()[te])
                    if model=='ArcFusionNet_current':
                        state=np.load(q/'preprocessing.npz');assert np.array_equal(state['fit_indices'],tr)
                        d=base.FIXED/ds/'fixed20';x=np.load(d/'temporal.npy',mmap_mode='r');sp=np.load(d/'spectrum.npy',mmap_mode='r')
                        bins=state['harmonic_indices'];assert len(bins)==32 and len(np.unique(bins))==32
                        np.testing.assert_allclose(state['temporal_mean'],x[tr].mean((0,3),keepdims=True),rtol=1e-6,atol=1e-6)
                        np.testing.assert_allclose(state['temporal_std'],np.maximum(x[tr].std((0,3),keepdims=True),1e-6),rtol=1e-6,atol=1e-6)
                        selected=sp[:,bins]
                        np.testing.assert_allclose(state['spectral_mean'],selected[tr].mean(0),rtol=1e-6,atol=1e-6)
                        np.testing.assert_allclose(state['spectral_std'],np.maximum(selected[tr].std(0),1e-6),rtol=1e-6,atol=1e-6)
                        assert receipt['protocol_sha256']==base.sha(out/'protocol.json')
                        for n,key in [('model.pt','model_sha256'),('preprocessing.npz','preprocessing_sha256'),('predictions.npz','predictions_sha256')]:
                            assert base.sha(q/n)==receipt[key]
                        forward=json.loads((q/'FORWARD_AUDIT.json').read_text())
                        assert forward['model_sha256']==base.sha(q/'model.pt') and forward['decision_changes']==0
                        assert forward['portable_model_sha256']==base.sha(q/'model_portable.pt')
                        value=z['probability'];threshold=.5;model_file=q/'model_portable.pt'
                    else:
                        assert receipt['protocol_sha256']==base.sha(base.OUT/'protocol.json')
                        assert receipt['predictions_sha256']==base.sha(q/'predictions.npz')
                        model_file=q/('model.npz' if model=='HYDRA_paper' else 'model.pt')
                        expected=old_audit[(old_audit.dataset==ds)&(old_audit.model==model)&(old_audit.fold==fold)&(old_audit.seed==seed)].iloc[0]
                        assert expected.checkpoint_sha256==base.sha(model_file)
                        value=z['score'];threshold=0 if model=='HYDRA_paper' else .5
                    coverage[te]+=1;scores[te]=value;pred[te]=value>=threshold
                    registry.append(dict(dataset=ds,baseline_id=names[model],model=model,seed=seed,fold=fold,
                        train=len(tr),test=len(te),folder=str(q),weights_sha256=base.sha(model_file),
                        weights_path=str(model_file),
                        checkpoint_format='L1_SVM_npz' if model=='HYDRA_paper' else 'standalone_state_dict',
                        score_kind='hinge_margin' if model=='HYDRA_paper' else 'positive_class_softmax',
                        prediction_sha256=base.sha(q/'predictions.npz'),indices_sha256=base.sha(q/'indices.npz'),
                        threshold=threshold,conditional_reconstruction=(model=='ArcFENet_reconstructed'),
                        current_only_adaptation=(model=='ArcFusionNet_current')))
                assert coverage.sum()==len(keep) and (coverage[keep]==1).all()
                y=table.iloc[keep].label.to_numpy();groups=table.iloc[keep].condition_key.to_numpy()
                m=cycle_metrics(y,scores[keep],groups,threshold)
                rows.append(dict(dataset=ds,baseline_id=names[model],model=model,seed=seed,**m))
                for key,g in table.iloc[keep].assign(score=scores[keep]).groupby('condition_key'):
                    mm=cycle_metrics(g.label.to_numpy(),g.score.to_numpy(),g.condition_key.to_numpy(),threshold)
                    conditions.append(dict(dataset=ds,model=model,seed=seed,condition=key,BAcc=mm['BAcc_condition']))
                np.savez(out/f'{ds}_{model}_s{seed}_oof.npz',indices=keep,labels=y,scores=scores[keep],prediction=pred[keep])
    df=pd.DataFrame(rows);df.to_csv(out/'seed_metrics.csv',index=False)
    fields=['BAcc_condition','BAcc_pooled','Accuracy','Precision','Recall','Specificity','F1','Macro_F1','FPR','FNR','ROC_AUC','AP','MCC']
    group=df.groupby(['dataset','baseline_id','model'],sort=False)
    mean=group[fields].mean().reset_index();sd=group[fields].std().reset_index()
    info=group.agg(seeds=('seed','nunique'),N=('N','first'),groups=('groups','first')).reset_index()
    mean=mean.merge(info,on=['dataset','baseline_id','model']);mean.to_csv(out/'main_results.csv',index=False)
    sd.to_csv(out/'seed_sd.csv',index=False)
    df[['dataset','baseline_id','seed','N','TP','FP','TN','FN']].to_csv(out/'confusion_per_seed.csv',index=False)
    cond=pd.DataFrame(conditions);cond.to_csv(out/'condition_metrics.csv',index=False)
    ci=[]
    for (ds,model),g in cond.groupby(['dataset','model']):
        v=g.groupby('condition').BAcc.mean().to_numpy();rng=np.random.default_rng(20261029)
        boot=v[rng.integers(len(v),size=(5000,len(v)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
        ci.append(dict(dataset=ds,model=model,lower=lo,upper=hi,unit='condition',seeds_averaged_first=True))
    pd.DataFrame(ci).to_csv(out/'bacc_condition_intervals.csv',index=False)
    base.save(out/'baseline_registry.json',dict(status='frozen_comparators',cohort='current reviewed grouped',
        entries=registry,models=names,fidelity='FENet reconstruction conditional; HYDRA config reproduction; Fusion current-only adaptation',
        later_comparisons='same cohorts/splits/thresholds/seeds/settings; do not replace with RCR-M0 or author alternating scores'))
    base.save(out/'AUDIT_PASS.json',dict(status='pass',outer_fits=len(registry),datasets=2,models=3,
        train_test_record_source_condition_disjoint=True,all_same_current_training_indices=True,OOF_unique=True,
        train_only_preprocessing=True,source_and_weight_identities=True,
        complete_original_author_methods=False,Fusion_full_voltage_method=False,
        FENet_numerics='conditional; see training-only sensitivity audit',gpu0_used=False))
    base.save(out/'RESULTS_COMPLETE.json',dict(status='complete',new_Fusion_fits=24,index_exact_Fusion_reuses=6,
        outer_fits_in_registry=70,summary_rows=6))
    print(mean.to_string(index=False))


if __name__=='__main__':main()
