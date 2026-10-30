"""Recompute all twelve predeclared arms and their paired mechanism comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import _bootstrap as boot
from arc_model.evaluation.cycle_metrics import cycle_metrics
from summarize_locked_routes_20261030 import paired_interval, sha

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'
CONFIG=ROOT/'configs/modules/phase_dynamic_evidence_20261030.yaml'


def holm(frame,ids):
    ids=frame.loc[ids].sort_values('p').index
    frame.loc[ids,'p_holm']=np.maximum.accumulate([min(1.,frame.loc[k,'p']*(len(ids)-i)) for i,k in enumerate(ids)])


def main():
    cfg=yaml.safe_load(CONFIG.read_text());datasets=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())
    metrics=[];conditions=[];strata=[];changes=[];choices=[];files=[];completed=[]
    for ds in cfg['datasets']:
        c=datasets[ds];table=pd.read_csv(ROOT/c['prepared_roots']['current_self_sync']/'cycles.csv')
        keep=np.load(ROOT/c['reviewed_indices']);splits=json.loads((ROOT/c['split_file']).read_text())
        for model in cfg['models']:
            parts={a:[] for a in cfg['arms']}
            for fold in range(5):
                folder=OUT/'runs'/ds/model/f'f{fold}';receipt=json.loads((folder/'COMPLETE.json').read_text());completed.append(receipt)
                expected=np.intersect1d(splits[fold]['test'],keep);train=np.asarray(splits[fold]['train'])
                assert np.array_equal(np.sort(train),np.setdiff1d(keep,expected))
                for key in ['record_id','condition_key','source_group']:
                    assert not set(table.iloc[train][key])&set(table.iloc[expected][key])
                for arm in cfg['arms']:
                    p=folder/(arm+'.npz');z=np.load(p);ix=z['indices'];y=z['y'];score=z['margin']
                    assert np.array_equal(ix,expected) and np.array_equal(y,table.label.to_numpy()[ix])
                    assert np.isfinite(score).all()
                    assert sha(p)==receipt['prediction_sha256'][p.name]
                    parts[arm].append(pd.DataFrame(dict(index=ix,y=y,score=score)))
                    files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
                    sp=folder/(arm+'_state.npz')
                    if sp.exists():
                        state=np.load(sp)
                        choices.append(dict(dataset=ds,model=model,fold=fold,arm=arm,dimension=int(state['dimension']),strength=float(state['strength']),quantile=float(state['selected_quantile']) if 'selected_quantile' in state else np.nan))
            baseline=pd.concat(parts['R0_parent']).sort_values('index').score.to_numpy()>=0
            for arm in cfg['arms']:
                allp=pd.concat(parts[arm]).sort_values('index');assert np.array_equal(allp['index'],keep)
                y=allp.y.to_numpy();s=allp.score.to_numpy();p=s>=0;g=table.iloc[keep].condition_key.to_numpy()
                metrics.append(dict(dataset=ds,model=model,arm=arm,seed=0,**cycle_metrics(y,s,g,threshold=0)))
                for group in np.unique(g):
                    ix=g==group;recall=p[ix&(y==1)].mean();fpr=p[ix&(y==0)].mean()
                    conditions.append(dict(dataset=ds,model=model,arm=arm,condition=group,BAcc=.5*(recall+1-fpr),Recall=recall,FPR=fpr))
                changes.append(dict(dataset=ds,model=model,arm=arm,
                    corrected_FP=int(((y==0)&baseline&~p).sum()),new_FP=int(((y==0)&~baseline&p).sum()),
                    corrected_FN=int(((y==1)&~baseline&p).sum()),new_FN=int(((y==1)&baseline&~p).sum())))
                valid=table.iloc[keep].kind.eq('pll').to_numpy()
                for name,mask in [('valid_RCR',valid),('nominal_fallback',~valid)]:
                    yy=y[mask];pp=p[mask];r=float(pp[yy==1].mean()) if (yy==1).any() else np.nan;f=float(pp[yy==0].mean()) if (yy==0).any() else np.nan
                    strata.append(dict(dataset=ds,model=model,arm=arm,stratum=name,N=len(yy),Recall=r,FPR=f,BAcc_pooled=.5*(r+1-f)))
    m=pd.DataFrame(metrics);gc=pd.DataFrame(conditions);pairs=[]
    for (ds,model),sub in gc.groupby(['dataset','model']):
        pivot=sub.pivot(index='condition',columns='arm',values='BAcc')
        for role in ['primary','diagnostic']:
            for candidate,reference in cfg['evaluation'][role+'_pairs']:
                v=(pivot[candidate]-pivot[reference]).to_numpy()
                pairs.append(dict(dataset=ds,model=model,candidate=candidate,reference=reference,role=role,**paired_interval(v)))
    paired=pd.DataFrame(pairs);paired['p_holm']=np.nan
    holm(paired,paired.index[paired.role=='primary'])
    for _,sub in paired[paired.role=='diagnostic'].groupby(['candidate','reference']):holm(paired,sub.index)
    assert len(m)==72 and len(paired)==60 and len(completed)==30
    assert sum(r['head_fits'] for r in completed)==420
    for name,frame in [('main_results',m),('condition_metrics',gc),('paired_deltas',paired),
                        ('reference_strata',pd.DataFrame(strata)),('correction_transitions',pd.DataFrame(changes)),('selection_choices',pd.DataFrame(choices))]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    status=dict(status='metrics_recomputed',configuration_count=12,new_configuration_count=8,
                completed_cell_folds=30,new_outer_arm_bundles=240,new_linear_head_fits=420,
                parity_diagnostic_head_fits=1,new_backbone_fits=0,metric_rows=72,paired_rows=60,
                primary_tests=30,primary_Holm_family='all30',seed=0,
                not_independent_confirmation=True,source_predictions=files)
    (OUT/'SUMMARY.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print((100*m.pivot(index=['dataset','model'],columns='arm',values='BAcc_condition')).round(3).to_string())
    print(paired[paired.role=='primary'].to_string(index=False))


if __name__=='__main__':main()
