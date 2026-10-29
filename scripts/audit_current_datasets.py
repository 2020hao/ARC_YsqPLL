"""Recompute dataset counts and leakage checks from the current frozen cohort."""
import json
import _bootstrap as boot
import numpy as np
import pandas as pd

ROOT=boot.ROOT
OUT=ROOT/'outputs/paper_baseline_completion_20261029/dataset_audit'
SOURCE=ROOT/'outputs/rcr_fundamental_consensus_20261028/detector'
COHORT=ROOT/'outputs/dynamic_restore_20261029'
RECORDS=ROOT/'outputs/continuous_pll_validation_20261027_v2/records'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=[];folds=[];devices=[];conditions=[]
    for ds in ['continuous5m','zenodo5501527']:
        tab=pd.read_csv(SOURCE/'data'/ds/'recovery/cycles.csv')
        keep=np.load(COHORT/f'{ds}_indices.npy');v=tab.iloc[keep]
        splits=json.loads((SOURCE/'data'/ds/'recovery/splits.json').read_text())
        docs=[json.loads(p.read_text()) for p in sorted((RECORDS/ds).glob('*/COMPLETE.json'))]
        assert len(docs)==tab.record_id.nunique()
        summary.append(dict(dataset=ds,native_fs=docs[0]['fs'],recordings=len(docs),
            unique_source_groups=tab.source_group.nunique(),conditions=v.condition_key.nunique(),
            target_appliances=v.appliance.nunique(),candidate_anchors=len(tab),
            normal=int(v.label.eq(0).sum()),arc=int(v.label.eq(1).sum()),N=len(v),
            excluded=len(tab)-len(v),duration_seconds=sum(d['duration_s'] for d in docs),
            original_guard40=int(tab.guard40.sum()),
            reviewed_extra_exclusions=int(tab.guard40.sum())-len(v),
            nominal_evaluated_seconds=.02*len(v)))
        for appliance,g in v.groupby('appliance',sort=True):
            devices.append(dict(dataset=ds,appliance=appliance,recordings=g.record_id.nunique(),
                conditions=g.condition_key.nunique(),normal=int(g.label.eq(0).sum()),arc=int(g.label.eq(1).sum()),N=len(g)))
        for key,g in v.groupby('condition_key',sort=True):
            conditions.append(dict(dataset=ds,condition=key,recordings=g.record_id.nunique(),
                normal=int(g.label.eq(0).sum()),arc=int(g.label.eq(1).sum()),N=len(g)))
        count=np.zeros(len(tab),int)
        for f,split in enumerate(splits):
            tr=np.array(split['train']);te=np.intersect1d(split['test'],keep)
            assert np.array_equal(tr,np.setdiff1d(keep,te))
            assert set(tab.iloc[tr].label)=={0,1} and set(tab.iloc[te].label)=={0,1}
            for col in ['record_id','condition_key','source_group']:
                assert not set(tab.iloc[tr][col])&set(tab.iloc[te][col])
            count[te]+=1
            a,b=tab.iloc[tr],tab.iloc[te]
            folds.append(dict(dataset=ds,fold=f,train=len(tr),train_normal=int(a.label.eq(0).sum()),
                train_arc=int(a.label.eq(1).sum()),test=len(te),test_normal=int(b.label.eq(0).sum()),
                test_arc=int(b.label.eq(1).sum()),train_groups=a.condition_key.nunique(),test_groups=b.condition_key.nunique(),
                train_recordings=a.record_id.nunique(),test_recordings=b.record_id.nunique()))
        assert (count[keep]==1).all() and count.sum()==len(keep)
    for name,rows in [('dataset_summary',summary),('fold_counts',folds),('appliance_counts',devices),('condition_counts',conditions)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'AUDIT.json').write_text(json.dumps(dict(status='pass',OOF_unique=True,
        source_record_condition_disjoint=True,train_is_current_reviewed_complement=True,
        source='current frozen cycles/splits and raw record receipts; counts not inferred from directory names'),indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(folds).to_string(index=False))


if __name__=='__main__':main()
