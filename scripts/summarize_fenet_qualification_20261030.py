"""Audit and report the FENet reconstruction qualification without selecting by test score."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import _bootstrap as boot
from qualify_fenet_20261030 import ROOT,OUT,DS,sha,location,indices,save
from arc_model.evaluation.cycle_metrics import cycle_metrics
from check_baseline_release import verify


def main():
    verify();spec=json.loads((OUT/'protocol.json').read_text());code={r['path']:r['sha256'] for r in spec['code']}
    if (OUT/'execution_revision_1.json').exists():
        a=json.loads((OUT/'execution_revision_1.json').read_text());b=json.loads((OUT/'execution_revision_2.json').read_text())
        assert code['scripts/qualify_fenet_20261030.py']==a['old_code_sha256'] and a['new_code_sha256']==b['old_code_sha256']
        snapshots=json.loads((OUT/'executed_source_snapshot.json').read_text())
        executed=next(row for row in snapshots if row['path']=='scripts/qualify_fenet_20261030.py')
        assert executed['sha256']==b['new_code_sha256'] and sha(ROOT/executed['snapshot'])==executed['sha256']
    else:
        assert code['scripts/qualify_fenet_20261030.py']==sha(ROOT/'scripts/qualify_fenet_20261030.py')
    assert code['scripts/fenet_vmd_graph_20261030.py']==sha(ROOT/'scripts/fenet_vmd_graph_20261030.py')
    for r in spec['identities']:assert sha(r['path'])==r['sha256']
    cap=[];seeds=[];groups=[];registrations=[];diag=[]
    for ds in DS:
        tab=pd.read_csv(ROOT/f'outputs/rcr_three_backbones_20261028/data/{ds}/fixed20/cycles.csv');keep=np.load(ROOT/f'outputs/dynamic_restore_20261029/{ds}_indices.npy');y=tab.label.to_numpy();g=tab.condition_key.to_numpy()
        for version,path in [('frozen_v1',ROOT/f'outputs/rcr_three_backbones_20261028/data/{ds}/fixed20/vmd_diagnostics.npz'),('repaired_v2',OUT/'data'/ds/'repaired_vmd_diagnostics.npz')]:
            z=np.load(path);c=np.sort(z['centers'][keep],axis=1)
            diag.append(dict(dataset=ds,version=version,N=len(keep),converged=int(z['converged'][keep].sum()),convergence_fraction=float(z['converged'][keep].mean()),median_iterations=float(np.median(z['iterations'][keep])),middle_median_hz=float(np.median(c[:,1])),high_median_hz=float(np.median(c[:,2])),high_98_112k_fraction=float(((c[:,2]>=98000)&(c[:,2]<=112000)).mean())))
        for version in ['frozen_v1','repaired_v2']:
            for seed in [0,1,2]:
                job=dict(dataset=ds,scope='same_record_capability',version=version,seed=seed,fold=-1);q=location(job);r=json.loads((q/'COMPLETE.json').read_text());p=np.load(q/'predictions.npz');ix=np.load(q/'indices.npz');assert sha(q/'predictions.npz')==r['predictions_sha256']
                tr,te,val=indices(job);assert np.array_equal(ix['train'],tr) and np.array_equal(ix['test'],te) and np.array_equal(ix['validation'],val)
                assert not np.intersect1d(np.r_[tr,val],te).size
                for part,where in [('test',q/'predictions.npz'),('validation',q/'validation.npz')]:
                    v=np.load(where);m=cycle_metrics(v['labels'],v['score'],g[v['indices']]);cap.append(dict(dataset=ds,version=version,seed=seed,partition=part,**m))
                m=json.loads((q/'train_metrics.json').read_text());cap.append(dict(dataset=ds,version=version,seed=seed,partition='train',**m))
        for seed in [0,1,2]:
            cover=np.zeros(len(tab),int);score=np.full(len(tab),np.nan)
            for fold in range(5):
                job=dict(dataset=ds,scope='grouped',version='repaired_v2',seed=seed,fold=fold);q=location(job);receipt=json.loads((q/'COMPLETE.json').read_text());tr,te,_=indices(job);ix=np.load(q/'indices.npz');z=np.load(q/'predictions.npz')
                assert np.array_equal(tr,ix['train']) and np.array_equal(te,ix['test']) and np.array_equal(te,z['indices']) and np.array_equal(y[te],z['labels'])
                assert sha(q/'model.pt')==receipt['model_sha256'] and sha(q/'predictions.npz')==receipt['predictions_sha256']
                assert receipt['input_sha256']==json.loads((OUT/'data'/ds/'IMAGES_COMPLETE.json').read_text())['rgb_sha256']
                for key in ['record_id','source_group','condition_key']:assert not set(tab.iloc[tr][key])&set(tab.iloc[te][key])
                assert len(json.loads((q/'history.json').read_text()))==20
                cover[te]+=1;score[te]=z['score'];registrations.append(dict(dataset=ds,baseline_id='B_FENET_fixed20_v2_numrepair',model='ArcFENet_reconstructed_numrepair',fold=fold,seed=seed,folder=str(q),weights_path=str(q/'model.pt'),weights_sha256=receipt['model_sha256'],input_sha256=receipt['input_sha256'],indices_sha256=sha(q/'indices.npz'),prediction_sha256=receipt['predictions_sha256'],threshold=.5,conditional_reconstruction=True))
            assert (cover[keep]==1).all() and cover.sum()==len(keep)
            m=cycle_metrics(y[keep],score[keep],g[keep]);seeds.append(dict(dataset=ds,model='ArcFENet_reconstructed_numrepair',seed=seed,**m))
            np.savez(OUT/f'{ds}_repaired_v2_s{seed}_oof.npz',indices=keep,labels=y[keep],scores=score[keep])
            old=np.load(ROOT/f'outputs/grouped_baseline_standard_20261029/{ds}_ArcFENet_reconstructed_s{seed}_oof.npz');assert np.array_equal(old['indices'],keep)
            for key in np.unique(g[keep]):
                local=g[keep]==key;ixs=keep[local]
                new=cycle_metrics(y[ixs],score[ixs],g[ixs]);prev=cycle_metrics(y[ixs],old['scores'][local],g[ixs]);groups.append(dict(dataset=ds,seed=seed,condition=key,new_BAcc=new['BAcc_condition'],old_BAcc=prev['BAcc_condition'],delta=new['BAcc_condition']-prev['BAcc_condition'],Recall=new['Recall'],FPR=new['FPR']))
    fields=['BAcc_condition','BAcc_pooled','Accuracy','Precision','Recall','Specificity','F1','FPR','FNR','ROC_AUC','AP','MCC']
    cap=pd.DataFrame(cap);seed=pd.DataFrame(seeds);group=pd.DataFrame(groups)
    cap.to_csv(OUT/'capability_seed_results.csv',index=False);cap.groupby(['dataset','version','partition'])[fields].mean().to_csv(OUT/'capability_means.csv');cap.groupby(['dataset','version','partition'])[fields].std().to_csv(OUT/'capability_seed_sd.csv')
    seed.to_csv(OUT/'grouped_seed_results.csv',index=False);seed.groupby('dataset')[fields].mean().to_csv(OUT/'grouped_means.csv');seed.groupby('dataset')[fields].std().to_csv(OUT/'grouped_seed_sd.csv');group.to_csv(OUT/'grouped_condition_results.csv',index=False);pd.DataFrame(diag).to_csv(OUT/'numerical_diagnostics.csv',index=False)
    paired=[]
    for ds,x in group.groupby('dataset'):
        d=x.groupby('condition').delta.mean().to_numpy();rng=np.random.default_rng(20261030);samples=d[rng.integers(len(d),size=(5000,len(d)))].mean(1);lo,hi=np.quantile(samples,[.025,.975]);paired.append(dict(dataset=ds,delta=float(d.mean()),lower=float(lo),upper=float(hi),unit='condition',seeds_averaged_first=True,interpretation='exploratory descriptive paired interval'))
    pd.DataFrame(paired).to_csv(OUT/'grouped_paired_intervals.csv',index=False)
    save(OUT/'baseline_v2_registry.json',dict(status='versioned_numerical_reconstruction',entries=registrations,old_baselines_unchanged=True,active_catalog_not_silently_changed=True,replication_scope='same published network and budget; declared numerical adapter/VMD choices; not unavailable author data/code equivalence'))
    records=list((OUT/'records').rglob('NATIVE_COMPLETE.json'));assert len(records)==151
    source=[]
    for p in records:
        r=json.loads(p.read_text());assert r['source_full_hash_rechecked'];source.append(dict(dataset=r['dataset'],record_id=r['record_id'],source=r['original_source'],sha256=r['source_sha256'],native_fs=r['fs']))
    pd.DataFrame(source).to_csv(OUT/'fresh_source_identity.csv',index=False)
    save(OUT/'RESULTS_COMPLETE.json',dict(status='complete',capability_fits=12,grouped_fits=30,fresh_original_sources=151,old_frozen_unchanged=True,GRAPH_PARITY_500=True,GRAPH_PARITY_2000=True,physical_gpu0_used=False,code_execution_revisions=['finite check buffer memory','duration loop variable'],affected_old_capability_durations_excluded=True,classification_results_unchanged_by_execution_fixes=True))
    report='ArcFENet资格验证与数值修订结果\n\n同一网络/HFDC/DDC/20轮训练预算，版本化FIR与VMD迭代修订；相同138/13原始来源、38374/1361审核队列，未删除不收敛样本。先固定技术修订，再评价分类；不按外折成绩选择版本。\n\n同源窗口能力\n'+cap[cap.partition=='test'].groupby(['dataset','version'])[fields].mean().to_string()+'\n\n五折三种子分组主表新版本\n'+seed.groupby('dataset')[fields].mean().to_string()+'\n\n与旧冻结版本的条件配对差\n'+pd.DataFrame(paired).to_string(index=False)+'\n\n数值行为\n'+pd.DataFrame(diag).to_string(index=False)+'\n\n同源能力不能替代未见条件泛化；数值迭代上限不收敛仍标记。没有作者数据/代码，保留条件重建名称。旧版本未覆盖，新模型按baseline_v2_registry.json精确权重与输入身份比较。旧能力核验耗时字段存在计时变量覆盖，已单独勘误，不用于成本结论；预测不受影响。\n'
    (OUT/'完整结果与解释.txt').write_text(report);print(report,flush=True)

if __name__=='__main__':main()
