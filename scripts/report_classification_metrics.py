"""All standard cycle metrics for new baselines and current adapted C0 studies."""
import numpy as np
import pandas as pd
import complete_paper_baselines as study
from arc_model.evaluation.cycle_metrics import cycle_metrics

OUT=study.OUT
METRICS=['BAcc_condition','BAcc_pooled','Accuracy','Precision','Recall','Specificity','F1','Macro_F1','FPR','FNR','ROC_AUC','AP','MCC']


def main():
    rows=[]
    for ds in study.DATASETS:
        tab=pd.read_csv(study.FIXED/ds/'fixed20/cycles.csv');keep=np.load(study.COHORT/f'{ds}_indices.npy')
        y=tab.iloc[keep].label.to_numpy();groups=tab.iloc[keep].condition_key.to_numpy()
        for model,seeds in [('ArcFENet_reconstructed',[0,1,2]),('HYDRA_paper',[0])]:
            for seed in seeds:
                z=np.load(OUT/f'{ds}_{model}_s{seed}_oof.npz')
                assert np.array_equal(z['indices'],keep) and np.array_equal(z['labels'],y)
                threshold=0 if model=='HYDRA_paper' else .5
                m=cycle_metrics(y,z['score'],groups,threshold)
                assert np.array_equal(z['prediction'],z['score']>=threshold)
                rows.append(dict(dataset=ds,model=model,arm='no_research_enhancement',seed=seed,threshold=threshold,**m))
        # Historical current continuous studies, explicitly distinct baselines.
        for model in ['ArcFENet','HYDRA','ArcFusionNet']:
            for seed in [0,1,2]:
                z=np.load(study.COHORT/f'{ds}_{model}_seed{seed}_oof.npz')
                assert np.array_equal(z['indices'],keep) and np.array_equal(z['labels'],y)
                for arm in ['M0','legacy_saved','uniform36','dynamic32']:
                    m=cycle_metrics(y,z[arm],groups,.5)
                    rows.append(dict(dataset=ds,model=model+'_RCR_adapted',arm=arm,seed=seed,threshold=.5,**m))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'all_classification_metrics_per_seed.csv',index=False)
    group=frame.groupby(['dataset','model','arm'],sort=False)
    means=group[METRICS].mean().reset_index();sds=group[METRICS].std().reset_index()
    info=group.agg(seeds=('seed','nunique'),N=('N','first'),groups=('groups','first')).reset_index()
    means=means.merge(info,on=['dataset','model','arm']);means.to_csv(OUT/'all_classification_metrics_mean.csv',index=False)
    sds.to_csv(OUT/'all_classification_metrics_seed_sd.csv',index=False)
    frame[['dataset','model','arm','seed','N','TP','FP','TN','FN']].to_csv(OUT/'confusion_counts_per_seed.csv',index=False)
    # Verify every historical metric under the unchanged recipe.
    old=pd.read_csv(study.COHORT/'seed_metrics.csv')
    for r in frame[frame.model.str.endswith('_RCR_adapted')].itertuples():
        m=r.model.removesuffix('_RCR_adapted')
        a=old[(old.dataset==r.dataset)&(old.model==m)&(old.arm==r.arm)&(old.seed==r.seed)].iloc[0]
        for now,previous in [('BAcc_condition','BAcc'),('Recall','Recall'),('FPR','FPR'),('F1','F1'),('AP','AP')]:
            assert abs(getattr(r,now)-a[previous])<1e-12
    lines=['# 周波级主流分类指标与完整结果','',
        '2026-10-29。正类为电弧（1），正常为0。全部来自各自测试折OOF预测重新计算，固定判定阈值，不按测试F1选阈值。',
        '', '## 指标定义与报告口径','',
        '| 指标 | 定义/意义 | 当前口径 |','|---|---|---|',
        '| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | 全部保留周波合并；易受类别比例影响 |',
        '| Precision | $TP/(TP+FP)$，报为电弧中有多少是真的 | 电弧正类 |',
        '| Recall / Sensitivity / TPR | $TP/(TP+FN)$，电弧检出比例 | 电弧正类 |',
        '| Specificity / TNR | $TN/(TN+FP)$，正常正确排除比例 | 正常类 |',
        '| F1 | $2TP/(2TP+FP+FN)$ | 电弧正类F1；另存Macro-F1 |',
        r'| FPR | $FP/(FP+TN)=1-\mathrm{Specificity}$ | 正常周波误报比例，不是每小时误报警次数 |',
        r'| FNR | $FN/(TP+FN)=1-\mathrm{Recall}$ | 电弧周波漏检比例 |',
        r'| pooled BAcc | $(\mathrm{Recall}+\mathrm{Specificity})/2$ | 周波合并后计算 |',
        r'| 条件等权BAcc | $G^{-1}\sum_{g=1}^{G}(\mathrm{Recall}_g+\mathrm{Specificity}_g)/2$ | 主指标，每条件等权 |',
        '| ROC-AUC | ROC曲线面积；跨阈值排序区分能力 | OOF原始分数；SVM用未校准margin |',
        '| PR-AUC（AP） | 按召回增量加权的Precision平均 | sklearn average_precision_score，不与梯形积分混称 |',
        r'| MCC | $\frac{TP\,TN-FP\,FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | 保存至CSV，范围[-1,1] |',
        '', 'TP是真电弧报电弧，FP是正常报电弧，TN是正常报正常，FN是电弧报正常。',
        '表中除MCC外均乘100；AUC/AP以百分数展示仅为格式转换。',
        '每种子先合并五折，得到一套完整OOF指标；表内再对种子取均值，SD另存CSV。',
        '种子不增加样本量，混淆矩阵按种子分别保留，不将重复预测叠成3倍样本。',
        '组间分数刻度可能不同，尤其SVM margin未跨折概率校准；AUC/AP与固定阈值指标分别解读。',
        '', '## 无本研究增强的新基线','',
        'ArcFENet为三种子论文重建；HYDRA是作者固定随机变换seed=0、原文FD配置和L1-SVM目标。完整ArcFusionNet原方法缺同步供电电压，未代填。','']
    columns=['BAcc_condition','Accuracy','Precision','Recall','F1','Specificity','FPR','ROC_AUC','AP']
    def table(t, key):
        v=['| 数据集 | '+key+' | BAcc条件 | Acc | Precision | Recall | F1 | Specificity | FPR↓ | ROC-AUC | AP |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in t.to_dict('records'):
            name=r['model'] if key=='模型' else r['model'].removesuffix('_RCR_adapted')+' / '+{'M0':'RCR-M0','legacy_saved':'旧C0','uniform36':'固定36','dynamic32':'去标记32'}[r['arm']]
            v.append('| '+('自采5 m' if r['dataset']=='continuous5m' else '公共')+' | '+name+' | '+' | '.join(f"{r[c]*100:.2f}" for c in columns)+' |')
        return v
    base=means[means.arm.eq('no_research_enhancement')];lines+=table(base,'模型')
    lines+=['','## RCR适配体系：原参照与动态增强','',
        '**这些是旧适配骨干，尤其HYDRA并非上表的新原文配置SVM。不能把旧C0成绩认作新SVM的增强结果。**','']
    for ds in study.DATASETS:
        lines+=table(means[means.dataset.eq(ds)&~means.arm.eq('no_research_enhancement')],'模型/方案')+['']
    lines+=['## 使用建议','',
        '论文主表报告条件等权BAcc、Recall、FPR、F1、AP；同时给出Accuracy、Precision、Specificity和ROC-AUC以便与电弧文献对照。',
        '二分类不平衡时不只报告Accuracy；高Recall同时高FPR不等于可靠检测。AUC/AP是排序指标，也不能替代固定阈值误报率。',
        '当前不计算报警事件时延、9周波规则或每小时报警率：筛选窗口主表不具备该层评价所需的完整覆盖与精确事件真值。',
        '', '本地CSV：`outputs/paper_baseline_completion_20261029/all_classification_metrics_{per_seed,mean,seed_sd}.csv`；',
        '`confusion_counts_per_seed.csv`为每种子整数混淆计数。',
        '重算：`python scripts/report_classification_metrics.py`。']
    text='\n'.join(lines)+'\n';(OUT/'论文分类指标_完整表格.md').write_text(text)
    (study.ROOT/'docs/CLASSIFICATION_METRICS.md').write_text(text)
    study.save(OUT/'METRICS_AUDIT.json',dict(status='pass',per_seed_rows=len(frame),summary_rows=len(means),
        unchanged_historical_recipe=True,positive_class='arc=1',thresholds={'HYDRA_paper':0,'probabilities':.5},
        confusion_counts_not_multiplied_by_seed=True))
    print(base[['dataset','model']+columns].to_string(index=False))


if __name__=='__main__':main()
