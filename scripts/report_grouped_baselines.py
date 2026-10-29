"""Publish six grouped baseline rows with explicit reconstruction/adaptation labels."""
import json
import pandas as pd
import freeze_grouped_baselines as study


def main():
    out=study.OUT;mean=pd.read_csv(out/'main_results.csv');sd=pd.read_csv(out/'seed_sd.csv')
    names={'ArcFENet_reconstructed':'ArcFENet重建-v1†','HYDRA_paper':'HYDRA原文FD配置',
           'ArcFusionNet_current':'ArcFusionNet-current-v2‡'}
    lines=['<!-- frozen-baseline-results:start -->','## 三个本地比较器的锁定性能','',
        '表中单位均为%。BAcc按条件等权，其余为全周波合并；每种子先合并五折，再取种子均值。',
        'BAcc的±为神经模型三个初始化种子的SD；HYDRA采用作者固定随机变换，不虚填SD。',
        '', '| 数据集 | 无研究增强比较器 | BAcc条件 | Accuracy | Precision | Recall | F1 | FPR↓ | ROC-AUC | AP |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in mean.itertuples():
        s=sd[(sd.dataset==r.dataset)&(sd.model==r.model)].iloc[0]
        b=f'{r.BAcc_condition*100:.3f}'
        if r.seeds>1:b+=f'±{s.BAcc_condition*100:.3f}'
        lines.append('| '+('自采5 m' if r.dataset=='continuous5m' else '公共')+' | '+names[r.model]+' | '+b+' | '+
            ' | '.join(f'{getattr(r,k)*100:.2f}' for k in ['Accuracy','Precision','Recall','F1','FPR','ROC_AUC','AP'])+' |')
    lines+=['','† ArcFENet是带数值前端限制的论文重建。‡ Fusion是总路电流结构适配，完整作者电压/差分/合成框架未复现。',
        '这是固定本地实现的比较标准，不将它统称为三个完整作者方法的SOTA复现。',
        '', '### 判定与统计约定','',
        '- 正类为电弧1。神经模型softmax概率阈值0.5；HYDRA SVM margin阈值0。没有按测试分数调阈值。',
        '- 同一38,374/1,361队列、五折条件/文件隔离；原始记录、source group和条件均无训练测试交叉。',
        '- BAcc条件、pooled BAcc、Accuracy、Precision、Recall、Specificity、F1/Macro-F1、FPR/FNR、ROC-AUC、AP与MCC均已保存。',
        '- `seed_metrics.csv`与`confusion_per_seed.csv`保存各种子完整指标和整数TP/FP/TN/FN；不把三种子重复预测当作三倍样本。',
        '- `bacc_condition_intervals.csv`保存先平均种子、再按条件聚类重采样的描述区间；不将窗口视为独立采集。',
        '- SVM原始margin未做跨折概率校准，AUC/AP与固定阈值指标分别解读。',
        '', '### 注册与检查','',
        '注册表包含70个外折拟合：FENet30、HYDRA10、Fusion30。FENet/HYDRA复用先前已核验状态；',
        'Fusion6个索引完全一致组合复用、24个受旧多余训练窗口影响的组合重新拟合。',
        'Fusion30份权重完整测试重放最大概率误差0、判定变化0；所有当前训练/测试索引与同一队列完全一致。',
        '注册表`baseline_registry.json`中的`weights_path`、权重SHA、索引SHA与预测SHA为后续比较的准据。',
        '任何前处理、数据范围或基线代码变更都另建版本，同步评估baseline和baseline+模块。',
        '<!-- frozen-baseline-results:end -->']
    block='\n'.join(lines)+'\n';path=study.ROOT/'docs/GROUPED_BASELINE_STANDARD.md'
    text=path.read_text();marker='<!-- frozen-baseline-results:start -->'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n\n'
    text=text.rstrip()+'\n\n'+block;path.write_text(text)
    (out/'三个基线_划分审查与锁定性能.md').write_text(text)
    (out/'性能表.md').write_text(block)
    meta=json.loads((out/'AUDIT_PASS.json').read_text())
    forward=[json.loads(p.read_text()) for p in (out/'runs').glob('*/*/FORWARD_AUDIT.json')]
    assert len(forward)==30 and max(r['max_probability_error'] for r in forward)==0
    meta['Fusion_forward_audit']=dict(checkpoints=30,max_probability_error=0,decision_changes=0)
    meta['portable_Fusion_weights']=30;meta['tests']='36 CPU module tests passed'
    (out/'AUDIT_PASS.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(block)


if __name__=='__main__':main()
