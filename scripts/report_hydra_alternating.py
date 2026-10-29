"""Summarize the author-style alternating test without replacing grouped results."""
import json
import pandas as pd
import validate_hydra_alternating as s


def main():
    p=s.OUT;means=pd.read_csv(p/'main_results.csv').set_index('dataset')
    sd=pd.read_csv(p/'repeat_sd.csv').set_index('dataset')
    refs=pd.read_csv(p/'grouped_references.csv');counts=pd.read_csv(p/'cohort_counts.csv').set_index('dataset')
    per=pd.read_csv(p/'repeat_metrics.csv')
    sparse_rows=[]
    for ds in s.DATASETS:
        for repeat in range(1,6):
            q=p/'runs'/ds/f'repeat{repeat}'
            state=json.loads((q/'solver.json').read_text())
            sparse_rows.append(dict(dataset=ds,repeat=repeat,**state))
    pd.DataFrame(sparse_rows).to_csv(p/'solver_objectives_and_sparsity.csv',index=False)
    lines=['# HYDRA交替窗口协议验证','',
        '## 目的与完成状态','',
        '核验按原论文同一连续录波交替切窗，是否可达到作者约99.63%的Accuracy。',
        '完成两个当前连续数据集各5次分层训练留出，作者MAT另5次；共15次精确L1-hinge SVM拟合，未加入PLL、C0或额外特征。',
        '', '## 主要结果','',
        '| 输入与评价路径 | 训练候选窗 | 固定测试窗 | Accuracy（均值±SD） | 条件等权BAcc | Recall | FPR↓ | F1 | ROC-AUC | AP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    names={'continuous5m':'自采审核窗口，交替划分','zenodo5501527':'公共审核窗口，交替划分','author_mat':'作者发布MAT及原划分'}
    for ds in s.DATASETS:
        r=means.loc[ds];d=sd.loc[ds];n=counts.loc[ds]
        b='—（无逐窗电器映射）' if ds=='author_mat' else f'{r.BAcc_condition*100:.3f}%'
        lines.append(f'| {names[ds]} | {int(n.train_pool)} | {int(n.test)} | **{r.Accuracy*100:.4f}±{d.Accuracy*100:.4f}%** | {b} | '+
            ' | '.join(f'{r[c]*100:.3f}%' for c in ['Recall','FPR','F1','ROC_AUC','AP'])+' |')
    author=means.loc['author_mat'];public=means.loc['zenodo5501527'];target=.996252
    lines+=['','作者归档最佳Accuracy为99.6252%（论文约99.63%）。',
        f'作者MAT本地复现与归档差{(author.Accuracy-target)*100:+.4f}个百分点；是否达到99%：'+('是。' if author.Accuracy>=.99 else '否。'),
        '当前公共审核窗口按交替协议是否达到99%：'+('是。' if public.Accuracy>=.99 else '否。'),
        '', '## 相同当前输入，分组评价与交替评价','',
        '| 数据集 | 整文件隔离，全部审核窗口Accuracy | 整文件隔离，仅相同交替测试窗Accuracy | 交替训练，固定测试窗Accuracy |',
        '|---|---:|---:|---:|']
    comparisons=[]
    for ds in s.DATASETS[:2]:
        old=refs[(refs.dataset==ds)&refs.reference.eq('grouped_full_reviewed')].iloc[0]
        match=refs[(refs.dataset==ds)&refs.reference.eq('grouped_same_alternating_test')].iloc[0]
        r=means.loc[ds];d=sd.loc[ds]
        lines.append(f'| {"自采5 m" if ds=="continuous5m" else "公共"} | {old.Accuracy*100:.4f}%（n={int(old.N)}） | {match.Accuracy*100:.4f}%（n={int(match.N)}） | {r.Accuracy*100:.4f}±{d.Accuracy*100:.4f}% |')
        comparisons.append(dict(dataset=ds,grouped_full_accuracy=old.Accuracy,grouped_same_test_accuracy=match.Accuracy,
            alternating_accuracy=r.Accuracy,delta_same_test_pp=100*(r.Accuracy-match.Accuracy)))
    pd.DataFrame(comparisons).to_csv(p/'protocol_comparison.csv',index=False)
    lines+=['','上述最后两列评价相同测试窗口，可避免把测试集合不同误认为性能差异。',
        '训练集组成与规模仍不同：分组模型不见测试电器/条件；交替模型见同录波的其他窗口，且仅使用训练候选的80%。',
        '因此它比较两种评价协议的整体影响，不把差值完全归为某一个实现细节。',
        '', '## 每次重复的Accuracy与错误数量','',
        '| 路径 | 重复 | Accuracy | FP | FN | 测试总数 |','|---|---:|---:|---:|---:|---:|']
    for r in per.itertuples():
        lines.append(f'| {names[r.dataset]} | {r.repeat} | {r.Accuracy*100:.4f}% | {r.FP} | {r.FN} | {r.N} |')
    lines+=['','## 固定实验定义','',
        '- 当前审核数据：沿用原生电流导出的50 kHz整流输入及已有1,024维HYDRA缓存；不修改标签、不按测试成绩排除电器。',
        '- 从原始连续文件的20 ms网格计算窗口序号：0、2、4…为训练候选，1、3、5…为固定测试。边界保护/审核排除后不重新编号。',
        '- 当前两数据集保留同一38,374/1,361个审核窗口，分别训练；公共交替训练池684窗、测试677窗。',
        '- 每次对训练候选做分层80%抽样，random_state=1…5；剩余20%不参与Scaler、拟合或调参；固定测试集不变。',
        '- 原文FD配置：20 ms、50 kHz、k8/g8/m5/stride1/divisor1、随机核seed0、lambda=0.001。',
        '- SparseScaler仅拟合当次80%训练子集；求解均值hinge加L1正则、自由截距的凸SVM，判定margin≥0为电弧。',
        '- 作者MAT直接沿用yTrain/yTest，均为1,334窗；使用未修改的作者CPU随机变换。NOARC=0、ARC=1已核对原文表2。',
        '- SVM使用CPU HiGHS精确求解，替代MATLAB SpaRSA；分层留出使用sklearn而非cvpartition。无神经网络训练，不需要占用GPU；所有worker均排除GPU0。',
        '', '## 解释与复现边界','',
        '交替划分检验同一录波/已见电器的窗口识别，不提供未见电器泛化证据。保留整文件隔离作为论文主评价，交替协议单列用于复现与解释差异。',
        '作者MAT与当前公共原始录波来自同一公开数据来源，不能合并成两个独立公共数据集。缺少可靠逐窗来源映射，不将MAT标签贴回DAT。',
        '当前审核窗口的标签范围、40 ms边界保护与本地因果重采样不同于作者MAT；仅改变划分不等于完整复刻作者预处理。',
        '五次重复共用固定测试样本，SD表示训练留出变动，不是五个独立测试集的不确定性；不据此制作独立样本显著性检验。',
        '原文曾搜索23,040个超参数组合；本次固定已公开最佳FD配置，不以本次测试结果选择lambda或核参数。',
        '作者MAT旧复现已有非零权重数量与作者不同的记录；本轮另存求解目标与稀疏度，不把接近的Accuracy解释为部署成本也已复现。',
        '不进行文件分数平均、投票、9周波报警或测试阈值调优；全部Accuracy为窗口级。',
        '', '## 产物与运行','',
        '服务器位置：`/data/hzy/experiments/ARC_model/outputs/hydra_alternating_protocol_20261029/`。',
        '`protocol.json`与首次冻结副本记录拟合前技术性表述修订；输入/划分/损失未改。',
        '`cohort_counts.csv`、`repeat_metrics.csv`、`main_results.csv`、`repeat_sd.csv`与`grouped_references.csv`保存完整指标。',
        '`runs/{dataset}/repeat{1..5}/`保存训练/丢弃/测试索引、Scaler、SVM权重、决策分数与求解器状态。',
        '`AUDIT_PASS.json`记录全部状态重放、训练内Scaler与原序号奇偶划分核对。',
        '', '```bash',
        'python scripts/validate_hydra_alternating.py --prepare',
        '# 三个并行worker；分类器是CPU线性规划，CUDA_VISIBLE_DEVICES仅明确排除GPU0。',
        'CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/validate_hydra_alternating.py --partition 0',
        'CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/validate_hydra_alternating.py --partition 1',
        'CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/validate_hydra_alternating.py --partition 2',
        'python scripts/validate_hydra_alternating.py --summarize',
        'python scripts/report_hydra_alternating.py',
        '```',
        '', '原论文：<https://doi.org/10.1109/OJIES.2025.3582482>，IV节、表2、表13；',
        '作者发布：<https://doi.org/10.5281/zenodo.15279701>。']
    text='\n'.join(lines)+'\n';(p/'交替切窗复现_完整结果与解释.md').write_text(text)
    (s.ROOT/'docs/HYDRA_ALTERNATING.md').write_text(text)
    print('\n'.join(lines[:22]))


if __name__=='__main__':main()
