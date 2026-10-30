"""Generate route tables and evidence-scoped explanations from completed results."""
from pathlib import Path
import json
import pandas as pd
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/locked_routes_20261030'
NAMES={'continuous5m':'自采5m','zenodo5501527':'公共数据'}
MODELS=['ArcFENet','HYDRA','ArcFusionNet']
ARMS=['baseline','calibration_only','dual_fixed20','dual_rcr','single_direct','single_retrained']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    assert summary['status']=='complete' and summary['metric_rows']==36
    m=pd.read_csv(OUT/'main_results.csv')
    p=pd.read_csv(OUT/'paired_deltas.csv')
    result=m.pivot(index=['dataset','model'],columns='arm',values='BAcc_condition')
    primary=[];trade=[];phase=[];pairs=[]
    for ds in NAMES:
        for model in MODELS:
            v=result.loc[(ds,model)]
            primary.append([NAMES[ds],model]+[f'{100*v[a]:.3f}' for a in ARMS])
            sub=m[(m.dataset==ds)&(m.model==model)].set_index('arm')
            trade.append([NAMES[ds],model]+[f'{100*sub.loc[a,"Recall"]:.2f} / {100*sub.loc[a,"FPR"]:.2f}' for a in ['baseline','dual_rcr','single_direct','single_retrained']])
            for candidate in ['dual_rcr','single_direct','single_retrained']:
                x=p[(p.dataset==ds)&(p.model==model)&(p.candidate==candidate)&(p.reference=='baseline')].iloc[0]
                pairs.append([NAMES[ds],model,candidate,f'{x.delta_pp:+.3f}',f'[{x.lo_pp:+.3f}, {x.hi_pp:+.3f}]',f'{x.p_holm:.4f}'])
            x=p[(p.dataset==ds)&(p.model==model)&(p.candidate=='dual_rcr')&(p.reference=='dual_fixed20')].iloc[0]
            phase.append([NAMES[ds],model,f'{x.delta_pp:+.3f}',f'[{x.lo_pp:+.3f}, {x.hi_pp:+.3f}]',f'{x.p_holm:.4f}'])
    primary_table=table(['数据集','模型','父基线seed0','仅校准','固定窗动态','RCR动态双分支','输入直接插入','对齐输入重训'],primary)
    trade_table=table(['数据集','模型','父基线 Recall/FPR','RCR双分支 Recall/FPR','直接输入 Recall/FPR','输入重训 Recall/FPR'],trade)
    phase_table=table(['数据集','模型','RCR−固定窗(pp)','95%条件配对区间(pp)','Holm p'],phase)
    pair_table=table(['数据集','模型','方案','相对父基线Δ(pp)','95%条件配对区间(pp)','Holm p'],pairs)
    text='''# 冻结父基线上的双分支和输入模块：完整seed0验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-10-30
- Verification Status: VERIFIED（保存状态前向重放及逐窗指标重算；非独立新数据确认）
- Version Label: locked_routes_20261030_seed0

两个数据集、三个独立骨干、五折、seed0的预定矩阵已完成。正式baseline仍保持原先
ArcFENet/Fusion三个种子的冻结表；本轮所有增量只与同seed0父模型配对。两数据集
分别拟合、分别评价，审核窗口仍为38,374/1,361，不混入旧裁剪数据。

## 六组完整结果

以下为条件等权周波BAcc，单位%。双分支使用相同冻结固定窗骨干，只改变旁路的
36维历史表征；单模块改变原输入，保持网络结构和参数维度。

'''+primary_table+'''
RCR动态双分支相对原模型五个组合均值正向；输入插件的直接插入和重训效果依赖
数据集与骨干。本轮保留全部预定组合，不以外测成绩自动选择每个电器/模型的路线。

## 相位坐标能否单独解释增益

固定窗与RCR动态使用相同10–100/100–400kHz频带、1/2/4/8周波历史、36维统计和
同容量纠错头。相对这个同信息对照，RCR三组均值更高、三组更低，六项区间均跨零：

'''+phase_table+'''
当前更明确的作用是：在原单窗骨干之外，加入原生高频跨周波动态并校正分数。
相位专属增量尚未建立，因此论文应把“能够建立坐标”与“坐标必然改善检测”分开。
纯输入重训还同时改变窗口支持、相位归一化插值及后续VMD/FFT条件，其总增益也
不能单独归因于相位位置。对齐保留故障畸变，改变的是参考位置与重采样坐标。

## 误报与召回的权衡

下表每格为电弧Recall / 正常FPR，单位%。RCR双分支六组FPR均下降，但部分组合
召回同时下降；自采Fusion的BAcc下降必须保留。不能只选FPR来宣称全部改善。

'''+trade_table+'''
## 配对增量与统计范围

每个窗口恰好一次外测，先合并五折；按47个条件或13个电器文件聚类重采样10,000次。
给出百分位95%区间及配对符号翻转检验；每类预定比较跨六组合做Holm校正。区间
和p值均为当前固定OOF的描述，训练折重叠及历史外测曝光限制其独立确认意义。
原始margin用于排序AUC/AP；固定阈值仍为零，不在测试上寻找最佳阈值。

'''+pair_table+'''
完整36行传统指标在本地`main_results.csv`，含Accuracy、Precision、Recall、
Specificity、F1、Macro-F1、FPR/FNR、ROC-AUC、AP、MCC及整数TP/FP/TN/FN。
参考有效/后备分层见`reference_strata.csv`：自采37,477/897，公共1,166/195。
分层采用pooled指标，样本构成与全队列不同，不将分层变化解释为独立因果效应。

## 热插拔开发判断

工程接口已经实现：各骨干独立的分数纠错状态可加载、开关和保存；纯输入插件使用
相同网络接口与逐窗后备掩码。关闭直接插件恢复冻结父模型；关闭重训插件保留新权重，
故不等于恢复旧性能。

优先保留分数纠错模块作为性能路线，保留固定窗同信息对照。输入模块继续作为独立
研究候选，其自采收益与公共不稳定结果并存。本轮不把任一路线提升为“所有骨干、
两个数据集统一有效”的默认配置，也不以零新增网络参数宣称端到端轻量化。

## 训练、缓存和核验

- 新增40个双分支内折拟合（20ArcFENet、20HYDRA），复用20个身份核验后的Fusion
  内折模型；冻结外层骨干30份复用，另保存60份动态纠错状态。
- 输入模块30个外折重新拟合，ArcFENet20轮/Fusion70轮/HYDRA原L1-hinge目标不缩减；
  直接插件仅用冻结权重，无新训练。初始化、打乱和训练内预处理与对应父版本匹配。
- 151份原生通道缓存重新核对完整通道SHA，绑定已有原始文件完整SHA及当前size/mtime；
  本轮不声称又读取并哈希了全部百GB级原始文件。HYDRA在原生1/2MHz电流上先整流，
  再滤波/降采样，其固定窗重建与冻结50kHz输入逐点一致。
- 保存状态独立重放、开关一致性、组隔离、诚实OOF、拟合索引和输入身份均有本地审计。
  双分支增强margin最大重放误差3.815e-6，关闭误差0；父模型跨分块概率舍入差最高
  1.907e-6，判决差0。纯输入插件的冻结父输出按原分块回放，具体误差见其审计。
- 两类、三次严格父推理检查在优化器启动前拦截了批次划分差异：两次FE平衡批次、一次
  历史完整heldout列表。修复推理批次后继续，未更改训练/评价集合；失败记录、执行
  源码快照与版本映射全部保留。GPU1释放后追加分担Fusion，调度修订单独记录。
- 训练只使用物理GPU1/2/3，GPU0保留。源数据、标签、权重和逐窗预测留本地。

## 统计解释审查（11/11）

1. Simpson：同时保留条件等权与pooled指标，并提供逐条件表。
2. 生态推断：结论限定当前采集条件；47条件不写成47种目标电器。
3. 选择偏差：审核/40ms保护预先固定，保留排除记录和标签误差说明。
4. 碰撞变量：参考有效分层只作描述，不用分层择优改主队列。
5. 基率：正常/电弧数量、Precision、FPR与Recall同时报告；不推算每小时报警。
6. 回归均值：所有增强均与同seed0冻结父模型配对，不只选最差组。
7. 幸存者：完整矩阵与后备周期保留，失败推理尝试归档。
8. 多重检验：五类预定比较各跨六组合校正，全部30行结果保留。
9. 分析路径：阶段方案、修复与调度留痕；本轮标为探索性，单种子不充作稳健确认。
10. 因果：历史、高频、校准、切窗变化分开解释；相位专属收益依据同信息对照。
11. 时间方向：历史特征只取过去，不跨采集/参考重启；OOF头不接触外测标签。
'''
    (OUT/'完整结果与热插拔可行性分析.md').write_text(text)
    doc=ROOT/'docs/LOCKED_ROUTES_20261030.md'
    body=doc.read_text().split('<!-- results:start -->')[0].rstrip()
    doc.write_text(body+'\n\n<!-- results:start -->\n## 已完成结果：seed0五折\n\n'+primary_table+
                   '\n所有新值都与相同seed0父模型配对。双分支5/6均值正向；相位专属六项区间均跨零。\n'
                   '完整传统指标、配对区间、参考有效性分层、状态重放和11项解释审查保存在本地\n'
                   '`outputs/locked_routes_20261030/完整结果与热插拔可行性分析.md`。\n'
                   '新结果不是三种子确认；冻结主表继续保留，六候选不自动升格默认。\n\n'+
                   '### 相同信息下的坐标比较\n\n'+phase_table+
                   '\n### 召回/误报同时审查\n\n'+trade_table+'<!-- results:end -->\n')
    print('REPORT_READY',len(m),len(p))


if __name__=='__main__':
    main()
