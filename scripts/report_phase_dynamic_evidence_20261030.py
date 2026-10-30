"""Render all predefined evidence groups without promoting a test-selected winner."""
from pathlib import Path
import json
import pandas as pd
import _bootstrap as boot
from report_locked_routes_20261030 import table

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'
DS={'continuous5m':'自采5m','zenodo5501527':'公共'}
MODELS=['ArcFENet','HYDRA','ArcFusionNet']


def metric_table(m,arms,headers):
    p=m.pivot(index=['dataset','model'],columns='arm',values='BAcc_condition');rows=[]
    for ds,name in DS.items():
        for model in MODELS:
            v=p.loc[(ds,model)];rows.append([name,model]+[f'{100*v[a]:.3f}' for a in arms])
    return table(['数据集','模型']+headers,rows)


def main():
    for name in ['HEAD_AUDIT.json','REAL_MODEL_REPLAY.json','independent_audit.json']:
        assert json.loads((OUT/name).read_text())['status']=='pass',name
    m=pd.read_csv(OUT/'main_results.csv');pairs=pd.read_csv(OUT/'paired_deltas.csv')
    contribution=metric_table(m,['R0_parent','A1_rcr_masks4_oldfusion','A2_rcr_stats32_oldfusion','R2_rcr36_old'],
                               ['原模型','仅4标记','仅32动态','完整36旧融合'])
    fusion=metric_table(m,['R0_parent','R2_rcr36_old','B1_rcr36_wholeblend','B2_rcr36_wholeblend_recall','P1_rcr40_phase'],
                         ['原模型','旧36融合','B1整路径融合','B2加召回约束','P1加位置四维'])
    phase=metric_table(m,['B2_rcr36_wholeblend_recall','P1_rcr40_phase','P2_fixed40_phase','P3_rcr40_shift','P4_rcr40_phasefree'],
                        ['B2无新增四维','P1正确位置','P2固定窗同信息','P3新增四维错位','P4同容量相位无关'])
    trade=[]
    for ds in DS:
        for model in MODELS:
            sub=m[(m.dataset==ds)&(m.model==model)].set_index('arm')
            trade.append([DS[ds],model]+[f'{100*sub.loc[a,"Recall"]:.2f}/{100*sub.loc[a,"FPR"]:.2f}' for a in
                                         ['R0_parent','R2_rcr36_old','B1_rcr36_wholeblend','B2_rcr36_wholeblend_recall','P1_rcr40_phase']])
    trade=table(['数据集','模型','原Recall/FPR','旧36 Recall/FPR','B1 Recall/FPR','B2 Recall/FPR','P1 Recall/FPR'],trade)
    text='''# 十二组机制验证：结果与论文主线判断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-10-30
- Verification Status: VERIFIED（实际拟合、独立状态与完整Model回放、逐窗指标重算）
- Version Label: phase_dynamic_evidence_20261030_seed0

主线为**电流自同步坐标下的跨周波高频动态可信纠错**。本轮把“动态信息”“整路径
融合”“新增绝对位置量”分别验证。12组全部完成，预先指定完整候选为P1，保留
全部负结果；没有根据外测结果另选每个骨干的最优配置。

## 工作量与复用

12配置＝4复用＋8新增；每配置覆盖双数据集×三骨干×五折，形成72套完整OOF指标。
30个数据集/骨干/折任务内生成240个新增外折方案、120个复用方案，共1,430,460条
预测。七个不同特征集合各做B拟合和完整训练OOF重拟合，合计420次小型线性拟合；
另1次数值一致性拟合单列。新增骨干训练0次，原38,374/1,361审核队列不变。

## 动态和历史标记分别做了什么

以下及后续BAcc均为条件等权周波指标，单位%。旧融合下比较A1、A2、R2：

'''+contribution+'''
总体上，仅4个历史标记不能解释大部分完整分支收益；仅32维动态已接近完整36。
但自采Fusion旧融合下仅动态和完整36都低于原模型，因此信息补充和融合方式需要
一起审查。该组消融直接对应旧融合R2，不等同于新融合下已逐列验证每个特征。

## 融合改造能否减少过度纠错

'''+fusion+'''
B1整路径有界融合在六个组合上均值都高于父模型；相对旧融合并非全部提高。
它是预定的诊断候选，不因六个均值正向就取代预定主方法P1或成为确认性赢家。
最有价值的变化是自采Fusion从旧分支88.749%变为B1的90.533%，高于父模型89.236%。
公共HYDRA从旧94.945%变为96.458%。这些比较的配对区间和校正结果见CSV，不能仅凭
均值将所有升级称为显著。

B2的训练内召回约束没有带来统一BAcc升级；约束在C中成立，不保证外测单组或均值
召回必然不降。B1与R2比较的是包含整路径退回、有界强度及条件权重选择的整套融合
规则；B2与B1才单独隔离召回约束。

'''+trade+'''
## 新增四维能否确立相位贡献

'''+phase+'''
P1相对原模型六个均值正向，但新增四维相对B2仅三个组合均值提高、三个下降；
相对同容量相位无关P4也仅两个提高、四个下降。正确位置没有稳定胜过固定窗或错位。
30项主比较统一Holm后，P1相对原模型的自采ArcFENet和Fusion通过筛选；相位位置
专属比较均未通过。这说明“完整模块有效”与“新增相位位置有效”仍是两个结论。

因此，本轮**不将新增四维纳入已验证的正式特征集合**，也不继续做这四列逐列消融
来追逐某个外折涨分。已有位置组尚未建立独立优势，逐列筛选不能代替这一缺口。
P1/P3只检验新增位置量的正确对齐，未破坏原36维整个动态坐标，不能扩大为自同步
前端从此无效的结论。

## 当前论文应采用的表述

保留参考坐标→跨周波动态→可信纠错的研究问题。现有较明确证据是：动态而非仅历史
标记提供主要补充，整路径退回可以改善部分过度纠错问题，模块可在原骨干之外独立
加载、开关和部署。相位参考作为组织前端保留，其独立性能必要性继续按同信息对照
判断；当前不能写成“新增四维相位特征已显著提升所有模型”。

下一版方法候选可优先研究36维整路径融合B1。它属于本轮诊断结果带来的探索方向；
若转为正式主方法，应补相同整路径融合下的固定窗36维对照和对应动态/标记消融，
而非拿旧融合固定窗R3代替。此处记录这一缺口，没有额外追加矩阵或自动默认切换。

## 统计、复现与边界

主比较P1分别对原模型、B2、固定窗同信息、错位和同容量相位无关，五类×六组合共
30项统一Holm；另外30项诊断比较按各六组合家庭校正。10,000次条件/电器文件聚类
重采样，区间为边际95%区间；同源周波不当作独立重复。五折训练重叠、历史外测
曝光及仅骨干seed0限制确认性推断。

420份头的标准化/PCA和拟合索引独立检查；60个诚实内折模型权重/预测/索引身份、
60份训练内校准、全部A/B/C/外测隔离通过。新增四维逐点独立重建；从151份原始
周期清单重建历史掩码，无未来或跨参考区段历史。240份模块的953,640行结果独立
算式重放误差0；240份真实Model状态GPU1回放最大margin误差3.815e-6，判决差0。
关闭及整路径零强度严格恢复父分数。GPU0未使用；无未解决失败。

## 11项解释审查

1. 条件等权与pooled指标分开，并保留逐条件表。
2. 不把47背景条件解释为47目标电器。
3. 审核队列及保护区固定，无外测择样本。
4. 有效/后备分层仅描述，不筛掉参考困难段。
5. 基率、Precision、Recall与FPR同时报告。
6. 配对同seed0父模型，不以最低阶段分数作为参照。
7. 12组完整保留，无只报完成或涨分组合。
8. 五类主比较全30统一校正，诊断检验单列。
9. 方法P1预先确定；B1的后续研究价值明确标为探索，不倒置为预先主检验。
10. 相位、信息量、容量和融合分别对照；不从模块总收益推出相位因果必要性。
11. 动态历史只向过去，标准化/PCA/校准/强度选择只用允许训练索引。

配置与12组用途见项目`docs/PHASE_DYNAMIC_EVIDENCE_20261030.md`；本目录包含
`main_results.csv`、`paired_deltas.csv`、`condition_metrics.csv`、`reference_strata.csv`、
`correction_transitions.csv`、`selection_choices.csv`与完整审计包。
'''
    (OUT/'十二组实验_主线与贡献定位分析.md').write_text(text)
    doc=ROOT/'docs/PHASE_DYNAMIC_EVIDENCE_20261030.md'
    old=doc.read_text().split('<!-- completed-results:start -->')[0].replace('Verification Status: PLANNED；完成状态与实际核验另存结果包','Verification Status: VERIFIED；12组实际拟合与保存状态重放已完成')
    doc.write_text(old.rstrip()+'\n\n<!-- completed-results:start -->\n## 已完成结果\n\n'+fusion+
                   '\n### 相位位置机制对照\n\n'+phase+
                   '\n新增四维尚未建立相对同信息、错位、同容量对照的独立优势，因此不提升为正式必选输入。\n'
                   'B1整路径36维融合六组合均值正向，作为后续探索候选保留；不自动替换预先指定P1或旧方法。\n'
                   '完整传统指标、区间、独立审计与解释见本地`outputs/phase_dynamic_evidence_20261030/十二组实验_主线与贡献定位分析.md`。\n'
                   '只使用骨干seed0，未完成独立新采集确认。\n<!-- completed-results:end -->\n')
    print('REPORT_READY',len(m),len(pairs))


if __name__=='__main__':main()
