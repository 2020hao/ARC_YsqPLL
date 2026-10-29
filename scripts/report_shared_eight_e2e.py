"""Build the result report only after all training and checkpoint audits pass."""

import json
import pandas as pd
import _bootstrap as boot

OUT=boot.ROOT/'outputs/shared_eight_e2e_20261029'


def main():
    audit=json.loads((OUT/'METRICS_AUDIT_PASS.json').read_text());assert audit['status']=='pass'
    replay=[json.loads((OUT/f'CHECKPOINT_AUDIT_GPU{i}_PASS.json').read_text()) for i in [1,2,3]]
    assert sum(x['fits'] for x in replay)==90 and all(x['status']=='pass' for x in replay)
    results=pd.read_csv(OUT/'results.csv');paired=pd.read_csv(OUT/'paired_deltas.csv')
    old=pd.read_csv(OUT/'frozen_comparison.csv');cost=pd.read_csv(OUT/'cost.csv')
    base=paired[(paired.candidate=='phase8')&(paired.baseline=='original')]
    fixed=paired[(paired.candidate=='phase8')&(paired.baseline=='fixed8')]
    positive=int((base.delta_pp>1e-10).sum());fixed_positive=int((fixed.delta_pp>1e-10).sum())
    lines=['# 统一8维特征：双数据集三骨干端到端训练验证','', '## Material Passport','',
        '- 日期：2026-10-29。','- 状态：90/90拟合、90份权重完整测试集前向重放、指标与划分核验完成。',
        '- 项目：`/data/hzy/experiments/ARC_model`。','- 输出：`/data/hzy/experiments/ARC_model/outputs/shared_eight_e2e_20261029`。',
        '- 范围：单训练种子、五折、当前连续数据的探索性实验；不是作者原始完整流程SOTA复现。','',
        '## 主要发现','',
        f'自同步8维相对原模型在{positive}/6组合中均值正向；相对固定窗同信息8维在{fixed_positive}/6组合中均值正向。',
        f'前一比较有{int((base.lo_pp>0).sum())}/6组的描述性95%条件区间完全大于零，Holm校正后p<0.05为{int((base.p_holm<.05).sum())}/6；后一比较分别为{int((fixed.lo_pp>0).sum())}/6和{int((fixed.p_holm<.05).sum())}/6。',
        '均值方向、区间与多重检验分开报告。区间跨零不能解释为等效；较高准确率不自动证明相位机制。','',
        '## 回答“是否被冻结骨干限制”','',
        '此前冻结编码器，只能重拟合分类头。本轮ArcFENet/ArcFusionNet从相同随机初始化开始，全部编码器和分类参数参与训练；三组的原有参数初值、批次顺序和训练随机状态一致，附加特征权重从0开始。已核验编码器首批梯度非零以及最终参数确实改变。',
        'HYDRA随机核在原算法中本来就是固定的，保留其定义；从头拟合全部原设计可训练分类参数，不把固定随机变换改为可学习卷积。因此HYDRA结果变化不能归因于“解冻了随机核”。',
        '旧冻结试验采用预训练权重和另一套分类头学习预算，本轮是从头完整训练。两轮的增量变化可以提示联合学习/优化方式的影响，不能单独隔离为冻结的因果效应。本轮保留cuDNN默认策略（允许TF32、非强制确定性），矩阵乘TF32关闭；旧训练明确禁用cuDNN TF32并启用确定性。这也是跨轮绝对指标不能直接当作冻结效应的原因。','',
        '## 三组设计与统一特征','',
        '| 组 | 骨干输入 | 新增特征 | 分类器接入 |','|---|---|---|---|',
        '| original | 相同RCR输入 | 无 | 原分类器 |',
        '| fixed8 | 相同RCR输入 | 固定20ms坐标计算的同8维 | 线性拼接的等价实现 |',
        '| phase8 | 相同RCR输入 | RCR坐标计算的同8维 | 同容量、同接入规则 |','',
        '8维定义未修改：10–100kHz、100–400kHz两频带各自的相对高频强度(2维)、参考0°/180°附近±18°能量占比(2维)、二阶圆周矩余弦/正弦分量(4维)。全部模型使用相同列、顺序、频带、相位箱和训练内标准化规则，没有按骨干筛选。能量由原始2/1MHz信号积分，低频分母为相位网格电流的能量代理，不能写成精确物理积分。',
        '神经模型使用已有独立模型文件中的extra线性项：W_b h+b+W_f f，数学上等价于分类器对[h,f]直接线性映射。不增加MLP、门控、注意力或跨周波历史。这是分类前特征接入；不是完全不改分类接口的纯电流输入前端。',
        '固定窗/自同步只改变新增特征，原骨干输入保持RCR不变。两种特征坐标对应的原始支持可能不同，因此检验完整特征生成流程，不能称为只改变相角的单因素实验。','',
        '## 数据与训练','',
        '自采38374周波、47条件；公共1361周波、13电器组。沿用审核标签、40ms边界保护、五折条件/采集分组隔离；整份录波不会同时进入同一折训练和测试。只使用连续自采5m总路CH0及公共C3电流；公共C2不进入模型。',
        '所有测试阈值固定0.5，不做测试集择优或阈值调参。RF32选择、时域/频谱标准化及HYDRA SparseScaler复用已核验训练行完全一致的原状态；新增8维标准化仅在对应训练折拟合。',
        '| 模型 | 轮数 | batch | 优化器 | 学习率 | weight decay | 精度 |','|---|---:|---:|---|---:|---:|---|',
        '| ArcFENet | 20 | 128 | Adam | 0.001 | 0.00001 | FP32张量，cuDNN允许TF32 |',
        '| HYDRA | 70 | 256 | AdamW | 0.01 | 0.0001 | FP32 |',
        '| ArcFusionNet | 70 | 64 | AdamW＋余弦调度 | 0.0001 | 0.01 | BF16计算，FP32参数/损失 |','',
        '同一模型三组训练设置相同。固定随机种子0，外折初始化种子由fold编号确定。训练全部预定轮数，保留最后一轮，不通过外折测试集挑模型。','',
        '## 主表：条件等权周波BAcc','',
        '| 数据集 | 模型 | 原模型 | 固定窗8维 | 自同步8维 | 相对原模型pp | 相对固定窗pp |',
        '|---|---|---:|---:|---:|---:|---:|']
    for (ds,m),g in results.groupby(['dataset','model'],sort=False):
        v=g.set_index('arm').BAcc
        lines.append(f'| {ds} | {m} | {v.original*100:.3f}% | {v.fixed8*100:.3f}% | {v.phase8*100:.3f}% | {(v.phase8-v.original)*100:+.3f} | {(v.phase8-v.fixed8)*100:+.3f} |')
    lines += ['', '## 配对比较','', '| 数据集 | 模型 | 比较 | 增量pp | 95%条件区间pp | Holm p |',
              '|---|---|---|---:|---|---:|']
    for x in paired.itertuples():lines.append(f'| {x.dataset} | {x.model} | {x.candidate}−{x.baseline} | {x.delta_pp:+.3f} | [{x.lo_pp:+.3f},{x.hi_pp:+.3f}] | {x.p_holm:.4f} |')
    lines += ['', '条件bootstrap5000次、组sign-flip20000次。12项phase8主比较统一Holm校正；6项fixed8−original作为次要比较另组校正。区间为未作同时性修正的95%描述区间。条件组对应训练折有重叠、数据已有历史测试曝光，不能视为独立确认。','',
              '## 全部分类指标','', '| 数据集 | 模型 | 组 | Recall | FPR | Precision | F1 | AP |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for x in results.itertuples():lines.append(f'| {x.dataset} | {x.model} | {x.arm} | {x.Recall*100:.2f}% | {x.FPR*100:.2f}% | {x.Precision*100:.2f}% | {x.F1*100:.2f}% | {x.AP:.4f} |')
    lines += ['', 'BAcc先在每个条件计算(Recall+Specificity)/2，再条件等权平均；其余指标为周波合并计算。AP为average_precision_score，混淆计数见results.csv。有效同步/后备窗分别报告于phase_validity.csv，后备不计作成功同步。','',
              '## 与冻结试验比较','', '| 数据集 | 模型 | 比较 | 冻结增量pp | 本轮增量pp | 增量变化pp |',
              '|---|---|---|---:|---:|---:|']
    for x in old.itertuples():lines.append(f'| {x.dataset} | {x.model} | {x.comparison} | {x.frozen_delta_pp:+.3f} | {x.e2e_delta_pp:+.3f} | {x.change_pp:+.3f} |')
    lines += ['', '此表只作跨方案描述，不能将变化全部归因于解冻。尤其HYDRA没有可解冻的原始随机核，其变化反映分类器初始化和优化方案等因素。','',
              '## 对接入方式的解释边界','',
              '解冻后，附加特征通过分类损失间接影响编码器学习，但最后的线性映射仍是加性组合，没有显式的特征条件调制或乘性交互。若某些组合仍不改善，可能存在信息重复、摘要损失或线性接口表达限制；本轮没有逐一隔离这些原因，不能直接认定“门控有问题”或“8维没有信息”。',
              '本轮不新增逐列消融。若8维整体尚未通过统一收益要求，不能将其升级为正式必选集；如果整体收益成立，仍需同一协议下的组/逐列移除和独立确认来定位贡献。','',
              '## 核验与成本','',
              f'90份保存模型全部测试窗口独立前向重放通过，最大概率误差{max(x["max_error"] for x in replay):.8g}，阈值判定零变化。来源哈希、训练/测试组隔离、初始原模型参数一致、特征标准化训练内拟合、跨模型同8维及OOF唯一覆盖通过。神经编码器梯度与参数更新均已核验。',
              '训练只使用物理GPU1、2、3，按折轮换三组在各GPU上的分配，GPU0未使用。输入数组常驻GPU以减少重复读盘传输；没有减少原定轮数或改变网络结构。','',
              '| 模型 | 累计模型训练秒数 | 最大单进程显存MiB |', '|---|---:|---:|']
    for m,g in cost.groupby('model',sort=False):lines.append(f'| {m} | {g.seconds.sum():.1f} | {g.peak_bytes.max()/1024**2:.1f} |')
    lines += ['', '累计秒数不是并行墙钟时间，也不包含原始读盘、PLL、带通、VMD/MBWVD等前端成本；不据此宣称整体轻量化。原始数据、标签明细、权重和逐条件结果保留本地。','',
              '## 当前结论','']
    if positive==6 and fixed_positive==6:
        lines += ['本轮六组均值均正向，说明统一8维联合训练路线值得保留；仍需按区间、多重检验和单种子范围限制结论，不能自动写成所有条件下有效。']
    else:
        lines += ['本轮没有形成六组共同正向的完整证据，因此继续保留统一8维候选，不按模型择列，也不将其写成所有骨干的正式必选输入。联合训练可以检验和缓解部分冻结限制，但尚不能承诺解决所有组合。']
    lines += ['同一套8维、同一接入规则和真实的不利结果均保留。下一步如研究接入结构，应保持这8维及划分不变，用同容量对照单独检验，不能同时换特征再把收益归给联合训练。']
    (OUT/'端到端统一8维_完整结果与分析.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status='complete',fits=90,datasets=2,models=3,folds=5,seeds=[0],feature_columns=8,
                 all_checkpoint_replays='pass',phase_vs_original_positive=positive,
                 phase_vs_fixed_positive=fixed_positive,gpu_physical=[1,2,3],gpu0_used=False)
    (OUT/'RESULTS_COMPLETE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
