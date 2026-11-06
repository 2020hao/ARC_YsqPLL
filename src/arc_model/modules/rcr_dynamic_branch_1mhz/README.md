# 1 MHz RCR跨周波动态校正模块

本包提供独立于骨干模型的因果电流前端、36维动态特征及输出校正。实现已通过组件一致性检查；当前462条数据上的三个基准校正性能仍需另行验证。模块不会自动训练，不继承旧自采数据的成绩。

## 接口

```python
import numpy as np
from arc_model.contracts import CurrentChunk, ScorePacket
from arc_model.modules.rcr_dynamic_branch_1mhz import RCRDynamicBranch1MHz

branch = RCRDynamicBranch1MHz()
# 单位、记录身份、原始采样率、样本位置显式提供；示例为2 ms原生1 MHz电流。
events = branch.observe(CurrentChunk(np.zeros(2000), 1_000_000, "record-1", 0))
scores = ScorePacket(np.array([-1.0, 2.0]), kind="svm_margin", threshold=0.0)
assert branch.correct(scores, enabled=False) is scores
```

- `observe(CurrentChunk)`接收任意长度的连续原始电流，返回已可用的完整周波特征事件；内部以2 ms块处理。新记录自动清空历史；同记录跳样须显式`reset()`，防止跨中断连接历史。
- `transform(raw, mask)`将未经标准化的`[N,9,5,64]`输入转换为`[N,36]`：四种时滞、两频带、四统计加四个连续历史标记。
- `fit(...)`仅在调用方显式提供训练索引、外测试索引、组、诚实训练组OOF分数及`FitIdentity`后执行离线拟合。模块验证所声明的索引与组隔离；OOF分数的生产模型和收据需由实验执行器另外保存。
- `correct(ScorePacket, features, enabled=False)`默认严格返回原对象，保留原分数域、dtype及判决；启用时必须已绑定`FitIdentity`。`correct/predict`的`expected_identity`用于逐项核对当前模型、父检查点、数据、划分和种子，模型实例适配器另强制核对本实例身份。`predict`提供返回分数数组的便捷接口。
- `save/load`保存并验证模型、父检查点、数据、划分、训练索引、种子和各算法版本身份。加载生产模型时传入`expected_identity`，拒绝混用其他模型或折的状态。

## 两个兼容状态版本

`Correction`及`legacy_dynamic_correction`保留旧alpha/PCA动态校正接口，供独立模型迁移及历史重放。`RCRDynamicBranch1MHz`使用明确的`phase_dynamic_v2`状态，支持残差校正或完整路径融合；两种状态不可相互冒用。旧兼容状态的原有身份字段继续生效，新状态额外记录完整`FitIdentity`。

完整路径融合为`raw + strength * gate * (calibrated - raw + delta)`；残差融合为`calibrated + strength * gate * delta`。因此完整路径的`strength=0`恢复原分数，而残差融合的`strength=0`仍保留校准。关闭模块则两种方式都严格恢复原分数。HYDRA保持SVM间隔，不能把间隔当成概率。

## 采样率和时间

原生1 MHz输入逐值通过。2 MHz使用原数据准备流程完全一致的103抽头Kaiser因果FIR后降采样，400/500 kHz通阻边界，25.5 μs群延迟；不使用未来补偿或零相位滤波。1 MHz是旁路高频分析规格，不覆盖骨干自己的输入接口，也不能把原生2 MHz成绩改称1 MHz成绩。

内部5 kHz电流只提供RCR参考，10–100/100–400 kHz活动仍在1 MHz电流上计算。RCR原生周波与20 ms名义后备周波使用各自历史；后备坐标不等同于观测到的相位。事件显式保留坐标、参考区段、epoch、相位状态、原始支持右端及实际可用时刻。

## 来源与验证

`provenance.json`记录提取时的原始源码及SHA256。参考来自`rcr_cycle_v2`，高频积分和周波组织来自原`CycleStream`的无模型部分，动态统计来自原`native_dynamic_summary`；融合来自`phase_dynamic_v2`。旧原文件保持不变。

CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/contracts tests/modules -q
```
