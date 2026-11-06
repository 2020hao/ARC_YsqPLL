# 1 MHz动态输出校正模块

实现：`src/arc_model/modules/rcr_dynamic_branch_1mhz/`。配置：`configs/modules/rcr_dynamic_branch_1mhz.yaml`。当前状态为工程迁移完成、462条三骨干检测收益待验证；本模块不继承旧队列性能。

## 电流与相位路径

输入为连续单通道电流：原生1 MHz直接通过；2 MHz使用103抽头因果Kaiser抗混叠形成1 MHz。通/阻边界为400/500 kHz，群延迟25.5 μs，不提前补偿。记录ID、源采样位置、时间原点和实际采样率必须显式提供。

高频活动在1 MHz电流上计算；内部5 kHz信号只为参考估计服务。参考、高频活动、有限历史及分箱沿用已记录公式版本。观测相位、预测保持与后备坐标分别标记，每个输出报告支持边界、可用时刻、参考区段及历史掩码。

相位恢复保证有明确的输出状态，不等于每个周波都有可观测、准确的相位真值。启动或参考异常按状态解释；无未来电流参与当前判决。

## 特征

每周波形成64相位箱，九周波历史下使用两频带的1/2/4/8周波动态：余弦相似度、平均绝对差、JS散度及对称峰箱份额变化，共32个连续量，加4个历史可用标记。模式及特征顺序由版本标识绑定。旧20箱静态候选和其他维数的历史方案保留为独立来源。

## 接入接口

```python
from arc_model.contracts import ScorePacket
from arc_model.modules.rcr_dynamic_branch_1mhz import RCRDynamicBranch1MHz

branch = RCRDynamicBranch1MHz()
scores = ScorePacket(original_margin, kind="svm_margin", threshold=0.0)
same = branch.correct(scores, enabled=False)
assert same is scores
```

上例为关闭行为的接口示意。启用需已拟合模块、36维动态特征和完整匹配的拟合身份。神经网络使用logit margin或显式概率，HYDRA使用SVM margin；不能将SVM margin直接当作概率融合。三个纠正实例各自的 `inference.predict_with_branch`负责接入。

| 接口 | 含义 |
|---|---|
| `observe(CurrentChunk)` | 因果消费原始电流，返回有时序和质量状态的周波事件 |
| `transform(profiles, mask)` | 将既有相位输入转为同一36维动态定义 |
| `fit(...)` | 显式离线拟合，要求训练索引、隔离组、诚实组外分数与身份 |
| `correct(ScorePacket, ..., enabled=False)` | 关闭时直接返回原对象；启用后按声明的分数语义校正 |
| `save/load` | 保存和校验参考、特征、模型、数据、折、种子与父权重身份 |
| `reset()` | 切换记录或时间中断时清除流式状态 |

训练状态按模型、数据、折与种子分别拟合。新接口启用时校验身份，文件加载验证实际检查点SHA256；跨模型、跨折或跨父权重使用会报错。状态关闭不触发校准或重新设阈值。

## 版本与验证

`legacy_correction.py`和 `legacy_dynamic_correction.py`保留旧alpha残差公式，供已有接口精确迁移；`fusion.py`与新持久化状态使用明确版本化的残差/整路径融合。不同版本不能仅因同为36维而互换状态。

CPU组件检查覆盖抗混叠分块一致性、原动态公式、两类融合、关闭恒等、身份隔离、训练组边界以及流式因果性。1 MHz采样适配、64箱特征和三骨干组合的实际检测效果需要在当前两库重新评价。
