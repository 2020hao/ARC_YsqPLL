# dual_arc_fenet

原骨干＋独立36维动态纠错分支；已完成当前父基线seed0五折探索验证，默认仍关闭。

- 父基线：`B_FENET_fixed20_v1`，发布：`baselines_v1_20261030`。
- 模型实现：`network.py`；本目录`model.py`为本角色入口，不导入其他八个模型包。
- 数据：`data_loader.py`独立校验形状、组隔离与当前审核掩码；`raw_reader.py`读取原生BIN/C3/NPY。
- 必须显式传入当前`reviewed_indices`；建议用注册表对应的`split_file`，不沿用旧多余训练窗口。
- 输入训练：`backbone_training.py`；指标：`evaluation.py`；配置与来源分别见`configs/models.yaml`中的`ArcFENet_DualBranch`节点、`provenance.json`。
- 精确历史成绩以不可变`performance.json`及原注册表为准；便携训练API写入新输出目录，不覆盖这些成绩，且不承诺不同运行精度下位级重训一致。
- 改进分支默认关闭；开启必须提供匹配父基线的已拟合状态/对齐输入。旧C0/静态8维/输入序列化成绩不能移植到本版本。

符合度：论文结构重建，VMD收敛与频带漂移限制仍存在，未确认为完整作者前端。

`training.fit_from_oof`要求诚实分组OOF及收据；HYDRA还需要训练内margin校准和校准-only对照。没有声称新父基线已取得旧C0分数。

本轮验证入口为`training.fit_from_nested_oof`，将基模拟合组、头拟合组与门控验证组隔离。
`fit_from_oof`保留旧研究兼容；新研究不要将其历史调门控结果代入本轮。

本轮结果：项目`docs/LOCKED_ROUTES_20261030.md`；完整指标见本目录`performance.json`。

新增`phase_dynamic_v2.py`用于12组机制实验；可变维度和整路径有界融合另立版本。
`docs/PHASE_DYNAMIC_EVIDENCE_20261030.md`说明结果；不替换旧`model.py`或默认方法。
