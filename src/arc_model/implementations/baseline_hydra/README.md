# baseline_hydra

固定代码与指标快照；不加入本研究同步/纠错。

- 父基线：`B_HYDRA_FD_fixed20_v1`，发布：`baselines_v1_20261030`。
- 模型实现：`network.py`；本目录`model.py`为本角色入口，不导入其他八个模型包。
- 数据：`data_loader.py`独立校验形状、组隔离与当前审核掩码；`raw_reader.py`读取原生BIN/C3/NPY。
- 必须显式传入当前`reviewed_indices`；建议用注册表对应的`split_file`，不沿用旧多余训练窗口。
- 输入训练：`backbone_training.py`；指标：`evaluation.py`；配置与来源分别见`configs/models.yaml`中的`HYDRA_Baseline`节点、`provenance.json`。
- 精确历史成绩以不可变`performance.json`及原注册表为准；便携训练API写入新输出目录，不覆盖这些成绩，且不承诺不同运行精度下位级重训一致。
- 改进分支默认关闭；开启必须提供匹配父基线的已拟合状态/对齐输入。旧C0/静态8维/输入序列化成绩不能移植到本版本。
