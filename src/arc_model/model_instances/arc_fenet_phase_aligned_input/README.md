# ArcFENet 相位对齐输入

模型ID：`arc_fenet_phase_aligned_input`。角色：`input_alignment`。状态：`engineering_migrated_current462_validation_pending`。

本目录固定网络/输入/训练及融合配置，`model.yaml`为本实例唯一配置；`create_model()`不能切换到另一个结构。网络和依赖闭包均在本目录，保持原构造顺序、参数名与缓冲区，用于既有state_dict严格重放。

- 网络：`network.py`；本地兼容基类：`_source_*.py`。
- 加载器：`data_loader.py`；预处理：`preprocessing.py`。
- 训练适配：`training.py`；推理：`inference.py`。
- 来源、原始类与哈希：`provenance.json`。
- 当前允许数据：462条自采及公共1357窗口协议。历史数据或成绩不迁入当前评价。

构建：`from arc_model.model_instances.arc_fenet_phase_aligned_input import create_model`。

历史结构仅迁移实现；不继承旧自采指标，当前数据上的性能状态以新协议实际验证为准。
