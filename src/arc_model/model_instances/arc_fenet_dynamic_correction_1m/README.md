# ArcFENet 动态输出校正

模型ID：`arc_fenet_dynamic_correction_1m`。角色：`output_correction`。状态：`engineering_migrated_current462_validation_pending`。

本目录固定网络/输入/训练及融合配置，`model.yaml`为本实例唯一配置；`create_model()`不能切换到另一个结构。网络和依赖闭包均在本目录，保持原构造顺序、参数名与缓冲区，用于既有state_dict严格重放。

- 网络：`network.py`；本地兼容基类：`_source_*.py`。
- 加载器：`data_loader.py`；预处理：`preprocessing.py`。
- 训练适配：`training.py`；推理：`inference.py`。
- 来源、原始类与哈希：`provenance.json`。
- 当前允许数据：462条自采及公共1357窗口协议。历史数据或成绩不迁入当前评价。

构建：`from arc_model.model_instances.arc_fenet_dynamic_correction_1m import create_model`。

1 MHz旁路仅完成工程接入，当前462条性能验证待完成；主干关闭旁路时沿用原输出。共享模块与本地网络分别保存状态。

新1MHz旁路接入：`inference.predict_with_branch(model, ..., branch=branch, dynamic=features, enabled=True)`。`enabled=False`直接返回原模型输出，既不校准也不调用旁路。兼容旧格式`attach_correction`与新版本共享分支状态分别管理，不互相冒充。

启用新分支时必须绑定本实例的`FitIdentity`。传入`expected_identity`做模型/检查点/数据/划分/训练组/种子/特征版本全字段校验，或先用本目录`inference.load_checkpoint`绑定实际检查点文件SHA256；建议加载时同时绑定完整expected_identity。关闭时不要求身份或分支状态。

严格身份模式：只加载文件SHA256尚不足以确认数据划分；启用时需要显式expected_identity，或load_checkpoint(expected_identity=...)保存的完整身份收据。
