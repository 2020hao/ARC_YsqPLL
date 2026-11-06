# RCR分层脉冲网络（加宽）

模型ID：`rcr_hierarchical_snn_wide`。角色：`direct_classifier`。状态：`completed_current_grouped_development`。

本目录固定网络/输入/训练及融合配置，`model.yaml`为本实例唯一配置；`create_model()`不能切换到另一个结构。网络和依赖闭包均在本目录，保持原构造顺序、参数名与缓冲区，用于既有state_dict严格重放。

- 网络：`network.py`；本地兼容基类：`_source_*.py`。
- 加载器：`data_loader.py`；预处理：`preprocessing.py`。
- 训练适配：`training.py`；推理：`inference.py`。
- 来源、原始类与哈希：`provenance.json`。
- 当前允许数据：462条自采及公共1357窗口协议。历史数据或成绩不迁入当前评价。

构建：`from arc_model.model_instances.rcr_hierarchical_snn_wide import create_model`。

当前双库分组开发结果来源于已保存研究产物；目录迁移不构成新训练或独立确认。训练期校准只拟合训练组OOF，折入同一个末层。

训练预算来源：`configs/modules/phase_hierarchy_20261105.yaml`，只复用参数定义，不启用历史自采数据。`training.train_batch`是本实例的训练组件；完整分组调度、BF16执行及OOF生成由明确研究入口负责，不将组件适配视为整条历史协议重放。
