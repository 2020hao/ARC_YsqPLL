# V2项目重构记录

日期：2026-11-06。本地工作分支：`refactor/model-layout-v2-20261106`。变更限于工程组织与显式接口；当前462条和公共1357窗口的既有训练结果不重新命名或替换。

## 已完成范围

| 层 | 改动 | 核验 |
|---|---|---|
| 契约 | 电流、时间、分数、特征及完整拟合身份 | 非法采样、分数语义、支持边界及状态混用拒绝 |
| 共享模块 | 1 MHz采样适配、RCR、高频活动、历史、动态、融合和状态 | 原公式/分块因果性/旁路恒等/训练隔离 |
| 模型 | 60个独立实例，各自网络、加载器、配置与训练/推理适配 | 初始化、前向、梯度、实际训练批次及检查点兼容 |
| 注册 | 60实例＋3封存基准，按角色或推荐状态筛选 | schema、重复ID、跨模型依赖与冻结哈希 |
| 数据 | 462条125051窗四折、公共1357窗五折 | 实际两库纳入数、来源组隔离和逐窗外折覆盖 |
| 运行 | 本机路径、缓存哈希、资源策略与产物身份 | 禁止GPU0、拒绝不匹配的恢复状态 |
| 文档与CI | 当前入口、目录清单、模块契约、公开导出 | 相对链接、CPU检查及源码快照 |

## 一致性证据

- 3个baseline及42个封存文件校验通过，原路径与原YAML节点保持不动。
- 913份既有算法、配置、测试及研究入口与重构前完整快照一致；新目录以来源哈希记录复制依赖。
- 当前9种自有方案×两个数据集的原四/五折×3种子，共243份已训练外折权重，使用真实保留窗口做旧/新CPU前向，输出逐值一致。抽样覆盖每个检查点的8个保留测试窗口，并严格加载完整参数和缓冲区。
- CPU旧/新逐值比较与CPU对历史GPU分数比较分开记录，不把跨设备差异当作目录迁移的数值变化。
- 新组件211项、旧接口225项，合计436项CPU测试通过。独立可公开代码包另通过184项CPU测试，1项私有权重检查因不随代码发布而跳过；完整收据留在本地重构输出目录。

本次没有启动新的数据集训练，不新增检测性能或SOTA结论。测试中的合成小批反传/拟合仅验证训练接口。1 MHz三骨干校正的当前数据集增益仍需要单独实验。

## 保留与兼容

原 `implementations/`、带日期的runner、输出缓存和符号链接保持原位，供历史重放和恢复。新注册表不读取旧自采成绩当作当前性能。旧 `arc-models`与 `arc-phase`保持兼容；当前管理使用 `arc-project`。

模型目录的本地 `_source_*.py`保留必要类和初始化顺序，是各实例内部实现，不是跨实例的共享网络。完整OOF研究调度依然保留在原版本runner；本轮提供独立训练适配和当前项目管理接口，没有伪装成所有历史研究的一键训练器。

## 核验命令

```bash
CUDA_VISIBLE_DEVICES='' arc-project validate
CUDA_VISIBLE_DEVICES='' arc-project data-audit self462
CUDA_VISIBLE_DEVICES='' arc-project data-audit public
CUDA_VISIBLE_DEVICES='' python scripts/check_baseline_release.py
CUDA_VISIBLE_DEVICES='' python -m pytest tests/contracts tests/modules tests/model_instances tests/orchestration tests/data_io tests/parity tests/phase_module tests/test_locked_plugin_routes.py -q
```

源码运行可用 `python scripts/project.py`替代 `arc-project`。现有权重的全实例迁移核验由 `scripts/audit_project_layout_v2.py`执行，私有证据目录必须显式提供；完整参数见 `--help`。

## 恢复与GitHub同步

重构前快照保存全部已跟踪和未跟踪源码，位置为本地 `outputs/refactor_layout_v2_20261106/`下对应时间子目录；`LATEST`指向本次快照。包含 `SOURCE_MANIFEST.json`、Git状态和已跟踪/暂存补丁。校验和与各次审计收据在同目录保存。

需要回退时，先按清单检查待恢复文件，再恢复相应阶段；保留当前新增实现，使用旧入口即可继续历史重放。不执行 `git clean`或整体 `reset --hard`，避免丢失用户原有未提交工作。

用户已授权将本次重构同步至GitHub的同名分支；主分支合并通过拉取请求单独审查。公开导出只含允许的源码、配置、测试和文档；本机paths、数据、权重、逐窗预测及审计产物在Git忽略目录。同步前用 `scripts/export_code_snapshot.py`生成独立可审查代码包。
