# 项目结构与职责（V2）

更新：2026-11-06。当前入口为 `arc-project`，源码方式为 `python scripts/project.py`。只使用462条自采和公共1357窗口协议。[模型目录](MODEL_CATALOG.md)列出所有实例，[数据协议](DATASETS.md)说明窗口及划分。

## 当前目录

```text
src/arc_model/
├── contracts/                 # 电流、时间、分数、特征及拟合身份
├── modules/
│   └── rcr_dynamic_branch_1mhz/# 唯一共享的1MHz动态输出校正模块
├── model_instances/           # 60个固定实例，每个单独目录
├── implementations/           # 3冻结baseline＋历史兼容实现
├── data_io/                   # 共享只读元数据、索引、路径与缓存身份
├── orchestration/             # 注册、配置、资源策略与运行清单
├── evaluation/                # 通用指标
└── cli.py                     # 当前项目CLI
configs/
├── registry.yaml              # 扫描独立实例并引用3个冻结基准
├── datasets/                  # 两个当前数据源，不包含旧自采入口
├── protocols/                 # 自采四折与公共五折
├── modules/                   # 共享模块规格；较早suite保留兼容
├── runtime/                   # GPU1/2/3，禁止GPU0
├── experiments/               # 当前比较计划
└── local/paths.yaml           # 本机数据路径，Git忽略
artifacts/                     # 新运行产物，Git忽略
outputs/                       # 既有缓存、结果与重构快照，原位保留
```

## 独立实例边界

一个固定网络、输入编码或融合方案对应一个模型ID和目录。不同折、种子属于运行重复，不复制源码。相位参考不同的TTFS对照也各自有独立目录。

各实例包含 `network.py`、`data_loader.py`、`preprocessing.py`、`training.py`、`inference.py`、`model.yaml`、`README.md`和 `provenance.json`。必要基类、神经元和特征计算复制到本实例内部，文件名前缀 `_source_`标明来源；这保留原构造顺序及权重键名，避免跨模型继承带来的联动修改。

新实例不得导入其他实例或旧 `implementations/`中的网络、加载器或模型组件。允许复用明确声明的共享旁路、数据契约、指标、日志与调度。注册器通过AST检查依赖边界。

## 配置唯一性

新实例的完整模型配置位于自己的 `model.yaml`，包含固定网络、输入、训练及评价设置，不继承其他模型的YAML。模块参数集中在模块配置中，数据纳入与分折集中在协议配置中，运行时保存解析后的配置。

三个baseline保留冻结目录和 `configs/models.yaml`中各自的封存节点。V2注册器验证原文件与节点哈希后引用它们，不创建第二份可冲突的默认配置，不修改冻结性能记录。

旧 `configs/own_models/rcr_peak_dynamic_candidate.yaml`保留为既有研究来源。新完整候选规范入口是 `model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/model.yaml`。

## 数据与运行产物

底层缓存可只读共享，各模型自己的加载器负责表示选择和输入适配。缓存身份包含原始来源、通道、采样率、滤波、参考、特征版本及配置；不能按数组形状判定可替换。

新运行路径：`artifacts/runs/<study>/<protocol>/<model>/fold-N/seed-N/`。运行身份同时绑定源码、模型配置、数据与划分；校正模块还绑定父模型检查点及训练组。保存权重、预处理、校准状态和预测时，不能只保存一个不带上下文的state_dict。

## 兼容入口

`arc-phase`和 `arc-models`继续服务原研究接口；带日期的旧脚本和依赖路径不移动。它们不是当前数据选择入口，旧自采拟合状态和成绩不参加当前比较。新CLI只接受当前两库，旧自采ID会直接拒绝。

当前CLI支持列出、校验、配置、CPU构建、数据审计、计划与报告。每实例提供实际训练适配；历史全研究的OOF调度仍按原脚本保存，迁移没有将所有历史训练器改造成一个统一训练命令。
