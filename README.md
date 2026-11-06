# ARC_YsqPLL

**基于电流自同步与跨周波动态信息的交流串联电弧检测。** 项目包含三个独立基准模型、可插拔动态输出校正模块，以及自有CNN–SNN直接分类模型。V2工程迁移已完成，当前入口为 `arc-project`。

> 当前数据范围（2026-11-06）：**最新已标注导出自采库468次，当前实验纳入462次＋公共Zenodo 5501527数据**。468次中含微波炉3次、台灯3次，按此前目标排除规则暂不纳入实验。旧自采数据集退出当前训练、评价和论文主表。

## 阅读导航

| 内容 | 位置 |
|---|---|
| 当前结构与职责 | [V2项目结构](docs/CURRENT_PROJECT_STRUCTURE.md) |
| 全部独立模型及配置 | [模型目录](docs/MODEL_CATALOG.md) |
| 安装与命令 | [快速开始](docs/QUICKSTART.md) |
| 共享1 MHz旁路接口 | [模块说明](docs/MODULE_INTERFACE.md) |
| 重构检查与回退 | [迁移记录](docs/MIGRATION.md) |
| 已完成实验、完整指标和逐负载结果 | [实验结果展示](docs/EXPERIMENT_RESULTS.md) |
| 数据类别、标签、窗口、划分与采样率 | [当前数据协议](docs/DATASETS.md) |
| 当前自有模型完整配置 | [峰值动态CNN–SNN](src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/model.yaml) |
| 动态摘要、峰值池化及容量对照 | [动态信息保留设计](docs/DYNAMIC_STATE_NET.md) |
| 训练组折外校准与单分类头 | [校准说明](docs/FOLDED_CALIBRATION.md) |
| 462条数据上的三个基准 | [1 MHz比较与复现边界](docs/SELF462_1MHZ_BASELINES.md) |
| 参数、MAC与各种权重的含义 | [加权与成本说明](docs/WEIGHTS_AND_COST_ACCOUNTING.md) |
| 第三方实现来源 | [来源声明](docs/THIRD_PARTY_NOTICES.md) |
| 修改记录 | [CHANGELOG](docs/CHANGELOG.md) |

## 当前研究路线

### 基准输出动态校正

原模型沿用自己的预处理、骨干与分类器。旁路从电流建立相位参考，提取高频活动和跨周波动态信息，再修正原模型的输出分数。

```text
电流 → 原模型预处理 → 原骨干与分类器 → 原始分数 ─────────┐
                                                        ├→ 校正判决
电流 → 相位参考 → 高频活动 → 跨周波动态 → 输出校正模块 ──┘
```

“双支路”描述上述结构，“可插拔”描述模块的接入和关闭能力。相位对齐后直接替换骨干输入是另一条实验路线，应单独登记。

**当前462条数据尚未完成三个基准的同队列动态旁路验证。** 历史校正成绩不移入当前对比表。

### 自有CNN–SNN直接分类

```text
原采样率电流 → RCR参考 → 九周波相位组织输入
                          ↓
              峰值保留CNN–SNN＋动态摘要
                          ↓
                     单一分类器
                          ↓
                    当前周波判决
```

当前候选使用原五通道输入，动态摘要与网络状态进入同一个分类器；不接收外部基准分数，不使用蒸馏。训练组折外正仿射校准并入最后线性层，推理时保持单一分类头。详细配置见[候选YAML](src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/model.yaml)。

## 当前数据集

最新人工标注并导出468次采集；按此前排除微波炉和台灯目标的规则，当前研究纳入462次。468是原始导出库规模，462是实验纳入规模；两者都属于最新人工标注数据，不是旧的138份队列。详细关系见[数据范围核对](docs/DATASET_SCOPE_20261106.md)。

| 项目 | 自采462条 | 公共Zenodo 5501527 |
|---|---:|---:|
| 原始录波/物理采集 | 462 | 13 |
| 原始电流采样率 | 2 MHz | 1 MHz |
| 工况或电器录波组 | 156 | 13 |
| 当前共同评价窗口 | 125,051 | 1,357 |
| 正常窗口 | 61,896 | 794 |
| 电弧窗口 | 63,155 | 563 |
| 外层划分 | 四折工况隔离 | 五折电器/录波组隔离 |
| 当前自有模型种子 | 0、1、2 | 0、1、2 |

自采目标负载为**吹风机186条、热水壶174条、电磁炉90条、电视12条**。微波炉、台灯不作为本队列目标负载，仍可能出现在背景电器组合中。自采以114采集设备CH0总路电流为检测输入，使用人工复核起弧/熄弧区间；公共数据使用C3电流，C2仅作标注辅助。

同一采集及同一工况的所有窗口在一轮外折中整体归组，不将同一文件的相邻窗口随机拆入训练与测试。两库分别训练、拟合预处理和报告指标。当前主评价为稳定区间的周波判定；九周波上下文不等于九周波连续报警。标签保护、历史支持和每折数量见[数据说明](docs/DATASETS.md)。

公共数据当前1,357窗口来自统一实时前端支持检查，区别于历史1,361窗口协议；比较时必须匹配保留索引。旧自采队列及其划分均已停用。

### 采样率与相位参考

- 当前自有模型：自采使用**原生2 MHz**表征，公共使用**原生1 MHz**表征。
- 已完成的自采1 MHz比较：2 MHz原始电流经因果抗混叠下采样得到1 MHz，是同一数据的数字采样率适配。
- PLL内部5 kHz参考只用于相位估计；10–100 kHz、100–400 kHz高频活动从对应高采样率电流计算。
- 1 MHz可插拔旁路已实现工程接口，当前462条三骨干的检测收益仍需验证；其规格不改变已完成原生2 MHz研究的身份。

## 模型与验证状态

| 清晰名称 | 当前目录，均在 `src/arc_model/` 下 | 状态 |
|---|---|---|
| ArcFENet本地论文重建 | `implementations/baseline_arc_fenet/` | 冻结，保留原路径 |
| HYDRA原文FD配置 | `implementations/baseline_hydra/` | 冻结，保留原路径 |
| ArcFusionNet仅电流适配 | `implementations/baseline_arc_fusion/` | 冻结，保留原路径 |
| 三基准动态输出校正 | `model_instances/*_dynamic_correction_1m/` | 每骨干独立，共享旁路；当前462性能待验证 |
| 三基准相位对齐输入 | `model_instances/*_phase_aligned_input/` | 每骨干独立的输入对照 |
| 当前完整自有CNN–SNN | `model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/` | 双库三种子开发验证，迁移核验通过 |
| 原模型、动态、容量、峰值及校准对照 | `model_instances/`中各自独立目录 | 当前9种方案分别保存，不靠一个arm参数切换 |
| 历史结构及SOGI/固定窗TTFS对照 | `model_instances/`中各自独立目录 | 保留结构和参数来源，当前462重训状态单列 |

总计**3个冻结baseline＋60个独立实例**。`arc-project list --recommended`只查看三个基准和当前完整自有模型。全部目录和配置见[模型目录](docs/MODEL_CATALOG.md)。

ArcFENet保留原数值重建及VMD不收敛记录；HYDRA采用原FD特征和同一凸目标的认证求解；Fusion-current属于仅电流架构适配。各自复现范围见[基准说明](docs/SELF462_1MHZ_BASELINES.md)。冻结版本位于 `releases/baselines_v1_20261030/`。

<!-- BEGIN VERIFIED_EXPERIMENTS -->

## 已完成实验展示

以下表格只使用新自采462次实验队列和当前公共1,357窗口。468是最新标注导出库总量；现有实验按既定目标规则纳入462次。指标以已有聚合CSV及通过的审计收据为依据。

### 双数据集主结果

自采原生2 MHz、四折工况隔离；公共原生1 MHz、五折电器/录波组隔离。两库分别训练，均为三种子、50轮、概率阈值0.5。Acc给出种子均值±标准差，其余主表给出均值。

| 数据集 | 模型 | Acc%（均值±SD） | 组等权BAcc% | Recall% | FPR% ↓ | F1% | AP% |
|---|---|---|---|---|---|---|---|
| 自采462次 | 原五通道CNN–SNN | 96.80 ± 0.36 | 96.62 | 96.70 | 3.11 | 96.82 | 99.04 |
| 自采462次 | 增广峰值动态模型＋正仿射校准（完整模型） | 97.75 ± 0.21 | 97.64 | 96.85 | 1.34 | 97.75 | 99.19 |
| 公共1,357窗口 | 原五通道CNN–SNN | 92.56 ± 1.17 | 90.48 | 90.76 | 6.17 | 91.04 | 92.51 |
| 公共1,357窗口 | 增广峰值动态模型＋正仿射校准（完整模型） | 97.05 ± 1.33 | 95.38 | 93.66 | 0.55 | 96.34 | 99.18 |

完整模型为8,249个推理参数、0.684960M矩阵MAC/输出，原模型为7,097参数、0.683808M。两者均为九周波上下文直接分类；校准已折入末层。原始滤波、RCR及动态摘要标量成本另计。

### 自采1 MHz基准与采样率对照

同125,051窗口、156工况、四折、seed0；1 MHz由2 MHz原始电流因果抗混叠得到。该表与上述原生2 MHz三种子研究分别报告。

| 方法 | Acc% | BAcc% | Recall% | FPR% ↓ | F1% | 参数 | 加权MAC（M） |
|---|---|---|---|---|---|---|---|
| ArcFENet本地论文重建 | 86.31 | 85.93 | 88.53 | 15.95 | 86.72 | 98,917 | 24.455680 |
| HYDRA原文FD配置 | 95.41 | 95.37 | 93.24 | 2.37 | 95.35 | 1,025 | 2.561024 |
| ArcFusionNet仅电流适配 | 94.07 | 94.06 | 94.94 | 6.81 | 94.18 | 918,258 | 751.134528 |
| 原五通道CNN–SNN，1 MHz | 97.12 | 96.94 | 96.81 | 2.57 | 97.14 | 7,097 | 0.683808 |
| 原五通道CNN–SNN，匹配2 MHz | 96.77 | 96.60 | 96.75 | 3.21 | 96.80 | 7,097 | 0.683808 |

三个基准使用单20 ms输入，CNN–SNN使用九周波上下文。ArcFENet为本地数值重建，Fusion-current为仅电流适配；HYDRA另有2,560个固定核系数，其FD变换计入加权运算。表中MAC均不等于包含全部预处理的端到端FLOPs。

### 九种模型方案与消融

| 方案 | 参数 | 自采BAcc%（均值±SD） | 公共BAcc%（均值±SD） |
|---|---|---|---|
| 原五通道CNN–SNN | 7,097 | 96.62 ± 0.42 | 90.48 ± 1.05 |
| 原模型＋动态摘要 | 8,249 | 96.54 ± 0.75 | 91.44 ± 0.56 |
| 仅扩大分类器容量 | 8,285 | 96.67 ± 0.54 | 90.76 ± 0.55 |
| 峰值池化＋动态摘要 | 8,249 | 96.85 ± 0.29 | 93.44 ± 0.24 |
| 峰值动态模型＋物理一致增广 | 8,249 | 97.48 ± 0.23 | 93.64 ± 0.34 |
| 早期相位LIF小模型 | 7,001 | 97.22 ± 0.10 | 83.78 ± 2.01 |
| 早期相位LIF宽模型 | 13,921 | 97.05 ± 0.19 | 82.64 ± 1.83 |
| 增广峰值动态模型＋偏置校准 | 8,249 | 97.45 ± 0.21 | 94.50 ± 0.86 |
| 增广峰值动态模型＋正仿射校准（完整模型） | 8,249 | 97.64 ± 0.17 | 95.38 ± 1.95 |

单独加入动态摘要没有形成双库一致增益；早期相位LIF在公共数据出现退化，保留完整结果。物理一致增广及正仿射校准在自采主比较中有校正后正向支持，公共相应增量尚未形成同样统计支持。

### 公共同队列强参照

| 公共同队列系统 | Acc%（均值±SD） | BAcc% | Recall% | FPR% ↓ | F1% | 参数 | 矩阵MAC（M） |
|---|---|---|---|---|---|---|---|
| SOGI＋原值TTFS全事件SNN | 97.22 ± 0.86 | 96.17 | 95.86 | 1.81 | 96.62 | 8,817 | 4.497472 |

该参照与当前公共数据的1,357个窗口、标签及外折一致，但参考、编码和骨干不同，属于完整系统比较。当前完整CNN–SNN误报更低，Acc与BAcc均值仍低于此参照。强成绩来源为原研究raw_sogi，不绑定配置不同的新独立实例。

### 验证状态与完整结果

| 实验 | 当前状态 |
|---|---|
| 9种自有模型方案，双库分组、三种子 | 已完成，54份整套OOF核验 |
| 自采1 MHz三基准与匹配采样率比较 | 已完成，四折、seed0、20个最终模型 |
| 公共同队列SOGI＋TTFS强参照 | 索引、标签和外折一致；独立系统参照 |
| 当前462次上的三基准动态输出校正 | 尚未完成同队列性能验证 |
| 当前公共1,357窗口的三个baseline统一补齐 | 尚未完成；旧1,361窗口结果不并入 |

[查看完整指标、参数计算量、逐负载结果和配对统计](docs/EXPERIMENT_RESULTS.md)。完整模型在公共电脑、水泵上的Recall分别为60.87%、45.83%，自采电视FPR为8.82%，这些失效情况也保留在逐负载表中。

<!-- END VERIFIED_EXPERIMENTS -->

## 安装与核验

建议Python 3.11或3.12，包声明最低Python 3.10。在服务器已有环境中可先执行 `conda activate NILMFormer`。

```bash
git clone https://github.com/2020hao/ARC_YsqPLL.git
cd ARC_YsqPLL
python -m pip install -e '.[dev]'

# 当前入口：校验结构、查看推荐模型，不训练
CUDA_VISIBLE_DEVICES='' arc-project validate
CUDA_VISIBLE_DEVICES='' arc-project list --recommended
CUDA_VISIBLE_DEVICES='' arc-project build rcr_cnn_snn_peak_dynamic_affine_calibrated

# CPU核验冻结基准
CUDA_VISIBLE_DEVICES='' python scripts/check_baseline_release.py

# 核验当前双库完整模型汇总，需要本机已有的私有数据与产物
CUDA_VISIBLE_DEVICES='' python scripts/report_model_upgrade_20261105.py --check-only
```

GitHub仓库提供源码与配置；原始录波、标签明细、权重和逐窗预测保留本地。源码方式为 `python scripts/project.py`。本机路径写入Git忽略的 `configs/local/paths.yaml`，模板为 [paths.example.yaml](configs/local/paths.example.yaml)。使用 `arc-project data-audit self462`或 `arc-project data-audit public`检查队列。

各实例提供独立训练适配；既有全研究训练仍有本地缓存、OOF调度和阶段依赖，当前模型YAML不是原始数据一键训练命令。

### 当前自有模型的分阶段入口

下列命令按顺序显示研究计划，**加 `--execute` 才执行数据检查及对应研究**。进入下一阶段前需要上一阶段完成和匹配的缓存、划分及源代码身份。

```bash
python scripts/run_experiment_suite.py --suite configs/modules/dynamic_state_20261105.yaml
python scripts/run_experiment_suite.py --suite configs/modules/phase_hierarchy_20261105.yaml
python scripts/run_experiment_suite.py --suite configs/modules/folded_calibration_20261105.yaml
```

当前自采1 MHz比较配置为 `configs/modules/self462_1m_baselines_20261105.yaml`。所有训练只使用**物理GPU1、2、3，GPU0保留给其他用户**；GPU编号由研究调度器管理。

`configs/experiment_suite.yaml`、`configs/datasets.yaml`和旧九模型注册器仍包含历史数据入口，尚待重构迁移。当前实验必须显式指定上面的现行研究配置，不能省略 `--suite` 使用旧默认队列。

## 重构后的目录与使用边界

```text
src/arc_model/
├── contracts/                  # 电流、时间、分数与拟合身份
├── modules/rcr_dynamic_branch_1mhz/ # 共享1 MHz动态校正旁路
├── model_instances/            # 每个固定模型独立目录与完整YAML
├── implementations/            # 三冻结baseline及历史兼容包
├── data_io/                    # 当前两库只读数据与缓存身份
├── orchestration/              # 注册、运行清单、路径与GPU策略
└── cli.py                      # arc-project当前入口
configs/{datasets,protocols,runtime,experiments}/
configs/local/                  # 本机路径，Git忽略
artifacts/                      # 新运行产物，Git忽略
outputs/                        # 原缓存、结果及快照，原位保留
```

每个实例有本地网络、加载器、预处理、训练适配、推理与 `model.yaml`，不导入其他模型的网络或加载器。共享1 MHz旁路有独立契约与状态版本；关闭时恢复原分数和判决，启用时校验模型、父权重、数据及分折身份。

当前9种自有方案的243份外折权重已在真实保留窗口上做CPU对照重放，新旧输出逐值一致。三个baseline封存检查通过。重构不产生新性能成绩；当前462条三基准校正实验仍待完成。

旧 `arc-phase`、`arc-models`与带日期研究脚本保留为兼容入口。历史自采从当前数据入口移除，旧源码与证据按[迁移记录](docs/MIGRATION.md)保留，后续不据此新增旧自采实验。

## 评价、记录与公开范围

主指标为组等权周波BAcc，同时报告Acc、Precision、Recall、Specificity、FPR、FNR、F1、Macro-F1、AP、ROC-AUC、MCC及混淆计数。标准化、特征选择和校准均在训练组内拟合；神经模型概率阈值0.5，HYDRA分类间隔阈值0。组级重采样用于配对统计，相邻周波不当作独立重复。

GitHub同步可移植源码、配置、测试和选定文档；原始数据、逐记录标签、权重、逐窗预测、私有论文及实验产物保留本地。README展示当前状态，历史过程记入[变更记录](docs/CHANGELOG.md)，历史文档不自动成为当前协议。
