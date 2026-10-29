# ARC_YsqPLL

**面向交流串联电弧检测的电流自同步相位表征模块。**

本项目研究如何用电流自身建立周波参考，在统一相位坐标下组织高频活动，并将这些信息接入 ArcFENet、HYDRA 和 ArcFusionNet 三类骨干。

> 当前版本包含**统一输入前端候选、三个独立模型文件和三个专用加载器**。已完成双数据集HYDRA输入试验，以及三骨干统一8维、五折、单种子的冻结骨干接入验证；8维接入尚未建立统一收益，辅助10维分支仍为独立候选，均未作为默认方法。历史C0成绩不作为新前端的效果证据。

最近变更：[统一8维三骨干验证](docs/SHARED_EIGHT_STUDY.md) · [变更记录](docs/CHANGELOG.md) · [独立模型与加载器](docs/INDEPENDENT_MODELS.md)。

[![CPU tests](https://github.com/2020hao/ARC_YsqPLL/actions/workflows/phase-module.yml/badge.svg)](https://github.com/2020hao/ARC_YsqPLL/actions/workflows/phase-module.yml)

## 阅读导航

- [项目在解决什么问题](#项目在解决什么问题)
- [整体流程](#整体流程)
- [统一8维三骨干接入验证](#统一8维三骨干接入验证)
- [快速验证结果](#快速验证结果)
- [三个独立模型与加载器](#三个独立模型与加载器)
- [快速开始](#快速开始)
- [数据与输入格式](#数据与输入格式)
- [从原始录波生成表征](#从原始录波生成表征)
- [10维候选特征](#10维候选特征)
- [接入三个模型](#接入三个模型)
- [目录与职责](#目录与职责)
- [验证与实验规则](#验证与实验规则)
- [Git与进一步开发](#git与进一步开发)

## 项目在解决什么问题

固定20 ms窗口默认电流周期始终相同。频率变化和窗口起点偏移，会让不同周波的对应位置发生错位，影响位置相关特征的比较。

本项目的研究路径是：

**建立电流参考 → 选择完整周波 → 组织输入表征 → 使用模型现有接口 → 验证是否改善判别。**

这里估计的是**电流参考相位**。相位归一化用于建立可比较的坐标，并不等于已经消除了所有干扰或恢复了电源电压相位。

### 当前完成情况

| 内容 | 状态 |
|---|---|
| 单通道原始电流读取及采样率检查 | 已实现 |
| 固定20 ms、SOGI、RCR、CFT参考接口 | 已实现 |
| 因果完整周波选择及后备窗口 | 已实现 |
| 原始采样率上的两频带能量积分 | 已实现 |
| 10维相位候选及相位无关／错位对照 | 已实现 |
| 仅在训练分区拟合的标准化 | 已实现 |
| 三个独立模型与三个专用加载器 | 已实现；新旧结构一致性检查通过 |
| 无新增分支的输入前端快速分类验证 | 双数据集HYDRA，四组×五折×一个种子，已完成 |
| 10维辅助分支的分类训练与贡献验证 | 尚未完成；不默认启用 |
| 三骨干统一8维特征接入 | 已完成双数据集五折、单种子冻结骨干筛查；未通过统一有效性要求 |
| 跨三骨干、多种子端到端独立确认 | 待完成 |

## 整体流程

```mermaid
flowchart TD
    A["原始总路电流：1 MHz / 2 MHz"] --> B["统一输入表征前端：估计参考并选择完整周波"]
    B --> C["模型专用表征：波形 / 时频图 / 时域矩阵与频谱"]
    C --> D["各模型自己的数据加载器"]
    D --> E["原骨干及原分类头：不新增辅助分支"]
    E --> F["周波级判别"]
```

上图表示本轮研究路径。前端内部仍需低频参考与原采样率高频处理，这是信号处理分工；不等于在模型旁增加特征编码网络。前端输出必须按模型输入语义分别定义，不能把序列拼接候选直接当作物理电流送入MBWVD。

10维候选与`PhaseFeatureAdapter`保留在项目中，专门用于将来辅助分支对照，**不在本轮四组输入实验中启用**。


**5 kHz仅供同步参考使用。** 两个高频带仍在原始1 MHz／2 MHz电流上计算，不从降采样后的参考信号恢复高频。

### 四种参考选项

| `--reference` | 含义 | 用途 |
|---|---|---|
| `fixed20` | 固定20 ms名义坐标 | 固定窗对照 |
| `sogi` | 现有稳健SOGI-PLL内核 | 同步基线 |
| `rcr` | 电流参考恢复，管理观测、有效段和坐标重置 | 当前准备入口默认值 |
| `cft` | 基波一致性恢复与条件趋势跟踪候选 | 单独验证其增益与退化 |

参考不足时保留固定窗后备，并标记其来源；后备窗口不计作成功同步。

## 统一8维三骨干接入验证

三个模型统一使用同一组8维，不按模型选择特征：两频带相对高频强度2维、参考过零邻域占比2维、二阶圆周矩余弦／正弦分量4维。它与后面的历史10维候选是不同定义，没有混用。

本轮保持RCR骨干输入一致，冻结编码器，在原分类器前直接拼接8维，不新增MLP或门控。对照包括原模型、仅重拟合原头、固定窗同信息8维，以及强度组／位置组移除和位置错位。分类前拼接属于特征接入验证，**不等同于完全不改模型接口的纯输入前端**。

| 数据集 | 模型 | 原模型BAcc | 固定窗8维 | 自同步8维 | 相对原模型 |
|---|---|---:|---:|---:|---:|
| 自采5 m | ArcFENet | 86.075% | 85.629% | 85.738% | −0.337 pp |
| 自采5 m | HYDRA | 88.631% | 87.806% | 88.727% | +0.096 pp |
| 自采5 m | ArcFusionNet | 88.357% | 88.174% | 88.200% | −0.157 pp |
| 公共数据 | ArcFENet | 80.317% | 80.771% | 80.543% | +0.226 pp |
| 公共数据 | HYDRA | 88.276% | 87.263% | 88.276% | 0.000 pp |
| 公共数据 | ArcFusionNet | 76.711% | 76.828% | 76.828% | +0.117 pp |

**结果不支持将8维确立为所有模型的必选输入。** 自采HYDRA中，自同步相对固定窗同信息8维提高0.921 pp；但整体收益和两组特征的必要性没有同时成立。固定8维继续作为统一候选保留，不按测试结果为各模型择列。

这是单预训练种子、冻结骨干的快速筛查，不是三骨干端到端训练。不能与下方HYDRA输入序列化试验或其他历史队列直接比较。完整定义、仅重拟合头对照、统计范围及产物位置见[验证说明](docs/SHARED_EIGHT_STUDY.md)。

## 快速验证结果

这是**HYDRA单种子探索性试验**，并非三个模型的新SOTA结果。两数据集分别训练，沿用同一五折组划分和经审核边界保护；自采38,374周波、47条件，公共1,361周波、13电器组。每个分类器898个参数，AdamW、70轮、batch=256，组间设置相同。

| 输入方案 | 自采5 m 条件等权BAcc | 公共数据 条件等权BAcc |
|---|---:|---:|
| 固定20 ms电流 | 78.10% | 82.66% |
| RCR自同步电流 | **88.43%** | 88.73% |
| 固定20 ms低高频序列化 | 82.67% | 96.06% |
| RCR自同步低高频序列化 | 86.85% | **97.23%** |

低高频序列化将500点低频电流与两个各250点的频带能量轮廓放入同一个1000长度输入，不增加网络分支。能量来自原始1/2 MHz信号；20个相位箱插值成250点不会创造新的采样信息。序列化结果是表征，**不是恢复出的真实电流波形**。

- 自同步电流相对固定窗：自采+10.33 pp，公共+6.07 pp；公共数据的组级95%描述区间跨零。
- 自同步低高频表征相对固定窗同信息表征：自采+4.18 pp，公共+1.17 pp；后者区间跨零。
- 低高频序列化相对自同步电流：自采−1.58 pp，公共+8.50 pp，收益没有同时覆盖两个数据集。

**当前决策：保留自同步输入前端路径，暂不把低高频拼接作为统一默认。** 完整指标、配对区间、权重重放记录保存在服务器`outputs/input_frontend_pilot_20261029/`。单种子、历史测试曝光及两种切窗原始支持不同，限制了机制与泛化结论；不把结果归因于纯相位因素，也不据此宣称轻量化。

## 三个独立模型与加载器

| 模型 | 独立模型文件 | 专用数据加载器 |
|---|---|---|
| ArcFENet | [arc_fenet.py](src/arc_model/models/standalone/arc_fenet.py) | [arc_fenet_loader.py](src/arc_model/data/arc_fenet_loader.py) |
| ArcFusionNet | [arc_fusion.py](src/arc_model/models/standalone/arc_fusion.py) | [arc_fusion_loader.py](src/arc_model/data/arc_fusion_loader.py) |
| HYDRA | [hydra.py](src/arc_model/models/standalone/hydra.py) | [hydra_loader.py](src/arc_model/data/hydra_loader.py) |

每个模型文件内置自己的网络组件，不引用另一个模型文件或项目通用网络块；每个加载器独立实现模型输入校验与组隔离。工厂只选择模型名称。旧模型目录作为一致性对照保留，新旧权重键、输出和梯度已核对。

加载器负责已准备数组；原始BIN/DAT读取、同步和时频计算仍属于前端。ArcFusionNet的FFT32频点必须由训练分区选择，加载器不会通过测试标签代选。具体形状、文件要求及调用示例见[独立模型与加载器说明](docs/INDEPENDENT_MODELS.md)。

## 快速开始

建议使用 Python 3.11 或 3.12：本地验证使用3.12，GitHub自动测试使用3.11。

```bash
git clone https://github.com/2020hao/ARC_YsqPLL.git
cd ARC_YsqPLL
python -m pip install -e '.[dev]'

# 查看候选规格，不运行训练
arc-phase design

# 用随机输入检查三个模型的接口、基线一致性与梯度
arc-phase model-smoke

# 运行可独立复现的模块测试
python -m pytest tests/phase_module -q
```

如只需CPU环境，可先从 [PyTorch CPU安装源](https://download.pytorch.org/whl/cpu) 安装PyTorch，再安装本项目。所有上述检查均可在CPU执行。

源码目录也可直接运行：

```bash
python scripts/phase_module.py design
```

### 当前公开命令

| 命令 | 作用 |
|---|---|
| `arc-phase design` | 打印候选模块规格 |
| `arc-phase prepare-record` | 从一份原始录波生成无标签波形与相位特征 |
| `arc-phase features` | 从已有频带积分与窗口元数据重新生成特征 |
| `arc-phase model-smoke` | 检查三个骨干的接入接口 |

这些命令不会自动启动正式分类训练。`prepare-record`也不会自动生成ArcFENet的MBWVD图像。

## 数据与输入格式

研究中两个数据集分别训练、分别报告：

| 数据来源 | 原始采样率 | 检测输入 | 标签依据 |
|---|---:|---|---|
| 自采连续5 m总路录波 | 2 MHz | NET9770_114的CH0 | 经审核的正常／电弧时间区间 |
| [Zenodo 5501527](https://doi.org/10.5281/zenodo.5501527) | 1 MHz | C3线路电流 | 作者状态定义、C2辅助核查及项目审核区间 |

公共C2是弧电压，**仅用于离线辅助标注**；检测入口拒绝将C2文件作为电流输入。后续HYDRA作者MAT复现包不作为另一个独立公共数据集。

### 支持的文件

| 格式 | 输入要求 |
|---|---|
| `.bin` | NET9770_114八通道格式；自动读取CH0，核对头部采样率和负载大小 |
| `.dat` | `C3_`开头的两列文本：时间戳、电流；核对时间间隔 |
| `.npy` | 一维、有限值、原生采样率的电流数组；采样率由参数明确声明 |

一个文件包含连续波形，随后可以切出多个窗口。多个窗口仍属于同一次采集，不能随机拆分后声称验证了跨电器泛化。

原始数据、逐记录标签、权重和实验结果保留在本地，不随代码仓库分发。

## 从原始录波生成表征

以下`/path/to/`需要替换为实际数据位置。每次使用新的输出目录，入口会拒绝覆盖已有非空目录。

### 1. 公共电流录波

```bash
arc-phase prepare-record \
  --current /path/to/C3_computer_carbonizedpath.dat \
  --sample-rate 1000000 \
  --reference rcr \
  --output outputs/public_computer_rcr_v1
```

### 2. 自采总路录波

```bash
arc-phase prepare-record \
  --current /path/to/NET9770_114_record.bin \
  --sample-rate 2000000 \
  --reference rcr \
  --output outputs/self_record_rcr_v1
```

### 3. 输出文件

其中`N`表示生成的窗口数量。

| 文件 | 形状／内容 | 用途 |
|---|---|---|
| `wave.npy` | `[N, 1000]` | 50 kHz低频支路的周波归一化波形 |
| `native.npy` | `[N, 5000]` | 250 kHz支路的周波归一化波形，供时频前端进一步处理 |
| `energy.npy` | `[N, 2, 20]` | 原始采样率积分得到的两频带、20相位箱能量 |
| `phase_features.npz` | 10维特征、列名、数值有效性及相位有效性 | 新增特征分支 |
| `windows.csv` | 窗口起止、决策锚点、可用时刻、来源与参考状态 | 对齐、因果性及标签映射核查 |
| `reference_clock.csv` | 同步时钟诊断；固定窗选项为空时钟 | 参考过程检查 |
| `receipt.json` | 原始文件SHA256、采样率、时间原点及准备状态 | 来源追溯 |

这些输出**没有类别标签**。正式训练前还需与审核区间对应、实施边界保护、建立按采集／条件隔离的划分，以及准备各骨干所需输入。

### 4. 从已有能量缓存重新提取特征

```bash
arc-phase features \
  --energy /path/to/energy.npy \
  --metadata /path/to/windows.csv \
  --mode phase \
  --output outputs/phase_features_v1
```

`windows.csv`也可以替换为现有、逐行对应的`cycles.csv`。元数据须包含`phase_valid`或`kind`列，且行顺序与能量数组完全一致。

| `--mode` | 内容 |
|---|---|
| `phase` | 位置相关的10维候选 |
| `phase_free` | 同维度的相位无关统计对照 |
| `shifted` | 每个周波独立循环移位后计算位置特征 |

`phase_free`用于普通信息补充对照，不等于保留了完全相同的信息；`shifted`用于进一步检查位置关系。

## 10维候选特征

频带固定为 **10–100 kHz** 和 **100–400 kHz**，每周波划分为20个相位箱。

| 特征家族 | 维数 | 表达的内容 |
|---|---:|---|
| 两频带对数总能量 | 2 | 高频活动强度 |
| 参考0°／180°邻域能量占比 | 2 | 高频活动是否集中于参考过零邻域 |
| 两半周能量不对称度 | 2 | 对应半周的活动是否不平衡 |
| 二阶圆周矩cos／sin分量 | 4 | 半周尺度的位置朝向与集中结构 |
| **合计** | **10** | 待验证的紧凑候选 |

首轮邻域固定为±18°。该定义及特征集合尚未通过分类消融确认，不代表所有电器的最佳选择。这里描述的是**高频能量相对于参考坐标的位置**，不是高频载波本身的相角。

两个有效性概念分别保存：

- `valid`：某个特征是否具有可用的数值能量依据；
- `phase_valid`：该窗口是否来自有效的原生同步参考。

标准化由`PhaseFeatureScaler`在显式训练索引上拟合；测试数据只调用`transform`。

## 接入三个模型

以下内容介绍**可选的10维辅助分支**，不是上面的无分支输入前端，也没有继承上表的分类结果。原骨干默认`extra=0`；只有显式构造`PhaseFeatureAdapter`时才创建附加分支。

新模块通过组合保留本地骨干，在分类层前融合相位特征嵌入。骨干内部卷积、随机核或Transformer在本轮结构重构中保持原实现。

| 骨干 | 基线输入准备 | 新分支接入位置 | 实现定位 |
|---|---|---|---|
| ArcFENet | 波形→VMD／MBWVD→图像 | 卷积表征之后、分类之前 | 论文结构重建 |
| HYDRA | 波形→随机核特征 | 核特征与相位嵌入拼接后分类 | 作者核算法的本地适配 |
| ArcFusionNet-current | 波形／导数与训练内选择的FFT频谱 | 时频融合表征之后、分类之前 | 仅电流适配 |

这些本地基线与作者完整方法的差异需要在实验中单独说明，尤其是HYDRA预处理／配置及ArcFusionNet的差分前端。

当前250 kHz波形支路沿用112.5 kHz低通，112 kHz附近存在衰减；这一已知适配限制及参考状态的解释见[模块设计](docs/PHASE_MODULE_DESIGN.md)。

### 最小接口示例

下面使用随机数据检查接入，不进行电弧性能评价：

```python
import numpy as np
import torch

from arc_model.methods.phase_representation import phase_features
from arc_model.training.phase_features import PhaseFeatureScaler
from arc_model.models.factory import make_model
from arc_model.models.phase_adapter import PhaseFeatureAdapter

rng = np.random.default_rng(0)
energy = rng.uniform(0.1, 1.0, size=(6, 2, 20))
batch = phase_features(energy, phase_valid=np.ones(6, dtype=bool))

# 示例中只用前4条拟合标准化，后2条不参与拟合。
scaler = PhaseFeatureScaler.fit(batch, np.array([0, 1, 2, 3]))
features = torch.from_numpy(scaler.transform(batch))

model = PhaseFeatureAdapter(make_model("ArcFENet"), "ArcFENet").eval()
images = torch.randn(6, 3, 100, 100)
with torch.no_grad():
    logits = model(images, features=features)
    baseline_logits = model(images, enabled=False)
print(logits.shape)  # torch.Size([6, 2])
```

融合头从基线分类器初始化，附加特征列置零。关闭模块调用基线路径；开启后的有效增强需要训练。当前实现属于**特征级融合候选**，与历史C0分数级残差是两套方法。

## 目录与职责

```text
ARC_YsqPLL/
├── configs/modules/phase_representation.yaml  # 模块规格说明
├── src/arc_model/
│   ├── data/current_recording.py              # 原始电流读取
│   ├── data/*_loader.py                      # 三个独立数据加载器
│   ├── models/standalone/                    # 三个自包含模型文件
│   ├── preprocessing/
│   │   ├── phase_reference.py                # 参考算法统一接口
│   │   └── phase_windows.py                  # 因果窗口与高频积分
│   ├── methods/phase_representation/          # 特征定义与对照
│   ├── training/phase_features.py             # 训练内标准化
│   ├── models/phase_adapter.py                # 可选辅助分支，不默认启用
│   └── experiments/phase_cli.py               # 命令组织
├── scripts/
│   ├── phase_module.py                       # 源码运行入口
│   └── export_code_snapshot.py               # 公开代码快照导出
├── tests/phase_module/                        # 可独立运行的模块测试
├── third_party/                              # 保留声明的参考源码
├── docs/                                     # 设计、分层和版本管理说明
└── outputs/                                  # 运行时生成，Git忽略
```

YAML用于说明当前候选规格；现有CLI参数见`--help`，尚未提供从该YAML自动驱动完整训练的入口。历史研究模块继续保留在源码中，公开运行入口以本文列出的四个命令为准。

## 验证与实验规则

模块测试覆盖：

- 活动强度与位置特征的区分；
- 180°极性翻转、相位无关及独立错位对照；
- 标准化仅使用训练数据；
- 原始文件格式检查及拒绝C2检测输入；
- 四种参考路径的前缀因果性与原始高频保留；
- 三个骨干的基线一致性、融合梯度与缺失输入处理。

本地对两个数据集各一份真实录波完成输入重放；模块测试25项通过。快速分类试验另存40份权重、训练内标准化状态与OOF预测，并检查训练/测试组隔离。两类验证分别回答工程正确性与探索性分类效果。

后续正式实验应遵循：

1. 两个数据集分别训练和报告，使用审核标签；
2. 按原始采集／条件分组，整份文件的周波不跨同一轮训练与测试；
3. 同步、固定窗、普通融合、错位对照使用可比的信息与预算；
4. 以条件等权周波BAcc为主，结合Recall、FPR、F1、AP和组级配对区间；
5. 保留不利结果，并计入参考、预处理、模型和附加分支的完整成本。

本研究服务器的训练GPU策略为**仅使用物理GPU1、2、3，保留GPU0**。本README中的测试命令可在CPU完成。

## Git与进一步开发

GitHub仅同步可复用代码、规格、测试和选定文档。历史论文生成器、数据、标签明细、权重及结果留在本地。

本地`research-baseline-20261029`标签及`research/local-main`分支保存重构前研究状态；日常公开开发以`main`为基线，避免将私有历史分支整体推送。

| 文档 | 内容 |
|---|---|
| [统一8维接入验证](docs/SHARED_EIGHT_STUDY.md) | 同一特征集的三骨干实验与当前结论 |
| [变更记录](docs/CHANGELOG.md) | 每次改动的目标、范围及验证 |
| [独立模型与加载器](docs/INDEPENDENT_MODELS.md) | 六个独立文件及输入契约 |
| [模块设计](docs/PHASE_MODULE_DESIGN.md) | 输入输出、特征定义、接入与消融 |
| [代码分层](docs/CODEBASE_GUIDE.md) | 各层职责和依赖方向 |
| [Git工作流](docs/GIT_WORKFLOW.md) | 版本记录、公开快照及本地证据管理 |
| [第三方来源](docs/THIRD_PARTY_NOTICES.md) | 参考实现、论文与来源说明 |

仓库保留第三方原有声明；数据的引用与许可遵循对应原始发布记录。
