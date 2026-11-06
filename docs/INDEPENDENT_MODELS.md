> 历史接口说明：本文保留原版本来源；当前模型目录、独立配置和入口见 [V2项目结构](CURRENT_PROJECT_STRUCTURE.md)及[模型目录](MODEL_CATALOG.md)。旧自采协议不再启用。

# 三个独立模型与专用加载器

更新：2026-10-29。模型独立化与新输入前端实验分别管理。

## 文件职责

| 模型 | 自包含模型文件 | 专用加载器 | 输入 |
|---|---|---|---|
| ArcFENet | `src/arc_model/models/standalone/arc_fenet.py` | `src/arc_model/data/arc_fenet_loader.py` | `rgb.npy`，`[N,3,100,100]` |
| ArcFusionNet | `src/arc_model/models/standalone/arc_fusion.py` | `src/arc_model/data/arc_fusion_loader.py` | `temporal.npy`，`[N,1,3,250]`；`spectrum.npy`，`[N,F]` |
| HYDRA | `src/arc_model/models/standalone/hydra.py` | `src/arc_model/data/hydra_loader.py` | `wave.npy`，`[N,1000]`；随机变换后448维 |

每个模型文件均可直接按文件路径导入，不引用项目中的另一个模型、通用网络块或模型基类。HFDC/DDC与时域卷积块分别内置在对应文件。HYDRA独立保留随机卷积、SparseScaler及本地线性分类器构建函数，并保留作者署名。

三个加载器各自实现元数据读取、形状验证、分组隔离与索引验证，不互相调用，也不继承公共加载器。仅依赖NumPy、Pandas、PyTorch及Python标准库。为便于逐文件检查，少量相同的校验逻辑有意分别保留。

加载器读取**模型准备缓存**，不是原始BIN/DAT解析器。原始电流读取和因果同步是独立的测量前端；本次没有复制三份PLL。`cycles.csv`中的标签必须来自已审核的区间，不在加载器中推断标签。

## 数据契约

每个模型缓存目录均包含：

- `cycles.csv`：`sample_id,label,condition_key,source_group,record_id`。样本ID唯一、标签0/1、关键字段不为空。
- `splits.json`：若干外折，每折`train`和`test`为该表的零基行号。它们不是record_id。
- 表中训练/测试不得有重叠行，也不得共享条件、采集来源或录波ID。
- 允许预先声明的边界保护样本不进入训练；所有被选入评价的样本恰好在一个外折测试集中出现。不能用测试结果决定保护区。
- 额外的前端缓存必须保持`sample_id`和标签顺序一致，之后才能配对比较。

ArcFusionNet的125项候选FFT谱不能直接全量送入原32项谱接口。调用方必须在外折**训练分区**选择32个频点，再将`selected_bins`传入加载器；加载器本身不接收测试标签，不替调用方做频点选择，也不自动选择前32项。

```python
from arc_model.data.arc_fenet_loader import ArcFENetLoader
from arc_model.models.standalone.arc_fenet import FENet

data = ArcFENetLoader("/absolute/path/to/prepared/ArcFENet")
train, test = data.indices(0)
batch = data.inputs(test[:2])
model = FENet(extra=0).eval()
logits = model(batch["image"])
```

```python
from arc_model.data.arc_fusion_loader import ArcFusionLoader
from arc_model.models.standalone.arc_fusion import FusionNet

data = ArcFusionLoader("/absolute/path/to/prepared/ArcFusionNet")
# train_selected_bins由调用方仅根据训练分区生成，并保存选择状态。
batch = data.inputs(test_rows, selected_bins=train_selected_bins)
logits = FusionNet(extra=0)(batch["temporal"], batch["spectrum"])
```

```python
from arc_model.data.hydra_loader import HydraLoader
from arc_model.models.standalone.hydra import transform_numpy, SparseScaler
import torch

data = HydraLoader("/absolute/path/to/prepared/HYDRA")
train, test = data.indices(0)
x = data.inputs(train)["wave"].numpy()
features = torch.from_numpy(transform_numpy(x, seed=0))
scaler = SparseScaler()
scaler.fit(features)  # 只拟合训练输入；测试输入复用该状态。
```

## 兼容性与检查

工厂`make_model`已转向两个独立神经模型；HYDRA现有特征入口和标准化入口转向其独立文件。工厂只是名称选择，不承载网络结构。

旧目录暂时保留，作为数值对照与历史权重来源。默认`extra=0`不新增分支；`extra>0`是保留的历史兼容接口，不是本轮实验配置。

检查包括：新旧state_dict键一致、严格加载、相同权重前向结果逐元素一致、梯度逐元素一致、HYDRA CPU变换逐元素一致、GPU实现误差检查，以及加载器尺寸和泄漏拒绝测试。

以上验证的是相对于**本地既有复现**的行为一致性。ArcFENet是论文重建，ArcFusionNet是仅电流适配，HYDRA分类训练器为本地实现；不能由这些检查推出已与作者完整实验等同。
