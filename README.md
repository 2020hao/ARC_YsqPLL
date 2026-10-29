# ARC_YsqPLL / ARC_model

电流自同步相位表征模块与三类串联电弧检测骨干的研究代码。按PILT方式区分配置、可复用实现、实验入口、测试和本地证据。

**当前状态：新模块为候选实现。** 已有C0的历史分类结果不代表本模块的检测效果。当前交付覆盖原始电流读取、参考估计、因果完整周波、原生频带积分、10维特征、训练内标准化和三个骨干的分类前接入；尚未执行本模块的正式双数据集分类训练。

## 安装与入口

```bash
python -m pip install -e '.[dev]'
arc-phase design
arc-phase model-smoke             # CPU接口检查，不是检测性能验证
python -m pytest tests/phase_module -q
```

未安装时也可以在项目根目录运行 `python scripts/phase_module.py ...`。

## 从原始电流准备候选表征

```bash
arc-phase prepare-record --current /path/to/C3_computer_carbonizedpath.dat \
  --sample-rate 1000000 --reference rcr --output outputs/public_example_rcr

arc-phase prepare-record --current /path/to/NET9770_114_record.bin \
  --sample-rate 2000000 --reference rcr --output outputs/self_example_rcr
```

- 支持NET9770_114的CH0（5m总路）、公共C3两列DAT、原生采样率的单通道NPY。
- 拒绝公共C2作为检测输入；C2只能离线辅助标注。
- 原始高频积分在2/1MHz完成；5kHz电流只用于参考估计。
- 输出 `wave.npy`（1000点）、`native.npy`（5000点）、`energy.npy`（2×20箱）、`phase_features.npz`、时间元数据和来源SHA256。
- 准备入口不产生类别标签，不把未标注区间视为正常，不自动生成ArcFENet图像或启动训练。
- `--reference` 可选 `fixed20/sogi/rcr/cft`；CFT为实验候选，不默认替代RCR。
- 不能对已有非空输出目录覆盖运行。

已有原生高频积分缓存可以只读重算特征：

```bash
arc-phase features --energy /path/to/energy.npy --metadata /path/to/cycles.csv \
  --mode phase --output outputs/phase_features_v1
# mode还支持phase_free和shifted，用于同容量、错位对照。
```

## 模块与模型接入

```python
from arc_model.methods.phase_representation import phase_features
from arc_model.training.phase_features import PhaseFeatureScaler
from arc_model.models.factory import make_model
from arc_model.models.phase_adapter import PhaseFeatureAdapter

batch = phase_features(energy, phase_valid=reference_valid)
scaler = PhaseFeatureScaler.fit(batch, train_indices)  # 仅外层训练数据
x_features = scaler.transform(batch)
model = PhaseFeatureAdapter(make_model('ArcFENet'), 'ArcFENet')
# logits = model(image_tensor, features=feature_tensor)
# baseline_logits = model(image_tensor, enabled=False)
```

HYDRA接入的是随机核变换后的向量及其线性分类器；ArcFusionNet同时传入时域和频域输入。新头从基线分类器初始化，附加特征列置零；关闭模块走原基线路径。开启后需训练，不是免训练插件。

10维候选为两频带对数总能量，以及每频带的参考过零邻域占比、半周不对称度、二阶圆周矩cos/sin。缺失能量与参考状态独立标记，不用测试集拟合归一化，也不把掩码当作真实相位误差。

## 项目分层

```text
configs/modules/                 候选模块声明
src/arc_model/data/              电流读取、来源与标签边界
src/arc_model/preprocessing/     滤波、参考时钟、已完成周波
src/arc_model/methods/phase_representation/  无标签、无模型的相位特征
src/arc_model/training/          训练分区内标准化
src/arc_model/models/            原本地骨干及特征级接入
src/arc_model/experiments/        薄入口和实验组织
scripts/phase_module.py          源码目录运行入口
 tests/phase_module/             特征、因果性、模型接口测试
outputs/                        本地结果，Git不跟踪
```

模型忠实度：ArcFENet为结构重建、HYDRA为作者核算法的本地适配、ArcFusionNet为仅电流适配；不称作者完整原始SOTA。当前250kHz输入的112.5kHz抗混叠滤波在112kHz附近有衰减，尚不宣称与作者采集链等价。

完整职责、对照设计与限制见 [模块设计](docs/PHASE_MODULE_DESIGN.md) 和 [代码分层](docs/CODEBASE_GUIDE.md)。

## 数据与实验管理

训练必须来自审核的标签，按原始录波/条件隔离，两个数据集分别训练报告。数据文件、录波、标签明细、权重、论文草稿和实验输出不进入公开仓库。公开的原始数据见 [Zenodo5501527](https://doi.org/10.5281/zenodo.5501527)；不要把后续作者MAT包当独立数据集。

新增模块不修改历史冻结脚本、模型或数据，也不复用历史得分充当新模块结果。本地历史脚本有绝对路径依赖，只作为证据保留；公开仓库聚焦可复用代码和模板。训练GPU仅可见物理1、2、3，CPU测试不申请GPU0。
