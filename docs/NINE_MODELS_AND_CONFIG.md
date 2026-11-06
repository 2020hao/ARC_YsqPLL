> 历史接口说明：本文保留原版本来源；当前模型目录、独立配置和入口见 [V2项目结构](CURRENT_PROJECT_STRUCTURE.md)及[模型目录](MODEL_CATALOG.md)。旧自采协议不再启用。

# 九个独立模型与PILT式YAML配置

## 代码与配置分工

参考`/data/hzy/experiments/NILM_PILT/PILT`的`configs/models.yaml`及模型注册表实现：
**一份模型YAML中九个完整独立节点**，结构参数用`model_kwargs`，训练参数用
`model_training_param`。没有YAML继承、锚点合并或相互覆盖；各节点独立调整。

| 原骨干 | baseline目录 | 双分支目录 | 热插拔目录 |
|---|---|---|---|
| ArcFENet | `baseline_arc_fenet` | `dual_arc_fenet` | `plugin_arc_fenet` |
| HYDRA | `baseline_hydra` | `dual_hydra` | `plugin_hydra` |
| Fusion-current | `baseline_arc_fusion` | `dual_arc_fusion` | `plugin_arc_fusion` |

九个活动包位于`src/arc_model/implementations/`。每个包拥有自己的`network.py`、
`model.py`、`data_loader.py`、`raw_reader.py`、`configuration.py`、训练入口、评价定义、
配置来源及性能状态，不导入另八个包或旧网络组件。重复代码是用户要求的独立性设计；
生成脚本只在构建期复用模板，运行时没有通用网络块。

旧`src/arc_model/models/`仍用于历史产物重放和来源核对，不作为新增实验的活动目录，
不因整理代码而移动、删改旧冻结源文件或权重。

- `configs/models.yaml`：九个模型节点，唯一活动模型参数来源。
- `configs/datasets.yaml`：两个当前数据集的位置、原生采样率、审核掩码与划分指纹。
- `configs/experiment_config.yaml`：选择模型/数据集、结果命名空间、并行与GPU调度。
- 实验默认`validate_only`，不会因读取配置启动训练；GPU只允许1/2/3，0保留。

## 九个模型节点

```text
ArcFENet_Baseline                 ArcFENet_DualBranch                 ArcFENet_Plugin
HYDRA_Baseline                    HYDRA_DualBranch                    HYDRA_Plugin
ArcFusionNet_Current_Baseline     ArcFusionNet_Current_DualBranch     ArcFusionNet_Current_Plugin
```

`model_kwargs`会实际传入本目录模型构造器：FENet的通道/FEM选项、Fusion的注意力头数/
FF宽度/层数/dropout/池化步长、HYDRA的核数/组数/核长/步长均可核对。输入维度必须
与配置匹配，未知参数不会被静默忽略。固定的论文输入合同与可调构造参数分开记录。
baseline节点受冻结校验保护，候选节点可独立改动。

`model_training_param`用于本目录训练入口，包括lr、wd、epochs、batch_size和SVM的
lambda；双分支头的训练参数在本节点`feature_module.training`。便携训练API写入新
输出目录，不覆盖冻结成绩；精确历史运行来源仍由`provenance.json`追溯。

## 三种角色的含义

**Baseline**：不含本研究PLL/C0/额外高频历史。三个目录的`performance.json`保存
当前分组基线六行指标及种子SD，数值直接来自已核验的标准表，不通过本次重构重训。

**双分支**：独立骨干加36维跨周波动态的分数纠错路径。骨干训练和纠错训练分开；
纠错入口必须得到组隔离的内层OOF及索引收据，校准仅限训练资料。HYDRA新基线是
SVM margin，不能当作旧BCE概率使用，须另有训练内校准及同容量校准-only对照。
骨干权重与纠错NPZ分开保存，不只存骨干`state_dict()`就声称完整模块已保存。

**热插拔**：默认是输入端的统一自同步表征，保持原骨干输入/输出形状，不添加辅助
特征分支。`enabled=False`保持当前传入骨干的原forward；它不意味着训练过的候选
权重会自动恢复冻结baseline成绩。恢复基准须加载对应冻结权重和固定20ms输入。
新的相位输入在现有接口内传入，失效掩码按窗口回退到原输入。

六个改进目录均标为`not_validated_on_locked_parent`，当前性能表为空。旧C0/36维、
静态8维和序列化输入成绩仅作为历史来源，不复制到新冻结父基线。目录拆分或关闭
开关的数值一致性测试，不等于分类增益验证。

## 冻结边界

发布：`releases/baselines_v1_20261030/`。冻结清单保护三个baseline目录的代码、
训练配方、来源、指标快照，以及`models.yaml`中对应三个节点的规范化内容。
六个候选节点修改不会使baseline配置随之变化。baseline内容或数值变化会让校验和CI
失败；需要修复时建立新版本，并同步重跑baseline和相应增强方案，不覆盖旧v1。

此处冻结的是**代码和已报告指标快照**。ArcFENet仍为有VMD数值限制的重建，Fusion
仍是仅电流结构适配，不能用冻结动作代替作者完整方法忠实度确认。详见
[原方法符合程度](BASELINE_FIDELITY_v2.json)和[分组比较标准](GROUPED_BASELINE_STANDARD.md)。

## 使用入口

```bash
arc-models list
arc-models validate
arc-models show --model ArcFENet_DualBranch
arc-models smoke
python scripts/check_baseline_release.py
```

未安装命令时，可用`PYTHONPATH=src python -m arc_model.implementations.registry list`。
各目录的`data_loader.from_project(dataset)`读取当前项目路径及分组掩码；不能把旧
448维HYDRA缓存传给新1,024维SVM。热插拔HYDRA的原文型相位输入仍须从原生电流
重建，不能把旧有符号/448维缓存改名冒用。

本次不启动新分类训练。验证关注九目录无组件交叉依赖、YAML参数生效、分支关闭与
同权重原forward一致，以及冻结70个外折状态经新baseline目录重放的指标不变。
