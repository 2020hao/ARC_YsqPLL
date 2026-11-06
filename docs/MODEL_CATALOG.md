# 独立模型目录（V2）

当前注册63项：3个冻结baseline，60个独立实例。每个固定网络、输入或融合方案有自己的目录；折和种子不是额外模型目录。

默认推荐视图为三个基准和完整峰值动态CNN–SNN。历史实例只迁移结构和参数来源，不转移旧自采成绩。

```bash
arc-project list --recommended
arc-project list --role direct_classifier
arc-project config rcr_cnn_snn_peak_dynamic_affine_calibrated
```

## 当前模型与工程接入

| 模型ID | 清晰名称 | 类型 | 配置 | 验证范围 |
|---|---|---|---|---|
| `baseline_arc_fenet` | ArcFENet本地论文重建（冻结基准） | 冻结基准 | [配置](../configs/models.yaml) | 462条1 MHz基准比较完成；当前公共协议另核验 |
| `baseline_hydra` | HYDRA原文FD配置（冻结基准） | 冻结基准 | [配置](../configs/models.yaml) | 462条1 MHz基准比较完成；当前公共协议另核验 |
| `baseline_arc_fusion` | ArcFusionNet仅电流适配（冻结基准） | 冻结基准 | [配置](../configs/models.yaml) | 462条1 MHz基准比较完成；当前公共协议另核验 |
| `arc_fenet_dynamic_correction_1m` | ArcFENet 动态输出校正 | 动态输出校正 | [配置](../src/arc_model/model_instances/arc_fenet_dynamic_correction_1m/model.yaml) | 工程接入完成；当前462性能待验证 |
| `arc_fenet_phase_aligned_input` | ArcFENet 相位对齐输入 | 相位输入对齐 | [配置](../src/arc_model/model_instances/arc_fenet_phase_aligned_input/model.yaml) | 工程接入完成；当前462性能待验证 |
| `arc_fusion_dynamic_correction_1m` | ArcFusionNet-current 动态输出校正 | 动态输出校正 | [配置](../src/arc_model/model_instances/arc_fusion_dynamic_correction_1m/model.yaml) | 工程接入完成；当前462性能待验证 |
| `arc_fusion_phase_aligned_input` | ArcFusionNet-current 相位对齐输入 | 相位输入对齐 | [配置](../src/arc_model/model_instances/arc_fusion_phase_aligned_input/model.yaml) | 工程接入完成；当前462性能待验证 |
| `hydra_dynamic_correction_1m` | HYDRA 动态输出校正 | 动态输出校正 | [配置](../src/arc_model/model_instances/hydra_dynamic_correction_1m/model.yaml) | 工程接入完成；当前462性能待验证 |
| `hydra_phase_aligned_input` | HYDRA 相位对齐输入 | 相位输入对齐 | [配置](../src/arc_model/model_instances/hydra_phase_aligned_input/model.yaml) | 工程接入完成；当前462性能待验证 |
| `rcr_cnn_snn_capacity_control` | RCR相位CNN–SNN（同容量读出对照） | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_capacity_control/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_dynamic_summary` | RCR相位CNN–SNN＋动态摘要 | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_dynamic_summary/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_original` | RCR相位CNN–SNN（原五通道） | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_original/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_peak_dynamic` | RCR峰值动态CNN–SNN | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_peak_dynamic_affine_calibrated` | RCR峰值动态CNN–SNN＋训练内正仿射校准 | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_affine_calibrated/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_peak_dynamic_augmented` | RCR峰值动态CNN–SNN＋物理一致增广 | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_augmented/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_cnn_snn_peak_dynamic_bias_calibrated` | RCR峰值动态CNN–SNN＋训练内偏置校准 | 直接分类 | [配置](../src/arc_model/model_instances/rcr_cnn_snn_peak_dynamic_bias_calibrated/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_hierarchical_snn_compact` | RCR分层脉冲网络（紧凑） | 直接分类 | [配置](../src/arc_model/model_instances/rcr_hierarchical_snn_compact/model.yaml) | 当前双库开发验证；迁移逐值核验 |
| `rcr_hierarchical_snn_wide` | RCR分层脉冲网络（加宽） | 直接分类 | [配置](../src/arc_model/model_instances/rcr_hierarchical_snn_wide/model.yaml) | 当前双库开发验证；迁移逐值核验 |

## 历史结构与参考对照

下列结构均有独立目录；历史训练预算保存在各自YAML，不代表已完成当前462队列重训。

| 模型ID | 结构名称 | 独立配置 |
|---|---|---|
| `coded_phase_map_clip` | coded phase map clip | [配置](../src/arc_model/model_instances/coded_phase_map_clip/model.yaml) |
| `coded_phase_map_raw` | coded phase map raw | [配置](../src/arc_model/model_instances/coded_phase_map_raw/model.yaml) |
| `coded_phase_map_sparse4` | coded phase map sparse4 | [配置](../src/arc_model/model_instances/coded_phase_map_sparse4/model.yaml) |
| `compact_phase_cnn` | 紧凑相位CNN | [配置](../src/arc_model/model_instances/compact_phase_cnn/model.yaml) |
| `compact_phase_cnn_snn` | 紧凑相位CNN–SNN | [配置](../src/arc_model/model_instances/compact_phase_cnn_snn/model.yaml) |
| `compact_phase_content_attention` | 紧凑相位CNN–SNN＋内容注意力池化 | [配置](../src/arc_model/model_instances/compact_phase_content_attention/model.yaml) |
| `compact_phase_static_attention` | 紧凑相位CNN–SNN＋静态位置池化 | [配置](../src/arc_model/model_instances/compact_phase_static_attention/model.yaml) |
| `current_phase_continuous` | 电流相位连续激活网络（早期对照） | [配置](../src/arc_model/model_instances/current_phase_continuous/model.yaml) |
| `current_phase_snn` | 电流相位CNN–SNN（早期实时版本） | [配置](../src/arc_model/model_instances/current_phase_snn/model.yaml) |
| `cycle_phase_continuous_level_laststate` | cycle phase continuous level laststate | [配置](../src/arc_model/model_instances/cycle_phase_continuous_level_laststate/model.yaml) |
| `cycle_phase_continuous_level_multistate` | cycle phase continuous level multistate | [配置](../src/arc_model/model_instances/cycle_phase_continuous_level_multistate/model.yaml) |
| `cycle_phase_continuous_normalized_laststate` | cycle phase continuous normalized laststate | [配置](../src/arc_model/model_instances/cycle_phase_continuous_normalized_laststate/model.yaml) |
| `cycle_phase_continuous_normalized_multistate` | cycle phase continuous normalized multistate | [配置](../src/arc_model/model_instances/cycle_phase_continuous_normalized_multistate/model.yaml) |
| `cycle_phase_spike_level_laststate` | cycle phase spike level laststate | [配置](../src/arc_model/model_instances/cycle_phase_spike_level_laststate/model.yaml) |
| `cycle_phase_spike_level_multistate` | cycle phase spike level multistate | [配置](../src/arc_model/model_instances/cycle_phase_spike_level_multistate/model.yaml) |
| `cycle_phase_spike_normalized_laststate` | cycle phase spike normalized laststate | [配置](../src/arc_model/model_instances/cycle_phase_spike_normalized_laststate/model.yaml) |
| `cycle_phase_spike_normalized_multistate` | cycle phase spike normalized multistate | [配置](../src/arc_model/model_instances/cycle_phase_spike_normalized_multistate/model.yaml) |
| `deep_phase_map_clip` | deep phase map clip | [配置](../src/arc_model/model_instances/deep_phase_map_clip/model.yaml) |
| `deep_phase_map_raw` | deep phase map raw | [配置](../src/arc_model/model_instances/deep_phase_map_raw/model.yaml) |
| `deep_phase_map_sparse4` | deep phase map sparse4 | [配置](../src/arc_model/model_instances/deep_phase_map_sparse4/model.yaml) |
| `detail_phase_map_allocation` | detail phase map allocation | [配置](../src/arc_model/model_instances/detail_phase_map_allocation/model.yaml) |
| `detail_phase_map_base` | detail phase map base | [配置](../src/arc_model/model_instances/detail_phase_map_base/model.yaml) |
| `detail_phase_map_both` | detail phase map both | [配置](../src/arc_model/model_instances/detail_phase_map_both/model.yaml) |
| `detail_phase_map_burst` | detail phase map burst | [配置](../src/arc_model/model_instances/detail_phase_map_burst/model.yaml) |
| `detail_phase_map_capacity` | detail phase map capacity | [配置](../src/arc_model/model_instances/detail_phase_map_capacity/model.yaml) |
| `fixed20_ttfs_snn_reference` | 固定20 ms参考＋TTFS脉冲主干 | [配置](../src/arc_model/model_instances/fixed20_ttfs_snn_reference/model.yaml) |
| `midband_phase_map_base` | midband phase map base | [配置](../src/arc_model/model_instances/midband_phase_map_base/model.yaml) |
| `midband_phase_map_both` | midband phase map both | [配置](../src/arc_model/model_instances/midband_phase_map_both/model.yaml) |
| `midband_phase_map_competition` | midband phase map competition | [配置](../src/arc_model/model_instances/midband_phase_map_competition/model.yaml) |
| `midband_phase_map_mid` | midband phase map mid | [配置](../src/arc_model/model_instances/midband_phase_map_mid/model.yaml) |
| `phase_cycle_cnn_history` | 相位CNN＋因果周波历史卷积 | [配置](../src/arc_model/model_instances/phase_cycle_cnn_history/model.yaml) |
| `phase_cycle_cnn_single` | 相位CNN（单周波对照） | [配置](../src/arc_model/model_instances/phase_cycle_cnn_single/model.yaml) |
| `phase_event_direct_continuous` | phase event direct continuous | [配置](../src/arc_model/model_instances/phase_event_direct_continuous/model.yaml) |
| `phase_event_direct_snn` | phase event direct snn | [配置](../src/arc_model/model_instances/phase_event_direct_snn/model.yaml) |
| `phase_event_rate_continuous` | phase event rate continuous | [配置](../src/arc_model/model_instances/phase_event_rate_continuous/model.yaml) |
| `phase_event_rate_snn` | phase event rate snn | [配置](../src/arc_model/model_instances/phase_event_rate_snn/model.yaml) |
| `phase_event_ttfs_continuous` | phase event ttfs continuous | [配置](../src/arc_model/model_instances/phase_event_ttfs_continuous/model.yaml) |
| `phase_event_ttfs_snn` | phase event ttfs snn | [配置](../src/arc_model/model_instances/phase_event_ttfs_snn/model.yaml) |
| `position_preserving_cnn_snn` | 位置保留相位CNN–SNN | [配置](../src/arc_model/model_instances/position_preserving_cnn_snn/model.yaml) |
| `scale_robust_batch_cosine` | scale robust batch cosine | [配置](../src/arc_model/model_instances/scale_robust_batch_cosine/model.yaml) |
| `scale_robust_batch_linear` | scale robust batch linear | [配置](../src/arc_model/model_instances/scale_robust_batch_linear/model.yaml) |
| `scale_robust_sample_cosine` | scale robust sample cosine | [配置](../src/arc_model/model_instances/scale_robust_sample_cosine/model.yaml) |
| `scale_robust_sample_linear` | scale robust sample linear | [配置](../src/arc_model/model_instances/scale_robust_sample_linear/model.yaml) |
| `self462_burst_phase_map` | 相位CNN–SNN＋高频脉冲集中度（自采候选） | [配置](../src/arc_model/model_instances/self462_burst_phase_map/model.yaml) |
| `sogi_ttfs_snn_reference` | SOGI参考＋TTFS脉冲主干 | [配置](../src/arc_model/model_instances/sogi_ttfs_snn_reference/model.yaml) |

## 每实例文件

`network.py`及本地 `_source_*.py`保留完整网络与必要来源依赖；`data_loader.py`和 `preprocessing.py`负责本模型输入；`training.py`与 `inference.py`负责训练和推理适配；`model.yaml`是唯一活动模型配置；`README.md`和 `provenance.json`记录职责与来源。

三个封存基准继续使用原目录和原封存配置节点，注册表引用并验证它们，不另造可冲突的默认配置。
