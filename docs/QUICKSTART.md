# 当前项目快速开始

当前数据为462条自采与公共1357窗口；旧自采不用于当前训练或结果对比。Python3.11/3.12，服务器可使用已有 `NILMFormer`环境。

```bash
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' arc-project validate
CUDA_VISIBLE_DEVICES='' arc-project list --recommended
CUDA_VISIBLE_DEVICES='' arc-project config rcr_cnn_snn_peak_dynamic_affine_calibrated
CUDA_VISIBLE_DEVICES='' arc-project build rcr_cnn_snn_peak_dynamic_affine_calibrated
```

源码方式用 `python scripts/project.py`替代 `arc-project`，无需重装即可访问新命令。

## 配置本机数据

复制 `configs/local/paths.example.yaml`到 `configs/local/paths.yaml`后填写真实路径，或设置同名环境变量。实际paths文件已被Git忽略，本机已绑定现有缓存。其他机器需要提供自己的数据路径与相同协议的加工产物。

```bash
CUDA_VISIBLE_DEVICES='' arc-project data-audit self462
CUDA_VISIBLE_DEVICES='' arc-project data-audit public
CUDA_VISIBLE_DEVICES='' arc-project plan
```

审计命令检查纳入数量、采集/工况隔离及外折覆盖，不训练。计划命令只显示比较计划。原始数据准备及完整研究训练目前仍有阶段依赖；配置文件或CPU构建通过不代表数据、权重或完整研究已经复现。

## 核验源码迁移

```bash
CUDA_VISIBLE_DEVICES='' python scripts/check_baseline_release.py
CUDA_VISIBLE_DEVICES='' python -m pytest tests/contracts tests/modules tests/model_instances tests/orchestration tests/data_io tests/parity -q
CUDA_VISIBLE_DEVICES='' python scripts/audit_project_layout_v2.py
```

使用私有证据做完整现有权重迁移核验时，额外显式传入 `--checkpoint-parity --evidence-root <既有outputs> --self-prepared <自采加工目录> --public-prepared <公共加工目录>`。该核验不拟合模型或生成标签。

当前CLI提供管理和校验；每个模型目录的 `training.py`保留自己的训练适配。原全研究脚本继续作为兼容执行器，当前研究必须显式选择462条及公共1357协议。GPU训练只允许物理1、2、3号，0号保留。
