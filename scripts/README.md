# 脚本入口

当前项目管理入口为 `python scripts/project.py`（安装后为 `arc-project`）：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/project.py validate
CUDA_VISIBLE_DEVICES='' python scripts/project.py list --recommended
CUDA_VISIBLE_DEVICES='' python scripts/project.py data-audit self462
CUDA_VISIBLE_DEVICES='' python scripts/project.py data-audit public
CUDA_VISIBLE_DEVICES='' python scripts/project.py plan
```

| 脚本 | 用途 |
|---|---|
| `project.py` | V2模型、配置与当前数据入口 |
| `check_baseline_release.py` | 冻结三基准源码及配置校验 |
| `audit_project_layout_v2.py` | 原源码保留与现有权重迁移一致性核验 |
| `report_model_upgrade_20261105.py --check-only` | 当前自有模型完整结果只读复核，需要本机产物 |
| `export_code_snapshot.py --output <空目录>` | 导出允许公开的代码和配置，不上传GitHub |

带日期的准备、训练和报告脚本保留原路径，以便重放既有研究。新工程逻辑进入 `src/arc_model/`；不再将实验runner作为另一个runner的通用工具模块。

旧 `run_experiments.py`、`run_experiment_suite.py`默认配置及 `legacy/`属于历史兼容入口。当前使用时必须显式指定现行研究配置，不能自动采用旧自采队列。所有GPU训练只用物理1/2/3，CPU核验禁用CUDA。

目录职责见 [CURRENT_PROJECT_STRUCTURE](../docs/CURRENT_PROJECT_STRUCTURE.md)，安装与命令见 [QUICKSTART](../docs/QUICKSTART.md)。
