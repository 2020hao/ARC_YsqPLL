> 历史接口说明：本文保留原版本来源；当前模型目录、独立配置和入口见 [V2项目结构](CURRENT_PROJECT_STRUCTURE.md)及[模型目录](MODEL_CATALOG.md)。旧自采协议不再启用。

# Codebase guide

The project follows PILT's separation of declarative configuration, small entries,
reusable implementations, tests and immutable local experiment evidence.

| Layer | Responsibility | Does not own |
|---|---|---|
| `data/current_recording.py` | Native sampling rate, current channel, timestamps | Labels inferred from shape |
| `preprocessing/phase_reference.py` | SOGI/RCR/CFT facade, low-rate current reference | Classification |
| `preprocessing/phase_windows.py` | Available complete cycles, native-HF integration | Folds or feature fitting |
| `methods/phase_representation/` | Fixed10D candidates, explicit masks, controls | Files, labels, model names |
| `training/phase_features.py` | Train-index-only scaler and serializable state | Test fitting |
| `models/phase_adapter.py` | Pre-classifier fusion around unchanged local models | Data paths or phase estimation |
| `experiments/phase_cli.py` | Argument parsing and artifact orchestration | Numerical algorithm definitions |
| `tests/phase_module/` | Scientific and interface contracts | Claims of detection efficacy |

Dependencies flow from entries to data/preprocessing/methods/models. Features do
not import models. Models do not import scripts, datasets or results. Evaluation
must consume saved predictions without updating model or preprocessing state.

## Source preservation

Original model files, SOGI/RCR numerical kernels, CFT candidate and historical C0
remain unchanged. The new wrapper adapts existing models by composition. This
preserves checkpoint interfaces and supports explicit parity checks.

The source directory still contains historical research implementations; they
remain import-compatible. Local frozen runners are not moved or reformatted:
completed evidence can depend on their byte identities. They are not part of the
public portable entry. Adding a new study does not justify rewriting old results.

## New experiments

The engineering entry is `arc-phase`, configured by
`configs/modules/phase_representation.yaml`. It prepares unlabeled records or
features, and checks model interfaces. It does not invent labels, materialize all
backbone representations or silently launch classification training.

Before a full study, freeze a dual-dataset protocol, reviewed interval mapping,
source/condition-disjoint folds, reference fallback policy, original model fidelity,
feature set, equal-information controls and budget. All retained descriptors need
retrained removal evidence. Source preparation is per-record; model evaluation is
separate per dataset. Physical GPU0 must remain unused by project training.

## Public and local material

GitHub contains reusable code, selected implementation docs, tests and config
examples. Raw recordings, per-record labels, model weights, drafts, all metrics
and the private pre-refactor baseline stay local. See `GIT_WORKFLOW.md`.
# 2026-10-29 独立模型入口

模型结构现在分别保存在`models/standalone/arc_fenet.py`、
`models/standalone/arc_fusion.py`和`models/standalone/hydra.py`。
三个文件不导入项目中的公共网络块，也不相互引用。

模型专用加载器分别为`data/arc_fenet_loader.py`、
`data/arc_fusion_loader.py`和`data/hydra_loader.py`。
它们不继承通用加载器。原始信号读取和相位估计继续属于前端。
详见[独立模型与加载器](INDEPENDENT_MODELS.md)及[变更记录](CHANGELOG.md)。
