# 论文实验复现

本入口对应 2026-11-13 中文稿的 CNN–SNN、双数据集消融及三个基准的输出校正实验。公开数据以 1,357 个窗口为准，自采主实验使用 462 次录波、156 个工况和 125,051 个共同窗口。

## 输入

- 自采：462 个已审核的 NET9770_114 BIN（2 MHz，读取 CH0）和随数据发放的 `self462_manifest.csv`。清单是人工复核后的数据版本，包含采集 ID、工况、起弧和熄弧时间，以及原 BIN 的路径。路径可以用 `--recordings-root` 重新定位。
- 公共：Zenodo 5501527 的 13 个 C3 DAT 文件，以及随数据发放的 `public_annotations.csv`。该文件保存人工复核后的 1,364 个候选锚点和 1,357 个保留标记。C2 用于此前的标注复核。重新划分时保留既有的文件归组，静态映射见 [DATASETS.md](DATASETS.md)。
- 两份标注清单属于数据交付包。仓库不包含录波、标签、检查点或离线预测。持有历史实验产物的项目维护者可用 `scripts/reproduction/export_annotations.py --help` 从冻结的清单和索引导出同一数据版本。

在 Linux 上使用 Python 3.12 和 CUDA PyTorch，安装本实验验证过的依赖组合。RTX 5090 的原运行环境使用 CUDA 13.0 和 PyTorch 2.13.0+cu130。

```sh
python -m pip install -r requirements-verified.txt
python -m pip install -e . --no-deps
```

## 从原始录波到结果

从仓库根目录运行。路径示例中的 `inputs` 为随数据交付的标注目录。GPU 选择按实际机器修改，当前服务器保留 0 号卡。

```sh
CUDA_VISIBLE_DEVICES="" python scripts/reproduction/public_data.py \
  --recordings /path/to/zenodo5501527 --annotations /path/to/inputs/public_annotations.csv

CUDA_VISIBLE_DEVICES="" python scripts/reproduction/self_data.py \
  --manifest /path/to/inputs/self462_manifest.csv --recordings-root /path/to/bin --workers 4

python scripts/reproduction/runner.py features --gpus 1
python scripts/reproduction/runner.py self-baselines --gpus 1
python scripts/reproduction/runner.py public-reference --gpus 1
python scripts/reproduction/runner.py dynamic_state --gpus 1 2 3
python scripts/reproduction/runner.py phase_hierarchy --gpus 1 2 3
python scripts/reproduction/runner.py folded_calibration --gpus 1 2 3
python scripts/reproduction/runner.py correction --gpus 1 2 3
```

数据准备生成 `cycles.csv`、保留索引、分组训练/测试索引和信号表示。自采四折按目标类型内的工况打乱顺序分配，种子固定为 20261105。公共五折按照已核对的 13 个文件分配。校准只读当折的外层训练样本，折外预测由两组内层训练生成。

三个基准和 CNN–SNN 保留各自的预处理、训练预算、校准阈值与评估代码。`features`、`self-baselines` 与 `public-reference` 生成基准表征和预测，后三个 CNN–SNN 阶段依次生成动态状态、层次模型和训练折内校准结果。复现使用独立的新检出目录。各阶段输出在 Git 忽略的 `outputs/` 下。最终 CNN–SNN 指标见 `folded_calibration_20261105/main_results.csv`。三个基准的四种校正对照、配对结果和回放审计见 `current_correction_20261109/`。只重跑公共校正时，向 `features` 和 `correction` 添加 `--dataset zenodo5501527`。

复现时应比较输入数组、折索引和逐窗预测，再核对 Accuracy、工况等权 BAcc、Recall 和 FPR。输出校正指标可用 `python scripts/reproduction/compare_metrics.py reference.csv rebuilt.csv --tolerance-pp 0` 检查。库版本、显卡及浮点归约顺序会影响训练数值。训练时间依赖实际硬件与调度。

本次从13份公开原始录波生成的五折索引、保留索引和CNN–SNN输入与冻结实验逐元素一致。15个原始公共CNN–SNN折×种子训练的预测逐元素一致。公共三基准校正已重新运行15个外折并通过得分复算。HYDRA四个版本的检测指标与已存结果一致，ArcFENet和Fusion-current有部分指标差异。完整自采重训仍需单独核验。

## 七算法补充表的来源和复现边界

表 IX–XI 来自另一次种子 1 的七算法双版本实验。暂存证据包含汇总指标、逐折混淆矩阵和耗时。四个额外骨干和七算法当天的训练、划分及逐窗预测的源产物尚未在此工作区找到。复现这些表需要该次运行的代码快照、模型预算、录波到各折的实际映射和各算法保留窗口的完整索引。此处的种子 0 入口对应论文表 VII–VIII 的三个基准。表 VII 中四个补充原模型的自采结果同样需要对应的原始训练源码和样本清单。
