"""Render current-cohort aggregate tables for README; never train or expose OOF.

Evidence paths are supplied explicitly. The generated document contains only
aggregate metrics; native data, labels, per-window scores and weights stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {
    "base": "原五通道CNN–SNN",
    "dynamic": "原模型＋动态摘要",
    "capacity": "仅扩大分类器容量",
    "maxpool_dynamic": "峰值池化＋动态摘要",
    "gain_phase": "峰值动态模型＋物理一致增广",
    "hierarchy_small": "早期相位LIF小模型",
    "hierarchy_wide": "早期相位LIF宽模型",
    "folded_bias": "增广峰值动态模型＋偏置校准",
    "folded_affine": "增广峰值动态模型＋正仿射校准（完整模型）",
}
DATASETS = {"self462": "自采462次", "zenodo5501527": "公共1,357窗口"}
BASELINES = {
    "ArcFENet_Baseline": "ArcFENet本地论文重建",
    "HYDRA_Baseline": "HYDRA原文FD配置",
    "ArcFusionNet_Current_Baseline": "ArcFusionNet仅电流适配",
    "cnn_snn_1m": "原五通道CNN–SNN，1 MHz",
    "cnn_snn_2m_control": "原五通道CNN–SNN，匹配2 MHz",
}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def percent(row, name, *, sd=False):
    value = f"{100 * row[name + '_mean']:.2f}"
    return value + (f" ± {100 * row[name + '_std']:.2f}" if sd else "")


def render(evidence: Path, output: Path):
    summary = evidence / "model_upgrade_summary_20261105"
    baseline = evidence / "self462_1m_baselines_20261105"
    for path in (summary / "REPORT_CHECK.json", baseline / "FINAL_CHECK.json"):
        if json.loads(path.read_text())["status"] != "passed":
            raise ValueError(f"Completed evidence required: {path}")
    metrics = pd.read_csv(summary / "完整模型指标_均值与标准差.csv")
    seeds = pd.read_csv(summary / "全部模型_逐种子指标.csv")
    loads = pd.read_csv(summary / "逐负载完整指标_均值与标准差.csv")
    costs = pd.read_csv(summary / "模型参数与计算量.csv").set_index("arm")
    effects = pd.read_csv(summary / "阶段主比较_区间与Holm.csv")
    cohort = pd.read_csv(summary / "数据与分组说明.csv").set_index("dataset")
    strong = pd.read_csv(summary / "旧强SNN公共参照.csv").iloc[0]
    reference = pd.read_csv(baseline / "metrics.csv").set_index("model")
    reference_cost = pd.read_csv(baseline / "model_cost.csv").set_index("model")
    if len(metrics) != 18 or set(metrics.arm) != set(ARMS):
        raise ValueError("Exactly nine current variants on both datasets required")
    for ds, expected in (("self462", (462, 125051, 156)), ("zenodo5501527", (13, 1357, 13))):
        row = cohort.loc[ds]
        if tuple(int(row[col]) for col in ("录波数", "窗口数", "工况或电器组数")) != expected:
            raise ValueError("Retired or incompatible cohort in evidence")
        rows = seeds[seeds.dataset == ds]
        if len(rows) != 27 or set(rows.seed) != {0, 1, 2} or set(rows.N) != {expected[1]}:
            raise ValueError("Incomplete seed coverage")
        for arm in ARMS:
            saved = metrics[(metrics.dataset == ds) & (metrics.arm == arm)].iloc[0]
            part = rows[rows.arm == arm]
            for name in ("Accuracy", "BAcc_condition", "Recall", "FPR", "F1", "AP"):
                np.testing.assert_allclose(saved[name + "_mean"], part[name].mean(), atol=2e-10, rtol=0)
                np.testing.assert_allclose(saved[name + "_std"], part[name].std(ddof=1), atol=2e-10, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    top_rows = []
    for ds in DATASETS:
        for arm in ("base", "folded_affine"):
            r = metrics[(metrics.dataset == ds) & (metrics.arm == arm)].iloc[0]
            top_rows.append([DATASETS[ds], ARMS[arm], percent(r, "Accuracy", sd=True),
                             *[percent(r, name) for name in ("BAcc_condition", "Recall", "FPR", "F1", "AP")]])
    top = table(["数据集", "模型", "Acc%（均值±SD）", "组等权BAcc%", "Recall%", "FPR% ↓", "F1%", "AP%"], top_rows)
    baseline_rows = []
    for key, name in BASELINES.items():
        r = reference.loc[key]
        cost = reference_cost.loc["cnn_snn_1m" if key == "cnn_snn_2m_control" else key]
        baseline_rows.append([name, *[f"{100 * r[m]:.2f}" for m in ("Acc", "BAcc", "Recall", "FPR", "F1")],
                              f"{int(r.parameters):,}", f"{float(cost.weighted_MAC_per_output)/1e6:.6f}"])
    base_table = table(["方法", "Acc%", "BAcc%", "Recall%", "FPR% ↓", "F1%", "参数", "加权MAC（M）"], baseline_rows)
    ablation_rows = []
    for arm, name in ARMS.items():
        parts = [metrics[(metrics.dataset == ds) & (metrics.arm == arm)].iloc[0] for ds in DATASETS]
        ablation_rows.append([name, f"{int(costs.loc[arm].parameters):,}",
                              *[percent(r, "BAcc_condition", sd=True) for r in parts]])
    ablation = table(["方案", "参数", "自采BAcc%（均值±SD）", "公共BAcc%（均值±SD）"], ablation_rows)
    strong_table = table(["公共同队列系统", "Acc%（均值±SD）", "BAcc%", "Recall%", "FPR% ↓", "F1%", "参数", "矩阵MAC（M）"],
                         [["SOGI＋原值TTFS全事件SNN", percent(strong, "Accuracy", sd=True),
                           *[percent(strong, n) for n in ("BAcc_condition", "Recall", "FPR", "F1")],
                           f"{int(strong.parameters):,}", f"{strong.matrix_MAC_M:.6f}"]])
    status = table(["实验", "当前状态"], [
        ["9种自有模型方案，双库分组、三种子", "已完成，54份整套OOF核验"],
        ["自采1 MHz三基准与匹配采样率比较", "已完成，四折、seed0、20个最终模型"],
        ["公共同队列SOGI＋TTFS强参照", "索引、标签和外折一致；独立系统参照"],
        ["当前462次上的三基准动态输出校正", "尚未完成同队列性能验证"],
        ["当前公共1,357窗口的三个baseline统一补齐", "尚未完成；旧1,361窗口结果不并入"],
    ])
    brief = "\n\n".join([
        "## 已完成实验展示",
        "以下表格只使用新自采462次实验队列和当前公共1,357窗口。468是最新标注导出库总量；现有实验按既定目标规则纳入462次。指标以已有聚合CSV及通过的审计收据为依据。",
        "### 双数据集主结果",
        "自采原生2 MHz、四折工况隔离；公共原生1 MHz、五折电器/录波组隔离。两库分别训练，均为三种子、50轮、概率阈值0.5。Acc给出种子均值±标准差，其余主表给出均值。", top,
        "完整模型为8,249个推理参数、0.684960M矩阵MAC/输出，原模型为7,097参数、0.683808M。两者均为九周波上下文直接分类；校准已折入末层。原始滤波、RCR及动态摘要标量成本另计。",
        "### 自采1 MHz基准与采样率对照",
        "同125,051窗口、156工况、四折、seed0；1 MHz由2 MHz原始电流因果抗混叠得到。该表与上述原生2 MHz三种子研究分别报告。", base_table,
        "三个基准使用单20 ms输入，CNN–SNN使用九周波上下文。ArcFENet为本地数值重建，Fusion-current为仅电流适配；HYDRA另有2,560个固定核系数，其FD变换计入加权运算。表中MAC均不等于包含全部预处理的端到端FLOPs。",
        "### 九种模型方案与消融", ablation,
        "单独加入动态摘要没有形成双库一致增益；早期相位LIF在公共数据出现退化，保留完整结果。物理一致增广及正仿射校准在自采主比较中有校正后正向支持，公共相应增量尚未形成同样统计支持。",
        "### 公共同队列强参照", strong_table,
        "该参照与当前公共数据的1,357个窗口、标签及外折一致，但参考、编码和骨干不同，属于完整系统比较。当前完整CNN–SNN误报更低，Acc与BAcc均值仍低于此参照。强成绩来源为原研究raw_sogi，不绑定配置不同的新独立实例。",
        "### 验证状态与完整结果", status,
        "[查看完整指标、参数计算量、逐负载结果和配对统计](docs/EXPERIMENT_RESULTS.md)。完整模型在公共电脑、水泵上的Recall分别为60.87%、45.83%，自采电视FPR为8.82%，这些失效情况也保留在逐负载表中。",
    ])
    (output / "README_EXPERIMENT_BLOCK.md").write_text(brief + "\n")
    detail = ["# 当前数据集的已有实验结果", "",
              "本页由已完成研究的聚合结果生成，保留正向、无增益和退化结果。自采实验纳入462次、125,051窗口、156工况；公共1,357窗口、13录波组。468次原始导出库与462次实验队列的关系见[数据范围](DATASET_SCOPE_20261106.md)。", "", status, ""]
    measures = [("Accuracy", "Acc%"), ("BAcc_condition", "组等权BAcc%"), ("Precision", "Precision%"),
                ("Recall", "Recall%"), ("FPR", "FPR% ↓"), ("F1", "F1%"), ("AP", "AP%"), ("ROC_AUC", "ROC-AUC%")]
    for ds, name in DATASETS.items():
        detail += ["## " + name + "：九方案完整主指标", "", "全部指标为三种子均值±标准差，保留全部外折OOF；不是最佳种子成绩。", "",
                   table(["方案", *[title for _, title in measures]],
                         [[ARMS[arm], *[percent(metrics[(metrics.dataset == ds) & (metrics.arm == arm)].iloc[0], metric, sd=True) for metric, _ in measures]] for arm in ARMS]), ""]
        rows = []
        for arm in ARMS:
            r = metrics[(metrics.dataset == ds) & (metrics.arm == arm)].iloc[0]
            rows.append([ARMS[arm], *[percent(r, m) for m in ("Specificity", "FNR", "Macro_F1")], f"{r.MCC_mean:.4f}",
                         *[f"{r[m + '_mean']:.2f}" for m in ("TP", "FP", "TN", "FN")]])
        detail += ["### 补充指标与混淆计数", "", "混淆计数为三种子均值，允许小数；单个种子的计数仍为整数。", "",
                   table(["方案", "Specificity%", "FNR%", "Macro-F1%", "MCC", "TP均值", "FP均值", "TN均值", "FN均值"], rows), ""]
    detail += ["## 自采1 MHz基准与采样率比较", "", "该研究为seed0，固定125,051窗口/四折；不要与三种子表混成同预算排名。", "", base_table, "",
               "三个基准的完整概率指标（包括AP、ROC-AUC和MCC）：", "",
               table(["方法", "Precision%", "Specificity%", "AP%", "ROC-AUC%", "MCC", "TP", "FP", "TN", "FN"],
                     [[BASELINES[k], *[f"{100 * reference.loc[k, m]:.2f}" for m in ("Precision", "Specificity", "AP", "ROC_AUC")],
                       f"{reference.loc[k, 'MCC']:.4f}", *[int(reference.loc[k, m]) for m in ("TP", "FP", "TN", "FN")]] for k in BASELINES]), "",
               "## 公共同队列SOGI＋TTFS强参照", "", strong_table, "",
               "该行来自phase_event_raw研究中的raw_sogi、非分解输入，全部15份外折训练/测试索引与当前公共协议一致；不是新sogi_ttfs_snn_reference配置的已完成成绩，也不是参考或编码的单因素对照。", "",
               "## 参数与模型计算量", "",
               table(["方案", "参数", "矩阵MAC（M）", "名义矩阵MFLOPs", "LIF标量更新", "训练期校准标量"],
                     [[ARMS[a], f"{int(costs.loc[a].parameters):,}", f"{costs.loc[a].matrix_MAC/1e6:.6f}",
                       f"{costs.loc[a].nominal_matrix_FLOPs/1e6:.6f}", int(costs.loc[a].LIF_scalar_updates), int(costs.loc[a].calibration_fitted_scalars)] for a in ARMS]), "",
               "1 MAC按2个名义FLOPs换算。滤波、RCR、动态摘要、归一化、激活、池化和数据搬运未统一并入矩阵MAC；不据此推断硬件能耗。正仿射的两个训练期标量折入原分类层，不新增推理参数槽。", ""]
    public_names = {"computer": "台式电脑", "dolcegusto": "咖啡机", "drill550": "550 W电钻", "drill710": "710 W电钻",
                    "foodprocessor": "食品加工机", "halogen": "卤素灯", "heater": "暖风机", "jigsaw135": "曲线锯", "kettle": "电水壶",
                    "printer": "打印机", "pump": "水泵", "vacuumcleaner1200": "1200 W吸尘器", "vacuumcleaner1250": "1250 W吸尘器"}
    for ds, name in DATASETS.items():
        part = loads[(loads.dataset == ds) & (loads.arm == "folded_affine")]
        expected = 4 if ds == "self462" else 13
        if len(part) != expected:
            raise ValueError("Incomplete per-appliance coverage")
        detail += ["## " + name + "：完整模型逐负载结果", "", "以下为三种子均值，全部目标均保留。", "",
                   table(["负载", "窗口", "工况/组数", "Acc%", "BAcc%", "Recall%", "FPR% ↓", "F1%"],
                         [[public_names.get(r.load, r.load), int(r.N), int(r.groups),
                           *[percent(r, m) for m in ("Accuracy", "BAcc_condition", "Recall", "FPR", "F1")]] for _, r in part.iterrows()]), ""]
    detail += ["## 分阶段配对统计", "",
               "增量单位pp；区间以工况/电器录波组配对bootstrap获得。Holm校正在各阶段的预设比较族内执行；均值改善不自动等于部件必要性。", "",
               table(["数据集", "比较", "ΔBAcc（pp）", "95%组配对区间", "Holm p", "校正后正向支持"],
                     [[DATASETS[r.dataset], ARMS[r.candidate] + " − " + ARMS[r.comparator], f"{r.delta_pp:+.3f}",
                       f"[{r.CI_low_pp:+.3f}, {r.CI_high_pp:+.3f}]", f"{r.p_holm:.4g}", "是" if r.positive_adjusted_support else "否"] for _, r in effects.iterrows()]), "",
               "## 数据和来源", "",
               "源聚合表：`outputs/model_upgrade_summary_20261105/完整模型指标_均值与标准差.csv`、`模型参数与计算量.csv`、`逐负载完整指标_均值与标准差.csv`、`阶段主比较_区间与Holm.csv`；三基准来源为 `outputs/self462_1m_baselines_20261105/metrics.csv`与 `model_cost.csv`。核验收据分别为 `REPORT_CHECK.json`和 `FINAL_CHECK.json`。", "",
               "公开页面只展示聚合表。原始电流、逐记录标签、逐窗OOF、权重及Excel留本地；当前数据组经历多轮开发，现有结果为分组开发评价。", ""]
    (output / "EXPERIMENT_RESULTS.md").write_text("\n".join(detail))
    files = [summary / n for n in ("完整模型指标_均值与标准差.csv", "全部模型_逐种子指标.csv", "逐负载完整指标_均值与标准差.csv", "模型参数与计算量.csv", "阶段主比较_区间与Holm.csv", "数据与分组说明.csv", "旧强SNN公共参照.csv", "REPORT_CHECK.json")]
    files += [baseline / n for n in ("metrics.csv", "model_cost.csv", "FINAL_CHECK.json")]
    receipt = {"status": "passed", "current_variants": 9, "current_datasets": 2, "seed_summary_rows_checked": 54,
               "per_appliance_groups_displayed": 17, "old_self_results_included": False, "training_started": False,
               "sources": [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (output / "SHOWCASE_CHECK.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence_root, args.output)
