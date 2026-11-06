"""Build a CPU-only inventory of the three completed direct-model studies.

This script never fits, selects, calibrates, or changes a model. It refuses to
write a report until all three FINAL_CHECK.json receipts say ``passed``. Every
included out-of-fold score is checked against the unchanged cohort, labels,
fold/seed metrics, and condition metrics before being summarized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap as boot
from arc_model.evaluation.cycle_metrics import cycle_metrics


ROOT = boot.ROOT
DEFAULT_OUT = ROOT / "outputs/model_upgrade_summary_20261105"
STAGES = {
    "dynamic_state": ROOT / "outputs/dynamic_state_20261105",
    "phase_hierarchy": ROOT / "outputs/phase_hierarchy_20261105",
    "folded_calibration": ROOT / "outputs/folded_calibration_20261105",
}
ARM_STAGE = {
    "base": "dynamic_state",
    "dynamic": "dynamic_state",
    "capacity": "dynamic_state",
    "maxpool_dynamic": "dynamic_state",
    "gain_phase": "phase_hierarchy",
    "hierarchy_small": "phase_hierarchy",
    "hierarchy_wide": "phase_hierarchy",
    "folded_bias": "folded_calibration",
    "folded_affine": "folded_calibration",
}
LABELS = {
    "base": "原CNN–SNN",
    "dynamic": "显式动态摘要",
    "capacity": "仅增大分类器容量",
    "maxpool_dynamic": "最大池化＋动态摘要",
    "gain_phase": "最大池化动态＋物理一致增广",
    "hierarchy_small": "早期相位LIF小模型",
    "hierarchy_wide": "早期相位LIF宽模型",
    "folded_bias": "物理一致增广＋折入偏置校准",
    "folded_affine": "物理一致增广＋折入正仿射校准",
    "raw_sogi_reference": "旧强参照：SOGI＋TTFS全事件SNN",
}
DATASETS = ("self462", "zenodo5501527")
DS_LABELS = {"self462": "完整自采462条", "zenodo5501527": "公共连续数据"}
METRICS = [
    "BAcc_condition", "BAcc_pooled", "Accuracy", "Precision", "Recall",
    "Specificity", "FPR", "FNR", "F1", "Macro_F1", "ROC_AUC", "AP",
    "MCC", "TP", "FP", "TN", "FN",
]
PERCENT = set(METRICS[:12])
TITLES = {
    "BAcc_condition": "工况等权BAcc", "BAcc_pooled": "汇总BAcc",
    "Accuracy": "Acc", "Precision": "Precision", "Recall": "Recall",
    "Specificity": "Specificity", "FPR": "FPR", "FNR": "FNR", "F1": "F1",
    "Macro_F1": "Macro-F1", "ROC_AUC": "ROC-AUC", "AP": "AP", "MCC": "MCC",
}
KEY_METRICS = ["BAcc_condition", "Accuracy", "Precision", "Recall", "FPR", "F1", "AP", "ROC_AUC"]
REFER = ROOT / "outputs/phase_event_raw_20261104"
SELF = ROOT / "outputs/self462_1m_baselines_20261105"
PUBLIC = ROOT / "outputs/map_input_calibration_20261105/data/zenodo5501527"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(path: Path, label: str | None = None) -> str:
    return f"[{label or path.name}]({path.resolve()})"


def require_completed():
    receipts = {}
    errors = []
    for stage, folder in STAGES.items():
        receipt = folder / "FINAL_CHECK.json"
        if not receipt.exists():
            errors.append(f"{stage}: 缺少 {receipt}")
            continue
        data = json.loads(receipt.read_text())
        if data.get("status") != "passed":
            errors.append(f"{stage}: FINAL_CHECK状态为{data.get('status')!r}")
        else:
            receipts[stage] = data
        for name in ("main_results.csv", "seed_metrics.csv", "per_condition_seed.csv", "paired_effects.csv", "protocol.json"):
            if not (folder / name).exists():
                errors.append(f"{stage}: 缺少 {name}")
        if stage != "folded_calibration" and not (folder / "model_cost.csv").exists():
            errors.append(f"{stage}: 缺少 model_cost.csv")
    if errors:
        raise SystemExit("三阶段尚未全部通过，未生成或覆盖报告：\n" + "\n".join(errors))
    return receipts


def cohorts():
    self_table = pd.read_csv(SELF / "cycles.csv")
    public_table = pd.read_csv(PUBLIC / "cycles.csv")
    tables = {"self462": self_table, "zenodo5501527": public_table}
    kept = {"self462": np.arange(len(self_table)), "zenodo5501527": np.load(PUBLIC / "keep.npy")}
    rows = []
    for ds in DATASETS:
        table, ids = tables[ds], kept[ds]
        load = "load" if ds == "self462" else "appliance"
        assert load in table, f"缺少逐负载列：{ds}/{load}"
        assert table.iloc[ids].groupby("condition_key").label.nunique().min() == 2
        rows.append(dict(dataset=ds,数据集=DS_LABELS[ds],录波数=table.iloc[ids].record_id.nunique(),
                         工况或电器组数=table.iloc[ids].condition_key.nunique(),窗口数=len(ids),
                         正常窗口=int((table.iloc[ids].label == 0).sum()),
                         电弧窗口=int((table.iloc[ids].label == 1).sum()),外折数=4 if ds == "self462" else 5,
                         训练种子="0,1,2",原生采样率="2 MHz" if ds == "self462" else "1 MHz"))
    assert len(kept["self462"]) == 125051 and self_table.record_id.nunique() == 462
    assert len(kept["zenodo5501527"]) == 1357
    return tables, kept, pd.DataFrame(rows)


def assert_row(saved, actual, context):
    for metric in ("N", "groups", *METRICS):
        if metric in saved and not pd.isna(saved[metric]):
            np.testing.assert_allclose(float(saved[metric]), actual[metric], atol=2e-10, rtol=2e-10,
                                       err_msg=f"{context}: {metric}")


def aggregate(frame, keys):
    result = frame.groupby(keys, sort=False)[METRICS].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    return result.reset_index()


def collect_verified(tables, kept):
    source_files = set()
    stage_seed = {}; stage_condition = {}; stage_main = {}
    for name, folder in STAGES.items():
        stage_seed[name] = pd.read_csv(folder / "seed_metrics.csv")
        stage_condition[name] = pd.read_csv(folder / "per_condition_seed.csv")
        stage_main[name] = pd.read_csv(folder / "main_results.csv")
        source_files.update(folder / item for item in
                            ("FINAL_CHECK.json", "seed_metrics.csv", "main_results.csv", "per_condition_seed.csv", "paired_effects.csv", "protocol.json"))
    seed_rows = []; condition_rows = []; load_rows = []; checks = []
    for ds in DATASETS:
        table, ids = tables[ds], kept[ds]
        y = table.label.to_numpy(); g = table.condition_key.to_numpy()
        load_col = "load" if ds == "self462" else "appliance"
        for arm, stage in ARM_STAGE.items():
            for seed in range(3):
                filename = STAGES[stage] / f"{ds}_{arm}_s{seed}_OOF.npz"
                source_files.add(filename)
                z = np.load(filename)
                np.testing.assert_array_equal(z["indices"], ids, err_msg=str(filename))
                np.testing.assert_array_equal(z["labels"], y[ids], err_msg=str(filename))
                probability = z["probability"]
                assert probability.shape == ids.shape and np.isfinite(probability).all()
                assert (probability >= 0).all() and (probability <= 1).all()
                measured = cycle_metrics(y[ids], probability, g[ids])
                saved = stage_seed[stage]
                saved = saved[(saved.dataset == ds) & (saved.arm == arm) & (saved.seed == seed)]
                assert len(saved) == 1, (ds, arm, seed)
                assert_row(saved.iloc[0], measured, (ds, arm, seed))
                seed_rows.append(dict(dataset=ds, arm=arm, seed=seed, **measured))
                for condition in np.unique(g[ids]):
                    take = g[ids] == condition; ii = ids[take]
                    rr = cycle_metrics(y[ii], probability[take], g[ii])
                    source = stage_condition[stage]
                    source = source[(source.dataset == ds) & (source.arm == arm) &
                                    (source.seed == seed) & (source.condition == condition)]
                    assert len(source) == 1
                    assert_row(source.iloc[0], rr, (ds, arm, seed, condition))
                    condition_rows.append(dict(dataset=ds, arm=arm, seed=seed, condition=condition,
                                               load=str(table.iloc[ii[0]][load_col]), **rr))
                loads = table.iloc[ids][load_col].to_numpy()
                for load in np.unique(loads):
                    take = loads == load; ii = ids[take]
                    load_rows.append(dict(dataset=ds, arm=arm, seed=seed, load=load,
                                          **cycle_metrics(y[ii], probability[take], g[ii])))
                checks.append(dict(dataset=ds,arm=arm,seed=seed,N=len(ids),labels_equal=True,
                                   indices_equal=True,metrics_equal=True,condition_metrics_equal=True))
    seeds = pd.DataFrame(seed_rows); conditions = pd.DataFrame(condition_rows); loads = pd.DataFrame(load_rows)
    summary = aggregate(seeds, ["dataset", "arm"])
    for _, row in summary.iterrows():
        stage = ARM_STAGE[row.arm]
        original = stage_main[stage]
        original = original[(original.dataset == row.dataset) & (original.arm == row.arm)]
        assert len(original) == 1
        for metric in METRICS:
            for stat in ("mean", "std"):
                field = f"{metric}_{stat}"
                if field in original:
                    np.testing.assert_allclose(row[field], original.iloc[0][field], atol=2e-10, rtol=2e-10)
    # Reused controls are checked, rather than silently counted as new arms.
    for ds in DATASETS:
        for seed in range(3):
            a = STAGES["dynamic_state"] / f"{ds}_maxpool_dynamic_s{seed}_OOF.npz"
            b = STAGES["phase_hierarchy"] / f"{ds}_maxpool_dynamic_s{seed}_OOF.npz"
            c = STAGES["phase_hierarchy"] / f"{ds}_gain_phase_s{seed}_OOF.npz"
            d = STAGES["folded_calibration"] / f"{ds}_uncalibrated_s{seed}_OOF.npz"
            for left, right in ((a,b),(c,d)):
                source_files.update((left,right))
                x,z = np.load(left),np.load(right)
                np.testing.assert_array_equal(x["indices"],z["indices"])
                np.testing.assert_allclose(x["probability"],z["probability"],atol=2e-6,rtol=0)
                np.testing.assert_array_equal(x["probability"]>=.5,z["probability"]>=.5)
    assert len(seeds) == 54 and len(summary) == 18
    return seeds, conditions, loads, summary, pd.DataFrame(checks), source_files


def model_costs():
    sources = set(); frames = {}
    for stage in ("dynamic_state","phase_hierarchy"):
        path=STAGES[stage]/"model_cost.csv";sources.add(path);frames[stage]=pd.read_csv(path)
    rows=[]
    for arm,stage in ARM_STAGE.items():
        cost_stage="phase_hierarchy" if stage=="folded_calibration" else stage
        cost_arm="gain_phase" if stage=="folded_calibration" else arm
        c=frames[cost_stage];c=c[c.arm==cost_arm]
        assert len(c)==1
        row=c.iloc[0].to_dict();row.update(arm=arm,model=LABELS[arm],source=str(STAGES[cost_stage]/"model_cost.csv"),
                calibration_fitted_scalars=1 if arm=="folded_bias" else 2 if arm=="folded_affine" else 0,
                added_inference_parameters=0 if stage=="folded_calibration" else np.nan)
        row["matrix_MAC_M"]=float(row["matrix_MAC"])/1e6
        row["nominal_matrix_MFLOPs"]=float(row["nominal_matrix_FLOPs"])/1e6
        if stage=="folded_calibration":
            assert int(row["parameters"])==8249 and int(row["matrix_MAC"])==684960
        rows.append(row)
    return pd.DataFrame(rows),sources


def paired_effects(conditions, cohort):
    rows=[];sources=set()
    for stage,folder in STAGES.items():
        path=folder/"paired_effects.csv";sources.add(path)
        original=pd.read_csv(path)
        family_size=len(original)
        assert family_size==(4 if stage=="folded_calibration" else 6)
        for _,r in original.iterrows():
            ds=r.dataset
            # Stage3's uncalibrated scores are the verified phase2 gain_phase control.
            cand=r.candidate;ref="gain_phase" if r.comparator=="uncalibrated" else r.comparator
            wide=conditions[conditions.dataset==ds].groupby(["condition","arm"]).BAcc_condition.mean().unstack()
            diff=wide[cand]-wide[ref]
            np.testing.assert_allclose(diff.mean()*100,r.delta_pp,atol=2e-10,rtol=2e-10)
            row=r.to_dict();row.update(stage=stage,comparator=ref,groups=len(diff),
                      windows=int(cohort[cohort.dataset==ds].窗口数.iloc[0]),
                      Holm_family_size=family_size,source=str(path))
            row["positive_adjusted_support"]=bool(r.CI_low_pp>0 and r.p_holm<.05)
            assert 0<=r.p<=r.p_holm+1e-12<=1+1e-12
            rows.append(row)
    return pd.DataFrame(rows),sources


def strong_reference(table, keep):
    saved=pd.read_csv(REFER/"seed_metrics.csv")
    selected=saved[(saved.dataset=="zenodo5501527")&(saved.arm=="raw_sogi")]
    assert len(selected)==3
    rows=[];loads=[];sources={REFER/"seed_metrics.csv",REFER/"main_results.csv"}
    y=table.label.to_numpy();g=table.condition_key.to_numpy()
    for seed in range(3):
        path=REFER/f"zenodo5501527_raw_sogi_s{seed}_OOF.npz";sources.add(path);z=np.load(path)
        np.testing.assert_array_equal(z["indices"],keep);np.testing.assert_array_equal(z["labels"],y[keep])
        m=cycle_metrics(y[keep],z["probability"],g[keep]);assert_row(selected[selected.seed==seed].iloc[0],m,("strong_reference",seed))
        rows.append(dict(dataset="zenodo5501527",arm="raw_sogi_reference",seed=seed,**m))
        load=table.iloc[keep].appliance.to_numpy()
        for name in np.unique(load):
            take=load==name;ii=keep[take]
            loads.append(dict(dataset="zenodo5501527",arm="raw_sogi_reference",seed=seed,load=name,
                              **cycle_metrics(y[ii],z["probability"][take],g[ii])))
    op=REFER/"operations_gpu2.csv";sources.add(op);operations=pd.read_csv(op)
    operations=operations[(operations.dataset=="zenodo5501527")&(operations.arm=="raw_sogi")]
    assert operations.parameters.nunique()==operations.dense_weighted_MAC.nunique()==1
    result=aggregate(pd.DataFrame(rows),["dataset","arm"])
    result["parameters"]=int(operations.parameters.iloc[0])
    result["matrix_MAC_M"]=float(operations.dense_weighted_MAC.iloc[0])/1e6
    result["scope"]="同1357窗、SOGI参考和TTFS编码；不是同参考/同编码的网络单因素对照"
    return result,pd.DataFrame(rows),pd.DataFrame(loads),sources


def mean_sd(row,metric):
    factor=100 if metric in PERCENT else 1
    decimals=2 if metric in PERCENT else 3
    return f"{row[metric+'_mean']*factor:.{decimals}f}±{row[metric+'_std']*factor:.{decimals}f}"


def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join("| "+" | ".join(str(v).replace("|","／").replace("\n"," ") for v in row)+" |" for row in rows)


def reading_table(summary,cost,ds):
    headers=["方案","参数","模型MAC（M）"]+[TITLES[m]+"（%）" for m in KEY_METRICS]
    rows=[]
    for arm in ARM_STAGE:
        r=summary[(summary.dataset==ds)&(summary.arm==arm)].iloc[0];c=cost[cost.arm==arm].iloc[0]
        rows.append([LABELS[arm],f"{int(c.parameters):,}",f"{c.matrix_MAC_M:.6f}"]+[mean_sd(r,m) for m in KEY_METRICS])
    return table(headers,rows)


def write_excel(path,summary,seeds,loads,load_summary,cost,effects,cohort,strong,strong_loads,calibration):
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.utils import get_column_letter
    wb=Workbook();wb.remove(wb.active)
    def sheet(name,frame):
        ws=wb.create_sheet(name);ws.append(list(frame.columns))
        for row in frame.itertuples(index=False,name=None):
            ws.append([None if isinstance(v,float) and not np.isfinite(v) else v.item() if isinstance(v,np.generic) else v for v in row])
        ws.freeze_panes="C2";ws.auto_filter.ref=ws.dimensions;ws.sheet_view.zoomScale=85
        for cell in ws[1]:
            cell.font=Font(name="Arial",bold=True,color="FFFFFF",size=10);cell.fill=PatternFill("solid",fgColor="173B52")
            cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
        ws.row_dimensions[1].height=32
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font=Font(name="Arial",size=10);cell.alignment=Alignment(vertical="top",wrap_text=True)
                if cell.row%2==0:cell.fill=PatternFill("solid",fgColor="F0F5F8")
                if isinstance(cell.value,float):cell.number_format="0.000000" if "MAC" in str(ws.cell(1,cell.column).value) else "0.0000"
        for i,h in enumerate(frame.columns,1):ws.column_dimensions[get_column_letter(i)].width=40 if h in ("方案","model","load","condition","source","scope","成本范围") else 19
        return ws
    for ds,name in [("self462","自采462主指标"),("zenodo5501527","公共主指标")]:
        rr=[]
        for arm in ARM_STAGE:
            r=summary[(summary.dataset==ds)&(summary.arm==arm)].iloc[0];c=cost[cost.arm==arm].iloc[0]
            row={"方案":LABELS[arm],"参数":int(c.parameters),"模型MAC（M）":float(c.matrix_MAC_M)}
            row.update({TITLES[m]+"（%）":mean_sd(r,m) for m in KEY_METRICS});rr.append(row)
        ws=sheet(name,pd.DataFrame(rr))
        # Rank only comparable performance means, never identifiers or counts.
        selected=summary[summary.dataset==ds].set_index("arm")
        for col,metric in enumerate(KEY_METRICS,4):
            values=np.array([selected.loc[a,metric+"_mean"] for a in ARM_STAGE]);unique=np.unique(values)
            unique=np.sort(unique) if metric=="FPR" else np.sort(unique)[::-1]
            for i,value in enumerate(values,2):
                if np.isclose(value,unique[0],rtol=0,atol=1e-12):ws.cell(i,col).font=Font(name="Arial",size=10,bold=True,color="0000FF")
                elif len(unique)>1 and np.isclose(value,unique[1],rtol=0,atol=1e-12):ws.cell(i,col).font=Font(name="Arial",size=10,bold=False,color="FF0000")
    sheet("全指标均值与SD",summary)
    sheet("全部逐种子",seeds)
    sheet("逐负载均值与SD",load_summary)
    sheet("逐负载逐种子",loads)
    sheet("参数与计算量",cost)
    sheet("阶段主比较CI与Holm",effects)
    sheet("数据与划分",cohort)
    sheet("旧强SNN参照",strong)
    sheet("旧强参照逐负载",strong_loads)
    sheet("训练内校准系数",calibration)
    wb.save(path)


def generate(out:Path,check_only=False):
    receipts=require_completed()
    tables,kept,cohort=cohorts()
    seeds,conditions,loads,summary,checks,sources=collect_verified(tables,kept)
    sources.update({Path(__file__),SELF/"cycles.csv",SELF/"splits.json",PUBLIC/"cycles.csv",PUBLIC/"keep.npy"})
    sources.update(PUBLIC/f"roles_f{fold}.npz" for fold in range(5))
    cost,cost_sources=model_costs();sources|=cost_sources
    effects,effect_sources=paired_effects(conditions,cohort);sources|=effect_sources
    strong,strong_seeds,strong_loads,strong_sources=strong_reference(tables["zenodo5501527"],kept["zenodo5501527"]);sources|=strong_sources
    calibration_path=STAGES["folded_calibration"]/"calibration_coefficients.csv";sources.add(calibration_path)
    calibration=pd.read_csv(calibration_path)
    assert len(calibration)==54 and calibration.success.all()
    load_summary=aggregate(loads,["dataset","arm","load"])
    for c in ("N","groups"):
        dimension=loads.groupby(["dataset","arm","load"],sort=False)[c].first().reset_index()
        assert loads.groupby(["dataset","arm","load"])[c].nunique().max()==1
        load_summary=load_summary.merge(dimension,on=["dataset","arm","load"],validate="one_to_one")
    summary=summary.merge(cost,on="arm",how="left",validate="many_to_one")
    source_inventory=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(sources)]
    audit=dict(status="passed",three_FINAL_CHECK_passed=True,model_variants=9,datasets=2,seeds=3,
               OOF_checked=len(checks),OOF_indices_labels_metrics_equal=True,load_rows=len(loads),
               public_load_column="appliance",self_load_column="load",source_files=source_inventory,
               new_training=False,threshold=.5,independent_SOTA_claim=False,
               stages=receipts)
    if check_only:
        print(json.dumps({k:v for k,v in audit.items() if k not in ("stages","source_files")},ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    exports={"完整模型指标_均值与标准差.csv":summary,"全部模型_逐种子指标.csv":seeds,
             "逐工况完整指标_逐种子.csv":conditions,"逐负载完整指标_逐种子.csv":loads,
             "逐负载完整指标_均值与标准差.csv":load_summary,"模型参数与计算量.csv":cost,
             "阶段主比较_区间与Holm.csv":effects,"数据与分组说明.csv":cohort,
             "旧强SNN公共参照.csv":strong,"旧强SNN公共逐种子.csv":strong_seeds,
             "旧强SNN公共逐负载.csv":strong_loads,"训练内校准系数.csv":calibration,
             "报告OOF逐项核验.csv":checks}
    for name,frame in exports.items():frame.to_csv(out/name,index=False,encoding="utf-8-sig")
    report=["# 自有CNN–SNN三阶段升级：完整462条与公共数据验证汇总", "",
        "本报告合并三阶段全部九种预先指定方案，不选取最佳种子。每个方案在两套数据分别训练，以三个完整OOF种子的均值±标准差报告。所有三阶段FINAL_CHECK均已通过；本脚本重新核对54份OOF的索引、标签、全指标及逐工况结果。", "",
        "## 数据与评价口径", "",
        table(["数据集","录波数","工况／电器组","窗口","正常／电弧","外折","采样率"],
              [[r.数据集,r.录波数,r.工况或电器组数,r.窗口数,f"{r.正常窗口}／{r.电弧窗口}",r.外折数,r.原生采样率] for _,r in cohort.iterrows()]), "",
        "自采固定NET9770_114 CH0、5 m总路单电流；同工况重复录波和全部周波同折。公共每种电器对应一份连续录波，同文件全部周波同折。两库训练、标准化和评价分开。原生2 MHz/1 MHz高频信息已进入固定五通道九周波表征；本轮复用可追溯缓存，不复制旧自采队列分数。公共保留不足九周波历史的44个窗口，缺失由掩码表示，不为改善成绩剔除。", "",
        "BAcc为工况／电器组等权；Acc、Precision、Recall、FPR、F1、AP、ROC-AUC为各完整OOF窗口汇总。FPR=FP/(FP+TN)，AP为average precision，不写成梯形积分PR-AUC。±是三种子标准差，不是折间独立重复的置信区间。阈值始终0.5、最终epoch，不根据外测挑检查点或阈值。", "",
        "## 九种方案与设计差异", "",
        table(["方案","作用"],[
            [LABELS["base"],"原7097参数、平均池化、末周波状态读出"],
            [LABELS["dynamic"],"原信息计算36维周期动态摘要，加入同一个分类器，不输入外部模型分数"],
            [LABELS["capacity"],"只扩分类器容量，检查参数增加能否解释收益"],
            [LABELS["maxpool_dynamic"],"两次平均池化改最大池化，保留局部响应，再结合动态摘要"],
            [LABELS["gain_phase"],"训练内共享九周波的传感增益0.708–1.413与统一相位±2箱扰动；验证不增广"],
            [LABELS["hierarchy_small"],"在池化前维护多层相位LIF状态的小模型"],
            [LABELS["hierarchy_wide"],"相同早期相位LIF思路的较宽模型"],
            [LABELS["folded_bias"],"训练工况内部二折OOF拟合一个全局偏置，折入现有末层"],
            [LABELS["folded_affine"],"训练工况内部二折OOF拟合正尺度与偏置，折入现有末层"],
        ]), "",
        "增广遵守物理关系：低频电流乘g，两带log10功率加2log10(g)，分布差分不乘增益；相同样本九周波共用相位滚动。卷积和脉冲状态与动态信息经同一个最终分类器判别，无蒸馏、外部基准打分或推理纠正分支。校准属于标准分数标定，不作为新的深度架构贡献。", "",
        "## 完整自采462条结果", "",reading_table(summary,cost,"self462"),"",
        "## 公共数据结果", "",reading_table(summary,cost,"zenodo5501527"),"",
        "## 当前最优观测与跨库取舍", ""]
    for ds in DATASETS:
        dd=summary[summary.dataset==ds]
        best=dd.loc[dd.BAcc_condition_mean.idxmax()];acc=dd.loc[dd.Accuracy_mean.idxmax()];fpr=dd.loc[dd.FPR_mean.idxmin()]
        report.append(f"{DS_LABELS[ds]}：本轮BAcc均值最高的是{LABELS[best.arm]}（{mean_sd(best,'BAcc_condition')}%）；Acc均值最高的是{LABELS[acc.arm]}（{mean_sd(acc,'Accuracy')}%）；FPR均值最低的是{LABELS[fpr.arm]}（{mean_sd(fpr,'FPR')}%）。这里的‘最高／最低’只描述这组固定开发比较，不代表独立确认或全领域最优。")
        report.append("")
    gain_notes=[]
    for ds in DATASETS:
        dd=summary[summary.dataset==ds].set_index("arm");r=dd.loc["gain_phase"];base=dd.loc["base"]
        gain_notes.append(f"{DS_LABELS[ds]}的Acc为{mean_sd(r,'Accuracy')}%、BAcc为{mean_sd(r,'BAcc_condition')}%、FPR为{mean_sd(r,'FPR')}%；相对原模型Acc均值{100*(r.Accuracy_mean-base.Accuracy_mean):+.3f} pp、BAcc均值{100*(r.BAcc_condition_mean-base.BAcc_condition_mean):+.3f} pp")
    report += ["结构与训练路线层面，当前继续保留最大池化动态＋物理一致增广为跨库一致改进候选。"+"；".join(gain_notes)+"。折入校准作为该候选的单独标定验证，并不改称新的网络结构。",""]
    rejected=[]
    public_rows=summary[summary.dataset=="zenodo5501527"].set_index("arm")
    for arm in ("hierarchy_small","hierarchy_wide"):
        r=public_rows.loc[arm]
        if r.BAcc_condition_mean<public_rows.loc["base","BAcc_condition_mean"]:
            rejected.append(f"{LABELS[arm]}公共BAcc为{mean_sd(r,'BAcc_condition')}%、Acc为{mean_sd(r,'Accuracy')}%")
    if rejected:
        report += ["本轮早期相位LIF存在负向结果："+"；".join(rejected)+"。这些候选保留完整结果，不因脉冲状态更早或模型更宽而默认采用。",""]
    changes=[]
    for arm in list(ARM_STAGE)[1:]:
        delta=[]
        for ds in DATASETS:
            dd=summary[summary.dataset==ds].set_index("arm");delta.append(float((dd.loc[arm,"BAcc_condition_mean"]-dd.loc["base","BAcc_condition_mean"])*100))
        changes.append([LABELS[arm],f"{delta[0]:+.3f}",f"{delta[1]:+.3f}","两库均值均正向" if min(delta)>0 else "存在跨库取舍或未改善"])
    report += [table(["方案相对原模型","自采BAcc增量pp","公共BAcc增量pp","方向"],changes),"",
               "均值正向与统计支持分别报告。不能把每个数据集不同方案的最优值拼成同一模型；若某候选只改善一库，保留其负向结果和逐负载情况。", "",
               "## 各阶段主比较、条件配对区间与多重校正", ""]
    effect_rows=[]
    for _,r in effects.iterrows():
        effect_rows.append([r.stage,DS_LABELS[r.dataset],LABELS[r.candidate]+" − "+LABELS[r.comparator],
                            f"{r.delta_pp:+.3f}",f"[{r.CI_low_pp:+.3f}, {r.CI_high_pp:+.3f}]",int(r.groups),
                            f"{r.p_holm:.4g}",int(r.Holm_family_size),"是" if r.positive_adjusted_support else "否"])
    report += [table(["阶段","数据","比较","ΔBAcc pp","95%工况配对区间","组数","Holm p","校正族大小","校正后正向支持"],effect_rows),"",
        "区间以工况／电器录波组重采样，先对三种子求组内平均，再进行10000次组配对bootstrap。p为组差值正负的双侧符号检验，按各阶段预设6／6／4项主比较分别做Holm校正；这些分阶段校正不是跨全部开发历史的统一校正。有限13公共电器组会使检验较保守，窗口数量不能当作独立样本数增加显著性。", "",
        "## 参数与计算量", "",
        table(["方案","网络参数","矩阵MAC（M）","名义矩阵MFLOPs","LIF标量更新","训练另拟合校准标量"],
              [[LABELS[r.arm],f"{int(r.parameters):,}",f"{r.matrix_MAC_M:.6f}",f"{r.nominal_matrix_MFLOPs:.6f}",r.get("LIF_scalar_updates","另计"),int(r.calibration_fitted_scalars)] for _,r in cost.iterrows()]),"",
        "MAC只计卷积、线性及投影矩阵，名义按1 MAC=2 FLOPs换算。原生滤波、RCR拟合、动态摘要的softmax/log/归约、归一化、池化、激活、内存搬运和LIF标量操作未统一并入该矩阵计数，不能据此宣称整机能耗。校准将W改为aW、偏置改为ab+c，网络推理仍为8249参数、684960矩阵MAC，没有新增推理层；训练额外拟合的1／2个标量仍明确报告。", "",
        "## 同公共队列的旧强SNN参照", "",
        table(["方案","BAcc%","Acc%","Recall%","FPR%","F1%","AP%","ROC-AUC%","参数","矩阵MAC（M）"],
              [[LABELS["raw_sogi_reference"]]+[mean_sd(strong.iloc[0],m) for m in ["BAcc_condition","Accuracy","Recall","FPR","F1","AP","ROC_AUC"]]+[int(strong.iloc[0].parameters),f"{strong.iloc[0].matrix_MAC_M:.6f}"]]),"",
        "该参照与本轮公共OOF的1357窗口索引、标签一致，但使用SOGI参考和TTFS编码。它提供已达到性能的上下文，不是对RCR、编码或早期LIF的单因素验证。97.22%是三种子Acc均值，不是文件级指标，也不是最佳种子；其BAcc、Recall和误报同时保留。", "",
        "## 逐负载结果与审阅入口", "",
        "逐负载指标从各模型的完整OOF重新计算；自采用load列，公共用appliance列，全部四类自采负载和13类公共电器均保留。文件包含每个方案／种子的N、工况数、Precision、Recall、FPR、F1、AP、ROC-AUC和混淆计数；均值表与逐种子表分别提供，避免用最佳种子掩盖电脑、水泵或电视等负载上的失败。", "",
        link(out/"逐负载完整指标_均值与标准差.csv")+"；"+link(out/"逐负载完整指标_逐种子.csv")+"；"+link(out/"全部模型_逐种子指标.csv"),"",
        "## 证据范围与后续比较", "",
        "本轮沿用已被多轮研究使用的公开分组，是版本化开发证据。九种方案和全部不利结果均报告，不能把目前最高均值称为独立确认的SOTA。完整462条的三基准双支路尚需同队列验证；公共历史1361窗的纠正结果不与当前1357窗直接拼成严格排名。不同测量输入、参考方法、历史长度及训练协议须在最终论文主表统一或明确分层。", "",
        "校准的内层模型只见外训练工况的一半：公共约5–6个电器组，自采每半58–59工况。其OOF分数尺度向完整外训练模型迁移存在差异；凸优化成功或折入代数等价不保证外测指标提高。正尺度标定维持同一模型内部排序，主要作用于全局分数位置与判决，并不能修复已经反向的类间排序。", "",
        "## 原始文件与验证收据", ""]
    for stage,folder in STAGES.items():
        report += [f"**{stage}**："+"；".join(link(folder/n) for n in ["FINAL_CHECK.json","main_results.csv","seed_metrics.csv","per_condition_seed.csv","paired_effects.csv","protocol.json"]),""]
    report += ["公共强参照："+link(REFER/"main_results.csv")+"；"+link(REFER/"seed_metrics.csv"),"",
               "所有来源的SHA-256和本报告54份OOF逐项核对结果保存在同目录REPORT_CHECK.json与报告OOF逐项核验.csv。报告生成只使用CPU，不训练模型、不改源数据、旧检查点或原结果。",""]
    (out/"自有模型升级_三阶段完整验证报告.md").write_text("\n".join(report),encoding="utf-8")
    write_excel(out/"自有模型升级_双库完整指标.xlsx",summary,seeds,loads,load_summary,cost,effects,cohort,strong,strong_loads,calibration)
    (out/"REPORT_CHECK.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","output":str(out),"OOF_checked":len(checks),"model_variants":9,"datasets":2},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--check-only",action="store_true",help="Verify completed sources and OOF without creating output files")
    args=parser.parse_args()
    generate(args.output,args.check_only)
