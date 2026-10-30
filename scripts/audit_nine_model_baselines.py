"""Replay the frozen70-fold registry through the new independent packages."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import _bootstrap as boot
from arc_model.implementations.registry import create_model

ROOT = boot.ROOT
SOURCE = ROOT / "outputs/grouped_baseline_standard_20261029"
OUT = ROOT / "outputs/nine_model_refactor_20261030"
NAMES = {
    "ArcFENet_reconstructed": "ArcFENet_Baseline",
    "HYDRA_paper": "HYDRA_Baseline",
    "ArcFusionNet_current": "ArcFusionNet_Current_Baseline",
}


def main(part):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(part + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    rows = []
    entries = json.loads((SOURCE / "baseline_registry.json").read_text())["entries"]
    for i, e in enumerate(entries):
        if i % 3 != part:
            continue
        ds, kind = e["dataset"], e["model"]
        folder = Path(e["folder"])
        z = np.load(folder / "predictions.npz")
        ids = z["indices"]
        model = create_model(NAMES[kind], checkpoint=e["weights_path"])
        if kind == "HYDRA_paper":
            features = np.load(
                ROOT / "outputs/paper_baseline_completion_20261029/data" / ds / "author_hydra.npy",
                mmap_mode="r",
            )
            score = model.decision_function(features[ids])
            expected = z["score"]
            threshold = 0
        else:
            model.cuda().eval()
            d = ROOT / "outputs/rcr_three_backbones_20261028/data" / ds / "fixed20"
            full_ids = ids
            target = z["score"] if kind == "ArcFENet_reconstructed" else z["probability"]
            state = None
            if kind == "ArcFENet_reconstructed":
                x = np.load(d / "rgb.npy", mmap_mode="r")
                batches = np.array_split(full_ids, np.ceil(len(full_ids) / 128).astype(int))
            else:
                state = np.load(folder / "preprocessing.npz")
                x = np.load(d / "temporal.npy", mmap_mode="r")
                sp = np.load(d / "spectrum.npy", mmap_mode="r")
                receipt = json.loads((folder / "COMPLETE.json").read_text())
                if receipt["reuse"]:
                    old = np.load(
                        ROOT
                        / "outputs/rcr_three_backbones_20261028/runs"
                        / ds
                        / "fixed20/ArcFusionNet"
                        / f"f{e['fold']}_s{e['seed']}/outer/M0/predictions.npz"
                    )
                    full_ids = old["indices"]
                    target = old["probability"]
                batches = [full_ids[j : j + 64] for j in range(0, len(full_ids), 64)]
            parts = []
            with torch.no_grad():
                for ix in batches:
                    if state is None:
                        xx = torch.tensor(np.array(x[ix]), device="cuda")
                        ss = None
                    else:
                        xx = torch.tensor(
                            (x[ix] - state["temporal_mean"]) / state["temporal_std"], device="cuda"
                        )
                        ss = torch.tensor(
                            (
                                (sp[ix][:, state["harmonic_indices"]] - state["spectral_mean"])
                                / state["spectral_std"]
                            )[:, None],
                            device="cuda",
                        )
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=state is not None):
                        value = model(xx, ss)
                    parts.append(value.float().softmax(-1)[:, 1].cpu().numpy())
            value = np.concatenate(parts)
            lookup = {int(v): j for j, v in enumerate(full_ids)}
            view = np.array([lookup[int(v)] for v in ids])
            score = value[view]
            expected = target[view]
            threshold = 0.5
            del model
            torch.cuda.empty_cache()
        err = float(abs(score - expected).max())
        changes = int(((score >= threshold) != (expected >= threshold)).sum())
        assert err < 5e-5 and changes == 0, (ds, kind, e["fold"], e["seed"], err, changes)
        rows.append(
            dict(
                dataset=ds,
                model=kind,
                fold=e["fold"],
                seed=e["seed"],
                score_max_error=err,
                decision_changes=changes,
                N=len(ids),
            )
        )
    (OUT / f"baseline_replay_part{part}.json").write_text(json.dumps(rows, indent=2))
    print("PASS", part, len(rows), max(r["score_max_error"] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--partition", type=int, required=True)
    main(p.parse_args().partition)
