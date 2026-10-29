"""Aggregate disjoint outer folds; seeds stay separate and partial coverage is explicit."""

from pathlib import Path
import json
from .aggregation import evaluate_records


def summarize(directory, dataset):
    groups = {}
    for path in sorted(Path(directory).glob("*/fold*.json")):
        meta = json.loads(path.read_text())
        saved_split = meta.get('split', meta)
        if saved_split.get("protocol") != "condition_grouped_cv5":
            continue
        key = (meta["model"], meta["variant"], meta["mode"], meta["seed"])
        group = groups.setdefault(key, dict(records={}, folds=[], configs=[]))
        fold = meta["fold_index"]
        split = dataset.splits[fold]
        if set(saved_split["test_records"]) != set(split["test_records"]):
            raise ValueError(f"Unexpected test records in {path}")
        if fold in group["folds"]:
            raise ValueError(f"Duplicate fold: {path}")
        group["folds"].append(fold)
        group["configs"].append(meta.get("config", {}).get("training"))
        for row in meta["predictions"]:
            rid = row["record_id"]
            if rid in group["records"] or rid not in split["test_records"]:
                raise ValueError(f"Duplicate/non-test record: {path}, {rid}")
            if row['label'] != dataset.ledger[rid]['label']:
                raise ValueError(f'Label mismatch: {path}, {rid}')
            group["records"][rid] = row
    results = []
    for (model, variant, mode, seed), group in groups.items():
        if any(c != group["configs"][0] for c in group["configs"]):
            raise ValueError("Cannot combine folds with different training settings")
        rows = list(group["records"].values())
        result = dict(
            model=model,
            variant=variant,
            mode=mode,
            seed=seed,
            folds=sorted(group["folds"]),
            record_count=len(rows),
            complete_main_cohort=len(rows) == len(dataset.records),
            metrics=evaluate_records(rows, model),
        )
        conditions = {}
        for row in rows:
            conditions.setdefault(dataset.ledger[row['record_id']]['condition_key'], []).append(row)
        result['condition_equal_bacc'] = sum(
            evaluate_records(part, model)['record']['bacc']
            for part in conditions.values()
        ) / len(conditions)
        results.append(result)
    if not results:
        raise ValueError(f"No main-cohort fold results in {directory}")
    return results
