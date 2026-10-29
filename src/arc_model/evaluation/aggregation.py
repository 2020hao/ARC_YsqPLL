"""Aggregate window decisions without mixing record means and first-window metrics."""

import numpy as np
from .metrics import metrics


def record_results(scores, indices, dataset, model):
    key = "score" if model == "HYDRA" else "probability"
    rids = dataset.record_ids[indices]
    result = []
    for rid in np.unique(rids):
        record = dataset.ledger[int(rid)]
        values = scores[rids == rid]
        result.append(
            dict(
                record_id=int(rid),
                label=record["label"],
                appliance=record["appliance"],
                background=record["background"],
                **{key: float(values.mean()), "first_" + key: float(values[0])},
                windows=len(values),
            )
        )
    return result


def evaluate_records(rows, model):
    key, threshold = ("score", 0.0) if model == "HYDRA" else ("probability", 0.5)
    labels = [r["label"] for r in rows]
    return {
        resolution: metrics(labels, [r[column] for r in rows], threshold)
        for resolution, column in [("record", key), ("first_window", "first_" + key)]
    }
