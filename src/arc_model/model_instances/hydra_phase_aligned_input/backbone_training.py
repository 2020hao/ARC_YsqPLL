"""Independent author-FD SVM fit on prepared author features."""

from pathlib import Path
import json
import numpy as np
from .network import Model
from .configuration import load_config


def fit(features, labels, train_indices, test_indices, output, *, groups):
    out = Path(output).resolve()
    package = Path(__file__).resolve().parent
    if out == package or package in out.parents:
        raise ValueError("Never write into a source/frozen package")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    tr = np.asarray(train_indices, dtype=int)
    te = np.asarray(test_indices, dtype=int)
    if np.intersect1d(tr, te).size:
        raise ValueError("Train/test overlap")
    groups = np.asarray(groups)
    if set(groups[tr]) & set(groups[te]):
        raise ValueError("Main baseline training requires condition/record-disjoint groups")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    import yaml

    (out / "resolved_model.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    )
    model = Model(**cfg["model_kwargs"])
    diagnostics = model.fit(
        features, labels, tr, lambda_l1=cfg["model_training_param"]["lambda_l1"]
    )
    np.savez(out / "model.npz", **model.state)
    np.savez(out / "indices.npz", train=tr, test=te)
    np.savez(
        out / "predictions.npz",
        indices=te,
        labels=np.asarray(labels)[te],
        score=model.decision_function(np.asarray(features)[te]),
    )
    (out / "run.json").write_text(
        json.dumps(
            dict(performance_status="new_run_not_the_frozen_scores", solver=diagnostics), indent=2
        )
    )
    return model
