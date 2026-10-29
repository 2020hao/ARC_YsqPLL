"""Train one HYDRA outer fold with grouped inner regularization selection."""

import time
import numpy as np
from arc_model.models.hydra.preprocessing import Features
from arc_model.models.hydra.selection import select_alpha
from arc_model.models.hydra.classifier import train as train_classifier
from arc_model.evaluation.aggregation import record_results


def train_fold(config, dataset, fold, stem, device=None):
    start = time.perf_counter()
    tr, te = dataset.indices(fold)
    physical = dataset.physical(config.variant, config.mode)
    features = dataset.array(f"hydra_features_seed{config.seed}")[:, config.variant]
    method = {"base": 0, "raw": 4, "enhanced": 2, "enhanced_low": 2}[config.mode]
    records = np.array(dataset.splits[fold]["train_records"])
    alpha, val = select_alpha(
        features,
        physical,
        dataset.labels,
        dataset.record_ids,
        dataset.ledger,
        records,
        method,
        config.seed,
        group="condition_key",
    )
    transform = Features(features, physical, dataset.labels, method).fit(tr)
    classifier, warning = train_classifier(
        transform.transform(tr), dataset.labels[tr], alpha, config.seed
    )
    scores = classifier.decision_function(transform.transform(te))
    state = dict(
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        scaler_mu=transform.sc.mu.numpy(),
        scaler_sigma=transform.sc.sigma.numpy(),
        scaler_epsilon=transform.sc.epsilon.numpy(),
    )
    if config.mode != "base":
        state.update(
            feature_med=transform.med,
            feature_mad=transform.mad,
            feature_mean=transform.mu,
            feature_std=transform.sd,
        )
        if hasattr(dataset, 'feature_indices'):
            state['feature_indices'] = np.asarray(dataset.feature_indices, dtype=np.int64)
    np.savez_compressed(stem.with_suffix(".npz"), **state)
    np.savez_compressed(
        str(stem) + "_windows.npz",
        indices=te,
        record_ids=dataset.record_ids[te],
        labels=dataset.labels[te],
        scores=scores,
    )
    return dict(
        alpha=alpha,
        inner_bacc=val,
        convergence_warning=warning,
        iterations=int(classifier.n_iter_),
        parameters=int(np.count_nonzero(classifier.coef_)),
        elapsed=time.perf_counter() - start,
        predictions=record_results(scores, te, dataset, "HYDRA"),
    )
