"""L1 hinge classifier; Python SGD adaptation, not MATLAB SpaRSA."""

import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning


def train(x, y, alpha, seed):
    # Same L1 + hinge model family/objective; different optimizer from MATLAB.
    clf = SGDClassifier(
        loss="hinge",
        penalty="l1",
        alpha=alpha,
        fit_intercept=True,
        max_iter=3000,
        tol=1e-4,
        n_iter_no_change=15,
        random_state=seed,
        class_weight="balanced",
        shuffle=True,
        average=False,
    )
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    return clf, any(issubclass(w.category, ConvergenceWarning) for w in ws)
