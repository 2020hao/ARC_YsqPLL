"""Record/first-window metrics with the original decision thresholds."""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


def metrics(y, s, threshold):
    pred = np.asarray(s) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        bacc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        auc=roc_auc_score(y, s) if len(set(y)) == 2 else np.nan,
        mcc=matthews_corrcoef(y, pred),
        recall=tp / max(1, tp + fn),
        fpr=fp / max(1, fp + tn),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        n=len(y),
    )
