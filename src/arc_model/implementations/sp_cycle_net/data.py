"""Independent SP-CycleNet normalization, history loader and loss weights."""
import numpy as np


def training_statistics(profiles, histories, train_indices):
    # Only the current cycles of outer-training anchors fit the transform.
    current = profiles[histories[np.asarray(train_indices), -1]]
    mean = current.mean(axis=(0, 2), dtype=np.float64).astype('float32')
    std = np.maximum(current.std(axis=(0, 2), dtype=np.float64), 1e-6).astype('float32')
    return mean, std


def condition_class_weights(labels, groups, indices):
    indices = np.asarray(indices)
    result = np.zeros(len(labels), dtype='float32')
    for group in np.unique(groups[indices]):
        for label in (0, 1):
            subset = indices[(groups[indices] == group) & (labels[indices] == label)]
            if not len(subset):
                raise ValueError('Each training condition must contain both classes')
            result[subset] = 1. / len(subset)
    result[indices] /= result[indices].mean()
    return result
