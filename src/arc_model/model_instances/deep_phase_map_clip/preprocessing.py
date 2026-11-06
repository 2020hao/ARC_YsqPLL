"""Independent input transformation and training weights for the direct SNN."""
import numpy as np


def training_statistics(inputs, indices):
    current = inputs[np.asarray(indices), -1]
    mean = current.mean((0, 2), dtype=np.float64).astype('float32')
    std = np.maximum(current.std((0, 2), dtype=np.float64), 1e-6).astype('float32')
    return mean, std


def condition_class_weights(labels, groups, indices):
    indices = np.asarray(indices)
    result = np.zeros(len(labels), dtype='float32')
    for group in np.unique(groups[indices]):
        for label in (0, 1):
            ids = indices[(groups[indices] == group) & (labels[indices] == label)]
            if not len(ids):
                raise ValueError('Every training condition must contain both classes')
            result[ids] = 1. / len(ids)
    result[indices] /= result[indices].mean()
    return result
