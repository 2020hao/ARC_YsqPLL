"""Author HYDRA transform with the arc experiment's audited hyperparameters."""

import numpy as np
import torch
from arc_model.models.standalone.hydra import Hydra


def transform_windows(waves, seed: int, batch_size: int = 128):
    waves = np.asarray(waves, dtype=np.float32)
    if waves.ndim != 2 or waves.shape[1] != 1000 or len(waves) == 0:
        raise ValueError("Expected nonempty [windows, 1000] 50 kHz current input")
    transform = Hydra(1000, k=8, g=4, kernel_length=9, stride=2, divisor=1, seed=seed)
    with torch.no_grad():
        return transform.batch(torch.from_numpy(waves[:, None]), batch_size=batch_size).numpy()
