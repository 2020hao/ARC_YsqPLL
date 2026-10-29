"""Determinism and shared neural batching, independent of model architecture."""

import random
import numpy as np
import torch


def configure_runtime(device, threads=2):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        torch.cuda.set_device(dev)
    return dev


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensors(arrays, device):
    return tuple(torch.tensor(a, device=device) if a is not None else None for a in arrays)


@torch.no_grad()
def predict_probabilities(model, x, spectral, features, indices, batch_size=64):
    model.eval()
    if len(indices) == 0:
        raise ValueError("Cannot predict an empty window selection")
    return (
        torch.cat(
            [
                model(
                    x[b],
                    spectral[b] if spectral is not None else None,
                    features[b] if features is not None else None,
                ).softmax(-1)[:, 1]
                for b in indices.split(batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
