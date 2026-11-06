"""Validate physical GPU assignments without initializing CUDA."""

import os
from pathlib import Path

from .configuration import read_yaml


def validate_runtime(config):
    allowed = config.get("allowed_physical_gpus", [])
    forbidden = config.get("forbidden_physical_gpus", [0])
    if sorted(allowed) != [1, 2, 3] or 0 not in forbidden or set(allowed) & set(forbidden):
        raise ValueError("Training policy requires physical GPUs1,2,3 and forbids GPU0")
    return config


def device_environment(physical_gpu=None, *, config=None):
    policy = validate_runtime(config or {"allowed_physical_gpus": [1, 2, 3], "forbidden_physical_gpus": [0]})
    if physical_gpu is None:
        return {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if physical_gpu not in policy["allowed_physical_gpus"]:
        raise ValueError(f"Forbidden physical GPU: {physical_gpu}")
    # Launch a subprocess with this environment; never remap an already initialized process.
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(physical_gpu)}


def load_runtime(path: str | Path):
    return validate_runtime(read_yaml(path))

