"""Small, auditable subprocess launcher for registered experiments."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .catalog import ExperimentPlan


def validate_plan(plan: ExperimentPlan, root: Path) -> list[str]:
    errors: list[str] = []
    for relative in [plan.runner, plan.protocol]:
        if not (root / relative).is_file():
            errors.append(f"missing file: {relative}")
    if plan.summarizer and not (root / plan.summarizer).is_file():
        errors.append(f"missing file: {plan.summarizer}")
    if set(plan.datasets) != {"new5m", "zenodo5501527"}:
        errors.append("current studies must report new5m and zenodo5501527 separately")
    return errors


def run_partition(
    plan: ExperimentPlan, root: Path, partition: int, partitions: int, gpu: int
) -> None:
    if gpu not in (1, 2, 3):
        raise ValueError("Training must use physical GPU1,2,3; GPU0 is reserved")
    if partitions < 1 or not 0 <= partition < partitions:
        raise ValueError("Invalid worker partition")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        sys.executable,
        str(root / plan.runner),
        "--partition",
        str(partition),
        "--partitions",
        str(partitions),
        "--gpu",
        "0",
    ]
    subprocess.run(command, cwd=root, env=environment, check=True)


def summarize(plan: ExperimentPlan, root: Path) -> None:
    if plan.summarizer:
        subprocess.run([sys.executable, str(root / plan.summarizer)], cwd=root, check=True)
