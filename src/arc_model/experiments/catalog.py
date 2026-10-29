"""Typed experiment catalog inspired by the PILT project layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    runner: str
    summarizer: str | None
    protocol: str
    result_dir: str
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    folds: int
    seeds: tuple[int, ...]

    @property
    def fit_count(self) -> int:
        return len(self.datasets) * len(self.models) * self.folds * len(self.seeds)


@dataclass(frozen=True)
class ExperimentCatalog:
    project_root: Path
    experiments: dict[str, ExperimentPlan]

    def get(self, name: str) -> ExperimentPlan:
        try:
            return self.experiments[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.experiments))
            raise KeyError(f"unknown experiment {name!r}; choose one of: {choices}") from error


def load_catalog(path: str | Path, project_root: str | Path) -> ExperimentCatalog:
    project_root = Path(project_root).resolve()
    raw = yaml.safe_load(Path(path).read_text())
    experiments: dict[str, ExperimentPlan] = {}
    for name, entry in raw["experiments"].items():
        experiments[name] = ExperimentPlan(
            name=name,
            runner=entry["runner"],
            summarizer=entry.get("summarizer"),
            protocol=entry["protocol"],
            result_dir=entry["result_dir"],
            datasets=tuple(entry["datasets"]),
            models=tuple(entry["models"]),
            folds=int(entry.get("folds", 5)),
            seeds=tuple(int(seed) for seed in entry.get("seeds", range(5))),
        )
    return ExperimentCatalog(project_root=project_root, experiments=experiments)
