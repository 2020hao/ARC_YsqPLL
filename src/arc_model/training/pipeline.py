"""Outer-fold orchestration; completion JSON is the last artifact written."""

from dataclasses import asdict
from arc_model.data.dataset import PreparedDataset
from arc_model.utils.io import save_json
from arc_model.evaluation.aggregation import evaluate_records
from .common import configure_runtime
from . import neural, hydra


def run(config, fold=None):
    config.validate()
    dataset = PreparedDataset(config.path("prepared"), config.path("manifests"))
    folds = list(range(len(dataset.splits))) if fold is None else [fold]
    for number in folds:
        dataset.indices(number)
    output = (
        config.path("output") / f"{config.model}_v{config.variant}_{config.mode}_s{config.seed}"
    )
    # Do not silently overwrite existing results or treat a smoke run as a full run.
    for number in folds:
        if (output / f"fold{number}.json").exists():
            raise FileExistsError(f"{output}/fold{number}.json already exists; use a new --output")
    device = configure_runtime(config.device, config.threads)
    output.mkdir(parents=True, exist_ok=True)
    trainer = hydra if config.model == "HYDRA" else neural
    for number in folds:
        result = trainer.train_fold(config, dataset, number, output / f"fold{number}", device)
        split = dataset.splits[number]
        meta = dict(
            model=config.model,
            variant=config.variant,
            mode=config.mode,
            seed=config.seed,
            fold_index=number,
            **split,
            **result,
        )
        meta["config"] = asdict(config)
        meta["metrics"] = evaluate_records(result["predictions"], config.model)
        save_json(output / f"fold{number}.json", meta)
        print(f"DONE {config.model} fold={number} -> {output}", flush=True)
    return output
