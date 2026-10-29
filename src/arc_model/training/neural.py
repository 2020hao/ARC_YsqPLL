"""Train one neural outer fold; architecture and fitting statistics live elsewhere."""

import time
import numpy as np
import torch
from torch.nn import functional as F
from arc_model.models.factory import make_model
from arc_model.evaluation.aggregation import record_results
from .preprocessing import fit_state, apply_neural_state
from .common import seed_everything, to_tensors, predict_probabilities


def make_optimizer(model, settings):
    kind = settings["optimizer"]
    if kind not in ("adam", "adamw"):
        raise ValueError(f"Unknown optimizer {kind}")
    cls = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    optimizer = cls(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, settings["epochs"], eta_min=settings.get("min_learning_rate", 1e-6)
        )
        if settings.get("cosine_schedule")
        else None
    )
    return optimizer, scheduler


def train_epochs(model, tensors, labels, train, settings, runseed, device):
    x, spectral, features = tensors
    optimizer, scheduler = make_optimizer(model, settings)
    train = torch.tensor(train, device=device)
    generator = torch.Generator(device=device).manual_seed(runseed + 90000)
    history = []
    for epoch in range(settings["epochs"]):
        model.train()
        order = train[torch.randperm(len(train), device=device, generator=generator)]
        total, correct = 0.0, 0
        for ids in order.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                x[ids],
                spectral[ids] if spectral is not None else None,
                features[ids] if features is not None else None,
            )
            loss = F.cross_entropy(logits, labels[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            correct += int((logits.argmax(-1) == labels[ids]).sum())
        history.append(
            dict(
                epoch=epoch + 1,
                loss=total / len(train),
                accuracy=correct / len(train),
                lr=optimizer.param_groups[0]["lr"],
            )
        )
        if scheduler:
            scheduler.step()
        print(f"epoch={epoch + 1} loss={history[-1]['loss']:.6f}", flush=True)
    return history


def train_fold(config, dataset, fold, stem, device):
    train, test = dataset.indices(fold)
    runseed = config.seed * 1000 + fold
    seed_everything(runseed)
    state = fit_state(dataset, config, train, runseed)
    tensors = to_tensors(
        apply_neural_state(dataset, config.model, config.variant, config.mode, state), device
    )
    labels = torch.tensor(dataset.labels, device=device)
    model = make_model(config.model, 0 if config.mode == "base" else int(tensors[2].shape[-1])).to(device)
    start = time.perf_counter()
    history = train_epochs(model, tensors, labels, train, config.training, runseed, device)
    prob = predict_probabilities(model, *tensors, torch.tensor(test, device=device))
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    np.savez_compressed(str(stem) + "_preprocessing.npz", **state)
    np.savez_compressed(
        str(stem) + "_windows.npz",
        indices=test,
        record_ids=dataset.record_ids[test],
        labels=dataset.labels[test],
        probabilities=prob,
    )
    return dict(
        parameters=sum(p.numel() for p in model.parameters()),
        history=history,
        elapsed=time.perf_counter() - start,
        imported_baseline=False,
        predictions=record_results(prob, test, dataset, config.model),
    )
