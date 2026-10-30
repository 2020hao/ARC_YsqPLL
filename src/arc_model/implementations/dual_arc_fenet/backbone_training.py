"""Independent prepared-data training recipe. Writes new runs, never release metrics."""

from pathlib import Path
import json
import numpy as np
import torch
from .network import Model
from .configuration import load_config
from .preprocessing import fit_state, apply_state


def fit(loader, train_indices, test_indices, output, *, seed=0, device="cpu"):
    out = Path(output).resolve()
    package = Path(__file__).resolve().parent
    if out == package or package in out.parents:
        raise ValueError("Training output cannot be a frozen/source package directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    import yaml

    (out / "resolved_model.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    )
    tr = np.asarray(train_indices, dtype=int)
    te = np.asarray(test_indices, dtype=int)
    if np.intersect1d(tr, te).size:
        raise ValueError("Train/test overlap")
    for key in ["condition_key", "source_group", "record_id"]:
        if set(loader.table.iloc[tr][key]) & set(loader.table.iloc[te][key]):
            raise ValueError(key + " leakage")
    state = fit_state(loader, tr, seed)
    x, s = apply_state(loader, state)
    torch.backends.cuda.matmul.allow_tf32 = cfg["precision"]["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["precision"]["cudnn_tf32"]
    torch.backends.cudnn.deterministic = cfg["precision"]["cudnn_deterministic"]
    torch.manual_seed(seed)
    net = Model(**cfg["model_kwargs"]).to(device)
    raw = cfg["model_training_param"]
    settings = dict(
        epochs=raw["epochs"],
        optimizer=raw["optimizer"],
        lr=raw["lr"],
        weight_decay=raw["wd"],
        batch=raw["batch_size"],
    )
    kind = torch.optim.Adam if settings["optimizer"] == "Adam" else torch.optim.AdamW
    opt = kind(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, settings["epochs"], eta_min=raw["min_lr"])
        if raw["scheduler"] == "cosine"
        else None
    )
    history = []
    rng = torch.Generator().manual_seed(seed + 10000)
    for epoch in range(settings["epochs"]):
        net.train()
        order = tr[torch.randperm(len(tr), generator=rng).numpy()]
        total = 0.0
        for start in range(0, len(order), settings["batch"]):
            ids = order[start : start + settings["batch"]]
            a = torch.tensor(np.array(x[ids]), device=device)
            b = None if s is None else torch.tensor(np.array(s[ids]), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda")
                and cfg["precision"]["autocast"] == "bfloat16",
            ):
                z = net(a, b)
            loss = torch.nn.functional.cross_entropy(
                z.float(), torch.tensor(loader.labels[ids], device=device)
            )
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ids)
        if schedule:
            schedule.step()
        history.append(total / len(tr))
    net.eval()
    prob = []
    with torch.no_grad():
        for start in range(0, len(te), settings["batch"]):
            ids = te[start : start + settings["batch"]]
            a = torch.tensor(np.array(x[ids]), device=device)
            b = None if s is None else torch.tensor(np.array(s[ids]), device=device)
            prob.append(net(a, b).softmax(-1)[:, 1].cpu().numpy())
    torch.save(net.state_dict(), out / "model.pt")
    np.savez(out / "preprocessing.npz", **state)
    np.savez(
        out / "predictions.npz", indices=te, labels=loader.labels[te], score=np.concatenate(prob)
    )
    np.savez(out / "indices.npz", train=tr, test=te)
    (out / "run.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                history=history,
                performance_status="new_run_not_the_frozen_scores",
                recipe="portable reference runner; exact historical execution sources in provenance.json",
            ),
            indent=2,
        )
    )
    return net, state
