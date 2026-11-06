"""Local training primitives; orchestration supplies disjoint reviewed rows."""
import torch
from torch.nn import functional as F
from .configuration import load_config
from .preprocessing import training_statistics, condition_class_weights


def make_optimizer(model):
    cfg = load_config()["training"]
    return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])


def train_batch(model, optimizer, x, mask, labels, sample_weights, *, summary=None, augmentation_generator=None):
    cfg = load_config()["training"]
    if len(x) < 2:
        raise ValueError("Merge singleton training tails; do not silently drop rows")
    model.train()
    if cfg.get("physical_augmentation", False):
        if augmentation_generator is None or not hasattr(model, "input_mean"):
            raise ValueError("Explicit augmentation RNG and installed training statistics required")
        from .augmentation import augment_standardized
        x = augment_standardized(x, model.input_mean, model.input_std, augmentation_generator)
    optimizer.zero_grad(set_to_none=True)
    extra = {} if summary is None else {"summary": summary}
    if hasattr(model, "spiking") or hasattr(model, "leak_logit") or hasattr(model, "lif1"):
        logits, activity = model(x, mask, return_activity=True, **extra)
    else:
        logits, activity = model(x, mask, **extra), x.new_zeros(())
    labels = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
    sample_weights = torch.as_tensor(sample_weights, dtype=logits.dtype, device=logits.device)
    if sample_weights.shape != labels.shape or not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
        raise ValueError("Finite nonnegative training-only sample weights required")
    loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * sample_weights).mean() + (cfg.get("spike_activity_penalty") or 0.0) * activity
    loss.backward()
    if cfg.get("gradient_clip") is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
    optimizer.step()
    return float(loss.detach())
