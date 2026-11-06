"""Local checkpoint replay; preprocessing remains training-only."""
from pathlib import Path
import torch
from .network import Model


def load_checkpoint(path, *, device="cpu"):
    model = Model()
    model.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_logits(model, inputs, mask, *, summary=None):
    model.eval()
    kwargs = {} if summary is None else {"summary": summary}
    return model(inputs, mask, **kwargs)
