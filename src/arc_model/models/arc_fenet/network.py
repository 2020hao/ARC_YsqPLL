"""Checkpoint-compatible Arc FENet with optional physical-feature logit branch."""

from torch import nn
from .backbone import ArcFENet


class FENet(nn.Module):
    def __init__(self, extra=0):
        super().__init__()
        self.backbone = ArcFENet()
        self.extra = nn.Linear(extra, 2, bias=False) if extra else None

    def forward(self, x, spectral=None, features=None):
        y = self.backbone(x)
        return y if self.extra is None else y + self.extra(features)
