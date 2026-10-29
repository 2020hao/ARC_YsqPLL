"""Paper-reconstructed Arc FENet backbone (not author source)."""

import torch
from torch import nn
from .layers import DifferentialConv


class ArcFENet(nn.Module):
    def __init__(self, fusion="sum", enhanced=True):
        super().__init__()
        self.fusion = fusion
        self.enhanced = enhanced
        if enhanced:
            width = 16 if fusion == "sum" else 8
            self.hf = DifferentialConv(width, "hfdc")
            self.dd = DifferentialConv(width, "ddc")
        else:
            self.front = nn.Conv2d(3, 16, 3, padding=1)
        layers = []
        for a, b in [(16, 32), (32, 64), (64, 128)]:
            layers += [
                nn.Conv2d(a, b, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            ]
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(128 * 2 * 2, 2)

    def forward(self, x):
        if self.enhanced:
            a, b = self.hf(x), self.dd(x)
            x = a + b if self.fusion == "sum" else torch.cat([a, b], dim=1)
        else:
            x = self.front(x)
        return self.head(self.layers(x).flatten(1))
