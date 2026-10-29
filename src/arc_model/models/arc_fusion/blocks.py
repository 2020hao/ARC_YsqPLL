"""Temporal convolution block used by the current-only ArcFusionNet adaptation."""

from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, a, b, residual):
        super().__init__()
        layers = []
        for incoming, kernel in [(a, 100), (b, 50), (b, 20)]:
            layers += [
                nn.Conv2d(incoming, b, (1, kernel), padding="same"),
                nn.ReLU(),
                nn.BatchNorm2d(b),
            ]
        self.net = nn.Sequential(*layers)
        self.short = nn.Conv2d(a, b, 1) if residual else None

    def forward(self, x):
        y = self.net(x)
        return y if self.short is None else y + self.short(x)
