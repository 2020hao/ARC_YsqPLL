"""Self-contained arc_fenet local reconstruction. No cross-model imports.

Numerical structure/state-dict keys match the existing audited local model.
This is not a claim of author-source equivalence. Extra branch is disabled by default.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DifferentialConv(nn.Module):
    """HFDC Fig.9 and DDC Fig.10, implemented as equivalent learnable kernels."""

    def __init__(self, out_channels, kind):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, padding=1)
        self.kind = kind
        self.raw_a = nn.Parameter(torch.zeros(()))
        if kind == "ddc":
            self.raw_b = nn.Parameter(torch.zeros(()))

    def effective_weight(self):
        w = self.conv.weight
        if self.kind == "hfdc":
            mu = self.raw_a.sigmoid()
            mask = torch.ones_like(w) * (-mu / 8)
            mask[:, :, 1, 1] = 1
            return w * mask
        # Figure 10 anchors X' to the TOP-LEFT sample x11, not the centre.
        alpha = self.raw_a.sigmoid()
        beta = alpha * self.raw_b.sigmoid()  # 0 < beta < alpha < 1
        decay = torch.stack(
            [alpha * 0 + 1, alpha, beta, alpha, alpha, beta, beta, beta, beta]
        ).reshape(3, 3)
        corner = torch.zeros_like(w)
        corner[:, :, 0, 0] = (w * decay).sum((-1, -2))
        return w - corner

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.conv.bias, padding=1)


class ArcFENet(nn.Module):
    def __init__(self, fusion="sum", enhanced=True, channels=(16, 32, 64, 128)):
        super().__init__()
        channels = tuple(channels)
        if len(channels) != 4 or any(int(v) <= 0 for v in channels):
            raise ValueError("Four positive CNN widths required")
        self.fusion = fusion
        self.enhanced = enhanced
        if enhanced:
            width = channels[0] if fusion == "sum" else channels[0] // 2
            self.hf = DifferentialConv(width, "hfdc")
            self.dd = DifferentialConv(width, "ddc")
        else:
            self.front = nn.Conv2d(3, channels[0], 3, padding=1)
        layers = []
        for a, b in list(zip(channels[:-1], channels[1:])):
            layers += [
                nn.Conv2d(a, b, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            ]
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1] * 2 * 2, 2)

    def forward(self, x):
        if self.enhanced:
            a, b = self.hf(x), self.dd(x)
            x = a + b if self.fusion == "sum" else torch.cat([a, b], dim=1)
        else:
            x = self.front(x)
        return self.head(self.layers(x).flatten(1))


class FENet(nn.Module):
    def __init__(self, extra=0, fusion="sum", enhanced=True, channels=(16, 32, 64, 128)):
        super().__init__()
        self.backbone = ArcFENet(fusion=fusion, enhanced=enhanced, channels=channels)
        self.extra = nn.Linear(extra, 2, bias=False) if extra else None

    def forward(self, x, spectral=None, features=None):
        y = self.backbone(x)
        return y if self.extra is None else y + self.extra(features)


Model = FENet
