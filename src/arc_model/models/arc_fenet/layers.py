"""HFDC/DDC operators; numerical definitions preserved from audited reconstruction."""

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
