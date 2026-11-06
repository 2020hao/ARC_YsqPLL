"""Phase-preserving CNN followed by shared, causal cycle-axis convolutions.

The input consists of native-rate-derived summaries, not raw HF waveforms.
Missing history is masked; the mask is never supplied as a learned feature.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PhaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, 24, 1, bias=False),
                                  nn.GroupNorm(4, 24), nn.GELU())
        self.scales = nn.ModuleList([
            nn.Conv1d(24, 24, k, padding=k//2, padding_mode='circular',
                      groups=24, bias=False) for k in (3, 7, 15)])
        self.fuse = nn.Conv1d(72, 32, 1, bias=False)
        self.skip = nn.Conv1d(24, 32, 1, bias=False)
        self.norm = nn.GroupNorm(4, 32)

    def forward(self, x):
        h = self.stem(x)
        h = self.fuse(torch.cat([layer(h) for layer in self.scales], dim=1)) + self.skip(h)
        return F.avg_pool1d(F.gelu(self.norm(h)), 4)


class CausalCycleBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(32, 32, 3, dilation=dilation, groups=32, bias=False)
        self.pointwise = nn.Conv1d(32, 32, 1, bias=False)
        # Normalize channels at each time separately: never normalize over time.
        self.norm = nn.LayerNorm(32)

    def forward(self, x, mask):
        x = x * mask
        h = self.pointwise(self.depthwise(F.pad(x, (2*self.dilation, 0))))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return (x + F.gelu(h)) * mask


class SPCycleNet(nn.Module):
    def __init__(self, use_history=True):
        super().__init__()
        self.use_history = use_history
        self.phase = PhaseEncoder()
        self.history = nn.ModuleList([CausalCycleBlock(1), CausalCycleBlock(3)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 32),
                                  nn.LayerNorm(32), nn.GELU(), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, x, mask):
        batch, cycles, channels, bins = x.shape
        if channels != 3 or bins != 64:
            raise ValueError('Expected [batch, history, 3, 64]')
        if not self.use_history:
            return self.head(self.phase(x[:, -1])).flatten()
        f = self.phase(x.reshape(batch*cycles, channels, bins)).reshape(batch, cycles, 32, 16)
        if self.use_history:
            u = f.permute(0, 3, 2, 1).reshape(batch*16, 32, cycles)
            m = mask[:, None, None, :].expand(batch, 16, 1, cycles).reshape(batch*16, 1, cycles)
            for block in self.history:
                u = block(u, m)
            last = u[:, :, -1].reshape(batch, 16, 32).transpose(1, 2)
        return self.head(last).flatten()
