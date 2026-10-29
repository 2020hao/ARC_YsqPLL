"""Self-contained arc_fusion local reconstruction. No cross-model imports.

Numerical structure/state-dict keys match the existing audited local model.
This is not a claim of author-source equivalence. Extra branch is disabled by default.
"""

from torch import nn
from torch.nn import functional as F


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


class FusionNet(nn.Module):
    """Fig.7: 3 derivative rows, Conv2D temporal, RF FFT-32 spectral, transformers.

    The paper omits tokenization, transformer width/heads/feedforward/dropout and
    pooling stride. We use spatial tokens (3*250), d_model=64, heads=4, two
    encoder layers in each branch (Fig.7), FF=128, dropout=.1; spectral pool
    kernel=3, stride=2, padding=1. Eq.6 cross-attention has no extra projection.
    These explicit choices are NOT claimed to recover undisclosed author code.
    """

    def __init__(self, extra=0):
        super().__init__()
        self.temporal = nn.Sequential(
            TemporalBlock(1, 32, True), TemporalBlock(32, 32, False), TemporalBlock(32, 64, True)
        )
        layers = []
        for a, b in [(1, 16), (16, 32), (32, 64)]:
            layers += [
                nn.Conv1d(a, b, 5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(b),
                nn.Conv1d(b, b, 5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(b),
                nn.MaxPool1d(3, stride=2, padding=1),
            ]
        self.spectral = nn.Sequential(*layers)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.se = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(64, 2)
        self.extra = nn.Linear(extra, 2, bias=False) if extra else None

    def forward(self, x, spectral, features=None):
        t = self.temporal(x).flatten(2).transpose(1, 2)
        s = self.spectral(spectral).transpose(1, 2)
        t, s = self.te(t), self.se(s)
        fused = F.scaled_dot_product_attention(s[:, None], t[:, None], t[:, None], dropout_p=0.0)[
            :, 0
        ]
        y = self.head(fused.mean(1))
        return y if self.extra is None else y + self.extra(features)
