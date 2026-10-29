"""Feature-level adapters around unchanged local backbones; no experiment paths."""

from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class PhaseFeatureAdapter(nn.Module):
    """Concatenate phase embedding before a jointly trainable linear classifier.

    Backbones are supplied by the caller. A new head starts from the original
    classifier with zero feature-column weights. ``enabled=False`` calls the
    untouched baseline forward path. This is not the historical C0 score head.
    """

    def __init__(self, backbone, name, feature_dim=10, embedding_dim=16):
        super().__init__()
        if name not in ("ArcFENet", "ArcFusionNet", "HYDRA"):
            raise ValueError("Unsupported backbone")
        if feature_dim < 1 or embedding_dim < 1:
            raise ValueError("Feature and embedding widths must be positive")
        self.backbone, self.name = deepcopy(backbone), name
        backbone = self.backbone
        head = (
            backbone.backbone.head
            if name == "ArcFENet"
            else backbone.head
            if name == "ArcFusionNet"
            else backbone
        )
        if not isinstance(head, nn.Linear):
            raise ValueError("An explicit linear baseline classifier is required")
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.GELU())
        self.fusion_head = nn.Linear(head.in_features + embedding_dim, head.out_features)
        with torch.no_grad():
            self.fusion_head.weight.zero_()
            self.fusion_head.weight[:, : head.in_features].copy_(head.weight)
            self.fusion_head.bias.copy_(
                head.bias if head.bias is not None else torch.zeros_like(self.fusion_head.bias)
            )
        # Retain original classifier for exact disabled/fallback predictions.
        # Freeze its now-inactive parameters; active fusion weights train normally.
        for parameter in head.parameters():
            parameter.requires_grad_(False)

    def baseline(self, x, spectral=None):
        if self.name == "HYDRA":
            return self.backbone(x)
        return self.backbone(x, spectral)

    def encode(self, x, spectral=None):
        if self.name == "HYDRA":
            return x
        if self.name == "ArcFENet":
            b = self.backbone.backbone
            if b.enhanced:
                a, c = b.hf(x), b.dd(x)
                z = a + c if b.fusion == "sum" else torch.cat([a, c], dim=1)
            else:
                z = b.front(x)
            return b.layers(z).flatten(1)
        b = self.backbone
        t = b.te(b.temporal(x).flatten(2).transpose(1, 2))
        s = b.se(b.spectral(spectral).transpose(1, 2))
        return F.scaled_dot_product_attention(s[:, None], t[:, None], t[:, None], dropout_p=0.0)[
            :, 0
        ].mean(1)

    def forward(self, x, spectral=None, features=None, *, enabled=True, available=None):
        if not enabled:
            return self.baseline(x, spectral)
        if features is None or features.shape != (len(x), self.feature_encoder[0].in_features):
            raise ValueError("Aligned phase feature matrix required")
        if available is not None:
            if available.shape != (len(x),) or available.dtype != torch.bool:
                raise ValueError("availability must be a boolean per-window mask")
            if not bool(available.any()):
                return self.baseline(x, spectral)
        h = self.encode(x, spectral)
        u = self.feature_encoder(features)
        logits = self.fusion_head(torch.cat([h, u], dim=-1))
        if available is not None and not bool(available.all()):
            # Use the same representation once; avoids a second BN/dropout pass.
            head = (
                self.backbone.backbone.head
                if self.name == "ArcFENet"
                else self.backbone.head
                if self.name == "ArcFusionNet"
                else self.backbone
            )
            logits = torch.where(available[:, None], logits, head(h))
        return logits
