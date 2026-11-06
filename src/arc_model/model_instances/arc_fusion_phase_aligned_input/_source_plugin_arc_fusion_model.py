"""Parameter-identical input-frontend candidate; no auxiliary branch."""

import torch
from ._source_plugin_arc_fusion_network import Model as Original


class Model(Original):
    def forward(self, x, spectral=None, *, aligned=None, enabled=False, available=None):
        if not enabled:
            return super().forward(x, spectral)
        if aligned is None:
            raise ValueError("Aligned model-native inputs required")
        a = aligned["temporal"] if isinstance(aligned, dict) else aligned
        b = aligned.get("spectrum") if isinstance(aligned, dict) else spectral
        if a.shape != x.shape:
            raise ValueError("Frontend must preserve the original input shape")
        if spectral is not None and (b is None or b.shape != spectral.shape):
            raise ValueError("Preserve the spectral32 interface")
        if available is not None:
            if available.shape != (len(x),) or available.dtype != torch.bool:
                raise ValueError("Per-window boolean availability required")
            a = torch.where(available.reshape((-1,) + (1,) * (a.ndim - 1)), a, x)
            if b is not None:
                b = torch.where(available[:, None, None], b, spectral)
        return super().forward(a, b)
