"""Direct classification from recurrent phase states and explicit dynamics.

The existing self-owned CNN--LIF encoder is retained; there is one final
classifier, with no external detector scores, teacher or correction head.
Four arms are exposed for controlled comparisons. Existing source files and
the three sealed paper baselines are never modified by this implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from arc_model.implementations.map_input_calibration.deep_network import (
    DeepCodedPhaseMap,
    Spike,
)

from .summary import mask_dynamic_summary, native_dynamic_summary

PARAMETER_COUNTS = {
    "base": 7097,
    "dynamic": 8249,
    "capacity": 8285,
    "maxpool_dynamic": 8249,
}


class DynamicStateNet(DeepCodedPhaseMap):
    """Independent candidate wrapper around the self-owned 7097-parameter net.

    Inputs are already standardized per channel by outer-training statistics.
    ``set_statistics`` stores those five-channel statistics so native log10
    powers can be restored for deterministic summaries. The first 32 summary
    features are standardized by separate training-only statistics, while the
    final four history markers remain binary. Invalid histories stay zero
    even after standardization.

    ``forward(..., summary=native36)`` accepts a deterministic cached *raw*
    summary. Omitting it computes the same summary from restored raw inputs.
    """

    def __init__(self, arm="dynamic"):
        if arm not in PARAMETER_COUNTS:
            raise ValueError(f"Unknown dynamic-state arm: {arm}")
        super().__init__("raw")
        self.arm = arm
        self.register_buffer("input_mean", torch.zeros(5))
        self.register_buffer("input_std", torch.ones(5))
        self.register_buffer("dynamic_mean", torch.zeros(36))
        self.register_buffer("dynamic_std", torch.ones(36))

        old_hidden = self.readout_hidden
        if arm in ("dynamic", "maxpool_dynamic"):
            self.readout_hidden = nn.Linear(164, 32)
            with torch.no_grad():
                self.readout_hidden.weight[:, :128].copy_(old_hidden.weight)
                self.readout_hidden.weight[:, 128:].zero_()
                self.readout_hidden.bias.copy_(old_hidden.bias)
        elif arm == "capacity":
            old_norm, old_readout = self.readout_norm, self.readout
            self.readout_hidden = nn.Linear(128, 41)
            self.readout_norm = nn.BatchNorm1d(41)
            self.readout = nn.Linear(41, 1)
            with torch.no_grad():
                self.readout_hidden.weight[:32].copy_(old_hidden.weight)
                self.readout_hidden.bias[:32].copy_(old_hidden.bias)
                self.readout_norm.weight[:32].copy_(old_norm.weight)
                self.readout_norm.bias[:32].copy_(old_norm.bias)
                self.readout_norm.running_mean[:32].copy_(old_norm.running_mean)
                self.readout_norm.running_var[:32].copy_(old_norm.running_var)
                self.readout.weight[:, :32].copy_(old_readout.weight)
                self.readout.weight[:, 32:].zero_()
                self.readout.bias.copy_(old_readout.bias)

        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != PARAMETER_COUNTS[arm]:
            raise RuntimeError(f"Unexpected parameter count for {arm}: {actual}")

    @torch.no_grad()
    def set_statistics(self, mean5, std5, mean36, std36):
        """Install statistics fitted outside this module on training data only."""
        for name, values, length in (
            ("input_mean", mean5, 5),
            ("input_std", std5, 5),
            ("dynamic_mean", mean36, 36),
            ("dynamic_std", std36, 36),
        ):
            target = getattr(self, name)
            values = torch.as_tensor(values, device=target.device, dtype=target.dtype)
            if values.shape != (length,) or not torch.isfinite(values).all():
                raise ValueError(f"{name} must contain {length} finite values")
            if name.endswith("std") and not (values > 0).all():
                raise ValueError(f"{name} must be strictly positive")
            target.copy_(values)
        return self

    def _normalized_summary(self, x, mask, summary):
        if summary is None:
            raw = x * self.input_std[None, None, :, None] + self.input_mean[
                None, None, :, None
            ]
            summary = native_dynamic_summary(raw, mask)
        else:
            summary = summary.to(device=x.device, dtype=x.dtype)
        native = mask_dynamic_summary(summary, mask)
        values = (native[:, :32] - self.dynamic_mean[:32]) / self.dynamic_std[:32]
        values = torch.where(
            native[:, 32:, None].bool(),
            values.reshape(-1, 4, 8),
            torch.zeros_like(values.reshape(-1, 4, 8)),
        )
        return native, torch.cat([values.flatten(1), native[:, 32:]], dim=1)

    def forward(
        self,
        x,
        mask,
        return_activity=False,
        return_details=False,
        *,
        summary=None,
    ):
        if x.ndim != 4 or x.shape[1:] != (9, 5, 64):
            raise ValueError("Expected standardized input [batch,9,5,64]")
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [batch,9]")
        if self.arm == "base":
            return super().forward(x, mask, return_activity, return_details)

        batch, cycles, _, _ = x.shape
        z = (x * mask[:, :, None, None]).reshape(batch * cycles, 5, 64)
        z = F.gelu(self.n1(self.c1(z)))
        pool = F.max_pool1d if self.arm == "maxpool_dynamic" else F.avg_pool1d
        z = pool(F.gelu(self.n2(self.pw1(self.dw1(z)))), 4)
        z = pool(F.gelu(self.n3(self.pw2(self.dw2(z)))), 2)
        z = z + F.gelu(
            self.ms_norm(
                self.ms_fuse(torch.cat([conv(z) for conv in self.multiscale], dim=1))
            )
        )
        current = self.n4(self.input_map(z)).reshape(batch, cycles, 16, 8)
        leak = self.leak_logit.sigmoid()[None, :, None]
        threshold = (F.softplus(self.threshold_raw) + 0.05)[None, :, None]
        potential = torch.zeros_like(current[:, 0])
        trace = torch.zeros_like(potential)
        events = []
        for step in range(9):
            valid = mask[:, step, None, None]
            u = (potential * leak + current[:, step]) * valid
            spikes = Spike.apply(u - threshold) * valid
            potential = (u - threshold * spikes.detach()) * valid
            trace = (0.8 * trace + 0.2 * spikes) * valid
            events.append(spikes)
        representation = (potential / threshold + trace).flatten(1)

        if self.arm in ("dynamic", "maxpool_dynamic"):
            native, normalized = self._normalized_summary(x, mask, summary)
            # Splitting the two input blocks preserves the original 128-input
            # linear function when the added columns are zero at initialization.
            hidden = F.linear(
                representation,
                self.readout_hidden.weight[:, :128].contiguous(),
                self.readout_hidden.bias,
            ) + F.linear(normalized, self.readout_hidden.weight[:, 128:])
        else:
            native = normalized = None
            hidden = self.readout_hidden(representation)
        logits = self.readout(F.gelu(self.readout_norm(hidden))).flatten()
        activity = torch.stack(events, dim=1).sum() / (mask.sum().clamp_min(1) * 128)
        if return_details:
            return logits, {
                "representation": representation,
                "membrane": potential,
                "trace": trace,
                "spikes": torch.stack(events, dim=1),
                "activity": activity,
                "native_summary": native,
                "normalized_summary": normalized,
            }
        return (logits, activity) if return_activity else logits


def build_model(arm="base"):
    """Use the untouched original class for the base arm whenever possible."""
    return DeepCodedPhaseMap("raw") if arm == "base" else DynamicStateNet(arm)
