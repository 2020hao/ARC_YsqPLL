"""Establish cycle memory before pooling at three phase resolutions.

Every LIF sequence has nine physical-cycle steps, without TTFS microsteps or
future cycles. Binary spike sequences pass to subsequent convolution stages;
only the final membrane/threshold and spike trace form an analog readout.
One classifier receives that readout and the existing native dynamic summary.
No external model, score correction, teacher, or old baseline is used.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._source_dynamic_state_net_network import DynamicStateNet
from ._source_map_input_calibration_deep_network import Spike

CHANNELS = {"small": (16, 24, 16), "wide": (24, 32, 24)}
READOUT_WIDTH = {"small": 32, "wide": 48}
PARAMETER_COUNTS = {"small": 7001, "wide": 13921}
MATRIX_MACS = {"small": 451104, "wide": 759792}
LIF_UPDATES = {"small": 18432, "wide": 26496}


class _PhaseLIF(nn.Module):
    """Shared channel parameters with separate phase-position membrane states."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.leak_logit = nn.Parameter(torch.full((channels,), math.log(4)))
        self.threshold_raw = nn.Parameter(
            torch.full((channels,), math.log(math.expm1(0.45)))
        )

    def forward(self, current, mask):
        if current.ndim != 4 or current.shape[1:3] != (9, self.channels):
            raise ValueError("LIF current must have nine steps and its configured channels")
        leak = self.leak_logit.sigmoid()[None, :, None]
        threshold = (F.softplus(self.threshold_raw) + 0.05)[None, :, None]
        potential = torch.zeros_like(current[:, 0])
        trace = torch.zeros_like(potential)
        events = []
        for step in range(9):
            valid = (mask[:, step, None, None] > 0.5).to(current.dtype)
            u = (potential * leak + current[:, step]) * valid
            spikes = Spike.apply(u - threshold) * valid
            potential = (u - threshold * spikes.detach()) * valid
            trace = (0.8 * trace + 0.2 * spikes) * valid
            events.append(spikes)
        events = torch.stack(events, dim=1)
        denominator = (mask > 0.5).sum().clamp_min(1) * current.shape[2] * current.shape[3]
        activity = events.sum() / denominator
        return events, potential, trace, threshold, activity


class PhaseHierarchyNet(nn.Module):
    """Three-stage 64/32/16-bin cycle memory with one final direct classifier.

    ``x`` is the original five-channel representation standardized by external
    training-fold statistics. Native summary reconstruction, training-only
    summary scaling, raw availability bits and cached-summary validation reuse
    the already frozen DynamicStateNet methods without changing their files.

    ``size='small'`` uses channels16/24/16 and a32-unit readout. ``wide`` uses
    channels24/32/24 and a48-unit readout. Both keep a single classification
    output. MAC accounting excludes the deterministic summary, normalization,
    pooling comparisons, surrogate/LIF scalar operations and raw frontend.
    """

    spiking = True

    def __init__(self, size="small"):
        super().__init__()
        if size not in CHANNELS:
            raise ValueError(f"Unknown phase hierarchy size: {size}")
        self.size = size
        self.input_coding = "raw"
        c1, c2, c3 = CHANNELS[size]
        width = READOUT_WIDTH[size]
        self.channels = (c1, c2, c3)
        self.register_buffer("input_mean", torch.zeros(5))
        self.register_buffer("input_std", torch.ones(5))
        self.register_buffer("dynamic_mean", torch.zeros(36))
        self.register_buffer("dynamic_std", torch.ones(36))

        self.stem = nn.Conv1d(5, c1, 5, padding=2, padding_mode="circular", bias=False)
        self.n1 = nn.BatchNorm1d(c1)
        self.lif1 = _PhaseLIF(c1)
        self.dw1 = nn.Conv1d(
            c1, c1, 7, padding=3, padding_mode="circular", groups=c1, bias=False
        )
        self.pw1 = nn.Conv1d(c1, c2, 1, bias=False)
        self.n2 = nn.BatchNorm1d(c2)
        self.lif2 = _PhaseLIF(c2)
        self.dw2 = nn.Conv1d(
            c2, c2, 5, padding=2, padding_mode="circular", groups=c2, bias=False
        )
        self.pw2 = nn.Conv1d(c2, c3, 1, bias=False)
        self.n3 = nn.BatchNorm1d(c3)
        self.lif3 = _PhaseLIF(c3)
        self.readout_hidden = nn.Linear(c3 * 8 + 36, width)
        self.readout_norm = nn.BatchNorm1d(width)
        self.readout = nn.Linear(width, 1)
        with torch.no_grad():
            self.readout_hidden.weight[:, -36:].zero_()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != PARAMETER_COUNTS[size]:
            raise RuntimeError(f"Unexpected parameter count for {size}: {actual}")

    def set_statistics(self, mean5, std5, mean36, std36):
        """Install independently fitted outer-training-fold statistics."""
        return DynamicStateNet.set_statistics(self, mean5, std5, mean36, std36)

    def _normalized_summary(self, x, mask, summary):
        return DynamicStateNet._normalized_summary(self, x, mask, summary)

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
        if mask.shape != x.shape[:2] or mask.device != x.device:
            raise ValueError("mask must match input batch/cycles and device")
        batch, cycles, _, _ = x.shape
        c1, c2, c3 = self.channels
        valid = mask > 0.5
        inputs = torch.where(valid[:, :, None, None], x, torch.zeros_like(x))

        current1 = self.n1(self.stem(inputs.reshape(batch * cycles, 5, 64)))
        current1 = current1.reshape(batch, cycles, c1, 64)
        spikes1, membrane1, trace1, threshold1, activity1 = self.lif1(current1, mask)
        pooled1 = F.max_pool1d(spikes1.reshape(batch * cycles, c1, 64), 2)

        current2 = self.n2(self.pw1(self.dw1(pooled1)))
        current2 = current2.reshape(batch, cycles, c2, 32)
        spikes2, membrane2, trace2, threshold2, activity2 = self.lif2(current2, mask)
        pooled2 = F.max_pool1d(spikes2.reshape(batch * cycles, c2, 32), 2)

        current3 = self.n3(self.pw2(self.dw2(pooled2)))
        current3 = current3.reshape(batch, cycles, c3, 16)
        spikes3, membrane3, trace3, threshold3, activity3 = self.lif3(current3, mask)
        representation = F.max_pool1d(membrane3 / threshold3 + trace3, 2).flatten(1)
        native, normalized = self._normalized_summary(x, mask, summary)
        # A real Linear module call makes dense matrix MAC hooks complete.
        hidden = self.readout_hidden(torch.cat([representation, normalized], dim=1))
        logits = self.readout(F.gelu(self.readout_norm(hidden))).flatten()
        rates = torch.stack([activity1, activity2, activity3])
        activity = rates.mean()
        if return_details:
            return logits, {
                "representation": representation,
                "membrane": membrane3,
                "trace": trace3,
                "spikes": spikes3,
                "activity": activity,
                "layer_rates": rates,
                "layer_spikes": (spikes1, spikes2, spikes3),
                "layer_membranes": (membrane1, membrane2, membrane3),
                "layer_traces": (trace1, trace2, trace3),
                "layer_thresholds": (threshold1, threshold2, threshold3),
                "native_summary": native,
                "normalized_summary": normalized,
            }
        return (logits, activity) if return_activity else logits

    def cost_accounting(self):
        """Dense per-output matrix/state counts, not measured sparse energy."""
        return {
            "parameters": PARAMETER_COUNTS[self.size],
            "matrix_MAC": MATRIX_MACS[self.size],
            "nominal_matrix_FLOPs": 2 * MATRIX_MACS[self.size],
            "LIF_scalar_updates": LIF_UPDATES[self.size],
            "history_cycles": 9,
            "LIF_phase_positions": (64, 32, 16),
            "scope": "Conv/Linear matrix MAC only; frontend, summary, BN, pooling, LIF scalar operations excluded",
        }


def build_model(size="small"):
    return PhaseHierarchyNet(size)
