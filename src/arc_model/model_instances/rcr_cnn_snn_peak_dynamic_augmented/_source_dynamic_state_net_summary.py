"""Deterministic, label-free dynamics from native log10 band-power profiles.

Ordering is lag(1,2,4,8), then band(10--100/100--400 kHz), then cosine,
mean absolute difference, natural-log JS divergence and symmetric peak-share
change. The final four columns indicate contiguous history availability.

The symmetric peak change is (max(p)-max(q))/(max(p)+max(q)+1e-12), matching
the historical statistical definition. These values are freshly calculated
from the current five-channel profiles; no old-cohort feature cache is reused.
"""

import math

import torch

LAGS = (1, 2, 4, 8)
STATISTICS = ("cosine", "MAD", "JS_nats", "symmetric_peak_share_change")
DYNAMIC_FEATURE_NAMES = tuple(
    f"lag{lag}_band{band}_{stat}"
    for lag in LAGS
    for band in (0, 1)
    for stat in STATISTICS
) + tuple(f"history_lag{lag}_available" for lag in LAGS)


def history_availability(mask):
    """Require current, lag endpoint and every intervening cycle to be valid."""
    if mask.ndim != 2 or mask.shape[1] != 9:
        raise ValueError("mask must have shape [batch,9]")
    valid = mask > 0.5
    return torch.stack([valid[:, -lag - 1 :].all(dim=1) for lag in LAGS], dim=1)


def mask_dynamic_summary(summary, mask):
    """Clear missing statistics and preserve their four raw binary flags.

    Cached summaries may declare less history than the mask. They can never
    declare more: a gap anywhere in a lag's support makes that lag invalid.
    """
    if summary.ndim != 2 or summary.shape != (mask.shape[0], 36):
        raise ValueError("summary must have shape [batch,36]")
    available = history_availability(mask) & (summary[:, 32:] > 0.5)
    flags = available.to(summary.dtype)
    values = summary[:, :32].reshape(-1, 4, 8)
    values = torch.where(available[:, :, None], values, torch.zeros_like(values))
    return torch.cat([values.flatten(1), flags], dim=1)


def native_dynamic_summary(raw, mask):
    """Return [batch,36] causal statistics from raw [batch,9,5,64] input.

    ``raw`` contains unstandardized native low-frequency profiles, two log10
    band-power profiles and two adjacent-cycle distribution differences.
    Only the two log-power profiles are used here. Normalizing in log space
    avoids overflow and reconstructs the bin-average-power activity shares.
    Large finite log-power contrasts can yield exact zero probabilities;
    epsilon-regularized JS terms remain finite in that case.

    No rolling or cross-batch lookup is used. All selected histories belong
    to the same sample, and missing support yields zeros plus a zero flag.
    ``mask`` must identify any reference reset or discontinuity in advance.
    """
    if raw.ndim != 4 or raw.shape[1:] != (9, 5, 64):
        raise ValueError("raw must have shape [batch,9,5,64]")
    if not raw.is_floating_point():
        raise TypeError("raw must have floating-point dtype")
    if mask.shape != raw.shape[:2] or mask.device != raw.device:
        raise ValueError("mask must match raw batch/cycles and device")

    # Half precision cannot represent the JS epsilon. Keep these reductions
    # at least float32 while preserving float64 callers' reference precision.
    work = raw if raw.dtype in (torch.float32, torch.float64) else raw.float()
    valid = mask > 0.5
    log_power = torch.where(
        valid[:, :, None, None], work[:, :, 1:3], torch.zeros_like(work[:, :, 1:3])
    )
    distributions = torch.softmax(log_power * math.log(10.0), dim=-1)
    current = distributions[:, -1]
    available = history_availability(mask)
    eps = 1e-12
    per_lag = []
    for index, lag in enumerate(LAGS):
        previous = distributions[:, -1 - lag]
        midpoint = 0.5 * (current + previous)
        cosine = (current * previous).sum(-1) / (
            torch.linalg.vector_norm(current, dim=-1)
            * torch.linalg.vector_norm(previous, dim=-1)
            + eps
        )
        mad = (current - previous).abs().mean(-1)
        js = 0.5 * (
            (current * torch.log((current + eps) / (midpoint + eps))).sum(-1)
            + (previous * torch.log((previous + eps) / (midpoint + eps))).sum(-1)
        )
        peak_current = current.amax(-1)
        peak_previous = previous.amax(-1)
        peak_change = (peak_current - peak_previous) / (
            peak_current + peak_previous + eps
        )
        statistics = torch.stack([cosine, mad, js.clamp_min(0), peak_change], dim=-1)
        statistics = torch.where(
            available[:, index, None, None], statistics, torch.zeros_like(statistics)
        )
        per_lag.append(statistics.flatten(1))
    return torch.cat([*per_lag, available.to(work.dtype)], dim=1).to(raw.dtype)
