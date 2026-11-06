"""Unified direct CNN--SNN with explicit multi-lag dynamic-state readout."""

from .network import PARAMETER_COUNTS, DynamicStateNet, build_model
from .summary import DYNAMIC_FEATURE_NAMES, LAGS, native_dynamic_summary

__all__ = [
    "DYNAMIC_FEATURE_NAMES",
    "LAGS",
    "PARAMETER_COUNTS",
    "DynamicStateNet",
    "build_model",
    "native_dynamic_summary",
]
