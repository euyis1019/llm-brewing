"""Diagnostic metrics — per-sample layer-wise indicators.

Computes:
  - FPCL (First Probe-Correct Layer): earliest layer where probe is correct
  - FJC (First Joint-Correct Layer): earliest layer where BOTH probe and CSD
    are correct simultaneously
  - CSD tail confidence: max per-class average confidence in the tail window
    (used for Misresolved vs Unresolved distinction)

These metrics consume SampleMethodResult objects (from probing and CSD).
"""

from __future__ import annotations

import logging

import numpy as np

from brewing.schema import SampleMethodResult

logger = logging.getLogger(__name__)

# Tail window starts at 3/4 of total layers
TAIL_FRACTION = 0.75


def compute_fpcl(
    probe_result: SampleMethodResult,
) -> int | None:
    """First Probe-Correct Layer: first layer where probe predicts correctly."""
    for layer_idx, val in enumerate(probe_result.layer_values):
        if val > 0.5:  # correctness flag
            return layer_idx
    return None


def compute_fjc(
    probe_result: SampleMethodResult,
    csd_result: SampleMethodResult,
) -> int | None:
    """First Joint-Correct Layer: first layer where BOTH probe and CSD are correct."""
    n_layers = min(len(probe_result.layer_values), len(csd_result.layer_values))
    for layer_idx in range(n_layers):
        if probe_result.layer_values[layer_idx] > 0.5 and \
           csd_result.layer_values[layer_idx] > 0.5:
            return layer_idx
    return None


def compute_csd_tail_confidence(
    csd_result: SampleMethodResult,
    n_layers: int,
) -> float:
    """Max average confidence in the tail window (>= 3L/4)."""
    tail_start = int(n_layers * TAIL_FRACTION)
    if csd_result.layer_confidences is None:
        # layer_values is correctness (0/1), NOT confidence — cannot use as fallback
        logger.warning(
            "CSD result for sample '%s' has no layer_confidences; "
            "cannot compute tail confidence, returning 0.0 (conservative Unresolved)",
            csd_result.sample_id,
        )
        return 0.0

    # Max of per-class average confidence in tail window
    tail_confs = csd_result.layer_confidences[tail_start:]  # (tail_len, C)
    if len(tail_confs) == 0:
        return 0.0
    # For each class, compute mean confidence across tail layers
    mean_per_class = tail_confs.mean(axis=0)  # (C,)
    return float(np.max(mean_per_class))
