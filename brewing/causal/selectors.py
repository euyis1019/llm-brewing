"""Sample selectors for causal validation experiments.

Each selector filters samples from a DiagnosticResult based on
experiment-specific criteria and returns selection records with
skip reasons for rejected samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brewing.schema import (
    DiagnosticResult,
    HiddenStateCache,
    Sample,
    SampleCausalResult,
    SampleDiagnostic,
)


@dataclass
class SelectedSample:
    """A sample selected for causal intervention."""
    sample: Sample
    diagnostic: SampleDiagnostic
    source_layer: int
    cache_index: int  # index into cache hidden_states


def select_fjc_samples(
    samples: list[Sample],
    diagnostics: DiagnosticResult,
    cache: HiddenStateCache,
) -> tuple[list[SelectedSample], list[SampleCausalResult]]:
    """Select samples with non-null FJC for activation patching.

    Returns:
        (selected, skipped): selected samples ready for intervention,
        and SampleCausalResult entries for skipped samples with reasons.
    """
    diag_by_id = {sd.sample_id: sd for sd in diagnostics.sample_diagnostics}
    cache_id_set = set(cache.sample_ids)
    cache_index_map = {sid: i for i, sid in enumerate(cache.sample_ids)}

    selected: list[SelectedSample] = []
    skipped: list[SampleCausalResult] = []

    for sample in samples:
        sid = sample.id

        # Check diagnostic exists
        diag = diag_by_id.get(sid)
        if diag is None:
            skipped.append(SampleCausalResult(
                sample_id=sid, selected=False,
                skip_reason="no_diagnostic",
            ))
            continue

        # Check FJC exists
        if diag.fjc is None:
            skipped.append(SampleCausalResult(
                sample_id=sid, selected=False,
                skip_reason="fjc_is_none",
            ))
            continue

        # Check cache entry exists
        if sid not in cache_id_set:
            skipped.append(SampleCausalResult(
                sample_id=sid, selected=False,
                skip_reason="not_in_cache",
            ))
            continue

        selected.append(SelectedSample(
            sample=sample,
            diagnostic=diag,
            source_layer=diag.fjc,
            cache_index=cache_index_map[sid],
        ))

    return selected, skipped
