"""Outcome classification and diagnostic aggregation.

Responsible for: classifying each sample into one of four outcomes
(Resolved / Overprocessed / Misresolved / Unresolved) based on FPCL, FJC,
model output, and CSD tail confidence; aggregating per-sample diagnostics
into DiagnosticResult; and the disk-based entry point for S3.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from pathlib import Path

from brewing.schema import (
    DiagnosticResult, Granularity, HiddenStateCache, MethodResult,
    Outcome, Sample, SampleDiagnostic, SampleMethodResult,
    load_samples,
)
from .metrics import compute_fpcl, compute_fjc, compute_csd_tail_confidence

logger = logging.getLogger(__name__)

# Confidence threshold for Misresolved vs Unresolved
MISRESOLVED_THRESHOLD = 0.5


def classify_outcome(
    fpcl: int | None,
    fjc: int | None,
    model_output: str,
    answer: str,
    csd_tail_confidence: float,
) -> Outcome:
    """Classify a sample into one of four outcomes (or NO_BREWING).

    If FPCL is None, probing never found the answer at any layer — brewing
    never started, so the four-outcome taxonomy does not apply.
    """
    if fpcl is None:
        return Outcome.NO_BREWING
    if fjc is not None:
        if model_output == answer:
            return Outcome.RESOLVED
        else:
            return Outcome.OVERPROCESSED
    else:
        if csd_tail_confidence >= MISRESOLVED_THRESHOLD:
            return Outcome.MISRESOLVED
        else:
            return Outcome.UNRESOLVED


def diagnose_sample(
    sample: Sample,
    probe_result: SampleMethodResult,
    csd_result: SampleMethodResult,
    model_output: str,
    n_layers: int,
) -> SampleDiagnostic:
    """Compute all diagnostic indicators for a single sample."""
    fpcl = compute_fpcl(probe_result)
    fjc = compute_fjc(probe_result, csd_result)
    delta_brew = (fjc - fpcl) if (fjc is not None and fpcl is not None) else None

    csd_tail_conf = compute_csd_tail_confidence(csd_result, n_layers)
    outcome = classify_outcome(fpcl, fjc, model_output, sample.answer, csd_tail_conf)

    return SampleDiagnostic(
        sample_id=sample.id,
        fpcl=fpcl,
        fjc=fjc,
        delta_brew=delta_brew,
        outcome=outcome,
        model_output=model_output,
        csd_tail_confidence=csd_tail_conf,
    )


def run_diagnostics(
    samples: list[Sample],
    probe_result: MethodResult,
    csd_result: MethodResult,
    model_predictions: dict[str, str] | None = None,
    n_layers: int | None = None,
) -> DiagnosticResult:
    """Run full diagnostics on a set of samples.

    Args:
        samples: The eval samples
        probe_result: MethodResult from linear probing (per_sample)
        csd_result: MethodResult from CSD (per_sample)
        model_predictions: {sample_id: model_output} dict.
            If None, uses empty string (outcome will be Overprocessed/Misresolved/Unresolved).
        n_layers: Number of layers. If None, inferred from results.
    """
    if probe_result.granularity != Granularity.PER_SAMPLE:
        raise ValueError("Probe result must be per_sample granularity")
    if csd_result.granularity != Granularity.PER_SAMPLE:
        raise ValueError("CSD result must be per_sample granularity")

    # Build lookup
    probe_by_id = {sr.sample_id: sr for sr in probe_result.sample_results}
    csd_by_id = {sr.sample_id: sr for sr in csd_result.sample_results}

    if model_predictions is None:
        model_predictions = {}

    # Infer n_layers
    if n_layers is None:
        first_probe = probe_result.sample_results[0] if probe_result.sample_results else None
        n_layers = len(first_probe.layer_values) if first_probe else 0

    sample_diagnostics: list[SampleDiagnostic] = []

    for sample in samples:
        sid = sample.id
        if sid not in probe_by_id or sid not in csd_by_id:
            logger.warning("Sample '%s' missing from probe or CSD results, skipping", sid)
            continue

        model_output = model_predictions.get(sid, "")

        diag = diagnose_sample(
            sample=sample,
            probe_result=probe_by_id[sid],
            csd_result=csd_by_id[sid],
            model_output=model_output,
            n_layers=n_layers,
        )
        sample_diagnostics.append(diag)

    # Compute aggregates — outcome_distribution excludes NO_BREWING samples
    outcome_counts = Counter(sd.outcome for sd in sample_diagnostics)
    n_no_brewing = outcome_counts.get(Outcome.NO_BREWING, 0)
    brewing_total = len(sample_diagnostics) - n_no_brewing
    denom = brewing_total or 1
    BREWING_OUTCOMES = [Outcome.RESOLVED, Outcome.OVERPROCESSED, Outcome.MISRESOLVED, Outcome.UNRESOLVED]
    outcome_distribution = {
        o.value: outcome_counts.get(o, 0) / denom
        for o in BREWING_OUTCOMES
    }
    outcome_distribution["no_brewing_count"] = n_no_brewing
    outcome_distribution["no_brewing_rate"] = n_no_brewing / (len(sample_diagnostics) or 1)

    fpcls = [sd.fpcl for sd in sample_diagnostics if sd.fpcl is not None]
    fjcs = [sd.fjc for sd in sample_diagnostics if sd.fjc is not None]
    delta_brews = [sd.delta_brew for sd in sample_diagnostics if sd.delta_brew is not None]

    n_layers_f = float(n_layers) if n_layers > 0 else 1.0

    return DiagnosticResult(
        model_id=probe_result.model_id,
        eval_dataset_id=probe_result.eval_dataset_id,
        benchmark=samples[0].benchmark if samples else "unknown",
        subset=samples[0].subset if samples and all(s.subset == samples[0].subset for s in samples) else None,
        sample_diagnostics=sample_diagnostics,
        outcome_distribution=outcome_distribution,
        mean_fpcl_normalized=float(np.mean(fpcls)) / n_layers_f if fpcls else None,
        mean_fjc_normalized=float(np.mean(fjcs)) / n_layers_f if fjcs else None,
        mean_delta_brew=float(np.mean(delta_brews)) if delta_brews else None,
    )


def group_diagnostics_by_difficulty(
    samples: list[Sample],
    diagnostics: DiagnosticResult,
    group_key: str,
) -> dict[str, dict[str, Any]]:
    """Group diagnostic results by a difficulty dimension.

    Returns dict mapping group_value -> {outcome_distribution, mean_fpcl, ...}
    """
    sample_map = {s.id: s for s in samples}
    diag_map = {sd.sample_id: sd for sd in diagnostics.sample_diagnostics}

    groups: dict[str, list[SampleDiagnostic]] = {}
    for sd in diagnostics.sample_diagnostics:
        sample = sample_map.get(sd.sample_id)
        if sample is None or sample.difficulty is None:
            continue
        key_val = str(sample.difficulty.get(group_key, "unknown"))
        groups.setdefault(key_val, []).append(sd)

    BREWING_OUTCOMES = [Outcome.RESOLVED, Outcome.OVERPROCESSED, Outcome.MISRESOLVED, Outcome.UNRESOLVED]
    result = {}
    for key_val, diags in sorted(groups.items()):
        outcomes = Counter(sd.outcome for sd in diags)
        n_no_brewing = outcomes.get(Outcome.NO_BREWING, 0)
        brewing_total = len(diags) - n_no_brewing
        denom = brewing_total or 1
        fpcls = [sd.fpcl for sd in diags if sd.fpcl is not None]
        fjcs = [sd.fjc for sd in diags if sd.fjc is not None]
        dbs = [sd.delta_brew for sd in diags if sd.delta_brew is not None]

        dist = {o.value: outcomes.get(o, 0) / denom for o in BREWING_OUTCOMES}
        dist["no_brewing_count"] = n_no_brewing
        dist["no_brewing_rate"] = n_no_brewing / (len(diags) or 1)

        result[key_val] = {
            "n_samples": len(diags),
            "outcome_distribution": dist,
            "mean_fpcl": float(np.mean(fpcls)) if fpcls else None,
            "mean_fjc": float(np.mean(fjcs)) if fjcs else None,
            "mean_delta_brew": float(np.mean(dbs)) if dbs else None,
        }

    return result


def run_diagnostics_from_disk(
    results_dir: Path | str,
    *,
    # ResourceKey-based resolution
    key: "ResourceKey | None" = None,
    # Legacy explicit-path overrides (still supported for flexibility)
    model_id: str | None = None,
    eval_dataset_id: str | None = None,
    probe_result_path: Path | str | None = None,
    csd_result_path: Path | str | None = None,
    cache_path: Path | str | None = None,
    samples_path: Path | str | None = None,
    output_path: Path | str | None = None,
    allow_no_cache: bool = False,
) -> DiagnosticResult:
    """Run diagnostics from persisted files on disk.

    This is the decoupled entry point for S3: it loads MethodResult files,
    the eval cache (for model_predictions), and samples from disk, then
    delegates to run_diagnostics() for the actual computation.

    Supports two resolution modes:
      1. ResourceKey-based: pass a ``key`` with model_id, method will be
         set automatically for probe/csd lookups.
      2. Explicit paths: pass probe_result_path, csd_result_path, etc.

    Args:
        results_dir: Root output directory (the same output_root used by
            Orchestrator / ResourceManager).
        key: ResourceKey for structured resolution (model_id must be set).
        model_id: Model identifier (ignored when key is provided).
        eval_dataset_id: Dataset identifier (ignored when key is provided).
        probe_result_path: Explicit path to the linear_probing MethodResult JSON.
        csd_result_path: Explicit path to the CSD MethodResult JSON.
        cache_path: Explicit path to the HiddenStateCache .npz file.
        samples_path: Explicit path to the samples.json file.
        output_path: Where to save the DiagnosticResult JSON.

    Returns:
        The computed DiagnosticResult.
    """
    from brewing.resources import ResourceKey, ResourceManager

    results_dir = Path(results_dir)
    rm = ResourceManager(results_dir)

    # --- Resolve probe result ---
    if probe_result_path is not None:
        probe_result = MethodResult.load(Path(probe_result_path))
    elif key is not None:
        probe_key = ResourceKey(
            benchmark=key.benchmark, split=key.split, task=key.task,
            seed=key.seed, model_id=key.model_id, method="linear_probing",
        )
        probe_result = rm.resolve_result(probe_key)
        if probe_result is None:
            raise FileNotFoundError(
                f"Probe result not found at {rm.result_path(probe_key)}"
            )
    else:
        if model_id is None or eval_dataset_id is None:
            raise ValueError(
                "model_id and eval_dataset_id are required when "
                "neither key nor probe_result_path is provided"
            )
        raise ValueError(
            "Legacy string-based resolution is no longer supported. "
            "Use a ResourceKey or provide explicit paths."
        )

    # --- Resolve CSD result ---
    if csd_result_path is not None:
        csd_result = MethodResult.load(Path(csd_result_path))
    elif key is not None:
        csd_key = ResourceKey(
            benchmark=key.benchmark, split=key.split, task=key.task,
            seed=key.seed, model_id=key.model_id, method="csd",
        )
        csd_result = rm.resolve_result(csd_key)
        if csd_result is None:
            raise FileNotFoundError(
                f"CSD result not found at {rm.result_path(csd_key)}"
            )
    else:
        raise ValueError(
            "Legacy string-based resolution is no longer supported. "
            "Use a ResourceKey or provide explicit paths."
        )

    # Infer identifiers from loaded results if not provided
    if model_id is None:
        model_id = probe_result.model_id
    if eval_dataset_id is None:
        eval_dataset_id = probe_result.eval_dataset_id

    # --- Resolve samples ---
    if samples_path is not None:
        samples = load_samples(Path(samples_path))
    elif key is not None:
        loaded = rm.resolve_dataset(key)
        if loaded is None:
            raise FileNotFoundError(
                f"Dataset not found at {rm.dataset_dir(key)}"
            )
        _, samples = loaded
    else:
        raise FileNotFoundError(
            "Cannot resolve samples without key or explicit samples_path"
        )

    # --- Resolve eval cache for model_predictions and n_layers ---
    model_predictions: dict[str, str] | None = None
    n_layers: int | None = None

    if cache_path is not None:
        cache = HiddenStateCache.load(Path(cache_path))
    elif key is not None:
        cache = rm.resolve_cache(key)
    else:
        cache = None

    if cache is not None:
        model_predictions = {
            sid: pred
            for sid, pred in zip(cache.sample_ids, cache.model_predictions)
        }
        n_layers = cache.n_layers
    elif not allow_no_cache:
        raise FileNotFoundError(
            "Eval cache not found. Diagnostics require the eval cache for "
            "model_predictions (needed for outcome classification). "
            "Pass allow_no_cache=True to run without it (outcomes will be biased)."
        )
    else:
        logger.warning(
            "Running diagnostics without eval cache. model_predictions will be "
            "empty — all samples with FJC will be classified as Overprocessed."
        )

    # --- Run diagnostics (pure computation) ---
    diag = run_diagnostics(
        samples=samples,
        probe_result=probe_result,
        csd_result=csd_result,
        model_predictions=model_predictions,
        n_layers=n_layers,
    )

    # --- Persist ---
    if output_path is not None:
        save_path = Path(output_path)
    elif key is not None:
        save_path = rm.diagnostic_path(key)
    else:
        save_path = results_dir / "diagnostics.json"

    diag.save(save_path)
    logger.info("Diagnostics saved to %s", save_path)

    return diag
