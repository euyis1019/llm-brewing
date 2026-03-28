"""Diagnostics engine — S3 of the Brewing pipeline (decoupled post-processing).

Responsible for: consuming persisted MethodResult files (from probing + CSD)
and producing DiagnosticResult with per-sample FPCL, FJC, Δ_brew, outcome
classification, and aggregate statistics.

NOT responsible for: running methods (that's S2/Orchestrator), model loading,
or dataset construction. S3 reads from disk and writes to disk — it has no
dependency on the Orchestrator.

Two entry points:
  - run_diagnostics(): in-memory, takes MethodResult objects directly
  - run_diagnostics_from_disk(): loads everything from ResourceManager paths
"""

from .metrics import (
    TAIL_FRACTION,
    compute_csd_tail_confidence,
    compute_fjc,
    compute_fpcl,
)
from .outcome import (
    MISRESOLVED_THRESHOLD,
    classify_outcome,
    diagnose_sample,
    group_diagnostics_by_difficulty,
    run_diagnostics,
    run_diagnostics_from_disk,
)

__all__ = [
    "TAIL_FRACTION",
    "MISRESOLVED_THRESHOLD",
    "compute_fpcl",
    "compute_fjc",
    "compute_csd_tail_confidence",
    "classify_outcome",
    "diagnose_sample",
    "run_diagnostics",
    "run_diagnostics_from_disk",
    "group_diagnostics_by_difficulty",
]
