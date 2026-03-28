"""Schema package — all data structures for the Brewing framework.

This is the single import point for all types. Consumers should do:
    from brewing.schema import Sample, HiddenStateCache, ...

Contains three files:
  - types.py:     Enums, Sample, DatasetManifest, HiddenStateCache, FitArtifact
  - results.py:   SampleMethodResult, MethodResult, DiagnosticResult, RunConfig
  - benchmark.py: BenchmarkSpec, SubsetSpec, compatibility checks

No runtime logic lives here — only dataclasses, enums, and serialization.
"""

from .types import (
    AnswerMeta,
    AnswerType,
    DatasetManifest,
    DatasetPurpose,
    FitArtifact,
    FitPolicy,
    FitStatus,
    Granularity,
    HiddenStateCache,
    Outcome,
    Sample,
    SingleTokenRequirement,
    load_samples,
    save_samples,
)
from .results import (
    DiagnosticResult,
    MethodConfig,
    MethodRequirements,
    MethodResult,
    RunConfig,
    SampleDiagnostic,
    SampleMethodResult,
)
from .benchmark import BenchmarkSpec, SubsetSpec, check_compatibility

__all__ = [
    # enums
    "AnswerType", "DatasetPurpose", "FitPolicy", "FitStatus",
    "Granularity", "Outcome", "SingleTokenRequirement",
    # sample
    "Sample", "load_samples", "save_samples",
    # benchmark
    "AnswerMeta", "BenchmarkSpec", "SubsetSpec",
    # dataset
    "DatasetManifest",
    # cache
    "HiddenStateCache",
    # artifact
    "FitArtifact",
    # method result
    "MethodConfig", "MethodRequirements", "MethodResult", "SampleMethodResult",
    # config
    "RunConfig",
    # diagnostics
    "DiagnosticResult", "SampleDiagnostic",
    # compat
    "check_compatibility",
]
