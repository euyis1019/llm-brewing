# 2. Brewing Framework Skeleton Implementation

**Date**: 2026-03-27
**Scope**: Full framework skeleton from schema to CLI

## What was done

Implemented the complete Brewing framework skeleton in `Brewing/brewing/`, covering all 7 phases from the implementation plan.

### Files created/modified

#### Core schema (`brewing/data.py`)

- Upgraded from minimal dataclasses to full schema layer
- Added: `DatasetManifest`, `HiddenStateCache`, `FitArtifact`, `MethodResult`, `SampleMethodResult`, `DiagnosticResult`, `SampleDiagnostic`
- Added enums: `DatasetPurpose`, `FitPolicy`, `FitStatus`, `Granularity`
- All structures have `save()`/`load()` round-trip serialization
- HiddenStateCache uses `.npz` + JSON sidecar (avoids numpy object array pickle issues)

#### Registry (`brewing/registry.py`)

- Central registry for benchmarks and methods
- Benchmarks auto-register on import (e.g., CUE-Bench)

#### CUE-Bench (`brewing/benchmarks/cue_bench.py`)

- Full 6-subset definition including `function_call`
- Datagen integration: `load_generated_dataset()`, `generate_and_convert()`
- Fixture samples for testing (1 per subset)
- `build_eval_dataset()` with fallback chain: disk → generate → fixture

#### Resource manager (`brewing/resources.py`)

- `ResourceManager` with resolve-or-build pattern
- Handles: datasets, hidden caches, fit artifacts, method results
- Path organization: `datasets/`, `caches/`, `artifacts/`, `results/`
- `resolve_artifact_with_policy()` implements `auto`/`force`/`eval_only`

#### Cache builder (`brewing/cache_builder.py`)

- `build_hidden_cache()`: extracts (N, L, D) hidden states
- Supports HuggingFace and nnterp model backends
- `make_synthetic_cache()`: structured random data for testing

#### Method interface (`brewing/methods/base.py`)

- `AnalysisMethod` → `CacheOnlyMethod` / `ModelOnlineMethod`
- Unified `run()` interface with full lifecycle support

#### Linear Probing (`brewing/methods/linear_probing.py`)

- Per-layer LogisticRegression (11-class: 0-9 + residual)
- Full fit/eval flow with artifact persistence
- Auto-registers as `"linear_probing"`

#### CSD (`brewing/methods/csd.py`)

- Patchscope-based information readiness
- Baseline subtraction
- Uses nnterp tracing when available, graceful fallback
- Auto-registers as `"csd"`

#### Diagnostics (`brewing/diagnostics.py`)

- `compute_fpcl()`, `compute_fjc()`, `classify_outcome()`
- `run_diagnostics()`: full S3 pipeline from MethodResult[] → DiagnosticResult
- `group_diagnostics_by_difficulty()`: per-dimension aggregation

#### Orchestrator (`brewing/orchestrator.py`)

- `RunConfig` dataclass for all configuration
- `Orchestrator.run()`: S0→S1→S2→S3→S4 pipeline per subset
- Handles model-available and no-model (synthetic cache) modes

#### CLI (`brewing/cli.py`, `brewing/__main__.py`)

- `python -m brewing.cli` entry point
- Supports: `--fixture`, `--no-model`, `--subset`, `--methods`, `--fit-policy`, `--config`

#### Tests

- `tests/test_schema.py`: 14 tests for data structure round-trips
- `tests/test_diagnostics.py`: 10 tests for FPCL/FJC/outcome/full diagnostics
- `tests/test_resources.py`: 7 tests for dataset/cache/artifact lifecycle
- `tests/test_e2e.py`: 2 end-to-end smoke tests
- **33 tests, all passing**

## Key decisions

1. **HiddenStateCache serialization**: `.npz` for numeric data + `.meta.json` sidecar for string metadata. Avoids numpy object array pickle issues.
2. **Training data for probing**: The orchestrator currently uses eval data as train data (same dataset for both). In production runs, `train_dataset_id` should point to a separate dataset. This is a known simplification for the skeleton.
3. **CSD fallback**: When nnterp is not available, CSD returns zero logits as fallback. This allows the pipeline to run end-to-end for testing, but real CSD requires a loaded model.
4. **Synthetic cache**: For no-model testing, generates structured random hidden states where answer signal grows with layer depth (simulates real probing dynamics).
5. **Per-subset processing**: The orchestrator processes each subset independently. Cross-subset analysis (e.g., comparing outcome fingerprints) is a post-processing step, not built into the orchestrator loop.

## Verified behaviors

- Schema round-trips (all structures serialize/deserialize correctly)
- Cache reuse (second run loads from disk, no re-computation)
- Artifact reuse with `fit_policy=auto` (loads existing) and `force` (re-trains)
- `fit_policy=eval_only` raises error when artifact missing
- Diagnostics produce correct FPCL/FJC/outcome for known inputs
- function_call datagen integration works (54 samples with `samples_per_config=3`)

