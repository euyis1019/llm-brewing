# Brewing

Implementation for the experiments in *"From Brewing to Resolution: Tracing the
Internal Lifecycle of Code Reasoning in LLMs"*.

Standard accuracy only tells you whether a model got the answer right, not how
that answer formed inside the network. Brewing studies code reasoning as an
internal **layer-wise lifecycle**: answers often become linearly readable from
hidden states *before* the model itself can decode them. That intermediate
regime is what we call `brewing`.

Two complementary views track this lifecycle:

- `linear_probing` (Φ_P) — **information availability**: is the answer already
  linearly readable from the hidden state?
- `csd` (Φ_C) — **information readiness**: can the model itself decode the answer
  from that state?

Their gap over the layer axis (the *brewing gap*) sorts each example into one of
four outcomes: `resolved`, `overprocessed`, `misresolved`, `unresolved`.

> 🚧 Next up (2026-07-10): a more essential, more *general* line of research.

## Design Philosophy: Config-Driven

Brewing is **config-driven**. There is exactly one entry point —

```bash
python -m brewing --config path/to/config.yaml [--verbose]
```

— and a single YAML file declares everything about a run: which `mode` to
execute, which benchmark and subsets, which model, which methods, and where
outputs land. The CLI parses no behavioral flags; it only loads the config,
decides whether a model must be brought online, and hands off to the
orchestrator. This has a few consequences worth knowing:

- **A run is fully described by its config.** No hidden CLI state, no
  environment-dependent branching. The same YAML reproduces the same run, which
  is why the configs under `brewing/config/experiments/` *are* the experiment
  record.
- **Modes compose into a pipeline, not a monolith.** Each run does one stage
  (`cache_only`, `train_probing`, `eval`, `diagnostics`, `causal_validation`).
  You chain them by running configs in sequence; intermediate artifacts on disk
  are the interface between stages.
- **Resolve-or-build everywhere.** Datasets, caches, and probe artifacts are
  keyed deterministically (benchmark × split × subset × seed × model × method).
  If an artifact already exists it is loaded, not recomputed — so reruns are
  cheap and interrupted jobs resume naturally.
- **Diagnostics are decoupled.** `diagnostics` reads saved method results from
  disk and needs no model. The expensive GPU stages and the cheap analysis stage
  never have to run together.

The framework keeps its interface boundaries (`BenchmarkSpec`,
`AnalysisMethod`, `MethodResult`) generic so new benchmarks and methods slot in
without touching the orchestrator.

## Pipeline

```
S0  Dataset resolve / build
S1  Hidden-state cache extraction
S2  Method run (Probing / CSD) → MethodResult on disk
─── pipeline boundary ───
S3  Diagnostics (independent post-processing): FPCL / FJC / outcome labels
```

| Mode | Purpose | Needs online model |
| --- | --- | --- |
| `cache_only` | Build/load dataset and extract hidden-state caches | Yes, unless all caches exist |
| `train_probing` | Train per-layer probe artifacts from train-split caches | Yes, unless required caches exist |
| `eval` | Run `linear_probing` / `csd` on eval data | Depends on selected methods |
| `diagnostics` | Compute FPCL, FJC, brewing gap, outcome labels from saved results | No |
| `causal_validation` | Intervention experiments over existing S0–S3 artifacts | Yes |

Typical end-to-end flow: `cache_only` → `train_probing` → `eval` →
`diagnostics` → (optionally) `causal_validation`.

## Installation

```bash
pip install -e .            # minimal
pip install -e .[model]     # model-backed runs (cache extraction, CSD, causal)
pip install -e .[dev]       # test dependencies
```

Python `>=3.10` is required.

## Quick Start

The smallest runnable path uses the built-in fixture (no model, no setup):

```bash
python -m brewing --config brewing/config/example_single_task.yaml
```

That config sets `use_fixture: true`, one subset (`value_tracking`), and one
method (`linear_probing`) — the fastest way to verify the CLI and output layout.

Other example configs:

```text
brewing/config/example_single_task.yaml
brewing/config/example_probing_tune.yaml
brewing/config/example_local_model.yaml
brewing/config/example_full_reference.yaml
brewing/config/experiments/*.yaml
```

## Benchmark

The default benchmark is `CUE-Bench`, six code-reasoning subsets, all with
single-digit answers (0–9):

`value_tracking`, `computing`, `conditional`, `function_call`, `loop`,
`loop_unrolled`.

## Outputs

All artifacts go under `output_root` (default `brewing_output/`), organized by
benchmark / split / subset / seed / model / method:

```text
datasets/...    caches/...    artifacts/...    results/...    run_summary.json
```

Path logic lives in `brewing/resources.py`.

## Repository Layout

```text
brewing/
  benchmarks/   Benchmark specs, builders, adapters, built-in data
  causal/       Causal intervention backends and validators
  config/       Example and experiment YAML configs
  diagnostics/  Outcome taxonomy and aggregate metrics (S3)
  methods/      Linear probing and CSD
  pipelines/    cache_only / train_probing / eval / diagnostics / causal_validation
  schema/       Shared dataclasses and serialization
  cli.py        CLI entry point for `python -m brewing`
docs/           Architecture notes and per-mode behavior
scripts/        Smoke tests and experiment helpers
tests/          Unit and integration tests
```

## Documentation

- [docs/project_overview.md](docs/project_overview.md) — high-level architecture
- [docs/running_modes.md](docs/running_modes.md) — per-mode behavior
- [brewing/config/README.md](brewing/config/README.md) — config reference

## Tests

```bash
pytest                              # unit + integration suite
python scripts/test_e2e_smoke.py    # model-backed end-to-end (assumes local assets)
```

## Citation

```bibtex
@misc{brewing2026,
  title={From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs},
  author={Chen, Siyue and Guo, Yifu and Lu, Yuquan and Xu, Zishan and Lin, Jiaye and
          Lin, Jianbo and Zhang, Siyu and Yang, Cheng and Li, Junxin and Li, Yujia and
          Huo, Yu and Wang, Ruixuan},
  year={2026},
  note={Under review},
}
```
