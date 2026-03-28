# 8. YAML Config-Only CLI + Quantization Support

## What was done

### P0: YAML-config-only CLI
- Rewrote `brewing/cli.py` from argparse+JSON to YAML-config-only interface
- CLI now accepts only `--config path/to/config.yaml [--verbose]`
- All old experiment parameters removed: `--model`, `--subset`, `--methods`, `--fixture`, `--no-model`, `--output`, `--benchmark`, `--data-dir`, `--seed`, `--samples-per-config`, `--fit-policy`, `--batch-size`, `--device`
- YAML is read with `yaml.safe_load()`, top-level must be a mapping, fields map 1:1 to `RunConfig`
- Refactored `main()` into testable helpers:
  - `load_config()` — reads and validates YAML into RunConfig
  - `needs_model_online()` — checks method requirements via registry (not hardcoded)
  - `build_model_load_kwargs()` — constructs model load kwargs based on quantization
- "Needs model online" check now uses `get_method_class(name)().requirements().needs_model_online` instead of hardcoded `["csd"]`
- Added `pyyaml>=6.0` to `pyproject.toml` dependencies
- `main()` accepts optional `argv` parameter for testability

### P1: Quantization support
- Added `quantization: str | None = None` field to `RunConfig`
- Validation in `__post_init__`: only `None`, `"int8"`, `"int4"` are legal; anything else raises `ValueError`
- `build_model_load_kwargs()` handles three cases:
  - `None` → `torch_dtype=torch.float16`
  - `"int8"` → `BitsAndBytesConfig(load_in_8bit=True)`
  - `"int4"` → `BitsAndBytesConfig(load_in_4bit=True)`
- `device_map="auto"` and `output_hidden_states=True` preserved in all cases
- `bitsandbytes` not added to `pyproject.toml` — runtime lazy import

### Example configs
- `brewing/config/example_anchor.yaml` — 7B anchor model, full methods
- `brewing/config/example_14b_int8.yaml` — 14B model with INT8 quantization
- `brewing/config/example_single_task.yaml` — local smoke test (fixture + probing only)
- Removed `.gitkeep` from `brewing/config/`

### Tests
- New `tests/test_cli.py` with 14 tests covering:
  - YAML parsing, defaults, missing file, non-mapping YAML
  - quantization valid/invalid values
  - `needs_model_online` using registry (not hardcoded)
  - `build_model_load_kwargs` for None/int8/int4 (skipped without torch)
  - CLI argparse: missing --config exits, valid config invokes Orchestrator

### Fixture expansion
- `pure_copying` previously had only 1 fixture sample, which made probing's train/eval split fail
- Added 4 more `pure_copying` fixtures (total 5) so the smoke config can complete the full probing path: split → fit → evaluate

## Files changed
- `brewing/cli.py` — full rewrite
- `brewing/schema/results.py` — added `quantization` field + `__post_init__` validation
- `brewing/benchmarks/cue_bench/fixtures.py` — added 4 `pure_copying` fixtures (1→5)
- `pyproject.toml` — added `pyyaml>=6.0`
- `brewing/config/example_anchor.yaml` — new
- `brewing/config/example_14b_int8.yaml` — new
- `brewing/config/example_single_task.yaml` — new
- `brewing/config/.gitkeep` — deleted
- `tests/test_cli.py` — new
- `docs_dev_brewing/8_yaml_config_and_quantization.md` — this file

## Key decisions

1. **YAML-only, no fallback to argparse**: The old CLI allowed both `--config` (JSON) and individual args. Now `--config` is required and YAML-only. This enforces reproducibility — one config file = one run.

2. **Registry-based model loading check**: Instead of hardcoding `["csd"]` in the CLI, the check iterates `config.methods`, looks up each via `get_method_class()`, and checks `.requirements().needs_model_online`. Adding a new ModelOnlineMethod automatically triggers model loading.

3. **Validation in `RunConfig.__post_init__`**: Quantization validation is in the dataclass itself, so any construction path (YAML, programmatic, tests) gets the same check.

4. **`example_single_task.yaml` only has `linear_probing`**: CSD is a `ModelOnlineMethod` — it requires a loaded model. The single-task config is meant for local smoke tests without GPU, so including CSD would force a model load attempt and fail on CPU-only machines.

## Test results
- 48 passed, 3 skipped (model kwargs tests skip without torch/transformers)
- Smoke test (`example_single_task.yaml --verbose`): 5 fixture samples → 4 train / 1 eval → probing fit + eval → `method_linear_probing: ok`
