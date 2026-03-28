# 11. Hierarchical Output Directory Layout

## What changed

Refactored the Brewing output directory from a flat layout to a hierarchical one that encodes benchmark, split, task, seed, and model in the directory path.

## New layout

```
{output_root}/
├── datasets/{benchmark}/{split}/{task}/seed{seed}/
├── caches/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
├── artifacts/{benchmark}/{task}/{model_id_safe}/{method}/seed{seed}/
└── results/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
```

## Key decisions

- **benchmark name**: `cuebench` (filesystem-friendly, via `RunConfig.benchmark_path_safe`)
- **ResourceKey**: frozen dataclass replacing flat `dataset_id` strings for path derivation
- **HiddenStateCache meta**: changed from `{name}.npz.meta.json` to sibling `meta.json`
- **RunConfig**: removed `eval_dataset_id` and `train_dataset_id` fields (superseded by ResourceKey)
- **YAML configs**: all COLM configs now share `output_root: brewing_output`

## Files changed

- `brewing/resources.py` — added `ResourceKey`, rewrote all path helpers and resolve/save methods
- `brewing/schema/types.py` — `HiddenStateCache.save()`/`load()` take optional `meta_path` param
- `brewing/schema/results.py` — `RunConfig` simplified, added `benchmark_path_safe` property
- `brewing/orchestrator.py` — uses `ResourceKey` throughout, `_make_key()` helper
- `brewing/methods/linear_probing.py` — `train()` takes `artifact_key: ResourceKey`
- `brewing/methods/csd.py` — imports ResourceKey (minimal change)
- `brewing/diagnostics/outcome.py` — `run_diagnostics_from_disk()` accepts `key: ResourceKey`
- `brewing/config/colm/*.yaml` — shared `output_root`, removed `train_dataset_id`
- `brewing/config/example_*.yaml` — same cleanup
- `brewing/config/colm/README.md` — updated conventions
- `tests/test_resources.py` — all tests use ResourceKey
- `tests/test_diagnostics.py` — uses ResourceKey for disk tests
- `tests/test_orchestrator.py` — tests `_train_key_for_subset` and `_make_key`
- `tests/test_linear_probing.py` — uses `artifact_key` param
- `tests/test_e2e.py` — uses ResourceKey
- `tests/test_schema.py` — updated HiddenStateCache round-trip
- `tests/test_cli.py` — added `benchmark_path_safe` test

## New files

- `scripts/migrate_output_layout.py` — migration script for old flat layout to new hierarchy
