# COLM Batch Configs

These YAML files are the default batch configs for the 9 models in
`/path/to/cue/COLM_REQUIREMENTS.md`.

Conventions:

- `data_dir` points to `/path/to/cue/data/colm_v1/eval`
- each run covers all 6 CUE-Bench subsets
- all models share `output_root: brewing_output` (hierarchical layout separates by benchmark/task/model)
- probing is evaluation-only in the main run
- probing artifacts are resolved via ResourceKey (benchmark + task + model + seed)
- the 14B model uses `quantization: int8`

Run one config:

```bash
uv run python -m brewing --config brewing/config/colm/qwen25_coder_7b.yaml --verbose
```

Run all configs:

```bash
for f in brewing/config/colm/*.yaml; do
  uv run python -m brewing --config "$f" --verbose
done
```
