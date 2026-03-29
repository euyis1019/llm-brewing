# 16: Causal Validation MVP — Activation Patching at FJC

**Date**: 2026-03-29

## Summary

Implemented a minimal viable causal validation subsystem for Brewing. This adds a new pipeline mode `causal_validation` with the first concrete experiment: **Activation Patching at FJC** — reads the real hidden state at the FJC layer from the eval cache and injects it into a neutral target prompt (patchscope-style) to verify the FJC layer is causally sufficient for the answer.

## What was done

### New files

| File | Purpose |
|------|---------|
| `brewing/causal/__init__.py` | Package init, exports |
| `brewing/causal/base.py` | `CausalValidator` ABC + `VALIDATOR_REGISTRY` |
| `brewing/causal/backend.py` | `InterventionBackend` abstraction + `NNsightInterventionBackend` + `FakeInterventionBackend` (testing) |
| `brewing/causal/selectors.py` | `select_fjc_samples()` — filters samples with valid FJC, returns skip reasons |
| `brewing/causal/activation_patching.py` | `ActivationPatchingFJC` validator — the actual experiment |
| `brewing/pipelines/causal_validation.py` | `CausalValidationPipeline` — loads S1/S2/S3 from disk, runs validators |
| `tests/test_causal_validation.py` | 23 tests covering schema, resources, selectors, validator, pipeline, CLI, registry |
| `docs_dev_brewing/16_causal_validation_mvp.md` | This file |

### Modified files

| File | Change |
|------|--------|
| `brewing/schema/results.py` | Added `SampleCausalResult`, `CausalValidationResult`, `causal_validation` to `VALID_MODES`, `causal_validation` field on `RunConfig` |
| `brewing/schema/__init__.py` | Export new types |
| `brewing/resources.py` | `causal_result_dir/path`, `save_causal_result`, `resolve_causal_result` |
| `brewing/pipelines/__init__.py` | Register `CausalValidationPipeline` in `PIPELINE_REGISTRY` |
| `brewing/cli.py` | `needs_model_online()` returns True for `causal_validation` mode |
| `tests/test_pipelines.py` | Updated registry/mode tests to include `causal_validation` |

## Key design decisions

1. **Separate `brewing/causal/` package** — not in `methods/` because causal validation depends on S3 diagnostics (FJC/Outcome), so it's conceptually a post-S3 verification layer, not an S2 analysis method.

2. **Patchscope-style activation patching** — reads the real FJC hidden state from cache and injects it into a neutral target prompt (default: `'# The value of x is "'`). This is NOT zero ablation; the experiment tests whether the FJC layer contains causally sufficient information that transfers to a different context.

3. **InterventionBackend abstraction** — thin layer between validators and nnsight. `InterventionRequest` carries both `source_prompt` (provenance) and `target_prompt` (where injection happens). `NNsightInterventionBackend` matches dtype to target layer's actual dtype (no hardcoded float16). `FakeInterventionBackend` enables full testing without GPU.

4. **Strict disk-only loading** — `CausalValidationPipeline` NEVER regenerates datasets or caches. All S0/S1/S2/S3 artifacts must pre-exist from prior runs. This prevents desynchronization between causal validation inputs and the artifacts they depend on.

5. **Config flows through to validators** — YAML `causal_validation` section supports per-experiment config (e.g., `activation_patching_fjc.intervention.target_prompt`). Pipeline extracts the experiment-specific section and passes it as `config=` kwarg to the validator.

6. **Declarative selector pattern** — `select_fjc_samples()` returns both selected samples and explicit skip-reason records, so no sample is silently dropped.

7. **Separate disk path** — `output_root/causal/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/{experiment}.json`. Not mixed into existing `results/` tree.

8. **Validator registry** — `@register_validator` decorator parallels the method registry. Future `layer_skipping` and `reinjection` validators just need to implement `CausalValidator.run()` and register.

## Extension points for future experiments

- **Layer Skipping**: Implement `CausalValidator` subclass, add a `select_overprocessed_samples()` selector, use `InterventionBackend` to skip layers after FJC.
- **Re-injection**: Implement multi-round validator with `round_idx` tracking in `SampleCausalResult`, add `select_unresolved_samples()` selector.

## Test results

```
118 passed, 9 warnings in 12.19s
```

All 91 existing tests + 27 new causal validation tests pass.
