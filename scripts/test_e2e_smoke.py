#!/usr/bin/env python3
"""End-to-end smoke test: probing eval → CSD → save results → diagnostics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_e2e_smoke.py \
        --model-size 0.5B --task value_tracking --n 10
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import numpy as np
import torch

from brewing.schema import (
    HiddenStateCache, MethodConfig, Sample, Granularity,
)
from brewing.methods.linear_probing import LinearProbing
from brewing.methods.csd import CSD
from brewing.resources import ResourceKey, ResourceManager
from brewing.diagnostics.outcome import run_diagnostics

MODEL_MAP = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "3B":   "Qwen/Qwen2.5-Coder-3B",
    "7B":   "Qwen/Qwen2.5-Coder-7B",
    "14B":  "Qwen/Qwen2.5-Coder-14B",
}
MODEL_DIR = Path("/path/to/cue/models")
OUTPUT_ROOT = Path("brewing_output")
DATA_DIR = Path("brewing/benchmarks/cue_bench/data/eval")
CACHE_ROOT = OUTPUT_ROOT / "caches" / "cuebench" / "eval"
# Use a separate output root for smoke test to avoid polluting real results
SMOKE_OUTPUT = Path("brewing_output/_smoke_test")


def load_subset(model_id: str, task: str, n: int):
    """Load first n samples + their cached hidden states."""
    model_safe = model_id.replace("/", "__")
    cache_dir = CACHE_ROOT / task / "seed42" / model_safe

    hs = np.load(cache_dir / "hidden_states.npz")
    hidden_states = hs[list(hs.keys())[0]]  # (N, L, D)

    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)

    with open(DATA_DIR / f"{task}.json") as f:
        raw = json.load(f)
    all_samples = {s["id"]: s for s in raw}

    sample_ids = meta["sample_ids"][:n]
    model_preds = meta.get("model_predictions", [""] * len(meta["sample_ids"]))[:n]
    samples = []
    for sid in sample_ids:
        s = all_samples[sid]
        samples.append(Sample(
            id=s["id"], benchmark="CUE-Bench", subset=task,
            prompt=s["prompt"], answer=s["answer"],
            metadata=s.get("metadata", {}),
        ))

    cache = HiddenStateCache(
        model_id=model_id,
        hidden_states=hidden_states[:n],
        sample_ids=sample_ids,
        model_predictions=model_preds,
    )
    return samples, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.5B", choices=MODEL_MAP.keys())
    parser.add_argument("--task", default="value_tracking")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model_size]
    model_path = MODEL_DIR / model_id
    task = args.task
    n = args.n

    print(f"=== E2E Smoke Test ===")
    print(f"Model: {model_id} | Task: {task} | N: {n}")
    print(f"Smoke output: {SMOKE_OUTPUT}")
    print()

    # ── Load data ──
    samples, cache = load_subset(model_id, task, n)
    print(f"Loaded {len(samples)} samples, cache shape: {cache.hidden_states.shape}")

    # Resource managers: real one for reading artifacts, smoke one for writing results
    rm_real = ResourceManager(str(OUTPUT_ROOT))
    rm_smoke = ResourceManager(str(SMOKE_OUTPUT))

    # ── Step 1: Probing eval ──
    print("\n── Step 1: Probing eval ──")
    t0 = time.time()
    lp = LinearProbing()
    lp_config = MethodConfig(
        method="linear_probing",
        benchmark="CUE-Bench",
        config={
            "eval_dataset_id": f"smoke-{task}",
            "answer_space": [str(d) for d in range(10)],
            "resource_key_benchmark": "cuebench",
            "resource_key_task": task,
            "resource_key_seed": 42,
            "fit_policy": "eval_only",
        },
    )
    probe_result = lp.run(
        config=lp_config,
        eval_samples=samples,
        eval_cache=cache,
        resources=rm_real,  # reads artifacts from real output
    )
    t_probe = time.time() - t0
    print(f"  Probing done in {t_probe:.2f}s")
    print(f"  Sample results: {len(probe_result.sample_results)}")

    # Save probing result
    probe_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id, method="linear_probing",
    )
    probe_path = rm_smoke.save_result(probe_key, probe_result)
    print(f"  Saved to: {probe_path}")
    assert probe_path.exists(), f"Probe result not saved: {probe_path}"

    # ── Step 2: CSD ──
    print("\n── Step 2: CSD ──")
    t0 = time.time()
    from nnsight import LanguageModel
    model = LanguageModel(str(model_path), dtype=torch.float16, device_map="auto")
    t_load = time.time() - t0
    print(f"  Model loaded in {t_load:.1f}s")

    csd = CSD()
    csd_config = MethodConfig(
        method="csd",
        benchmark="CUE-Bench",
        config={
            "target_prompt": '# The value of x is "',
            "answer_space": [str(d) for d in range(10)],
            "eval_dataset_id": f"smoke-{task}",
        },
    )

    t0 = time.time()
    csd_result = csd.run(
        config=csd_config,
        eval_samples=samples,
        eval_cache=cache,
        resources=rm_smoke,
        model=model,
    )
    t_csd = time.time() - t0
    print(f"  CSD done in {t_csd:.2f}s ({t_csd/n:.2f}s/sample)")
    print(f"  Sample results: {len(csd_result.sample_results)}")

    # Save CSD result
    csd_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id, method="csd",
    )
    csd_path = rm_smoke.save_result(csd_key, csd_result)
    print(f"  Saved to: {csd_path}")
    assert csd_path.exists(), f"CSD result not saved: {csd_path}"

    # ── Step 3: Diagnostics ──
    print("\n── Step 3: Diagnostics ──")
    t0 = time.time()

    model_predictions = {
        sid: pred for sid, pred in zip(cache.sample_ids, cache.model_predictions)
    }

    diag = run_diagnostics(
        samples=samples,
        probe_result=probe_result,
        csd_result=csd_result,
        model_predictions=model_predictions,
        n_layers=cache.n_layers,
    )
    t_diag = time.time() - t0
    print(f"  Diagnostics done in {t_diag:.3f}s")

    # Print outcome distribution
    print(f"\n  Outcome distribution (n={n}):")
    for outcome, frac in diag.outcome_distribution.items():
        count = int(frac * n)
        print(f"    {outcome:<15s}: {count:>3d} ({frac:.1%})")

    print(f"\n  FPCL (norm): {diag.mean_fpcl_normalized}")
    print(f"  FJC  (norm): {diag.mean_fjc_normalized}")
    print(f"  ΔBrew:       {diag.mean_delta_brew}")

    # Save diagnostic result
    diag_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id,
    )
    diag_path = rm_smoke.save_diagnostic(diag_key, diag)
    print(f"\n  Saved to: {diag_path}")
    assert diag_path.exists(), f"Diagnostic result not saved: {diag_path}"

    # ── Step 4: Verify reload ──
    print("\n── Step 4: Verify reload from disk ──")
    from brewing.schema import MethodResult, DiagnosticResult
    probe_reload = MethodResult.load(probe_path)
    csd_reload = MethodResult.load(csd_path)
    diag_reload = DiagnosticResult.load(diag_path)
    print(f"  Probe result: {len(probe_reload.sample_results)} samples ✓")
    print(f"  CSD result:   {len(csd_reload.sample_results)} samples ✓")
    print(f"  Diagnostics:  {len(diag_reload.sample_diagnostics)} samples, "
          f"outcomes={diag_reload.outcome_distribution} ✓")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"E2E SMOKE TEST PASSED")
    print(f"  Probing:     {t_probe:.2f}s")
    print(f"  Model load:  {t_load:.1f}s")
    print(f"  CSD:         {t_csd:.2f}s")
    print(f"  Diagnostics: {t_diag:.3f}s")
    print(f"  All results saved under: {SMOKE_OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
