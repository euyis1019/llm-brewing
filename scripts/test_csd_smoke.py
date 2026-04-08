#!/usr/bin/env python3
"""Smoke test: run CSD on a handful of samples to verify the implementation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_csd_smoke.py \
        --model-size 0.5B --task value_tracking --n 5
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
from brewing.methods.csd import CSD
from brewing.resources import ResourceManager

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


def load_subset(model_id: str, task: str, n: int):
    """Load first n samples + their cached hidden states."""
    model_safe = model_id.replace("/", "__")
    cache_dir = CACHE_ROOT / task / "seed42" / model_safe

    # Load hidden states
    hs = np.load(cache_dir / "hidden_states.npz")
    hidden_states = hs[list(hs.keys())[0]]  # (N, L, D)

    # Load meta for sample_ids and model_predictions
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)

    # Load samples
    with open(DATA_DIR / f"{task}.json") as f:
        raw = json.load(f)
    all_samples = {s["id"]: s for s in raw}

    sample_ids = meta["sample_ids"][:n]
    samples = []
    for sid in sample_ids:
        s = all_samples[sid]
        samples.append(Sample(
            id=s["id"], benchmark="CUE-Bench", subset=task,
            prompt=s["prompt"], answer=s["answer"],
            metadata=s.get("metadata", {}),
        ))

    # Slice cache
    cache = HiddenStateCache(
        model_id=model_id,
        hidden_states=hidden_states[:n],
        sample_ids=sample_ids,
        model_predictions=meta.get("model_predictions", [""] * n)[:n],
    )
    return samples, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.5B", choices=MODEL_MAP.keys())
    parser.add_argument("--task", default="value_tracking")
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model_size]
    model_path = MODEL_DIR / model_id
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"=== CSD Smoke Test ===")
    print(f"Model: {model_id} ({args.model_size})")
    print(f"Task:  {args.task}")
    print(f"N:     {args.n}")

    # Load samples + cache
    samples, cache = load_subset(model_id, args.task, args.n)
    print(f"Cache shape: {cache.hidden_states.shape}")
    print(f"Samples: {[s.id for s in samples]}")
    print(f"Answers: {[s.answer for s in samples]}")

    # Load model via nnsight
    print(f"\nLoading model from {model_path} ...")
    t0 = time.time()
    from nnsight import LanguageModel
    model = LanguageModel(str(model_path), dtype=torch.float16, device_map="auto")
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s")

    # Run CSD
    csd = CSD()
    method_config = MethodConfig(
        method="csd",
        benchmark="CUE-Bench",
        config={
            "target_prompt": '# The value of x is "',
            "answer_space": [str(d) for d in range(10)],
            "eval_dataset_id": f"smoke-{args.task}",
        },
    )

    print(f"\nRunning CSD on {args.n} samples ...")
    t0 = time.time()
    result = csd.run(
        config=method_config,
        eval_samples=samples,
        eval_cache=cache,
        resources=ResourceManager(str(OUTPUT_ROOT)),
        model=model,
    )
    t_csd = time.time() - t0
    print(f"CSD completed in {t_csd:.1f}s ({t_csd/args.n:.2f}s/sample)")

    # Print per-sample results
    print(f"\n{'='*60}")
    print(f"{'Sample':<30} {'Answer':>6} {'FPCL':>5} {'Tail3':>6}")
    print(f"{'='*60}")
    for sr, sample in zip(result.sample_results, samples):
        # Find first correct layer
        fpcl = None
        for li, v in enumerate(sr.layer_values):
            if v > 0.5:
                fpcl = li
                break
        fpcl_str = str(fpcl) if fpcl is not None else "-"

        # Tail 3 layers accuracy
        tail3 = sr.layer_values[-3:].mean()

        print(f"{sample.id:<30} {sample.answer:>6} {fpcl_str:>5} {tail3:>6.2f}")

    # Layer-wise accuracy curve
    n_layers = cache.n_layers
    layer_acc = np.zeros(n_layers)
    for sr in result.sample_results:
        layer_acc += sr.layer_values
    layer_acc /= len(result.sample_results)

    print(f"\nLayer-wise CSD accuracy (avg over {args.n} samples):")
    for li in range(n_layers):
        bar = "#" * int(layer_acc[li] * 40)
        print(f"  L{li:02d}: {layer_acc[li]:.2f} {bar}")

    print(f"\nPer-sample time: {t_csd/args.n:.2f}s")
    print(f"Estimated full task (810 samples): {t_csd/args.n * 810 / 60:.1f} min")


if __name__ == "__main__":
    main()
