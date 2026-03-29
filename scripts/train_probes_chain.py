#!/usr/bin/env python3
"""Train per-layer linear probes using chain_ pooled data and save artifacts.

chain_ naming convention:
  chain_train = original train (3240) + 4/5 of eval (648) = 3888 samples
  chain_es    = held-out 1/5 of eval (162) — used inside _fit_probes for ES
  eval (full) = all 810 eval samples — used for final accuracy reporting

This script does NOT modify framework code. It constructs a pooled
HiddenStateCache + Sample list, then calls the existing
LinearProbing.train() which handles ES splitting internally.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import torch

from brewing.schema import HiddenStateCache, Sample
from brewing.methods.linear_probing import LinearProbing, DIGIT_CLASSES
from brewing.resources import ResourceKey, ResourceManager

# ── Config ──
OUTPUT_ROOT = Path("brewing_output")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
CACHE_ROOT = OUTPUT_ROOT / "caches" / "cuebench"

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]
MODELS = {
    "0.5B":      "Qwen/Qwen2.5-Coder-0.5B",
    "1.5B":      "Qwen/Qwen2.5-Coder-1.5B",
    "3B":        "Qwen/Qwen2.5-Coder-3B",
    "7B":        "Qwen/Qwen2.5-Coder-7B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B-Base",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B-Base",
}

CHAIN_PROBE_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 0.05,
    "epochs": 1000,
    "patience": 50,
    "batch_size": 99999,  # full-batch: set larger than any dataset
}
EVAL_POOL_RATIO = 4 / 5  # 4/5 eval goes into chain_train
CHAIN_SEED = 42


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_samples_from_json(split: str, task: str) -> list[Sample]:
    path = DATA_ROOT / split / f"{task}.json"
    with open(path) as f:
        raw = json.load(f)
    return [
        Sample(
            id=s["id"], benchmark="CUE-Bench", subset=task,
            prompt=s["prompt"], answer=s["answer"],
            metadata=s.get("metadata", {}),
        )
        for s in raw
    ]


def build_chain_train(
    task: str, model_id: str,
) -> tuple[HiddenStateCache, list[Sample], np.ndarray, np.ndarray, list[Sample]]:
    """Build chain_train cache by pooling train + 4/5 eval.

    Returns:
        chain_cache: pooled HiddenStateCache (3888 samples)
        chain_samples: corresponding Sample list
        eval_full_hs: full eval hidden states (810) for final reporting
        eval_full_labels: full eval labels
        eval_full_samples: full eval Sample list
    """
    mk = safe_model_id(model_id)

    # Load train
    train_hs = np.load(CACHE_ROOT / "train" / task / "seed42" / mk / "hidden_states.npz")["hidden_states"]
    train_samples = load_samples_from_json("train", task)

    # Load eval
    eval_hs = np.load(CACHE_ROOT / "eval" / task / "seed42" / mk / "hidden_states.npz")["hidden_states"]
    eval_samples = load_samples_from_json("eval", task)

    n_eval = len(eval_samples)
    rng = np.random.RandomState(CHAIN_SEED)
    perm = rng.permutation(n_eval)
    split_idx = int(n_eval * EVAL_POOL_RATIO)  # 648

    chain_eval_idx = perm[:split_idx]   # 4/5 eval → into chain_train

    # Pool: train + 4/5 eval
    chain_hs = np.concatenate([train_hs, eval_hs[chain_eval_idx]], axis=0)
    chain_samples = train_samples + [eval_samples[i] for i in chain_eval_idx]

    chain_cache = HiddenStateCache(
        hidden_states=chain_hs,
        model_id=model_id,
        sample_ids=[s.id for s in chain_samples],
        token_position="last",
    )

    eval_full_labels = np.array([int(s.answer) for s in eval_samples])

    return chain_cache, chain_samples, eval_hs, eval_full_labels, eval_samples


def main():
    resources = ResourceManager(OUTPUT_ROOT)
    prober = LinearProbing()

    all_results = {}
    t_total = time.time()

    for mname, model_id in MODELS.items():
        all_results[mname] = {}
        print(f"\n{'='*60}")
        print(f"  {mname}: {model_id}")
        print(f"{'='*60}")

        for task in TASKS:
            print(f"\n  [{mname}] {task}")
            print(f"  {'─'*40}")

            # Build chain_train
            chain_cache, chain_samples, eval_hs, eval_labels, eval_samples = \
                build_chain_train(task, model_id)
            print(f"  chain_train: {chain_cache.n_samples} samples, "
                  f"{chain_cache.n_layers} layers, dim={chain_cache.hidden_dim}")

            # Artifact key (split-agnostic)
            artifact_key = ResourceKey(
                benchmark="cuebench",
                split="artifact",
                task=task,
                seed=CHAIN_SEED,
                model_id=model_id,
                method="linear_probing",
            )

            # Train
            t0 = time.time()
            artifact, probes = prober.train(
                resources=resources,
                train_samples=chain_samples,
                train_cache=chain_cache,
                artifact_key=artifact_key,
                probe_params=CHAIN_PROBE_PARAMS,
                answer_space=DIGIT_CLASSES,
                overwrite=True,
                probe_type="linear",
            )
            dt = time.time() - t0

            # Eval on full eval set (810)
            n_layers = chain_cache.n_layers
            per_layer_eval = {}
            best_eval_acc = 0
            best_layer = 0

            for li in range(n_layers):
                probe = probes[li]
                preds = probe.predict(eval_hs[:, li, :])
                acc = float(np.mean(preds == eval_labels))
                per_layer_eval[li] = acc
                if acc > best_eval_acc:
                    best_eval_acc = acc
                    best_layer = li

            # Also get train metrics from artifact
            train_metrics = artifact.fit_metrics.get("per_layer", {})
            best_train_acc = train_metrics.get(str(best_layer), {}).get("train_accuracy", 0)

            all_results[mname][task] = {
                "eval_acc": best_eval_acc,
                "best_layer": best_layer,
                "train_acc": best_train_acc,
                "n_chain_train": chain_cache.n_samples,
                "time_s": round(dt, 1),
            }

            gap = best_train_acc - best_eval_acc
            print(f"  → eval={best_eval_acc:.1%} train={best_train_acc:.1%} "
                  f"gap={gap:+.1%} @L{best_layer} ({dt:.1f}s)")

        # Free GPU memory between models
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n\n{'='*90}")
    print(f"SUMMARY: chain_train probing (n=3888 pooled, wd=0.05, linear)")
    print(f"{'='*90}")

    header = f"{'Model':>12s}"
    for t in TASKS:
        header += f" {t[:10]:>10s}"
    header += f" {'avg':>8s}"
    print(header)
    print("─" * len(header))

    for mname in MODELS:
        r = all_results[mname]
        accs = [r[t]["eval_acc"] for t in TASKS]
        row = f"{mname:>12s}"
        for t in TASKS:
            a = r[t]["eval_acc"]
            l = r[t]["best_layer"]
            row += f" {a:>6.0%}@L{l:<2d}"
        row += f" {np.mean(accs):>7.1%}"
        print(row)

    print(f"\nTotal time: {time.time() - t_total:.0f}s")

    # Save summary
    summary_path = OUTPUT_ROOT / "probe_experiments" / "chain_train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
