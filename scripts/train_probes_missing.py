#!/usr/bin/env python3
"""Train chain probes for models that are missing probe artifacts.

Same logic as train_probes_chain.py but only runs for specified models.
No GPU needed — purely sklearn on pre-computed caches.
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

OUTPUT_ROOT = Path("brewing_output")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
CACHE_ROOT = OUTPUT_ROOT / "caches" / "cuebench"

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]

# Only train these models
MODELS_TO_TRAIN = {
    "Llama2-7B":    "meta-llama/Llama-2-7b-hf",
    "CodeLlama-7B": "codellama/CodeLlama-7b-hf",
}

CHAIN_PROBE_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 0.05,
    "epochs": 1000,
    "patience": 35,
    "batch_size": 99999,
}
EVAL_POOL_RATIO = 5 / 6
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


def build_chain_train(task, model_id):
    model_safe = safe_model_id(model_id)

    # Load train cache
    train_dir = CACHE_ROOT / "train" / task / f"seed{CHAIN_SEED}" / model_safe
    train_hs = np.load(train_dir / "hidden_states.npz")
    train_hs = train_hs[list(train_hs.keys())[0]]
    with open(train_dir / "meta.json") as f:
        train_meta = json.load(f)
    train_samples = load_samples_from_json("train", task)
    train_by_id = {s.id: s for s in train_samples}
    train_samples = [train_by_id[sid] for sid in train_meta["sample_ids"]]

    # Load eval cache
    eval_dir = CACHE_ROOT / "eval" / task / f"seed{CHAIN_SEED}" / model_safe
    eval_hs = np.load(eval_dir / "hidden_states.npz")
    eval_hs = eval_hs[list(eval_hs.keys())[0]]
    with open(eval_dir / "meta.json") as f:
        eval_meta = json.load(f)
    eval_samples_all = load_samples_from_json("eval", task)
    eval_by_id = {s.id: s for s in eval_samples_all}
    eval_samples = [eval_by_id[sid] for sid in eval_meta["sample_ids"]]

    # Pool: train + 5/6 eval → chain_train
    n_eval = len(eval_samples)
    n_pool = int(n_eval * EVAL_POOL_RATIO)

    rng = np.random.RandomState(CHAIN_SEED)
    perm = rng.permutation(n_eval)
    pool_idx = perm[:n_pool]

    chain_hs = np.concatenate([train_hs, eval_hs[pool_idx]], axis=0)
    chain_samples = train_samples + [eval_samples[i] for i in pool_idx]

    chain_cache = HiddenStateCache(
        model_id=model_id,
        hidden_states=chain_hs,
        sample_ids=[s.id for s in chain_samples],
        model_predictions=[""] * len(chain_samples),
    )

    # Eval labels for reporting
    eval_labels = np.array([int(s.answer) for s in eval_samples])

    return chain_cache, chain_samples, eval_hs, eval_labels, eval_samples


def main():
    resources = ResourceManager(OUTPUT_ROOT)
    prober = LinearProbing()

    for mname, model_id in MODELS_TO_TRAIN.items():
        print(f"\n{'='*60}")
        print(f"  Training probes: {mname} ({model_id})")
        print(f"{'='*60}")

        for task in TASKS:
            # Check if already exists
            artifact_key = ResourceKey(
                benchmark="cuebench", split="artifact", task=task,
                seed=CHAIN_SEED, model_id=model_id, method="linear_probing",
            )
            existing = resources.resolve_artifact(artifact_key)
            if existing is not None:
                print(f"  [{mname}] {task}: already exists, skipping")
                continue

            print(f"\n  [{mname}] {task}")
            chain_cache, chain_samples, eval_hs, eval_labels, eval_samples = \
                build_chain_train(task, model_id)
            print(f"  chain_train: {chain_cache.n_samples} samples, "
                  f"{chain_cache.n_layers} layers, dim={chain_cache.hidden_dim}")

            t0 = time.time()
            artifact, probes = prober.train(
                resources=resources,
                train_samples=chain_samples,
                train_cache=chain_cache,
                artifact_key=artifact_key,
                probe_params=CHAIN_PROBE_PARAMS,
                answer_space=DIGIT_CLASSES,
                overwrite=False,
                probe_type="linear",
            )
            dt = time.time() - t0

            # Quick eval
            n_layers = chain_cache.n_layers
            best_acc, best_layer = 0, 0
            for li in range(n_layers):
                preds = probes[li].predict(eval_hs[:, li, :])
                acc = float(np.mean(preds == eval_labels))
                if acc > best_acc:
                    best_acc = acc
                    best_layer = li

            print(f"  → eval={best_acc:.1%} @L{best_layer} ({dt:.1f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
