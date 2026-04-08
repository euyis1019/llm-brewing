#!/usr/bin/env python3
"""Rebuild datasets with more samples, rebuild caches, train and evaluate probes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""

This script handles the full pipeline:
1. Generate datasets with configurable samples_per_config
2. Split into train/eval (80/20, stratified by config)
3. Build hidden state caches using loaded models
4. Train probes and evaluate on eval set

Usage:
    python scripts/rebuild_and_probe.py --model 7B --task value_tracking --samples-per-config 1500
    python scripts/rebuild_and_probe.py --model all --task all --samples-per-config 1500
"""

import argparse
import json
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ──
OUTPUT_ROOT = Path("brewing_output")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
MODELS_DIR = Path("/path/to/cue/models/Qwen")

MODELS = {
    "1.5B": ("Qwen/Qwen2.5-Coder-1.5B", MODELS_DIR / "Qwen2.5-Coder-1.5B"),
    "1.5B-Inst": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", MODELS_DIR / "Qwen2.5-Coder-1.5B-Instruct"),
    "3B": ("Qwen/Qwen2.5-Coder-3B", MODELS_DIR / "Qwen2.5-Coder-3B"),
    "7B": ("Qwen/Qwen2.5-Coder-7B", MODELS_DIR / "Qwen2.5-Coder-7B"),
}

TASKS = {
    "value_tracking": "value_tracking",
    "computing": "computing",
    "conditional": "conditional",
    "function_call": "function_call",
    "loop": "loop",
    "loop_unrolled": "loop_unrolled",
}


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


# ── Step 1: Generate data ──

def generate_data(task: str, seed: int, samples_per_config: int) -> list[dict]:
    """Generate dataset for a task using the datagen modules."""
    import importlib
    mod = importlib.import_module(f"brewing.benchmarks.cue_bench.datagen.{task}")
    return mod.generate_dataset(seed=seed, samples_per_config=samples_per_config)


def split_data(data: list[dict], train_ratio: float = 0.8, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split data into train/eval, stratified by config (first 3 metadata dims)."""
    rng = random.Random(seed)

    # Group by config
    from collections import defaultdict
    groups = defaultdict(list)
    for sample in data:
        m = sample["metadata"]
        keys = sorted([k for k in m if k not in ("result_var", "sample_idx")])[:3]
        config = tuple(str(m[k]) for k in keys)
        groups[config].append(sample)

    train, eval_ = [], []
    for config, samples in groups.items():
        rng.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        train.extend(samples[:n_train])
        eval_.extend(samples[n_train:])

    return train, eval_


# ── Step 2: Build hidden state cache ──

def build_cache(
    model, tokenizer, samples: list[dict], model_id: str,
    batch_size: int = 8, device: str = "cuda:0",
) -> np.ndarray:
    """Extract last-token hidden states from all layers."""
    all_hidden = []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_start in tqdm(
        range(0, len(samples), batch_size),
        desc="Building cache",
        total=(len(samples) + batch_size - 1) // batch_size,
    ):
        batch = samples[batch_start:batch_start + batch_size]
        prompts = [s["prompt"] for s in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Skip embedding layer
        hidden_layers = outputs.hidden_states[1:]
        n_layers = len(hidden_layers)

        attention_mask = inputs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1

        batch_hs = np.zeros(
            (len(prompts), n_layers, hidden_layers[0].shape[-1]),
            dtype=np.float32,
        )
        for layer_idx, layer_states in enumerate(hidden_layers):
            for sample_idx in range(len(prompts)):
                pos = last_positions[sample_idx]
                batch_hs[sample_idx, layer_idx] = (
                    layer_states[sample_idx, pos].cpu().float().numpy()
                )

        all_hidden.append(batch_hs)

    return np.concatenate(all_hidden, axis=0)


# ── Step 3: Train and evaluate probes ──

def train_and_eval_probes(
    train_hs: np.ndarray, eval_hs: np.ndarray,
    train_labels: np.ndarray, eval_labels: np.ndarray,
    lr: float = 1e-3, weight_decay: float = 0.1,
    epochs: int = 2000, patience: int = 50, batch_size: int = 512,
    probe_type: str = "linear",
    device: str = "cuda:0",
) -> dict:
    """Train per-layer probes and return results."""
    n_layers = train_hs.shape[1]
    n_classes = 10

    # Split train into fit/validation
    try:
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=0.1,
            random_state=42, stratify=train_labels,
        )
    except ValueError:
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=0.1, random_state=42,
        )

    results = {"per_layer": {}}
    best_eval = 0
    best_layer = 0

    pbar = tqdm(range(n_layers), desc="Training probes", unit="layer")
    for li in pbar:
        X_all = train_hs[:, li, :]
        X_ev = eval_hs[:, li, :]

        # Standardize on training split
        mean = X_all[tr_idx].mean(axis=0)
        std = X_all[tr_idx].std(axis=0) + 1e-8
        X_all_s = (X_all - mean) / std
        X_ev_s = (X_ev - mean) / std

        dim = X_all.shape[1]
        if probe_type == "mlp":
            model = nn.Sequential(
                nn.Linear(dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(64, n_classes),
            ).to(device)
        else:
            model = nn.Linear(dim, n_classes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X_all_s[tr_idx]).float().to(device)
        yt = torch.from_numpy(train_labels[tr_idx]).long().to(device)
        Xv = torch.from_numpy(X_all_s[val_idx]).float().to(device)
        yv = torch.from_numpy(train_labels[val_idx]).long().to(device)

        n_tr = len(tr_idx)
        best_vl = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            model.train()
            if n_tr > batch_size:
                perm = torch.randperm(n_tr, device=device)
                for start in range(0, n_tr, batch_size):
                    idx = perm[start:start + batch_size]
                    optimizer.zero_grad()
                    loss_fn(model(Xt[idx]), yt[idx]).backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                loss_fn(model(Xt), yt).backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), yv).item()
            if vl < best_vl:
                best_vl = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # Train accuracy
            preds_tr = model(torch.from_numpy(X_all_s).float().to(device)).argmax(1).cpu().numpy()
            tr_acc = float(np.mean(preds_tr == train_labels))
            # Eval accuracy
            preds_ev = model(torch.from_numpy(X_ev_s).float().to(device)).argmax(1).cpu().numpy()
            ev_acc = float(np.mean(preds_ev == eval_labels))

        results["per_layer"][li] = {
            "train_acc": round(tr_acc, 4),
            "eval_acc": round(ev_acc, 4),
        }

        if ev_acc > best_eval:
            best_eval = ev_acc
            best_layer = li

        pbar.set_postfix_str(
            f"tr={tr_acc:.0%} ev={ev_acc:.0%} best={best_eval:.0%}@L{best_layer}"
        )

    results["best_layer"] = best_layer
    results["best_eval_acc"] = round(best_eval, 4)
    results["last_eval_acc"] = results["per_layer"][n_layers - 1]["eval_acc"]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="7B")
    parser.add_argument("--task", "-t", default="value_tracking")
    parser.add_argument("--samples-per-config", "-n", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache building, use existing")
    parser.add_argument("--no-save", action="store_true", help="Don't save caches to disk (keep in memory only)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = {"all": MODELS}.get(args.model, {args.model: MODELS[args.model]})
    tasks = TASKS.keys() if args.task == "all" else [args.task]

    for model_name, (model_id, model_path) in models.items():
        model_safe = safe_model_id(model_id)

        # Load model once for all tasks
        if not args.skip_cache:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"\nLoading model: {model_id} from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=torch.float16,
                device_map=args.device,
            )
            model.eval()
        else:
            model = tokenizer = None

        for task in tasks:
            print(f"\n{'='*60}")
            print(f"  {model_name} / {task} / n={args.samples_per_config}")
            print(f"{'='*60}")

            # Step 1: Generate data
            print("[1/4] Generating data...")
            data = generate_data(task, seed=args.seed, samples_per_config=args.samples_per_config)
            train_data, eval_data = split_data(data, seed=args.seed)
            train_labels = np.array([int(s["answer"]) for s in train_data])
            eval_labels = np.array([int(s["answer"]) for s in eval_data])
            print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
            print(f"  Label dist (train): {dict(Counter(train_labels))}")

            # Cache paths
            cache_dir = OUTPUT_ROOT / "caches_expanded" / f"n{args.samples_per_config}"
            train_cache_path = cache_dir / "train" / task / f"seed{args.seed}" / model_safe / "hidden_states.npz"
            eval_cache_path = cache_dir / "eval" / task / f"seed{args.seed}" / model_safe / "hidden_states.npz"

            if args.skip_cache and train_cache_path.exists() and eval_cache_path.exists():
                print("[2/4] Loading existing caches...")
                train_hs = np.load(train_cache_path)["hidden_states"]
                eval_hs = np.load(eval_cache_path)["hidden_states"]
            else:
                # Step 2: Build caches
                print("[2/4] Building train cache...")
                train_hs = build_cache(
                    model, tokenizer, train_data, model_id,
                    batch_size=args.batch_size, device=args.device,
                )
                if not args.no_save:
                    train_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(train_cache_path, hidden_states=train_hs)

                print("[3/4] Building eval cache...")
                eval_hs = build_cache(
                    model, tokenizer, eval_data, model_id,
                    batch_size=args.batch_size, device=args.device,
                )
                if not args.no_save:
                    eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(eval_cache_path, hidden_states=eval_hs)

            print(f"  Train cache: {train_hs.shape}")
            print(f"  Eval cache: {eval_hs.shape}")

            # Step 3: Train and evaluate probes
            print(f"[4/4] Training {args.probe_type} probes...")
            results = train_and_eval_probes(
                train_hs, eval_hs, train_labels, eval_labels,
                probe_type=args.probe_type, device=args.device,
            )

            print(f"\n  RESULT: best_eval={results['best_eval_acc']:.1%} @ L{results['best_layer']}")
            print(f"          last_layer={results['last_eval_acc']:.1%}")

            # Show last 10 layers detail
            n_layers = train_hs.shape[1]
            start = max(0, n_layers - 10)
            for li in range(start, n_layers):
                d = results["per_layer"][li]
                marker = " <-- best" if li == results["best_layer"] else ""
                print(f"    L{li:2d}: train={d['train_acc']:.1%} eval={d['eval_acc']:.1%}{marker}")

            # Save results
            result_path = OUTPUT_ROOT / "probe_experiments" / f"expanded_{model_name}_{task}_n{args.samples_per_config}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)

        # Free model
        if model is not None:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
