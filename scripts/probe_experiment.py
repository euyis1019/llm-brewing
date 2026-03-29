#!/usr/bin/env python3
"""Standalone probe experiment script.

Loads hidden-state caches directly (no model needed), trains probes
with various configurations, and reports eval accuracy.

Usage:
    python scripts/probe_experiment.py                    # run all
    python scripts/probe_experiment.py --model 7B         # one model
    python scripts/probe_experiment.py --task value_tracking --model 7B
    python scripts/probe_experiment.py --method pca_linear  # specific method
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Constants ──

CACHE_ROOT = Path("brewing_output/caches/cuebench")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
RESULTS_DIR = Path("brewing_output/probe_experiments")

MODELS = {
    "1.5B": "Qwen__Qwen2.5-Coder-1.5B",
    "1.5B-Instruct": "Qwen__Qwen2.5-Coder-1.5B-Instruct",
    "3B": "Qwen__Qwen2.5-Coder-3B",
    "7B": "Qwen__Qwen2.5-Coder-7B",
}

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]

DIGIT_CLASSES = [str(d) for d in range(10)]


def load_cache(split: str, task: str, model_key: str) -> np.ndarray:
    path = CACHE_ROOT / split / task / "seed42" / model_key / "hidden_states.npz"
    return np.load(path)["hidden_states"]


def load_labels(split: str, task: str) -> np.ndarray:
    path = DATA_ROOT / split / f"{task}.json"
    with open(path) as f:
        data = json.load(f)
    space_map = {str(d): d for d in range(10)}
    return np.array([space_map.get(s["answer"], 10) for s in data])


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Probe training methods ──

def train_probe_linear(X_train, y_train, X_val, y_val, hidden_dim, n_classes,
                       lr=1e-3, epochs=2000, batch_size=512, weight_decay=0.01,
                       patience=50, device=None):
    """Standard linear probe (current baseline)."""
    if device is None:
        device = get_device()

    model = nn.Linear(hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    n_tr = len(X_tr)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        if n_tr > batch_size:
            perm = torch.randperm(n_tr, device=device)
            for start in range(0, n_tr, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss_fn(model(X_tr), y_tr).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_probe_pca_linear(X_train, y_train, X_val, y_val, hidden_dim, n_classes,
                           n_components=256, lr=1e-3, epochs=2000, batch_size=512,
                           weight_decay=0.01, patience=50, device=None):
    """PCA dimensionality reduction + linear probe."""
    if device is None:
        device = get_device()

    pca = PCA(n_components=min(n_components, X_train.shape[0], X_train.shape[1]))
    X_tr_pca = pca.fit_transform(X_train)
    X_v_pca = pca.transform(X_val)
    d = X_tr_pca.shape[1]

    model = nn.Linear(d, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_tr_pca).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_v_pca).float().to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    n_tr = len(X_tr)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        if n_tr > batch_size:
            perm = torch.randperm(n_tr, device=device)
            for start in range(0, n_tr, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss_fn(model(X_tr), y_tr).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, pca


def train_probe_pca_mlp(X_train, y_train, X_val, y_val, hidden_dim, n_classes,
                        n_components=256, mlp_hidden=64, lr=1e-3, epochs=3000,
                        batch_size=512, weight_decay=0.01, patience=80,
                        dropout=0.1, device=None):
    """PCA + small MLP (bottleneck) probe."""
    if device is None:
        device = get_device()

    pca = PCA(n_components=min(n_components, X_train.shape[0], X_train.shape[1]))
    X_tr_pca = pca.fit_transform(X_train)
    X_v_pca = pca.transform(X_val)
    d = X_tr_pca.shape[1]

    model = nn.Sequential(
        nn.Linear(d, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, n_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_tr_pca).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_v_pca).float().to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    n_tr = len(X_tr)
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        if n_tr > batch_size:
            perm = torch.randperm(n_tr, device=device)
            for start in range(0, n_tr, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss_fn(model(X_tr), y_tr).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, pca


def evaluate(model, X, y, pca=None, device=None):
    """Compute accuracy."""
    if device is None:
        device = get_device()
    if pca is not None:
        X = pca.transform(X)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(preds == y))


# ── Main experiment runner ──

def run_experiment(model_name, model_key, task, method="baseline",
                   pca_components=256, mlp_hidden=64, device=None):
    """Run one model × task × method experiment, return per-layer eval accuracy."""
    if device is None:
        device = get_device()

    # Load data
    train_hs = load_cache("train", task, model_key)  # (N_tr, L, D)
    eval_hs = load_cache("eval", task, model_key)    # (N_ev, L, D)
    train_labels = load_labels("train", task)
    eval_labels = load_labels("eval", task)

    n_layers = train_hs.shape[1]
    hidden_dim = train_hs.shape[2]
    n_classes = 11  # 0-9 + residual

    # Split train into train/val for early stopping (90/10)
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
    best_eval_acc = 0
    best_layer = 0

    pbar = tqdm(range(n_layers), desc=f"{model_name}/{task}/{method}", unit="layer")
    for layer_idx in pbar:
        X_all_train = train_hs[:, layer_idx, :]
        X_eval = eval_hs[:, layer_idx, :]

        # Standardize
        mean = X_all_train[tr_idx].mean(axis=0)
        std = X_all_train[tr_idx].std(axis=0) + 1e-8
        X_all_train_s = (X_all_train - mean) / std
        X_eval_s = (X_eval - mean) / std

        X_tr = X_all_train_s[tr_idx]
        X_val = X_all_train_s[val_idx]
        y_tr = train_labels[tr_idx]
        y_val = train_labels[val_idx]

        pca = None
        if method == "baseline":
            model = train_probe_linear(
                X_tr, y_tr, X_val, y_val, hidden_dim, n_classes, device=device,
            )
        elif method == "pca_linear":
            model, pca = train_probe_pca_linear(
                X_tr, y_tr, X_val, y_val, hidden_dim, n_classes,
                n_components=pca_components, device=device,
            )
        elif method == "pca_mlp":
            model, pca = train_probe_pca_mlp(
                X_tr, y_tr, X_val, y_val, hidden_dim, n_classes,
                n_components=pca_components, mlp_hidden=mlp_hidden, device=device,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Eval accuracy
        eval_acc = evaluate(model, X_eval_s, eval_labels, pca=pca, device=device)
        train_acc = evaluate(model, X_all_train_s, train_labels, pca=pca, device=device)

        results["per_layer"][layer_idx] = {
            "train_acc": round(train_acc, 4),
            "eval_acc": round(eval_acc, 4),
        }

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_layer = layer_idx

        pbar.set_postfix_str(f"eval={eval_acc:.1%} best={best_eval_acc:.1%}@L{best_layer}")

    results["best_layer"] = best_layer
    results["best_eval_acc"] = round(best_eval_acc, 4)
    results["last_layer_eval_acc"] = results["per_layer"][n_layers - 1]["eval_acc"]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default=None, help="Model shortname (1.5B, 3B, 7B, etc.)")
    parser.add_argument("--task", "-t", default=None, help="Task name")
    parser.add_argument("--method", default=None,
                        help="Probe method: baseline, pca_linear, pca_mlp, or 'all'")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    args = parser.parse_args()

    models = {args.model: MODELS[args.model]} if args.model else MODELS
    tasks = [args.task] if args.task else TASKS
    methods = (
        ["baseline", "pca_linear", "pca_mlp"]
        if args.method in (None, "all")
        else [args.method]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results = {}

    for method in methods:
        for model_name, model_key in models.items():
            for task in tasks:
                key = f"{model_name}/{task}/{method}"
                print(f"\n{'='*60}")
                print(f"  {key}")
                print(f"{'='*60}")

                t0 = time.time()
                result = run_experiment(
                    model_name, model_key, task, method=method,
                    pca_components=args.pca_components,
                    mlp_hidden=args.mlp_hidden,
                    device=device,
                )
                result["time_s"] = round(time.time() - t0, 1)
                all_results[key] = result

                print(f"  Best eval acc: {result['best_eval_acc']:.1%} @ layer {result['best_layer']}")
                print(f"  Last layer:    {result['last_layer_eval_acc']:.1%}")
                print(f"  Time: {result['time_s']}s")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Best eval accuracy per model × task × method")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Task':<18} {'Method':<12} {'Best Acc':>8} {'Layer':>5} {'Last':>8}")
    print("-" * 80)
    for key, r in all_results.items():
        parts = key.split("/")
        print(f"{parts[0]:<20} {parts[1]:<18} {parts[2]:<12} "
              f"{r['best_eval_acc']:>7.1%} {r['best_layer']:>5} {r['last_layer_eval_acc']:>7.1%}")

    # Save results
    out_path = RESULTS_DIR / f"results_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
