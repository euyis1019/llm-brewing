#!/usr/bin/env python3
"""Probe experiment v2: focused on finding optimal probe config.

Key insight from v1: PCA destroys discriminative signal. Need heavy
regularization on the full-dimensional linear probe, or a supervised
bottleneck (trained end-to-end).

Strategies:
  A. Linear probe + heavy L2 (weight_decay sweep)
  B. Bottleneck MLP: Linear(D→k) → ReLU → Linear(k→C), small k
  C. Input dropout + linear
  D. Combination: dropout + bottleneck + L2
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CACHE_ROOT = Path("brewing_output/caches/cuebench")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
RESULTS_DIR = Path("brewing_output/probe_experiments")


MODELS = {
    "1.5B": ("Qwen__Qwen2.5-Coder-1.5B", 28, 1536),
    "1.5B-Inst": ("Qwen__Qwen2.5-Coder-1.5B-Instruct", 28, 1536),
    "3B": ("Qwen__Qwen2.5-Coder-3B", 36, 2048),
    "7B": ("Qwen__Qwen2.5-Coder-7B", 28, 3584),
}

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]


def load_cache(split, task, model_key):
    path = CACHE_ROOT / split / task / "seed42" / model_key / "hidden_states.npz"
    return np.load(path)["hidden_states"]


def load_labels(split, task):
    path = DATA_ROOT / split / f"{task}.json"
    with open(path) as f:
        data = json.load(f)
    space_map = {str(d): d for d in range(10)}
    return np.array([space_map.get(s["answer"], 10) for s in data])


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class ProbeConfig:
    name: str
    probe_type: str  # "linear", "bottleneck", "dropout_linear", "dropout_bottleneck"
    weight_decay: float = 0.01
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 512
    patience: int = 50
    bottleneck_dim: int = 32
    dropout: float = 0.0
    label_smoothing: float = 0.0


def build_probe(config: ProbeConfig, in_features: int, n_classes: int):
    layers = []
    if config.dropout > 0:
        layers.append(nn.Dropout(config.dropout))
    if config.probe_type in ("bottleneck", "dropout_bottleneck"):
        layers.extend([
            nn.Linear(in_features, config.bottleneck_dim),
            nn.BatchNorm1d(config.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(config.bottleneck_dim, n_classes),
        ])
    else:
        layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


def train_and_eval_one_layer(
    X_train_all, y_train_all, X_eval, y_eval,
    tr_idx, val_idx, config: ProbeConfig, device,
):
    """Train probe on one layer, return (train_acc, eval_acc, stopped_epoch)."""
    hidden_dim = X_train_all.shape[1]
    n_classes = 11

    # Standardize
    mean = X_train_all[tr_idx].mean(axis=0)
    std = X_train_all[tr_idx].std(axis=0) + 1e-8
    X_train_s = (X_train_all - mean) / std
    X_eval_s = (X_eval - mean) / std

    X_tr = torch.from_numpy(X_train_s[tr_idx]).float().to(device)
    y_tr = torch.from_numpy(y_train_all[tr_idx]).long().to(device)
    X_v = torch.from_numpy(X_train_s[val_idx]).float().to(device)
    y_v = torch.from_numpy(y_train_all[val_idx]).long().to(device)
    X_ev = torch.from_numpy(X_eval_s).float().to(device)
    y_ev = torch.from_numpy(y_eval).long().to(device)

    model = build_probe(config, hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    n_tr = len(tr_idx)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    stopped_epoch = config.epochs

    for epoch in range(config.epochs):
        model.train()
        if n_tr > config.batch_size:
            perm = torch.randperm(n_tr, device=device)
            for start in range(0, n_tr, config.batch_size):
                idx = perm[start:start + config.batch_size]
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
            if wait >= config.patience:
                stopped_epoch = epoch + 1
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        # Full train accuracy
        X_all = torch.from_numpy(X_train_s).float().to(device)
        train_acc = float((model(X_all).argmax(1) == torch.from_numpy(y_train_all).long().to(device)).float().mean())
        # Eval accuracy
        eval_acc = float((model(X_ev).argmax(1) == y_ev).float().mean())

    return train_acc, eval_acc, stopped_epoch


def run_one_experiment(model_name, task, config: ProbeConfig, device):
    """Run one model × task × config, return per-layer results."""
    model_key = MODELS[model_name][0]

    train_hs = load_cache("train", task, model_key)
    eval_hs = load_cache("eval", task, model_key)
    train_labels = load_labels("train", task)
    eval_labels = load_labels("eval", task)

    n_layers = train_hs.shape[1]

    try:
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=0.1,
            random_state=42, stratify=train_labels,
        )
    except ValueError:
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=0.1, random_state=42,
        )

    per_layer = {}
    best_eval = 0
    best_layer = 0

    pbar = tqdm(range(n_layers), desc=f"{config.name}", unit="L", leave=False)
    for li in pbar:
        X_tr_l = train_hs[:, li, :]
        X_ev_l = eval_hs[:, li, :]

        tr_acc, ev_acc, ep = train_and_eval_one_layer(
            X_tr_l, train_labels, X_ev_l, eval_labels,
            tr_idx, val_idx, config, device,
        )

        per_layer[li] = {"train": round(tr_acc, 4), "eval": round(ev_acc, 4), "epoch": ep}
        if ev_acc > best_eval:
            best_eval = ev_acc
            best_layer = li
        pbar.set_postfix_str(f"tr={tr_acc:.0%} ev={ev_acc:.0%} best={best_eval:.0%}@L{best_layer}")

    return {
        "per_layer": per_layer,
        "best_layer": best_layer,
        "best_eval": round(best_eval, 4),
        "last_eval": per_layer[n_layers - 1]["eval"],
    }


# ── Experiment configs ──

CONFIGS = {
    # A: Linear + weight_decay sweep
    "linear_wd0.01": ProbeConfig("linear_wd0.01", "linear", weight_decay=0.01),
    "linear_wd0.1": ProbeConfig("linear_wd0.1", "linear", weight_decay=0.1),
    "linear_wd1.0": ProbeConfig("linear_wd1.0", "linear", weight_decay=1.0),
    "linear_wd10": ProbeConfig("linear_wd10", "linear", weight_decay=10.0),

    # B: Bottleneck MLP
    "bn16_wd0.1": ProbeConfig("bn16_wd0.1", "bottleneck", bottleneck_dim=16, weight_decay=0.1),
    "bn32_wd0.1": ProbeConfig("bn32_wd0.1", "bottleneck", bottleneck_dim=32, weight_decay=0.1),
    "bn64_wd0.1": ProbeConfig("bn64_wd0.1", "bottleneck", bottleneck_dim=64, weight_decay=0.1),
    "bn128_wd0.1": ProbeConfig("bn128_wd0.1", "bottleneck", bottleneck_dim=128, weight_decay=0.1),

    # C: Dropout + linear
    "drop0.3_linear": ProbeConfig("drop0.3_linear", "dropout_linear", dropout=0.3, weight_decay=0.1),
    "drop0.5_linear": ProbeConfig("drop0.5_linear", "dropout_linear", dropout=0.5, weight_decay=0.1),
    "drop0.7_linear": ProbeConfig("drop0.7_linear", "dropout_linear", dropout=0.7, weight_decay=0.1),

    # D: Dropout + bottleneck
    "drop0.3_bn32": ProbeConfig("drop0.3_bn32", "dropout_bottleneck", dropout=0.3, bottleneck_dim=32, weight_decay=0.1),
    "drop0.5_bn64": ProbeConfig("drop0.5_bn64", "dropout_bottleneck", dropout=0.5, bottleneck_dim=64, weight_decay=0.1),

    # E: Label smoothing
    "linear_ls0.1": ProbeConfig("linear_ls0.1", "linear", weight_decay=0.1, label_smoothing=0.1),

    # F: Lower LR
    "linear_lr1e-4_wd0.1": ProbeConfig("linear_lr1e-4_wd0.1", "linear", lr=1e-4, weight_decay=0.1, epochs=5000, patience=100),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="7B")
    parser.add_argument("--task", "-t", default="value_tracking")
    parser.add_argument("--configs", "-c", default=None, help="Comma-separated config names, or 'all'")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.configs and args.configs != "all":
        config_names = args.configs.split(",")
    else:
        config_names = list(CONFIGS.keys())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for cname in config_names:
        config = CONFIGS[cname]
        print(f"\n{'='*60}")
        print(f"  {args.model}/{args.task} — {cname}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_one_experiment(args.model, args.task, config, device)
        result["time_s"] = round(time.time() - t0, 1)
        all_results[cname] = result
        print(f"  Best eval: {result['best_eval']:.1%} @ L{result['best_layer']}, "
              f"last: {result['last_eval']:.1%}, time: {result['time_s']}s")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: {args.model}/{args.task}")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Best Eval':>10} {'Layer':>6} {'Last':>10} {'Time':>8}")
    print("-" * 70)
    for cname in sorted(all_results, key=lambda k: -all_results[k]["best_eval"]):
        r = all_results[cname]
        print(f"{cname:<25} {r['best_eval']:>9.1%} {r['best_layer']:>6} "
              f"{r['last_eval']:>9.1%} {r['time_s']:>7.0f}s")

    out_path = RESULTS_DIR / f"v2_{args.model}_{args.task}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
