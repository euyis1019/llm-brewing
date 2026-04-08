#!/usr/bin/env python3
"""App C: Train probes for Coder-7B × computing with per-epoch metrics.

Produces per-epoch train/test accuracy and loss for all layers,
suitable for Fig C.1 (representative layer curves) and Fig C.2 (heatmap).

Uses the same chain_train pooling as train_probes_chain.py but records
per-epoch metrics instead of only final metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from brewing.schema import HiddenStateCache, Sample
from brewing.methods.linear_probing import (
    LinearProbe, _make_probe, _encode_labels, _get_probe_device, DIGIT_CLASSES,
)

# ── Config ──
OUTPUT_ROOT = Path("brewing_output")
DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
CACHE_ROOT = OUTPUT_ROOT / "caches" / "cuebench"

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
TASK = "computing"
SAFE_MODEL_ID = MODEL_ID.replace("/", "__")

# Training params — match chain_train but NO early stopping (run all epochs for curves)
EPOCHS = 200
PROBE_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 0.05,
    "batch_size": 99999,  # full-batch
}
CHAIN_SEED = 42
EVAL_POOL_RATIO = 5 / 6


def load_samples(split: str, task: str) -> list[Sample]:
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


def build_chain_data():
    """Build chain_train + held-out eval."""
    train_hs = np.load(
        CACHE_ROOT / "train" / TASK / "seed42" / SAFE_MODEL_ID / "hidden_states.npz"
    )["hidden_states"]
    train_samples = load_samples("train", TASK)

    eval_hs = np.load(
        CACHE_ROOT / "eval" / TASK / "seed42" / SAFE_MODEL_ID / "hidden_states.npz"
    )["hidden_states"]
    eval_samples = load_samples("eval", TASK)

    n_eval = len(eval_samples)
    rng = np.random.RandomState(CHAIN_SEED)
    perm = rng.permutation(n_eval)
    split_idx = int(n_eval * EVAL_POOL_RATIO)

    chain_eval_idx = perm[:split_idx]
    holdout_idx = perm[split_idx:]

    chain_hs = np.concatenate([train_hs, eval_hs[chain_eval_idx]], axis=0)
    chain_samples = train_samples + [eval_samples[i] for i in chain_eval_idx]

    return chain_hs, chain_samples, eval_hs, eval_samples


def main():
    print(f"App C Probing: {MODEL_ID} × {TASK}")
    print(f"Epochs: {EPOCHS} (no early stopping — full curves)")

    chain_hs, chain_samples, eval_hs, eval_samples = build_chain_data()

    n_samples, n_layers, hidden_dim = chain_hs.shape
    print(f"chain_train: {n_samples} samples, {n_layers} layers, dim={hidden_dim}")

    labels = _encode_labels([s.answer for s in chain_samples], DIGIT_CLASSES)
    eval_labels = _encode_labels([s.answer for s in eval_samples], DIGIT_CLASSES)
    n_classes = len(DIGIT_CLASSES)
    device = _get_probe_device()
    print(f"Device: {device}")

    # Train/val split within chain_train (90/10)
    indices = np.arange(n_samples)
    try:
        tr_idx, val_idx = train_test_split(
            indices, test_size=0.1, random_state=42, stratify=labels,
        )
    except ValueError:
        tr_idx, val_idx = train_test_split(
            indices, test_size=0.1, random_state=42,
        )

    y_all = torch.from_numpy(labels).long().to(device)
    y_eval = torch.from_numpy(eval_labels).long().to(device)

    lr = PROBE_PARAMS["lr"]
    wd = PROBE_PARAMS["weight_decay"]

    # per_epoch_metrics[layer][epoch] = {train_acc, val_acc, eval_acc, train_loss, val_loss}
    per_epoch_metrics = {}

    t0 = time.time()

    for layer_idx in tqdm(range(n_layers), desc="Layers", unit="layer"):
        X_np = chain_hs[:, layer_idx, :]
        X_eval_np = eval_hs[:, layer_idx, :]

        # Standardize on train split
        mean = X_np[tr_idx].mean(axis=0)
        std = X_np[tr_idx].std(axis=0) + 1e-8
        X_scaled = (X_np - mean) / std
        X_eval_scaled = (X_eval_np - mean) / std

        X_tr = torch.from_numpy(X_scaled[tr_idx]).float().to(device)
        y_tr = y_all[tr_idx]
        X_val = torch.from_numpy(X_scaled[val_idx]).float().to(device)
        y_val = y_all[val_idx]
        X_ev = torch.from_numpy(X_eval_scaled).float().to(device)

        probe = _make_probe("linear", hidden_dim, n_classes)
        probe.model.to(device)
        optimizer = torch.optim.Adam(probe.model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()

        epoch_records = []

        for epoch in range(EPOCHS):
            # Train step
            probe.model.train()
            optimizer.zero_grad()
            train_logits = probe.model(X_tr)
            train_loss = loss_fn(train_logits, y_tr)
            train_loss.backward()
            optimizer.step()

            # Eval (no grad)
            probe.model.eval()
            with torch.no_grad():
                # Train metrics
                tr_preds = train_logits.argmax(dim=1)
                tr_acc = float((tr_preds == y_tr).float().mean())
                tr_loss = float(train_loss)

                # Val metrics
                val_logits = probe.model(X_val)
                val_preds = val_logits.argmax(dim=1)
                val_acc = float((val_preds == y_val).float().mean())
                val_loss = float(loss_fn(val_logits, y_val))

                # Full eval set metrics
                ev_logits = probe.model(X_ev)
                ev_preds = ev_logits.argmax(dim=1)
                ev_acc = float((ev_preds == y_eval).float().mean())

            epoch_records.append({
                "train_acc": round(tr_acc, 4),
                "val_acc": round(val_acc, 4),
                "eval_acc": round(ev_acc, 4),
                "train_loss": round(tr_loss, 4),
                "val_loss": round(val_loss, 4),
            })

        per_epoch_metrics[layer_idx] = epoch_records

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Print summary: best eval acc per layer
    print(f"\n{'Layer':>6s} {'BestEval':>9s} {'@Epoch':>7s} {'FinalEval':>10s}")
    print("─" * 35)
    for li in range(n_layers):
        records = per_epoch_metrics[li]
        best_ep = max(range(EPOCHS), key=lambda e: records[e]["eval_acc"])
        best_acc = records[best_ep]["eval_acc"]
        final_acc = records[-1]["eval_acc"]
        print(f"  L{li:<3d} {best_acc:>8.1%}  ep={best_ep:<4d} {final_acc:>9.1%}")

    # Save
    out_dir = OUTPUT_ROOT / "probe_experiments" / "appc_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SAFE_MODEL_ID}__{TASK}.json"

    result = {
        "model_id": MODEL_ID,
        "task": TASK,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "epochs": EPOCHS,
        "n_chain_train": n_samples,
        "n_eval": len(eval_samples),
        "probe_params": PROBE_PARAMS,
        "elapsed_s": round(elapsed, 1),
        "per_epoch_metrics": {str(k): v for k, v in per_epoch_metrics.items()},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
