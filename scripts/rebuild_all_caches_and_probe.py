#!/usr/bin/env python3
"""Rebuild ALL caches using fixed cache_builder, then train+eval probes.

Uses the framework's own cache_builder (with the padding fix) in batch mode.
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

from brewing.cache_builder import build_hidden_cache
from brewing.schema import Sample, HiddenStateCache

DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
CACHE_ROOT = Path("brewing_output/caches/cuebench")

TASKS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
MODELS = {
    "1.5B":   ("Qwen/Qwen2.5-Coder-1.5B",   "/path/to/cue/models/Qwen/Qwen2.5-Coder-1.5B"),
    "3B":     ("Qwen/Qwen2.5-Coder-3B",      "/path/to/cue/models/Qwen/Qwen2.5-Coder-3B"),
    "7B":     ("Qwen/Qwen2.5-Coder-7B",      "/path/to/cue/models/Qwen/Qwen2.5-Coder-7B"),
}


def safe_model_id(mid: str) -> str:
    return mid.replace("/", "__")


def load_samples(split: str, task: str) -> list[Sample]:
    path = DATA_ROOT / split / f"{task}.json"
    with open(path) as f:
        raw = json.load(f)
    return [
        Sample(id=s["id"], benchmark="CUE-Bench", subset=task,
               prompt=s["prompt"], answer=s["answer"], metadata=s.get("metadata", {}))
        for s in raw
    ]


def rebuild_cache(model, tokenizer, model_id, split, task, batch_size=8, device="cuda:0"):
    """Rebuild one cache using the fixed cache_builder."""
    mk = safe_model_id(model_id)
    cache_dir = CACHE_ROOT / split / task / "seed42" / mk
    cache_path = cache_dir / "hidden_states.npz"
    meta_path = cache_dir / "meta.json"

    samples = load_samples(split, task)
    print(f"  Building {split}/{task} ({len(samples)} samples)...")

    cache = build_hidden_cache(
        model=model, tokenizer=tokenizer, samples=samples,
        model_id=model_id, batch_size=batch_size, device=device,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, hidden_states=cache.hidden_states)
    meta = {
        "model_id": cache.model_id,
        "sample_ids": cache.sample_ids,
        "token_position": cache.token_position,
        "model_predictions": cache.model_predictions,
        "metadata": cache.metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return cache.hidden_states


def train_eval_probes(train_hs, eval_hs, train_labels, eval_labels,
                      probe_type="linear", lr=1e-3, wd=0.1,
                      epochs=2000, patience=50, device="cuda:0"):
    """Train per-layer probes, return best eval acc and per-layer detail."""
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_labels)), test_size=0.1, random_state=42, stratify=train_labels)
    n_layers = train_hs.shape[1]
    dim = train_hs.shape[2]
    n_classes = 10
    best_eval = 0; best_layer = 0
    per_layer = {}

    for li in range(n_layers):
        X_all = train_hs[:, li, :]; X_ev = eval_hs[:, li, :]
        m = X_all[tr_idx].mean(0); s = X_all[tr_idx].std(0) + 1e-8
        X_all_s = (X_all - m) / s; X_ev_s = (X_ev - m) / s

        if probe_type == "mlp":
            probe = nn.Sequential(
                nn.Linear(dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(64, n_classes),
            ).to(device)
        else:
            probe = nn.Linear(dim, n_classes).to(device)

        opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(X_all_s[tr_idx]).float().to(device)
        yt = torch.from_numpy(train_labels[tr_idx]).long().to(device)
        Xv = torch.from_numpy(X_all_s[val_idx]).float().to(device)
        yv = torch.from_numpy(train_labels[val_idx]).long().to(device)

        best_vl = float("inf"); best_st = None; w = 0
        for ep in range(epochs):
            probe.train(); opt.zero_grad(); loss_fn(probe(Xt), yt).backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                vl = loss_fn(probe(Xv), yv).item()
            if vl < best_vl:
                best_vl = vl; best_st = {k: v.clone() for k, v in probe.state_dict().items()}; w = 0
            else:
                w += 1
                if w >= patience: break
        if best_st:
            probe.load_state_dict(best_st)
        probe.eval()
        with torch.no_grad():
            tr_acc = float((probe(torch.from_numpy(X_all_s).float().to(device)).argmax(1).cpu().numpy() == train_labels).mean())
            ev_acc = float((probe(torch.from_numpy(X_ev_s).float().to(device)).argmax(1).cpu().numpy() == eval_labels).mean())
        per_layer[li] = {"train": round(tr_acc, 4), "eval": round(ev_acc, 4)}
        if ev_acc > best_eval:
            best_eval = ev_acc; best_layer = li

    return best_eval, best_layer, per_layer


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    all_results = {}

    for mname, (mid, mpath) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Loading {mname}: {mid}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(mpath)
        mdl = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map=device)
        mdl.eval()

        mk = safe_model_id(mid)
        all_results[mname] = {}

        for task in TASKS:
            # Rebuild caches
            for split in ["train", "eval"]:
                rebuild_cache(mdl, tokenizer, mid, split, task, batch_size=8, device=device)

            # Load and probe
            train_hs = np.load(CACHE_ROOT / "train" / task / "seed42" / mk / "hidden_states.npz")["hidden_states"]
            eval_hs = np.load(CACHE_ROOT / "eval" / task / "seed42" / mk / "hidden_states.npz")["hidden_states"]

            with open(DATA_ROOT / "train" / f"{task}.json") as f:
                tl = np.array([int(s["answer"]) for s in json.load(f)])
            with open(DATA_ROOT / "eval" / f"{task}.json") as f:
                el = np.array([int(s["answer"]) for s in json.load(f)])

            best_acc, best_layer, per_layer = train_eval_probes(train_hs, eval_hs, tl, el, device=device)
            all_results[mname][task] = {"acc": best_acc, "layer": best_layer}
            print(f"  {task:20s}: {best_acc:.1%} @ L{best_layer}")

        del mdl; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Rebuilt caches (padding fix), linear probe (wd=0.1)")
    print(f"{'='*80}")
    header = f"{'Model':>8s}" + "".join(f"{t[:10]:>12s}" for t in TASKS) + f"{'avg':>10s}"
    print(header)
    print("-" * len(header))
    for mname in MODELS:
        accs = [all_results[mname][t]["acc"] for t in TASKS]
        row = f"{mname:>8s}" + "".join(f"{a:>11.1%}" for a in accs) + f"{np.mean(accs):>9.1%}"
        print(row)


if __name__ == "__main__":
    main()
