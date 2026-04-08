#!/usr/bin/env python3
"""Generate expanded training data, rebuild caches, and train probes.

Generates more samples_per_config for training, keeps original eval set.
Uses batch_size=64 for fast cache building.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
import time
import importlib
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from brewing.cache_builder import build_hidden_cache
from brewing.schema import Sample

DATA_ROOT = Path("brewing/benchmarks/cue_bench/data")
CACHE_ROOT = Path("brewing_output/caches/cuebench")
EXPANDED_CACHE = Path("brewing_output/caches_expanded_v2")

TASKS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
MODELS = {
    "1.5B": ("Qwen/Qwen2.5-Coder-1.5B", "/path/to/cue/models/Qwen/Qwen2.5-Coder-1.5B"),
    "3B":   ("Qwen/Qwen2.5-Coder-3B",   "/path/to/cue/models/Qwen/Qwen2.5-Coder-3B"),
    "7B":   ("Qwen/Qwen2.5-Coder-7B",   "/path/to/cue/models/Qwen/Qwen2.5-Coder-7B"),
}

SAMPLES_PER_CONFIG = 500  # up from 150
BATCH_SIZE = 64
DEVICE = "cuda:0"


def safe_model_id(mid: str) -> str:
    return mid.replace("/", "__")


def generate_train_data(task: str, seed: int = 42, samples_per_config: int = 500):
    """Generate training data with more samples, split 80/20."""
    mod = importlib.import_module(f"brewing.benchmarks.cue_bench.datagen.{task}")
    all_data = mod.generate_dataset(seed=seed, samples_per_config=samples_per_config)

    # Split 80/20 stratified by config
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in all_data:
        m = s["metadata"]
        keys = sorted([k for k in m if k not in ("result_var", "sample_idx")])[:3]
        cfg = tuple(str(m[k]) for k in keys)
        groups[cfg].append(s)

    train, eval_ = [], []
    for cfg, samples in groups.items():
        rng.shuffle(samples)
        n_train = int(len(samples) * 0.8)
        train.extend(samples[:n_train])
        eval_.extend(samples[n_train:])

    return train, eval_


def samples_to_brewing(data: list[dict], task: str) -> list[Sample]:
    return [
        Sample(id=s["id"], benchmark="CUE-Bench", subset=task,
               prompt=s["prompt"], answer=s["answer"], metadata=s.get("metadata", {}))
        for s in data
    ]


def train_probes(train_hs, eval_hs, tl, el, probe_type="linear",
                 lr=1e-3, wd=0.1, epochs=1000, patience=50, device=DEVICE):
    tr_idx, val_idx = train_test_split(
        np.arange(len(tl)), test_size=0.1, random_state=42, stratify=tl)
    n_layers = train_hs.shape[1]; dim = train_hs.shape[2]
    best_eval = 0; best_layer = 0; best_train = 0

    for li in range(n_layers):
        X_all = train_hs[:, li, :]; X_ev = eval_hs[:, li, :]
        m = X_all[tr_idx].mean(0); s = X_all[tr_idx].std(0) + 1e-8
        X_all_s = (X_all - m) / s; X_ev_s = (X_ev - m) / s

        if probe_type == "mlp":
            probe = nn.Sequential(
                nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 10),
            ).to(device)
        else:
            probe = nn.Linear(dim, 10).to(device)

        opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(X_all_s[tr_idx]).float().to(device)
        yt = torch.from_numpy(tl[tr_idx]).long().to(device)
        Xv = torch.from_numpy(X_all_s[val_idx]).float().to(device)
        yv = torch.from_numpy(tl[val_idx]).long().to(device)

        best_vl = float("inf"); best_st = None; w = 0
        for ep in range(epochs):
            probe.train(); opt.zero_grad(); loss_fn(probe(Xt), yt).backward(); opt.step()
            probe.eval()
            with torch.no_grad(): vl = loss_fn(probe(Xv), yv).item()
            if vl < best_vl:
                best_vl = vl; best_st = {k: v.clone() for k, v in probe.state_dict().items()}; w = 0
            else:
                w += 1
                if w >= patience: break
        if best_st: probe.load_state_dict(best_st)
        probe.eval()
        with torch.no_grad():
            tr_acc = float((probe(torch.from_numpy(X_all_s).float().to(device)).argmax(1).cpu().numpy() == tl).mean())
            ev_acc = float((probe(torch.from_numpy(X_ev_s).float().to(device)).argmax(1).cpu().numpy() == el).mean())
        if ev_acc > best_eval:
            best_eval = ev_acc; best_layer = li; best_train = tr_acc

    return best_eval, best_layer, best_train


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {}

    for mname, (mid, mpath) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {mname}: {mid}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(mpath)
        mdl = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map=DEVICE)
        mdl.eval()
        mk = safe_model_id(mid)

        all_results[mname] = {}
        for task in TASKS:
            cache_dir = EXPANDED_CACHE / f"n{SAMPLES_PER_CONFIG}" / "train" / task / "seed42" / mk
            cache_path = cache_dir / "hidden_states.npz"

            if cache_path.exists():
                print(f"  {task}: cache exists, loading...")
                train_hs = np.load(cache_path)["hidden_states"]
                with open(cache_dir / "labels.json") as f:
                    tl = np.array(json.load(f))
            else:
                print(f"  {task}: generating {SAMPLES_PER_CONFIG} samples/config...")
                train_data, _ = generate_train_data(task, samples_per_config=SAMPLES_PER_CONFIG)
                tl = np.array([int(s["answer"]) for s in train_data])
                print(f"    train: {len(train_data)} samples")

                train_samples = samples_to_brewing(train_data, task)
                cache = build_hidden_cache(
                    model=mdl, tokenizer=tokenizer, samples=train_samples,
                    model_id=mid, batch_size=BATCH_SIZE, device=DEVICE,
                )
                train_hs = cache.hidden_states
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, hidden_states=train_hs)
                with open(cache_dir / "labels.json", "w") as f:
                    json.dump(tl.tolist(), f)

            # Eval: use the ORIGINAL eval set (already rebuilt with padding fix)
            eval_hs = np.load(CACHE_ROOT / "eval" / task / "seed42" / mk / "hidden_states.npz")["hidden_states"]
            with open(DATA_ROOT / "eval" / f"{task}.json") as f:
                el = np.array([int(s["answer"]) for s in json.load(f)])

            # Train probes
            ev_acc, bl, tr_acc = train_probes(train_hs, eval_hs, tl, el, probe_type="linear")
            gap = tr_acc - ev_acc
            all_results[mname][task] = {"eval": ev_acc, "train": tr_acc, "layer": bl, "gap": gap,
                                        "n_train": len(tl)}
            print(f"    linear: eval={ev_acc:.1%} train={tr_acc:.1%} gap={gap:+.1%} @L{bl} (n={len(tl)})")

        del mdl; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: expanded data (n={SAMPLES_PER_CONFIG}/config), linear probe")
    print(f"{'='*80}")
    print(f"{'Model':>6s} {'n_train':>7s}", end="")
    for t in TASKS: print(f" {t[:8]:>10s}", end="")
    print(f" {'avg':>8s}")
    print("-" * 90)

    for mname in MODELS:
        r = all_results[mname]
        n = r[TASKS[0]]["n_train"]
        accs = [r[t]["eval"] for t in TASKS]
        gaps = [r[t]["gap"] for t in TASKS]
        print(f"{mname:>6s} {n:>7d}", end="")
        for a, g in zip(accs, gaps):
            print(f" {a:>6.0%}({g:+.0%})", end="")
        print(f" {np.mean(accs):>7.1%}")


if __name__ == "__main__":
    main()
