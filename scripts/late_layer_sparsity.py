#!/usr/bin/env python3
"""Late-Layer Sparsity Analysis (App E.3, A18).

Computes Hoyer sparsity per layer for Overprocessed vs non-OT samples.
Uses hidden state caches + diagnostics from the anchor model (Qwen2.5-Coder-7B).

Output:
  brewing_output/artifacts/sparsity/  — per-task CSVs + summary
"""

import json
import sys
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "brewing_output"
CACHE_ROOT = BASE / "caches" / "cuebench" / "eval"
DIAG_ROOT = BASE / "results" / "cuebench" / "eval"
OUT_DIR = BASE / "artifacts" / "sparsity"

ANCHOR = "Qwen__Qwen2.5-Coder-7B"
SEED = "seed42"
TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]


def hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer sparsity for a 1-D vector.

    Returns value in [0, 1]. Higher = sparser.
    0 when all elements equal, 1 when only one non-zero.
    """
    d = x.shape[0]
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x ** 2).sum())
    if l2 < 1e-12:
        return 0.0
    sqrt_d = np.sqrt(d)
    return float((sqrt_d - l1 / l2) / (sqrt_d - 1.0))


def load_diagnostics(task: str) -> list[dict]:
    path = DIAG_ROOT / task / SEED / ANCHOR / "diagnostics.json"
    with open(path) as f:
        data = json.load(f)
    return data["sample_diagnostics"]


def load_hidden_states(task: str) -> tuple[np.ndarray, list[str]]:
    """Returns (hidden_states (N, L, D), sample_ids)."""
    cache_dir = CACHE_ROOT / task / SEED / ANCHOR
    hs = np.load(cache_dir / "hidden_states.npz")["hidden_states"]
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    return hs, meta["sample_ids"]


def analyse_task(task: str) -> dict:
    diags = load_diagnostics(task)
    hs, cache_ids = load_hidden_states(task)
    n_samples, n_layers, hidden_dim = hs.shape

    # build id → index for cache
    id_to_idx = {sid: i for i, sid in enumerate(cache_ids)}

    # group sample indices by outcome
    groups: dict[str, list[int]] = {
        "overprocessed": [],
        "resolved": [],
        "misresolved": [],
        "unresolved": [],
    }
    for sd in diags:
        outcome = sd["outcome"]
        if outcome == "no_brewing":
            continue
        idx = id_to_idx.get(sd["sample_id"])
        if idx is None:
            continue
        groups[outcome].append(idx)

    # compute per-layer sparsity for each group
    result = {"task": task, "n_layers": n_layers}
    for grp_name, indices in groups.items():
        if not indices:
            result[grp_name] = {"n": 0, "mean": None, "std": None}
            continue
        subset = hs[indices]  # (n_grp, L, D)
        # vectorised hoyer per sample per layer
        l1 = np.abs(subset).sum(axis=2)      # (n_grp, L)
        l2 = np.sqrt((subset ** 2).sum(axis=2))  # (n_grp, L)
        sqrt_d = np.sqrt(hidden_dim)
        # avoid division by zero
        safe_l2 = np.where(l2 < 1e-12, 1.0, l2)
        sparsity = (sqrt_d - l1 / safe_l2) / (sqrt_d - 1.0)  # (n_grp, L)
        sparsity = np.where(l2 < 1e-12, 0.0, sparsity)

        mean_per_layer = sparsity.mean(axis=0)  # (L,)
        std_per_layer = sparsity.std(axis=0)    # (L,)
        result[grp_name] = {
            "n": len(indices),
            "mean": mean_per_layer.tolist(),
            "std": std_per_layer.tolist(),
        }

    # OT vs non-OT gap
    ot = result["overprocessed"]
    res = result["resolved"]
    if ot["mean"] is not None and res["mean"] is not None:
        gap = np.array(ot["mean"]) - np.array(res["mean"])
        result["ot_vs_resolved_gap"] = gap.tolist()
        result["max_gap_layer"] = int(np.argmax(gap))
        result["max_gap_value"] = float(gap.max())
        # spike: layer with max absolute sparsity among OT
        result["ot_spike_layer"] = int(np.argmax(ot["mean"]))
    else:
        result["ot_vs_resolved_gap"] = None
        result["max_gap_layer"] = None
        result["max_gap_value"] = None
        result["ot_spike_layer"] = None

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for task in TASKS:
        print(f"Processing {task}...")
        r = analyse_task(task)
        all_results.append(r)

        # per-task CSV: layer, ot_mean, ot_std, res_mean, res_std, gap
        csv_path = OUT_DIR / f"{task}_sparsity.csv"
        n_layers = r["n_layers"]
        with open(csv_path, "w") as f:
            f.write("layer,ot_mean,ot_std,resolved_mean,resolved_std,gap\n")
            for l in range(n_layers):
                ot_m = r["overprocessed"]["mean"][l] if r["overprocessed"]["mean"] else ""
                ot_s = r["overprocessed"]["std"][l] if r["overprocessed"]["std"] else ""
                re_m = r["resolved"]["mean"][l] if r["resolved"]["mean"] else ""
                re_s = r["resolved"]["std"][l] if r["resolved"]["std"] else ""
                gap = r["ot_vs_resolved_gap"][l] if r["ot_vs_resolved_gap"] else ""
                f.write(f"{l},{ot_m},{ot_s},{re_m},{re_s},{gap}\n")

    # summary
    summary_path = OUT_DIR / "summary.json"
    summary = []
    for r in all_results:
        summary.append({
            "task": r["task"],
            "n_overprocessed": r["overprocessed"]["n"],
            "n_resolved": r["resolved"]["n"],
            "ot_spike_layer": r["ot_spike_layer"],
            "max_gap_layer": r["max_gap_layer"],
            "max_gap_value": r["max_gap_value"],
        })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # print summary
    print("\n=== Sparsity Summary ===")
    print(f"{'Task':<18} {'N_OT':>5} {'N_Res':>5} {'OT_spike':>9} {'MaxGap_L':>9} {'MaxGap':>8}")
    for s in summary:
        print(f"{s['task']:<18} {s['n_overprocessed']:>5} {s['n_resolved']:>5} "
              f"{s['ot_spike_layer']:>9} {s['max_gap_layer']:>9} {s['max_gap_value']:>8.4f}")


if __name__ == "__main__":
    main()
