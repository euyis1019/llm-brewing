#!/usr/bin/env python3
"""Per-Difficulty Breakdown analysis for CUE-Bench.

For each task × each difficulty dimension, sweep along that dimension
(marginalizing over the other two) and compute outcome distribution +
brewing metrics per bin.

Anchor model: Qwen2.5-Coder-7B (28 layers).
"""

import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────

RESULTS_BASE = Path("/path/to/brewing_output/results/cuebench/eval")
DATASETS_BASE = Path("/path/to/brewing_output/datasets/cuebench/eval")
SEED = "seed42"

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]

TASK_DIMENSIONS = {
    "value_tracking": ["mechanism", "depth", "distractors"],
    "computing":      ["structure", "steps", "operators"],
    "conditional":    ["branch_type", "condition_type", "depth"],
    "function_call":  ["mechanism", "depth", "distractors"],
    "loop":           ["body_type", "iterations", "init_offset"],
    "loop_unrolled":  ["body_type", "iterations", "init_offset"],
}

# Dimension value orderings (for sorting bins meaningfully)
DIM_ORDER = {
    # value_tracking
    # value_tracking mechanisms
    "mechanism": ["function_chain", "container", "method_chain",
                  # function_call mechanisms
                  "arithmetic", "container_relay", "conditional_return"],
    "depth": [1, 2, 3],
    "distractors": [0, 1, 2],
    # computing
    "structure": ["func_arithmetic", "chained_calls", "accumulator"],
    "steps": [2, 3, 4],
    "operators": ["add", "add_sub", "add_mul"],
    # conditional
    "branch_type": ["elif_chain", "guard_clause", "sequential_if"],
    "condition_type": ["numeric", "membership", "boolean_flag"],
    # loop
    "body_type": ["simple_acc", "filter_count", "dual_var"],
    "iterations": [2, 3, 4],
    "init_offset": ["0", "low", "high"],
}

DEFAULT_MODEL = "Qwen__Qwen2.5-Coder-7B"
DEFAULT_N_LAYERS = 28

OUTCOMES = ["resolved", "overprocessed", "misresolved", "unresolved"]


def load_data(task: str, model_dir: str) -> tuple[list[dict], list[dict]]:
    """Load dataset samples and diagnostics for a task+model."""
    ds_path = DATASETS_BASE / task / SEED / "samples.json"
    diag_path = RESULTS_BASE / task / SEED / model_dir / "diagnostics.json"

    with open(ds_path) as f:
        samples = json.load(f)
    with open(diag_path) as f:
        diag = json.load(f)

    return samples, diag["sample_diagnostics"]


def compute_bin_stats(
    bin_samples: list[dict], n_layers: int
) -> dict:
    """Compute outcome distribution and brewing metrics for a bin."""
    n = len(bin_samples)
    if n == 0:
        return None

    # Count outcomes
    outcome_counts = defaultdict(int)
    for s in bin_samples:
        outcome_counts[s["outcome"]] += 1

    nb_count = outcome_counts.get("no_brewing", 0)
    n_classified = n - nb_count  # denominator for outcome %

    # Outcome percentages (excluding NB)
    pcts = {}
    for o in OUTCOMES:
        pcts[o] = outcome_counts.get(o, 0) / n_classified if n_classified > 0 else 0.0

    # FPCL (normalized) — only for samples with fpcl != null
    fpcl_vals = [s["fpcl"] / n_layers for s in bin_samples if s["fpcl"] is not None]
    fpcl_mean = sum(fpcl_vals) / len(fpcl_vals) if fpcl_vals else None

    # FJC — only for samples with fjc != null
    fjc_vals = [s["fjc"] / n_layers for s in bin_samples if s["fjc"] is not None]
    fjc_mean = sum(fjc_vals) / len(fjc_vals) if fjc_vals else None
    fjc_exist_pct = len(fjc_vals) / n if n > 0 else 0.0

    # ΔBrew — only for samples with delta_brew != null
    dbrew_vals = [s["delta_brew"] for s in bin_samples if s["delta_brew"] is not None]
    dbrew_mean = sum(dbrew_vals) / len(dbrew_vals) if dbrew_vals else None

    return {
        "N": n,
        "NB": nb_count,
        "Res%": pcts["resolved"],
        "OP%": pcts["overprocessed"],
        "MR%": pcts["misresolved"],
        "UR%": pcts["unresolved"],
        "FPCL_mean": fpcl_mean,
        "FJC_mean": fjc_mean,
        "FJC_exist%": fjc_exist_pct,
        "dBrew_mean": dbrew_mean,
    }


def run_sweep(
    task: str, dim: str, merged: list[dict], n_layers: int
) -> list[dict]:
    """Sweep along one dimension, marginalizing over the other two."""
    # Group by dimension value
    bins = defaultdict(list)
    for s in merged:
        val = s["difficulty"].get(dim)
        bins[val].append(s)

    # Sort bins by predefined order if available
    order = DIM_ORDER.get(dim, sorted(bins.keys(), key=str))
    # Filter to values that actually appear
    ordered_vals = [v for v in order if v in bins]
    # Add any values not in predefined order
    for v in sorted(bins.keys(), key=str):
        if v not in ordered_vals:
            ordered_vals.append(v)

    results = []
    for val in ordered_vals:
        stats = compute_bin_stats(bins[val], n_layers)
        if stats:
            stats["task"] = task
            stats["dimension"] = dim
            stats["bin"] = val
            results.append(stats)

    return results


def main():
    parser = argparse.ArgumentParser(description="Per-Difficulty Breakdown")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model directory name")
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS,
                        help="Number of layers for normalization")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: brewing_output/artifacts/per_difficulty)")
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        help="Tasks to process")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("/path/to/brewing_output/artifacts/per_difficulty")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        samples, diag_samples = load_data(task, args.model)

        # Build lookup: sample_id -> difficulty
        ds_lookup = {s["id"]: s["difficulty"] for s in samples}

        # Merge diagnostics with difficulty info
        merged = []
        for ds in diag_samples:
            sid = ds["sample_id"]
            if sid in ds_lookup:
                merged.append({**ds, "difficulty": ds_lookup[sid]})
            else:
                print(f"  WARNING: {sid} not found in dataset")

        dims = TASK_DIMENSIONS[task]
        for dim in dims:
            print(f"\n  Dimension: {dim}")
            print(f"  {'bin':<25} {'N':>4} {'NB':>3} {'Res%':>6} {'OP%':>6} "
                  f"{'MR%':>6} {'UR%':>6} {'FPCL':>6} {'FJC':>6} "
                  f"{'FJC%':>5} {'dBrew':>6}")
            print(f"  {'-'*95}")

            sweep = run_sweep(task, dim, merged, args.n_layers)
            for row in sweep:
                fpcl_s = f"{row['FPCL_mean']:.3f}" if row['FPCL_mean'] is not None else "  N/A"
                fjc_s = f"{row['FJC_mean']:.3f}" if row['FJC_mean'] is not None else "  N/A"
                dbrew_s = f"{row['dBrew_mean']:.2f}" if row['dBrew_mean'] is not None else "  N/A"

                print(f"  {str(row['bin']):<25} {row['N']:>4} {row['NB']:>3} "
                      f"{row['Res%']:>5.1%} {row['OP%']:>5.1%} "
                      f"{row['MR%']:>5.1%} {row['UR%']:>5.1%} "
                      f"{fpcl_s:>6} {fjc_s:>6} "
                      f"{row['FJC_exist%']:>4.1%} {dbrew_s:>6}")

            all_results.extend(sweep)

    # ── Save outputs ────────────────────────────────────────────────────

    model_tag = args.model.replace("/", "_").replace("__", "_")

    # JSON
    json_path = output_dir / f"per_difficulty_{model_tag}.json"
    # Convert for JSON serialization
    json_data = []
    for r in all_results:
        row = {k: v for k, v in r.items()}
        # Convert bin values to string for consistency
        row["bin"] = str(row["bin"])
        json_data.append(row)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # CSV
    csv_path = output_dir / f"per_difficulty_{model_tag}.csv"
    fieldnames = [
        "task", "dimension", "bin", "N", "NB",
        "Res%", "OP%", "MR%", "UR%",
        "FPCL_mean", "FJC_mean", "FJC_exist%", "dBrew_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k) for k in fieldnames}
            row["bin"] = str(row["bin"])
            # Format percentages
            for pct_col in ["Res%", "OP%", "MR%", "UR%", "FJC_exist%"]:
                if row[pct_col] is not None:
                    row[pct_col] = f"{row[pct_col]:.4f}"
            for float_col in ["FPCL_mean", "FJC_mean", "dBrew_mean"]:
                if row[float_col] is not None:
                    row[float_col] = f"{row[float_col]:.4f}"
            writer.writerow(row)
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
