"""
CSD digit-0 bias analysis:
1. Computing task, CodeLlama vs Llama-2: answer distribution for resolved samples,
   early-FJC answer distribution, and CSD layer 0-3 prediction distribution.
2. All 6 tasks, CodeLlama only: CSD layer 0-3 prediction distribution.
"""
import json
import os
from collections import Counter
from pathlib import Path

RESULTS_BASE = "/data/brewing_output/results/cuebench/eval"
DATASET_BASE = "/data/brewing_output/datasets/cuebench/eval"

MODELS = [
    "codellama__CodeLlama-7b-hf",
    "meta-llama__Llama-2-7b-hf",
]

TASKS = ["computing", "value_tracking", "conditional", "function_call", "loop", "loop_unrolled"]

EARLY_LAYERS = [0, 1, 2, 3]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_ground_truth_map(task):
    """Load samples.json to get sample_id -> answer mapping."""
    path = f"{DATASET_BASE}/{task}/seed42/samples.json"
    data = load_json(path)
    if data is None:
        return {}
    return {s["id"]: s["answer"] for s in data}


def print_counter(counter, label="", top_n=12):
    total = sum(counter.values())
    print(f"  {label} (n={total}):")
    for val, cnt in counter.most_common(top_n):
        pct = 100 * cnt / total if total else 0
        print(f"    '{val}': {cnt:5d}  ({pct:5.1f}%)")
    # Show digit-0 specifically if not in top
    if "0" not in dict(counter.most_common(top_n)):
        cnt0 = counter.get("0", 0)
        pct0 = 100 * cnt0 / total if total else 0
        print(f"    '0': {cnt0:5d}  ({pct0:5.1f}%)  [not in top {top_n}]")
    print()


def analyze_diagnostics(task, model, gt_map):
    """Analyze diagnostics.json: outcome distribution, answer distribution for resolved, early-FJC."""
    path = f"{RESULTS_BASE}/{task}/seed42/{model}/diagnostics.json"
    data = load_json(path)
    if data is None:
        print(f"  [diagnostics.json not found for {model}/{task}]")
        return

    samples = data["sample_diagnostics"]

    # Outcome distribution
    outcome_counter = Counter(s["outcome"] for s in samples)
    print(f"  Outcome distribution (n={len(samples)}):")
    for outcome, cnt in outcome_counter.most_common():
        print(f"    {outcome}: {cnt} ({100*cnt/len(samples):.1f}%)")
    print()

    # Resolved samples: ground truth answer distribution
    resolved = [s for s in samples if s["outcome"] == "resolved"]
    if resolved:
        gt_answers = Counter(gt_map.get(s["sample_id"], "?") for s in resolved)
        print_counter(gt_answers, "Ground truth answers (resolved samples)")

        model_outputs = Counter(s["model_output"] for s in resolved)
        print_counter(model_outputs, "Model outputs (resolved samples)")

    # Early FJC (fjc <= 3): answer distribution
    early_fjc = [s for s in samples if s["fjc"] is not None and s["fjc"] <= 3]
    if early_fjc:
        gt_early = Counter(gt_map.get(s["sample_id"], "?") for s in early_fjc)
        print_counter(gt_early, f"Ground truth answers (early FJC <= 3, n={len(early_fjc)})")

        model_early = Counter(s["model_output"] for s in early_fjc)
        print_counter(model_early, f"Model outputs (early FJC <= 3)")
    else:
        print("  [No samples with FJC <= 3]")
    print()


def analyze_csd_early_layers(task, model):
    """Analyze CSD predictions at early layers (0-3)."""
    path = f"{RESULTS_BASE}/{task}/seed42/{model}/csd.json"
    data = load_json(path)
    if data is None:
        print(f"  [csd.json not found for {model}/{task}]")
        return

    samples = data["sample_results"]
    n_samples = len(samples)
    n_layers = len(samples[0]["layer_predictions"]) if samples else 0
    print(f"  CSD early-layer predictions ({n_samples} samples, {n_layers} total layers):")

    for layer in EARLY_LAYERS:
        if layer >= n_layers:
            print(f"    Layer {layer}: [out of range]")
            continue
        preds = Counter(s["layer_predictions"][layer] for s in samples)
        total = sum(preds.values())
        # Show top predictions
        top = preds.most_common(5)
        top_str = ", ".join(f"'{v}':{c}({100*c/total:.1f}%)" for v, c in top)
        digit0_count = preds.get("0", 0)
        digit0_pct = 100 * digit0_count / total if total else 0
        print(f"    Layer {layer:2d}: digit-0={digit0_count}({digit0_pct:.1f}%) | top: {top_str}")
    print()


# ============================================================
# PART 1: Computing task, CodeLlama vs Llama-2
# ============================================================
print("=" * 80)
print("PART 1: Computing task — CodeLlama-7b vs Llama-2-7b")
print("=" * 80)

gt_map = get_ground_truth_map("computing")
print(f"\nGround truth answer distribution (all {len(gt_map)} samples):")
all_gt = Counter(gt_map.values())
print_counter(all_gt, "All samples")

for model in MODELS:
    print("-" * 60)
    print(f"Model: {model}")
    print("-" * 60)

    print("\n[Diagnostics]")
    analyze_diagnostics("computing", model, gt_map)

    print("[CSD early layers]")
    analyze_csd_early_layers("computing", model)


# ============================================================
# PART 2: All 6 tasks, CodeLlama only
# ============================================================
print("\n" + "=" * 80)
print("PART 2: All 6 tasks — CodeLlama-7b CSD early-layer predictions")
print("=" * 80)

model = "codellama__CodeLlama-7b-hf"
for task in TASKS:
    print(f"\n--- Task: {task} ---")
    analyze_csd_early_layers(task, model)
