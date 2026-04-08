#!/usr/bin/env python3
"""P0 calculations: A2 verification + §5.3 FJC correlation."""

import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent / "brewing_output"
DIAG = BASE / "results" / "cuebench" / "eval"
DS = BASE / "datasets" / "cuebench" / "eval"
SEED = "seed42"
TASKS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
ANCHOR = "Qwen__Qwen2.5-Coder-7B"


def a2_fjc_null_correct_rate():
    print("=== A2: FJC=null correct rate ===")
    total_null, total_correct = 0, 0
    for task in TASKS:
        with open(DIAG / task / SEED / ANCHOR / "diagnostics.json") as f:
            diags = json.load(f)["sample_diagnostics"]
        with open(DS / task / SEED / "samples.json") as f:
            answers = {s["id"]: s["answer"] for s in json.load(f)}

        fjc_null = [
            (d, answers[d["sample_id"]])
            for d in diags
            if d["fjc"] is None and d["sample_id"] in answers
        ]
        n_null = len(fjc_null)
        n_correct = sum(1 for d, ans in fjc_null if d["model_output"] == ans)
        total_null += n_null
        total_correct += n_correct
        rate = n_correct / n_null * 100 if n_null else 0
        print(f"  {task:<18} null={n_null:>4}  correct={n_correct:>3}  ({rate:.1f}%)")

    overall = total_correct / total_null * 100 if total_null else 0
    print(f"  {'TOTAL':<18} null={total_null:>4}  correct={total_correct:>3}  ({overall:.2f}%)")
    print(f"\n  Old result: 0.65%")
    return overall


def fjc_correlation():
    print("\n=== §5.3: Coder-7B vs Base-7B FJC correlation ===")
    models = {
        "Coder-7B": "Qwen__Qwen2.5-Coder-7B",
        "Base-7B": "Qwen__Qwen2.5-7B",
    }
    fjc = {m: [] for m in models}

    for task in TASKS:
        for label, mdir in models.items():
            p = DIAG / task / SEED / mdir / "diagnostics.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                fjc[label].append(d.get("mean_fjc_normalized"))
            else:
                fjc[label].append(None)

    print(f"\n  {'Task':<18} {'Coder-7B':>10} {'Base-7B':>10}")
    for i, t in enumerate(TASKS):
        c = f"{fjc['Coder-7B'][i]:.3f}" if fjc['Coder-7B'][i] is not None else "N/A"
        b = f"{fjc['Base-7B'][i]:.3f}" if fjc['Base-7B'][i] is not None else "N/A"
        print(f"  {t:<18} {c:>10} {b:>10}")

    pairs = [
        (c, b)
        for c, b in zip(fjc["Coder-7B"], fjc["Base-7B"])
        if c is not None and b is not None
    ]
    if len(pairs) >= 3:
        cv, bv = zip(*pairs)
        r, p = stats.pearsonr(cv, bv)
        shift = np.mean([c - b for c, b in pairs])
        print(f"\n  Pearson r = {r:.3f}, p = {p:.4f}")
        print(f"  Mean FJC shift (Coder - Base, normalized) = {shift:.4f}")
        print(f"  Mean FJC shift (layers, 28L model) = {shift * 28:.2f}")
        print(f"\n  Old result: r=0.90, p=0.014, shift=0.18 layers (0.6%)")
        return r, p, shift
    else:
        print("  Not enough paired data points")
        return None


if __name__ == "__main__":
    a2_fjc_null_correct_rate()
    fjc_correlation()
