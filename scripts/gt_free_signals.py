#!/usr/bin/env python3
"""GT-free Signal Analysis (§3.2, App F, A6).

Three GT-free signals derived from CSD outputs:
  1. Entropy  H(ℓ) = -Σ p_i log p_i   (lower = more confident in answer space)
  2. MaxConf  C(ℓ) = max_i p_i         (higher = more confident)
  3. NonDigit N(ℓ) = 1 - Σ_digits softmax(full_vocab)  (higher = less digit-focused)

Evaluates for:
  - Brewing zone detection: can we identify FPCL/FJC without ground truth?
  - OT detection: can we distinguish Overprocessed from Resolved?

Metrics: AUC, F1, agreement rate.

Output:
  brewing_output/artifacts/gt_free/  — summary JSON + console table
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "brewing_output"
DIAG_ROOT = BASE / "results" / "cuebench" / "eval"
CSD_ROOT = BASE / "results" / "cuebench" / "eval"
OUT_DIR = BASE / "artifacts" / "gt_free"

ANCHOR = "Qwen__Qwen2.5-Coder-7B"
SEED = "seed42"
TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]


def load_diagnostics(task: str) -> list[dict]:
    path = DIAG_ROOT / task / SEED / ANCHOR / "diagnostics.json"
    with open(path) as f:
        return json.load(f)["sample_diagnostics"]


def load_csd(task: str) -> list[dict]:
    path = CSD_ROOT / task / SEED / ANCHOR / "csd.json"
    with open(path) as f:
        return json.load(f)["sample_results"]


def compute_entropy(confidences: np.ndarray) -> np.ndarray:
    """Entropy per layer. confidences: (L, C)."""
    p = np.clip(confidences, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)  # (L,)


def compute_max_conf(confidences: np.ndarray) -> np.ndarray:
    return np.max(confidences, axis=1)  # (L,)


# ── Brewing zone detection ─────────────────────────────────────────────

def detect_brewing_nondigit(non_digit: np.ndarray, threshold: float = 0.5) -> int | None:
    """Estimate brewing as first layer where non-digit prob drops below threshold
    (model starts focusing on digit tokens)."""
    candidates = np.where(non_digit < threshold)[0]
    return int(candidates[0]) if len(candidates) > 0 else None


def detect_fjc_combined(max_conf: np.ndarray, non_digit: np.ndarray,
                        conf_thresh: float = 0.5, nd_thresh: float = 0.3) -> int | None:
    """Estimate FJC: first layer where max confidence > threshold AND non-digit < threshold.
    Both conditions must hold = model is confident AND focused on digits."""
    mask = (max_conf > conf_thresh) & (non_digit < nd_thresh)
    candidates = np.where(mask)[0]
    return int(candidates[0]) if len(candidates) > 0 else None


# ── OT detection features ─────────────────────────────────────────────

def extract_ot_features(entropy: np.ndarray, max_conf: np.ndarray,
                        non_digit: np.ndarray | None, n_layers: int) -> dict:
    tail_start = int(n_layers * 0.75)
    mid_start, mid_end = int(n_layers * 0.25), int(n_layers * 0.75)

    tail_entropy_mean = entropy[tail_start:].mean()
    mid_conf_max = max_conf[mid_start:mid_end].max() if mid_end > mid_start else 0.0
    conf_drop = mid_conf_max - max_conf[-1]
    mid_entropy_mean = entropy[mid_start:mid_end].mean() if mid_end > mid_start else 0.0
    entropy_rise = tail_entropy_mean - mid_entropy_mean

    feats = {
        "tail_entropy": float(tail_entropy_mean),
        "mid_conf_max": float(mid_conf_max),
        "conf_drop": float(conf_drop),
        "entropy_rise": float(entropy_rise),
        "final_conf": float(max_conf[-1]),
    }

    if non_digit is not None:
        # non-digit rise in tail: OT samples should show rising non-digit prob
        mid_nd = non_digit[mid_start:mid_end].mean() if mid_end > mid_start else 0.0
        tail_nd = non_digit[tail_start:].mean()
        feats["nondigit_rise"] = float(tail_nd - mid_nd)
        feats["tail_nondigit"] = float(tail_nd)

    return feats


def analyse_task(task: str) -> dict:
    diags = load_diagnostics(task)
    csds = load_csd(task)
    n_layers = len(csds[0]["layer_confidences"])

    csd_by_id = {s["sample_id"]: s for s in csds}

    # Check if non-digit data available
    has_nondigit = "extras" in csds[0] and "layer_non_digit_probs" in csds[0].get("extras", {})

    records = []
    for sd in diags:
        sid = sd["sample_id"]
        csd = csd_by_id.get(sid)
        if csd is None:
            continue

        confs = np.array(csd["layer_confidences"])  # (L, 10)
        entropy = compute_entropy(confs)
        max_conf = compute_max_conf(confs)

        non_digit = None
        if has_nondigit:
            non_digit = np.array(csd["extras"]["layer_non_digit_probs"])

        fpcl = sd["fpcl"]
        fjc = sd["fjc"]
        outcome = sd["outcome"]

        # GT-free FJC estimation
        if non_digit is not None:
            est_fjc = detect_fjc_combined(max_conf, non_digit)
        else:
            # fallback: confidence-only
            candidates = np.where(max_conf > 0.5)[0]
            est_fjc = int(candidates[0]) if len(candidates) > 0 else None

        ot_feats = extract_ot_features(entropy, max_conf, non_digit, n_layers)

        records.append({
            "sample_id": sid,
            "outcome": outcome,
            "gt_fpcl": fpcl,
            "gt_fjc": fjc,
            "est_fjc": est_fjc,
            **ot_feats,
        })

    # ── Brewing detection: FJC exist agreement ──
    fjc_exist_agree = sum(
        1 for r in records
        if (r["gt_fjc"] is not None) == (r["est_fjc"] is not None)
    )
    fjc_exist_agreement = fjc_exist_agree / len(records) if records else 0

    fjc_distances = [
        abs(r["gt_fjc"] - r["est_fjc"])
        for r in records
        if r["gt_fjc"] is not None and r["est_fjc"] is not None
    ]
    mean_fjc_distance = float(np.mean(fjc_distances)) if fjc_distances else None

    # ── OT detection: OT(1) vs Resolved(0) ──
    ot_labels, ot_scores = [], {}
    signal_names = ["conf_drop", "entropy_rise"]
    if has_nondigit:
        signal_names += ["nondigit_rise", "tail_nondigit"]

    for name in signal_names:
        ot_scores[name] = []

    for r in records:
        if r["outcome"] == "overprocessed":
            ot_labels.append(1)
        elif r["outcome"] == "resolved":
            ot_labels.append(0)
        else:
            continue
        for name in signal_names:
            ot_scores[name].append(r[name])

    ot_labels = np.array(ot_labels)
    ot_n = len(ot_labels)

    auc_per_signal = {}
    combined_auc = None
    combined_f1 = None
    combined_agreement = None

    if ot_n > 10 and ot_labels.sum() > 0 and (ot_n - ot_labels.sum()) > 0:
        z_scores = []
        for name in signal_names:
            sc = np.array(ot_scores[name])
            auc_per_signal[name] = float(roc_auc_score(ot_labels, sc))
            z = (sc - sc.mean()) / (sc.std() + 1e-8)
            z_scores.append(z)

        # combined = sum of z-scored signals
        sc_combined = sum(z_scores)
        combined_auc = float(roc_auc_score(ot_labels, sc_combined))

        # optimal threshold via Youden's J
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(ot_labels, sc_combined)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        preds = (sc_combined >= best_thresh).astype(int)
        combined_f1 = float(f1_score(ot_labels, preds))
        combined_agreement = float((preds == ot_labels).mean())

    # ── Overall agreement (paper's composite metric) ──
    # Two-part: (1) brewing existence, (2) OT vs Resolved among brewing samples
    overall_correct = 0
    overall_total = 0
    for r in records:
        if r["outcome"] == "no_brewing":
            continue
        overall_total += 1
        gt_brewing = r["gt_fjc"] is not None
        est_brewing = r["est_fjc"] is not None
        if gt_brewing == est_brewing:
            overall_correct += 1
    overall_agreement = overall_correct / overall_total if overall_total else 0

    return {
        "task": task,
        "n_samples": len(records),
        "n_layers": n_layers,
        "has_nondigit": has_nondigit,
        "fjc_exist_agreement": fjc_exist_agreement,
        "mean_fjc_distance": mean_fjc_distance,
        "n_ot_vs_resolved": ot_n,
        "auc_per_signal": auc_per_signal,
        "combined_auc": combined_auc,
        "combined_f1": combined_f1,
        "combined_agreement": combined_agreement,
        "overall_agreement": overall_agreement,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for task in TASKS:
        print(f"Processing {task}...")
        r = analyse_task(task)
        all_results.append(r)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print results ──
    has_nd = any(r["has_nondigit"] for r in all_results)

    print("\n=== Per-Signal AUC (OT vs Resolved) ===")
    header = f"{'Task':<18} {'conf_drop':>10} {'ent_rise':>10}"
    if has_nd:
        header += f" {'nd_rise':>10} {'tail_nd':>10}"
    header += f" {'combined':>10}"
    print(header)

    for r in all_results:
        auc = r["auc_per_signal"]
        line = f"{r['task']:<18}"
        line += f" {auc.get('conf_drop', 0):>10.3f}"
        line += f" {auc.get('entropy_rise', 0):>10.3f}"
        if has_nd:
            line += f" {auc.get('nondigit_rise', 0):>10.3f}"
            line += f" {auc.get('tail_nondigit', 0):>10.3f}"
        line += f" {r['combined_auc'] or 0:>10.3f}"
        print(line)

    print(f"\n=== Summary ===")
    print(f"{'Task':<18} {'N':>5} {'FJC_agr':>8} {'FJC_dist':>9} "
          f"{'OT_AUC':>8} {'OT_F1':>7} {'OT_agr':>7} {'Overall':>8}")
    for r in all_results:
        fjc_d = f"{r['mean_fjc_distance']:.1f}" if r['mean_fjc_distance'] is not None else "N/A"
        auc = f"{r['combined_auc']:.3f}" if r['combined_auc'] else "N/A"
        f1 = f"{r['combined_f1']:.3f}" if r['combined_f1'] else "N/A"
        ot_agr = f"{r['combined_agreement']:.3f}" if r['combined_agreement'] else "N/A"
        print(f"{r['task']:<18} {r['n_samples']:>5} {r['fjc_exist_agreement']:>8.3f} {fjc_d:>9} "
              f"{auc:>8} {f1:>7} {ot_agr:>7} {r['overall_agreement']:>8.3f}")

    # weighted averages
    total_n = sum(r["n_samples"] for r in all_results)
    w_overall = sum(r["overall_agreement"] * r["n_samples"] for r in all_results) / total_n
    w_fjc = sum(r["fjc_exist_agreement"] * r["n_samples"] for r in all_results) / total_n
    ot_results = [r for r in all_results if r["combined_auc"] is not None]
    if ot_results:
        ot_total = sum(r["n_ot_vs_resolved"] for r in ot_results)
        w_auc = sum(r["combined_auc"] * r["n_ot_vs_resolved"] for r in ot_results) / ot_total
        w_f1 = sum(r["combined_f1"] * r["n_ot_vs_resolved"] for r in ot_results) / ot_total
        w_ot_agr = sum(r["combined_agreement"] * r["n_ot_vs_resolved"] for r in ot_results) / ot_total
    else:
        w_auc = w_f1 = w_ot_agr = 0
    print(f"\n{'Weighted avg':<18} {total_n:>5} {w_fjc:>8.3f} {'':>9} "
          f"{w_auc:>8.3f} {w_f1:>7.3f} {w_ot_agr:>7.3f} {w_overall:>8.3f}")


if __name__ == "__main__":
    main()
