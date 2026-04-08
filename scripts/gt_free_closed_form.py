#!/usr/bin/env python3
"""GT-free Closed-Form Outcome Discrimination (§3.2 / Appendix F).

Derives four outcome categories from layer-wise CSD and Probing
*distributions* alone — no ground-truth labels required.

═══════════════════════════════════════════════════════════════════════════
Signal                 Formula                              Domain
───────────────────────────────────────────────────────────────────────────
CSD entropy            H_C^ℓ = −Σ Φ_C[t] ln Φ_C[t]         [0, ln C]
Probe entropy          H_P^ℓ = −Σ Φ_P[t] ln Φ_P[t]         [0, ln C]
Probe–CSD JSD          D^ℓ = JSD(Φ_P^ℓ ‖ Φ_C^ℓ)            [0, ln 2]
Entropy velocity       v_H^ℓ = ∂H_C/∂ℓ  (Savitzky-Golay)   ℝ
Argmax agreement       A^ℓ = 𝟙[argmax Φ_P = argmax Φ_C]    {0,1}
NonDigit mass          N^ℓ = 1 − Σ_{digits} softmax         [0,1]
═══════════════════════════════════════════════════════════════════════════

Normalized features (scale-free, comparable across models):

    ĥ  = H̄_tail / ln C           ∈ [0,1]   residual uncertainty
    d̂  = D̄_tail / ln 2           ∈ [0,1]   probe–CSD disagreement
    â  = Ā_tail                   ∈ [0,1]   argmax agreement
    ŝ  = |min v_H| / (ln C / L)  ≥ 0       normalised sharpening speed

Decision surface (axis-aligned, no learned weights):

    ┌── ŝ < α_brew ───────────────────→ NO_BREWING
    │
    ├── ĥ ≥ β_H ──────────────────────→ UNRESOLVED
    │
    ├── ĥ < β_H  AND  d̂ < γ_D ───────→ RESOLVED
    │
    ├── ĥ < β_H  AND  â < δ_A ───────→ MISRESOLVED
    │
    └── else ──────────────────────────→ OVERPROCESSED

Usage:
    python scripts/gt_free_closed_form.py [--models M1,M2,...] [--tasks T1,...]
    python scripts/gt_free_closed_form.py --sweep          # grid-search thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as sp_entropy

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "brewing_output"
EVAL_ROOT = BASE / "results" / "cuebench" / "eval"
OUT_DIR = BASE / "artifacts" / "gt_free_v2"

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]
SEED = "seed42"
C = 10  # answer space cardinality (digits 0–9)
LN_C = float(np.log(C))     # ≈ 2.3026
LN_2 = float(np.log(2))     # ≈ 0.6931


# ═══════════════════════════════════════════════════════════════════════════
# §1  Signal extraction
# ═══════════════════════════════════════════════════════════════════════════

def _smooth_deriv(sig: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    n = len(sig)
    w = min(window, n if n % 2 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 else poly + 3)
    if w > n:
        return np.gradient(sig)
    return savgol_filter(sig, window_length=w, polyorder=poly, deriv=1)


@dataclass
class SampleFeatures:
    """Per-sample normalised feature vector (model-agnostic)."""
    sample_id: str
    n_layers: int
    # normalised scalars
    h_hat: float       # ĥ  = H̄_csd_tail / ln C
    d_hat: float       # d̂  = D̄_jsd_tail / ln 2
    a_hat: float       # â  = mean argmax agreement in tail
    s_hat: float       # ŝ  = |min v_H| / (ln C / L)
    # raw curves (for diagnostics / plotting)
    H_csd: np.ndarray
    H_probe: np.ndarray
    jsd: np.ndarray
    v_H: np.ndarray
    agree: np.ndarray
    non_digit: np.ndarray | None


def extract_features(
    csd_confs: np.ndarray,     # (L, C)
    probe_confs: np.ndarray,   # (L, C)
    non_digit: np.ndarray | None,
    sample_id: str,
) -> SampleFeatures:
    L = csd_confs.shape[0]
    eps = 1e-10
    tail = int(L * 0.75)

    # normalise to valid distributions
    csd_p = np.clip(csd_confs, eps, None)
    csd_p /= csd_p.sum(axis=1, keepdims=True)
    probe_p = np.clip(probe_confs, eps, None)
    probe_p /= probe_p.sum(axis=1, keepdims=True)

    # --- layer-wise signals ---
    H_csd = sp_entropy(csd_p.T)      # (L,)
    H_probe = sp_entropy(probe_p.T)  # (L,)
    jsd = np.array([jensenshannon(probe_p[l], csd_p[l]) ** 2 for l in range(L)])
    v_H = _smooth_deriv(H_csd)
    agree = (np.argmax(csd_p, axis=1) == np.argmax(probe_p, axis=1)).astype(float)

    # --- normalised scalars ---
    h_hat = float(np.mean(H_csd[tail:])) / LN_C
    d_hat = float(np.mean(jsd[tail:])) / LN_2
    a_hat = float(np.mean(agree[tail:]))
    s_hat = float(np.abs(np.min(v_H))) / (LN_C / L)

    return SampleFeatures(
        sample_id=sample_id, n_layers=L,
        h_hat=h_hat, d_hat=d_hat, a_hat=a_hat, s_hat=s_hat,
        H_csd=H_csd, H_probe=H_probe, jsd=jsd, v_H=v_H,
        agree=agree, non_digit=non_digit,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §2  Closed-form decision surface
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Thresholds:
    alpha_brew: float = 0.30   # ŝ below this → no significant entropy drop
    beta_H: float = 0.58       # ĥ above this → unresolved (tail still uncertain)
    gamma_D: float = 0.22      # d̂ below this → probe–CSD agree → resolved
    delta_A: float = 0.55      # â below this → argmax disagree → misresolved


# Default thresholds — derived from information-theoretic reasoning:
#   β_H = 0.58 ≈ "tail entropy > 58% of max → more uncertain than certain"
#   γ_D = 0.22 ≈ "JSD > 22% of max → meaningful disagreement"
#   δ_A = 0.55 ≈ "argmax agrees less than 55% of tail layers → not aligned"
#   α_brew = 0.30 ≈ "entropy drop < 30% of one-layer-uniform-drop rate"
DEFAULT_THRESH = Thresholds()


def classify(feat: SampleFeatures, t: Thresholds) -> str:
    if feat.s_hat < t.alpha_brew:
        return "no_brewing"
    if feat.h_hat >= t.beta_H:
        return "unresolved"
    if feat.d_hat < t.gamma_D:
        return "resolved"
    if feat.a_hat < t.delta_A:
        return "misresolved"
    return "overprocessed"


# ═══════════════════════════════════════════════════════════════════════════
# §3  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def discover_models(task: str) -> list[str]:
    d = EVAL_ROOT / task / SEED
    if not d.exists():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and (p / "csd.json").exists() and (p / "linear_probing.json").exists()
    )


def load_task_model(task: str, model: str):
    base = EVAL_ROOT / task / SEED / model
    with open(base / "csd.json") as f:
        csd = json.load(f)["sample_results"]
    with open(base / "linear_probing.json") as f:
        probe = json.load(f)["sample_results"]
    diag_path = base / "diagnostics.json"
    diag = None
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)["sample_diagnostics"]
    return csd, probe, diag


def build_features(csd_results: list[dict], probe_results: list[dict]) -> list[SampleFeatures]:
    probe_map = {s["sample_id"]: s for s in probe_results}
    feats = []
    for cs in csd_results:
        sid = cs["sample_id"]
        ps = probe_map.get(sid)
        if ps is None or ps.get("layer_confidences") is None:
            continue
        nd = None
        if cs.get("extras") and "layer_non_digit_probs" in cs["extras"]:
            nd = np.array(cs["extras"]["layer_non_digit_probs"])
        feats.append(extract_features(
            np.array(cs["layer_confidences"]),
            np.array(ps["layer_confidences"]),
            nd, sid,
        ))
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# §4  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

OUTCOME_ORDER = ["resolved", "overprocessed", "misresolved", "unresolved", "no_brewing"]


def cohens_kappa(yt: list[str], yp: list[str]) -> float:
    labels = sorted(set(yt) | set(yp))
    n = len(yt)
    if n == 0:
        return 0.0
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    C = np.zeros((k, k), dtype=int)
    for t, p in zip(yt, yp):
        C[idx[t], idx[p]] += 1
    p_o = np.trace(C) / n
    p_e = sum(C[i, :].sum() * C[:, i].sum() for i in range(k)) / (n * n)
    return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0


@dataclass
class EvalRow:
    task: str
    model: str
    n: int
    n_layers: int
    acc: float
    kappa: float
    per_outcome: dict[str, dict[str, float]]  # outcome → {prec, rec, f1, support}
    confusion: dict[str, dict[str, int]]


def evaluate(
    feats: list[SampleFeatures],
    diag: list[dict],
    thresh: Thresholds,
    task: str,
    model: str,
) -> EvalRow:
    gt_map = {d["sample_id"]: d["outcome"] for d in diag}
    yt, yp = [], []
    for f in feats:
        gt = gt_map.get(f.sample_id)
        if gt is None:
            continue
        yt.append(gt)
        yp.append(classify(f, thresh))

    labels = sorted(set(yt) | set(yp))
    conf = {l: {l2: 0 for l2 in labels} for l in labels}
    for t, p in zip(yt, yp):
        conf[t][p] += 1

    per_out = {}
    for l in labels:
        tp = conf.get(l, {}).get(l, 0)
        p_total = sum(conf.get(t, {}).get(l, 0) for t in labels)
        t_total = sum(conf.get(l, {}).get(p, 0) for p in labels)
        prec = tp / p_total if p_total else 0
        rec = tp / t_total if t_total else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_out[l] = {"prec": prec, "rec": rec, "f1": f1, "support": t_total}

    n = len(yt)
    acc = sum(t == p for t, p in zip(yt, yp)) / n if n else 0
    return EvalRow(task, model, n, feats[0].n_layers if feats else 0,
                   acc, cohens_kappa(yt, yp), per_out, conf)


# ═══════════════════════════════════════════════════════════════════════════
# §4b  Binary signal-level AUC (continuous scores, no threshold needed)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BinaryAUCs:
    """AUC for binary sub-problems using continuous GT-free signals."""
    task: str
    model: str
    n: int
    # Brewing detection: has_brewing (resolved|OT|misresolved) vs no_brewing
    brewing_auc_h: float | None = None      # using ĥ (lower → brewing)
    brewing_auc_s: float | None = None      # using ŝ (higher → brewing)
    # OT detection among "has FJC" samples: OT(1) vs Resolved(0)
    ot_auc_h: float | None = None           # using ĥ (higher → OT)
    ot_auc_d: float | None = None           # using d̂ (higher → OT)
    ot_auc_a: float | None = None           # using â (lower → OT)
    # Misresolved detection among confident: Mis(1) vs Resolved(0)
    mis_auc_d: float | None = None          # using d̂ (higher → mis)
    mis_auc_a: float | None = None          # using â (lower → mis)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(labels) < 10 or labels.sum() == 0 or labels.sum() == len(labels):
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def compute_binary_aucs(
    feats: list[SampleFeatures],
    diag: list[dict],
    task: str,
    model: str,
) -> BinaryAUCs:
    gt_map = {d["sample_id"]: d["outcome"] for d in diag}
    matched = [(f, gt_map[f.sample_id]) for f in feats if f.sample_id in gt_map]

    result = BinaryAUCs(task=task, model=model, n=len(matched))

    # ── Brewing detection: outcomes with FPCL vs no_brewing ──
    brew_labels = np.array([
        0 if o in ("no_brewing", "unresolved") else 1 for _, o in matched
    ])
    h_vals = np.array([f.h_hat for f, _ in matched])
    s_vals = np.array([f.s_hat for f, _ in matched])
    result.brewing_auc_h = _safe_auc(brew_labels, -h_vals)   # lower h → brewing
    result.brewing_auc_s = _safe_auc(brew_labels, s_vals)     # higher s → brewing

    # ── OT detection: Overprocessed(1) vs Resolved(0) ──
    ot_pairs = [(f, o) for f, o in matched if o in ("overprocessed", "resolved")]
    if len(ot_pairs) > 10:
        ot_labels = np.array([1 if o == "overprocessed" else 0 for _, o in ot_pairs])
        result.ot_auc_h = _safe_auc(ot_labels, np.array([f.h_hat for f, _ in ot_pairs]))
        result.ot_auc_d = _safe_auc(ot_labels, np.array([f.d_hat for f, _ in ot_pairs]))
        result.ot_auc_a = _safe_auc(ot_labels, -np.array([f.a_hat for f, _ in ot_pairs]))

    # ── Misresolved detection: Mis(1) vs Resolved(0) ──
    mis_pairs = [(f, o) for f, o in matched if o in ("misresolved", "resolved")]
    if len(mis_pairs) > 10:
        mis_labels = np.array([1 if o == "misresolved" else 0 for _, o in mis_pairs])
        result.mis_auc_d = _safe_auc(mis_labels, np.array([f.d_hat for f, _ in mis_pairs]))
        result.mis_auc_a = _safe_auc(mis_labels, -np.array([f.a_hat for f, _ in mis_pairs]))

    return result


def print_binary_aucs(aucs: list[BinaryAUCs]):
    print("\n" + "=" * 100)
    print("  Binary Signal AUC (continuous scores — no threshold required)")
    print("=" * 100)

    by_model: dict[str, list[BinaryAUCs]] = defaultdict(list)
    for a in aucs:
        by_model[a.model].append(a)

    def _fmt(v): return f"{v:.3f}" if v is not None else "  -  "

    # Per-model table
    for model, aas in sorted(by_model.items()):
        print(f"\n  {model}")
        print(f"  {'Task':<18} {'Brew(ĥ)':>8} {'Brew(ŝ)':>8} "
              f"{'OT(ĥ)':>7} {'OT(d̂)':>7} {'OT(â)':>7} "
              f"{'Mis(d̂)':>8} {'Mis(â)':>7}")
        for a in sorted(aas, key=lambda x: x.task):
            print(f"  {a.task:<18} {_fmt(a.brewing_auc_h):>8} {_fmt(a.brewing_auc_s):>8} "
                  f"{_fmt(a.ot_auc_h):>7} {_fmt(a.ot_auc_d):>7} {_fmt(a.ot_auc_a):>7} "
                  f"{_fmt(a.mis_auc_d):>8} {_fmt(a.mis_auc_a):>7}")

    # Grand summary: weighted average AUC per signal
    print(f"\n{'─' * 100}")
    print(f"  {'Model':<42} {'Brew(ĥ)':>8} {'OT(ĥ)':>7} {'OT(d̂)':>7} {'Mis(d̂)':>8}")
    print(f"  {'':─<42} {'':─>8} {'':─>7} {'':─>7} {'':─>8}")
    for model, aas in sorted(by_model.items()):
        def _wavg(attr):
            vals = [(getattr(a, attr), a.n) for a in aas if getattr(a, attr) is not None]
            if not vals: return None
            return sum(v * n for v, n in vals) / sum(n for _, n in vals)
        print(f"  {model:<42} {_fmt(_wavg('brewing_auc_h')):>8} "
              f"{_fmt(_wavg('ot_auc_h')):>7} {_fmt(_wavg('ot_auc_d')):>7} "
              f"{_fmt(_wavg('mis_auc_d')):>8}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# §5  Threshold sweep (optional)
# ═══════════════════════════════════════════════════════════════════════════

def sweep_thresholds(
    all_feats: list[SampleFeatures],
    all_gt: dict[str, str],
) -> Thresholds:
    """Grid search over threshold space — find combo that maximises κ.

    Still closed-form rules, just with empirically optimal cut-points.
    Useful as an upper bound on what this decision surface can achieve.
    """
    best_kappa = -1.0
    best_t = DEFAULT_THRESH

    for alpha in np.arange(0.10, 0.60, 0.05):
        for beta in np.arange(0.40, 0.75, 0.03):
            for gamma in np.arange(0.10, 0.40, 0.03):
                for delta in np.arange(0.30, 0.80, 0.05):
                    t = Thresholds(alpha, beta, gamma, delta)
                    yt, yp = [], []
                    for f in all_feats:
                        gt = all_gt.get(f.sample_id)
                        if gt is None:
                            continue
                        yt.append(gt)
                        yp.append(classify(f, t))
                    k = cohens_kappa(yt, yp)
                    if k > best_kappa:
                        best_kappa = k
                        best_t = t

    print(f"  Sweep best κ = {best_kappa:.4f}")
    print(f"  α_brew={best_t.alpha_brew:.2f}  β_H={best_t.beta_H:.2f}  "
          f"γ_D={best_t.gamma_D:.2f}  δ_A={best_t.delta_A:.2f}")
    return best_t


# ═══════════════════════════════════════════════════════════════════════════
# §6  Pretty printing
# ═══════════════════════════════════════════════════════════════════════════

def print_feature_stats(all_feats: list[SampleFeatures], all_gt: dict[str, str]):
    """Print feature distributions by outcome — sanity check."""
    by_out: dict[str, list[SampleFeatures]] = defaultdict(list)
    for f in all_feats:
        gt = all_gt.get(f.sample_id)
        if gt:
            by_out[gt].append(f)

    print("\n  Feature distributions by outcome (all models pooled)")
    print(f"  {'Outcome':<16} {'N':>5} {'ĥ':>7} {'d̂':>7} {'â':>7} {'ŝ':>7}")
    print(f"  {'':─<16} {'':─>5} {'':─>7} {'':─>7} {'':─>7} {'':─>7}")
    for o in OUTCOME_ORDER:
        fs = by_out.get(o, [])
        if not fs:
            continue
        print(f"  {o:<16} {len(fs):>5}"
              f" {np.mean([f.h_hat for f in fs]):>7.3f}"
              f" {np.mean([f.d_hat for f in fs]):>7.3f}"
              f" {np.mean([f.a_hat for f in fs]):>7.3f}"
              f" {np.mean([f.s_hat for f in fs]):>7.3f}")
    print()


def print_results(rows: list[EvalRow], thresh: Thresholds):
    print("\n" + "=" * 92)
    print("  GT-free Closed-Form Outcome Discrimination")
    print(f"  Thresholds: α={thresh.alpha_brew:.2f}  β={thresh.beta_H:.2f}  "
          f"γ={thresh.gamma_D:.2f}  δ={thresh.delta_A:.2f}")
    print("=" * 92)

    by_model: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)

    for model, rs in sorted(by_model.items()):
        print(f"\n{'─' * 92}")
        print(f"  {model}  (L={rs[0].n_layers})")
        print(f"{'─' * 92}")
        print(f"  {'Task':<18} {'N':>5} {'Acc':>7} {'κ':>7}  "
              f"{'Res-F1':>7} {'OT-F1':>7} {'Mis-F1':>7} {'Unr-F1':>7} {'NB-F1':>7}")

        for r in sorted(rs, key=lambda x: x.task):
            def _f1(o):
                po = r.per_outcome.get(o, {})
                return f"{po['f1']:.3f}" if po.get("support", 0) > 0 else "   -  "
            print(f"  {r.task:<18} {r.n:>5} {r.acc:>7.3f} {r.kappa:>7.3f}  "
                  f"{_f1('resolved'):>7} {_f1('overprocessed'):>7} "
                  f"{_f1('misresolved'):>7} {_f1('unresolved'):>7} {_f1('no_brewing'):>7}")

        total = sum(r.n for r in rs)
        if total:
            wa = sum(r.acc * r.n for r in rs) / total
            wk = sum(r.kappa * r.n for r in rs) / total
            print(f"  {'WEIGHTED':.<18} {total:>5} {wa:>7.3f} {wk:>7.3f}")

    # Grand summary
    print(f"\n{'=' * 92}")
    print(f"  {'Model':<42} {'N':>6} {'Acc':>7} {'κ':>7}")
    print(f"  {'':─<42} {'':─>6} {'':─>7} {'':─>7}")
    for model, rs in sorted(by_model.items()):
        total = sum(r.n for r in rs)
        wa = sum(r.acc * r.n for r in rs) / total if total else 0
        wk = sum(r.kappa * r.n for r in rs) / total if total else 0
        print(f"  {model:<42} {total:>6} {wa:>7.3f} {wk:>7.3f}")

    all_n = sum(r.n for r in rows)
    if all_n:
        ga = sum(r.acc * r.n for r in rows) / all_n
        gk = sum(r.kappa * r.n for r in rows) / all_n
        print(f"  {'':─<42} {'':─>6} {'':─>7} {'':─>7}")
        print(f"  {'ALL':.<42} {all_n:>6} {ga:>7.3f} {gk:>7.3f}")
    print(f"\n  Random 5-way: Acc ≈ 0.200, κ = 0.000\n")


# ═══════════════════════════════════════════════════════════════════════════
# §7  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="Grid-search thresholds (upper bound)")
    ap.add_argument("--sweep-per-model", action="store_true",
                    help="Grid-search thresholds per model")
    ap.add_argument("--dump-json", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else TASKS

    all_models = set()
    for t in tasks:
        all_models.update(discover_models(t))
    all_models = sorted(all_models)
    if args.models:
        req = set(args.models.split(","))
        all_models = [m for m in all_models if m in req]

    print(f"Tasks: {tasks}")
    print(f"Models ({len(all_models)}): {', '.join(all_models)}")

    # ── Load everything ──
    model_task_feats: dict[str, dict[str, list[SampleFeatures]]] = {}
    model_task_diag: dict[str, dict[str, list[dict]]] = {}
    all_feats: list[SampleFeatures] = []
    all_gt: dict[str, str] = {}

    for model in all_models:
        model_task_feats[model] = {}
        model_task_diag[model] = {}
        for task in tasks:
            if model not in discover_models(task):
                continue
            try:
                csd, probe, diag = load_task_model(task, model)
            except Exception as e:
                print(f"  SKIP {task}/{model}: {e}")
                continue
            feats = build_features(csd, probe)
            if not feats:
                continue
            model_task_feats[model][task] = feats
            all_feats.extend(feats)
            if diag:
                model_task_diag[model][task] = diag
                for d in diag:
                    all_gt[d["sample_id"]] = d["outcome"]

    # ── Feature distribution overview ──
    print_feature_stats(all_feats, all_gt)

    # ── Determine thresholds ──
    if args.sweep:
        print("Running global threshold sweep...")
        thresh = sweep_thresholds(all_feats, all_gt)
    else:
        thresh = DEFAULT_THRESH
        print(f"Using default thresholds: α={thresh.alpha_brew:.2f}  β={thresh.beta_H:.2f}  "
              f"γ={thresh.gamma_D:.2f}  δ={thresh.delta_A:.2f}")

    # ── Evaluate ──
    rows: list[EvalRow] = []
    for model in all_models:
        if args.sweep_per_model:
            # per-model sweep
            m_feats = [f for fs in model_task_feats.get(model, {}).values() for f in fs]
            m_gt = {}
            for task, diags in model_task_diag.get(model, {}).items():
                for d in diags:
                    m_gt[d["sample_id"]] = d["outcome"]
            if m_feats and m_gt:
                print(f"  Sweeping {model}...")
                m_thresh = sweep_thresholds(m_feats, m_gt)
            else:
                m_thresh = thresh
        else:
            m_thresh = thresh

        for task in tasks:
            feats = model_task_feats.get(model, {}).get(task)
            diag = model_task_diag.get(model, {}).get(task)
            if feats is None or diag is None:
                continue
            rows.append(evaluate(feats, diag, m_thresh, task, model))

    print_results(rows, thresh)

    # ── Binary AUC analysis (threshold-free) ──
    all_aucs: list[BinaryAUCs] = []
    for model in all_models:
        for task in tasks:
            feats = model_task_feats.get(model, {}).get(task)
            diag = model_task_diag.get(model, {}).get(task)
            if feats is None or diag is None:
                continue
            all_aucs.append(compute_binary_aucs(feats, diag, task, model))
    print_binary_aucs(all_aucs)

    if args.dump_json:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rows:
            out.append({
                "task": r.task, "model": r.model,
                "n": r.n, "n_layers": r.n_layers,
                "acc": r.acc, "kappa": r.kappa,
                "per_outcome": r.per_outcome,
                "confusion": r.confusion,
            })
        path = OUT_DIR / "closed_form_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
