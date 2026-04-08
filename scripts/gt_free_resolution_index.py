#!/usr/bin/env python3
"""GT-free Resolution Index — detecting Resolved via internal dynamics.

Core insight: Resolution is not only a terminal state but a *dynamical
process*.  A model that merely arrives at low entropy is not the same as
one that underwent a clear phase transition — a sharp, monotonic collapse
of uncertainty followed by stable convergence.  We encode both endpoint
quality and path dynamics into a single scalar.

═══════════════════════════════════════════════════════════════════════════

§A  Layer-wise signals

    H_C^ℓ  = −Σ Φ_C[t] ln Φ_C[t]           CSD entropy             [0, ln C]
    H_P^ℓ  = −Σ Φ_P[t] ln Φ_P[t]           Probe entropy           [0, ln C]
    D^ℓ    = JSD(Φ_P^ℓ ‖ Φ_C^ℓ)            Probe–CSD divergence    [0, ln 2]
    A^ℓ    = 𝟙[argmax Φ_P = argmax Φ_C]    Argmax agreement        {0,1}

§B  Gradient signals (Savitzky-Golay, w=7, p=3)

    J^ℓ    = −∂H_C/∂ℓ                       "information flux" — positive when
                                              entropy decreases (information crystallises)
    ∂D/∂ℓ                                    JSD rate of change

§C  Normalised scalar features

  Endpoint:
    ĥ      = H̄_C^tail / ln C               residual CSD uncertainty
    ĥ_P    = H̄_P^tail / ln C               residual probe uncertainty
    d̂      = D̄^tail / ln 2                 probe–CSD divergence
    â      = Ā^tail                          argmax agreement rate

  Dynamics:
    ĵ_peak = max_ℓ J^ℓ / J₀                 normalised peak information flux
                 where J₀ = ln C / L          (one-layer uniform rate)
    ĵ_int  = ∫ max(J^ℓ,0) dℓ / ln C         total positive flux / max possible
    σ_tail = std(H_C^tail) / ln C            tail entropy stability
    ṫ_tail = mean(∂D/∂ℓ |_tail) / (ln2/L)   normalised tail JSD trend
    μ      = Σ 𝟙[J^ℓ > 0] / L              monotonicity fraction

═══════════════════════════════════════════════════════════════════════════

§D  Resolution Indices

Index 1 — ρ_mult  (endpoint only, baseline)
    ρ₁ = (1 − ĥ)(1 − d̂) · â

Index 2 — ρ_geo  (Boltzmann distance, endpoint only)
    ρ₂ = exp(−½ · Ψ),    Ψ = ĥ + ĥ_P + d̂

Index 3 — ρ_dyn  (Dynamical Resolution Functional)

    The key innovation: resolution = good endpoint × clear phase transition
    × stable convergence.  Three multiplicative gates:

    ρ₃ = Φ_state · Φ_flux · Φ_stab

    Φ_state = (1 − ĥ)(1 − d̂)
        Terminal state quality: CSD is confident and agrees with probe.

    Φ_flux  = tanh(ĵ_peak · μ)
        Information flux gate: the model experienced a strong, monotonic
        entropy collapse.  tanh saturates so that very large fluxes don't
        dominate; multiplication by μ (monotonicity) penalises oscillatory
        paths — a clean phase transition has μ → 1.

    Φ_stab  = exp(−λ · Ω)
        Tail stability gate (Lyapunov-inspired): penalises entropy
        increase in the tail, which is the signature of overprocessing.

        Ω = ∫_tail max(∂H_C/∂ℓ, 0) dℓ / ln C     "tail entropy regrowth"

        λ controls the penalty sharpness.  We use λ = 5 (one percent of
        regrowth halves the score).  Resolved samples have Ω ≈ 0;
        Overprocessed samples have Ω > 0 due to late-layer entropy rise.

Index 4 — ρ_path  (Resolution Functional with Path Integral)

    The central formulation.  Resolution requires not only convergence of the
    terminal state but also evidence of a clear *phase transition* in the
    layer-wise dynamics — a criterion that purely endpoint statistics cannot
    capture.

    ρ₄ = Φ_state · (1 + β · Φ_path)

    Φ_state = (1 − ĥ)(1 − d̂) · â                    [terminal convergence]

    Φ_path  = 𝒥^α · Λ^(1−α)                          [path quality]

    where:

    𝒥 = (1/ln C) ∫₀ᴸ max(−∂ℓ H_C, 0) dℓ             "information flux integral"
        Total entropy removed during the forward pass, normalised by the
        maximum possible (ln C).  Resolved samples have 𝒥 → 1; Unresolved
        samples have 𝒥 → 0.  This is ĵ_int.

    Λ = exp(−Var(H_C^tail) / σ₀²)                     "Lyapunov stability"
        σ₀ = ln C / (4L)
        Exponential penalty on tail entropy variance.  A resolved computation
        converges to a fixed point (Var → 0, Λ → 1); an overprocessed one
        exhibits late-layer oscillation (Var > 0, Λ < 1).

    β = 0.3   (bonus scale — path can boost score by ≤30%)
    α = 0.5   (equal weight to flux and stability)

    Interpretation: Φ_state certifies the destination; Φ_path certifies
    the journey.  The multiplicative-additive hybrid ensures that gradient
    information can only *help* — a sample with perfect terminal state but
    unclear dynamics still gets ρ₄ ≈ Φ_state, while a sample with a clean
    phase transition gets an additional boost.

═══════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/gt_free_resolution_index.py
    python scripts/gt_free_resolution_index.py --models Qwen__Qwen2.5-Coder-7B
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "brewing_output"
EVAL_ROOT = BASE / "results" / "cuebench" / "eval"
OUT_DIR = BASE / "artifacts" / "gt_free_v2"

TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]
SEED = "seed42"
C = 10
LN_C = float(np.log(C))
LN_2 = float(np.log(2))
LAMBDA_STAB = 5.0   # Lyapunov penalty coefficient (ρ_dyn)
BETA_PATH = 0.3     # path bonus scale (ρ_path)
ALPHA_PATH = 0.5    # flux vs stability balance (ρ_path)


# ═══════════════════════════════════════════════════════════════════════════
# §1  Signal & gradient extraction
# ═══════════════════════════════════════════════════════════════════════════

def _sg_deriv(sig: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    n = len(sig)
    w = min(window, n if n % 2 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 else poly + 3)
    if w > n:
        return np.gradient(sig)
    return savgol_filter(sig, window_length=w, polyorder=poly, deriv=1)


@dataclass
class SampleScores:
    sample_id: str
    n_layers: int
    # ── endpoint features ──
    h_hat: float         # ĥ
    h_probe: float       # ĥ_P
    d_hat: float         # d̂
    a_hat: float         # â
    # ── gradient features ──
    j_peak: float        # ĵ_peak  (normalised peak information flux)
    j_int: float         # ĵ_int   (normalised total positive flux)
    sigma_tail: float    # σ_tail  (tail entropy stability)
    jsd_tail_trend: float  # ṫ_tail  (normalised tail JSD trend)
    mu: float            # μ       (monotonicity fraction)
    omega: float         # Ω       (tail entropy regrowth)
    lyapunov: float      # Λ       (Lyapunov stability)
    # ── resolution indices ──
    rho_mult: float      # ρ₁
    rho_geo: float       # ρ₂
    rho_dyn: float       # ρ₃
    rho_path: float      # ρ₄ (Resolution Functional with Path Integral)


def extract(
    csd_confs: np.ndarray,     # (L, C)
    probe_confs: np.ndarray,   # (L, C)
    sample_id: str,
) -> SampleScores:
    L = csd_confs.shape[0]
    eps = 1e-10
    tail = int(L * 0.75)

    csd_p = np.clip(csd_confs, eps, None)
    csd_p /= csd_p.sum(axis=1, keepdims=True)
    probe_p = np.clip(probe_confs, eps, None)
    probe_p /= probe_p.sum(axis=1, keepdims=True)

    # ── layer-wise signals ──
    H_csd = sp_entropy(csd_p.T)
    H_probe = sp_entropy(probe_p.T)
    jsd_arr = np.array([jensenshannon(probe_p[l], csd_p[l]) ** 2 for l in range(L)])
    agree = (np.argmax(csd_p, axis=1) == np.argmax(probe_p, axis=1)).astype(float)

    # ── gradients ──
    dH = _sg_deriv(H_csd)       # ∂H_C/∂ℓ
    dD = _sg_deriv(jsd_arr)     # ∂D/∂ℓ
    J = -dH                     # information flux: positive = entropy decreasing

    J0 = LN_C / L              # uniform-rate normaliser

    # ── endpoint features ──
    h_hat = float(np.mean(H_csd[tail:])) / LN_C
    h_probe = float(np.mean(H_probe[tail:])) / LN_C
    d_hat = float(np.mean(jsd_arr[tail:])) / LN_2
    a_hat = float(np.mean(agree[tail:]))

    # ── gradient features ──
    j_peak = float(np.max(J)) / J0                              # ĵ_peak
    j_int = float(np.sum(np.maximum(J, 0))) / LN_C              # ĵ_int (total positive flux / max)
    sigma_tail = float(np.std(H_csd[tail:])) / LN_C             # σ_tail
    jsd_tail_trend = float(np.mean(dD[tail:])) / (LN_2 / L)     # ṫ_tail
    mu = float(np.mean(J > 0))                                  # μ (monotonicity)
    omega = float(np.sum(np.maximum(dH[tail:], 0))) / LN_C      # Ω (tail regrowth)

    # ── Lyapunov stability ──
    sigma_0 = LN_C / (4 * L)
    tail_var = float(np.var(H_csd[tail:]))
    lyapunov = float(np.exp(-tail_var / (sigma_0 ** 2)))         # Λ

    # ── resolution indices ──

    # ρ₁ — multiplicative endpoint AND-gate
    rho_mult = (1.0 - h_hat) * (1.0 - d_hat) * a_hat

    # ρ₂ — Boltzmann / info-geometric
    psi = h_hat + h_probe + d_hat
    rho_geo = float(np.exp(-0.5 * psi))

    # ρ₃ — dynamical (multiplicative gates)
    phi_state_dyn = (1.0 - h_hat) * (1.0 - d_hat)
    phi_flux = float(np.tanh(j_peak * mu))
    phi_stab = float(np.exp(-LAMBDA_STAB * omega))
    rho_dyn = phi_state_dyn * phi_flux * phi_stab

    # ρ₄ — Resolution Functional with Path Integral
    #   ρ₄ = Φ_state · (1 + β · 𝒥^α · Λ^(1-α))
    phi_state = (1.0 - h_hat) * (1.0 - d_hat) * a_hat
    phi_path = (j_int ** ALPHA_PATH) * (lyapunov ** (1.0 - ALPHA_PATH))
    rho_path = phi_state * (1.0 + BETA_PATH * phi_path)

    return SampleScores(
        sample_id=sample_id, n_layers=L,
        h_hat=h_hat, h_probe=h_probe, d_hat=d_hat, a_hat=a_hat,
        j_peak=j_peak, j_int=j_int, sigma_tail=sigma_tail,
        jsd_tail_trend=jsd_tail_trend, mu=mu, omega=omega,
        lyapunov=lyapunov,
        rho_mult=rho_mult, rho_geo=rho_geo,
        rho_dyn=rho_dyn, rho_path=rho_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §2  Data loading
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
    diag = None
    dp = base / "diagnostics.json"
    if dp.exists():
        with open(dp) as f:
            diag = json.load(f)["sample_diagnostics"]
    return csd, probe, diag


def build_scores(csd_list: list[dict], probe_list: list[dict]) -> list[SampleScores]:
    probe_map = {s["sample_id"]: s for s in probe_list}
    out = []
    for cs in csd_list:
        sid = cs["sample_id"]
        ps = probe_map.get(sid)
        if ps is None or ps.get("layer_confidences") is None:
            continue
        out.append(extract(
            np.array(cs["layer_confidences"]),
            np.array(ps["layer_confidences"]),
            sid,
        ))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# §3  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

INDEX_NAMES = ["rho_mult", "rho_geo", "rho_dyn", "rho_path"]
FEAT_NAMES = [
    "h_hat", "h_probe", "d_hat", "a_hat",
    "j_peak", "j_int", "sigma_tail", "jsd_tail_trend", "mu", "omega", "lyapunov",
]
PRETTY = {
    "rho_mult": "ρ_mult", "rho_geo": "ρ_geo",
    "rho_dyn": "ρ_dyn", "rho_path": "ρ_path",
}


@dataclass
class AUCRow:
    task: str
    model: str
    n: int
    n_resolved: int
    n_layers: int
    aucs: dict[str, float]
    best_f1s: dict[str, float]


def _safe_auc(labels, scores):
    if len(labels) < 10 or labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def eval_binary(scores: list[SampleScores], diag: list[dict],
                task: str, model: str) -> AUCRow | None:
    gt = {d["sample_id"]: d["outcome"] for d in diag}
    labels_list, vecs = [], {n: [] for n in INDEX_NAMES}
    for s in scores:
        o = gt.get(s.sample_id)
        if o is None:
            continue
        labels_list.append(1 if o == "resolved" else 0)
        for n in INDEX_NAMES:
            vecs[n].append(getattr(s, n))

    labels = np.array(labels_list)
    if labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
        return None

    aucs, f1s = {}, {}
    for n in INDEX_NAMES:
        sv = np.array(vecs[n])
        auc = _safe_auc(labels, sv)
        aucs[n] = auc if auc is not None else 0.5
        fpr, tpr, th = roc_curve(labels, sv)
        best = np.argmax(tpr - fpr)
        preds = (sv >= th[best]).astype(int)
        f1s[n] = float(f1_score(labels, preds))

    return AUCRow(task, model, len(labels), int(labels.sum()),
                  scores[0].n_layers, aucs, f1s)


def eval_features(all_scores: list[SampleScores], all_gt: dict[str, str]) -> dict[str, float]:
    """AUC of each individual feature for Resolved detection — feature ablation."""
    labels, feat_vecs = [], {n: [] for n in FEAT_NAMES}
    for s in all_scores:
        o = all_gt.get(s.sample_id)
        if o is None:
            continue
        labels.append(1 if o == "resolved" else 0)
        for n in FEAT_NAMES:
            feat_vecs[n].append(getattr(s, n))

    labels = np.array(labels)
    result = {}
    for n in FEAT_NAMES:
        sv = np.array(feat_vecs[n])
        # some features are inversely correlated (h_hat: lower = more resolved)
        auc_pos = _safe_auc(labels, sv)
        auc_neg = _safe_auc(labels, -sv)
        if auc_pos is not None and auc_neg is not None:
            if auc_neg > auc_pos:
                result[n] = (auc_neg, "−")
            else:
                result[n] = (auc_pos, "+")
        elif auc_pos is not None:
            result[n] = (auc_pos, "+")
        else:
            result[n] = (0.5, "?")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# §4  Printing
# ═══════════════════════════════════════════════════════════════════════════

def print_feature_ablation(feat_aucs: dict[str, tuple[float, str]]):
    print("\n" + "=" * 72)
    print("  Individual Feature AUC (Resolved vs Rest, all models pooled)")
    print("=" * 72)
    print(f"  {'Feature':<18} {'Type':<12} {'AUC':>7} {'Dir':>4}")
    print(f"  {'':─<18} {'':─<12} {'':─>7} {'':─>4}")

    # Sort by AUC descending
    endpoint_feats = ["h_hat", "h_probe", "d_hat", "a_hat"]
    gradient_feats = ["j_peak", "j_int", "sigma_tail", "jsd_tail_trend", "mu", "omega", "lyapunov"]

    print("  — Endpoint —")
    for n in sorted(endpoint_feats, key=lambda x: -feat_aucs[x][0]):
        auc, dir_ = feat_aucs[n]
        print(f"  {n:<18} {'endpoint':<12} {auc:>7.3f} {dir_:>4}")

    print("  — Gradient —")
    for n in sorted(gradient_feats, key=lambda x: -feat_aucs[x][0]):
        auc, dir_ = feat_aucs[n]
        print(f"  {n:<18} {'gradient':<12} {auc:>7.3f} {dir_:>4}")
    print()


def print_results(rows: list[AUCRow]):
    print("\n" + "=" * 108)
    print("  GT-free Resolution Index — Resolved vs Rest")
    print("=" * 108)
    print(f"  Indices:")
    print(f"    ρ_mult   = (1−ĥ)(1−d̂)·â                                     endpoint AND-gate")
    print(f"    ρ_geo    = exp(−½·[ĥ+ĥ_P+d̂])                               Boltzmann / info-geometric")
    print(f"    ρ_dyn    = (1−ĥ)(1−d̂) · tanh(ĵ_peak·μ) · exp(−λΩ)         dynamical resolution functional")
    print(f"    ρ_path   = Φ_state·(1+β·𝒥^α·Λ^(1−α))                       path integral functional")
    print()

    by_model: dict[str, list[AUCRow]] = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)

    for model, rs in sorted(by_model.items()):
        print(f"{'─' * 108}")
        print(f"  {model}  (L={rs[0].n_layers})")
        print(f"{'─' * 108}")
        print(f"  {'Task':<18} {'N':>5} {'Res%':>6}"
              f"  {'ρ_mult':>7} {'ρ_geo':>7} {'ρ_dyn':>7} {'ρ_path':>7}"
              f"  {'F1_mult':>8} {'F1_dyn':>7} {'F1_path':>8}")
        for r in sorted(rs, key=lambda x: x.task):
            rp = r.n_resolved / r.n * 100 if r.n else 0
            print(f"  {r.task:<18} {r.n:>5} {rp:>5.1f}%"
                  f"  {r.aucs['rho_mult']:>7.3f} {r.aucs['rho_geo']:>7.3f}"
                  f" {r.aucs['rho_dyn']:>7.3f} {r.aucs['rho_path']:>7.3f}"
                  f"  {r.best_f1s['rho_mult']:>8.3f}"
                  f" {r.best_f1s['rho_dyn']:>7.3f}"
                  f" {r.best_f1s['rho_path']:>8.3f}")
        total = sum(r.n for r in rs)
        if total:
            rt = sum(r.n_resolved for r in rs)
            rp = rt / total * 100
            w = {n: sum(r.aucs[n]*r.n for r in rs)/total for n in INDEX_NAMES}
            wf = {n: sum(r.best_f1s[n]*r.n for r in rs)/total for n in INDEX_NAMES}
            print(f"  {'WEIGHTED':.<18} {total:>5} {rp:>5.1f}%"
                  f"  {w['rho_mult']:>7.3f} {w['rho_geo']:>7.3f}"
                  f" {w['rho_dyn']:>7.3f} {w['rho_path']:>7.3f}"
                  f"  {wf['rho_mult']:>8.3f}"
                  f" {wf['rho_dyn']:>7.3f}"
                  f" {wf['rho_path']:>8.3f}")
        print()

    # Grand summary
    print("=" * 108)
    print(f"  {'Model':<42} {'N':>6} {'Res%':>6}"
          f"  {'ρ_mult':>7} {'ρ_geo':>7} {'ρ_dyn':>7} {'ρ_path':>7}"
          f"  {'F1_best':>8}")
    print(f"  {'':─<42} {'':─>6} {'':─>6}"
          f"  {'':─>7} {'':─>7} {'':─>7} {'':─>7}  {'':─>8}")
    for model, rs in sorted(by_model.items()):
        total = sum(r.n for r in rs)
        rt = sum(r.n_resolved for r in rs)
        rp = rt / total * 100 if total else 0
        w = {n: sum(r.aucs[n]*r.n for r in rs)/total for n in INDEX_NAMES}
        wf = {n: sum(r.best_f1s[n]*r.n for r in rs)/total for n in INDEX_NAMES}
        best_f1 = max(wf.values())
        print(f"  {model:<42} {total:>6} {rp:>5.1f}%"
              f"  {w['rho_mult']:>7.3f} {w['rho_geo']:>7.3f}"
              f" {w['rho_dyn']:>7.3f} {w['rho_path']:>7.3f}"
              f"  {best_f1:>8.3f}")

    all_n = sum(r.n for r in rows)
    if all_n:
        ra = sum(r.n_resolved for r in rows)
        gw = {n: sum(r.aucs[n]*r.n for r in rows)/all_n for n in INDEX_NAMES}
        gwf = {n: sum(r.best_f1s[n]*r.n for r in rows)/all_n for n in INDEX_NAMES}
        best_gf = max(gwf.values())
        print(f"  {'':─<42} {'':─>6} {'':─>6}"
              f"  {'':─>7} {'':─>7} {'':─>7} {'':─>7}  {'':─>8}")
        print(f"  {'ALL':.<42} {all_n:>6} {ra/all_n*100:>5.1f}%"
              f"  {gw['rho_mult']:>7.3f} {gw['rho_geo']:>7.3f}"
              f" {gw['rho_dyn']:>7.3f} {gw['rho_path']:>7.3f}"
              f"  {best_gf:>8.3f}")
    print(f"\n  Random: AUC = 0.500\n")


# ═══════════════════════════════════════════════════════════════════════════
# §5  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--tasks", type=str, default=None)
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

    print(f"Resolved-vs-Rest binary classification")
    print(f"Tasks: {len(tasks)},  Models: {len(all_models)}")

    # ── Load ──
    all_scores: list[SampleScores] = []
    all_gt: dict[str, str] = {}
    task_model_data: dict[str, dict[str, tuple]] = {}

    for model in all_models:
        for task in tasks:
            if model not in discover_models(task):
                continue
            try:
                csd, probe, diag = load_task_model(task, model)
            except Exception as e:
                print(f"  SKIP {task}/{model}: {e}")
                continue
            scores = build_scores(csd, probe)
            if not scores:
                continue
            all_scores.extend(scores)
            task_model_data.setdefault(model, {})[task] = (scores, diag)
            if diag:
                for d in diag:
                    all_gt[d["sample_id"]] = d["outcome"]

    # ── Feature ablation ──
    feat_aucs = eval_features(all_scores, all_gt)
    print_feature_ablation(feat_aucs)

    # ── Per task×model eval ──
    rows: list[AUCRow] = []
    for model in all_models:
        for task in tasks:
            entry = task_model_data.get(model, {}).get(task)
            if entry is None:
                continue
            scores, diag = entry
            if diag is None:
                continue
            row = eval_binary(scores, diag, task, model)
            if row:
                rows.append(row)

    print_results(rows)

    if args.dump_json:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rows:
            out.append({
                "task": r.task, "model": r.model,
                "n": r.n, "n_resolved": r.n_resolved,
                "aucs": r.aucs, "best_f1s": r.best_f1s,
            })
        path = OUT_DIR / "resolution_index_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
