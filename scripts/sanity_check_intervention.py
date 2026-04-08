#!/usr/bin/env python3
"""Sanity check: verify intervention mechanism on Resolved computing samples.

Tests:
1. Offset=0 (no-op): inject layer X output back at layer X. Must match
   unintervened output — if not, the nnsight intervention is broken.
2. Offset=+2 with restricted digit argmax: Resolved control should maintain
   ~90%+ (validates that answer_space fix works).
3. Offset=+2 global argmax: compare against test 2 to show the
   answer_space impact.
4. Re-injection with norm_match vs replace: show that norm matching
   improves over raw replacement for early→late injection.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "brewing_output"
MODEL_CACHE = Path("/home/gyf/CUE/models")
SEED = "seed42"
TASK = "computing"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
MODEL_DIR = "Qwen__Qwen2.5-Coder-7B"
N_SAMPLES = 10
ANSWER_SPACE = [str(d) for d in range(10)]


def load_data():
    with open(BASE / "results" / "cuebench" / "eval" / TASK / SEED / MODEL_DIR / "diagnostics.json") as f:
        diags = json.load(f)["sample_diagnostics"]
    cache_dir = BASE / "caches" / "cuebench" / "eval" / TASK / SEED / MODEL_DIR
    hs = np.load(cache_dir / "hidden_states.npz")["hidden_states"]
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    from brewing.schema import Sample
    with open(BASE / "datasets" / "cuebench" / "eval" / TASK / SEED / "samples.json") as f:
        samples = {s["id"]: Sample(**s) for s in json.load(f)}
    return diags, hs, meta, samples


def main():
    diags, hs, meta, samples = load_data()
    cache_map = {sid: i for i, sid in enumerate(meta["sample_ids"])}
    n_layers = hs.shape[1]

    # Select Resolved samples with valid FJC
    resolved = []
    for d in diags:
        sid = d["sample_id"]
        if (d["outcome"] == "resolved" and d["fjc"] is not None
                and sid in cache_map and sid in samples):
            resolved.append({
                "diag": d, "idx": cache_map[sid], "sample": samples[sid]
            })
    resolved = resolved[:N_SAMPLES]
    logger.info("Selected %d Resolved samples (FJC range: %s)",
                len(resolved), [e["diag"]["fjc"] for e in resolved])

    # Also select a few UR samples for re-injection test
    unresolved = []
    for d in diags:
        sid = d["sample_id"]
        if (d["outcome"] == "unresolved" and d["fpcl"] is not None
                and sid in cache_map and sid in samples):
            unresolved.append({
                "diag": d, "idx": cache_map[sid], "sample": samples[sid]
            })
    unresolved = unresolved[:N_SAMPLES]

    # Load model
    logger.info("Loading model...")
    from nnsight import LanguageModel
    model = LanguageModel(
        str(MODEL_CACHE / MODEL_ID),
        device_map="auto", torch_dtype=torch.float16, dispatch=True,
    )
    from brewing.causal.backend import NNsightInterventionBackend, InterventionRequest
    backend = NNsightInterventionBackend(model, model.tokenizer)
    logger.info("Model loaded. dtype=%s device=%s", backend._dtype, backend._device)

    # ── Test 1: Offset=0 no-op ──
    print("\n=== Test 1: Offset=0 no-op (inject layer X back at layer X) ===")
    for e in resolved[:3]:
        fjc = e["diag"]["fjc"]
        # Unintervened
        orig = backend.get_model_output(e["sample"].prompt)
        # Intervened at same layer (should be identical)
        resp = backend.run_interventions([InterventionRequest(
            sample_id=e["sample"].id,
            source_prompt=e["sample"].prompt,
            target_prompt=e["sample"].prompt,
            source_hidden=hs[e["idx"], fjc],
            target_layer=fjc,
            target_position=-1,
            answer_space=ANSWER_SPACE,
            baseline_subtract=False,
        )])[0]
        match = "PASS" if resp.intervened_output == e["sample"].answer else "FAIL"
        print(f"  {e['sample'].id}: FJC={fjc}, answer={e['sample'].answer}, "
              f"orig={orig}, intervened={resp.intervened_output} [{match}]")

    # ── Test 2: Offset=+2, restricted digit argmax ──
    print("\n=== Test 2: Offset=+2, answer_space=digits, no baseline_subtract ===")
    ok, total = 0, 0
    for e in resolved:
        fjc = e["diag"]["fjc"]
        tl = fjc + 2
        if tl >= n_layers:
            continue
        resp = backend.run_interventions([InterventionRequest(
            sample_id=e["sample"].id,
            source_prompt=e["sample"].prompt,
            target_prompt=e["sample"].prompt,
            source_hidden=hs[e["idx"], fjc],
            target_layer=tl,
            target_position=-1,
            answer_space=ANSWER_SPACE,
            baseline_subtract=False,
        )])[0]
        total += 1
        if resp.intervened_output == e["sample"].answer:
            ok += 1
    print(f"  Resolved maintain: {ok}/{total} ({ok/total*100:.1f}%)")

    # ── Test 3: Offset=+2, global argmax (old broken behavior) ──
    print("\n=== Test 3: Offset=+2, global argmax (answer_space=None) ===")
    ok3, total3 = 0, 0
    for e in resolved:
        fjc = e["diag"]["fjc"]
        tl = fjc + 2
        if tl >= n_layers:
            continue
        resp = backend.run_interventions([InterventionRequest(
            sample_id=e["sample"].id,
            source_prompt=e["sample"].prompt,
            target_prompt=e["sample"].prompt,
            source_hidden=hs[e["idx"], fjc],
            target_layer=tl,
            target_position=-1,
            answer_space=None,
        )])[0]
        total3 += 1
        if resp.intervened_output.strip() == e["sample"].answer:
            ok3 += 1
        else:
            # Show what the global argmax produces instead
            print(f"    {e['sample'].id}: expected={e['sample'].answer}, got='{resp.intervened_output}'")
    print(f"  Resolved maintain (global): {ok3}/{total3} ({ok3/total3*100:.1f}%)")
    print(f"  >> Delta from restricted argmax: {ok/total*100 - ok3/total3*100:+.1f}pp")

    # ── Test 4: Re-injection modes comparison ──
    if unresolved:
        print("\n=== Test 4: Re-injection at L-3, replace vs norm_match ===")
        tl = n_layers - 3
        for mode in ["replace", "norm_match"]:
            ok4, total4 = 0, 0
            for e in unresolved:
                fpcl = e["diag"]["fpcl"]
                resp = backend.run_interventions([InterventionRequest(
                    sample_id=e["sample"].id,
                    source_prompt=e["sample"].prompt,
                    target_prompt=e["sample"].prompt,
                    source_hidden=hs[e["idx"], fpcl],
                    target_layer=tl,
                    target_position=-1,
                    answer_space=ANSWER_SPACE,
                    baseline_subtract=False,
                    injection_mode=mode,
                )])[0]
                total4 += 1
                if resp.intervened_output == e["sample"].answer:
                    ok4 += 1
            print(f"  UR rescue ({mode}): {ok4}/{total4} ({ok4/total4*100:.1f}%)")

        # Also test Resolved control with norm_match
        for mode in ["replace", "norm_match"]:
            ok4r, total4r = 0, 0
            for e in resolved:
                fpcl = e["diag"].get("fpcl")
                if fpcl is None:
                    continue
                resp = backend.run_interventions([InterventionRequest(
                    sample_id=e["sample"].id,
                    source_prompt=e["sample"].prompt,
                    target_prompt=e["sample"].prompt,
                    source_hidden=hs[e["idx"], fpcl],
                    target_layer=tl,
                    target_position=-1,
                    answer_space=ANSWER_SPACE,
                    baseline_subtract=False,
                    injection_mode=mode,
                )])[0]
                total4r += 1
                if resp.intervened_output == e["sample"].answer:
                    ok4r += 1
            if total4r:
                print(f"  Res maintain ({mode}): {ok4r}/{total4r} ({ok4r/total4r*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
