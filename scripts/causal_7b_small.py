#!/usr/bin/env python3
"""Run both layer skipping and re-injection on 7B with small samples.

50 OT + 25 Res for skipping, 50 UR + 25 Res for re-injection, per task.
Single model load, both experiments sequentially.
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
MODEL_CACHE = Path("/path/to/cue/models")
SEED = "seed42"
TASKS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
MODEL_DIR = "Qwen__Qwen2.5-Coder-7B"

SKIP_OFFSETS = [2, 4, 6]
INJECT_FROM_END = [4, 3, 2]
ANSWER_SPACE = [str(d) for d in range(10)]  # digits 0-9


def load_diagnostics(task):
    with open(BASE / "results" / "cuebench" / "eval" / task / SEED / MODEL_DIR / "diagnostics.json") as f:
        return json.load(f)["sample_diagnostics"]

def load_cache(task):
    cache_dir = BASE / "caches" / "cuebench" / "eval" / task / SEED / MODEL_DIR
    hs = np.load(cache_dir / "hidden_states.npz")["hidden_states"]
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    return hs, meta

def load_samples(task):
    from brewing.schema import Sample
    with open(BASE / "datasets" / "cuebench" / "eval" / task / SEED / "samples.json") as f:
        return {s["id"]: Sample(**s) for s in json.load(f)}


def run_interventions_batch(backend, entries, hs, n_layers, source_layer_key, target_layers, label):
    """Generic: run interventions for a list of entries at multiple target layers."""
    from brewing.causal.backend import InterventionRequest

    results = {}
    for tl in target_layers:
        if tl < 0 or tl >= n_layers:
            continue
        requests = []
        request_meta = []
        for e in entries:
            sl = e["diag"][source_layer_key]
            if sl is None:
                continue
            requests.append(InterventionRequest(
                sample_id=e["sample"].id,
                source_prompt=e["sample"].prompt,
                target_prompt=e["sample"].prompt,
                source_hidden=hs[e["idx"], sl],
                target_layer=tl if isinstance(tl, int) else tl(e),
                target_position=-1,
                answer_space=None,
            ))
            request_meta.append({"outcome": e["diag"]["outcome"], "answer": e["sample"].answer})

        responses = backend.run_interventions(requests)
        correct_by_outcome = {}
        total_by_outcome = {}
        for resp, rm in zip(responses, request_meta):
            o = rm["outcome"]
            total_by_outcome[o] = total_by_outcome.get(o, 0) + 1
            if resp.intervened_output.strip() == rm["answer"]:
                correct_by_outcome[o] = correct_by_outcome.get(o, 0) + 1

        results[tl] = {"correct": correct_by_outcome, "total": total_by_outcome}
    return results


def main():
    logger.info("Loading model...")
    from nnsight import LanguageModel
    model_path = MODEL_CACHE / MODEL_ID
    model = LanguageModel(str(model_path), device_map="auto", torch_dtype=torch.float16, dispatch=True)
    tokenizer = model.tokenizer
    logger.info("Model loaded.")

    from brewing.causal.backend import NNsightInterventionBackend, InterventionRequest
    backend = NNsightInterventionBackend(model, tokenizer)

    skip_results = []
    reinj_results = []

    for task in TASKS:
        logger.info("=== %s ===", task)
        diags = load_diagnostics(task)
        hs, meta = load_cache(task)
        samples = load_samples(task)
        n_layers = hs.shape[1]
        cache_map = {sid: i for i, sid in enumerate(meta["sample_ids"])}

        # Categorize
        ot, res, ur = [], [], []
        for d in diags:
            sid = d["sample_id"]
            if sid not in cache_map or sid not in samples:
                continue
            e = {"diag": d, "idx": cache_map[sid], "sample": samples[sid]}
            if d["outcome"] == "overprocessed" and d["fjc"] is not None:
                ot.append(e)
            elif d["outcome"] == "resolved" and d["fjc"] is not None:
                res.append(e)
            elif d["outcome"] == "unresolved" and d["fpcl"] is not None:
                ur.append(e)

        # ── Layer Skipping: 50 OT + 25 Res ──
        skip_entries = ot[:50] + res[:25]
        task_skip = {"task": task, "n_ot": min(len(ot), 50), "n_res": min(len(res), 25), "offsets": {}}

        for offset in SKIP_OFFSETS:
            requests = []
            request_meta = []
            for e in skip_entries:
                fjc = e["diag"]["fjc"]
                tl = fjc + offset
                if tl >= n_layers:
                    continue
                requests.append(InterventionRequest(
                    sample_id=e["sample"].id,
                    source_prompt=e["sample"].prompt,
                    target_prompt=e["sample"].prompt,
                    source_hidden=hs[e["idx"], fjc],
                    target_layer=tl,
                    target_position=-1,
                    answer_space=ANSWER_SPACE,
                    baseline_subtract=False,
                ))
                request_meta.append({"outcome": e["diag"]["outcome"], "answer": e["sample"].answer})

            responses = backend.run_interventions(requests)
            ot_n, ot_ok, res_n, res_ok = 0, 0, 0, 0
            for resp, rm in zip(responses, request_meta):
                c = resp.intervened_output.strip() == rm["answer"]
                if rm["outcome"] == "overprocessed":
                    ot_n += 1; ot_ok += c
                else:
                    res_n += 1; res_ok += c

            task_skip["offsets"][offset] = {
                "ot_n": ot_n, "ot_rescued": ot_ok,
                "ot_rate": ot_ok/ot_n if ot_n else None,
                "res_n": res_n, "res_maintained": res_ok,
                "res_rate": res_ok/res_n if res_n else None,
            }
            logger.info("  SKIP +%d: OT %d/%d (%.1f%%) Res %d/%d (%.1f%%)",
                        offset, ot_ok, ot_n, ot_ok/ot_n*100 if ot_n else 0,
                        res_ok, res_n, res_ok/res_n*100 if res_n else 0)

        skip_results.append(task_skip)

        # ── Re-injection: 50 UR + 25 Res ──
        reinj_entries = ur[:50] + res[:25]
        task_reinj = {"task": task, "n_ur": min(len(ur), 50), "n_res": min(len(res), 25), "rounds": {}}

        for ri, from_end in enumerate(INJECT_FROM_END):
            tl = n_layers - from_end
            if tl < 0:
                continue
            requests = []
            request_meta = []
            for e in reinj_entries:
                fpcl = e["diag"].get("fpcl")
                if fpcl is None:
                    continue
                requests.append(InterventionRequest(
                    sample_id=e["sample"].id,
                    source_prompt=e["sample"].prompt,
                    target_prompt=e["sample"].prompt,
                    source_hidden=hs[e["idx"], fpcl],
                    target_layer=tl,
                    target_position=-1,
                    answer_space=ANSWER_SPACE,
                    baseline_subtract=False,
                ))
                request_meta.append({"outcome": e["diag"]["outcome"], "answer": e["sample"].answer})

            responses = backend.run_interventions(requests)
            ur_n, ur_ok, res_n, res_ok = 0, 0, 0, 0
            for resp, rm in zip(responses, request_meta):
                c = resp.intervened_output.strip() == rm["answer"]
                if rm["outcome"] == "unresolved":
                    ur_n += 1; ur_ok += c
                else:
                    res_n += 1; res_ok += c

            task_reinj["rounds"][f"round_{ri+1}"] = {
                "target_layer": tl,
                "ur_n": ur_n, "ur_rescued": ur_ok,
                "ur_rate": ur_ok/ur_n if ur_n else None,
                "res_n": res_n, "res_maintained": res_ok,
                "res_rate": res_ok/res_n if res_n else None,
            }
            logger.info("  REINJ L=%d: UR %d/%d (%.1f%%) Res %d/%d (%.1f%%)",
                        tl, ur_ok, ur_n, ur_ok/ur_n*100 if ur_n else 0,
                        res_ok, res_n, res_ok/res_n*100 if res_n else 0)

        reinj_results.append(task_reinj)

    # Save
    out_dir = BASE / "artifacts"
    (out_dir / "layer_skipping").mkdir(parents=True, exist_ok=True)
    (out_dir / "reinjection").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layer_skipping" / "results_7B_small.json", "w") as f:
        json.dump(skip_results, f, indent=2)
    with open(out_dir / "reinjection" / "results_7B_small.json", "w") as f:
        json.dump(reinj_results, f, indent=2)

    # Summary
    print("\n=== Layer Skipping (7B, 50 OT + 25 Res per task) ===")
    print(f"{'Task':<18} {'Offset':>7} {'OT_rescue':>15} {'Res_maintain':>15}")
    for r in skip_results:
        for off, d in r["offsets"].items():
            ot_s = f"{d['ot_rescued']}/{d['ot_n']} ({d['ot_rate']*100:.1f}%)" if d['ot_rate'] is not None else "N/A"
            re_s = f"{d['res_maintained']}/{d['res_n']} ({d['res_rate']*100:.1f}%)" if d['res_rate'] is not None else "N/A"
            print(f"  {r['task']:<18} +{off:<6} {ot_s:>15} {re_s:>15}")

    print("\n=== Re-injection (7B, 50 UR + 25 Res per task) ===")
    print(f"{'Task':<18} {'Round':>7} {'Layer':>6} {'UR_rescue':>15} {'Res_maintain':>15}")
    for r in reinj_results:
        for rnd, d in r["rounds"].items():
            ur_s = f"{d['ur_rescued']}/{d['ur_n']} ({d['ur_rate']*100:.1f}%)" if d['ur_rate'] is not None else "N/A"
            re_s = f"{d['res_maintained']}/{d['res_n']} ({d['res_rate']*100:.1f}%)" if d['res_rate'] is not None else "N/A"
            print(f"  {r['task']:<18} {rnd:>7} L={d['target_layer']:>3} {ur_s:>15} {re_s:>15}")


if __name__ == "__main__":
    main()
