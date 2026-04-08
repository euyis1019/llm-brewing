#!/usr/bin/env python3
"""Run re-injection for a single injection mode. Called by reinjection_modes_compare.py."""

import argparse
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
TASKS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
MODEL_DIR = "Qwen__Qwen2.5-Coder-7B"
INJECT_FROM_END = [4, 3, 2]
ANSWER_SPACE = [str(d) for d in range(10)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logger.info("Loading model for mode=%s alpha=%.2f ...", args.mode, args.alpha)
    from nnsight import LanguageModel
    model = LanguageModel(str(MODEL_CACHE / MODEL_ID), device_map="auto", torch_dtype=torch.float16, dispatch=True)
    from brewing.causal.backend import NNsightInterventionBackend, InterventionRequest
    backend = NNsightInterventionBackend(model, model.tokenizer)
    logger.info("Model loaded.")

    results = []
    for task in TASKS:
        logger.info("=== %s [%s] ===", task, args.mode)
        diags = load_diagnostics(task)
        hs, meta = load_cache(task)
        samples = load_samples(task)
        n_layers = hs.shape[1]
        cache_map = {sid: i for i, sid in enumerate(meta["sample_ids"])}

        ur, res = [], []
        for d in diags:
            sid = d["sample_id"]
            if sid not in cache_map or sid not in samples:
                continue
            e = {"diag": d, "idx": cache_map[sid], "sample": samples[sid]}
            if d["outcome"] == "unresolved" and d["fpcl"] is not None:
                ur.append(e)
            elif d["outcome"] == "resolved" and d["fjc"] is not None:
                res.append(e)

        entries = ur[:50] + res[:25]
        task_result = {"task": task, "mode": args.mode, "alpha": args.alpha,
                       "n_ur": min(len(ur), 50), "n_res": min(len(res), 25),
                       "rounds": {}}

        for ri, from_end in enumerate(INJECT_FROM_END):
            tl = n_layers - from_end
            if tl < 0:
                continue
            requests = []
            request_meta = []
            for e in entries:
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
                    injection_mode=args.mode,
                    injection_alpha=args.alpha,
                    target_original_hidden=hs[e["idx"], tl] if args.mode != "replace" else None,
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

            task_result["rounds"][f"round_{ri+1}"] = {
                "target_layer": tl,
                "ur_n": ur_n, "ur_rescued": ur_ok,
                "ur_rate": ur_ok / ur_n if ur_n else None,
                "res_n": res_n, "res_maintained": res_ok,
                "res_rate": res_ok / res_n if res_n else None,
            }
            logger.info("  L=%d: UR %d/%d (%.1f%%) Res %d/%d (%.1f%%)",
                        tl, ur_ok, ur_n, ur_ok / ur_n * 100 if ur_n else 0,
                        res_ok, res_n, res_ok / res_n * 100 if res_n else 0)

        results.append(task_result)

        # Cleanup between tasks
        import gc
        backend._baseline_cache.clear()
        backend._output_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
