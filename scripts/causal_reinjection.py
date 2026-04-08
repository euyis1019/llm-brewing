#!/usr/bin/env python3
"""Re-injection experiment (§3.2, App E.4, A5).

For Unresolved samples: inject the FPCL-layer hidden state (where info
first appeared) into late layers, hoping to "rescue" the sample.

Multi-round: inject at layer (n_layers - k) for k rounds, checking if
the model produces the correct answer after each injection.

Usage:
    python scripts/causal_reinjection.py --smoke
    python scripts/causal_reinjection.py --model 0.5B
    python scripts/causal_reinjection.py --model 7B
"""

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

MODELS = {
    "0.5B": ("Qwen/Qwen2.5-Coder-0.5B", "Qwen__Qwen2.5-Coder-0.5B"),
    "7B": ("Qwen/Qwen2.5-Coder-7B", "Qwen__Qwen2.5-Coder-7B"),
}

# Injection targets: inject at these layers (from end)
# Round k injects at n_layers - INJECT_FROM_END[k]
INJECT_FROM_END = [4, 3, 2]  # 3 rounds: inject at L-4, L-3, L-2


def load_diagnostics(task, model_dir):
    path = BASE / "results" / "cuebench" / "eval" / task / SEED / model_dir / "diagnostics.json"
    with open(path) as f:
        return json.load(f)


def load_cache(task, model_dir):
    cache_dir = BASE / "caches" / "cuebench" / "eval" / task / SEED / model_dir
    hs = np.load(cache_dir / "hidden_states.npz")["hidden_states"]
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)
    return hs, meta


def load_samples(task):
    from brewing.schema import Sample
    path = BASE / "datasets" / "cuebench" / "eval" / task / SEED / "samples.json"
    with open(path) as f:
        return {s["id"]: Sample(**s) for s in json.load(f)}


def run_reinjection_task(model, tokenizer, task, model_dir, max_ur=None, max_res=None):
    from brewing.causal.backend import NNsightInterventionBackend, InterventionRequest

    diag_data = load_diagnostics(task, model_dir)
    diags = diag_data["sample_diagnostics"]
    hs, meta = load_cache(task, model_dir)
    samples = load_samples(task)
    n_layers = hs.shape[1]

    cache_id_to_idx = {sid: i for i, sid in enumerate(meta["sample_ids"])}

    # Select Unresolved (with valid FPCL) and Resolved controls
    ur_samples = []
    res_samples = []
    for d in diags:
        if d["fpcl"] is None:
            continue
        sid = d["sample_id"]
        if sid not in cache_id_to_idx or sid not in samples:
            continue
        entry = {"diag": d, "idx": cache_id_to_idx[sid], "sample": samples[sid]}
        if d["outcome"] == "unresolved":
            ur_samples.append(entry)
        elif d["outcome"] == "resolved":
            res_samples.append(entry)

    if max_ur is not None:
        ur_samples = ur_samples[:max_ur]
    if max_res is not None:
        res_samples = res_samples[:max_res]

    logger.info("  %s: %d Unresolved, %d Resolved control", task, len(ur_samples), len(res_samples))

    backend = NNsightInterventionBackend(model, tokenizer)

    results = {"task": task, "n_layers": n_layers, "rounds": {}}

    for round_idx, from_end in enumerate(INJECT_FROM_END):
        target_layer = n_layers - from_end
        if target_layer < 0:
            continue

        requests = []
        request_meta = []

        for entry in ur_samples + res_samples:
            d = entry["diag"]
            fpcl = d["fpcl"]
            sample = entry["sample"]

            # Inject FPCL hidden state at target_layer of the same prompt
            fpcl_hidden = hs[entry["idx"], fpcl]

            requests.append(InterventionRequest(
                sample_id=sample.id,
                source_prompt=sample.prompt,
                target_prompt=sample.prompt,
                source_hidden=fpcl_hidden,
                target_layer=target_layer,
                target_position=-1,
                answer_space=None,
            ))
            request_meta.append({
                "outcome": d["outcome"],
                "answer": sample.answer,
                "fpcl": fpcl,
            })

        if not requests:
            continue

        responses = backend.run_interventions(requests)

        ur_rescued = 0
        ur_total = 0
        res_maintained = 0
        res_total = 0

        for resp, rmeta in zip(responses, request_meta):
            correct = resp.intervened_output.strip() == rmeta["answer"]
            if rmeta["outcome"] == "unresolved":
                ur_total += 1
                if correct:
                    ur_rescued += 1
            else:
                res_total += 1
                if correct:
                    res_maintained += 1

        ur_rate = ur_rescued / ur_total if ur_total else None
        res_rate = res_maintained / res_total if res_total else None

        round_label = f"round_{round_idx+1}"
        results["rounds"][round_label] = {
            "target_layer": target_layer,
            "inject_from_end": from_end,
            "ur_n": ur_total,
            "ur_rescued": ur_rescued,
            "ur_rescue_rate": ur_rate,
            "res_n": res_total,
            "res_maintained": res_maintained,
            "res_maintain_rate": res_rate,
        }
        logger.info("    round %d (layer %d): UR rescue=%d/%d (%.1f%%), Res=%d/%d (%.1f%%)",
                     round_idx + 1, target_layer,
                     ur_rescued, ur_total, (ur_rate or 0) * 100,
                     res_maintained, res_total, (res_rate or 0) * 100)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="0.5B", choices=["0.5B", "7B"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    model_id, model_dir = MODELS[args.model]
    tasks = args.tasks or (["computing"] if args.smoke else TASKS)
    max_ur = 50 if args.smoke else None
    max_res = 25 if args.smoke else None

    model_path = MODEL_CACHE / model_id
    if not model_path.exists():
        model_path = MODEL_CACHE / model_id.replace("/", "--")
    logger.info("Loading model %s from %s", model_id, model_path)

    from nnsight import LanguageModel
    model = LanguageModel(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    tokenizer = model.tokenizer
    logger.info("Model loaded.")

    all_results = []
    for task in tasks:
        logger.info("=== %s ===", task)
        r = run_reinjection_task(model, tokenizer, task, model_dir, max_ur, max_res)
        all_results.append(r)

    out_dir = BASE / "artifacts" / "reinjection"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else args.model
    out_path = out_dir / f"results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out_path)

    print(f"\n=== Re-injection Summary ({args.model}) ===")
    print(f"{'Task':<18} {'Round':>6} {'Layer':>6} {'UR_rescue':>12} {'Res_maintain':>14}")
    for r in all_results:
        for rnd, data in r["rounds"].items():
            ur_str = f"{data['ur_rescued']}/{data['ur_n']} ({data['ur_rescue_rate']*100:.1f}%)" if data.get('ur_rescue_rate') is not None else "N/A"
            res_str = f"{data['res_maintained']}/{data['res_n']} ({data['res_maintain_rate']*100:.1f}%)" if data.get('res_maintain_rate') is not None else "N/A"
            print(f"  {r['task']:<18} {rnd:>6} L={data['target_layer']:>3} {ur_str:>12} {res_str:>14}")


if __name__ == "__main__":
    main()
