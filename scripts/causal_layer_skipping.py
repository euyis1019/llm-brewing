#!/usr/bin/env python3
"""Layer Skipping experiment (§3.2, App E.2, A4).

For Overprocessed samples: skip post-FJC layers by injecting the FJC-layer
hidden state into a later layer of the SAME prompt. If the model then
produces the correct answer, overprocessing is confirmed as the failure cause.

Uses the existing InterventionBackend but with source_prompt == target_prompt
(self-intervention, not patchscope to neutral prompt).

Prototype: Qwen2.5-Coder-0.5B on computing, 50 OT + 25 Resolved controls.
Full: anchor model (Qwen2.5-Coder-7B) on all tasks.

Usage:
    # Smoke test: 0.5B, computing, 50 samples
    python scripts/causal_layer_skipping.py --smoke

    # Full: 0.5B, all tasks, all OT+Resolved
    python scripts/causal_layer_skipping.py --model 0.5B

    # Anchor: 7B, all tasks
    python scripts/causal_layer_skipping.py --model 7B
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

# Skip targets: inject FJC hidden state at FJC+k for each k
SKIP_OFFSETS = [2, 4, 6]


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


def run_layer_skipping_task(model, tokenizer, task, model_dir, max_ot=None, max_res=None):
    """Run layer skipping on one task. Returns summary dict."""
    from brewing.causal.backend import NNsightInterventionBackend, InterventionRequest

    diag_data = load_diagnostics(task, model_dir)
    diags = diag_data["sample_diagnostics"]
    hs, meta = load_cache(task, model_dir)
    samples = load_samples(task)
    n_layers = hs.shape[1]

    cache_id_to_idx = {sid: i for i, sid in enumerate(meta["sample_ids"])}

    # Select OT and Resolved samples with valid FJC
    ot_samples = []
    res_samples = []
    for d in diags:
        if d["fjc"] is None:
            continue
        sid = d["sample_id"]
        if sid not in cache_id_to_idx or sid not in samples:
            continue
        entry = {"diag": d, "idx": cache_id_to_idx[sid], "sample": samples[sid]}
        if d["outcome"] == "overprocessed":
            ot_samples.append(entry)
        elif d["outcome"] == "resolved":
            res_samples.append(entry)

    if max_ot is not None:
        ot_samples = ot_samples[:max_ot]
    if max_res is not None:
        res_samples = res_samples[:max_res]

    logger.info("  %s: %d OT, %d Resolved selected", task, len(ot_samples), len(res_samples))

    backend = NNsightInterventionBackend(model, tokenizer)

    results = {"task": task, "n_layers": n_layers, "offsets": {}}

    for offset in SKIP_OFFSETS:
        ot_correct = 0
        ot_total = 0
        res_correct = 0
        res_total = 0

        requests = []
        request_meta = []  # track (outcome, answer) per request

        for entry in ot_samples + res_samples:
            d = entry["diag"]
            fjc = d["fjc"]
            target_layer = fjc + offset
            if target_layer >= n_layers:
                continue

            sample = entry["sample"]
            fjc_hidden = hs[entry["idx"], fjc]  # hidden state AT FJC

            # Self-intervention: inject FJC hidden state at target_layer of the SAME prompt
            requests.append(InterventionRequest(
                sample_id=sample.id,
                source_prompt=sample.prompt,
                target_prompt=sample.prompt,  # same prompt!
                source_hidden=fjc_hidden,
                target_layer=target_layer,
                target_position=-1,
                answer_space=None,  # global argmax
            ))
            request_meta.append({
                "outcome": d["outcome"],
                "answer": sample.answer,
                "fjc": fjc,
            })

        if not requests:
            results["offsets"][offset] = {"ot_rescue": None, "res_maintain": None}
            continue

        responses = backend.run_interventions(requests)

        for resp, rmeta in zip(responses, request_meta):
            correct = resp.intervened_output.strip() == rmeta["answer"]
            if rmeta["outcome"] == "overprocessed":
                ot_total += 1
                if correct:
                    ot_correct += 1
            else:
                res_total += 1
                if correct:
                    res_correct += 1

        ot_rate = ot_correct / ot_total if ot_total else None
        res_rate = res_correct / res_total if res_total else None

        results["offsets"][offset] = {
            "ot_n": ot_total,
            "ot_rescued": ot_correct,
            "ot_rescue_rate": ot_rate,
            "res_n": res_total,
            "res_maintained": res_correct,
            "res_maintain_rate": res_rate,
        }
        logger.info("    offset=+%d: OT rescue=%d/%d (%.1f%%), Res maintain=%d/%d (%.1f%%)",
                     offset, ot_correct, ot_total, (ot_rate or 0) * 100,
                     res_correct, res_total, (res_rate or 0) * 100)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="0.5B", choices=["0.5B", "7B"])
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 task, 50 OT + 25 Res")
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    model_id, model_dir = MODELS[args.model]
    tasks = args.tasks or (["computing"] if args.smoke else TASKS)
    max_ot = 50 if args.smoke else None
    max_res = 25 if args.smoke else None

    # Load model
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
        r = run_layer_skipping_task(model, tokenizer, task, model_dir, max_ot, max_res)
        all_results.append(r)

    # Save
    out_dir = BASE / "artifacts" / "layer_skipping"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else args.model
    out_path = out_dir / f"results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", out_path)

    # Summary table
    print(f"\n=== Layer Skipping Summary ({args.model}) ===")
    print(f"{'Task':<18} {'Offset':>7} {'OT_rescue':>12} {'Res_maintain':>14}")
    for r in all_results:
        for offset, data in r["offsets"].items():
            ot_str = f"{data['ot_rescued']}/{data['ot_n']} ({data['ot_rescue_rate']*100:.1f}%)" if data.get('ot_rescue_rate') is not None else "N/A"
            res_str = f"{data['res_maintained']}/{data['res_n']} ({data['res_maintain_rate']*100:.1f}%)" if data.get('res_maintain_rate') is not None else "N/A"
            print(f"  {r['task']:<18} +{offset:<6} {ot_str:>12} {res_str:>14}")


if __name__ == "__main__":
    main()
