#!/usr/bin/env python3
"""Full eval runner: Probing eval + CSD + Diagnostics for all models x tasks.

Runs S2 (probing eval + CSD) and S3 (diagnostics) on pre-computed caches.
Supports dual-GPU parallel execution via multiprocessing.

Usage:
    # Single GPU
    python scripts/run_full_eval.py --gpu 0 --models all

    # Dual GPU (default: splits models across GPU 0 and GPU 1)
    python scripts/run_full_eval.py --models all --dual-gpu

    # In tmux
    tmux new -s eval
    /path/to/conda/envs/cue/bin/python scripts/run_full_eval.py --models all --dual-gpu
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
import logging
import os
import time
import traceback
from multiprocessing import Process, Queue

import numpy as np
import torch

from brewing.schema import (
    HiddenStateCache, MethodConfig, MethodResult, Sample,
    SampleMethodResult, Granularity,
)
from brewing.methods.linear_probing import LinearProbing
from brewing.diagnostics.outcome import run_diagnostics
from brewing.resources import ResourceKey, ResourceManager

# ── Constants ──

MODEL_DIR = Path("/path/to/cue/models")
OUTPUT_ROOT = Path("brewing_output")
DATA_DIR = Path("brewing/benchmarks/cue_bench/data/eval")

ALL_TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]

# Ordered by priority: Qwen2.5-Coder first (small→large), then others
MODEL_GROUPS = {
    "qwen25coder": [
        "Qwen/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-14B",
    ],
    "qwen25": [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
    ],
    "qwen3": [
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
    ],
    "extra": [
        "deepseek-ai/deepseek-coder-6.7b-base",
        "meta-llama/Llama-2-7b-hf",
        "codellama/CodeLlama-7b-hf",
    ],
}

# "all" = all groups in priority order
ALL_MODELS = (
    MODEL_GROUPS["qwen25coder"]
    + MODEL_GROUPS["qwen25"]
    + MODEL_GROUPS["qwen3"]
    + MODEL_GROUPS["extra"]
)

TARGET_PROMPT = '# The value of x is "'
ANSWER_SPACE = [str(d) for d in range(10)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_eval")


# ── Data loading ──

def load_eval_data(model_id: str, task: str, max_samples: int | None = None):
    """Load eval samples + hidden state cache from disk."""
    model_safe = model_id.replace("/", "__")
    cache_dir = OUTPUT_ROOT / "caches" / "cuebench" / "eval" / task / "seed42" / model_safe

    hs_path = cache_dir / "hidden_states.npz"
    meta_path = cache_dir / "meta.json"

    hs = np.load(hs_path)
    hidden_states = hs[list(hs.keys())[0]]  # (N, L, D)

    with open(meta_path) as f:
        meta = json.load(f)

    with open(DATA_DIR / f"{task}.json") as f:
        raw = json.load(f)
    all_samples = {s["id"]: s for s in raw}

    sample_ids = meta["sample_ids"]
    model_preds = meta.get("model_predictions", [""] * len(sample_ids))

    if max_samples is not None and max_samples < len(sample_ids):
        sample_ids = sample_ids[:max_samples]
        model_preds = model_preds[:max_samples]
        hidden_states = hidden_states[:max_samples]

    samples = []
    for sid in sample_ids:
        s = all_samples[sid]
        samples.append(Sample(
            id=s["id"], benchmark="CUE-Bench", subset=task,
            prompt=s["prompt"], answer=s["answer"],
            metadata=s.get("metadata", {}),
        ))

    cache = HiddenStateCache(
        model_id=model_id,
        hidden_states=hidden_states,
        sample_ids=sample_ids,
        model_predictions=model_preds,
    )
    return samples, cache


# ── CSD via HF hooks (from csd.py _run_per_sample) ──

def get_answer_token_ids(tokenizer, answer_space):
    ids = []
    for ans in answer_space:
        tids = tokenizer.encode(ans, add_special_tokens=False)
        ids.append(tids[0] if tids else 0)
    return ids


def get_baseline_logits(model, tokenizer, target_prompt, answer_token_ids):
    inputs = tokenizer(target_prompt, return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1].cpu().float().numpy()
    return np.array([logits[tid] for tid in answer_token_ids])


def run_csd_hf(model, tokenizer, eval_samples, eval_cache, n_layers):
    """Run CSD using HF hook injection (no nnsight)."""
    answer_token_ids = get_answer_token_ids(tokenizer, ANSWER_SPACE)
    baseline_logits = get_baseline_logits(model, tokenizer, TARGET_PROMPT, answer_token_ids)

    device = next(model.parameters()).device
    target_inputs = tokenizer(TARGET_PROMPT, return_tensors="pt").to(device)
    patch_pos = target_inputs.input_ids.shape[1] - 1

    sample_results = []

    for i, sample in enumerate(eval_samples):
        h_all = eval_cache.hidden_states[i]  # (L, D)
        layer_vals = np.zeros(n_layers)
        layer_preds = []
        layer_confs = np.zeros((n_layers, len(ANSWER_SPACE)))

        for layer_idx in range(n_layers):
            h = torch.tensor(
                h_all[layer_idx], dtype=torch.float16
            ).to(device)

            def make_hook(h_src, pos):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if hidden.shape[1] > pos:
                        hidden[0, pos, :] = h_src
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return hook_fn

            hook = model.model.layers[layer_idx].register_forward_hook(
                make_hook(h, patch_pos)
            )
            try:
                with torch.no_grad():
                    outputs = model(**target_inputs)
                logits = outputs.logits[0, -1].cpu().float().numpy()
            finally:
                hook.remove()

            answer_logits = np.array([logits[tid] for tid in answer_token_ids])
            adjusted = answer_logits - baseline_logits

            pred_idx = int(np.argmax(adjusted))
            pred_label = ANSWER_SPACE[pred_idx]
            correct = float(pred_label == sample.answer)

            exp_adj = np.exp(adjusted - np.max(adjusted))
            norm_probs = exp_adj / exp_adj.sum()

            layer_vals[layer_idx] = correct
            layer_preds.append(pred_label)
            layer_confs[layer_idx] = norm_probs

        sample_results.append(SampleMethodResult(
            sample_id=sample.id,
            layer_values=layer_vals,
            layer_predictions=layer_preds,
            layer_confidences=layer_confs,
        ))

        if (i + 1) % 100 == 0:
            logger.info("  CSD progress: %d/%d samples", i + 1, len(eval_samples))

    return sample_results


# ── Per-model runner ──

def result_exists(rm, model_id, task):
    """Check if all three result files exist."""
    probe_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id, method="linear_probing",
    )
    csd_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id, method="csd",
    )
    diag_key = ResourceKey(
        benchmark="cuebench", split="eval", task=task,
        seed=42, model_id=model_id,
    )
    return (
        rm.result_path(probe_key).exists()
        and rm.result_path(csd_key).exists()
        and rm.diagnostic_path(diag_key).exists()
    )


def run_model(model_id, tasks, gpu_id, skip_existing=True, max_samples=None):
    """Run all tasks for a single model on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    rm = ResourceManager(str(OUTPUT_ROOT))
    model_safe = model_id.replace("/", "__")

    # Check which tasks need work
    if skip_existing:
        pending = [t for t in tasks if not result_exists(rm, model_id, t)]
        if not pending:
            logger.info("[GPU %d] %s: all tasks done, skipping", gpu_id, model_id)
            return
        skipped = len(tasks) - len(pending)
        if skipped:
            logger.info("[GPU %d] %s: skipping %d done tasks", gpu_id, model_id, skipped)
        tasks = pending

    # Resolve model path
    parts = model_id.split("/")
    if len(parts) == 2:
        model_path = MODEL_DIR / parts[0] / parts[1]
    else:
        model_path = MODEL_DIR / model_id
    if not model_path.exists():
        logger.error("[GPU %d] Model not found: %s", gpu_id, model_path)
        return

    # Load model for CSD
    logger.info("[GPU %d] Loading model %s ...", gpu_id, model_id)
    t_load = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    except Exception:
        logger.error("[GPU %d] Failed to load %s:\n%s", gpu_id, model_id, traceback.format_exc())
        return
    t_load = time.time() - t_load
    logger.info("[GPU %d] %s loaded in %.1fs", gpu_id, model_id, t_load)

    completed = 0
    for task in tasks:
        logger.info("[GPU %d] %s / %s — starting", gpu_id, model_id, task)
        t_task = time.time()
        try:
            samples, cache = load_eval_data(model_id, task, max_samples)
            n_layers = cache.n_layers
            n_samples = len(samples)
            logger.info("  Loaded %d samples, %d layers", n_samples, n_layers)

            # ── Probing eval ──
            probe_key = ResourceKey(
                benchmark="cuebench", split="eval", task=task,
                seed=42, model_id=model_id, method="linear_probing",
            )
            probe_path = rm.result_path(probe_key)

            if skip_existing and probe_path.exists():
                logger.info("  Probing: loading existing result")
                probe_result = MethodResult.load(probe_path)
            else:
                t0 = time.time()
                lp = LinearProbing()
                lp_config = MethodConfig(
                    method="linear_probing",
                    benchmark="CUE-Bench",
                    config={
                        "eval_dataset_id": f"cuebench-{task}-eval-seed42",
                        "answer_space": ANSWER_SPACE,
                        "resource_key_benchmark": "cuebench",
                        "resource_key_task": task,
                        "resource_key_seed": 42,
                        "fit_policy": "eval_only",
                    },
                )
                try:
                    probe_result = lp.run(
                        config=lp_config,
                        eval_samples=samples,
                        eval_cache=cache,
                        resources=rm,
                    )
                    rm.save_result(probe_key, probe_result)
                    logger.info("  Probing done in %.2fs", time.time() - t0)
                except FileNotFoundError as e:
                    logger.warning("  Probing skipped (no artifact): %s", e)
                    probe_result = None

            # ── CSD ──
            csd_key = ResourceKey(
                benchmark="cuebench", split="eval", task=task,
                seed=42, model_id=model_id, method="csd",
            )
            csd_path = rm.result_path(csd_key)

            if skip_existing and csd_path.exists():
                logger.info("  CSD: loading existing result")
                csd_result = MethodResult.load(csd_path)
            else:
                t0 = time.time()
                csd_sample_results = run_csd_hf(
                    model, tokenizer, samples, cache, n_layers
                )
                csd_result = MethodResult(
                    method="csd",
                    model_id=model_id,
                    granularity=Granularity.PER_SAMPLE,
                    eval_dataset_id=f"cuebench-{task}-eval-seed42",
                    sample_results=csd_sample_results,
                )
                rm.save_result(csd_key, csd_result)
                t_csd = time.time() - t0
                logger.info(
                    "  CSD done in %.1fs (%.3fs/sample)",
                    t_csd, t_csd / n_samples,
                )

            # ── Diagnostics ──
            diag_key = ResourceKey(
                benchmark="cuebench", split="eval", task=task,
                seed=42, model_id=model_id,
            )
            diag_path = rm.diagnostic_path(diag_key)

            if skip_existing and diag_path.exists():
                logger.info("  Diagnostics: already exists, skipping")
            elif probe_result is None:
                logger.warning("  Diagnostics skipped: no probe result")
            else:
                model_predictions = {
                    sid: pred
                    for sid, pred in zip(cache.sample_ids, cache.model_predictions)
                }
                diag = run_diagnostics(
                    samples=samples,
                    probe_result=probe_result,
                    csd_result=csd_result,
                    model_predictions=model_predictions,
                    n_layers=n_layers,
                )
                rm.save_diagnostic(diag_key, diag)

                # Print outcome summary
                dist = diag.outcome_distribution
                dist_str = " | ".join(f"{k}: {v:.1%}" for k, v in dist.items())
                logger.info("  Outcomes: %s", dist_str)

            completed += 1
            logger.info(
                "[GPU %d] %s / %s — done (%.1fs)",
                gpu_id, model_id, task, time.time() - t_task,
            )

        except Exception:
            logger.error(
                "[GPU %d] %s / %s — FAILED:\n%s",
                gpu_id, model_id, task, traceback.format_exc(),
            )

    # Free GPU memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("[GPU %d] %s: finished %d/%d tasks", gpu_id, model_id, completed, len(tasks))


# ── GPU worker (for multiprocessing) ──

def gpu_worker(model_list, tasks, gpu_id, skip_existing, result_queue, max_samples=None):
    """Worker process: runs a list of models sequentially on one GPU."""
    t_start = time.time()
    for model_id in model_list:
        try:
            run_model(model_id, tasks, gpu_id, skip_existing, max_samples)
        except Exception:
            logger.error("[GPU %d] %s — unhandled error:\n%s",
                         gpu_id, model_id, traceback.format_exc())
    elapsed = time.time() - t_start
    result_queue.put((gpu_id, len(model_list), elapsed))


def split_models_for_dual_gpu(models):
    """Split models into two roughly balanced lists for dual GPU.

    Strategy: alternate assignment, with larger models (14B) going to
    different GPUs when possible.
    """
    gpu0, gpu1 = [], []
    for i, m in enumerate(models):
        if i % 2 == 0:
            gpu0.append(m)
        else:
            gpu1.append(m)
    return gpu0, gpu1


# ── CLI ──

def resolve_models(group_name):
    if group_name == "all":
        return list(ALL_MODELS)
    parts = group_name.split(",")
    result = []
    for p in parts:
        p = p.strip()
        if p in MODEL_GROUPS:
            result.extend(MODEL_GROUPS[p])
        else:
            # Treat as direct model ID
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full eval runner: Probing + CSD + Diagnostics"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device (single-GPU mode). Ignored if --dual-gpu is set.",
    )
    parser.add_argument(
        "--dual-gpu", action="store_true",
        help="Use both GPU 0 and GPU 1 in parallel.",
    )
    parser.add_argument(
        "--models", default="all",
        help="Model group: qwen25coder, qwen25, qwen3, extra, all, or comma-separated.",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task list. Default: all 6 tasks.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip (model, task) combos with existing results (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Force re-run even if results exist.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per task (default: all). Use to speed up slow models.",
    )
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS
    models = resolve_models(args.models)

    total_combos = len(models) * len(tasks)
    logger.info("=" * 60)
    logger.info("Full Eval Runner")
    logger.info("Models: %d | Tasks: %d | Combos: %d", len(models), len(tasks), total_combos)
    if args.max_samples:
        logger.info("Max samples: %d", args.max_samples)
    logger.info("Skip existing: %s", skip_existing)
    if args.dual_gpu:
        logger.info("Mode: dual-GPU (GPU 0 + GPU 1)")
    else:
        logger.info("Mode: single-GPU (GPU %d)", args.gpu)
    logger.info("=" * 60)

    for i, m in enumerate(models):
        logger.info("  [%2d] %s", i + 1, m)
    logger.info("")

    t_total = time.time()

    if args.dual_gpu:
        gpu0_models, gpu1_models = split_models_for_dual_gpu(models)
        logger.info("GPU 0: %d models | GPU 1: %d models", len(gpu0_models), len(gpu1_models))
        logger.info("GPU 0: %s", [m.split("/")[-1] for m in gpu0_models])
        logger.info("GPU 1: %s", [m.split("/")[-1] for m in gpu1_models])

        q = Queue()
        p0 = Process(target=gpu_worker, args=(gpu0_models, tasks, 0, skip_existing, q, args.max_samples))
        p1 = Process(target=gpu_worker, args=(gpu1_models, tasks, 1, skip_existing, q, args.max_samples))

        p0.start()
        p1.start()
        p0.join()
        p1.join()

        # Collect results
        while not q.empty():
            gpu_id, n_models, elapsed = q.get()
            logger.info("GPU %d finished: %d models in %.1fs", gpu_id, n_models, elapsed)
    else:
        for model_id in models:
            run_model(model_id, tasks, args.gpu, skip_existing, args.max_samples)

    t_total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("All done in %.1fs (%.1f min)", t_total, t_total / 60)
    logger.info("=" * 60)

    # Summary: count result files
    rm = ResourceManager(str(OUTPUT_ROOT))
    done, missing = 0, 0
    for model_id in models:
        for task in tasks:
            diag_key = ResourceKey(
                benchmark="cuebench", split="eval", task=task,
                seed=42, model_id=model_id,
            )
            if rm.diagnostic_path(diag_key).exists():
                done += 1
            else:
                missing += 1
                logger.warning("  MISSING: %s / %s", model_id, task)
    logger.info("Results: %d/%d complete, %d missing", done, done + missing, missing)


if __name__ == "__main__":
    main()
