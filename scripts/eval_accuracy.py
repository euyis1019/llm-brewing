"""
eval_accuracy.py — CUE-Bench accuracy evaluation script
========================================================
Greedy decoding only; does not go through the Brewing cache/probe pipeline.
The model predicts the next token for each item and is scored against the
ground-truth single-character answer (0-9).

Usage:
    python scripts/eval_accuracy.py --model /path/to/model [options]

Options:
    --model         Model path (required)
    --data_dir      Eval data directory (default: brewing/benchmarks/cue_bench/data/eval)
    --tasks         Comma-separated task names to run (default: all 6)
    --batch_size    Inference batch size (default: 16)
    --max_samples   Max samples per task, for debugging (default: all)
    --output        Result save path (default: none, print only)
    --dtype         Model precision bfloat16 / float16 / float32 (default: bfloat16)

Examples:
    # Evaluate Qwen3.5-35B-A3B-Base
    python scripts/eval_accuracy.py \
        --model /path/to/model/Qwen/Qwen3.5-35B-A3B-Base \
        --batch_size 8

    # Run only two tasks, quick check
    python scripts/eval_accuracy.py \
        --model /path/to/model \
        --tasks value_tracking,computing \
        --max_samples 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "value_tracking",
    "computing",
    "conditional",
    "function_call",
    "loop",
    "loop_unrolled",
]

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "brewing" / "benchmarks" / "cue_bench" / "data" / "eval"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_data(data_dir: Path, task: str, max_samples: int | None) -> list[dict]:
    path = data_dir / f"{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path) as f:
        samples = json.load(f)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def predict_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: str,
) -> list[str]:
    """
    Run single-step greedy decoding on a batch of prompts and return the
    predicted next-token strings.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        # Only need 1 new token
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # output_ids shape: (batch, prompt_len + 1)
    # Take the last column, i.e. the newly generated token
    new_token_ids = output_ids[:, inputs["input_ids"].shape[1]]
    predictions = [tokenizer.decode([tid]).strip() for tid in new_token_ids.tolist()]
    return predictions


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def eval_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict],
    task: str,
    batch_size: int,
    device: str,
) -> dict:
    """Return a detailed result dict for the task."""
    prompts = [s["prompt"] for s in samples]
    answers = [s["answer"] for s in samples]
    ids = [s.get("id", str(i)) for i, s in enumerate(samples)]

    all_preds = []
    n = len(prompts)

    print(f"  [{task}] {n} samples, batch_size={batch_size}", flush=True)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        preds = predict_batch(model, tokenizer, batch_prompts, device)
        all_preds.extend(preds)

        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        speed = done / elapsed
        print(f"    {done}/{n}  ({speed:.1f} samples/s)", end="\r", flush=True)

    print()  # newline

    # Compute accuracy
    correct = sum(p == a for p, a in zip(all_preds, answers))
    accuracy = correct / n

    # Group statistics by difficulty
    breakdown = {}
    for sample, pred, ans in zip(samples, all_preds, answers):
        meta = sample.get("metadata", {})
        # Use the first difficulty dimension as the grouping key (differs per task)
        dim_key, dim_val = _primary_dimension(task, meta)
        key = f"{dim_key}={dim_val}"
        if key not in breakdown:
            breakdown[key] = {"correct": 0, "total": 0}
        breakdown[key]["total"] += 1
        if pred == ans:
            breakdown[key]["correct"] += 1

    for k, v in breakdown.items():
        v["accuracy"] = v["correct"] / v["total"]

    return {
        "task": task,
        "n_samples": n,
        "n_correct": correct,
        "accuracy": accuracy,
        "breakdown": breakdown,
        "per_sample": [
            {"id": sid, "pred": p, "answer": a, "correct": p == a}
            for sid, p, a in zip(ids, all_preds, answers)
        ],
    }


def _primary_dimension(task: str, meta: dict) -> tuple[str, str]:
    """Pick the most representative complexity dimension per task for breakdown."""
    dim_map = {
        "value_tracking": "depth",
        "computing": "steps",
        "conditional": "depth",
        "function_call": "depth",
        "loop": "iterations",
        "loop_unrolled": "iterations",
    }
    key = dim_map.get(task, "depth")
    return key, str(meta.get(key, "?"))


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_results(results: list[dict], model_path: str) -> None:
    print()
    print("=" * 60)
    print(f"  Model: {model_path}")
    print("=" * 60)
    print(f"  {'Task':<20} {'Accuracy':>8}  {'Correct/Total':>12}")
    print("  " + "-" * 44)

    total_correct = 0
    total_n = 0
    for r in results:
        acc_str = f"{r['accuracy']:.1%}"
        cnt_str = f"{r['n_correct']}/{r['n_samples']}"
        print(f"  {r['task']:<20} {acc_str:>8}  {cnt_str:>12}")
        total_correct += r["n_correct"]
        total_n += r["n_samples"]

    print("  " + "-" * 44)
    overall = total_correct / total_n
    print(f"  {'Overall':<20} {overall:.1%}  {total_correct}/{total_n}")
    print("=" * 60)

    # Per-task difficulty breakdown
    print()
    for r in results:
        print(f"  [{r['task']}] breakdown by difficulty dimension:")
        for k, v in sorted(r["breakdown"].items()):
            print(f"    {k:<25} {v['accuracy']:.1%}  ({v['correct']}/{v['total']})")
        print()


# ---------------------------------------------------------------------------
# Model loading (compatible with text-only and VL checkpoints)
# ---------------------------------------------------------------------------

def _load_model(model_path: str, torch_dtype) -> AutoModelForCausalLM:
    """
    Load the model with checkpoint-type awareness:
    - Text-only checkpoint (e.g. Qwen2.5-7B): use AutoModelForCausalLM directly.
    - VL checkpoint (e.g. Qwen3.5-9B-Base, weight keys contain
      model.language_model.*): load via the architecture class for text-only
      inference to avoid key-prefix mismatch.
    - DeepSeek MoE: same standard AutoModelForCausalLM path, trust_remote_code=True.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_vl = hasattr(cfg, "text_config") and cfg.text_config is not None

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if is_vl:
        # VL checkpoint: load via architectures[0] to avoid key-prefix mismatch
        arch = (cfg.architectures or [""])[0]
        import transformers as _tf
        model_cls = getattr(_tf, arch, None)
        if model_cls is None:
            print(f"[WARN] Architecture class {arch} not found, falling back to AutoModelForCausalLM", file=sys.stderr)
            model_cls = AutoModelForCausalLM
        print(f"[Model] VL checkpoint detected, loading with {model_cls.__name__} (text inference mode)")
        return model_cls.from_pretrained(model_path, **load_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CUE-Bench accuracy evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR, help="Eval data directory")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names, default all")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per task (for debugging)")
    parser.add_argument("--output", type=Path, default=None, help="Result JSON save path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    # Parse task list
    tasks = ALL_TASKS
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        unknown = set(tasks) - set(ALL_TASKS)
        if unknown:
            print(f"[ERROR] Unknown tasks: {unknown}", file=sys.stderr)
            sys.exit(1)

    # Check data directory
    if not args.data_dir.exists():
        print(f"[ERROR] Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    if device == "cuda":
        n_gpu = torch.cuda.device_count()
        print(f"[GPU]  {n_gpu} device(s): " + ", ".join(
            f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)"
            for i in range(n_gpu)
        ))

    # Load tokenizer
    print(f"[Model] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding for generation tasks

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[Model] Loading weights (dtype={args.dtype}, device_map=auto)...")
    t_load = time.time()
    model = _load_model(args.model, torch_dtype)
    model.eval()
    print(f"[Model] Loaded in {time.time() - t_load:.1f}s")

    # Evaluate task by task
    all_results = []
    for task in tasks:
        print(f"\n>> Task: {task}")
        samples = load_task_data(args.data_dir, task, args.max_samples)
        result = eval_task(model, tokenizer, samples, task, args.batch_size, device)
        all_results.append(result)
        print(f"   Accuracy: {result['accuracy']:.1%}  ({result['n_correct']}/{result['n_samples']})")

    # Print summary
    print_results(all_results, args.model)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "model": args.model,
            "dtype": args.dtype,
            "tasks": all_results,
            "overall_accuracy": sum(r["n_correct"] for r in all_results)
                               / sum(r["n_samples"] for r in all_results),
        }
        # per_sample can be large; controlled separately whether to write
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Result] Saved to {args.output}")


if __name__ == "__main__":
    main()
