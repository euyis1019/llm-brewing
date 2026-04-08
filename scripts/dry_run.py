"""Dry-run script: extract and persist hidden-state caches (S0 + S1 only).

Loads a model, iterates over all 6 subsets x {eval, train}, and saves
hidden states + model predictions to disk. No analysis methods are run.

Usage:
    python scripts/dry_run.py \
        --model-path /path/to/cue/models/Qwen/Qwen2.5-Coder-7B \
        --model-id Qwen/Qwen2.5-Coder-7B \
        --gpu 0
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure brewing package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brewing.benchmarks.cue_bench.builder import load_generated_dataset
from brewing.cache_builder import build_hidden_cache
from brewing.resources import ResourceKey, ResourceManager
from brewing.schema import DatasetManifest, DatasetPurpose, Sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dry_run")

DATA_ROOT = Path(__file__).resolve().parent.parent / "brewing" / "benchmarks" / "cue_bench" / "data"
SUBSETS = ["value_tracking", "computing", "conditional", "function_call", "loop", "loop_unrolled"]
SEED = 42
BENCHMARK = "cuebench"


def load_model(model_path: str, gpu: int):
    """Load model and tokenizer onto specified GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu}"
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s onto %s", model_path, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def run_dry(
    model,
    tokenizer,
    model_id: str,
    output_root: str,
    batch_size: int,
    device: str,
):
    resources = ResourceManager(output_root)
    total_t0 = time.time()

    for split in ["eval", "train"]:
        data_dir = DATA_ROOT / split
        for subset in SUBSETS:
            key = ResourceKey(
                benchmark=BENCHMARK,
                split=split,
                task=subset,
                seed=SEED,
                model_id=model_id,
            )

            # Check if cache already exists
            existing = resources.resolve_cache(key)
            if existing is not None:
                logger.info("SKIP  %s / %s (cache exists)", split, subset)
                continue

            logger.info("START %s / %s", split, subset)
            t0 = time.time()

            # Load samples
            samples = load_generated_dataset(data_dir, task_name=subset)
            logger.info("  samples: %d", len(samples))

            # Save dataset if not exists
            if resources.resolve_dataset(key) is None:
                purpose = DatasetPurpose.EVAL if split == "eval" else DatasetPurpose.TRAIN
                manifest = DatasetManifest(
                    dataset_id=key.dataset_id,
                    purpose=purpose,
                    benchmark="CUE-Bench",
                    subset=subset,
                    sample_ids=[s.id for s in samples],
                    generation_config={"seed": SEED},
                    seed=SEED,
                )
                resources.save_dataset(key, manifest, samples)

            # Build and save cache
            cache = build_hidden_cache(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                model_id=model_id,
                batch_size=batch_size,
                device=device,
            )
            resources.save_cache(key, cache)

            # Log model accuracy as a bonus
            correct = sum(
                1 for pred, s in zip(cache.model_predictions, samples)
                if pred == s.answer
            )
            acc = correct / len(samples) * 100
            elapsed = time.time() - t0
            logger.info(
                "  DONE %s / %s -- acc=%.1f%% (%d/%d), shape=%s, time=%.1fs",
                split, subset, acc, correct, len(samples),
                cache.hidden_states.shape, elapsed,
            )

            # Free cache memory
            del cache
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_t0
    logger.info("ALL DONE for %s in %.1fs", model_id, total_elapsed)


def main():
    parser = argparse.ArgumentParser(description="Dry-run: cache hidden states")
    parser.add_argument("--model-path", required=True, help="Local path to model")
    parser.add_argument("--model-id", required=True, help="Canonical model ID (e.g. Qwen/Qwen2.5-Coder-7B)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--output-root", default="brewing_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward pass")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model, tokenizer = load_model(args.model_path, args.gpu)
    run_dry(model, tokenizer, args.model_id, args.output_root, args.batch_size, device)


if __name__ == "__main__":
    main()
