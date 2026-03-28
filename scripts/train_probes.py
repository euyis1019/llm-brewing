#!/usr/bin/env python
"""Train linear probes for the Brewing pipeline.

This script handles the full training flow that is external to the
eval-only orchestrator pipeline:

  1. Generate training data (via datagen)
  2. Load model & extract hidden states (S1)
  3. Train per-layer logistic regression probes
  4. Persist artifacts via ResourceManager

Usage:
    python scripts/train_probes.py --config scripts/train_config.yaml
    python scripts/train_probes.py --config scripts/train_config.yaml --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Ensure Brewing package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brewing.benchmarks.cue_bench import generate_and_convert
from brewing.cache_builder import build_hidden_cache
from brewing.methods.linear_probing import LinearProbing
from brewing.resources import ResourceKey, ResourceManager
from brewing.schema import DatasetManifest, DatasetPurpose

logger = logging.getLogger(__name__)

BENCHMARK = "cuebench"


def load_train_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train linear probes")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_train_config(args.config)

    model_id: str = cfg["model_id"]
    subsets: list[str] = cfg.get("subsets", [
        "value_tracking", "computing", "conditional",
        "function_call", "loop", "loop_unrolled",
    ])
    output_root: str = cfg.get("output_root", "brewing_output")
    seed: int = cfg.get("seed", 42)
    samples_per_config: int | None = cfg.get("samples_per_config", None)
    batch_size: int = cfg.get("batch_size", 8)
    quantization: str | None = cfg.get("quantization", None)
    overwrite: bool = cfg.get("overwrite", False)
    probe_params: dict = cfg.get("probe_params", {
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 1000,
    })

    rm = ResourceManager(output_root)

    # ---- Load model ----
    logger.info("Loading model: %s", model_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kwargs = {"device_map": "auto", "output_hidden_states": True}
    if quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    logger.info("Model loaded")

    lp = LinearProbing()

    for subset in subsets:
        train_key = ResourceKey(
            benchmark=BENCHMARK,
            split="train",
            task=subset,
            seed=seed,
            model_id=model_id,
            method="linear_probing",
        )
        logger.info("=" * 60)
        logger.info("Subset: %s  (key: %s/%s/seed%d)", subset, BENCHMARK, subset, seed)
        logger.info("=" * 60)

        # ---- S0: Generate training data ----
        ds_key = ResourceKey(
            benchmark=BENCHMARK, split="train", task=subset, seed=seed,
        )
        existing = rm.resolve_dataset(ds_key)
        if existing is not None:
            logger.info("Training dataset already on disk, reusing")
            _, train_samples = existing
        else:
            logger.info("Generating training data ...")
            train_samples = generate_and_convert(
                subset, seed=seed, samples_per_config=samples_per_config,
            )
            manifest = DatasetManifest(
                dataset_id=ds_key.dataset_id,
                purpose=DatasetPurpose.TRAIN,
                benchmark="CUE-Bench",
                subset=subset,
                sample_ids=[s.id for s in train_samples],
                generation_config={
                    "seed": seed,
                    "samples_per_config": samples_per_config,
                },
                seed=seed,
            )
            rm.save_dataset(ds_key, manifest, train_samples)
        logger.info("Training samples: %d", len(train_samples))

        # ---- S1: Build hidden state cache ----
        cache_key = ResourceKey(
            benchmark=BENCHMARK, split="train", task=subset,
            seed=seed, model_id=model_id,
        )
        existing_cache = rm.resolve_cache(cache_key)
        if existing_cache is not None:
            logger.info("Hidden state cache already on disk, reusing")
            train_cache = existing_cache
        else:
            logger.info("Extracting hidden states ...")
            t0 = time.time()
            train_cache = build_hidden_cache(
                model=model,
                tokenizer=tokenizer,
                samples=train_samples,
                model_id=model_id,
                batch_size=batch_size,
            )
            elapsed = time.time() - t0
            logger.info(
                "Cache built: shape=%s, time=%.1fs",
                train_cache.hidden_states.shape, elapsed,
            )
            rm.save_cache(cache_key, train_cache)

        # ---- Train probes ----
        logger.info("Training probes ...")
        t0 = time.time()
        try:
            artifact, probes = lp.train(
                resources=rm,
                train_samples=train_samples,
                train_cache=train_cache,
                artifact_key=train_key,
                probe_params=probe_params,
                overwrite=overwrite,
            )
            elapsed = time.time() - t0
            logger.info(
                "Probes trained: %d layers, time=%.1fs, artifact=%s",
                len(probes), elapsed, artifact.artifact_id,
            )
            # Print per-layer train accuracy summary
            per_layer = artifact.fit_metrics.get("per_layer", {})
            accs = [v["train_accuracy"] for v in per_layer.values()]
            if accs:
                logger.info(
                    "Train accuracy: min=%.3f, max=%.3f, mean=%.3f",
                    min(accs), max(accs), sum(accs) / len(accs),
                )
        except FileExistsError:
            logger.info("Probe artifact already exists, skipping (use overwrite: true to retrain)")

    logger.info("=" * 60)
    logger.info("All done. Output: %s/", output_root)


if __name__ == "__main__":
    main()
