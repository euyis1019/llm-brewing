#!/usr/bin/env python3
"""Re-run CSD for the anchor model to populate layer_non_digit_probs.

Only runs CSD (skips probing). Uses existing caches. Overwrites csd.json.
Does NOT touch diagnostics or probing results.

Usage:
    python scripts/rerun_csd_anchor.py [--tasks computing conditional ...]
    python scripts/rerun_csd_anchor.py  # all 6 tasks
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure brewing is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "brewing_output"
MODEL_CACHE = Path("/path/to/cue/models")
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
SEED = "seed42"
ANCHOR_DIR = "Qwen__Qwen2.5-Coder-7B"

ALL_TASKS = [
    "value_tracking", "computing", "conditional",
    "function_call", "loop", "loop_unrolled",
]


def load_samples(task: str):
    """Load eval samples from brewing_output/datasets/."""
    from brewing.schema import Sample
    samples_path = BASE / "datasets" / "cuebench" / "eval" / task / SEED / "samples.json"
    with open(samples_path) as f:
        data = json.load(f)
    return [Sample(**s) for s in data]


def load_cache(task: str):
    """Load hidden state cache."""
    from brewing.schema import HiddenStateCache
    cache_dir = BASE / "caches" / "cuebench" / "eval" / task / SEED / ANCHOR_DIR
    meta_path = cache_dir / "meta.json"
    hs_path = cache_dir / "hidden_states.npz"

    with open(meta_path) as f:
        meta = json.load(f)

    hs = np.load(hs_path)["hidden_states"]

    return HiddenStateCache(
        model_id=meta["model_id"],
        sample_ids=meta["sample_ids"],
        hidden_states=hs,
        token_position=meta.get("token_position", "last"),
        model_predictions=meta.get("model_predictions", []),
        metadata=meta.get("metadata", {}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    args = parser.parse_args()

    # Load model once
    logger.info("Loading model %s ...", MODEL_ID)
    from nnsight import LanguageModel
    import torch

    model_path = MODEL_CACHE / MODEL_ID.replace("/", "--")
    if not model_path.exists():
        # try without -- separator
        model_path = MODEL_CACHE / MODEL_ID.split("/")[-1]
    if not model_path.exists():
        model_path = MODEL_CACHE / MODEL_ID

    logger.info("Resolved model path: %s", model_path)
    model = LanguageModel(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        dispatch=True,
    )
    logger.info("Model loaded.")

    from brewing.methods.csd import CSD
    from brewing.schema import MethodConfig

    csd = CSD()

    for task in args.tasks:
        logger.info("=== %s ===", task)

        samples = load_samples(task)
        cache = load_cache(task)
        logger.info("  %d samples, cache shape (%d, %d, %d)",
                     len(samples), cache.n_layers, cache.hidden_dim, len(cache.sample_ids))

        # Align samples with cache order
        sample_by_id = {s.id: s for s in samples}
        ordered_samples = [sample_by_id[sid] for sid in cache.sample_ids]

        config = MethodConfig(
            method="csd",
            benchmark="CUE-Bench",
            config={
                "eval_dataset_id": f"cuebench/eval/{task}/{SEED}",
            },
        )

        result = csd.run(
            config=config,
            eval_samples=ordered_samples,
            eval_cache=cache,
            resources=None,
            model=model,
        )

        # Verify extras populated
        n_with_extras = sum(1 for sr in result.sample_results if "layer_non_digit_probs" in sr.extras)
        logger.info("  %d/%d samples have layer_non_digit_probs", n_with_extras, len(result.sample_results))

        # Save
        out_path = BASE / "results" / "cuebench" / "eval" / task / SEED / ANCHOR_DIR / "csd.json"
        result.save(out_path)
        logger.info("  Saved to %s", out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
