"""Rebuild computing dataset + caches after sample ID fix.

Runs one model on a specified CUDA device.  Launch 4 processes in parallel:

  CUDA_VISIBLE_DEVICES=0 python -m scripts.rebuild_computing Qwen/Qwen2.5-Coder-1.5B &
  CUDA_VISIBLE_DEVICES=0 python -m scripts.rebuild_computing Qwen/Qwen2.5-Coder-7B &
  CUDA_VISIBLE_DEVICES=1 python -m scripts.rebuild_computing Qwen/Qwen2.5-Coder-1.5B-Instruct &
  CUDA_VISIBLE_DEVICES=1 python -m scripts.rebuild_computing Qwen/Qwen2.5-Coder-3B &
  wait
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rebuild_computing")

import brewing.benchmarks  # noqa: F401
import brewing.methods.linear_probing  # noqa: F401
import brewing.methods.csd  # noqa: F401

from brewing.schema import RunConfig, DatasetPurpose
from brewing.registry import get_benchmark
from brewing.resources import ResourceManager
from brewing.pipelines.base import PipelineBase

OUTPUT_ROOT = "brewing_output"


class RebuildPipeline(PipelineBase):
    """Minimal pipeline: S0 (dataset) + S1 (cache) for both splits."""

    def run(self, model=None, tokenizer=None):
        for split, purpose in [("train", DatasetPurpose.TRAIN), ("eval", DatasetPurpose.EVAL)]:
            key = self.make_key("computing", split)
            logger.info("[%s] Resolving dataset: %s", self.config.model_id, split)
            manifest, samples = self.resolve_dataset("computing", key, purpose=purpose)
            logger.info("  -> %d samples, %d unique IDs", len(samples), len(set(s.id for s in samples)))

            logger.info("[%s] Building cache: %s", self.config.model_id, split)
            cache = self.resolve_hidden_cache(key, samples, model, tokenizer)
            logger.info("  -> shape (%d, %d, %d)", cache.n_samples, cache.n_layers, cache.hidden_dim)

        return {"status": "ok", "model": self.config.model_id}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m scripts.rebuild_computing <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    logger.info("Rebuilding computing caches for: %s", model_id)

    config = RunConfig(
        benchmark="CUE-Bench",
        model_id=model_id,
        subsets=["computing"],
        methods=[],
        output_root=OUTPUT_ROOT,
        seed=42,
        mode="cache_only",
        batch_size=16,
    )

    resources = ResourceManager(OUTPUT_ROOT)
    benchmark = get_benchmark(config.benchmark)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_BASE = "/path/to/cue/models"
    # Map HF model_id to local path (Qwen/Qwen2.5-Coder-7B -> /path/to/cue/models/Qwen/Qwen2.5-Coder-7B)
    local_path = f"{MODEL_BASE}/{model_id}"

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    logger.info("Loading model (HF backend, local): %s", local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    pipeline = RebuildPipeline(config, resources, benchmark)
    result = pipeline.run(model=model, tokenizer=tokenizer)
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
