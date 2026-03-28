"""Orchestrator — thin entry point that delegates to pipeline factory.

External API is unchanged: Orchestrator(config).run(model, tokenizer)
Internal logic is dispatched to the appropriate pipeline based on
config.mode via create_pipeline().

Also exposes helper methods (_make_key, _train_key_for_subset) for
backward compatibility with existing tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schema import RunConfig
from .registry import get_benchmark
from .resources import ResourceKey, ResourceManager
from .pipelines import create_pipeline

logger = logging.getLogger(__name__)

__all__ = ["Orchestrator", "RunConfig"]


class Orchestrator:
    """Main pipeline orchestrator."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.resources = ResourceManager(config.output_root)
        self.benchmark = get_benchmark(config.benchmark)
        self.subsets = config.subsets or self.benchmark.subset_names

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        """Execute the pipeline for the configured mode.

        Args:
            model: Pre-loaded model (required for cache_only, train_probing,
                   and eval with CSD). Not needed for diagnostics.
            tokenizer: Pre-loaded tokenizer.

        Returns:
            Summary dict with paths to all outputs.
        """
        pipeline = create_pipeline(self.config, self.resources, self.benchmark)
        results_summary = pipeline.run(model=model, tokenizer=tokenizer)

        # Save summary
        summary_path = Path(self.config.output_root) / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        return results_summary

    def _make_key(self, task: str, split: str, **overrides: Any) -> ResourceKey:
        """Build a ResourceKey from config + task/split."""
        return ResourceKey(
            benchmark=self.config.benchmark_path_safe,
            split=split,
            task=task,
            seed=self.config.seed,
            model_id=overrides.get("model_id", self.config.model_id),
            method=overrides.get("method"),
        )

    def _train_key_for_subset(self, subset_name: str) -> ResourceKey:
        """Build a train ResourceKey for a subset."""
        return self._make_key(subset_name, "train")
