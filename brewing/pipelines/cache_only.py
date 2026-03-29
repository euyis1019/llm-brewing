"""CacheOnlyPipeline — S0 -> S1 only.

Resolves/builds datasets and hidden-state caches, then stops.
Useful for pre-computing caches without running any analysis methods.
"""

from __future__ import annotations

import logging
from typing import Any

from brewing.schema import DatasetPurpose
from .base import PipelineBase

logger = logging.getLogger(__name__)


class CacheOnlyPipeline(PipelineBase):
    """S0 -> S1: dataset resolve + cache build, no method execution."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        splits = self.config.splits or ["eval"]

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[cache_only] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                subset_result: dict[str, Any] = {}
                for split in splits:
                    purpose = (
                        DatasetPurpose.TRAIN if split == "train"
                        else DatasetPurpose.EVAL
                    )
                    key = self.make_key(subset_name, split)
                    manifest, samples = self.resolve_dataset(
                        subset_name, key, purpose=purpose,
                    )
                    cache = self.resolve_hidden_cache(
                        key, samples, model, tokenizer
                    )
                    subset_result[split] = {
                        "n_samples": len(samples),
                        "n_layers": cache.n_layers,
                        "hidden_dim": cache.hidden_dim,
                    }
                results_summary["subsets"][subset_name] = subset_result
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary
