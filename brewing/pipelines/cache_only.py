"""CacheOnlyPipeline — S0 -> S1 only.

Resolves/builds datasets and hidden-state caches, then stops.
Useful for pre-computing caches without running any analysis methods.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import PipelineBase

logger = logging.getLogger(__name__)


class CacheOnlyPipeline(PipelineBase):
    """S0 -> S1: dataset resolve + cache build, no method execution."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[cache_only] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                eval_key = self.make_key(subset_name, "eval")
                manifest, eval_samples = self.resolve_dataset(
                    subset_name, eval_key
                )
                eval_cache = self.resolve_hidden_cache(
                    eval_key, eval_samples, model, tokenizer
                )
                results_summary["subsets"][subset_name] = {
                    "n_eval_samples": len(eval_samples),
                    "n_layers": eval_cache.n_layers,
                    "hidden_dim": eval_cache.hidden_dim,
                }
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary
