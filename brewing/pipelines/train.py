"""TrainPipeline — S0 -> S1 -> probe training -> artifact persistence.

Resolves/builds a *train* dataset and cache, then trains linear probes
and persists the resulting artifact.
"""

from __future__ import annotations

import logging
from typing import Any

from brewing.methods.linear_probing import LinearProbing, DEFAULT_PROBE_PARAMS, DIGIT_CLASSES
from .base import PipelineBase

logger = logging.getLogger(__name__)


class TrainPipeline(PipelineBase):
    """S0 -> S1 -> fit: train probing artifacts."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[train_probing] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                subset_result = self._run_subset(subset_name, model, tokenizer)
                results_summary["subsets"][subset_name] = subset_result
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary

    def _run_subset(
        self,
        subset_name: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # S0: resolve train dataset
        train_key = self.make_key(subset_name, "train")
        from brewing.schema import DatasetPurpose
        _, train_samples = self.resolve_dataset(
            subset_name, train_key, purpose=DatasetPurpose.TRAIN,
        )
        result["n_train_samples"] = len(train_samples)
        logger.info("Train dataset: %d samples", len(train_samples))

        # S1: resolve train hidden cache
        train_cache = self.resolve_hidden_cache(
            train_key, train_samples, model, tokenizer
        )
        result["n_layers"] = train_cache.n_layers
        result["hidden_dim"] = train_cache.hidden_dim

        # Fit: train probes
        prober = LinearProbing()
        lp_config = self.config.method_configs.get("linear_probing", {})
        probe_params = lp_config.get("probe_params", DEFAULT_PROBE_PARAMS)
        answer_space = lp_config.get(
            "answer_space",
            self.benchmark.answer_meta.answer_space
            if hasattr(self.benchmark, "answer_meta")
            else DIGIT_CLASSES,
        )
        overwrite = lp_config.get("overwrite", False)

        artifact_key = self.make_key(
            subset_name, "train", method="linear_probing"
        )

        artifact, _ = prober.train(
            resources=self.resources,
            train_samples=train_samples,
            train_cache=train_cache,
            artifact_key=artifact_key,
            probe_params=probe_params,
            answer_space=answer_space,
            overwrite=overwrite,
        )
        result["artifact_id"] = artifact.artifact_id
        result["fit_status"] = "trained"

        return result
