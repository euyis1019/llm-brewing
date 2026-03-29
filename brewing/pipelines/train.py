"""TrainPipeline — S0 -> S1 -> probe training -> artifact persistence.

Resolves/builds a *train* dataset and cache, then trains linear probes
and persists the resulting artifact.  Optionally validates on the eval
split and reports per-layer accuracy so you can sanity-check probe
quality before committing to a full eval run.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from brewing.methods.linear_probing import (
    LinearProbing, DEFAULT_PROBE_PARAMS, DIGIT_CLASSES, _encode_labels,
)
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

        artifact, probes = prober.train(
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

        # ── Optional: validate on eval split ──
        validate = lp_config.get("validate_on_eval", False)
        if validate:
            eval_metrics = self._validate_on_eval(
                subset_name, probes, answer_space, model, tokenizer,
            )
            result["eval_validation"] = eval_metrics

        return result

    # -----------------------------------------------------------------
    # Eval-split validation
    # -----------------------------------------------------------------

    def _validate_on_eval(
        self,
        subset_name: str,
        probes: list,
        answer_space: list[str],
        model: Any,
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Run trained probes on the eval split and report per-layer accuracy."""
        from brewing.schema import DatasetPurpose

        logger.info("[validate_on_eval] Resolving eval dataset for '%s'", subset_name)

        eval_key = self.make_key(subset_name, "eval")
        _, eval_samples = self.resolve_dataset(
            subset_name, eval_key, purpose=DatasetPurpose.EVAL,
        )
        eval_cache = self.resolve_hidden_cache(
            eval_key, eval_samples, model, tokenizer,
        )

        labels = _encode_labels(
            [s.answer for s in eval_samples], answer_space,
        )
        n_layers = eval_cache.n_layers
        n_eval = len(eval_samples)

        from tqdm import tqdm

        per_layer_acc: dict[str, float] = {}
        pbar = tqdm(range(n_layers), desc="Validating on eval", unit="layer")
        for layer_idx in pbar:
            X = eval_cache.hidden_states[:, layer_idx, :]
            preds = probes[layer_idx].predict(X)
            acc = float(np.mean(preds == labels))
            per_layer_acc[str(layer_idx)] = round(acc, 4)
            pbar.set_postfix(acc=f"{acc:.2%}")

        # Summary stats
        accs = list(per_layer_acc.values())
        best_layer = max(per_layer_acc, key=per_layer_acc.get)  # type: ignore[arg-type]
        summary = {
            "n_eval_samples": n_eval,
            "per_layer_accuracy": per_layer_acc,
            "best_layer": int(best_layer),
            "best_accuracy": per_layer_acc[best_layer],
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "last_layer_accuracy": accs[-1],
        }

        logger.info(
            "[validate_on_eval] %s — best: layer %s (%.2f%%), last layer: %.2f%%",
            subset_name, best_layer, summary["best_accuracy"] * 100,
            summary["last_layer_accuracy"] * 100,
        )

        return summary
