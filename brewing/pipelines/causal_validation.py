"""CausalValidationPipeline — runs causal validation experiments.

Loads pre-computed S0/S1/S2/S3 artifacts from disk and runs causal
interventions (e.g., activation patching at FJC) using an online model.

Unlike other pipelines, this one NEVER rebuilds datasets or caches.
All inputs must already exist on disk from prior S0-S3 runs.
"""

from __future__ import annotations

import logging
from typing import Any

from brewing.schema import (
    CausalValidationResult,
    DiagnosticResult,
    MethodResult,
)
from brewing.resources import ResourceKey
from .base import PipelineBase

logger = logging.getLogger(__name__)


class CausalValidationPipeline(PipelineBase):
    """Pipeline for causal validation experiments.

    All inputs are loaded strictly from disk — no fallback generation.

    Requires (all must exist):
      - eval dataset (S0)
      - eval cache (S1)
      - linear_probing result (S2)
      - csd result (S2)
      - diagnostics result (S3)
      - online model for interventions

    Flow:
      1. Load eval dataset from disk (fail if missing)
      2. Load eval cache from disk (fail if missing)
      3. Load probing, CSD, diagnostics from disk (fail if missing)
      4. Build intervention backend
      5. Select experiment(s) from config
      6. Run validator(s), passing per-experiment config
      7. Save CausalValidationResult
    """

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        causal_config = self.config.causal_validation
        experiments = causal_config.get("experiments", ["activation_patching_fjc"])

        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[causal_validation] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                subset_result = self._run_subset(
                    subset_name, experiments, causal_config, model, tokenizer,
                )
                results_summary["subsets"][subset_name] = subset_result
            except Exception as e:
                logger.error(
                    "Failed on subset '%s': %s", subset_name, e, exc_info=True,
                )
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary

    def _run_subset(
        self,
        subset_name: str,
        experiments: list[str],
        causal_config: dict,
        model: Any,
        tokenizer: Any,
    ) -> dict[str, Any]:
        eval_key = self.make_key(subset_name, "eval")

        # S0: load eval dataset from disk — NEVER rebuild
        existing = self.resources.resolve_dataset(eval_key)
        if existing is None:
            raise FileNotFoundError(
                f"Eval dataset not found at {self.resources.dataset_dir(eval_key)}. "
                "Causal validation requires a pre-existing eval dataset "
                "from a prior S0 run. It will not regenerate data to avoid "
                "desynchronization with existing cache/results/diagnostics."
            )
        _manifest, samples = existing

        # S1: load eval cache from disk
        cache_key = self.make_key(subset_name, "eval", model_id=self.config.model_id)
        cache = self.resources.resolve_cache(cache_key)
        if cache is None:
            raise FileNotFoundError(
                f"Eval cache not found at {self.resources.cache_path(cache_key)}. "
                "Causal validation requires a pre-built eval cache."
            )

        # S2: load method results from disk
        probe_key = self.make_key(
            subset_name, "eval", model_id=self.config.model_id,
            method="linear_probing",
        )
        probe_result = self.resources.resolve_result(probe_key)
        if probe_result is None:
            raise FileNotFoundError(
                f"Probing result not found at {self.resources.result_path(probe_key)}"
            )

        csd_key = self.make_key(
            subset_name, "eval", model_id=self.config.model_id,
            method="csd",
        )
        csd_result = self.resources.resolve_result(csd_key)
        if csd_result is None:
            raise FileNotFoundError(
                f"CSD result not found at {self.resources.result_path(csd_key)}"
            )

        # S3: load diagnostics from disk
        diag_key = self.make_key(subset_name, "eval", model_id=self.config.model_id)
        diagnostics = self.resources.resolve_diagnostic(diag_key)
        if diagnostics is None:
            raise FileNotFoundError(
                f"Diagnostics not found at {self.resources.diagnostic_path(diag_key)}"
            )

        # Build intervention backend
        backend = self._build_backend(model, tokenizer)

        # Run each experiment, passing per-experiment config
        from brewing.causal import get_validator

        subset_result: dict[str, Any] = {}
        for experiment in experiments:
            logger.info("[causal_validation] Running experiment: %s", experiment)
            validator = get_validator(experiment)

            # Extract per-experiment config section
            experiment_config = causal_config.get(experiment, {})

            result = validator.run(
                samples=samples,
                cache=cache,
                diagnostics=diagnostics,
                backend=backend,
                config=experiment_config,
            )

            # Save result
            self.resources.save_causal_result(eval_key, experiment, result)
            subset_result[experiment] = result.summary

        return subset_result

    def _build_backend(self, model: Any, tokenizer: Any) -> Any:
        """Build the appropriate intervention backend."""
        if model is not None and tokenizer is not None:
            from brewing.causal.backend import NNsightInterventionBackend
            return NNsightInterventionBackend(model, tokenizer)
        else:
            raise RuntimeError(
                "Causal validation requires an online model and tokenizer. "
                "model=%s, tokenizer=%s" % (model, tokenizer)
            )
