"""Orchestrator — coordinates the S0->S1->S2 Brewing pipeline.

Responsible for: driving the end-to-end pipeline from dataset resolution
through hidden-state caching to method execution and result persistence.

NOT responsible for: diagnostics (S3 is decoupled — see
brewing.diagnostics.outcome.run_diagnostics_from_disk), model loading
(caller provides model/tokenizer), or benchmark-specific data generation
(delegated to benchmark adapters via registry).

Pipeline stages:
  S0:  Resolve/build eval dataset
  S1:  Resolve/build eval hidden-state cache
  S2:  Run analysis methods (probing, CSD, etc.) -> persist MethodResult
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schema import (
    DatasetManifest,
    DatasetPurpose,
    HiddenStateCache,
    MethodConfig,
    MethodResult,
    RunConfig,
    Sample,
)
from .registry import get_benchmark, get_method_class
from .resources import ResourceKey, ResourceManager

logger = logging.getLogger(__name__)

# Re-export RunConfig for backwards compatibility
__all__ = ["Orchestrator", "RunConfig"]


class Orchestrator:
    """Main pipeline orchestrator."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.resources = ResourceManager(config.output_root)
        self.benchmark = get_benchmark(config.benchmark)

        # Determine subsets
        self.subsets = config.subsets or self.benchmark.subset_names

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        """Execute the full pipeline.

        Args:
            model: Pre-loaded model (required for cache building and CSD).
                   If None, only works with pre-existing caches.
            tokenizer: Pre-loaded tokenizer.

        Returns:
            Summary dict with paths to all outputs.
        """
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                subset_result = self._run_subset(
                    subset_name, model=model, tokenizer=tokenizer
                )
                results_summary["subsets"][subset_name] = subset_result
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

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

    def _run_subset(
        self,
        subset_name: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        """Run pipeline for a single subset."""
        result: dict[str, Any] = {}

        # ---- S0: Resolve/build eval dataset ----
        eval_key = self._make_key(subset_name, "eval")

        manifest, eval_samples = self._resolve_eval_dataset(
            subset_name, eval_key
        )
        result["n_eval_samples"] = len(eval_samples)
        logger.info("Eval dataset: %d samples", len(eval_samples))

        # ---- S1: Resolve/build eval hidden cache ----
        eval_cache = self._resolve_hidden_cache(
            eval_key, eval_samples, model, tokenizer
        )
        result["n_layers"] = eval_cache.n_layers
        result["hidden_dim"] = eval_cache.hidden_dim

        # ---- S2: Run methods ----
        method_results: dict[str, MethodResult] = {}

        for method_name in self.config.methods:
            logger.info("Running method: %s", method_name)
            try:
                result_key = self._make_key(subset_name, "eval", method=method_name)
                mr = self._run_method(
                    method_name, subset_name, eval_key,
                    eval_samples, eval_cache, model, tokenizer,
                )
                method_results[method_name] = mr
                self.resources.save_result(result_key, mr)
                result[f"method_{method_name}"] = "ok"
            except Exception as e:
                logger.error("Method '%s' failed: %s", method_name, e, exc_info=True)
                result[f"method_{method_name}"] = f"error: {e}"

        return result

    def _resolve_eval_dataset(
        self,
        subset_name: str,
        key: ResourceKey,
    ) -> tuple[DatasetManifest, list[Sample]]:
        """S0: Resolve or build eval dataset."""
        existing = self.resources.resolve_dataset(key)
        if existing is not None:
            return existing

        # Build dataset
        if self.config.use_fixture:
            from .benchmarks.cue_bench import FIXTURE_SAMPLES
            samples = [s for s in FIXTURE_SAMPLES if s.subset == subset_name]
        elif self.config.data_dir:
            from .benchmarks.cue_bench import load_generated_dataset
            samples = load_generated_dataset(
                Path(self.config.data_dir), subset_name
            )
        else:
            # Try to generate
            try:
                from .benchmarks.cue_bench import generate_and_convert
                samples = generate_and_convert(
                    subset_name,
                    seed=self.config.seed,
                    samples_per_config=self.config.samples_per_config,
                )
            except (ImportError, ValueError):
                from .benchmarks.cue_bench import FIXTURE_SAMPLES
                samples = [s for s in FIXTURE_SAMPLES if s.subset == subset_name]
                logger.warning(
                    "datagen not available for '%s', using fixture", subset_name
                )

        manifest = DatasetManifest(
            dataset_id=key.dataset_id,
            purpose=DatasetPurpose.EVAL,
            benchmark=self.config.benchmark,
            subset=subset_name,
            sample_ids=[s.id for s in samples],
            generation_config={
                "seed": self.config.seed,
                "samples_per_config": self.config.samples_per_config,
            },
            seed=self.config.seed,
        )

        self.resources.save_dataset(key, manifest, samples)
        return manifest, samples

    def _resolve_hidden_cache(
        self,
        key: ResourceKey,
        samples: list[Sample],
        model: Any,
        tokenizer: Any,
    ) -> HiddenStateCache:
        """S1: Resolve or build hidden state cache for the eval dataset."""
        existing = self.resources.resolve_cache(key)
        if existing is not None:
            return existing

        if model is not None and tokenizer is not None:
            from .cache_builder import build_hidden_cache
            cache = build_hidden_cache(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                model_id=self.config.model_id,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
        else:
            # TESTING-ONLY fallback: no model available, create random cache.
            # This path should never be hit in production runs — it exists
            # only for --fixture / --no-model smoke tests.
            from tests.helpers import make_synthetic_cache
            logger.warning(
                "No model available, creating synthetic cache (testing-only fallback)"
            )
            cache = make_synthetic_cache(
                n_samples=len(samples),
                n_layers=28,  # hardcoded for Qwen2.5-Coder-7B — see DESIGN note
                hidden_dim=64,  # small for testing
                sample_ids=[s.id for s in samples],
                model_id=self.config.model_id,
                answers=[s.answer for s in samples],
                seed=self.config.seed,
            )

        self.resources.save_cache(key, cache)
        return cache

    def _run_method(
        self,
        method_name: str,
        subset_name: str,
        eval_key: ResourceKey,
        eval_samples: list[Sample],
        eval_cache: HiddenStateCache,
        model: Any,
        tokenizer: Any,
    ) -> MethodResult:
        """S2: Run a single method."""
        method_cls = get_method_class(method_name)
        method = method_cls()

        # Build method config
        user_config = self.config.method_configs.get(method_name, {})

        # Pass ResourceKey info through the config dict for methods
        mc = MethodConfig(
            method=method_name,
            benchmark=self.config.benchmark,
            config={
                "eval_dataset_id": eval_key.dataset_id,
                "answer_space": self.benchmark.answer_meta.answer_space,
                "resource_key_benchmark": eval_key.benchmark,
                "resource_key_task": eval_key.task,
                "resource_key_seed": eval_key.seed,
                **user_config,
            },
        )

        if method.requirements().trained:
            mc.config.setdefault("fit_policy", self.config.fit_policy)
            # Build train key info for the method
            train_key = self._make_key(subset_name, "train")
            mc.config.setdefault("train_resource_key_split", "train")

        return method.run(
            config=mc,
            eval_samples=eval_samples,
            eval_cache=eval_cache,
            resources=self.resources,
            model=model,
        )

    def _train_key_for_subset(self, subset_name: str) -> ResourceKey:
        """Build a train ResourceKey for a subset."""
        return self._make_key(subset_name, "train")
