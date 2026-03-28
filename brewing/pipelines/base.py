"""PipelineBase — shared S0/S1 logic for all pipeline modes.

Extracts the resolve-or-build patterns for datasets and hidden-state
caches from the former Orchestrator, so pipeline subclasses can reuse
them without copy-paste.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from brewing.schema import (
    DatasetManifest,
    DatasetPurpose,
    HiddenStateCache,
    RunConfig,
    Sample,
)
from brewing.registry import get_benchmark
from brewing.resources import ResourceKey, ResourceManager

logger = logging.getLogger(__name__)


class PipelineBase(ABC):
    """Abstract base for all pipeline modes."""

    def __init__(
        self,
        config: RunConfig,
        resources: ResourceManager,
        benchmark: Any,
    ):
        self.config = config
        self.resources = resources
        self.benchmark = benchmark
        self.subsets = config.subsets or benchmark.subset_names

    @abstractmethod
    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        """Execute this pipeline mode. Returns a summary dict."""
        ...

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------

    def make_key(self, task: str, split: str, **overrides: Any) -> ResourceKey:
        """Build a ResourceKey from config + task/split."""
        return ResourceKey(
            benchmark=self.config.benchmark_path_safe,
            split=split,
            task=task,
            seed=self.config.seed,
            model_id=overrides.get("model_id", self.config.model_id),
            method=overrides.get("method"),
        )

    # -----------------------------------------------------------------
    # S0: Resolve/build dataset
    # -----------------------------------------------------------------

    def resolve_dataset(
        self,
        subset_name: str,
        key: ResourceKey,
        purpose: DatasetPurpose = DatasetPurpose.EVAL,
    ) -> tuple[DatasetManifest, list[Sample]]:
        """S0: Resolve or build a dataset (eval or train)."""
        existing = self.resources.resolve_dataset(key)
        if existing is not None:
            return existing

        if self.config.use_fixture:
            from brewing.benchmarks.cue_bench import FIXTURE_SAMPLES
            samples = [s for s in FIXTURE_SAMPLES if s.subset == subset_name]
        elif self.config.data_dir:
            from brewing.benchmarks.cue_bench import load_generated_dataset
            samples = load_generated_dataset(
                Path(self.config.data_dir), subset_name
            )
        else:
            try:
                from brewing.benchmarks.cue_bench import generate_and_convert
                samples = generate_and_convert(
                    subset_name,
                    seed=self.config.seed,
                    samples_per_config=self.config.samples_per_config,
                )
            except (ImportError, ValueError):
                from brewing.benchmarks.cue_bench import FIXTURE_SAMPLES
                samples = [s for s in FIXTURE_SAMPLES if s.subset == subset_name]
                logger.warning(
                    "datagen not available for '%s', using fixture", subset_name
                )

        manifest = DatasetManifest(
            dataset_id=key.dataset_id,
            purpose=purpose,
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

    # -----------------------------------------------------------------
    # S1: Resolve/build hidden-state cache
    # -----------------------------------------------------------------

    def resolve_hidden_cache(
        self,
        key: ResourceKey,
        samples: list[Sample],
        model: Any,
        tokenizer: Any,
    ) -> HiddenStateCache:
        """S1: Resolve or build hidden state cache."""
        existing = self.resources.resolve_cache(key)
        if existing is not None:
            return existing

        if model is not None and tokenizer is not None:
            from brewing.cache_builder import build_hidden_cache
            cache = build_hidden_cache(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                model_id=self.config.model_id,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
        else:
            from tests.helpers import make_synthetic_cache
            logger.warning(
                "No model available, creating synthetic cache (testing-only fallback)"
            )
            cache = make_synthetic_cache(
                n_samples=len(samples),
                n_layers=28,
                hidden_dim=64,
                sample_ids=[s.id for s in samples],
                model_id=self.config.model_id,
                answers=[s.answer for s in samples],
                seed=self.config.seed,
            )

        self.resources.save_cache(key, cache)
        return cache
