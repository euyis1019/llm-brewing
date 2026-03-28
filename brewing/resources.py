"""Resource management: resolve-or-build pattern for persistent artifacts.

Responsible for: deterministic path layout, save/load lifecycle, and
the resolve-or-build pattern for all persistent objects in the framework.

NOT responsible for: deciding *what* to build (that's the Orchestrator),
or any analysis logic.

Managed resource types:
  - DatasetManifest + Samples
  - HiddenStateCache
  - FitArtifact (trained probe weights, etc.)
  - MethodResult (per-method output)
  - DiagnosticResult (S3 output)

Directory layout under output_root:
    {output_root}/
    ├── datasets/{benchmark}/{split}/{task}/seed{seed}/
    │   ├── manifest.json
    │   └── samples.json
    ├── caches/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
    │   ├── hidden_states.npz
    │   └── meta.json
    ├── artifacts/{benchmark}/{task}/{model_id_safe}/{method}/seed{seed}/
    │   ├── metadata.json
    │   └── model.pkl
    └── results/{benchmark}/{split}/{task}/seed{seed}/{model_id_safe}/
        ├── {method}.json
        └── diagnostics.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    DatasetManifest, DatasetPurpose, DiagnosticResult, FitArtifact, FitPolicy,
    HiddenStateCache, MethodResult, Sample, load_samples, save_samples,
)

logger = logging.getLogger(__name__)


def _safe_model_id(model_id: str) -> str:
    """Convert model_id to filesystem-safe string."""
    return model_id.replace("/", "__")


@dataclass(frozen=True)
class ResourceKey:
    """Structured key for locating resources in the output directory.

    Replaces the old flat dataset_id string with a hierarchical identity.
    """
    benchmark: str       # "cuebench"
    split: str           # "eval" / "train"
    task: str            # "computing", "conditional", etc.
    seed: int = 42
    model_id: str | None = None   # not needed for datasets
    method: str | None = None     # only for results/artifacts

    @property
    def dataset_id(self) -> str:
        """Derive a dataset_id string for metadata compatibility."""
        return f"{self.benchmark}-{self.task}-{self.split}-seed{self.seed}"

    @property
    def model_id_safe(self) -> str:
        """Filesystem-safe model ID."""
        if self.model_id is None:
            raise ValueError("ResourceKey.model_id is None; cannot derive model_id_safe")
        return _safe_model_id(self.model_id)


class ResourceManager:
    """Resolve-or-build resource manager.

    All persistent state goes under output_root. Resources are identified
    by deterministic paths derived from ResourceKey.
    """

    def __init__(self, output_root: Path | str):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Path helpers
    # =================================================================

    def dataset_dir(self, key: ResourceKey) -> Path:
        return (self.output_root / "datasets" /
                key.benchmark / key.split / key.task / f"seed{key.seed}")

    def manifest_path(self, key: ResourceKey) -> Path:
        return self.dataset_dir(key) / "manifest.json"

    def samples_path(self, key: ResourceKey) -> Path:
        return self.dataset_dir(key) / "samples.json"

    def cache_dir(self, key: ResourceKey) -> Path:
        return (self.output_root / "caches" /
                key.benchmark / key.split / key.task / f"seed{key.seed}" /
                key.model_id_safe)

    def cache_path(self, key: ResourceKey) -> Path:
        return self.cache_dir(key) / "hidden_states.npz"

    def cache_meta_path(self, key: ResourceKey) -> Path:
        return self.cache_dir(key) / "meta.json"

    def artifact_dir(self, key: ResourceKey) -> Path:
        if key.method is None:
            raise ValueError("ResourceKey.method is required for artifact paths")
        return (self.output_root / "artifacts" /
                key.benchmark / key.task / key.model_id_safe /
                key.method / f"seed{key.seed}")

    def artifact_meta_path(self, key: ResourceKey) -> Path:
        return self.artifact_dir(key) / "metadata.json"

    def artifact_model_path(self, key: ResourceKey) -> Path:
        return self.artifact_dir(key) / "model.pkl"

    def result_path(self, key: ResourceKey) -> Path:
        if key.method is None:
            raise ValueError("ResourceKey.method is required for result paths")
        return (self.output_root / "results" /
                key.benchmark / key.split / key.task / f"seed{key.seed}" /
                key.model_id_safe / f"{key.method}.json")

    def diagnostic_path(self, key: ResourceKey) -> Path:
        return (self.output_root / "results" /
                key.benchmark / key.split / key.task / f"seed{key.seed}" /
                key.model_id_safe / "diagnostics.json")

    # =================================================================
    # Dataset resolve/build
    # =================================================================

    def resolve_dataset(
        self, key: ResourceKey
    ) -> tuple[DatasetManifest, list[Sample]] | None:
        """Try to load an existing dataset. Returns None if not found."""
        mp = self.manifest_path(key)
        sp = self.samples_path(key)
        if mp.exists() and sp.exists():
            manifest = DatasetManifest.load(mp)
            samples = load_samples(sp)
            logger.info("Loaded dataset '%s' (%d samples)", key.dataset_id, len(samples))
            return manifest, samples
        return None

    def save_dataset(
        self, key: ResourceKey, manifest: DatasetManifest, samples: list[Sample]
    ) -> Path:
        """Persist a dataset."""
        manifest.save(self.manifest_path(key))
        save_samples(samples, self.samples_path(key))
        logger.info("Saved dataset '%s' (%d samples)", key.dataset_id, len(samples))
        return self.dataset_dir(key)

    def resolve_or_build_dataset(
        self,
        key: ResourceKey,
        build_fn: Any,  # Callable[[], tuple[DatasetManifest, list[Sample]]]
    ) -> tuple[DatasetManifest, list[Sample]]:
        """Load if exists, otherwise build and save."""
        existing = self.resolve_dataset(key)
        if existing is not None:
            return existing
        logger.info("Building dataset '%s'...", key.dataset_id)
        manifest, samples = build_fn()
        self.save_dataset(key, manifest, samples)
        return manifest, samples

    # =================================================================
    # Hidden state cache resolve/build
    # =================================================================

    def resolve_cache(
        self, key: ResourceKey
    ) -> HiddenStateCache | None:
        """Try to load an existing cache."""
        cp = self.cache_path(key)
        if cp.exists():
            cache = HiddenStateCache.load(cp, self.cache_meta_path(key))
            logger.info(
                "Loaded cache for '%s' on '%s' (%d samples, %d layers)",
                key.model_id, key.dataset_id, cache.n_samples, cache.n_layers,
            )
            return cache
        return None

    def save_cache(self, key: ResourceKey, cache: HiddenStateCache) -> Path:
        """Persist a hidden state cache."""
        cp = self.cache_path(key)
        cache.save(cp, self.cache_meta_path(key))
        logger.info(
            "Saved cache for '%s' on '%s' shape=%s",
            cache.model_id, key.dataset_id, cache.hidden_states.shape,
        )
        return cp

    def resolve_or_build_cache(
        self,
        key: ResourceKey,
        build_fn: Any,  # Callable[[], HiddenStateCache]
    ) -> HiddenStateCache:
        """Load if exists, otherwise build and save."""
        existing = self.resolve_cache(key)
        if existing is not None:
            return existing
        logger.info("Building cache for '%s' on '%s'...", key.model_id, key.dataset_id)
        cache = build_fn()
        self.save_cache(key, cache)
        return cache

    # =================================================================
    # Fit artifact resolve/build
    # =================================================================

    def resolve_artifact(self, key: ResourceKey) -> FitArtifact | None:
        """Try to load an existing artifact's metadata."""
        mp = self.artifact_meta_path(key)
        if mp.exists():
            artifact = FitArtifact.load_metadata(mp)
            logger.info("Loaded artifact for %s/%s/%s", key.method, key.model_id, key.task)
            return artifact
        return None

    def save_artifact(
        self,
        key: ResourceKey,
        artifact: FitArtifact,
        model_data: Any = None,
    ) -> Path:
        """Persist an artifact (metadata + optional model data)."""
        artifact.save_metadata(self.artifact_meta_path(key))
        if model_data is not None:
            import pickle
            model_path = self.artifact_model_path(key)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
        logger.info("Saved artifact for %s/%s/%s", key.method, key.model_id, key.task)
        return self.artifact_dir(key)

    def load_artifact_model(self, key: ResourceKey) -> Any:
        """Load the trained model data for an artifact."""
        import pickle
        mp = self.artifact_model_path(key)
        if not mp.exists():
            raise FileNotFoundError(
                f"No model file for artifact at {mp}"
            )
        with open(mp, "rb") as f:
            return pickle.load(f)

    def resolve_artifact_with_policy(
        self,
        key: ResourceKey,
        fit_policy: FitPolicy,
        fit_fn: Any,  # Callable[[], tuple[FitArtifact, Any]]
    ) -> tuple[FitArtifact, Any, bool]:
        """Resolve an artifact according to fit_policy.

        Returns:
            (artifact, model_data, was_trained)
        """
        if fit_policy == FitPolicy.FORCE:
            logger.info("fit_policy=force, training artifact for %s/%s", key.method, key.task)
            artifact, model_data = fit_fn()
            self.save_artifact(key, artifact, model_data)
            return artifact, model_data, True

        existing = self.resolve_artifact(key)
        if existing is not None:
            model_data = self.load_artifact_model(key)
            return existing, model_data, False

        if fit_policy == FitPolicy.EVAL_ONLY:
            raise FileNotFoundError(
                f"fit_policy=eval_only but artifact not found at {self.artifact_dir(key)}"
            )

        # AUTO: not found, train
        logger.info("fit_policy=auto, artifact not found, training for %s/%s", key.method, key.task)
        artifact, model_data = fit_fn()
        self.save_artifact(key, artifact, model_data)
        return artifact, model_data, True

    # =================================================================
    # Method results
    # =================================================================

    def resolve_result(
        self, key: ResourceKey
    ) -> MethodResult | None:
        """Try to load an existing method result."""
        rp = self.result_path(key)
        if rp.exists():
            return MethodResult.load(rp)
        return None

    def save_result(self, key: ResourceKey, result: MethodResult) -> Path:
        rp = self.result_path(key)
        result.save(rp)
        logger.info("Saved result for %s/%s/%s",
                     key.method, key.model_id, key.task)
        return rp

    # =================================================================
    # Diagnostic results
    # =================================================================

    def resolve_diagnostic(
        self, key: ResourceKey
    ) -> DiagnosticResult | None:
        """Try to load an existing diagnostic result."""
        dp = self.diagnostic_path(key)
        if dp.exists():
            return DiagnosticResult.load(dp)
        return None

    def save_diagnostic(self, key: ResourceKey, result: DiagnosticResult) -> Path:
        dp = self.diagnostic_path(key)
        result.save(dp)
        logger.info("Saved diagnostic for %s/%s/%s",
                     key.model_id, key.task, key.split)
        return dp
