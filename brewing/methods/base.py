"""Base classes for analysis methods.

Defines the method type hierarchy:

  AnalysisMethod (abstract)
  ├── CacheOnlyMethod   — only needs HiddenStateCache, no GPU at eval time
  │                       (e.g., Linear Probing)
  └── ModelOnlineMethod  — needs the model loaded and online at eval time
                           (e.g., CSD, Activation Patching)

Both axes combine with trained/training-free:
  - Training-required methods manage FitArtifact via ResourceManager
  - Training-free methods skip the fit step entirely

All methods produce MethodResult as output.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from brewing.schema import (
    FitArtifact, FitPolicy, FitStatus, Granularity,
    HiddenStateCache, MethodConfig, MethodRequirements, MethodResult,
    Sample, SampleMethodResult,
)
from brewing.resources import ResourceManager

logger = logging.getLogger(__name__)


class AnalysisMethod(abc.ABC):
    """Abstract base for all analysis methods."""

    name: str  # e.g. "linear_probing", "csd"

    @abc.abstractmethod
    def requirements(self) -> MethodRequirements:
        """Declare what this method needs from the benchmark."""
        ...

    @abc.abstractmethod
    def run(
        self,
        config: MethodConfig,
        eval_samples: list[Sample],
        eval_cache: HiddenStateCache,
        resources: ResourceManager,
        model: Any = None,
        train_samples: list[Sample] | None = None,
        train_cache: HiddenStateCache | None = None,
    ) -> MethodResult:
        """Execute the method and return results.

        For training-required methods, this handles the full
        fit-or-load + evaluate flow.
        """
        ...


class CacheOnlyMethod(AnalysisMethod):
    """Method that only needs cached hidden states (no GPU at eval)."""

    def requirements(self) -> MethodRequirements:
        reqs = self._requirements()
        reqs.needs_model_online = False
        return reqs

    @abc.abstractmethod
    def _requirements(self) -> MethodRequirements:
        ...


class ModelOnlineMethod(AnalysisMethod):
    """Method that needs the model loaded and online."""

    def requirements(self) -> MethodRequirements:
        reqs = self._requirements()
        reqs.needs_model_online = True
        return reqs

    @abc.abstractmethod
    def _requirements(self) -> MethodRequirements:
        ...
