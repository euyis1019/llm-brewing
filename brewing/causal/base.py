"""Base class and registry for causal validators."""

from __future__ import annotations

import abc
from typing import Any

from brewing.schema import (
    CausalValidationResult,
    DiagnosticResult,
    HiddenStateCache,
    Sample,
)


class CausalValidator(abc.ABC):
    """Abstract base for causal validation experiments.

    Each validator implements a specific causal verification experiment
    (e.g., activation patching at FJC, layer skipping, re-injection).
    """

    name: str  # e.g. "activation_patching_fjc"

    @abc.abstractmethod
    def run(
        self,
        samples: list[Sample],
        cache: HiddenStateCache,
        diagnostics: DiagnosticResult,
        backend: Any,  # InterventionBackend
        *,
        config: dict | None = None,
        **kwargs: Any,
    ) -> CausalValidationResult:
        """Execute the causal validation experiment.

        Args:
            samples: Eval samples.
            cache: Hidden state cache for eval samples.
            diagnostics: S3 diagnostic results (contains FJC, outcomes, etc.).
            backend: InterventionBackend for running interventions.
            config: Per-experiment config dict from YAML
                (selector, intervention, decoding settings).
            **kwargs: Experiment-specific arguments.

        Returns:
            CausalValidationResult with per-sample results and summary.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VALIDATOR_REGISTRY: dict[str, type[CausalValidator]] = {}


def register_validator(cls: type[CausalValidator]) -> type[CausalValidator]:
    """Class decorator to register a CausalValidator."""
    VALIDATOR_REGISTRY[cls.name] = cls
    return cls


def get_validator(name: str) -> CausalValidator:
    """Instantiate a registered validator by name."""
    cls = VALIDATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown causal validator: {name!r}. "
            f"Available: {list(VALIDATOR_REGISTRY.keys())}"
        )
    return cls()
