"""Pipeline factory — mode-based dispatch for Brewing runs.

Each RunConfig.mode maps to a concrete PipelineBase subclass.
The create_pipeline() factory is the single dispatch point used
by Orchestrator.run().
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import PipelineBase
from .cache_only import CacheOnlyPipeline
from .causal_validation import CausalValidationPipeline
from .diagnostics import DiagnosticsPipeline
from .eval import EvalPipeline
from .train import TrainPipeline

if TYPE_CHECKING:
    from brewing.schema import RunConfig
    from brewing.resources import ResourceManager

PIPELINE_REGISTRY: dict[str, type[PipelineBase]] = {
    "cache_only": CacheOnlyPipeline,
    "train_probing": TrainPipeline,
    "eval": EvalPipeline,
    "diagnostics": DiagnosticsPipeline,
    "causal_validation": CausalValidationPipeline,
}


def create_pipeline(
    config: "RunConfig",
    resources: "ResourceManager",
    benchmark: Any,
) -> PipelineBase:
    """Create a pipeline instance based on config.mode."""
    cls = PIPELINE_REGISTRY.get(config.mode)
    if cls is None:
        raise ValueError(
            f"Unknown mode: {config.mode!r}. "
            f"Available: {list(PIPELINE_REGISTRY.keys())}"
        )
    return cls(config, resources, benchmark)


__all__ = [
    "PipelineBase",
    "CacheOnlyPipeline",
    "CausalValidationPipeline",
    "DiagnosticsPipeline",
    "EvalPipeline",
    "TrainPipeline",
    "PIPELINE_REGISTRY",
    "create_pipeline",
]
