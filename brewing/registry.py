"""Benchmark and method registry — plugin discovery for the framework.

Responsible for: maintaining a name→spec/class mapping so the Orchestrator
can look up benchmarks and methods by string name without importing them
directly.

Registration happens at import time via side effects:
  - ``import brewing.benchmarks`` registers CUE-Bench
  - ``import brewing.methods.linear_probing`` registers linear_probing
  - ``import brewing.methods.csd`` registers csd

To add a new benchmark or method, call register_benchmark/register_method
at module scope in the relevant file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import BenchmarkSpec
    from .methods.base import AnalysisMethod

_BENCHMARKS: dict[str, BenchmarkSpec] = {}
_METHODS: dict[str, type[AnalysisMethod]] = {}


def register_benchmark(spec: BenchmarkSpec) -> None:
    _BENCHMARKS[spec.name] = spec


def get_benchmark(name: str) -> BenchmarkSpec:
    if name not in _BENCHMARKS:
        raise KeyError(
            f"Benchmark '{name}' not registered. "
            f"Available: {list(_BENCHMARKS.keys())}"
        )
    return _BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    return list(_BENCHMARKS.keys())


def register_method(name: str, cls: type[AnalysisMethod]) -> None:
    _METHODS[name] = cls


def get_method_class(name: str) -> type[AnalysisMethod]:
    if name not in _METHODS:
        raise KeyError(
            f"Method '{name}' not registered. "
            f"Available: {list(_METHODS.keys())}"
        )
    return _METHODS[name]


def list_methods() -> list[str]:
    return list(_METHODS.keys())
