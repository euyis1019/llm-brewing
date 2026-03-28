"""Benchmark specification and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import AnswerMeta, SingleTokenRequirement
from .results import MethodRequirements


# ---------------------------------------------------------------------------
# SubsetSpec / BenchmarkSpec
# ---------------------------------------------------------------------------

@dataclass
class SubsetSpec:
    name: str
    category: str  # "data_flow" / "control_flow" / "data_control"
    difficulty_schema: dict = field(default_factory=dict)
    generate_fn: Callable | None = None
    question_suffix: str | None = None


@dataclass
class BenchmarkSpec:
    name: str
    domain: str
    answer_meta: AnswerMeta
    prompt_template: str | None = None
    subsets: list[SubsetSpec] = field(default_factory=list)

    def get_subset(self, name: str) -> SubsetSpec:
        for s in self.subsets:
            if s.name == name:
                return s
        raise KeyError(f"Subset '{name}' not found in {self.name}")

    @property
    def subset_names(self) -> list[str]:
        return [s.name for s in self.subsets]


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def check_compatibility(
    requirements: MethodRequirements,
    benchmark: BenchmarkSpec,
) -> list[tuple[str, str]]:
    """Return list of (severity, message). Empty = fully compatible."""
    issues: list[tuple[str, str]] = []
    am = benchmark.answer_meta

    if requirements.needs_answer_space and am.answer_space is None:
        issues.append(("INCOMPATIBLE",
                        f"Method needs finite answer space, but {benchmark.name} has none"))

    if requirements.single_token_answer == SingleTokenRequirement.REQUIRED:
        if am.max_answer_tokens is None:
            issues.append(("WARNING", "Answer token count unknown"))
        elif am.max_answer_tokens > 1:
            issues.append(("INCOMPATIBLE",
                            f"Method needs single-token answer, but max is {am.max_answer_tokens}"))

    if requirements.single_token_answer == SingleTokenRequirement.PREFERRED:
        if (am.max_answer_tokens or 999) > 1:
            issues.append(("WARNING", "Multi-token answers may change layer comparison semantics"))

    return issues
