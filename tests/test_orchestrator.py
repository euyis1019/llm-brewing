"""Tests for orchestration helpers under eval-only probing semantics."""

import brewing.benchmarks  # noqa: F401
import brewing.methods.linear_probing  # noqa: F401

from brewing.orchestrator import Orchestrator
from brewing.schema import RunConfig


def test_train_key_for_subset_uses_benchmark_path_safe(tmp_path):
    orch = Orchestrator(
        RunConfig(
            output_root=str(tmp_path / "out"),
            methods=["linear_probing"],
            subsets=["loop"],
            use_fixture=True,
        )
    )
    key = orch._train_key_for_subset("loop")
    assert key.benchmark == "cuebench"
    assert key.split == "train"
    assert key.task == "loop"
    assert key.seed == 42


def test_train_key_uses_config_seed(tmp_path):
    orch = Orchestrator(
        RunConfig(
            output_root=str(tmp_path / "out"),
            methods=["linear_probing"],
            subsets=["conditional"],
            use_fixture=True,
            seed=7,
        )
    )
    key = orch._train_key_for_subset("conditional")
    assert key.seed == 7
    assert key.dataset_id == "cuebench-conditional-train-seed7"


def test_make_key_includes_model_id(tmp_path):
    orch = Orchestrator(
        RunConfig(
            output_root=str(tmp_path / "out"),
            methods=["linear_probing"],
            subsets=["computing"],
            use_fixture=True,
            model_id="Qwen/Qwen2.5-Coder-7B",
        )
    )
    key = orch._make_key("computing", "eval")
    assert key.model_id == "Qwen/Qwen2.5-Coder-7B"
    assert key.model_id_safe == "Qwen__Qwen2.5-Coder-7B"
