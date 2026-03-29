"""Tests for pipeline factory and individual pipeline modes."""

from unittest.mock import patch

import pytest

import brewing.benchmarks  # noqa: F401
import brewing.methods.linear_probing  # noqa: F401
import brewing.methods.csd  # noqa: F401

from brewing.schema import RunConfig
from brewing.orchestrator import Orchestrator
from brewing.pipelines import (
    PipelineBase,
    CacheOnlyPipeline,
    DiagnosticsPipeline,
    EvalPipeline,
    TrainPipeline,
    PIPELINE_REGISTRY,
    create_pipeline,
)
from brewing.resources import ResourceManager
from brewing.registry import get_benchmark
from tests.helpers import make_synthetic_cache


def _mock_resolve_hidden_cache(self, key, samples, model, tokenizer):
    """Test-only: build a synthetic cache instead of requiring a real model."""
    existing = self.resources.resolve_cache(key)
    if existing is not None:
        return existing
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


# ---------------------------------------------------------------------------
# RunConfig mode validation
# ---------------------------------------------------------------------------

def test_mode_default_is_eval():
    rc = RunConfig()
    assert rc.mode == "eval"


def test_mode_valid_values():
    for mode in ("cache_only", "train_probing", "eval", "diagnostics"):
        rc = RunConfig(mode=mode)
        assert rc.mode == mode


def test_mode_invalid_raises():
    with pytest.raises(ValueError, match="Invalid mode"):
        RunConfig(mode="bogus")


def test_mode_backward_compat_no_mode_field(tmp_path):
    """Config without mode field defaults to eval (backward compat)."""
    import yaml
    from brewing.cli import load_config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({"benchmark": "CUE-Bench"}))
    rc = load_config(cfg)
    assert rc.mode == "eval"


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def test_pipeline_registry_has_all_modes():
    assert set(PIPELINE_REGISTRY.keys()) == {
        "cache_only", "train_probing", "eval", "diagnostics",
    }


def test_create_pipeline_returns_correct_type(tmp_path):
    rm = ResourceManager(tmp_path / "out")
    bm = get_benchmark("CUE-Bench")

    for mode, expected_cls in [
        ("eval", EvalPipeline),
        ("cache_only", CacheOnlyPipeline),
        ("train_probing", TrainPipeline),
        ("diagnostics", DiagnosticsPipeline),
    ]:
        rc = RunConfig(mode=mode, output_root=str(tmp_path / "out"))
        pipeline = create_pipeline(rc, rm, bm)
        assert isinstance(pipeline, expected_cls)


def test_create_pipeline_unknown_mode(tmp_path):
    """Factory raises on invalid mode (shouldn't happen if RunConfig validates)."""
    rm = ResourceManager(tmp_path / "out")
    bm = get_benchmark("CUE-Bench")
    rc = RunConfig(output_root=str(tmp_path / "out"))
    # Force invalid mode past validation
    object.__setattr__(rc, "mode", "nonexistent")
    with pytest.raises(ValueError, match="Unknown mode"):
        create_pipeline(rc, rm, bm)


# ---------------------------------------------------------------------------
# Orchestrator delegates to pipeline
# ---------------------------------------------------------------------------

@patch.object(PipelineBase, "resolve_hidden_cache", _mock_resolve_hidden_cache)
def test_orchestrator_eval_mode_fixture(tmp_path):
    """Default mode=eval runs the same fixture pipeline as before."""
    rc = RunConfig(
        output_root=str(tmp_path / "out"),
        methods=["linear_probing"],
        subsets=["value_tracking"],
        use_fixture=True,
    )
    orch = Orchestrator(rc)
    # This will fail at S2 because no probe artifact exists, but
    # S0 + S1 should succeed, verifying the pipeline delegation works.
    result = orch.run()
    assert "subsets" in result
    assert "value_tracking" in result["subsets"]


@patch.object(PipelineBase, "resolve_hidden_cache", _mock_resolve_hidden_cache)
def test_orchestrator_cache_only_mode(tmp_path):
    """cache_only mode runs S0+S1 and produces dataset + cache."""
    rc = RunConfig(
        mode="cache_only",
        output_root=str(tmp_path / "out"),
        subsets=["value_tracking"],
        use_fixture=True,
    )
    orch = Orchestrator(rc)
    result = orch.run()
    subset_r = result["subsets"]["value_tracking"]
    assert "n_eval_samples" in subset_r
    assert "n_layers" in subset_r
    # No method results should appear
    assert not any(k.startswith("method_") for k in subset_r)


@patch.object(PipelineBase, "resolve_hidden_cache", _mock_resolve_hidden_cache)
def test_orchestrator_train_mode(tmp_path):
    """train_probing mode runs S0+S1+fit and produces artifact."""
    rc = RunConfig(
        mode="train_probing",
        output_root=str(tmp_path / "out"),
        subsets=["value_tracking"],
        use_fixture=True,
    )
    orch = Orchestrator(rc)
    result = orch.run()
    subset_r = result["subsets"]["value_tracking"]
    assert subset_r.get("fit_status") == "trained"
    assert "artifact_id" in subset_r


# ---------------------------------------------------------------------------
# CLI needs_model_online with mode awareness
# ---------------------------------------------------------------------------

def test_needs_model_cache_only():
    from brewing.cli import needs_model_online
    rc = RunConfig(mode="cache_only", methods=["linear_probing"])
    assert needs_model_online(rc) is True


def test_needs_model_train_probing():
    from brewing.cli import needs_model_online
    rc = RunConfig(mode="train_probing", methods=["linear_probing"])
    assert needs_model_online(rc) is True


def test_needs_model_diagnostics():
    from brewing.cli import needs_model_online
    rc = RunConfig(mode="diagnostics", methods=["linear_probing", "csd"])
    assert needs_model_online(rc) is False


def test_needs_model_eval_probing_only():
    from brewing.cli import needs_model_online
    rc = RunConfig(mode="eval", methods=["linear_probing"])
    assert needs_model_online(rc) is False


def test_needs_model_eval_with_csd():
    from brewing.cli import needs_model_online
    rc = RunConfig(mode="eval", methods=["linear_probing", "csd"])
    assert needs_model_online(rc) is True


# ---------------------------------------------------------------------------
# TrainPipeline: validate_on_eval
# ---------------------------------------------------------------------------

@patch.object(PipelineBase, "resolve_hidden_cache", _mock_resolve_hidden_cache)
def test_train_with_validate_on_eval(tmp_path):
    """train_probing with validate_on_eval=True also reports eval accuracy."""
    rc = RunConfig(
        mode="train_probing",
        output_root=str(tmp_path / "out"),
        subsets=["value_tracking"],
        use_fixture=True,
        method_configs={
            "linear_probing": {
                "validate_on_eval": True,
            },
        },
    )
    orch = Orchestrator(rc)
    result = orch.run()
    subset_r = result["subsets"]["value_tracking"]
    assert subset_r.get("fit_status") == "trained"
    # Validation metrics should be present
    ev = subset_r.get("eval_validation")
    assert ev is not None
    assert "per_layer_accuracy" in ev
    assert "best_layer" in ev
    assert "best_accuracy" in ev
    assert "n_eval_samples" in ev
    assert ev["n_eval_samples"] > 0
    # Accuracies should be in [0, 1]
    assert 0.0 <= ev["best_accuracy"] <= 1.0


@patch.object(PipelineBase, "resolve_hidden_cache", _mock_resolve_hidden_cache)
def test_train_without_validate_on_eval(tmp_path):
    """train_probing with validate_on_eval=False (default) skips eval."""
    rc = RunConfig(
        mode="train_probing",
        output_root=str(tmp_path / "out"),
        subsets=["value_tracking"],
        use_fixture=True,
    )
    orch = Orchestrator(rc)
    result = orch.run()
    subset_r = result["subsets"]["value_tracking"]
    assert subset_r.get("fit_status") == "trained"
    assert "eval_validation" not in subset_r
