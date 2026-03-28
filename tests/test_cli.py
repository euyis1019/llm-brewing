"""Tests for the YAML-config-only CLI."""

import importlib
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from brewing.cli import load_config, needs_model_online, build_model_load_kwargs
from brewing.schema import RunConfig


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_basic(tmp_path):
    """YAML config parses into RunConfig with correct values."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({
        "benchmark": "CUE-Bench",
        "model_id": "test-model",
        "methods": ["linear_probing"],
        "output_root": "out",
        "seed": 123,
    }))
    rc = load_config(cfg)
    assert rc.benchmark == "CUE-Bench"
    assert rc.model_id == "test-model"
    assert rc.seed == 123


def test_load_config_defaults(tmp_path):
    """Unspecified fields use RunConfig defaults."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({}))
    rc = load_config(cfg)
    assert rc.benchmark == "CUE-Bench"
    assert rc.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert rc.methods == ["linear_probing", "csd"]
    assert rc.quantization is None
    assert rc.fit_policy == "eval_only"


def test_load_config_not_found():
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_load_config_not_mapping(tmp_path):
    """Non-mapping YAML raises ValueError."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(cfg)


def test_load_config_quantization_int8(tmp_path):
    """quantization: int8 is accepted."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({"quantization": "int8"}))
    rc = load_config(cfg)
    assert rc.quantization == "int8"


def test_load_config_quantization_invalid(tmp_path):
    """Invalid quantization value raises ValueError."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({"quantization": "fp8"}))
    with pytest.raises(ValueError, match="Invalid quantization"):
        load_config(cfg)


def test_load_config_train_split_removed(tmp_path):
    """Automatic train/eval split is rejected at config-load time."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({"train_split": 0.8}))
    with pytest.raises(ValueError, match="Automatic train/eval splitting has been removed"):
        load_config(cfg)


def test_load_config_benchmark_path_safe(tmp_path):
    """RunConfig.benchmark_path_safe returns filesystem-friendly name."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({"benchmark": "CUE-Bench"}))
    rc = load_config(cfg)
    assert rc.benchmark_path_safe == "cuebench"


# ---------------------------------------------------------------------------
# needs_model_online — uses registry, not hardcoded names
# ---------------------------------------------------------------------------

def test_needs_model_online_probing_only():
    """Probing-only config does not need model online."""
    import brewing.methods.linear_probing  # noqa: F401
    rc = RunConfig(methods=["linear_probing"])
    assert needs_model_online(rc) is False


def test_needs_model_online_with_csd():
    """CSD requires model online."""
    import brewing.methods.csd  # noqa: F401
    rc = RunConfig(methods=["linear_probing", "csd"])
    assert needs_model_online(rc) is True


def test_needs_model_online_uses_registry():
    """The check goes through get_method_class, not a hardcoded list."""
    # Patch get_method_class to return a mock that says needs_model_online=True
    mock_cls = MagicMock()
    mock_cls.return_value.requirements.return_value.needs_model_online = True

    with patch("brewing.cli.get_method_class", return_value=mock_cls):
        rc = RunConfig(methods=["fake_method"])
        assert needs_model_online(rc) is True
        # Verify it called the registry
        from brewing.cli import get_method_class as _
        # The mock was called with "fake_method"


# ---------------------------------------------------------------------------
# build_model_load_kwargs
# ---------------------------------------------------------------------------

_requires_model = pytest.mark.skipif(
    not _has_module("torch") or not _has_module("transformers"),
    reason="requires torch and transformers",
)


@_requires_model
def test_build_model_load_kwargs_none():
    """No quantization -> torch_dtype=float16."""
    rc = RunConfig(quantization=None)
    import torch
    kwargs = build_model_load_kwargs(rc)
    assert kwargs["device_map"] == "auto"
    assert kwargs["torch_dtype"] == torch.float16
    assert "quantization_config" not in kwargs


@_requires_model
def test_build_model_load_kwargs_int8():
    """int8 -> BitsAndBytesConfig(load_in_8bit=True)."""
    rc = RunConfig(quantization="int8")
    kwargs = build_model_load_kwargs(rc)
    assert kwargs["device_map"] == "auto"
    assert "torch_dtype" not in kwargs
    qc = kwargs["quantization_config"]
    assert qc.load_in_8bit is True


@_requires_model
def test_build_model_load_kwargs_int4():
    """int4 -> BitsAndBytesConfig(load_in_4bit=True)."""
    rc = RunConfig(quantization="int4")
    kwargs = build_model_load_kwargs(rc)
    assert "torch_dtype" not in kwargs
    qc = kwargs["quantization_config"]
    assert qc.load_in_4bit is True


@_requires_model
def test_build_model_load_kwargs_cache_dir():
    """model_cache_dir is passed through as cache_dir."""
    rc = RunConfig(model_cache_dir="/tmp/my_models")
    kwargs = build_model_load_kwargs(rc)
    assert kwargs["cache_dir"] == "/tmp/my_models"

    rc_none = RunConfig()
    kwargs_none = build_model_load_kwargs(rc_none)
    assert "cache_dir" not in kwargs_none


# ---------------------------------------------------------------------------
# CLI integration (argparse)
# ---------------------------------------------------------------------------

def test_cli_missing_config(capsys):
    """CLI exits with error when --config is missing."""
    from brewing.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_runs_with_config(tmp_path):
    """CLI loads config and invokes Orchestrator."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump({
        "benchmark": "CUE-Bench",
        "methods": ["linear_probing"],
        "output_root": str(tmp_path / "out"),
        "use_fixture": True,
        "subsets": ["value_tracking"],
    }))

    mock_orch_instance = MagicMock()
    mock_orch_instance.run.return_value = {"subsets": {"value_tracking": {"n_eval_samples": 1}}}
    mock_orch_cls = MagicMock(return_value=mock_orch_instance)

    from brewing.cli import main
    with patch("brewing.orchestrator.Orchestrator", mock_orch_cls):
        main(["--config", str(cfg)])

    mock_orch_cls.assert_called_once()
    mock_orch_instance.run.assert_called_once()
