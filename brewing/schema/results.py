"""Method results, diagnostic results, and pipeline configuration.

Output types produced by the pipeline:
  - SampleMethodResult: per-sample, per-layer output from a single method
  - MethodResult: collection of SampleMethodResult for a (method, model, dataset) run
  - SampleDiagnostic / DiagnosticResult: per-sample outcome classification (S3 output)

Also contains:
  - MethodRequirements / MethodConfig: method capability declarations
  - RunConfig: top-level pipeline configuration dataclass
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    FitPolicy,
    FitStatus,
    Granularity,
    Outcome,
    SingleTokenRequirement,
)


# ---------------------------------------------------------------------------
# SampleMethodResult / MethodResult
# ---------------------------------------------------------------------------

@dataclass
class SampleMethodResult:
    """Per-sample, per-layer result from a method."""
    sample_id: str
    layer_values: np.ndarray  # (L,) correctness or metric per layer
    layer_predictions: list[str] | np.ndarray | None = None  # (L,) predictions
    layer_confidences: np.ndarray | None = None  # (L, C) full distribution
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"sample_id": self.sample_id}
        d["layer_values"] = self.layer_values.tolist()
        if self.layer_predictions is not None:
            if isinstance(self.layer_predictions, np.ndarray):
                d["layer_predictions"] = self.layer_predictions.tolist()
            else:
                d["layer_predictions"] = list(self.layer_predictions)
        if self.layer_confidences is not None:
            d["layer_confidences"] = self.layer_confidences.tolist()
        if self.extras:
            d["extras"] = self.extras
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleMethodResult:
        return cls(
            sample_id=d["sample_id"],
            layer_values=np.array(d["layer_values"]),
            layer_predictions=(
                d.get("layer_predictions")
                if isinstance(d.get("layer_predictions"), list) and
                   any(isinstance(x, str) for x in d.get("layer_predictions", []))
                else np.array(d["layer_predictions"]) if d.get("layer_predictions") is not None
                else None
            ),
            layer_confidences=(
                np.array(d["layer_confidences"]) if d.get("layer_confidences") is not None
                else None
            ),
            extras=d.get("extras", {}),
        )


@dataclass
class MethodResult:
    """Result of a method on (model, dataset)."""
    method: str
    model_id: str
    granularity: Granularity
    eval_dataset_id: str

    # per-sample results (when granularity == PER_SAMPLE)
    sample_results: list[SampleMethodResult] = field(default_factory=list)

    # aggregate results (when granularity == AGGREGATE)
    layer_values: np.ndarray | None = None  # (L,)
    extras: dict = field(default_factory=dict)

    # training-required method fields
    train_dataset_id: str | None = None
    train_size: int | None = None
    fit_artifact_id: str | None = None
    fit_status: FitStatus | None = None
    fit_metrics_summary: dict | None = None

    def get_sample_result(self, sample_id: str) -> SampleMethodResult:
        for sr in self.sample_results:
            if sr.sample_id == sample_id:
                return sr
        raise KeyError(f"No result for sample '{sample_id}'")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d: dict[str, Any] = {
            "method": self.method,
            "model_id": self.model_id,
            "granularity": self.granularity.value,
            "eval_dataset_id": self.eval_dataset_id,
        }
        if self.granularity == Granularity.PER_SAMPLE:
            d["sample_results"] = [sr.to_dict() for sr in self.sample_results]
        if self.layer_values is not None:
            d["layer_values"] = self.layer_values.tolist()
        if self.extras:
            d["extras"] = self.extras
        for fld in ("train_dataset_id", "train_size", "fit_artifact_id",
                     "fit_metrics_summary"):
            val = getattr(self, fld)
            if val is not None:
                d[fld] = val
        if self.fit_status is not None:
            d["fit_status"] = self.fit_status.value
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> MethodResult:
        with open(path) as f:
            d = json.load(f)
        sample_results = [
            SampleMethodResult.from_dict(sr)
            for sr in d.get("sample_results", [])
        ]
        return cls(
            method=d["method"],
            model_id=d["model_id"],
            granularity=Granularity(d["granularity"]),
            eval_dataset_id=d["eval_dataset_id"],
            sample_results=sample_results,
            layer_values=np.array(d["layer_values"]) if "layer_values" in d else None,
            extras=d.get("extras", {}),
            train_dataset_id=d.get("train_dataset_id"),
            train_size=d.get("train_size"),
            fit_artifact_id=d.get("fit_artifact_id"),
            fit_status=FitStatus(d["fit_status"]) if d.get("fit_status") else None,
            fit_metrics_summary=d.get("fit_metrics_summary"),
        )


# ---------------------------------------------------------------------------
# MethodRequirements / MethodConfig
# ---------------------------------------------------------------------------

@dataclass
class MethodRequirements:
    needs_answer_space: bool = False
    single_token_answer: SingleTokenRequirement = SingleTokenRequirement.NOT_NEEDED
    needs_model_online: bool = False
    trained: bool = False
    custom_config_schema: dict = field(default_factory=dict)


@dataclass
class MethodConfig:
    method: str
    benchmark: str
    config: dict = field(default_factory=dict)

    @property
    def fit_policy(self) -> FitPolicy:
        return FitPolicy(self.config.get("fit_policy", "eval_only"))

    @property
    def train_dataset_id(self) -> str | None:
        return self.config.get("train_dataset_id")


# ---------------------------------------------------------------------------
# SampleDiagnostic / DiagnosticResult
# ---------------------------------------------------------------------------

@dataclass
class SampleDiagnostic:
    """Per-sample diagnostic indicators."""
    sample_id: str
    fpcl: int | None = None
    fjc: int | None = None
    delta_brew: int | None = None
    outcome: Outcome | None = None
    model_output: str | None = None
    csd_tail_confidence: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.outcome is not None:
            d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleDiagnostic:
        d = dict(d)
        if d.get("outcome") is not None:
            d["outcome"] = Outcome(d["outcome"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DiagnosticResult:
    """Full diagnostic output for a (model, subset) pair."""
    model_id: str
    eval_dataset_id: str
    benchmark: str
    subset: str | None = None

    sample_diagnostics: list[SampleDiagnostic] = field(default_factory=list)

    # aggregate
    outcome_distribution: dict[str, float] = field(default_factory=dict)
    mean_fpcl_normalized: float | None = None
    mean_fjc_normalized: float | None = None
    mean_delta_brew: float | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "eval_dataset_id": self.eval_dataset_id,
            "benchmark": self.benchmark,
            "subset": self.subset,
            "sample_diagnostics": [sd.to_dict() for sd in self.sample_diagnostics],
            "outcome_distribution": self.outcome_distribution,
            "mean_fpcl_normalized": self.mean_fpcl_normalized,
            "mean_fjc_normalized": self.mean_fjc_normalized,
            "mean_delta_brew": self.mean_delta_brew,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> DiagnosticResult:
        with open(path) as f:
            d = json.load(f)
        return cls(
            model_id=d["model_id"],
            eval_dataset_id=d["eval_dataset_id"],
            benchmark=d["benchmark"],
            subset=d.get("subset"),
            sample_diagnostics=[SampleDiagnostic.from_dict(sd)
                                for sd in d.get("sample_diagnostics", [])],
            outcome_distribution=d.get("outcome_distribution", {}),
            mean_fpcl_normalized=d.get("mean_fpcl_normalized"),
            mean_fjc_normalized=d.get("mean_fjc_normalized"),
            mean_delta_brew=d.get("mean_delta_brew"),
        )


# ---------------------------------------------------------------------------
# SampleCausalResult / CausalValidationResult
# ---------------------------------------------------------------------------

@dataclass
class SampleCausalResult:
    """Per-sample result from a causal validation experiment."""
    sample_id: str
    selected: bool
    skip_reason: str | None = None  # why this sample was not selected

    # Intervention details (only set when selected=True)
    source_layer: int | None = None
    target_layer: int | None = None
    round_idx: int | None = None  # for multi-round experiments (re-injection)

    original_output: str | None = None
    intervened_output: str | None = None
    original_correct: bool | None = None
    intervened_correct: bool | None = None

    effect_label: str | None = None  # "flipped", "maintained", "other"
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> SampleCausalResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CausalValidationResult:
    """Full result of a causal validation experiment."""
    experiment: str  # e.g. "activation_patching_fjc"
    model_id: str
    eval_dataset_id: str
    benchmark: str
    subset: str | None = None

    sample_results: list[SampleCausalResult] = field(default_factory=list)

    # Summary metrics
    summary: dict = field(default_factory=dict)
    # e.g. {"n_selected": 10, "n_effective": 8, "n_flipped": 5, "flip_rate": 0.625}

    extras: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d: dict[str, Any] = {
            "experiment": self.experiment,
            "model_id": self.model_id,
            "eval_dataset_id": self.eval_dataset_id,
            "benchmark": self.benchmark,
            "subset": self.subset,
            "sample_results": [sr.to_dict() for sr in self.sample_results],
            "summary": self.summary,
            "extras": self.extras,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CausalValidationResult:
        with open(path) as f:
            d = json.load(f)
        return cls(
            experiment=d["experiment"],
            model_id=d["model_id"],
            eval_dataset_id=d["eval_dataset_id"],
            benchmark=d["benchmark"],
            subset=d.get("subset"),
            sample_results=[
                SampleCausalResult.from_dict(sr)
                for sr in d.get("sample_results", [])
            ],
            summary=d.get("summary", {}),
            extras=d.get("extras", {}),
        )


# ---------------------------------------------------------------------------
# RunConfig — configuration for a single Brewing run
# ---------------------------------------------------------------------------

VALID_MODES = ("cache_only", "train_probing", "eval", "diagnostics", "causal_validation")


@dataclass
class RunConfig:
    """Configuration for a single Brewing run."""
    mode: str = "eval"  # cache_only | train_probing | eval | diagnostics
    benchmark: str = "CUE-Bench"
    subsets: list[str] | None = None  # None = all subsets
    splits: list[str] | None = None  # None = pipeline default (e.g. ["eval"]); cache_only uses ["train", "eval"]
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    methods: list[str] = field(default_factory=lambda: ["linear_probing", "csd"])
    output_root: str = "brewing_output"

    # Dataset config
    data_dir: str | None = None  # Path to pre-generated data
    seed: int = 42
    samples_per_config: int | None = None  # None = use default

    # Training config (for probing artifacts)
    train_split: float | None = None  # Deprecated: automatic split is no longer supported
    fit_policy: str = "eval_only"

    # Method-specific configs
    method_configs: dict[str, dict] = field(default_factory=dict)

    # Execution config
    batch_size: int = 8
    device: str | None = None  # Auto-detect
    use_fixture: bool = False  # If True, use fixture samples only

    # Model weights
    model_cache_dir: str | None = None  # Path to local HF model cache

    # Causal validation config (for mode="causal_validation")
    causal_validation: dict = field(default_factory=dict)

    # Quantization
    quantization: str | None = None  # None, "int8", "int4"

    _VALID_QUANTIZATIONS = (None, "int8", "int4")

    # Filesystem-safe benchmark name for paths
    _BENCHMARK_PATH_MAP = {
        "CUE-Bench": "cuebench",
    }

    def __post_init__(self):
        import warnings

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode={self.mode!r}. "
                f"Must be one of: {VALID_MODES}"
            )
        if self.quantization not in self._VALID_QUANTIZATIONS:
            raise ValueError(
                f"Invalid quantization={self.quantization!r}. "
                f"Must be one of: {self._VALID_QUANTIZATIONS}"
            )
        FitPolicy(self.fit_policy)
        if self.train_split is not None:
            raise ValueError(
                "Automatic train/eval splitting has been removed from Brewing. "
                "Prepare train/eval datasets externally and train probing "
                "artifacts in a separate script."
            )
        if self.mode == "train_probing" and self.data_dir is None:
            warnings.warn(
                "mode='train_probing' with data_dir=None: training data will be "
                "dynamically generated or loaded from output_root. Set data_dir "
                "to point to the official train split for reproducible experiments.",
                UserWarning,
                stacklevel=2,
            )
        if self.mode == "diagnostics" and self.methods:
            warnings.warn(
                "mode='diagnostics' ignores the 'methods' field. Diagnostics "
                "loads probe and CSD results from disk, it does not run methods.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def benchmark_path_safe(self) -> str:
        """Filesystem-safe benchmark name."""
        return self._BENCHMARK_PATH_MAP.get(self.benchmark, self.benchmark.lower().replace("-", ""))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
