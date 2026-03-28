"""Base types and data containers for the Brewing framework.

Defines the core value objects that flow through the pipeline:
  - Enumerations (Outcome, FitPolicy, Granularity, etc.)
  - Sample: a single benchmark item (prompt + answer + metadata)
  - DatasetManifest: metadata about a persisted dataset
  - HiddenStateCache: (N, L, D) activation tensor + per-sample predictions
  - FitArtifact: metadata about a trained model artifact (e.g., probe weights)
  - AnswerMeta: answer-space description for benchmark compatibility checks

All types are plain dataclasses with to_dict/from_dict serialization.
No runtime dependencies on model code or external libraries beyond numpy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AnswerType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    FREE_TEXT = "free_text"
    CODE = "code"


class SingleTokenRequirement(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    NOT_NEEDED = "not_needed"


class Outcome(str, Enum):
    RESOLVED = "resolved"
    OVERPROCESSED = "overprocessed"
    MISRESOLVED = "misresolved"
    UNRESOLVED = "unresolved"


class DatasetPurpose(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    CALIBRATION = "calibration"


class FitPolicy(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    EVAL_ONLY = "eval_only"


class FitStatus(str, Enum):
    LOADED = "loaded"
    TRAINED = "trained"


class Granularity(str, Enum):
    PER_SAMPLE = "per_sample"
    AGGREGATE = "aggregate"


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    id: str
    benchmark: str
    subset: str
    prompt: str
    answer: str
    difficulty: dict | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Sample:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2, ensure_ascii=False)


def load_samples(path: Path) -> list[Sample]:
    with open(path) as f:
        return [Sample.from_dict(d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# AnswerMeta
# ---------------------------------------------------------------------------

@dataclass
class AnswerMeta:
    answer_type: AnswerType
    answer_space: list[str] | None = None
    max_answer_tokens: int | None = None


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    dataset_id: str
    purpose: DatasetPurpose
    benchmark: str
    subset: str | None = None
    sample_ids: list[str] = field(default_factory=list)
    generation_config: dict = field(default_factory=dict)
    seed: int | None = None
    version: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["purpose"] = self.purpose.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DatasetManifest:
        d = dict(d)
        d["purpose"] = DatasetPurpose(d["purpose"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# HiddenStateCache
# ---------------------------------------------------------------------------

@dataclass
class HiddenStateCache:
    """Activations cache: N samples x L layers x D hidden_dim."""
    model_id: str
    sample_ids: list[str]
    hidden_states: np.ndarray  # (N, L, D)
    token_position: str | list[int] = "last"
    model_predictions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def n_layers(self) -> int:
        return self.hidden_states.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.hidden_states.shape[2]

    def save(self, path: Path, meta_path: Path | None = None) -> None:
        """Save hidden states and metadata.

        Args:
            path: Path for the .npz file.
            meta_path: Path for the metadata JSON. If None, uses
                ``meta.json`` as a sibling of *path*.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, hidden_states=self.hidden_states)
        if meta_path is None:
            meta_path = path.parent / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": self.model_id,
            "sample_ids": self.sample_ids,
            "token_position": self.token_position,
            "model_predictions": self.model_predictions,
            "metadata": self.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path, meta_path: Path | None = None) -> HiddenStateCache:
        """Load hidden states and metadata.

        Args:
            path: Path to the .npz file.
            meta_path: Path to the metadata JSON. If None, uses
                ``meta.json`` as a sibling of *path*.
        """
        data = np.load(path, allow_pickle=False)
        hidden_states = data["hidden_states"]
        if meta_path is None:
            meta_path = path.parent / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        return cls(
            model_id=meta["model_id"],
            sample_ids=meta["sample_ids"],
            hidden_states=hidden_states,
            token_position=meta["token_position"],
            model_predictions=meta.get("model_predictions", []),
            metadata=meta.get("metadata", {}),
        )

    def get_sample_states(self, sample_id: str) -> np.ndarray:
        """Return (L, D) hidden states for a single sample."""
        idx = self.sample_ids.index(sample_id)
        return self.hidden_states[idx]


# ---------------------------------------------------------------------------
# FitArtifact
# ---------------------------------------------------------------------------

@dataclass
class FitArtifact:
    artifact_id: str
    method: str
    model_id: str
    train_dataset_id: str
    train_cache_id: str | None = None
    fit_config: dict = field(default_factory=dict)
    fit_metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_artifact_id(method: str, model_id: str, train_dataset_id: str,
                         fit_config: dict | None = None) -> str:
        """Deterministic artifact ID from key dimensions."""
        parts = [method, model_id, train_dataset_id]
        if fit_config:
            config_str = json.dumps(fit_config, sort_keys=True)
            parts.append(hashlib.sha256(config_str.encode()).hexdigest()[:8])
        return "__".join(p.replace("/", "_") for p in parts)

    def save_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load_metadata(cls, path: Path) -> FitArtifact:
        with open(path) as f:
            return cls(**json.load(f))
