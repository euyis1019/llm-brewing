"""Linear Probing — Information Availability (Phi_P).

Responsible for: fitting per-layer logistic regression probes on cached
hidden states and evaluating pre-trained probes to produce per-sample,
per-layer correctness and confidence values.

NOT responsible for: hidden state extraction (cache_builder), outcome
classification (diagnostics/), or model loading.

Cache-only, training-required method. Depends on a pre-built
HiddenStateCache but does NOT need the model online at eval time.

Important runtime contract:
  - `LinearProbing.train(...)` is the explicit training entry point.
  - `LinearProbing.run(...)` is evaluation-only and requires an existing
    serialized probe artifact.

COLM config (locked):
  - LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
  - 11-class: digits 0-9 + residual class (index 10)
  - Train/eval split is handled outside Brewing
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from brewing.schema import (
    FitArtifact, FitPolicy, FitStatus, Granularity,
    HiddenStateCache, MethodConfig, MethodRequirements, MethodResult,
    Sample, SampleMethodResult, SingleTokenRequirement,
)
from brewing.methods.base import CacheOnlyMethod
from brewing.registry import register_method
from brewing.resources import ResourceKey, ResourceManager

logger = logging.getLogger(__name__)

DEFAULT_PROBE_PARAMS = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 1000,
}

# Answer space for CUE-Bench: digits 0-9 + residual
DIGIT_CLASSES = [str(d) for d in range(10)]
RESIDUAL_CLASS = "__residual__"
ALL_CLASSES = DIGIT_CLASSES + [RESIDUAL_CLASS]
N_CLASSES = len(ALL_CLASSES)


def _encode_labels(answers: list[str], answer_space: list[str]) -> np.ndarray:
    """Encode string answers to integer labels.

    Answers not in answer_space get mapped to len(answer_space) (residual).
    """
    space_map = {a: i for i, a in enumerate(answer_space)}
    residual_idx = len(answer_space)
    return np.array([space_map.get(a, residual_idx) for a in answers])


def _artifact_key_from_config(config: MethodConfig, model_id: str, split: str = "train") -> ResourceKey:
    """Build a ResourceKey for artifact resolution from method config."""
    return ResourceKey(
        benchmark=config.config.get("resource_key_benchmark", "cuebench"),
        split=split,
        task=config.config.get("resource_key_task", "unknown"),
        seed=config.config.get("resource_key_seed", 42),
        model_id=model_id,
        method="linear_probing",
    )


class LinearProbing(CacheOnlyMethod):
    """Per-layer logistic regression probe."""

    name = "linear_probing"

    def _requirements(self) -> MethodRequirements:
        return MethodRequirements(
            needs_answer_space=True,
            single_token_answer=SingleTokenRequirement.NOT_NEEDED,
            trained=True,
        )

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
        probe_params = config.config.get("probe_params", DEFAULT_PROBE_PARAMS)
        fit_policy = config.fit_policy
        answer_space = config.config.get("answer_space", DIGIT_CLASSES)

        if fit_policy != FitPolicy.EVAL_ONLY:
            raise ValueError(
                "LinearProbing.run() is evaluation-only. Train probes in a "
                "separate script via LinearProbing.train(...) and set "
                "fit_policy=eval_only for evaluation runs."
            )
        if train_samples is not None or train_cache is not None:
            raise ValueError(
                "LinearProbing.run() does not accept train_samples/train_cache. "
                "Training must happen in a separate explicit fit step."
            )

        n_layers = eval_cache.n_layers

        # ---- Resolve artifact via ResourceKey ----
        artifact_key = _artifact_key_from_config(config, eval_cache.model_id, split="train")

        artifact = resources.resolve_artifact(artifact_key)
        if artifact is None:
            raise FileNotFoundError(
                f"Probe artifact not found at {resources.artifact_dir(artifact_key)}. "
                "Train it first via LinearProbing.train(...)."
            )
        probes = resources.load_artifact_model(artifact_key)
        artifact_answer_space = artifact.metadata.get("answer_space")
        if artifact_answer_space is not None and artifact_answer_space != answer_space:
            raise ValueError(
                "Configured answer_space does not match the trained probe artifact. "
                f"artifact={artifact_answer_space}, config={answer_space}"
            )
        if len(probes) != n_layers:
            raise ValueError(
                "Probe artifact layer count does not match eval cache. "
                f"artifact={len(probes)}, eval_cache={n_layers}"
            )

        sample_results: list[SampleMethodResult] = []
        for i, sample in enumerate(eval_samples):
            h = eval_cache.hidden_states[i]  # (L, D)
            layer_preds = []
            layer_vals = []
            layer_confs = []

            for layer_idx in range(n_layers):
                probe = probes[layer_idx]
                x = h[layer_idx].reshape(1, -1)
                pred_idx = int(probe.predict(x)[0])
                proba = probe.predict_proba(x)[0]

                pred_label = (
                    answer_space[pred_idx]
                    if pred_idx < len(answer_space)
                    else RESIDUAL_CLASS
                )
                correct = float(pred_label == sample.answer)

                layer_preds.append(pred_label)
                layer_vals.append(correct)
                layer_confs.append(proba)

            sample_results.append(SampleMethodResult(
                sample_id=sample.id,
                layer_values=np.array(layer_vals),
                layer_predictions=layer_preds,
                layer_confidences=np.array(layer_confs),
            ))

        return MethodResult(
            method=self.name,
            model_id=eval_cache.model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=config.config.get("eval_dataset_id", "unknown"),
            sample_results=sample_results,
            train_dataset_id=None,
            train_size=None,
            fit_artifact_id=artifact.artifact_id,
            fit_status=FitStatus.LOADED,
            fit_metrics_summary=artifact.fit_metrics,
        )

    def train(
        self,
        resources: ResourceManager,
        train_samples: list[Sample],
        train_cache: HiddenStateCache,
        *,
        artifact_key: ResourceKey,
        probe_params: dict | None = None,
        answer_space: list[str] | None = None,
        overwrite: bool = False,
    ) -> tuple[FitArtifact, list]:
        """Train probes explicitly and persist the resulting artifact.

        This is intentionally separate from `run()` so evaluation never
        performs hidden training steps or implicit dataset splitting.
        """
        if len(train_samples) != train_cache.n_samples:
            raise ValueError(
                "train_samples and train_cache must have the same number of samples. "
                f"samples={len(train_samples)}, cache={train_cache.n_samples}"
            )
        probe_params = probe_params or DEFAULT_PROBE_PARAMS
        answer_space = answer_space or DIGIT_CLASSES

        if resources.resolve_artifact(artifact_key) is not None and not overwrite:
            raise FileExistsError(
                f"Probe artifact already exists at {resources.artifact_dir(artifact_key)}. "
                "Pass overwrite=True to replace it."
            )

        artifact, probes = self._fit_probes(
            train_samples=train_samples,
            train_cache=train_cache,
            answer_space=answer_space,
            probe_params=probe_params,
            artifact_key=artifact_key,
        )
        resources.save_artifact(artifact_key, artifact, probes)
        return artifact, probes

    def _fit_probes(
        self,
        train_samples: list[Sample],
        train_cache: HiddenStateCache,
        answer_space: list[str],
        probe_params: dict,
        artifact_key: ResourceKey,
    ) -> tuple[FitArtifact, list]:
        from sklearn.linear_model import LogisticRegression

        labels = _encode_labels(
            [s.answer for s in train_samples], answer_space
        )
        n_layers = train_cache.n_layers

        logger.info(
            "Fitting %d probes on %d samples (model=%s)",
            n_layers, len(train_samples), train_cache.model_id,
        )

        probes = []
        fit_metrics: dict[str, Any] = {"per_layer": {}}
        t0 = time.time()

        for layer_idx in range(n_layers):
            X = train_cache.hidden_states[:, layer_idx, :]  # (N, D)
            clf = LogisticRegression(**probe_params)
            clf.fit(X, labels)

            train_acc = float(clf.score(X, labels))
            fit_metrics["per_layer"][str(layer_idx)] = {
                "train_accuracy": train_acc,
                "n_iter": int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else None,
            }
            probes.append(clf)

        fit_metrics["total_time_s"] = time.time() - t0
        fit_metrics["n_layers"] = n_layers
        fit_metrics["n_train"] = len(train_samples)

        # artifact_id is kept for metadata/serialization compatibility
        artifact_id = FitArtifact.make_artifact_id(
            method="linear_probing",
            model_id=train_cache.model_id,
            train_dataset_id=artifact_key.dataset_id,
            fit_config=probe_params,
        )

        artifact = FitArtifact(
            artifact_id=artifact_id,
            method="linear_probing",
            model_id=train_cache.model_id,
            train_dataset_id=artifact_key.dataset_id,
            train_cache_id=None,
            fit_config=probe_params,
            fit_metrics=fit_metrics,
            metadata={
                "n_classes": N_CLASSES,
                "answer_space": answer_space,
            },
        )

        return artifact, probes


# Register
register_method("linear_probing", LinearProbing)
