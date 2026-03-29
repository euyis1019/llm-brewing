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

Implementation: PyTorch nn.Linear (or optional MLP) trained with Adam on
GPU (if available), wrapped in a sklearn-compatible interface for
predict/predict_proba.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn

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
    "lr": 1e-3,
    "epochs": 2000,
    "batch_size": 512,
    "weight_decay": 0.1,
}

# Answer space for CUE-Bench: digits 0-9
DIGIT_CLASSES = [str(d) for d in range(10)]
N_CLASSES = len(DIGIT_CLASSES)


def _encode_labels(answers: list[str], answer_space: list[str]) -> np.ndarray:
    """Encode string answers to integer labels.

    Answers not in answer_space raise ValueError.
    """
    space_map = {a: i for i, a in enumerate(answer_space)}
    labels = []
    for a in answers:
        if a not in space_map:
            raise ValueError(
                f"Answer '{a}' not in answer_space {answer_space}. "
                "All samples must have valid answers."
            )
        labels.append(space_map[a])
    return np.array(labels)


def _get_probe_device() -> torch.device:
    """Pick the best available device for probe training."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class _ProbeBase:
    """Shared predict/score/standardize logic for all probe types."""

    mean: np.ndarray | None
    std: np.ndarray | None

    @property
    def model(self) -> nn.Module:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._standardize(X)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(next(self.model.parameters()).device))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._standardize(X)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(next(self.model.parameters()).device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is not None and self.std is not None:
            return (X - self.mean) / self.std
        return X

    def to_cpu(self):
        self.model.cpu()
        return self


class LinearProbe(_ProbeBase):
    """Single-layer linear probe."""

    def __init__(self, in_features: int, n_classes: int):
        self.in_features = in_features
        self.n_classes = n_classes
        self.linear = nn.Linear(in_features, n_classes)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def model(self) -> nn.Module:
        return self.linear


class MLPProbe(_ProbeBase):
    """Two-layer MLP probe (hidden_dim // 4)."""

    def __init__(self, in_features: int, n_classes: int):
        self.in_features = in_features
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, n_classes),
        )
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def model(self) -> nn.Module:
        return self.net


def _make_probe(probe_type: str, in_features: int, n_classes: int):
    if probe_type == "mlp":
        return MLPProbe(in_features, n_classes)
    return LinearProbe(in_features, n_classes)


def _artifact_key_from_config(config: MethodConfig, model_id: str) -> ResourceKey:
    """Build a ResourceKey for artifact resolution from method config.

    Artifact paths are split-agnostic (a trained probe is not tied to a
    particular data split), so ``split`` defaults to a neutral value.
    """
    return ResourceKey(
        benchmark=config.config.get("resource_key_benchmark", "cuebench"),
        split="artifact",
        task=config.config.get("resource_key_task", "unknown"),
        seed=config.config.get("resource_key_seed", 42),
        model_id=model_id,
        method="linear_probing",
    )


class LinearProbing(CacheOnlyMethod):
    """Per-layer linear probe (PyTorch nn.Linear + Adam)."""

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
        artifact_key = _artifact_key_from_config(config, eval_cache.model_id)

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

                pred_label = answer_space[pred_idx]
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
        probe_type: str = "linear",
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
            probe_type=probe_type,
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
        probe_type: str = "linear",
    ) -> tuple[FitArtifact, list]:
        from tqdm import tqdm

        labels = _encode_labels(
            [s.answer for s in train_samples], answer_space
        )
        n_layers = train_cache.n_layers
        n_samples = len(train_samples)
        hidden_dim = train_cache.hidden_dim
        n_classes = len(answer_space)

        lr = probe_params.get("lr", 1e-3)
        epochs = probe_params.get("epochs", 2000)
        batch_size = probe_params.get("batch_size", 512)
        weight_decay = probe_params.get("weight_decay", 1e-2)
        device = _get_probe_device()

        logger.info(
            "Fitting %d probes on %d samples (model=%s, device=%s, epochs=%d)",
            n_layers, n_samples, train_cache.model_id, device, epochs,
        )

        y_tensor = torch.from_numpy(labels).long().to(device)

        probes: list[LinearProbe] = []
        fit_metrics: dict[str, Any] = {"per_layer": {}}
        t0 = time.time()

        layer_pbar = tqdm(
            range(n_layers),
            desc=f"Training probes ({n_layers} layers × {n_samples} samples)",
            unit="layer",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        # Split off 10% of training data for early stopping validation
        from sklearn.model_selection import train_test_split
        patience = probe_params.get("patience", 50)

        split_indices = np.arange(n_samples)
        try:
            tr_idx, es_idx = train_test_split(
                split_indices, test_size=0.1, random_state=42, stratify=labels,
            )
        except ValueError:
            # Stratified split fails when some classes have <2 members (e.g. fixtures)
            tr_idx, es_idx = train_test_split(
                split_indices, test_size=0.1, random_state=42,
            )

        for layer_idx in layer_pbar:
            X_np = train_cache.hidden_states[:, layer_idx, :]  # (N, D)

            # Standardize on train split only
            mean = X_np[tr_idx].mean(axis=0)
            std = X_np[tr_idx].std(axis=0) + 1e-8
            X_scaled = (X_np - mean) / std

            X_tr = torch.from_numpy(X_scaled[tr_idx]).float().to(device)
            y_tr = y_tensor[tr_idx]
            X_es = torch.from_numpy(X_scaled[es_idx]).float().to(device)
            y_es = y_tensor[es_idx]

            n_tr = len(tr_idx)

            # Build probe
            probe = _make_probe(probe_type, hidden_dim, n_classes)
            probe.mean = mean
            probe.std = std
            probe.model.to(device)

            optimizer = torch.optim.Adam(probe.model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            # Training loop with early stopping
            probe.model.train()
            use_minibatch = n_tr > batch_size
            best_val_loss = float("inf")
            best_state = None
            wait = 0
            stopped_epoch = epochs

            for epoch in range(epochs):
                if use_minibatch:
                    perm = torch.randperm(n_tr, device=device)
                    for start in range(0, n_tr, batch_size):
                        idx = perm[start:start + batch_size]
                        optimizer.zero_grad()
                        loss_fn(probe.model(X_tr[idx]), y_tr[idx]).backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss_fn(probe.model(X_tr), y_tr).backward()
                    optimizer.step()

                # Check validation loss every epoch
                with torch.no_grad():
                    val_loss = loss_fn(probe.model(X_es), y_es).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in probe.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        stopped_epoch = epoch + 1
                        break

            # Restore best weights
            if best_state is not None:
                probe.model.load_state_dict(best_state)

            # Eval on full training set
            probe.model.eval()
            X_all = torch.from_numpy(X_scaled).float().to(device)
            with torch.no_grad():
                preds = probe.model(X_all).argmax(dim=1)
                train_acc = float((preds == y_tensor).float().mean().item())

            fit_metrics["per_layer"][str(layer_idx)] = {
                "train_accuracy": train_acc,
                "stopped_epoch": stopped_epoch,
                "best_val_loss": round(best_val_loss, 4),
            }

            # Move to CPU for serialization
            probe.to_cpu()
            probes.append(probe)

            layer_pbar.set_postfix_str(
                f"L{layer_idx} acc={train_acc:.1%} ep={stopped_epoch}"
            )

        fit_metrics["total_time_s"] = time.time() - t0
        fit_metrics["n_layers"] = n_layers
        fit_metrics["n_train"] = n_samples
        fit_metrics["device"] = str(device)

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
                "n_classes": n_classes,
                "answer_space": answer_space,
                "probe_type": probe_type,
            },
        )

        return artifact, probes


# Register
register_method("linear_probing", LinearProbing)
