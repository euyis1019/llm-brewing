"""Intervention backend abstraction for causal validation.

Provides a minimal interface for running hidden-state interventions
without coupling causal validators to nnsight internals.

The NNsightInterventionBackend wraps existing nnsight_ops functions.
FakeInterventionBackend is provided for testing without a real model.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class InterventionRequest:
    """A single intervention request.

    Describes: "inject source_hidden at target_layer/target_position
    of target_prompt, then read the model's next-token output."

    The source_prompt is the original prompt (for provenance tracking);
    the intervention itself is performed on target_prompt.
    """
    sample_id: str
    source_prompt: str         # original eval prompt (for reference)
    target_prompt: str         # prompt to run intervention on
    source_hidden: np.ndarray  # (hidden_dim,) the hidden state to inject
    target_layer: int
    target_position: int       # token position to patch (-1 = last)
    answer_space: list[str] | None = None  # restrict comparison to these tokens (e.g. ["0".."9"])
    baseline_subtract: bool = True  # when answer_space is set: True=CSD-consistent (change from baseline), False=absolute argmax within answer_space
    injection_mode: str = "replace"  # "replace" | "alpha_blend" | "norm_match"
    injection_alpha: float = 1.0  # blending weight for alpha_blend mode: h = (1-α)*h_orig + α*h_source
    target_original_hidden: np.ndarray | None = None  # original hidden state at target layer (for norm_match/alpha_blend pre-computation)


@dataclass
class InterventionResponse:
    """Result of a single intervention."""
    sample_id: str
    original_output: str    # target prompt output WITHOUT intervention
    intervened_output: str  # target prompt output WITH intervention


class InterventionBackend(abc.ABC):
    """Abstract backend for running interventions on a model."""

    @abc.abstractmethod
    def run_interventions(
        self,
        requests: list[InterventionRequest],
    ) -> list[InterventionResponse]:
        """Run a batch of interventions and return responses."""
        ...

    @abc.abstractmethod
    def get_model_output(self, prompt: str) -> str:
        """Get the model's unintervened next-token output for a prompt."""
        ...


class NNsightInterventionBackend(InterventionBackend):
    """Backend that uses nnsight for interventions.

    Wraps the existing patchscope_lens and model inference capabilities
    from nnsight_ops.  Dtype of injected tensors is matched to the
    model's parameter dtype (not from proxy objects).

    When answer_space is set on a request, uses either:
    - CSD-consistent semantics (baseline subtraction + argmax in answer space)
    - Restricted argmax without baseline subtraction (for causal experiments)
    controlled by the request's baseline_subtract flag.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._baseline_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._output_cache: dict[str, str] = {}
        # Resolve dtype/device from model parameters (proxy objects don't
        # expose these — see nnsight_ops.patchscope_lens for precedent).
        _param = next(model.parameters())
        self._dtype = _param.dtype
        self._device = _param.device

    def _get_answer_token_ids(self, answer_space: list[str]) -> list[int]:
        ids = []
        for ans in answer_space:
            tids = self.tokenizer.encode(ans, add_special_tokens=False)
            ids.append(tids[-1] if tids else 0)
        return ids

    def _get_baseline_logits(
        self, target_prompt: str, answer_token_ids: list[int],
    ) -> np.ndarray:
        """Baseline answer logits for the target prompt (cached)."""
        cache_key = (target_prompt, tuple(answer_token_ids))
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        from brewing.nnsight_ops import get_next_token_logits

        with self.model.trace(target_prompt):
            logits = get_next_token_logits(self.model).cpu().save()

        logits_np = logits[0].detach().cpu().float().numpy()  # (vocab_size,)
        baseline = np.array([logits_np[tid] for tid in answer_token_ids])
        self._baseline_cache[cache_key] = baseline
        return baseline

    def _get_layer_device(self, layer_idx: int):
        """Get the actual device for a specific layer (handles device_map='auto')."""
        from brewing.nnsight_ops import get_layers
        return next(get_layers(self.model)[layer_idx].parameters()).device

    def run_interventions(
        self,
        requests: list[InterventionRequest],
    ) -> list[InterventionResponse]:
        import torch
        from brewing.nnsight_ops import (
            get_layer_output,
            get_next_token_logits,
        )

        responses = []
        for req in requests:
            # Pre-compute the injection vector outside the trace to minimize
            # nnsight proxy operations (which accumulate GPU memory).
            src = req.source_hidden  # numpy (hidden_dim,)

            if req.injection_mode == "replace":
                inject_np = src
            elif req.injection_mode == "norm_match":
                if req.target_original_hidden is None:
                    raise ValueError(
                        "norm_match requires target_original_hidden "
                        "(pass the cached hidden state at target_layer)")
                tgt = req.target_original_hidden
                src_norm = float(np.linalg.norm(src)) + 1e-8
                tgt_norm = float(np.linalg.norm(tgt))
                inject_np = src * (tgt_norm / src_norm)
            elif req.injection_mode == "alpha_blend":
                if req.target_original_hidden is None:
                    raise ValueError(
                        "alpha_blend requires target_original_hidden "
                        "(pass the cached hidden state at target_layer)")
                tgt = req.target_original_hidden
                a = req.injection_alpha
                inject_np = (1 - a) * tgt + a * src
            else:
                raise ValueError(f"Unknown injection_mode: {req.injection_mode}")

            # Resolve device for the target layer (may differ from self._device
            # when device_map="auto" spreads layers across GPUs).
            layer_device = self._get_layer_device(req.target_layer)
            inject_tensor = torch.tensor(
                inject_np,
            ).to(dtype=self._dtype, device=layer_device)

            # Single proxy op inside trace: simple assignment
            with self.model.trace(req.target_prompt):
                layer_out = get_layer_output(self.model, req.target_layer)
                layer_out[0, req.target_position] = inject_tensor

                logits = get_next_token_logits(self.model).cpu().save()

            logits_np = logits[0].detach().cpu().float().numpy()  # (vocab_size,)
            del logits  # free trace-retained GPU tensors early
            torch.cuda.empty_cache()

            if req.answer_space is not None:
                answer_ids = self._get_answer_token_ids(req.answer_space)
                answer_logits = np.array([logits_np[tid] for tid in answer_ids])

                if req.baseline_subtract:
                    # CSD-consistent: baseline subtraction + answer-space argmax
                    baseline = self._get_baseline_logits(req.target_prompt, answer_ids)
                    scores = answer_logits - baseline
                else:
                    # Absolute argmax within answer space (for causal experiments)
                    scores = answer_logits

                pred_idx = int(np.argmax(scores))
                intervened_output = req.answer_space[pred_idx]
                original_output = self.get_model_output(req.target_prompt)
            else:
                # Global argmax (legacy behavior)
                intervened_token_id = int(np.argmax(logits_np))
                intervened_output = self.tokenizer.decode(
                    [intervened_token_id], skip_special_tokens=True
                ).strip()
                original_output = self.get_model_output(req.target_prompt)

            responses.append(InterventionResponse(
                sample_id=req.sample_id,
                original_output=original_output,
                intervened_output=intervened_output,
            ))

        return responses

    def get_model_output(self, prompt: str) -> str:
        if prompt in self._output_cache:
            return self._output_cache[prompt]

        from brewing.nnsight_ops import get_next_token_logits

        with self.model.trace(prompt):
            logits = get_next_token_logits(self.model).cpu().save()

        token_id = logits.argmax(dim=-1).item()
        result = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
        self._output_cache[prompt] = result
        return result


class FakeInterventionBackend(InterventionBackend):
    """Fake backend for testing — returns deterministic results.

    By default: original_output is the baseline_output for the target
    prompt; intervened_output comes from the predictions dict (simulating
    successful information transfer from source hidden state).

    Args:
        predictions: {sample_id: intervened_output} for each sample.
        baseline_output: what the target prompt produces without
            intervention (default "0").
    """

    def __init__(
        self,
        predictions: dict[str, str] | None = None,
        baseline_output: str = "0",
    ):
        self.predictions = predictions or {}
        self.baseline_output = baseline_output

    def run_interventions(
        self,
        requests: list[InterventionRequest],
    ) -> list[InterventionResponse]:
        responses = []
        for req in requests:
            # Original = target prompt's unintervened output
            original = self.baseline_output
            # Intervened = what the model outputs after patching in
            # the source hidden state
            intervened = self.predictions.get(req.sample_id, self.baseline_output)
            responses.append(InterventionResponse(
                sample_id=req.sample_id,
                original_output=original,
                intervened_output=intervened,
            ))
        return responses

    def get_model_output(self, prompt: str) -> str:
        return self.baseline_output
