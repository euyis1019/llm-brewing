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
    target layer's actual dtype (not hardcoded).
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

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
            # Get target prompt's original output (no intervention)
            original_output = self.get_model_output(req.target_prompt)

            # Run intervention: inject source hidden into target prompt
            with self.model.trace(req.target_prompt):
                layer_out = get_layer_output(self.model, req.target_layer)
                # Match dtype and device of the target layer
                target_dtype = layer_out.dtype
                target_device = layer_out.device
                source = torch.tensor(
                    req.source_hidden,
                ).to(dtype=target_dtype, device=target_device)
                layer_out[0, req.target_position] = source
                logits = get_next_token_logits(self.model).cpu().save()

            intervened_token_id = logits.argmax(dim=-1).item()
            intervened_output = self.tokenizer.decode(
                [intervened_token_id], skip_special_tokens=True
            ).strip()

            responses.append(InterventionResponse(
                sample_id=req.sample_id,
                original_output=original_output,
                intervened_output=intervened_output,
            ))

        return responses

    def get_model_output(self, prompt: str) -> str:
        from brewing.nnsight_ops import get_next_token_logits

        with self.model.trace(prompt):
            logits = get_next_token_logits(self.model).cpu().save()

        token_id = logits.argmax(dim=-1).item()
        return self.tokenizer.decode([token_id], skip_special_tokens=True).strip()


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
