"""NNsight-based model operations — the tracing/intervention backend.

Responsible for: providing thin wrappers around nnsight's tracing API so
that cache_builder and CSD don't need to know nnsight internals.

NOT responsible for: any analysis logic, metric computation, or I/O.

Capabilities:
  - Layer access helpers (get_layers, get_layer_output, get_logits, etc.)
  - Hidden state extraction (get_token_activations)
  - Patchscope-style interventions (patchscope_lens: inject hidden state
    into a target prompt and read out next-token probabilities)

Adapted from nnterp (https://github.com/Butanium/nnterp), with nnterp's
import dependency removed — Brewing depends directly on nnsight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import torch as th
from nnsight import LanguageModel
from nnsight.intervention.tracing.globals import Object

TraceTensor = Union[th.Tensor, Object]
GetModuleOutput = Callable[[LanguageModel, int], TraceTensor]


# ---------------------------------------------------------------------------
# Layer access helpers
# ---------------------------------------------------------------------------

def _unpack_tuple(tensor_or_tuple: TraceTensor) -> TraceTensor:
    if isinstance(tensor_or_tuple, tuple):
        return tensor_or_tuple[0]
    return tensor_or_tuple


def get_layers(model: LanguageModel) -> list:
    if hasattr(model, "layers"):
        return model.layers
    return model.model.layers


def get_num_layers(model: LanguageModel) -> int:
    return len(get_layers(model))


def get_layer(model: LanguageModel, layer: int):
    return get_layers(model)[layer]


def get_layer_output(model: LanguageModel, layer: int) -> TraceTensor:
    return _unpack_tuple(get_layer(model, layer).output)


def get_logits(model: LanguageModel) -> TraceTensor:
    return model.output.logits


def get_next_token_probs(model: LanguageModel) -> TraceTensor:
    return get_logits(model)[:, -1, :].softmax(-1)


def get_next_token_logits(model: LanguageModel) -> TraceTensor:
    return get_logits(model)[:, -1, :]


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@th.no_grad()
def get_token_activations(
    model: LanguageModel,
    prompts: list[str] | str,
    layers: list[int] | None = None,
    get_activations: GetModuleOutput | None = None,
    idx: int = -1,
) -> th.Tensor:
    """Collect hidden states of a specific token at each layer.

    Args:
        model: NNsight LanguageModel / StandardizedTransformer.
        prompts: Prompts to process.
        layers: Layers to collect (default: all).
        get_activations: Custom extraction function (default: layer output).
        idx: Token position to extract (-1 = last token, requires left-padding).

    Returns:
        Tensor of shape ``(num_layers, num_prompts, hidden_size)`` on CPU.
    """
    if get_activations is None:
        get_activations = get_layer_output
    if layers is None:
        layers = list(range(get_num_layers(model)))

    acts = []
    with model.trace(prompts) as tracer:
        for layer in layers:
            acts.append(get_activations(model, layer)[:, idx].cpu().save())
        tracer.stop()
    return th.stack(acts)


# ---------------------------------------------------------------------------
# Patchscope dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TargetPrompt:
    """A prompt with a designated position for hidden-state injection."""
    prompt: str
    index_to_patch: int


@dataclass
class TargetPromptBatch:
    """Batched target prompts for patchscope interventions."""
    prompts: list[str]
    index_to_patch: th.Tensor

    @classmethod
    def from_target_prompt(cls, prompt: TargetPrompt, batch_size: int):
        return cls(
            [prompt.prompt] * batch_size,
            th.tensor([prompt.index_to_patch] * batch_size),
        )

    @classmethod
    def from_target_prompts(cls, prompts_: list[TargetPrompt]):
        return cls(
            [p.prompt for p in prompts_],
            th.tensor([p.index_to_patch for p in prompts_]),
        )

    def __len__(self):
        return len(self.prompts)

    @staticmethod
    def auto(
        target_prompt: TargetPrompt | list[TargetPrompt] | TargetPromptBatch,
        batch_size: int,
    ) -> TargetPromptBatch:
        if isinstance(target_prompt, TargetPrompt):
            return TargetPromptBatch.from_target_prompt(target_prompt, batch_size)
        elif isinstance(target_prompt, list):
            return TargetPromptBatch.from_target_prompts(target_prompt)
        elif isinstance(target_prompt, TargetPromptBatch):
            return target_prompt
        raise ValueError(
            f"Expected TargetPrompt, list[TargetPrompt], or TargetPromptBatch, "
            f"got {type(target_prompt)}"
        )


# ---------------------------------------------------------------------------
# Patchscope lens
# ---------------------------------------------------------------------------

@th.no_grad()
def patchscope_lens(
    nn_model: LanguageModel,
    source_prompts: list[str] | str | None = None,
    target_patch_prompts: TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None = None,
    layers: list[int] | None = None,
    latents: th.Tensor | None = None,
    return_logits: bool = False,
) -> th.Tensor:
    """Inject hidden states into a target prompt and read out next-token output.

    For each layer, replaces the hidden state at the designated position in
    *target_patch_prompts* with the corresponding latent, runs the rest of
    the forward pass, and collects next-token output.

    Args:
        nn_model: NNsight LanguageModel.
        source_prompts: Source prompts to extract latents from (mutually
            exclusive with *latents*).
        target_patch_prompts: Where to inject.
        layers: Layers to intervene on (default: all).
        latents: Pre-extracted hidden states, shape
            ``(num_layers, num_sources, hidden_size)``.
        return_logits: If True, return raw logits instead of probabilities.

    Returns:
        Tensor of shape ``(num_sources, num_layers, vocab_size)`` containing
        next-token **logits** (if *return_logits*) or **probabilities** on CPU.
    """
    if latents is not None:
        num_sources = latents.shape[1]
    else:
        if source_prompts is None:
            raise ValueError("Provide either source_prompts or latents")
        if isinstance(source_prompts, str):
            source_prompts = [source_prompts]
        num_sources = len(source_prompts)

    if target_patch_prompts is None:
        raise ValueError("target_patch_prompts is required")
    target_patch_prompts = TargetPromptBatch.auto(target_patch_prompts, num_sources)

    if layers is None:
        layers = list(range(get_num_layers(nn_model)))

    if latents is None:
        latents = get_token_activations(nn_model, source_prompts, layers=layers)
    elif source_prompts is not None:
        raise ValueError("Cannot provide both source_prompts and latents")

    output_fn = get_next_token_logits if return_logits else get_next_token_probs

    # Determine device and dtype outside trace context (proxy objects don't expose these)
    param = next(nn_model.parameters())
    device = param.device
    dtype = param.dtype

    results_l = []
    for idx, layer in enumerate(layers):
        with nn_model.trace(target_patch_prompts.prompts):
            get_layer_output(nn_model, layer)[
                th.arange(num_sources), target_patch_prompts.index_to_patch
            ] = latents[idx].to(device=device, dtype=dtype)
            results_l.append(output_fn(nn_model).cpu().save())

    results = th.cat(results_l, dim=0)
    return results.reshape(len(layers), num_sources, -1).transpose(0, 1)
