"""Hidden state cache builder — S1 of the Brewing pipeline.

Responsible for: extracting last-token hidden states from all layers for a
batch of samples, producing an HiddenStateCache.

NOT responsible for: model loading, dataset construction, or any analysis.

Two extraction backends:
  - NNsight: used when the model has both `.trace()` and `.layers`
    attributes (i.e., an nnsight LanguageModel). Delegates to
    nnsight_ops.get_token_activations().
  - HuggingFace: fallback for plain HF AutoModelForCausalLM, using
    output_hidden_states=True.

Backend selection is a simple hasattr check (see `is_nnsight_model`),
not a full capability probe — models that expose `.trace()` but use a
non-standard layer structure may need the nnsight_ops helpers to be
extended (nnsight_ops.get_layers already handles both `model.layers`
and `model.model.layers`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .schema import HiddenStateCache, Sample

logger = logging.getLogger(__name__)

# Batch size for forward passes
DEFAULT_BATCH_SIZE = 8


def build_hidden_cache(
    model: Any,
    tokenizer: Any,
    samples: list[Sample],
    model_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token_position: str = "last",
    device: str | None = None,
) -> HiddenStateCache:
    """Extract hidden states from all layers for all samples.

    Auto-detects the model backend: uses nnsight tracing when available,
    falls back to HuggingFace output_hidden_states.

    Args:
        model: The loaded model (nnsight LanguageModel or HF AutoModelForCausalLM)
        tokenizer: The tokenizer
        samples: List of samples to process
        model_id: Model identifier string
        batch_size: Number of samples per forward pass
        token_position: "last" for last token position
        device: Device to use (inferred from model if None)
    """
    import torch

    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except (StopIteration, AttributeError):
            device = "cpu"

    all_hidden_states = []
    model_predictions = []

    # Backend detection: nnsight LanguageModel exposes .trace() and .layers.
    # This is a narrow check — see module docstring for limitations.
    is_nnsight_model = hasattr(model, "trace") and hasattr(model, "layers")

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start:batch_start + batch_size]
        prompts = [s.prompt for s in batch_samples]

        logger.info(
            "Processing batch %d-%d / %d",
            batch_start, batch_start + len(batch_samples), len(samples),
        )

        if is_nnsight_model:
            batch_hs, batch_preds = _extract_nnsight(
                model, tokenizer, prompts, device
            )
        else:
            batch_hs, batch_preds = _extract_hf(
                model, tokenizer, prompts, device
            )

        all_hidden_states.append(batch_hs)
        model_predictions.extend(batch_preds)

    # Stack all batches: (N, L, D)
    hidden_states = np.concatenate(all_hidden_states, axis=0)

    return HiddenStateCache(
        model_id=model_id,
        sample_ids=[s.id for s in samples],
        hidden_states=hidden_states,
        token_position=token_position,
        model_predictions=model_predictions,
        metadata={
            "n_layers": int(hidden_states.shape[1]),
            "hidden_dim": int(hidden_states.shape[2]),
            "dtype": str(hidden_states.dtype),
        },
    )


def _extract_hf(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Extract using HuggingFace's output_hidden_states=True."""
    import torch

    # Pad from left for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer+embedding
    # Skip embedding layer (index 0)
    hidden_layers = outputs.hidden_states[1:]  # L layers
    n_layers = len(hidden_layers)

    # Get last token position for each sample (accounting for padding)
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

    batch_size = len(prompts)
    hidden_dim = hidden_layers[0].shape[-1]

    # Extract last-token hidden states: (batch, L, D)
    batch_hs = np.zeros((batch_size, n_layers, hidden_dim), dtype=np.float32)
    for layer_idx, layer_states in enumerate(hidden_layers):
        for sample_idx in range(batch_size):
            pos = last_positions[sample_idx]
            batch_hs[sample_idx, layer_idx] = (
                layer_states[sample_idx, pos].cpu().float().numpy()
            )

    # Model predictions: greedy decode from final logits
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    predictions = []
    for sample_idx in range(batch_size):
        pos = last_positions[sample_idx]
        pred_id = logits[sample_idx, pos].argmax().item()
        pred_token = tokenizer.decode([pred_id]).strip()
        predictions.append(pred_token)

    return batch_hs, predictions


def _extract_nnsight(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Extract using nnsight tracing backend.

    Uses get_token_activations (nnsight_ops) for hidden state extraction,
    and a traced forward pass for model predictions.
    """
    import torch
    from brewing.nnsight_ops import get_token_activations

    # get_token_activations returns shape (num_layers, num_prompts, hidden_size)
    # It handles tokenization and padding internally
    activations = get_token_activations(
        model,
        prompts=prompts,
        layers=None,  # all layers
        idx=-1,  # last token
    )

    # activations: (L, N, D) -> transpose to (N, L, D)
    batch_hs = activations.cpu().float().numpy()
    batch_hs = np.transpose(batch_hs, (1, 0, 2))  # (N, L, D)

    # Get model predictions via traced forward pass
    with model.trace(prompts) as tracer:
        logits = model.output.logits.save()

    predictions = []
    logits_val = logits.value

    # nnsight LanguageModel exposes .tokenizer
    tok = model.tokenizer
    inputs = tok(prompts, return_tensors="pt", padding=True)
    attention_mask = inputs["attention_mask"].to(logits_val.device)
    last_positions = attention_mask.sum(dim=1) - 1

    for sample_idx in range(len(prompts)):
        pos = last_positions[sample_idx]
        pred_id = logits_val[sample_idx, pos].argmax().item()
        pred_token = tok.decode([pred_id]).strip()
        predictions.append(pred_token)

    return batch_hs, predictions

