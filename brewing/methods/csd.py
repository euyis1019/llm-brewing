"""Context-Stripped Decoding (CSD) — Information Readiness (Φ_C).

Responsible for: determining at which layer the model can internally decode
the answer from its hidden state alone (without source-code context).

NOT responsible for: hidden state extraction (that's cache_builder),
outcome classification (that's diagnostics/), or model loading.

Model-online, training-free method.
For each sample and each layer:
  1. Use pre-extracted hidden state h^ℓ from source prompt (via HiddenStateCache)
  2. Inject h^ℓ into target prompt (question only, no code context)
  3. Continue forward from layer ℓ+1
  4. Compare output against answer (with baseline subtraction)

Two execution paths:
  - Batch path: uses patchscope_lens (nnsight_ops) — faster, preferred.
  - Per-sample fallback: uses manual HF forward hooks — slower.
    Used when nnsight tracing fails.

Both paths operate in logit space for baseline subtraction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from brewing.schema import (
    Granularity, HiddenStateCache, MethodConfig, MethodRequirements,
    MethodResult, Sample, SampleMethodResult, SingleTokenRequirement,
)
from brewing.methods.base import ModelOnlineMethod
from brewing.registry import register_method
from brewing.resources import ResourceKey, ResourceManager

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PROMPT = '# The value of x is "'


class CSD(ModelOnlineMethod):
    """Context-Stripped Decoding for information readiness."""

    name = "csd"

    def _requirements(self) -> MethodRequirements:
        return MethodRequirements(
            needs_answer_space=False,
            single_token_answer=SingleTokenRequirement.PREFERRED,
            trained=False,
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
        target_prompt = config.config.get("target_prompt", DEFAULT_TARGET_PROMPT)
        answer_space = config.config.get("answer_space", [str(d) for d in range(10)])
        n_layers = eval_cache.n_layers

        if model is None:
            raise ValueError("CSD requires a model to be loaded (model-online method)")

        # Try batch patchscope_lens first (much faster)
        try:
            sample_results = self._run_batch_patchscope(
                model, eval_samples, eval_cache,
                target_prompt, answer_space, n_layers,
            )
        except Exception as e:
            logger.warning("Batch patchscope failed (%s), falling back to per-sample", e)
            sample_results = self._run_per_sample(
                model, eval_samples, eval_cache,
                target_prompt, answer_space, n_layers,
            )

        return MethodResult(
            method=self.name,
            model_id=eval_cache.model_id,
            granularity=Granularity.PER_SAMPLE,
            eval_dataset_id=config.config.get("eval_dataset_id", "unknown"),
            sample_results=sample_results,
        )

    def _run_batch_patchscope(
        self,
        model: Any,
        eval_samples: list[Sample],
        eval_cache: HiddenStateCache,
        target_prompt: str,
        answer_space: list[str],
        n_layers: int,
    ) -> list[SampleMethodResult]:
        """Use patchscope_lens for batch processing.

        patchscope_lens(return_logits=True) returns raw logits for all layers
        in one call per sample. Baseline subtraction in logit space, consistent
        with the per-sample fallback path.
        """
        import torch
        from brewing.nnsight_ops import patchscope_lens, TargetPrompt

        tokenizer = model.tokenizer
        answer_token_ids = self._get_answer_token_ids(tokenizer, answer_space)
        target = TargetPrompt(target_prompt, -1)

        # Baseline in logit space (same as per-sample path)
        baseline_logits = self._get_baseline_logits_nnsight(
            model, target_prompt, answer_token_ids
        )

        sample_results: list[SampleMethodResult] = []

        for i, sample in enumerate(eval_samples):
            # latents for patchscope_lens: shape (num_layers, 1, hidden_dim)
            h_all = eval_cache.hidden_states[i]  # (L, D)
            latents = torch.tensor(h_all, dtype=torch.float32).unsqueeze(1)  # (L, 1, D)

            # patchscope_lens returns (num_source_prompts, num_layers, vocab_size)
            logits_all = patchscope_lens(
                nn_model=model,
                source_prompts=None,
                target_patch_prompts=target,
                layers=list(range(n_layers)),
                latents=latents,
                return_logits=True,
            )
            # logits_all: (1, L, vocab_size)
            logits_np = logits_all[0].cpu().float().numpy()  # (L, vocab_size)

            layer_vals = np.zeros(n_layers)
            layer_preds: list[str] = []
            layer_confs = np.zeros((n_layers, len(answer_space)))

            for layer_idx in range(n_layers):
                answer_logits = np.array([
                    logits_np[layer_idx, tid] for tid in answer_token_ids
                ])
                adjusted = answer_logits - baseline_logits

                pred_idx = int(np.argmax(adjusted))
                pred_label = answer_space[pred_idx]
                correct = float(pred_label == sample.answer)

                exp_adj = np.exp(adjusted - np.max(adjusted))
                norm_probs = exp_adj / exp_adj.sum()

                layer_vals[layer_idx] = correct
                layer_preds.append(pred_label)
                layer_confs[layer_idx] = norm_probs

            sample_results.append(SampleMethodResult(
                sample_id=sample.id,
                layer_values=layer_vals,
                layer_predictions=layer_preds,
                layer_confidences=layer_confs,
            ))

        return sample_results

    def _run_per_sample(
        self,
        model: Any,
        eval_samples: list[Sample],
        eval_cache: HiddenStateCache,
        target_prompt: str,
        answer_space: list[str],
        n_layers: int,
    ) -> list[SampleMethodResult]:
        """Fallback: per-sample, per-layer patchscope using manual hook injection.

        Based on legacy/shared/patchscopes.py pattern.
        """
        import torch

        tokenizer = model.tokenizer
        answer_token_ids = self._get_answer_token_ids(tokenizer, answer_space)

        # Baseline
        baseline_logits = self._get_baseline_logits_manual(
            model, tokenizer, target_prompt, answer_token_ids
        )

        sample_results: list[SampleMethodResult] = []

        for i, sample in enumerate(eval_samples):
            h_all = eval_cache.hidden_states[i]  # (L, D)
            layer_vals = np.zeros(n_layers)
            layer_preds: list[str] = []
            layer_confs = np.zeros((n_layers, len(answer_space)))

            for layer_idx in range(n_layers):
                h = torch.tensor(
                    h_all[layer_idx], dtype=torch.float16
                ).to(next(model.parameters()).device)

                # Hook-based injection (legacy pattern)
                target_inputs = tokenizer(
                    target_prompt, return_tensors="pt"
                ).to(h.device)
                patch_pos = target_inputs.input_ids.shape[1] - 1

                def make_hook(h_src, pos):
                    def hook_fn(module, input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        if hidden.shape[1] > pos:
                            hidden[0, pos, :] = h_src
                        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                    return hook_fn

                hook = model.model.layers[layer_idx].register_forward_hook(
                    make_hook(h, patch_pos)
                )
                try:
                    with torch.no_grad():
                        outputs = model(**target_inputs)
                    logits = outputs.logits[0, -1].cpu().float().numpy()
                finally:
                    hook.remove()

                answer_logits = np.array([logits[tid] for tid in answer_token_ids])
                adjusted = answer_logits - baseline_logits

                pred_idx = int(np.argmax(adjusted))
                pred_label = answer_space[pred_idx]
                correct = float(pred_label == sample.answer)

                exp_adj = np.exp(adjusted - np.max(adjusted))
                norm_probs = exp_adj / exp_adj.sum()

                layer_vals[layer_idx] = correct
                layer_preds.append(pred_label)
                layer_confs[layer_idx] = norm_probs

            sample_results.append(SampleMethodResult(
                sample_id=sample.id,
                layer_values=layer_vals,
                layer_predictions=layer_preds,
                layer_confidences=layer_confs,
            ))

        return sample_results

    def _get_baseline_logits_nnsight(
        self,
        model: Any,
        target_prompt: str,
        answer_token_ids: list[int],
    ) -> np.ndarray:
        """Get baseline answer logits using nnsight tracing.

        Returns raw logits for each answer token when running the target
        prompt without any hidden-state injection, consistent with the
        per-sample fallback path.
        """
        from brewing.nnsight_ops import get_logits as _get_logits
        with model.trace(target_prompt) as tracer:
            logits = _get_logits(model)[:, -1, :].cpu().save()
        logits_np = logits.value[0].cpu().float().numpy()  # (vocab_size,)
        return np.array([logits_np[tid] for tid in answer_token_ids])

    def _get_baseline_logits_manual(
        self,
        model: Any,
        tokenizer: Any,
        target_prompt: str,
        answer_token_ids: list[int],
    ) -> np.ndarray:
        """Get baseline answer logits using direct HF forward pass.

        Returns actual logits (not probs), consistent with the per-sample
        fallback path which also operates in logit space.
        """
        import torch
        inputs = tokenizer(target_prompt, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1].cpu().float().numpy()
        return np.array([logits[tid] for tid in answer_token_ids])

    @staticmethod
    def _get_answer_token_ids(tokenizer: Any, answer_space: list[str]) -> list[int]:
        """Get token IDs for each answer in the answer space."""
        ids = []
        for ans in answer_space:
            tids = tokenizer.encode(ans, add_special_tokens=False)
            ids.append(tids[0] if tids else 0)
        return ids


# Register
register_method("csd", CSD)
