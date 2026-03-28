"""EvalPipeline — S0 -> S1 -> S2 (method execution).

This is the default pipeline mode, equivalent to the original
Orchestrator._run_subset logic.
"""

from __future__ import annotations

import logging
from typing import Any

from brewing.schema import (
    MethodConfig,
    MethodResult,
)
from brewing.registry import get_method_class
from .base import PipelineBase

logger = logging.getLogger(__name__)


class EvalPipeline(PipelineBase):
    """S0 -> S1 -> S2: dataset resolve, cache build, method eval."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[eval] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                subset_result = self._run_subset(subset_name, model, tokenizer)
                results_summary["subsets"][subset_name] = subset_result
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        return results_summary

    def _run_subset(
        self,
        subset_name: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        # S0
        eval_key = self.make_key(subset_name, "eval")
        manifest, eval_samples = self.resolve_dataset(subset_name, eval_key)
        result["n_eval_samples"] = len(eval_samples)
        logger.info("Eval dataset: %d samples", len(eval_samples))

        # S1
        eval_cache = self.resolve_hidden_cache(eval_key, eval_samples, model, tokenizer)
        result["n_layers"] = eval_cache.n_layers
        result["hidden_dim"] = eval_cache.hidden_dim

        # S2
        for method_name in self.config.methods:
            logger.info("Running method: %s", method_name)
            try:
                result_key = self.make_key(subset_name, "eval", method=method_name)
                mr = self._run_method(
                    method_name, subset_name, eval_key,
                    eval_samples, eval_cache, model, tokenizer,
                )
                self.resources.save_result(result_key, mr)
                result[f"method_{method_name}"] = "ok"
            except Exception as e:
                logger.error("Method '%s' failed: %s", method_name, e, exc_info=True)
                result[f"method_{method_name}"] = f"error: {e}"

        return result

    def _run_method(
        self,
        method_name: str,
        subset_name: str,
        eval_key: Any,
        eval_samples: list,
        eval_cache: Any,
        model: Any,
        tokenizer: Any,
    ) -> MethodResult:
        method_cls = get_method_class(method_name)
        method = method_cls()

        user_config = self.config.method_configs.get(method_name, {})

        mc = MethodConfig(
            method=method_name,
            benchmark=self.config.benchmark,
            config={
                "eval_dataset_id": eval_key.dataset_id,
                "answer_space": self.benchmark.answer_meta.answer_space,
                "resource_key_benchmark": eval_key.benchmark,
                "resource_key_task": eval_key.task,
                "resource_key_seed": eval_key.seed,
                **user_config,
            },
        )

        if method.requirements().trained:
            mc.config.setdefault("fit_policy", self.config.fit_policy)
            mc.config.setdefault("train_resource_key_split", "train")

        return method.run(
            config=mc,
            eval_samples=eval_samples,
            eval_cache=eval_cache,
            resources=self.resources,
            model=model,
        )
