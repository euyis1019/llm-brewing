"""DiagnosticsPipeline — S3 only.

Runs diagnostics from persisted MethodResult files on disk.
Does not require the model to be online.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brewing.diagnostics.outcome import run_diagnostics_from_disk
from .base import PipelineBase

logger = logging.getLogger(__name__)


class DiagnosticsPipeline(PipelineBase):
    """S3: run diagnostics from disk-persisted results."""

    def run(self, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
        results_summary: dict[str, Any] = {"subsets": {}}

        for subset_name in self.subsets:
            logger.info("=" * 60)
            logger.info("[diagnostics] Processing subset: %s", subset_name)
            logger.info("=" * 60)

            try:
                key = self.make_key(subset_name, "eval")
                diag = run_diagnostics_from_disk(
                    results_dir=self.config.output_root,
                    key=key,
                )
                results_summary["subsets"][subset_name] = {
                    "outcome_distribution": diag.outcome_distribution,
                    "mean_fpcl_normalized": diag.mean_fpcl_normalized,
                    "mean_fjc_normalized": diag.mean_fjc_normalized,
                    "mean_delta_brew": diag.mean_delta_brew,
                    "n_samples": len(diag.sample_diagnostics),
                }
            except Exception as e:
                logger.error("Failed on subset '%s': %s", subset_name, e, exc_info=True)
                results_summary["subsets"][subset_name] = {"error": str(e)}

        # Write aggregated summary JSON for easy downstream consumption
        self._write_summary(results_summary)

        return results_summary

    def _write_summary(self, results_summary: dict[str, Any]) -> None:
        """Write a flat summary JSON with one row per subset."""
        output_root = Path(self.config.output_root)
        model_id_safe = self.config.model_id.replace("/", "__")
        summary_path = (
            output_root / "diagnostics_summary" / f"{model_id_safe}.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for subset_name, data in results_summary.get("subsets", {}).items():
            if "error" in data:
                continue
            rows.append({
                "model_id": self.config.model_id,
                "task": subset_name,
                **data,
            })

        summary_path.write_text(json.dumps(rows, indent=2, default=str))
        logger.info("Diagnostics summary written to %s", summary_path)
