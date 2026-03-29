"""CLI entry point for the Brewing framework.

Responsible for: parsing CLI arguments, loading YAML config, constructing
RunConfig, optionally loading a model, and delegating to Orchestrator.run().

NOT responsible for: any analysis logic or result interpretation.

Usage:
    python -m brewing --config path/to/config.yaml           # standard run
    python -m brewing --config path/to/config.yaml --verbose  # debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from brewing.schema import RunConfig
from brewing.registry import get_benchmark, get_method_class


def load_config(config_path: str | Path) -> RunConfig:
    """Read a YAML config file and return a validated RunConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"YAML config must be a mapping, got {type(config_dict).__name__}"
        )
    return RunConfig(**config_dict)


def _all_caches_exist(config: RunConfig) -> bool:
    """Check whether all required hidden-state caches already exist on disk."""
    from brewing.resources import ResourceManager, ResourceKey

    rm = ResourceManager(config.output_root)
    benchmark_safe = config.benchmark_path_safe
    subsets = config.subsets

    if subsets is None:
        benchmark = get_benchmark(config.benchmark)
        subsets = benchmark.subset_names

    if config.splits:
        splits = config.splits
    elif config.mode == "train_probing":
        splits = ["train"]
        lp_config = config.method_configs.get("linear_probing", {})
        if lp_config.get("validate_on_eval", False):
            splits.append("eval")
    else:
        splits = ["eval"]

    for split in splits:
        for subset in subsets:
            key = ResourceKey(
                benchmark=benchmark_safe,
                split=split,
                task=subset,
                seed=config.seed,
                model_id=config.model_id,
            )
            if not rm.cache_path(key).exists():
                return False
    return True


def needs_model_online(config: RunConfig) -> bool:
    """Check whether the configured mode requires the model to be loaded."""
    if config.mode in ("cache_only", "train_probing"):
        if not _all_caches_exist(config):
            return True
        return False
    if config.mode == "diagnostics":
        return False
    if config.mode == "causal_validation":
        return True  # always needs model for interventions
    # mode == "eval": check methods
    for method_name in config.methods:
        method_cls = get_method_class(method_name)
        if method_cls().requirements().needs_model_online:
            return True
    return False


def build_model_load_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build kwargs for nnsight LanguageModel / HF from_pretrained."""
    import torch

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": torch.float16,
    }

    if config.model_cache_dir is not None:
        load_kwargs["cache_dir"] = config.model_cache_dir

    if config.quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config.quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    return load_kwargs


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Brewing: Layer-wise Mechanistic Interpretability Framework"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure benchmarks and methods are registered, Eric: Lazy load can be done?
    import brewing.benchmarks  # noqa: F401
    import brewing.methods.linear_probing  # noqa: F401
    import brewing.methods.csd  # noqa: F401

    # Load config
    config = load_config(args.config)

    # Load model if needed
    model = None
    tokenizer = None

    if needs_model_online(config):  # Depends on mode
        logging.info("Loading model: %s", config.model_id)
        try:
            load_kwargs = build_model_load_kwargs(config)
            # cache_only mode: plain HF model (no nnsight trace overhead)
            # other modes: nnsight LanguageModel (needed for interventions)
            if config.mode == "cache_only":
                from transformers import AutoModelForCausalLM, AutoTokenizer

                hf_kwargs = {k: v for k, v in load_kwargs.items()
                             if k != "dtype"}
                hf_kwargs["torch_dtype"] = load_kwargs.get("dtype")
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id, **hf_kwargs,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_id,
                    cache_dir=load_kwargs.get("cache_dir"),
                )
                logging.info("Model loaded as HF AutoModelForCausalLM (cache_only)")
            else:
                from nnsight import LanguageModel

                model = LanguageModel(config.model_id, **load_kwargs)
                tokenizer = model.tokenizer
                logging.info("Model loaded as nnsight LanguageModel")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{config.model_id}': {e}"
            ) from e

    # Run
    from brewing.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    results = orchestrator.run(model=model, tokenizer=tokenizer)

    # Print summary
    print("\n" + "=" * 60)
    print("Brewing Run Complete")
    print("=" * 60)
    for subset_name, subset_result in results.get("subsets", {}).items():
        print(f"\n  {subset_name}:")
        if isinstance(subset_result, dict):
            for k, v in subset_result.items():
                if k == "diagnostics" and isinstance(v, dict):
                    print(f"    {k}:")
                    for dk, dv in v.items():
                        print(f"      {dk}: {dv}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    {subset_result}")

    print(f"\nOutput: {config.output_root}/")


if __name__ == "__main__":
    main()
