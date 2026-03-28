"""Inspect all cached hidden states on disk.

Reports: model, dataset, shape, dtype, sample count, predictions count,
file size, and modification time. Flags any anomalies (mismatched counts,
missing meta, unexpected shapes).

Usage:
    python scripts/inspect_caches.py [--output-root brewing_output]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def inspect_cache_dir(output_root: Path):
    cache_root = output_root / "caches"
    if not cache_root.exists():
        print(f"No caches directory at {cache_root}")
        return

    # Collect all .npz files
    npz_files = sorted(cache_root.rglob("*.npz"))
    meta_files = {p.stem.replace(".npz", ""): p for p in cache_root.rglob("*.meta.json")}

    if not npz_files:
        print("No .npz cache files found.")
        return

    # Group by model
    by_model: dict[str, list[Path]] = {}
    for f in npz_files:
        model_dir = f.parent.name
        by_model.setdefault(model_dir, []).append(f)

    expected_subsets = [
        "value_tracking", "computing", "conditional",
        "function_call", "loop", "loop_unrolled",
    ]
    expected_datasets = []
    for split in ["eval", "train"]:
        for subset in expected_subsets:
            expected_datasets.append(f"cue-{subset}-{split}-seed42")

    total_size = 0
    issues = []

    for model_id in sorted(by_model):
        files = by_model[model_id]
        print(f"\n{'='*70}")
        print(f"  Model: {model_id}")
        print(f"  Files: {len(files)} .npz")
        print(f"{'='*70}")

        dataset_ids_found = set()

        for f in sorted(files, key=lambda p: p.name):
            dataset_id = f.stem  # e.g. cue-loop-eval-seed42
            dataset_ids_found.add(dataset_id)

            fsize = f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            total_size += fsize

            # Load and inspect
            try:
                data = np.load(f, allow_pickle=True)
                keys = list(data.keys())

                hs = data.get("hidden_states")
                preds = data.get("model_predictions")

                hs_shape = tuple(hs.shape) if hs is not None else None
                hs_dtype = str(hs.dtype) if hs is not None else None
                n_preds = len(preds) if preds is not None else 0

                # Check consistency
                flag = ""
                if hs is not None and preds is not None:
                    if hs.shape[0] != len(preds):
                        flag = " *** MISMATCH: hs samples != preds count"
                        issues.append(f"{model_id}/{dataset_id}: {flag}")

                # Check for NaN/Inf
                if hs is not None:
                    nan_count = int(np.isnan(hs).sum())
                    inf_count = int(np.isinf(hs).sum())
                    if nan_count > 0 or inf_count > 0:
                        flag += f" *** NaN={nan_count} Inf={inf_count}"
                        issues.append(f"{model_id}/{dataset_id}: NaN={nan_count} Inf={inf_count}")

                print(f"  {dataset_id}")
                print(f"    shape={hs_shape}  dtype={hs_dtype}  preds={n_preds}  "
                      f"size={sizeof_fmt(fsize)}  mtime={mtime}{flag}")

                # Check meta file
                meta_key = dataset_id
                meta_path = f.parent / f"{dataset_id}.npz.meta.json"
                if meta_path.exists():
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    meta_model = meta.get("model_id", "?")
                    if meta_model.replace("/", "__") != model_id and meta_model != model_id.replace("__", "/"):
                        flag = f" *** META model_id mismatch: {meta_model}"
                        issues.append(f"{model_id}/{dataset_id}: {flag}")
                        print(f"    meta: model_id={meta_model}{flag}")
                else:
                    print(f"    meta: MISSING")
                    issues.append(f"{model_id}/{dataset_id}: missing .meta.json")

                data.close()
            except Exception as e:
                print(f"  {dataset_id}")
                print(f"    *** ERROR loading: {e}")
                issues.append(f"{model_id}/{dataset_id}: load error: {e}")

        # Check completeness
        missing = set(expected_datasets) - dataset_ids_found
        extra = dataset_ids_found - set(expected_datasets)
        if missing:
            print(f"\n  MISSING datasets ({len(missing)}):")
            for m in sorted(missing):
                print(f"    - {m}")
            issues.append(f"{model_id}: missing {len(missing)} datasets")
        if extra:
            print(f"\n  EXTRA datasets ({len(extra)}):")
            for e in sorted(extra):
                print(f"    + {e}")

    # Also check datasets dir
    ds_root = output_root / "datasets"
    if ds_root.exists():
        ds_files = sorted(ds_root.rglob("*.json"))
        print(f"\n{'='*70}")
        print(f"  Datasets directory: {len(ds_files)} files")
        print(f"{'='*70}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Models:      {len(by_model)}")
    print(f"  Total files: {len(npz_files)} .npz")
    print(f"  Total size:  {sizeof_fmt(total_size)}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    ! {iss}")
    else:
        print(f"\n  No issues found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="brewing_output")
    args = parser.parse_args()
    inspect_cache_dir(Path(args.output_root))
