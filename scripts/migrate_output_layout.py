#!/usr/bin/env python3
"""Migrate old flat Brewing output layout to new hierarchical layout.

Old layout:
  datasets/{dataset_id}/manifest.json + samples.json
  caches/{model_safe}/{dataset_id}.npz + .npz.meta.json

New layout:
  datasets/cuebench/{split}/{task}/seed{seed}/manifest.json + samples.json
  caches/cuebench/{split}/{task}/seed{seed}/{model_safe}/hidden_states.npz + meta.json

Usage:
  python scripts/migrate_output_layout.py --output-root brewing_output
  python scripts/migrate_output_layout.py --output-root brewing_output --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


DATASET_ID_RE = re.compile(r"^cue-(.+)-(eval|train)-seed(\d+)$")

# Directories to skip
SKIP_DIRS = {"debug_copying", "smoke_test"}


def parse_dataset_id(dataset_id: str) -> tuple[str, str, int] | None:
    """Parse a dataset_id string into (task, split, seed).

    Returns None if the string doesn't match the expected pattern.
    """
    m = DATASET_ID_RE.match(dataset_id)
    if m is None:
        return None
    task, split, seed_str = m.groups()
    return task, split, int(seed_str)


def migrate_datasets(root: Path, dry_run: bool = False) -> int:
    """Migrate datasets/ from flat to hierarchical layout."""
    datasets_dir = root / "datasets"
    if not datasets_dir.exists():
        print("  No datasets/ directory found, skipping.")
        return 0

    count = 0
    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS:
            print(f"  SKIP: datasets/{entry.name} (in skip list)")
            continue

        parsed = parse_dataset_id(entry.name)
        if parsed is None:
            print(f"  SKIP: datasets/{entry.name} (does not match pattern)")
            continue

        task, split, seed = parsed
        new_dir = datasets_dir / "cuebench" / split / task / f"seed{seed}"

        if new_dir.exists():
            print(f"  SKIP: datasets/{entry.name} -> {new_dir.relative_to(root)} (already exists)")
            continue

        print(f"  MOVE: datasets/{entry.name}/ -> {new_dir.relative_to(root)}/")
        if not dry_run:
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(new_dir))
        count += 1

    return count


def migrate_caches(root: Path, dry_run: bool = False) -> int:
    """Migrate caches/ from flat to hierarchical layout."""
    caches_dir = root / "caches"
    if not caches_dir.exists():
        print("  No caches/ directory found, skipping.")
        return 0

    count = 0
    for model_dir in sorted(caches_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in SKIP_DIRS:
            print(f"  SKIP: caches/{model_dir.name} (in skip list)")
            continue
        # Check if already migrated (cuebench dir exists)
        if model_dir.name == "cuebench":
            print(f"  SKIP: caches/cuebench (already new layout)")
            continue

        model_safe = model_dir.name

        for npz_file in sorted(model_dir.glob("*.npz")):
            dataset_id = npz_file.stem  # e.g. "cue-computing-eval-seed42"
            parsed = parse_dataset_id(dataset_id)
            if parsed is None:
                print(f"  SKIP: caches/{model_safe}/{npz_file.name} (does not match pattern)")
                continue

            task, split, seed = parsed
            new_dir = caches_dir / "cuebench" / split / task / f"seed{seed}" / model_safe

            if (new_dir / "hidden_states.npz").exists():
                print(f"  SKIP: caches/{model_safe}/{dataset_id}.npz -> ... (already exists)")
                continue

            # Move .npz -> hidden_states.npz
            new_npz = new_dir / "hidden_states.npz"
            print(f"  MOVE: caches/{model_safe}/{dataset_id}.npz -> {new_npz.relative_to(root)}")

            # Move .npz.meta.json -> meta.json
            old_meta = Path(str(npz_file) + ".meta.json")
            new_meta = new_dir / "meta.json"

            if not dry_run:
                new_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(npz_file), str(new_npz))
                if old_meta.exists():
                    shutil.move(str(old_meta), str(new_meta))
                    print(f"  MOVE: caches/{model_safe}/{dataset_id}.npz.meta.json -> {new_meta.relative_to(root)}")
            else:
                if old_meta.exists():
                    print(f"  MOVE: caches/{model_safe}/{dataset_id}.npz.meta.json -> {new_meta.relative_to(root)}")

            count += 1

    return count


def cleanup_empty_dirs(root: Path, dry_run: bool = False) -> int:
    """Remove empty directories under root."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        if not any(p.iterdir()):
            print(f"  RMDIR: {p.relative_to(root)}/")
            if not dry_run:
                p.rmdir()
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Brewing output from flat to hierarchical layout"
    )
    parser.add_argument(
        "--output-root",
        default="brewing_output",
        help="Root output directory (default: brewing_output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    if not root.exists():
        print(f"Output root {root} does not exist.")
        return

    if args.dry_run:
        print("=== DRY RUN (no files will be moved) ===\n")

    print("--- Migrating datasets/ ---")
    ds_count = migrate_datasets(root, dry_run=args.dry_run)

    print("\n--- Migrating caches/ ---")
    cache_count = migrate_caches(root, dry_run=args.dry_run)

    print("\n--- Cleaning up empty directories ---")
    dir_count = cleanup_empty_dirs(root, dry_run=args.dry_run)

    print(f"\nDone. Moved {ds_count} datasets, {cache_count} cache files, removed {dir_count} empty dirs.")


if __name__ == "__main__":
    main()
