"""Download raw evaluation data files from the Hugging Face Hub.

Only plain data files (parquet / json / jsonl; plus the official GPQA csv, which is
only attempted because the repo is the canonical source) are fetched with
hf_hub_download.  No dataset loading scripts, no trust_remote_code.

Each source goes into its own directory <raw_root>/<name>/.  A manifest with repo ids,
pinned commit hashes, file sizes and sha256 digests (or the failure reason) is written
to --manifest (keep it outside the raw dirs).

Usage (run with an isolated interpreter, proxy without SOCKS):
  env -u all_proxy -u ALL_PROXY python -I download_raw.py \
      --raw-root GD/data/raw --manifest GD/data/raw_manifest.json [--only a,b]
"""
import argparse
import hashlib
import json
import os
import sys
import time

# name -> (repo_id, [files]) ; files are relative paths inside the dataset repo.
SOURCES = {
    "gsm8k": ("openai/gsm8k", ["main/test-00000-of-00001.parquet"]),
    "math500": ("HuggingFaceH4/MATH-500", ["test.jsonl"]),
    "aime24": ("HuggingFaceH4/aime_2024", ["data/train-00000-of-00001.parquet"]),
    # second copy of AIME 2024, only used to cross-check answers
    "aime24_maxwelljia": ("Maxwell-Jia/AIME_2024", ["aime_2024_problems.parquet"]),
    "aime25": ("MathArena/aime_2025", ["data/train-00000-of-00001.parquet"]),
    # second copy of AIME 2025, only used to cross-check answers
    "aime25_opencompass": ("opencompass/AIME2025", ["aime2025-I.jsonl", "aime2025-II.jsonl"]),
    "aime26": ("MathArena/aime_2026", ["data/train-00000-of-00001.parquet"]),
    "amc": ("AI-MO/aimo-validation-amc", ["data/train-00000-of-00001.parquet"]),
    "olympiad": ("Hothan/OlympiadBench",
                 ["OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"]),
    # 675-row copy (Qwen2.5-Math eval version); only used to diff against the official 674
    "olympiad_knoveleng": ("knoveleng/OlympiadBench", ["data/train-00000-of-00001.parquet"]),
    "minerva": ("math-ai/minervamath", ["test.jsonl"]),
    # official GPQA (gated); attempted first
    "gpqa_official": ("Idavidrein/gpqa", ["gpqa_diamond.csv"]),
    # public parquet mirror with the original GPQA columns (Question, Correct Answer,
    # Incorrect Answer 1-3, Record ID, ...)
    "gpqa_jingzeshi": ("JingzeShi/gpqa_diamond", ["data/test-00000-of-00001.parquet"]),
    # independent second mirror (Question / Choices / Correct Choice), cross-check only
    "gpqa_talzoomanzoo": ("talzoomanzoo/gpqa_diamond_raw", ["data/train-00000-of-00001.parquet"]),
}

ALLOWED_EXT = (".parquet", ".json", ".jsonl", ".csv")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--only", default="", help="comma-separated subset of source names")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    names = [n for n in SOURCES if not args.only or n in args.only.split(",")]
    manifest = {}
    if os.path.exists(args.manifest):
        with open(args.manifest) as f:
            manifest = json.load(f)

    for name in names:
        repo_id, files = SOURCES[name]
        entry = {"repo_id": repo_id, "files": {}, "status": "ok",
                 "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        out_dir = os.path.join(args.raw_root, name)
        try:
            info = api.dataset_info(repo_id)
            entry["revision"] = info.sha
            entry["gated"] = getattr(info, "gated", None)
            entry["last_modified"] = str(getattr(info, "last_modified", None))
            os.makedirs(out_dir, exist_ok=True)
            for fn in files:
                assert fn.endswith(ALLOWED_EXT), fn
                local = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset",
                                        revision=info.sha, local_dir=out_dir)
                entry["files"][fn] = {"local_path": os.path.relpath(local, args.raw_root),
                                      "bytes": os.path.getsize(local),
                                      "sha256": sha256_of(local)}
        except Exception as e:  # record and continue
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            if os.path.isdir(out_dir) and not any(
                    x for x in os.listdir(out_dir) if not x.startswith(".")):
                pass  # leave empty dir; it documents the attempt
        manifest[name] = entry
        print(name, repo_id, entry.get("revision"), entry["status"], entry.get("error", ""),
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
