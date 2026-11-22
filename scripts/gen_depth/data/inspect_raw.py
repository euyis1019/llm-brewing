"""Print schema + a few example rows of raw data files (inspection only).

Usage: python -I inspect_raw.py <file> [n_rows] [max_chars]
"""
import json
import sys


def load(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    raise ValueError(path)


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rows = load(path)
    print("file:", path, "rows:", len(rows))
    if rows:
        print("keys:", list(rows[0].keys()))
    for r in rows[:n]:
        print("-" * 60)
        for k, v in r.items():
            s = repr(v)
            print(f"  {k}: {s[:mx]}{'...' if len(s) > mx else ''}")


if __name__ == "__main__":
    main()
