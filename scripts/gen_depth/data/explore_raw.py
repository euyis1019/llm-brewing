"""Exploratory statistics over raw files (no outputs written).

Usage: python -I explore_raw.py <raw_root>
"""
import collections
import json
import os
import sys

import pandas as pd


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    R = sys.argv[1]
    # ---------------- olympiad
    ob = pd.read_parquet(os.path.join(
        R, "olympiad/OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"))
    kn = pd.read_parquet(os.path.join(R, "olympiad_knoveleng/data/train-00000-of-00001.parquet"))
    print("olympiad rows", len(ob), "knoveleng rows", len(kn))
    ids_ob, ids_kn = set(ob["id"]), set(kn["id"])
    print("ids only in knoveleng:", sorted(ids_kn - ids_ob), "only in official:",
          sorted(ids_ob - ids_kn), "dup ids official:", len(ob) - len(ids_ob),
          "dup ids kn:", len(kn) - len(ids_kn))
    for i in sorted(ids_kn - ids_ob):
        r = kn[kn["id"] == i].iloc[0]
        print("  missing-in-official id", i, repr(r["question"][:300]), list(r["final_answer"]))
    dups = kn[kn.duplicated("id", keep=False)]
    for _, r in dups.iterrows():
        print("  kn dup", r["id"], repr(r["question"][:120]), list(r["final_answer"]))
    print("modality", collections.Counter(ob["modality"]))
    print("question_type", collections.Counter(ob["question_type"]))
    print("is_multiple_answer", collections.Counter(ob["is_multiple_answer"]))
    print("answer_type", collections.Counter(ob["answer_type"]))
    print("n final_answer", collections.Counter(len(x) for x in ob["final_answer"]))
    print("unit non-null", ob["unit"].notna().sum(), collections.Counter(ob["unit"].dropna()))
    print("error non-null", ob["error"].notna().sum(), collections.Counter(ob["error"].dropna()))
    print("context non-null", ob["context"].notna().sum())
    imgs = [c for c in ob.columns if c.startswith("image_")]
    print("rows with any image", int(ob[imgs].notna().any(axis=1).sum()))
    print("--- multi final_answer examples")
    for _, r in ob[ob["final_answer"].map(len) != 1].head(8).iterrows():
        print("  ", r["id"], r["is_multiple_answer"], r["answer_type"], list(r["final_answer"]),
              r["unit"], r["error"])
    print("--- is_multiple_answer single-entry examples")
    for _, r in ob[ob["is_multiple_answer"].astype(bool)].head(
            12).iterrows():
        print("  ", r["id"], r["answer_type"], list(r["final_answer"]), r["unit"], r["error"])
    print("--- unit examples")
    for _, r in ob[ob["unit"].notna()].iterrows():
        print("  ", r["id"], r["answer_type"], list(r["final_answer"]), repr(r["unit"]),
              r["error"], repr(r["question"][-160:]))
    print("--- error examples")
    for _, r in ob[ob["error"].notna()].head(30).iterrows():
        print("  ", r["id"], r["answer_type"], list(r["final_answer"]), repr(r["error"]))
    print("--- context examples")
    for _, r in ob[ob["context"].notna()].head(3).iterrows():
        print("  ", r["id"], repr(str(r["context"])[:200]), "| Q:", repr(r["question"][:120]))
    print("--- Interval / Tuple / Equation examples")
    for t in ["Interval", "Tuple", "Equation"]:
        sub = ob[ob["answer_type"].astype(str).str.contains(t)]
        print(t, len(sub), [list(x) for x in sub["final_answer"].head(10)])

    # ---------------- amc
    amc = pd.read_parquet(os.path.join(R, "amc/data/train-00000-of-00001.parquet"))
    nonint = amc[amc["answer"].map(lambda a: float(a) != int(float(a)))]
    print("amc rows", len(amc), "answer dtype", amc["answer"].dtype, "non-integer answers",
          list(nonint["answer"]), "ids", sorted(amc["id"])[:5], "...", "unique ids",
          amc["id"].nunique())
    print("amc years", collections.Counter(u.split("/")[-2][:9] for u in amc["url"]))

    # ---------------- gsm8k
    g = pd.read_parquet(os.path.join(R, "gsm8k/main/test-00000-of-00001.parquet"))
    tails = [a.split("####")[-1].strip() for a in g["answer"]]
    n_hash = collections.Counter(a.count("####") for a in g["answer"])
    weird = [t for t in tails if not t.replace(",", "").lstrip("-").isdigit()]
    print("gsm8k rows", len(g), "#### counts", n_hash, "non-integer tails", weird[:20],
          "with comma", sum("," in t for t in tails))

    # ---------------- math500
    m = jl(os.path.join(R, "math500/test.jsonl"))
    print("math500 rows", len(m), "levels", collections.Counter(x["level"] for x in m),
          "subjects", collections.Counter(x["subject"] for x in m))

    # ---------------- minerva
    mv = jl(os.path.join(R, "minerva/test.jsonl"))
    print("minerva rows", len(mv), "keys", list(mv[0].keys()))
    print("minerva answers sample", [x["answer"] for x in mv[:60]])

    # ---------------- aime cross-checks
    a24 = pd.read_parquet(os.path.join(R, "aime24/data/train-00000-of-00001.parquet"))
    a24b = pd.read_parquet(os.path.join(R, "aime24_maxwelljia/aime_2024_problems.parquet"))
    print("aime24 urls", [u.split("/")[-1] for u in a24["url"]][:4], "years",
          collections.Counter(a24["year"]))
    print("aime24 ids", list(a24["id"]))
    print("aime24_mj ids", list(a24b["ID"]))
    a25 = pd.read_parquet(os.path.join(R, "aime25/data/train-00000-of-00001.parquet"))
    a26 = pd.read_parquet(os.path.join(R, "aime26/data/train-00000-of-00001.parquet"))
    print("aime25 idx", list(a25["problem_idx"]), "answers", list(a25["answer"]))
    print("aime26 idx", list(a26["problem_idx"]), "answers", list(a26["answer"]))
    oc = jl(os.path.join(R, "aime25_opencompass/aime2025-I.jsonl")) + jl(
        os.path.join(R, "aime25_opencompass/aime2025-II.jsonl"))
    print("aime25 opencompass answers", [x["answer"] for x in oc])

    # ---------------- gpqa
    gj = pd.read_parquet(os.path.join(R, "gpqa_jingzeshi/data/test-00000-of-00001.parquet"))
    gt = pd.read_parquet(os.path.join(R, "gpqa_talzoomanzoo/data/train-00000-of-00001.parquet"))
    print("gpqa jingzeshi rows", len(gj), "unique Record ID", gj["Record ID"].nunique(),
          "null correct", gj["Correct Answer"].isna().sum(),
          "null inc", [gj[f"Incorrect Answer {k}"].isna().sum() for k in (1, 2, 3)])
    print("gpqa talzoomanzoo rows", len(gt), "Choices null", gt["Choices"].isna().sum())
    print("gpqa domains", collections.Counter(gj["High-level domain"]))


if __name__ == "__main__":
    main()
