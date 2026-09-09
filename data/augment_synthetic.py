"""Synthetic data augmentation built entirely from the raw data we already
have (data/cache/teacher_labels.jsonl) -- no new Ollama/teacher calls needed.

The 800 teacher-labeled pairs are skewed: ~57% label 0 (irrelevant), ~33%
label 2 (highly relevant), only ~9% label 1 (moderately relevant). Random
cross-pairing (query from one row + item from another) is the usual cheap way
to synthesize more training data, but it can only ever manufacture MORE label-0
examples -- a random query/item pair is almost always irrelevant -- which
would make the already-dominant class even more dominant. Since the observed
failure mode is the model *over-predicting* irrelevant on real matches, that
would make things worse, not better.

Instead this generates two kinds of synthetic rows:

1. Label-preserving perturbations of the minority classes (1 and 2): querying
   with fewer words, or scoring against just the item's core title instead of
   the full title+brand+bullet text. The relevance judgment should hold under
   these changes, so the original label is reused as-is.
2. A modest, controlled batch of new label-0 negatives via true random
   cross-pairing, filtered to drop any pair that accidentally shares a word
   with the query (to avoid quietly mislabeling a real match as irrelevant).
"""
import argparse
import json
import random
import re
from pathlib import Path


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def truncate_query(query: str) -> str | None:
    words = query.split()
    if len(words) < 2:
        return None
    return " ".join(words[: max(1, len(words) // 2)])


def truncate_item(item_text: str) -> str | None:
    title = item_text.split("|")[0].strip()
    if not title or title == item_text.strip():
        return None
    return title


def perturb_minority_classes(rows: list[dict]) -> list[dict]:
    synthetic = []
    for row in rows:
        if row["teacher_label"] not in (1, 2):
            continue
        short_q = truncate_query(row["query"])
        if short_q:
            synthetic.append({
                **row, "query": short_q,
                "teacher_reason": row["teacher_reason"] + " [synthetic: shortened query]",
                "synthetic": True,
            })
        short_item = truncate_item(row["item_text"])
        if short_item:
            synthetic.append({
                **row, "item_text": short_item,
                "teacher_reason": row["teacher_reason"] + " [synthetic: title-only item]",
                "synthetic": True,
            })
    return synthetic


def random_negatives(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    real_pairs = {(r["query"], r["product_id"]) for r in rows}
    synthetic = []
    attempts = 0
    while len(synthetic) < count and attempts < count * 20:
        attempts += 1
        q_row = rng.choice(rows)
        i_row = rng.choice(rows)
        if q_row["query"] == i_row["query"]:
            continue
        if (q_row["query"], i_row["product_id"]) in real_pairs:
            continue
        if tokenize(q_row["query"]) & tokenize(i_row["item_text"]):
            continue  # shares a word -- not a safe auto-negative, skip
        synthetic.append({
            "query": q_row["query"],
            "product_id": i_row["product_id"],
            "item_text": i_row["item_text"],
            "teacher_label": 0,
            "teacher_reason": "synthetic: random cross-pair, no token overlap with query",
            "reference_relevance": None,
            "synthetic": True,
        })
    return synthetic


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/cache/teacher_labels.jsonl")
    ap.add_argument("--out", default="data/cache/teacher_labels_augmented.jsonl")
    ap.add_argument("--num-negatives", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.inp).open()]
    minority_synthetic = perturb_minority_classes(rows)
    negative_synthetic = random_negatives(rows, args.num_negatives, args.seed)

    combined = rows + minority_synthetic + negative_synthetic
    random.Random(args.seed).shuffle(combined)

    with Path(args.out).open("w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    import collections
    counts = collections.Counter(r["teacher_label"] for r in combined)
    total = len(combined)
    print(f"Original: {len(rows)} rows")
    print(f"+ {len(minority_synthetic)} label-preserving perturbations (classes 1/2)")
    print(f"+ {len(negative_synthetic)} controlled random negatives (class 0)")
    print(f"= {total} total rows -> {args.out}")
    for label in sorted(counts):
        print(f"  label {label}: {counts[label]:4d}  ({100 * counts[label] / total:.1f}%)")


if __name__ == "__main__":
    main()
