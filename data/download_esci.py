"""Pull a manageable query-item sample from the public Amazon ESCI dataset.

ESCI (github.com/amazon-science/esci-data) stands in for DoorDash's proprietary
query-item ad corpus: it has real search queries paired with real product
listings and a human-graded relevance label per pair, at a similar spirit
(if not scale) to what the blog post describes seeding its LLM teacher with.

We deliberately do NOT use ESCI's own esci_label as the training target --
that would skip the teacher step. It's kept in the output only as a reference
column so you can later sanity-check the Claude teacher's labels against
human judgments.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ESCI_TO_REFERENCE = {
    "Exact": 2,
    "Substitute": 1,
    "Complement": 1,
    "Irrelevant": 0,
}


def build_item_text(row: dict) -> str:
    parts = [row.get("product_title") or ""]
    if row.get("product_brand"):
        parts.append(f"Brand: {row['product_brand']}")
    if row.get("product_bullet_point"):
        parts.append(row["product_bullet_point"].split("\n")[0])
    return " | ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locale", default="us")
    ap.add_argument("--limit", type=int, default=1500,
                     help="number of query-item pairs to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/esci_sample.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset("tasksource/esci", split="train", streaming=True)

    by_label = {"Exact": [], "Substitute": [], "Complement": [], "Irrelevant": []}
    target_per_label = args.limit // 4

    for row in ds:
        if row["product_locale"] != args.locale:
            continue
        label = row["esci_label"]
        if label not in by_label or len(by_label[label]) >= target_per_label:
            if all(len(v) >= target_per_label for v in by_label.values()):
                break
            continue
        by_label[label].append(row)

    pairs = [r for rows in by_label.values() for r in rows]
    random.shuffle(pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps({
                "query": row["query"].strip(),
                "product_id": row["product_id"],
                "item_text": build_item_text(row),
                "reference_esci_label": row["esci_label"],
                "reference_relevance": ESCI_TO_REFERENCE[row["esci_label"]],
            }) + "\n")

    print(f"Wrote {len(pairs)} query-item pairs to {out_path}")
    for label, rows in by_label.items():
        print(f"  {label}: {len(rows)}")


if __name__ == "__main__":
    main()
