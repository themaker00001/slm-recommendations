"""Samples the query-item pairs that data/build_seed_labels.py's hand-written
judgments are keyed to, from the ESCI sample and the demo-catalog domain
pairs -- run this first (and don't change the seed) if you want to
regenerate data/cache/seed_candidates.jsonl before re-checking
build_seed_labels.py's LABELS list against it. The committed
data/seed_labels_claude.jsonl already carries the full query/item text
alongside each judgment, so it doesn't depend on this file surviving.
"""
import json
import random
from pathlib import Path

ESCI_PATH = "data/raw/esci_sample.jsonl"
GROCERY_PATH = "data/raw/grocery_domain_pairs.jsonl"
CATALOG_PATH = "data/samples/demo_catalog.json"
OUT_PATH = "data/cache/seed_candidates.jsonl"


def main():
    esci = [json.loads(l) for l in open(ESCI_PATH)]
    grocery = [json.loads(l) for l in open(GROCERY_PATH)]
    catalog = {item["item_id"]: item["group"] for item in json.load(open(CATALOG_PATH))}

    rng = random.Random(42)

    # Stratify roughly by ESCI's own reference label so the sample spans the
    # relevance spectrum, not just whatever the stream happened to serve first.
    by_ref = {}
    for r in esci:
        by_ref.setdefault(r.get("reference_relevance"), []).append(r)
    esci_sample = []
    for _, rows in by_ref.items():
        esci_sample.extend(rng.sample(rows, min(7, len(rows))))
    rng.shuffle(esci_sample)
    esci_sample = esci_sample[:28]

    # From the catalog domain pairs: prioritize same-group pairs (the
    # interesting/ambiguous ones, including the deliberate keyword traps),
    # plus a modest sample of cross-group pairs for contrast.
    same_group = [r for r in grocery if r["query"] == catalog[r["product_id"]]]
    cross_group = [r for r in grocery if r["query"] != catalog[r["product_id"]]]
    grocery_sample = same_group + rng.sample(cross_group, 12)
    rng.shuffle(grocery_sample)
    grocery_sample = grocery_sample[:27]

    combined = (
        [{"source": "esci", "query": r["query"], "product_id": r["product_id"],
          "item_text": r["item_text"]} for r in esci_sample]
        + [{"source": "grocery", "query": r["query"], "product_id": r["product_id"],
            "item_text": r["item_text"]} for r in grocery_sample]
    )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    print(f"{len(combined)} candidates written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
