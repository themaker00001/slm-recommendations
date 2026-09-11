"""Second batch of seed-set candidates, used to scale data/seed_labels_claude.jsonl
from 48 to 155 examples -- see data/sample_seed_candidates.py for the first
batch. Excludes every (query, product_id) pair already present in
data/seed_labels_claude.jsonl at run time, so re-running this after adding
more batches would correctly sample a further-new batch, not literally
reproduce this one -- data/seed_labels_claude.jsonl itself is the
reproducible source of truth for what's actually in the seed set.
"""
import json
import random
from pathlib import Path

ESCI_PATH = "data/raw/esci_sample.jsonl"
GROCERY_PATH = "data/raw/grocery_domain_pairs.jsonl"
CATALOG_PATH = "data/samples/demo_catalog.json"
EXISTING_PATH = "data/seed_labels_claude.jsonl"
OUT_PATH = "data/cache/seed_candidates_batch2.jsonl"


def main():
    esci = [json.loads(l) for l in open(ESCI_PATH)]
    grocery = [json.loads(l) for l in open(GROCERY_PATH)]
    catalog = {item["item_id"]: item["group"] for item in json.load(open(CATALOG_PATH))}

    already_used = set()
    for line in open(EXISTING_PATH):
        row = json.loads(line)
        already_used.add((row["query"], row["product_id"]))

    rng = random.Random(7)

    by_ref = {}
    for r in esci:
        key = (r["query"], r["product_id"])
        if key in already_used:
            continue
        by_ref.setdefault(r.get("reference_relevance"), []).append(r)
    esci_new = []
    for _, rows in by_ref.items():
        esci_new.extend(rng.sample(rows, min(18, len(rows))))
    rng.shuffle(esci_new)
    esci_new = esci_new[:55]

    same_group = [r for r in grocery if r["query"] == catalog[r["product_id"]]
                  and (r["query"], r["product_id"]) not in already_used]
    cross_group = [r for r in grocery if r["query"] != catalog[r["product_id"]]
                   and (r["query"], r["product_id"]) not in already_used]
    grocery_new = same_group + rng.sample(cross_group, min(20, len(cross_group)))
    rng.shuffle(grocery_new)
    grocery_new = grocery_new[:55]

    combined = (
        [{"source": "esci", "query": r["query"], "product_id": r["product_id"],
          "item_text": r["item_text"]} for r in esci_new]
        + [{"source": "grocery", "query": r["query"], "product_id": r["product_id"],
            "item_text": r["item_text"]} for r in grocery_new]
    )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    print(f"{len(combined)} new candidates written -> {OUT_PATH} "
          f"({len(esci_new)} esci, {len(grocery_new)} grocery)")


if __name__ == "__main__":
    main()
