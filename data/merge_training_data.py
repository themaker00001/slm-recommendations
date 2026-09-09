"""Merges the ESCI-labeled data with the domain-matched grocery-catalog data
into one training file, oversampling the (much smaller) domain-matched pool
so it isn't diluted into irrelevance by the larger ESCI pool.

Why oversampling is needed at all: training on ESCI's long-tail e-commerce
search phrases alone produces a model that behaves inconsistently on the
demo catalog's short, generic queries -- it isn't that ESCI data is "wrong",
it's a different query/item style than what the frontend actually tests
against. Mixing in a small amount of real, teacher-labeled, demo-catalog-style
data (data/build_grocery_domain_pairs.py + teacher/label_with_ollama.py)
closes that style gap, but only if that data isn't drowned out: 71 real
examples inside ~1000 total rows barely moves the gradient. Repeating them
--repeat times gives the domain-matched signal proportionally more weight
during training without needing more teacher labels.

Why the split has to happen BEFORE oversampling, not after: if you duplicate
a row 8 times and only then hand the whole file to a random 80/20 split,
some copies of the same example land in "train" and others in "val" purely by
chance -- so the model can end up validated on an example it was also trained
on, quietly inflating validation accuracy. Splitting the *unique* examples
first, then oversampling only the training side, avoids that. Rows are
written with an explicit "split" field; model/train.py uses it directly
instead of re-splitting randomly when every row in the file carries one.
"""
import argparse
import collections
import json
import random
from pathlib import Path


def split(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * frac)) if len(rows) > 4 else 0
    return rows[n_val:], rows[:n_val]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esci-labels", default="data/cache/teacher_labels.jsonl")
    ap.add_argument("--grocery-labels", default="data/cache/grocery_domain_labels.jsonl")
    ap.add_argument("--catalog", default="data/samples/demo_catalog.json")
    ap.add_argument("--out", default="data/cache/teacher_labels_merged.jsonl")
    ap.add_argument("--repeat", type=int, default=8,
                     help="how many times to repeat each domain-matched training example")
    ap.add_argument("--num-cross-negatives", type=int, default=100,
                     help="how many of the (very many) trivial cross-category negatives to keep")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    catalog = {item["item_id"]: item["group"] for item in json.loads(Path(args.catalog).read_text())}
    grocery_rows = [json.loads(line) for line in Path(args.grocery_labels).open()]
    esci_rows = [json.loads(line) for line in Path(args.esci_labels).open()]

    same_group = [r for r in grocery_rows if r["query"] == catalog[r["product_id"]]]
    cross_group = [r for r in grocery_rows if r["query"] != catalog[r["product_id"]]]
    cross_nonzero = [r for r in cross_group if r["teacher_label"] != 0]
    cross_zero = [r for r in cross_group if r["teacher_label"] == 0]

    domain_matched = same_group + cross_nonzero
    rng = random.Random(args.seed)
    cross_zero_sample = rng.sample(cross_zero, min(args.num_cross_negatives, len(cross_zero)))

    esci_train, esci_val = split(esci_rows, args.val_fraction, seed=args.seed + 1)
    domain_train, domain_val = split(domain_matched, args.val_fraction, seed=args.seed + 2)
    negs_train, negs_val = split(cross_zero_sample, args.val_fraction, seed=args.seed + 3)

    train_rows = esci_train + domain_train * args.repeat + negs_train
    val_rows = esci_val + domain_val + negs_val  # val stays unique, never duplicated

    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"

    combined = train_rows + val_rows
    rng.shuffle(combined)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    print(f"domain-matched: {len(domain_matched)} unique -> "
          f"train {len(domain_train)} (x{args.repeat} = {len(domain_train) * args.repeat}), "
          f"val {len(domain_val)} (kept unique, not oversampled)")
    print(f"esci: train {len(esci_train)}, val {len(esci_val)}")
    print(f"cross-category negatives: train {len(negs_train)}, val {len(negs_val)}")
    print(f"TOTAL: train {len(train_rows)}, val {len(val_rows)} -> {out_path}")
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        c = collections.Counter(r["teacher_label"] for r in rows)
        print(f"  {name} label distribution:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
