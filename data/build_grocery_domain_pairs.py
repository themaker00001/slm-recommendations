"""Builds query-item pairs in the SAME style/domain as the demo catalog and
the queries actually tested in the frontend -- short grocery/retail queries
against short product titles -- rather than ESCI's long-tail e-commerce
search phrases. This is real data to be labeled by the teacher (not
synthetic/auto-labeled), aimed at closing the domain gap between training
data (ESCI) and the demo catalog it's actually evaluated against.

Every catalog item is paired with every suggested query (the full cross
product), so the teacher gets a chance to judge every combination -- most
cross-group pairs are genuinely irrelevant, some are moderately relevant
(e.g. "fruits" query against an orange), and each item's own group query
includes the intentional keyword-overlap traps already built into the
catalog (e.g. "salt" query against "Salted Pretzel Sticks").
"""
import json
from pathlib import Path

CATALOG_PATH = Path("data/samples/demo_catalog.json")
OUT_PATH = Path("data/raw/grocery_domain_pairs.jsonl")


def item_text(item: dict) -> str:
    return f"{item['title']} | Brand: {item['brand']}"


def main():
    catalog = json.loads(CATALOG_PATH.read_text())
    queries = sorted({item["group"] for item in catalog})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT_PATH.open("w") as f:
        for query in queries:
            for item in catalog:
                f.write(json.dumps({
                    "query": query,
                    "product_id": item["item_id"],
                    "item_text": item_text(item),
                    "reference_esci_label": None,
                    "reference_relevance": None,
                }) + "\n")
                n += 1

    print(f"Wrote {n} query-item pairs ({len(queries)} queries x {len(catalog)} items) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
