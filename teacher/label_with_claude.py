"""LLM teacher: label query-item pairs 0/1/2 for relevance, offline and at scale.

Mirrors the blog's teacher step: a fine-tuned/prompted LLM produces relevance
supervision in a batch job, which the small bi-encoder student later learns
to reproduce online. Here the "fine-tuning" is a carefully written rubric
prompt against Claude instead of a fine-tuned closed-source model -- for a
demo corpus this size, prompting is enough to get consistent labels; at real
production scale you'd fine-tune as the post does.

Usage:
    export ANTHROPIC_API_KEY=...
    python teacher/label_with_claude.py --in data/raw/esci_sample.jsonl \
        --out data/cache/teacher_labels.jsonl --limit 500

Pass --dry-run to exercise the whole script (parsing, caching, resume logic)
without spending API credits -- it substitutes a deterministic keyword-overlap
heuristic for the Claude call. Never treat --dry-run output as real supervision.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

RUBRIC = """You are grading search-ad relevance for a shopping search engine, \
the same way a human rater would. Given a search QUERY and a candidate ITEM, \
output exactly one relevance label:

0 = Irrelevant: the item shares no meaningful intent match with the query; \
showing it would feel like noise.
1 = Moderately relevant: an acceptable substitute (e.g. Pepsi for a "coke" \
query) or a secondary intent (e.g. orange juice for "orange"), missing some \
specific attribute the query implied.
2 = Highly relevant: a direct hit that precisely satisfies the query's intent.

Judge intent, not keyword overlap -- a title that merely contains the query's \
words is not automatically relevant. For example, a query for "salt" should \
score plain table/kosher salt as 2, but should score "salt & vinegar potato \
chips" as 0: it only shares a word with the query, not the intent (buying salt).

Respond with ONLY a JSON object on a single line: {{"label": 0|1|2, "reason": "<=12 words"}}

QUERY: {query}
ITEM: {item}
"""


def dry_run_label(query: str, item: str) -> dict:
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    i_tokens = set(re.findall(r"[a-z0-9]+", item.lower()))
    overlap = len(q_tokens & i_tokens) / max(len(q_tokens), 1)
    label = 2 if overlap > 0.6 else 1 if overlap > 0.2 else 0
    return {"label": label, "reason": f"dry-run token overlap={overlap:.2f}"}


def call_claude(client, model: str, query: str, item: str, retries: int = 4) -> dict:
    prompt = RUBRIC.format(query=query, item=item)
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else json.loads(text)
            label = int(parsed["label"])
            if label not in (0, 1, 2):
                raise ValueError(f"label out of range: {label}")
            return {"label": label, "reason": parsed.get("reason", "")}
        except Exception as exc:  # noqa: BLE001 -- retry on any transient/parse failure
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/raw/esci_sample.jsonl")
    ap.add_argument("--out", default="data/cache/teacher_labels.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="use a token-overlap heuristic instead of calling Claude")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                already_done.add((row["query"], row["product_id"]))
        print(f"Resuming: {len(already_done)} pairs already labeled")

    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            sys.exit("anthropic package not installed; run `pip install anthropic` "
                      "or pass --dry-run")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set. Export it, or pass --dry-run to "
                      "smoke-test the pipeline without calling the API.")
        client = anthropic.Anthropic(api_key=api_key)

    rows = []
    with in_path.open() as f:
        for line in f:
            row = json.loads(line)
            if (row["query"], row["product_id"]) in already_done:
                continue
            rows.append(row)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Labeling {len(rows)} pairs with "
          f"{'dry-run heuristic' if args.dry_run else args.model}")

    n_agree = 0
    with out_path.open("a") as f:
        for i, row in enumerate(rows):
            if args.dry_run:
                result = dry_run_label(row["query"], row["item_text"])
            else:
                result = call_claude(client, args.model, row["query"], row["item_text"])

            record = {
                "query": row["query"],
                "product_id": row["product_id"],
                "item_text": row["item_text"],
                "teacher_label": result["label"],
                "teacher_reason": result["reason"],
                "reference_relevance": row.get("reference_relevance"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            if row.get("reference_relevance") is not None and \
                    result["label"] == row["reference_relevance"]:
                n_agree += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} labeled")

    if rows:
        print(f"Done. Teacher/reference agreement: {n_agree}/{len(rows)} "
              f"({100 * n_agree / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
