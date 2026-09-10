"""LLM teacher, soft-label variant: instead of keeping only the teacher's
single best answer (0, 1, or 2), this keeps its full confidence distribution
over all three -- the actual probabilities it assigned each option before
picking the most likely one. That distribution is what real knowledge
distillation trains the student against (via a soft cross-entropy / KL loss
in model/distill_loss.py), instead of a single "correct answer" label.

This is a genuinely different technique from teacher/label_with_ollama.py,
not just a variant -- see the module docstring on model/distill_loss.py and
model/train_distill.py for what changes as a result, and README.md's
"Real distillation vs. label generation" section for the full comparison.

Mechanically: Ollama's native /api/chat can return `logprobs` for each
generated token when `think: false` (so the answer token comes first, not
buried after a chain-of-thought). Prompting for a bare single digit, rather
than a JSON object with a reason, is what makes the *first* generated token
the actual answer -- necessary to get clean logprobs on it, at the cost of
losing the one-sentence "reason" the hard-label teacher also captured.

Usage:
    python teacher/label_with_ollama_soft.py --in data/raw/esci_sample.jsonl \
        --out data/cache/teacher_labels_soft.jsonl --model qwen3:14b
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

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

Respond with ONLY the single digit 0, 1, or 2. No other text.

QUERY: {query}
ITEM: {item}"""

OLLAMA_URL = "http://localhost:11434/api/chat"


def soft_label(model: str, query: str, item: str, retries: int = 3,
                base_url: str = OLLAMA_URL) -> dict:
    """Returns {"probs": [p0, p1, p2], "hard_label": argmax}."""
    prompt = RUBRIC.format(query=query, item=item)
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(base_url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
                "logprobs": True,
                "top_logprobs": 10,
            }, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            top = data["logprobs"][0]["top_logprobs"]

            logprobs_by_digit = {}
            for entry in top:
                tok = entry["token"].strip()
                if tok in ("0", "1", "2") and tok not in logprobs_by_digit:
                    logprobs_by_digit[tok] = entry["logprob"]

            if not logprobs_by_digit:
                raise ValueError(f"no 0/1/2 token in top_logprobs: {top}")

            # Convert log-probabilities to probabilities and renormalize over
            # just {0,1,2} -- the model's mass outside these three tokens
            # (which should be near-zero given the prompt) is discarded.
            raw = {d: math.exp(lp) for d, lp in logprobs_by_digit.items()}
            for d in ("0", "1", "2"):
                raw.setdefault(d, 1e-6)  # digit didn't appear in top-k: treat as near-zero, not zero
            total = sum(raw.values())
            probs = [raw["0"] / total, raw["1"] / total, raw["2"] / total]
            hard_label = max(range(3), key=lambda i: probs[i])
            return {"probs": probs, "hard_label": hard_label}
        except Exception as exc:  # noqa: BLE001 -- retry on any transient/parse failure
            last_exc = exc
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"soft labeling failed after {retries} attempts: {last_exc}")


def check_ollama_reachable(base_url: str = OLLAMA_URL) -> bool:
    try:
        requests.get(base_url.replace("/api/chat", "/api/tags"), timeout=3)
        return True
    except requests.RequestException:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/raw/esci_sample.jsonl")
    ap.add_argument("--out", default="data/cache/teacher_labels_soft.jsonl")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not check_ollama_reachable():
        sys.exit("Can't reach Ollama at localhost:11434. Run `ollama serve`.")

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                already_done.add((row["query"], row["product_id"]))
        print(f"Resuming: {len(already_done)} pairs already soft-labeled")

    rows = []
    with in_path.open() as f:
        for line in f:
            row = json.loads(line)
            if (row["query"], row["product_id"]) in already_done:
                continue
            rows.append(row)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Soft-labeling {len(rows)} pairs with local Ollama model {args.model}")

    with out_path.open("a") as f:
        for i, row in enumerate(rows):
            result = soft_label(args.model, row["query"], row["item_text"])
            record = {
                "query": row["query"],
                "product_id": row["product_id"],
                "item_text": row["item_text"],
                "teacher_label": result["hard_label"],
                "teacher_probs": result["probs"],
                "reference_relevance": row.get("reference_relevance"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} labeled")

    print("Done.")


if __name__ == "__main__":
    main()
