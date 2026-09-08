"""LLM teacher, fully local: labels query-item pairs 0/1/2 via a bigger model
running in Ollama, instead of a cloud API. This is the default teacher for
this project -- no API key, no external dependency, nothing leaves the
machine. The small bi-encoder student (model/bi_encoder.py) still ends up
tiny and fast; only the offline labeling step uses a bigger local model.

Usage:
    ollama serve                      # if not already running
    ollama pull qwen3:14b             # or any model you have pulled
    python teacher/label_with_ollama.py --in data/raw/esci_sample.jsonl \
        --out data/cache/teacher_labels.jsonl --model qwen3:14b --limit 500

Pass --dry-run to exercise the script's parsing/caching/resume logic with a
deterministic keyword-overlap heuristic instead of a real model call.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.label_with_claude import RUBRIC, dry_run_label  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"


def call_ollama(model: str, query: str, item: str, retries: int = 3,
                 base_url: str = OLLAMA_URL) -> dict:
    prompt = RUBRIC.format(query=query, item=item)
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(base_url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,  # skip chain-of-thought for reasoning models (e.g. qwen3)
                "options": {"temperature": 0},
            }, timeout=120)
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
            # Strip a <think>...</think> block if the model ignores think=false
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else json.loads(text)
            label = int(parsed["label"])
            if label not in (0, 1, 2):
                raise ValueError(f"label out of range: {label}")
            return {"label": label, "reason": parsed.get("reason", "")}
        except Exception as exc:  # noqa: BLE001 -- retry on any transient/parse failure
            last_exc = exc
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"Ollama labeling failed after {retries} attempts: {last_exc}")


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
    ap.add_argument("--out", default="data/cache/teacher_labels.jsonl")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="use a token-overlap heuristic instead of calling Ollama")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not check_ollama_reachable():
        sys.exit("Can't reach Ollama at localhost:11434. Run `ollama serve`, "
                  "or pass --dry-run to smoke-test without it.")

    already_done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                already_done.add((row["query"], row["product_id"]))
        print(f"Resuming: {len(already_done)} pairs already labeled")

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
          f"{'dry-run heuristic' if args.dry_run else f'local Ollama model {args.model}'}")

    n_agree = 0
    with out_path.open("a") as f:
        for i, row in enumerate(rows):
            if args.dry_run:
                result = dry_run_label(row["query"], row["item_text"])
            else:
                result = call_ollama(args.model, row["query"], row["item_text"])

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
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} labeled")

    if rows:
        print(f"Done. Teacher/reference agreement: {n_agree}/{len(rows)} "
              f"({100 * n_agree / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
