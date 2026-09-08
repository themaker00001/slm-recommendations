"""Demonstrates the production serving shape from the blog: query/item
embeddings are computed once and cached, and only the cheap bilinear layer
runs per request -- here simulated as an in-memory dict instead of a real KV
store, and as a pre-auction relevance filter over a list of candidate items.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.bi_encoder import BiEncoderRelevanceModel

RELEVANCE_LABELS = {0: "IRRELEVANT (filtered pre-auction)", 1: "MODERATELY RELEVANT",
                     2: "HIGHLY RELEVANT"}


class RelevanceServer:
    """Stands in for: offline cron job embeds queries/items -> KV store cache
    -> online request fetches cached embeddings -> bilinear scoring."""

    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = BiEncoderRelevanceModel(ckpt["backbone"], ckpt["embed_dim"], ckpt["head"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["backbone"])
        self.embedding_cache = {}  # simulates the KV store

    def _embed(self, text: str, cache_key: str, max_len: int) -> torch.Tensor:
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        enc = self.tokenizer(text, truncation=True, max_length=max_len,
                              padding="max_length", return_tensors="pt")
        with torch.no_grad():
            emb = self.model.encode(enc["input_ids"], enc["attention_mask"])
        self.embedding_cache[cache_key] = emb
        return emb

    def embed_query(self, query: str) -> torch.Tensor:
        return self._embed(query, f"q::{query}", max_len=32)

    def embed_item(self, item_id: str, item_text: str) -> torch.Tensor:
        return self._embed(item_text, f"i::{item_id}", max_len=96)

    def score(self, query: str, item_id: str, item_text: str):
        query_emb = self.embed_query(query)
        item_emb = self.embed_item(item_id, item_text)
        with torch.no_grad():
            logits = self.model.score(query_emb, item_emb)
            label = self.model.predict_labels(query_emb, item_emb).item()
            relevance = self.model.relevance_score(logits).item()
        return label, relevance

    def filter_and_rank(self, query: str, candidates: list[dict], min_label: int = 1):
        """The pre-auction relevance gate: score every candidate ad, drop
        anything below min_label, and rank the survivors by relevance."""
        scored = []
        for c in candidates:
            label, relevance = self.score(query, c["item_id"], c["item_text"])
            scored.append({**c, "predicted_label": label, "relevance_score": relevance})
        kept = [c for c in scored if c["predicted_label"] >= min_label]
        kept.sort(key=lambda c: c["relevance_score"], reverse=True)
        dropped = [c for c in scored if c["predicted_label"] < min_label]
        return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/bi_encoder.pt")
    ap.add_argument("--query", required=True)
    ap.add_argument("--candidates", required=True,
                     help="path to a JSON file: a list of {item_id, item_text}")
    args = ap.parse_args()

    server = RelevanceServer(args.checkpoint)
    candidates = json.loads(Path(args.candidates).read_text())

    kept, dropped = server.filter_and_rank(args.query, candidates)

    print(f"\nQuery: {args.query!r}\n")
    print(f"Kept (ad-eligible, ranked by relevance):")
    for c in kept:
        print(f"  [{RELEVANCE_LABELS[c['predicted_label']]:>32}] "
              f"score={c['relevance_score']:.2f}  {c['item_text'][:70]}")

    print(f"\nFiltered out before auction:")
    for c in dropped:
        print(f"  [{RELEVANCE_LABELS[c['predicted_label']]:>32}] "
              f"score={c['relevance_score']:.2f}  {c['item_text'][:70]}")


if __name__ == "__main__":
    main()
