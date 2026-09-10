"""Semantic retrieval: dense embedding similarity search over the catalog,
recall-oriented like keyword_retrieve (serve/app.py) and meant to be unioned
with it, not replace it -- exactly the "add vector search alongside keyword
search" upgrade real hybrid search systems make.

Deliberately uses a general-purpose sentence-embedding model
(all-MiniLM-L6-v2) rather than the relevance bi-encoder's own embeddings.
The bi-encoder was trained through a bilinear scorer, not a contrastive
objective -- nothing pushed its query/item embeddings to sit close together
in cosine-similarity terms for a true match, so raw cosine similarity in
that space isn't guaranteed to mean what we'd want it to. all-MiniLM-L6-v2
is trained specifically so cosine similarity reflects semantic relatedness,
which is the actual property retrieval needs.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def item_text(item: dict) -> str:
    return f"{item['title']} {item['brand']} {item['category']}"


class SemanticRetriever:
    def __init__(self, catalog: list[dict], model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.catalog = catalog
        texts = [item_text(item) for item in catalog]
        self.item_embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 8, min_similarity: float = 0.2) -> list[tuple[dict, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.item_embeddings @ query_emb  # cosine similarity (both sides normalized)
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(self.catalog[i], float(sims[i])) for i in ranked_idx if sims[i] >= min_similarity]
