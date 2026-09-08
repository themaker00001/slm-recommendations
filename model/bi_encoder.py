"""The production-shaped student model: a DistilBERT bi-encoder.

Matches the blog's final production config: query and item pass through the
SAME encoder (shared weights, not two separately-trained towers), CLS pooling,
a linear projection down to a compact embedding, and a bilinear layer that
turns two embeddings into a relevance score. Query/item embeddings are meant
to be computed once (see serve/predict.py) and reused -- the bilinear layer is
the only thing that needs to run per online request.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_BACKBONE = "distilbert-base-uncased"
DEFAULT_EMBED_DIM = 64
NUM_CLASSES = 3  # 0 = irrelevant, 1 = moderately relevant, 2 = highly relevant


class BiEncoderRelevanceModel(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, embed_dim: int = DEFAULT_EMBED_DIM,
                 head: str = "cross_entropy"):
        super().__init__()
        assert head in ("cross_entropy", "coral")
        self.head_type = head
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size
        self.embed_dim = embed_dim

        # Direct linear projection for dimensionality reduction -- the blog
        # found this simpler and just as effective in production as Matryoshka
        # training, so that's what we implement here.
        self.projection = nn.Linear(hidden_size, embed_dim)

        # Bilinear scorer: score = query_emb^T W item_emb.
        # cross_entropy head -> 3 independent class logits.
        # coral head -> a single shared scalar score, turned into ordinal
        # threshold logits by CoralHead (see coral_loss.py).
        if head == "cross_entropy":
            self.bilinear = nn.Bilinear(embed_dim, embed_dim, NUM_CLASSES)
        else:
            self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)
            from .coral_loss import CoralHead
            self.coral_head = CoralHead()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Shared tower: text -> CLS-pooled, projected embedding. Used for
        both queries and items, since the blog's production model shares
        encoder weights across the two towers."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS pooling
        return self.projection(cls)

    def forward(self, query_ids, query_mask, item_ids, item_mask) -> torch.Tensor:
        query_emb = self.encode(query_ids, query_mask)
        item_emb = self.encode(item_ids, item_mask)
        return self.score(query_emb, item_emb)

    def score(self, query_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Online step: combine two pre-computed embeddings into logits.
        This is the only part of the model that needs to run at request time
        if embeddings are precomputed and cached, matching the blog's serving
        design (offline cron job + KV store + online bilinear layer)."""
        raw = self.bilinear(query_emb, item_emb)
        if self.head_type == "cross_entropy":
            return raw  # (B, NUM_CLASSES)
        return self.coral_head(raw)  # (B, NUM_CLASSES - 1)

    def predict_labels(self, query_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        logits = self.score(query_emb, item_emb)
        if self.head_type == "cross_entropy":
            return logits.argmax(dim=1)
        from .coral_loss import coral_predict
        return coral_predict(logits)

    def relevance_score(self, logits: torch.Tensor) -> torch.Tensor:
        """A single continuous score per pair, used to rank items within a
        query for Precision@2 / NDCG@10 -- not just the discrete label."""
        if self.head_type == "cross_entropy":
            probs = torch.softmax(logits, dim=1)
            levels = torch.arange(NUM_CLASSES, device=logits.device, dtype=probs.dtype)
            return (probs * levels).sum(dim=1)
        return torch.sigmoid(logits).sum(dim=1)
