"""A small personalization model: predicts P(this persona engages with this
item), trained on the simulated interactions from simulate_users.py.

Same two-tower shape as the relevance bi-encoder (model/bi_encoder.py), for
the same reason DoorDash's own design runs relevance and a personalization/
quality model in parallel rather than as one model: they answer different
questions (does this match the query's intent? vs. would this specific user
want it?) and can be combined after both finish, keeping either one easy to
replace independently.

The user "tower" here is a learned embedding per persona rather than text --
there's no natural language description of a user the way there is of a
query, just their id and interaction history, so a lookup embedding is the
standard approach (this is what "collaborative filtering" embeddings are).
The item tower is a small feature vector (category + normalized price)
rather than the relevance model's full text encoder, since personalization
cares about coarse attributes (category, price point), not the item's exact
wording.
"""
import torch
import torch.nn as nn

EMBED_DIM = 16


class PersonalizationModel(nn.Module):
    def __init__(self, num_personas: int, num_categories: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.user_embedding = nn.Embedding(num_personas, embed_dim)
        # Item features: one-hot category + normalized price (1 dim).
        self.item_projection = nn.Linear(num_categories + 1, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, persona_idx: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(persona_idx)
        item_emb = self.item_projection(item_features)
        # Dot product + bias -> a single engagement logit per (user, item) pair.
        return (user_emb * item_emb).sum(dim=-1) + self.bias

    def engagement_prob(self, persona_idx: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(persona_idx, item_features))
