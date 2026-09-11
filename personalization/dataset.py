"""Tokenizes persona/item interactions into feature vectors: item features
are a category one-hot plus normalized price, not text -- see model.py for
why personalization uses coarse attributes rather than the relevance model's
text encoder.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .simulate_users import CATEGORIES, PERSONAS

PERSONA_IDS = list(PERSONAS.keys())
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def item_features(group: str, price: float, price_min: float, price_max: float) -> list[float]:
    one_hot = [0.0] * len(CATEGORIES)
    one_hot[CATEGORY_INDEX[group]] = 1.0
    price_norm = (price - price_min) / (price_max - price_min) if price_max > price_min else 0.5
    return one_hot + [price_norm]


class PersonaInteractionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = [json.loads(line) for line in Path(jsonl_path).open()]
        prices = [row["price"] for row in self.rows]
        self.price_min, self.price_max = min(prices), max(prices)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        features = item_features(row["group"], row["price"], self.price_min, self.price_max)
        return {
            "persona_idx": torch.tensor(PERSONA_IDS.index(row["persona_id"]), dtype=torch.long),
            "item_features": torch.tensor(features, dtype=torch.float32),
            "engaged": torch.tensor(row["engaged"], dtype=torch.float32),
        }
