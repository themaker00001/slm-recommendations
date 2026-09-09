"""Tokenizes teacher-labeled query-item pairs for the bi-encoder student."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RelevanceDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_query_len: int = 32,
                 max_item_len: int = 96):
        self.rows = []
        with Path(jsonl_path).open() as f:
            for line in f:
                row = json.loads(line)
                if row.get("teacher_label") is None:
                    continue
                self.rows.append(row)
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_item_len = max_item_len

    def __len__(self):
        return len(self.rows)

    def explicit_split_indices(self):
        """If every row carries a "split": "train"/"val" field (set by a data
        prep script that needs to control the split itself -- e.g. to keep
        oversampled duplicates of the same underlying example entirely on one
        side, which a random split could otherwise leak across), return
        (train_indices, val_indices). Returns None if the field isn't present
        on every row, so train.py falls back to its own random split."""
        if not self.rows or any("split" not in row for row in self.rows):
            return None
        train_idx = [i for i, row in enumerate(self.rows) if row["split"] == "train"]
        val_idx = [i for i, row in enumerate(self.rows) if row["split"] == "val"]
        return train_idx, val_idx

    def __getitem__(self, idx):
        row = self.rows[idx]
        query_enc = self.tokenizer(row["query"], truncation=True,
                                    max_length=self.max_query_len,
                                    padding="max_length", return_tensors="pt")
        item_enc = self.tokenizer(row["item_text"], truncation=True,
                                   max_length=self.max_item_len,
                                   padding="max_length", return_tensors="pt")
        return {
            "query_ids": query_enc["input_ids"].squeeze(0),
            "query_mask": query_enc["attention_mask"].squeeze(0),
            "item_ids": item_enc["input_ids"].squeeze(0),
            "item_mask": item_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(row["teacher_label"], dtype=torch.long),
            "query_text": row["query"],
        }


def group_by_query(rows):
    """Used by evaluate.py for query-level ranking metrics (Precision@2, NDCG@10)."""
    groups = {}
    for row in rows:
        groups.setdefault(row["query_text"], []).append(row)
    return groups
