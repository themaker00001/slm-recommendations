"""CORAL: COnsistent RAnk Logits ordinal regression, the alternative loss the
blog post evaluated against plain cross-entropy for the 0/1/2 relevance scale.

Standard ordinal-regression trick (Cao et al., 2020): instead of K independent
class logits, learn ONE shared scalar score per example plus K-1 per-threshold
biases, and predict K-1 binary "is relevance > k?" probabilities. Sharing the
scalar across thresholds (rather than giving each threshold its own weights,
as plain cross-entropy effectively does) is what makes the thresholds behave
consistently as relevance increases, instead of learning contradictory cutoffs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bi_encoder import NUM_CLASSES

NUM_THRESHOLDS = NUM_CLASSES - 1  # relevance > 0, relevance > 1


class CoralHead(nn.Module):
    """Turns a single shared scalar score into NUM_THRESHOLDS ordinal logits."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(NUM_THRESHOLDS))

    def forward(self, scalar_score: torch.Tensor) -> torch.Tensor:
        # scalar_score: (B, 1) -> (B, NUM_THRESHOLDS)
        return scalar_score + self.bias


def levels_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """label 0 -> [0, 0], label 1 -> [1, 0], label 2 -> [1, 1]
    (i.e. levels[:, k] = 1{label > k})."""
    thresholds = torch.arange(NUM_THRESHOLDS, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    levels = levels_from_labels(labels)
    return F.binary_cross_entropy_with_logits(logits, levels)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Sum of passed thresholds gives the predicted ordinal class."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)
