"""Accuracy plus the query-level ranking metrics the blog reports online
(Precision@2, NDCG@10), computed here offline against the teacher labels
standing in for their fine-tuned-GPT relevance judges."""
import math
from collections import defaultdict


def accuracy(preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / max(len(labels), 1)


def precision_at_k(ranked_labels, k=2, positive_threshold=1):
    """Fraction of the top-k ranked items that are at least moderately relevant."""
    top_k = ranked_labels[:k]
    if not top_k:
        return None
    hits = sum(1 for label in top_k if label >= positive_threshold)
    return hits / len(top_k)


def ndcg_at_k(ranked_labels, ideal_labels, k=10):
    def dcg(labels):
        return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(labels[:k]))

    ideal = dcg(sorted(ideal_labels, reverse=True))
    if ideal == 0:
        return None
    return dcg(ranked_labels) / ideal


def evaluate_ranking(query_texts, preds, labels, scores):
    """Groups predictions by query, ranks items within each query by predicted
    score, and reports mean Precision@2 / NDCG@10 alongside plain accuracy."""
    by_query = defaultdict(list)
    for q, pred, label, score in zip(query_texts, preds, labels, scores):
        by_query[q].append((score, label))

    p_at_2, ndcg_at_10 = [], []
    for q, items in by_query.items():
        ranked = [label for _, label in sorted(items, key=lambda x: x[0], reverse=True)]
        p = precision_at_k(ranked, k=2)
        n = ndcg_at_k(ranked, [label for _, label in items], k=10)
        if p is not None:
            p_at_2.append(p)
        if n is not None:
            ndcg_at_10.append(n)

    return {
        "accuracy": accuracy(preds, labels),
        "precision_at_2": sum(p_at_2) / len(p_at_2) if p_at_2 else None,
        "ndcg_at_10": sum(ndcg_at_10) / len(ndcg_at_10) if ndcg_at_10 else None,
        "num_queries": len(by_query),
    }
