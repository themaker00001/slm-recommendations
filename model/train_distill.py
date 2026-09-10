"""Trains the student via real knowledge distillation (soft cross-entropy
against the teacher's full probability distribution, model/distill_loss.py)
instead of ordinary supervised learning against a single hard label
(model/train.py). Deliberately a separate script, separate dataset class,
separate default checkpoint path -- this is meant to be a segregated,
directly comparable experiment against the hard-label baseline, not a
replacement for it. Run model/train.py on the same data file's hard labels
to get the matched baseline.

The resulting checkpoint is architecturally identical to a normal
cross-entropy checkpoint (same BiEncoderRelevanceModel, head="cross_entropy")
and loads into serve/predict.py or serve/app.py exactly the same way --
only how it was *trained* differs.
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.bi_encoder import BiEncoderRelevanceModel, DEFAULT_BACKBONE, DEFAULT_EMBED_DIM
from model.distill_dataset import DistillDataset
from model.distill_loss import distillation_loss
from model.evaluate import evaluate_ranking


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None, temperature=2.0):
    training = optimizer is not None
    model.train(training)

    all_preds, all_hard_labels, all_scores, all_queries = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        query_ids = batch["query_ids"].to(device)
        query_mask = batch["query_mask"].to(device)
        item_ids = batch["item_ids"].to(device)
        item_mask = batch["item_mask"].to(device)
        teacher_probs = batch["teacher_probs"].to(device)
        hard_labels = batch["hard_label"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(query_ids, query_mask, item_ids, item_mask)
            loss = distillation_loss(logits, teacher_probs, temperature)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(hard_labels)
        preds = logits.argmax(dim=1)
        scores = model.relevance_score(logits)

        all_preds.extend(preds.detach().cpu().tolist())
        all_hard_labels.extend(hard_labels.detach().cpu().tolist())
        all_scores.extend(scores.detach().cpu().tolist())
        all_queries.extend(batch["query_text"])

    # Accuracy here is measured against the teacher's *hard* argmax label,
    # purely so this number is directly comparable to model/train.py's --
    # the model itself never sees that hard label during distillation training.
    metrics = evaluate_ranking(all_queries, all_preds, all_hard_labels, all_scores)
    metrics["loss"] = total_loss / max(len(all_hard_labels), 1)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cache/teacher_labels_soft.jsonl")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/bi_encoder_distilled.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device()
    print(f"Device: {device}  (real distillation, T={args.temperature})")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    dataset = DistillDataset(args.data, tokenizer)
    if len(dataset) == 0:
        sys.exit(f"No soft-labeled rows found in {args.data}. "
                  f"Run teacher/label_with_ollama_soft.py first.")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BiEncoderRelevanceModel(args.backbone, args.embed_dim, head="cross_entropy").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.temperature)
        val_metrics = run_epoch(model, val_loader, device, None, args.temperature)
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_p@2={val_metrics['precision_at_2']} val_ndcg@10={val_metrics['ndcg_at_10']}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "backbone": args.backbone,
                "embed_dim": args.embed_dim,
                "head": "cross_entropy",
                "val_accuracy": best_acc,
                "trained_via": "distillation",
                "temperature": args.temperature,
            }, out_path)
            print(f"  saved new best checkpoint (val_acc={best_acc:.4f}) -> {out_path}")

    print(f"Best validation accuracy (vs. teacher's hard label): {best_acc:.4f}")


if __name__ == "__main__":
    main()
