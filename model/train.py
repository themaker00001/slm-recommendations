"""Trains the DistilBERT bi-encoder student on teacher-generated labels.

At DoorDash's scale this runs distributed data-parallel across many GPUs; at
this demo's scale (a few thousand teacher-labeled pairs) a single
GPU/MPS/CPU device is plenty, so DDP setup is left out rather than faked.
Swapping in `torch.nn.parallel.DistributedDataParallel` around the model
below is the only change needed to scale this up.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.bi_encoder import BiEncoderRelevanceModel, DEFAULT_BACKBONE, DEFAULT_EMBED_DIM
from model.coral_loss import coral_loss, coral_predict
from model.dataset import RelevanceDataset
from model.evaluate import evaluate_ranking


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None, loss_type="cross_entropy"):
    training = optimizer is not None
    model.train(training)

    all_preds, all_labels, all_scores, all_queries = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        query_ids = batch["query_ids"].to(device)
        query_mask = batch["query_mask"].to(device)
        item_ids = batch["item_ids"].to(device)
        item_mask = batch["item_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(query_ids, query_mask, item_ids, item_mask)
            if loss_type == "cross_entropy":
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                loss = coral_loss(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (logits.argmax(dim=1) if loss_type == "cross_entropy"
                 else coral_predict(logits))
        scores = model.relevance_score(logits)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(scores.detach().cpu().tolist())
        all_queries.extend(batch["query_text"])

    metrics = evaluate_ranking(all_queries, all_preds, all_labels, all_scores)
    metrics["loss"] = total_loss / max(len(all_labels), 1)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cache/teacher_labels.jsonl")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    ap.add_argument("--loss", choices=["cross_entropy", "coral"], default="cross_entropy")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/bi_encoder.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    dataset = RelevanceDataset(args.data, tokenizer)
    if len(dataset) == 0:
        sys.exit(f"No labeled rows found in {args.data}. Run teacher/label_with_claude.py first.")

    explicit_split = dataset.explicit_split_indices()
    if explicit_split is not None:
        # Respect a data-prep-assigned split (e.g. oversampled duplicates of
        # the same underlying example, kept entirely on one side so a random
        # split can't leak copies of a training example into validation).
        train_idx, val_idx = explicit_split
        train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  (explicit split from data file)")
    else:
        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(args.seed))
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BiEncoderRelevanceModel(args.backbone, args.embed_dim, head=args.loss,
                                     dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.loss)
        val_metrics = run_epoch(model, val_loader, device, None, args.loss)
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_p@2={val_metrics['precision_at_2']} val_ndcg@10={val_metrics['ndcg_at_10']}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "backbone": args.backbone,
                "embed_dim": args.embed_dim,
                "head": args.loss,
                "val_accuracy": best_acc,
            }, out_path)
            print(f"  saved new best checkpoint (val_acc={best_acc:.4f}) -> {out_path}")

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
