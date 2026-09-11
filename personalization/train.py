"""Trains the personalization model on simulated persona interactions.
Tiny model, tiny data (a few hundred rows) -- this trains in seconds on CPU,
unlike the relevance bi-encoder which needs a real encoder forward pass.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personalization.dataset import PersonaInteractionDataset, PERSONA_IDS, CATEGORIES
from personalization.model import PersonalizationModel


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    for batch in loader:
        with torch.set_grad_enabled(training):
            logits = model(batch["persona_idx"], batch["item_features"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["engaged"])
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == batch["engaged"]).sum().item()
        n += len(batch["engaged"])
        total_loss += loss.item() * len(batch["engaged"])

    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/cache/persona_interactions.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/personalization.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    dataset = PersonaInteractionDataset(args.data)
    if len(dataset) == 0:
        sys.exit(f"No interactions found in {args.data}. Run personalization/simulate_users.py first.")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = PersonalizationModel(num_personas=len(PERSONA_IDS), num_categories=len(CATEGORIES))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, None)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_personas": len(PERSONA_IDS),
                "num_categories": len(CATEGORIES),
                "persona_ids": PERSONA_IDS,
                "categories": CATEGORIES,
                "val_accuracy": best_acc,
            }, out_path)

    print(f"Best validation accuracy: {best_acc:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
