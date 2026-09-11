"""Closes the one step of the blog's pipeline this repo otherwise skips
entirely: fine-tuning a pretrained LLM on a human-labeled seed set to turn
it into a relevance-judging teacher, instead of just prompting a pretrained
model zero-shot (teacher/label_with_ollama.py's approach everywhere else in
this repo).

The "human" labels here are data/seed_labels_claude.jsonl -- Claude reading
each query/item pair directly and judging it against the rubric, NOT
independent human raters. Say that plainly wherever this script's results
get reported; it's a real, named gap versus DoorDash's actual 700K-human-
label seed set, not a detail to gloss over.

Base model: Qwen2.5-0.5B-Instruct (Apache 2.0, ungated, small enough to
LoRA fine-tune on a laptop in well under a minute). The prompt asks for a
bare digit (0/1/2) as the very next token, exactly like
teacher/label_with_ollama_soft.py's soft-labeling approach -- so training
only needs to teach the model to emit the right single token, and
evaluation is just "did it get that one token right."

Usage:
    python teacher/fine_tune_teacher.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teacher.label_with_claude import RUBRIC  # noqa: E402 -- same rubric as every other teacher

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_prompt(tokenizer, query: str, item: str) -> str:
    rubric = RUBRIC.format(query=query, item=item).replace(
        'Respond with ONLY a JSON object on a single line: {"label": 0|1|2, "reason": "<=12 words"}',
        "Respond with ONLY the single digit 0, 1, or 2. No other text.",
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": rubric}], tokenize=False, add_generation_prompt=True)


def load_seed_set(path: str, val_fraction: float = 0.2, seed: int = 0):
    rows = [json.loads(line) for line in Path(path).open()]
    import random
    random.Random(seed).shuffle(rows)
    n_val = max(4, int(len(rows) * val_fraction))
    return rows[n_val:], rows[:n_val]


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], device: str) -> float:
    model.eval()
    correct = 0
    for row in rows:
        prompt = build_prompt(tokenizer, row["query"], row["item_text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
        pred_token = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:])
        pred = pred_token.strip()
        correct += int(pred == str(row["teacher_label"]))
    return correct / len(rows)


def train_lora(model, tokenizer, train_rows: list[dict], device: str,
                epochs: int, lr: float) -> None:
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    examples = []
    for row in train_rows:
        prompt = build_prompt(tokenizer, row["query"], row["item_text"])
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        label_id = tokenizer.encode(str(row["teacher_label"]), add_special_tokens=False)
        eos_id = torch.tensor([tokenizer.eos_token_id])
        full_ids = torch.cat([prompt_ids, torch.tensor(label_id), eos_id])
        # Mask the prompt out of the loss -- only the label digit (and the
        # EOS that ends the turn) should ever be predicted or penalized.
        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100
        examples.append((full_ids, labels))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for input_ids, labels in examples:
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            out = model(input_ids=input_ids, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        print(f"Epoch {epoch}: train_loss={total_loss / len(examples):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/seed_labels_claude.jsonl")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/fine_tuned_teacher_lora")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    train_rows, val_rows = load_seed_set(args.data)
    print(f"Seed set: {len(train_rows)} train, {len(val_rows)} val")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)

    print("\n=== Zero-shot (before fine-tuning) ===")
    zero_shot_acc = evaluate(model, tokenizer, val_rows, device)
    print(f"Zero-shot accuracy on held-out seed examples: {zero_shot_acc:.4f}")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n=== Fine-tuning on the Claude-labeled seed set ===")
    train_lora(model, tokenizer, train_rows, device, args.epochs, args.lr)

    print("\n=== After fine-tuning ===")
    fine_tuned_acc = evaluate(model, tokenizer, val_rows, device)
    print(f"Fine-tuned accuracy on the SAME held-out seed examples: {fine_tuned_acc:.4f}")

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    print(f"\nSaved LoRA adapter -> {out_path}")
    print(f"\nSummary: zero-shot {zero_shot_acc:.4f} -> fine-tuned {fine_tuned_acc:.4f} "
          f"on {len(val_rows)} held-out seed examples")


if __name__ == "__main__":
    main()
