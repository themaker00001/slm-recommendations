# SLM Relevance

A small-language-model relevance filter for search results/ads, built by
reimplementing the architecture described in DoorDash's engineering blog post
["Using small language models to serve more relevant DoorDash search
ads"](https://careersatdoordash.com/blog/small-language-models-to-serve-more-relevant-doordash-search-ads/)
(June 2026).

## The idea

Keyword retrieval optimizes for recall and personalization optimizes for
engagement — neither guarantees that a candidate result actually matches what
the query meant. DoorDash's fix is a dedicated, user-agnostic **query-item
relevance model** that sits before the ad auction and filters out results
that are objectively off-target, on a 3-level scale:

| Label | Meaning |
|---|---|
| 0 | Irrelevant — no meaningful intent match |
| 1 | Moderately relevant — an acceptable substitute or secondary intent |
| 2 | Highly relevant — a direct hit on the query's intent |

Labeling millions of query-item pairs by hand doesn't scale, so the system
uses a **teacher-student split**:

- **Teacher (offline, big, slow, expensive):** an LLM labels query-item pairs
  in batch, mirroring human judgment.
- **Student (online, small, fast, cheap):** a compact BERT-family bi-encoder
  trains on the teacher's labels and serves real-time relevance scores at
  the latency budget an ad auction needs.

## What this repo implements

| Blog concept | This repo |
|---|---|
| Proprietary 700k-example human-labeled query-item ad corpus | [`data/download_esci.py`](data/download_esci.py) samples the public [Amazon ESCI](https://github.com/amazon-science/esci-data) query-product relevance dataset as a stand-in |
| Fine-tuned closed-source LLM teacher, labels 6 months of production traffic offline | [`teacher/label_with_ollama.py`](teacher/label_with_ollama.py) prompts a bigger **local** model (via [Ollama](https://ollama.com)) with the same 0/1/2 rubric described in the post — nothing leaves the machine. [`teacher/label_with_claude.py`](teacher/label_with_claude.py) is an optional cloud alternative if you'd rather use the Claude API. |
| DistilBERT bi-encoder student, shared encoder weights, CLS pooling, 64-dim linear-projected embeddings, online bilinear scorer | [`model/bi_encoder.py`](model/bi_encoder.py) |
| Cross-entropy vs. CORAL ordinal regression loss comparison | [`model/coral_loss.py`](model/coral_loss.py), selectable via `--loss` |
| AdamW training, validation-accuracy checkpoint selection | [`model/train.py`](model/train.py) |
| Precision@2 / NDCG@10 online evaluation metrics | [`model/evaluate.py`](model/evaluate.py) |
| Offline embedding cache (cron job + KV store) + cheap online bilinear scoring, relevance filter before the auction | [`serve/predict.py`](serve/predict.py) (CLI) and [`serve/app.py`](serve/app.py) (local web UI, see below) |

**Fully local by default.** The "big model labels data for the small model"
split doesn't require a cloud API: the teacher step runs against whatever
model you already have pulled in Ollama (default `qwen3:14b`), and the
student (DistilBERT, ~66M params) trains and serves entirely on-device. The
Claude-API teacher is kept as an opt-in alternative, not the default.

**Honest scope note:** this is a from-scratch reimplementation of the
*architecture*, not DoorDash's system or data. It swaps their proprietary ads
corpus for a public dataset and their fine-tuned closed-source teacher for a
prompted local model, and it trains on a demo-sized sample (thousands, not
millions, of pairs) on a single machine rather than distributed across many
GPUs. The modeling choices (bi-encoder shape, embedding dimension, losses,
serving split) follow the post directly.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Local teacher (default) — needs Ollama running with a model pulled:
ollama serve                 # if not already running in the background
ollama pull qwen3:14b        # or any model you already have, e.g. llama3.2

# Optional cloud teacher instead:
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the pipeline

```bash
# Smoke-test everything with no model calls at all (heuristic teacher, 1 epoch):
./scripts/run_pipeline.sh --dry-run

# Real run, fully local (Ollama teacher, the default):
LIMIT=1500 EPOCHS=5 ./scripts/run_pipeline.sh

# Real run with the Claude API teacher instead:
TEACHER=claude LIMIT=1500 EPOCHS=5 ./scripts/run_pipeline.sh
```

Or step by step:

```bash
python data/download_esci.py --limit 1500
python teacher/label_with_ollama.py --model qwen3:14b --limit 1500   # or --dry-run
python model/train.py --loss cross_entropy --epochs 5                # or --loss coral
python serve/predict.py --query "salt" --candidates data/samples/salt_candidates.json
```

`serve/predict.py` prints which candidate ads survive the relevance filter
and how they'd rank, e.g. for the query "salt": table salt and kosher salt
rank as highly relevant, "salt & vinegar chips" gets caught as a weak
keyword-overlap match (exactly the failure mode the blog opens with), and
milk gets filtered out entirely.

## Frontend: testing it interactively

```bash
python serve/app.py
# open http://localhost:8000
```

A single-page local UI (FastAPI backend, plain HTML/JS frontend, no build
step, no CDN dependency) for trying queries and candidate items by hand:

- **Run local SLM** — scores every candidate with the trained bi-encoder
  checkpoint and ranks them, exactly like `serve/predict.py`.
- **Ask local teacher (Ollama)** — labels the same pairs live with your
  chosen local Ollama model, so you can compare the small student's
  predictions against the bigger local model's judgment side by side.
- A status bar shows whether a trained checkpoint is loaded and whether
  Ollama is reachable, so it's obvious what's missing if something's blank.

Nothing here calls out to the internet — the backend only talks to the local
checkpoint file and to `localhost:11434` (Ollama).

## Repo layout

```
data/download_esci.py         sample public query-item pairs
teacher/label_with_ollama.py  LLM teacher (local, default): batch-labels pairs 0/1/2 via Ollama
teacher/label_with_claude.py  LLM teacher (cloud, optional): same rubric via the Claude API
model/bi_encoder.py           DistilBERT bi-encoder student
model/coral_loss.py           CORAL ordinal-regression loss + head
model/dataset.py              tokenization / batching
model/train.py                training loop
model/evaluate.py             accuracy, Precision@2, NDCG@10
serve/predict.py              CLI: cached-embedding scoring + relevance gate
serve/app.py                  local web UI backend (FastAPI)
serve/static/                 local web UI frontend (HTML/CSS/JS)
scripts/run_pipeline.sh       end-to-end orchestration
```

## Scaling this up

- Swap `data/download_esci.py` for your own query-item corpus (any source of
  `{query, item_text}` pairs works — the teacher doesn't need ESCI's own
  labels, only text).
- `model/train.py` runs on a single device (CUDA/MPS/CPU auto-detected);
  wrapping the model in `torch.nn.parallel.DistributedDataParallel` is the
  only change needed to go multi-GPU, as the original post does.
- Swap in a bigger/smaller Ollama model via `--model` on
  `teacher/label_with_ollama.py` to trade teacher quality for labeling speed.
