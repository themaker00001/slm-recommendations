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
python model/train.py --data data/cache/teacher_labels.jsonl --epochs 5
python serve/predict.py --query "salt" --candidates data/samples/salt_candidates.json
```

`serve/predict.py` prints which candidate ads survive the relevance filter
and how they'd rank, e.g. for the query "salt": table salt and kosher salt
rank as highly relevant, "salt & vinegar chips" gets caught as a weak
keyword-overlap match (exactly the failure mode the blog opens with), and
milk gets filtered out entirely.

## Training data: what actually moved the needle (and what didn't)

Training on ESCI alone produces a model that's inconsistent on the frontend's
demo catalog — not because ESCI is bad data, but because ESCI is long,
specific e-commerce search phrases (`"someday is not a day of the week
shirt"`) while the catalog and its queries are short and generic (`"salt"`,
`"orange"`). Three things were tried, in order, testing against the actual
demo catalog each time rather than trusting the offline validation number
alone:

1. **More epochs on the same 800 ESCI pairs** — did nothing. Validation
   accuracy plateaued at the same 75% it hit by epoch 4; training loss kept
   dropping toward zero while validation loss climbed. Textbook overfitting,
   not a fix.
2. **Synthetic data augmented from the raw ESCI pairs** (label-preserving
   query/item truncation for the underrepresented classes, plus a small
   batch of auto-labeled random cross-pairs) — [`data/augment_synthetic.py`](data/augment_synthetic.py).
   Validation accuracy rose to 87%, but real behavior on the demo catalog
   didn't improve — if anything some cases got worse (a query that used to
   show varied scores collapsed to "everything is irrelevant"). The
   validation number was partly measuring the synthetic examples' own
   easiness, not real generalization. Kept in the repo since it's a real,
   reusable technique — just not the fix for *this* problem.
3. **Real domain-matched labeled data** — [`data/build_grocery_domain_pairs.py`](data/build_grocery_domain_pairs.py)
   generates every (query, item) pair from the demo catalog's own 11
   categories crossed with all 55 items, and the local Ollama teacher labels
   all 605 of them for real. Mixing the ~71 useful pairs (same-category
   matches, including the deliberate keyword-overlap traps, plus a few
   genuinely ambiguous cross-category hits) straight into the 800 ESCI pairs
   improved two hand-tested cases ("salt", "orange") but broke four other
   categories completely (headphones, candles, fruits, dairy all came back
   100% wrong) — 71 examples spread across 11 categories inside ~1000 total
   rows is too thin a signal per category to reliably learn 11 separate
   associations. **Oversampling** those 71 examples 8x (repeating them in the
   training file so they carry proportionally more weight) fixed that: full
   catalog re-test came back correct or reasonable on 10 of the 11
   categories, with only individual item-level misses left (a coffee mug
   marked "highly relevant", an unrelated soup for "candles").

   That run's 85% validation accuracy turned out to be **inflated by a data
   leak** (see below) — after fixing it, honest validation accuracy on this
   same data is **73.2%**, and the real catalog behavior is roughly a wash
   against the leaky version (some categories better, a couple worse,
   nothing decisively different). That's the actual point: the leaky 85%
   never corresponded to better real-world quality, it just *looked* better
   on a metric that was quietly cheating.

[`data/merge_training_data.py`](data/merge_training_data.py) does this
merge-and-oversample step. One methodological detail it handles that's easy
to get wrong: **the train/val split has to happen before oversampling, not
after.** Duplicating a row 8 times and then handing the whole file to a
random 80/20 split lets some copies land in "train" and others in "val" by
chance — so the model can be validated on an example it was also trained on,
quietly inflating the accuracy number. The script splits the *unique*
examples first, then oversamples only the training side; `model/train.py`
respects an explicit `"split"` field per row when the whole file carries one,
instead of re-splitting randomly.

```bash
python data/build_grocery_domain_pairs.py
python teacher/label_with_ollama.py \
  --in data/raw/grocery_domain_pairs.jsonl \
  --out data/cache/grocery_domain_labels.jsonl
python data/merge_training_data.py --repeat 8
python model/train.py --data data/cache/teacher_labels_merged.jsonl --epochs 8
```

## Frontend: testing it interactively

```bash
python serve/app.py
# open http://localhost:8000
```

A local search-engine-style UI (FastAPI backend, plain HTML/JS frontend, no
build step, no CDN dependency) that demonstrates the whole funnel the blog
describes, not just the model in isolation:

- **Search bar + suggestion chips** run a small built-in demo catalog
  (`data/samples/demo_catalog.json`, ~30 items across salt/coffee/
  headphones/birthday-candles/orange/healthy-snacks — several picked to
  match the blog's own examples) through a **naive keyword retrieval** step
  first (`serve/app.py`'s `keyword_retrieve`) — recall-optimized, no notion
  of intent, so it happily retrieves "salt & vinegar chips" for "salt".
- The **relevance filter toggle** shows the difference this project exists
  to make: off, you see raw keyword-retrieval results, irrelevant items
  included; on, the trained SLM scores and reranks the same candidates and
  moves anything predicted irrelevant into a collapsed "filtered out"
  section — the same pre-auction gate `serve/predict.py` does from the CLI.
- Each result card shows a color-coded relevance chip and a **"Why?"**
  expander with the SLM's score, plus an **"Ask local teacher"** button that
  calls Ollama live for that one item so you can compare the small model's
  call against the bigger local model's judgment on the spot.
- An **Advanced** panel below the results lets you type any custom
  query/item pair and score it with the SLM or the teacher directly, for
  testing outside the demo catalog.
- A status popover shows whether a trained checkpoint is loaded and whether
  Ollama is reachable.

Nothing here calls out to the internet — the backend only talks to the local
checkpoint file and to `localhost:11434` (Ollama).

## Repo layout

```
data/download_esci.py           sample public ESCI query-item pairs
data/build_grocery_domain_pairs.py  generate demo-catalog-style query x item pairs
data/merge_training_data.py     leakage-safe merge + oversample of ESCI + domain-matched data
data/augment_synthetic.py       label-preserving text perturbation (didn't fix the real problem, kept anyway)
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
