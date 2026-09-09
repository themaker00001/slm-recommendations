#!/usr/bin/env bash
# End-to-end demo: download -> teacher labeling -> domain-matched data ->
# leakage-safe merge -> train -> serve.
#
# Fully local run, default (needs `ollama serve` running with TEACHER_MODEL pulled):
#   ./scripts/run_pipeline.sh
#
# Cloud teacher instead (needs ANTHROPIC_API_KEY exported, costs API credits):
#   TEACHER=claude ./scripts/run_pipeline.sh
#
# Smoke test with no model calls at all (heuristic teacher, 1 epoch, no merge):
#   ./scripts/run_pipeline.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

LIMIT=${LIMIT:-1500}
EPOCHS=${EPOCHS:-8}
TEACHER=${TEACHER:-ollama}       # ollama | claude
TEACHER_MODEL=${TEACHER_MODEL:-qwen3:14b}
REPEAT=${REPEAT:-8}

echo "== 1/6 Downloading ESCI sample =="
python data/download_esci.py --limit "$LIMIT"

echo "== 2/6 Teacher labeling ESCI pairs ($TEACHER) =="
if $DRY_RUN; then
  python "teacher/label_with_${TEACHER}.py" --dry-run --limit "$LIMIT"
  EPOCHS=1
elif [[ "$TEACHER" == "ollama" ]]; then
  python teacher/label_with_ollama.py --model "$TEACHER_MODEL" --limit "$LIMIT"
else
  python teacher/label_with_claude.py --limit "$LIMIT"
fi

if $DRY_RUN; then
  echo "== 3-4/6 Skipped in --dry-run (no domain-matched data) =="
  TRAIN_DATA="data/cache/teacher_labels.jsonl"
else
  echo "== 3/6 Building demo-catalog-style query x item pairs =="
  python data/build_grocery_domain_pairs.py

  echo "== 4/6 Teacher labeling domain-matched pairs =="
  python teacher/label_with_ollama.py --model "$TEACHER_MODEL" \
    --in data/raw/grocery_domain_pairs.jsonl \
    --out data/cache/grocery_domain_labels.jsonl

  echo "== 5/6 Merging (leakage-safe split, oversampling domain-matched data) =="
  python data/merge_training_data.py --repeat "$REPEAT"
  TRAIN_DATA="data/cache/teacher_labels_merged.jsonl"
fi

echo "== 6/6 Training student bi-encoder =="
python model/train.py --data "$TRAIN_DATA" --epochs "$EPOCHS"

echo "== Serving demo =="
python serve/predict.py --query "salt" --candidates data/samples/salt_candidates.json

echo "Done. Try the frontend: python serve/app.py, then open http://localhost:8000"
