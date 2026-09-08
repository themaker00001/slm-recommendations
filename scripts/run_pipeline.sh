#!/usr/bin/env bash
# End-to-end demo: download -> teacher labeling -> train -> evaluate -> serve.
#
# Fully local run, default (needs `ollama serve` running with TEACHER_MODEL pulled):
#   ./scripts/run_pipeline.sh
#
# Cloud teacher instead (needs ANTHROPIC_API_KEY exported, costs API credits):
#   TEACHER=claude ./scripts/run_pipeline.sh
#
# Smoke test with no model calls at all (heuristic teacher, 1 epoch):
#   ./scripts/run_pipeline.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

LIMIT=${LIMIT:-1500}
EPOCHS=${EPOCHS:-5}
TEACHER=${TEACHER:-ollama}       # ollama | claude
TEACHER_MODEL=${TEACHER_MODEL:-qwen3:14b}

echo "== 1/4 Downloading ESCI sample =="
python data/download_esci.py --limit "$LIMIT"

echo "== 2/4 Teacher labeling ($TEACHER) =="
if $DRY_RUN; then
  python "teacher/label_with_${TEACHER}.py" --dry-run --limit "$LIMIT"
  EPOCHS=1
elif [[ "$TEACHER" == "ollama" ]]; then
  python teacher/label_with_ollama.py --model "$TEACHER_MODEL" --limit "$LIMIT"
else
  python teacher/label_with_claude.py --limit "$LIMIT"
fi

echo "== 3/4 Training student bi-encoder =="
python model/train.py --epochs "$EPOCHS"

echo "== 4/4 Serving demo =="
python serve/predict.py --query "salt" --candidates data/samples/salt_candidates.json

echo "Done. Try the frontend: python serve/app.py, then open http://localhost:8000"
