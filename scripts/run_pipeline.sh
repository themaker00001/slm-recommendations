#!/usr/bin/env bash
# End-to-end demo: download -> teacher labeling -> train -> evaluate -> serve.
#
# Real run (needs ANTHROPIC_API_KEY exported, costs API credits):
#   ./scripts/run_pipeline.sh
#
# Smoke test with no API calls (heuristic teacher, tiny epoch count):
#   ./scripts/run_pipeline.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

LIMIT=${LIMIT:-1500}
EPOCHS=${EPOCHS:-5}

echo "== 1/4 Downloading ESCI sample =="
python data/download_esci.py --limit "$LIMIT"

echo "== 2/4 Teacher labeling =="
if $DRY_RUN; then
  python teacher/label_with_claude.py --dry-run --limit "$LIMIT"
  EPOCHS=1
else
  python teacher/label_with_claude.py --limit "$LIMIT"
fi

echo "== 3/4 Training student bi-encoder =="
python model/train.py --epochs "$EPOCHS"

echo "== 4/4 Serving demo =="
python serve/predict.py --query "salt" --candidates data/samples/salt_candidates.json

echo "Done."
