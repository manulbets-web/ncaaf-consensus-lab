#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SEASON WEEK" >&2
  echo "Example: $0 2026 2" >&2
  exit 2
fi
SEASON="$1"
WEEK="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

python scripts/refresh_predictiontracker_mirror.py --root . --season "$SEASON" --week "$WEEK"

git add \
  data/current/ncaapredictions.csv \
  data/current/predictiontracker_mirror_status.json \
  data/derived/predictiontracker_source_status.json \
  data/raw/predictiontracker \
  data/snapshots/predictiontracker/mirror

if ! git diff --cached --quiet; then
  git commit -m "data: PredictionTracker ${SEASON} week ${WEEK} refresh"
  git push origin main
  echo "Pushed refreshed PredictionTracker mirror to GitHub."
else
  echo "PredictionTracker source is unchanged; nothing new to push."
fi
