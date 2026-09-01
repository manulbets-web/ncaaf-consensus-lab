#!/usr/bin/env bash
set -euo pipefail
SEASON="${1:-}"
WEEK="${2:-}"
if [[ -z "$SEASON" || -z "$WEEK" ]]; then
  echo "Usage: ./refresh_cfbpicker_local_and_push.sh SEASON WEEK" >&2
  exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Refresh PredictionTracker first.  Its latest verified game list is the live
# slate authority used to keep Tableau's cumulative Wk 1-2 view from carrying
# already-played games into the CFB Picker cache.
python scripts/refresh_predictiontracker_mirror.py \
  --root "$HERE" \
  --season "$SEASON" \
  --week "$WEEK"

python scripts/scrape_cfbpicker_current.py \
  --root "$HERE" \
  --season "$SEASON" \
  --week "$WEEK" \
  --strict \
  --headed \
  --system-chrome \
  --no-github-fallback

git add -f data/current/ncaapredictions.csv \
  data/current/predictiontracker_mirror_status.json \
  data/derived/predictiontracker_source_status.json \
  data/current/cfbpicker_current_long.csv \
  data/current/cfbpicker_mirror_status.json \
  data/derived/cfbpicker_current_source_status.json \
  data/derived/cfbpicker_live_model_mapping.csv \
  "data/derived/cfbpicker_live_model_mapping_${SEASON}.csv" \
  "data/cfbpicker/collectable_pickers_${SEASON}.txt" 2>/dev/null || true
if [[ -d data/snapshots/predictiontracker/mirror ]]; then
  git add -f data/snapshots/predictiontracker/mirror
fi
if [[ -d data/snapshots/cfbpicker/current ]]; then
  git add -f data/snapshots/cfbpicker/current
fi
if ! git diff --cached --quiet; then
  git commit -m "data: CFB Picker ${SEASON} week ${WEEK} refresh"
  git push origin main
else
  echo "CFB Picker mirror already current; no Git changes."
fi
