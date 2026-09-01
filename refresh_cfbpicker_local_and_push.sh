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
python scripts/scrape_cfbpicker_current.py \
  --root "$HERE" \
  --season "$SEASON" \
  --week "$WEEK" \
  --strict \
  --headed \
  --system-chrome \
  --no-github-fallback

git add -f data/current/cfbpicker_current_long.csv \
  data/current/cfbpicker_mirror_status.json \
  data/derived/cfbpicker_current_source_status.json \
  data/derived/cfbpicker_live_model_mapping.csv \
  "data/derived/cfbpicker_live_model_mapping_${SEASON}.csv" \
  "data/cfbpicker/collectable_pickers_${SEASON}.txt" 2>/dev/null || true
if [[ -d data/snapshots/cfbpicker/current ]]; then
  git add -f data/snapshots/cfbpicker/current
fi
if ! git diff --cached --quiet; then
  git commit -m "data: CFB Picker ${SEASON} week ${WEEK} refresh"
  git push origin main
else
  echo "CFB Picker mirror already current; no Git changes."
fi
