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

# The repository intentionally ignores data/raw/ and data/snapshots/ in normal
# deployments.  The live mirror files themselves are not ignored, while the
# prospective mirror snapshots are intentionally force-added so they remain a
# permanent research record.  Do NOT include an ignored path in the ordinary
# git add call: with `set -e`, Git exits non-zero and the push never happens.
git add \
  data/current/ncaapredictions.csv \
  data/current/predictiontracker_mirror_status.json \
  data/derived/predictiontracker_source_status.json

git add -f data/snapshots/predictiontracker/mirror

if ! git diff --cached --quiet; then
  git commit -m "data: PredictionTracker ${SEASON} week ${WEEK} refresh"
  git push origin main
  echo "Pushed refreshed PredictionTracker mirror to GitHub."
else
  echo "PredictionTracker source is unchanged; nothing new to push."
fi

# Fail loudly if the two files the Connect app requires are not actually tracked.
for required in \
  data/current/ncaapredictions.csv \
  data/current/predictiontracker_mirror_status.json
do
  if ! git ls-files --error-unmatch "$required" >/dev/null 2>&1; then
    echo "ERROR: required mirror file is not tracked by Git: $required" >&2
    exit 1
  fi
done

# Raw GitHub may take a few seconds to expose a just-pushed commit.  Verify the
# exact endpoint the Connect app uses before declaring the sync ready.
RAW_META="https://raw.githubusercontent.com/manulbets-web/ncaaf-consensus-lab/main/data/current/predictiontracker_mirror_status.json"
verified=0
for attempt in 1 2 3 4 5 6; do
  if curl -fsSL -H 'Cache-Control: no-cache' "${RAW_META}?_verify=$(date +%s)" >/dev/null 2>&1; then
    verified=1
    break
  fi
  sleep 5
done
if [[ "$verified" -eq 1 ]]; then
  echo "Verified: GitHub raw mirror metadata is reachable."
else
  echo "WARNING: push completed, but GitHub raw mirror metadata was not reachable yet." >&2
  echo "Wait 20-30 seconds, then click Refresh PredictionTracker in the app." >&2
fi
