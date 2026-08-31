#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/manulbets-web/ncaaf-consensus-lab.git"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not on PATH." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
fi

git branch -M main
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

echo "Preflight: verifying v3.5.21 Patrick preset..."
python - <<'PYVERIFY'
from pathlib import Path
p = Path("strategy_lab/app.py")
s = p.read_text(encoding="utf-8")
expected = [
    "PATRICK_HOLDOUT_WEEKS = 6",
    "PATRICK_MIN_SIZE = 3",
    "PATRICK_MAX_SIZE = 6",
    "PATRICK_FINALISTS = 50",
    "PATRICK_POOL_N = 35",
    'PATRICK_POOL_METRIC = "wilson"',
    "PATRICK_POOL_MIN_BETS = 25",
    "PATRICK_MIN_AVAILABLE = 3",
    "PATRICK_MIN_SEARCH_BETS = 50",
    'PATRICK_RANK_METRIC = "ats"',
    'PATRICK_OVERLAP_THRESHOLD = 0.50',
    'committee_model_exposure_table',
    'committee_line_reference_table',
]
missing = [x for x in expected if x not in s]
if missing:
    raise SystemExit("REFUSING TO PUSH: stale Patrick preset detected:\n  " + "\n  ".join(missing))
refresh_expected = [
    "cached prior-week rows were NOT used",
    "Download prospective snapshots",
    "upcoming_refresh_audit",
]
current_week = Path("strategy_lab/current_week.py").read_text(encoding="utf-8")
scraper = Path("scripts/scrape_predictiontracker.py").read_text(encoding="utf-8")
refresh_missing = [x for x in refresh_expected if x not in s and x not in current_week and x not in scraper]
if "save_prospective_current_week_snapshot" not in current_week:
    refresh_missing.append("save_prospective_current_week_snapshot")
if "PT_MIRROR_CSV_URL" not in current_week:
    refresh_missing.append("GitHub mirror fallback")
if refresh_missing:
    raise SystemExit("REFUSING TO PUSH: stale v3.5.21 refresh code detected:\n  " + "\n  ".join(refresh_missing))
print("Verified: Top 35 | Wilson | min 25 | sizes 3–6 | ATS rank | 50 finalists | Jaccard 0.50 | exposure audit | open/midweek/close grading | strict PT mirror + snapshots")
PYVERIFY

git add .
if ! git diff --cached --quiet; then
  git commit -m "Deploy NCAAF Consensus Lab v3.5.21"
else
  echo "No new changes to commit."
fi

echo "Pushing to $REPO_URL"
git push -u origin main

echo
echo "GitHub push complete. In Posit Connect Cloud, publish from GitHub:"
echo "  repository: manulbets-web/ncaaf-consensus-lab"
echo "  branch:     main"
echo "  primary:    app.py"
