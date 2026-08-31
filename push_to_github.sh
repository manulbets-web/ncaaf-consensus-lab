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

# v3.5.25: release rebuilds must never delete the verified PredictionTracker
# mirror. If the working tree is missing mirror metadata, recover the newest
# historical commit that still contains it, together with the matching CSV and
# prospective mirror snapshots. This also self-heals repos affected by v3.5.21.
recover_latest_pt_mirror() {
  local meta="data/current/predictiontracker_mirror_status.json"
  [[ -f "$meta" ]] && return 0
  [[ -d .git ]] || return 0

  local commit=""
  while IFS= read -r c; do
    if git cat-file -e "${c}:${meta}" 2>/dev/null; then
      commit="$c"
      break
    fi
  done < <(git rev-list HEAD --all -- "$meta" 2>/dev/null || true)

  if [[ -z "$commit" ]]; then
    echo "No prior PredictionTracker mirror found in Git history; continuing without one."
    return 0
  fi

  echo "Recovering PredictionTracker mirror state from ${commit:0:12}..."
  for path in \
    data/current/ncaapredictions.csv \
    data/current/predictiontracker_mirror_status.json \
    data/derived/predictiontracker_source_status.json
  do
    if git cat-file -e "${commit}:${path}" 2>/dev/null; then
      mkdir -p "$(dirname "$path")"
      git show "${commit}:${path}" > "$path"
    fi
  done

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    mkdir -p "$(dirname "$path")"
    git show "${commit}:${path}" > "$path"
  done < <(git ls-tree -r --name-only "$commit" -- data/snapshots/predictiontracker/mirror 2>/dev/null || true)
}

recover_latest_pt_mirror

# Validate that mirror metadata and current CSV travel together.
python - <<'PYMIRROR'
from pathlib import Path
import hashlib, json
meta = Path("data/current/predictiontracker_mirror_status.json")
csv = Path("data/current/ncaapredictions.csv")
if meta.exists():
    if not csv.exists():
        raise SystemExit("REFUSING TO PUSH: PredictionTracker mirror metadata exists but current CSV is missing.")
    m = json.loads(meta.read_text(encoding="utf-8"))
    expected = str(m.get("canonical_sha256") or "").strip()
    actual = hashlib.sha256(csv.read_bytes()).hexdigest()
    if expected and actual != expected:
        raise SystemExit(
            "REFUSING TO PUSH: PredictionTracker mirror metadata/CSV hash mismatch.\n"
            f"  metadata: {expected[:12]}\n  csv:      {actual[:12]}\n"
            "Run ./refresh_predictiontracker_local_and_push.sh SEASON WEEK before deploying."
        )
    print(
        "Verified PredictionTracker mirror: "
        f"season={m.get('season')} week={m.get('week')} rows={m.get('rows')} hash={actual[:12]}"
    )
else:
    print("PredictionTracker mirror metadata not present; direct cloud refresh may remain blocked until a local mirror is created.")
PYMIRROR

echo "Preflight: verifying v3.5.25 Patrick preset..."
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
    '5 · Line Movement',
    'line_fixed_reprice_table',
    'line_migration_summary_table',
    'run_line_pipelines',
    'line_pipeline_cross_matrix_table',
    'line_pipeline_fixed_reprice_table',
    'line_pipeline_model_selection_table',
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
    raise SystemExit("REFUSING TO PUSH: stale v3.5.25 refresh code detected:\n  " + "\n  ".join(refresh_missing))
line_path = Path("strategy_lab/line_movement.py")
if not line_path.exists():
    raise SystemExit("REFUSING TO PUSH: strategy_lab/line_movement.py is missing")
line_text = line_path.read_text(encoding="utf-8")
line_missing = [x for x in [
    "fixed_bet_repricing", "bet_set_overlap", "signal_migration_detail",
    "opening_line_qc", "clean_line_history_for_analysis",
    "classify_open_line_anomalies", "run_line_specific_pipelines",
    "cross_reference", "fixed_repricing", "model_selection_comparison"
] if x not in line_text]
for x in ["line_active_strategy_status", '"discovery_periods": tuple(search_periods)', 'result["search_periods"]', "analysis_line_history()"]:
    if x not in s:
        line_missing.append(x)
if line_missing:
    raise SystemExit("REFUSING TO PUSH: stale v3.5.25 line-movement module detected:\n  " + "\n  ".join(line_missing))
print("Verified: Top 35 | Wilson | min 25 | sizes 3-6 | ATS rank | 50 finalists | Jaccard 0.50 | exposure audit | Open-line anomaly QC | frozen usable-week split | fixed-bet repricing | signal migration/CLV | independent line pipelines | 3x3 architecture/execution matrix | pipeline price decay | cross-pipeline model weights | strict PT mirror + snapshots")
PYVERIFY

git add .
if ! git diff --cached --quiet; then
  git commit -m "Deploy NCAAF Consensus Lab v3.5.25"
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
