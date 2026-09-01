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

# v3.5.31: release rebuilds must never delete the verified PredictionTracker
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

cfb_meta = Path("data/current/cfbpicker_mirror_status.json")
cfb_csv = Path("data/current/cfbpicker_current_long.csv")
if cfb_meta.exists():
    if not cfb_csv.exists():
        raise SystemExit("REFUSING TO PUSH: CFB Picker mirror metadata exists but current CSV is missing.")
    cm = json.loads(cfb_meta.read_text(encoding="utf-8"))
    cfb_expected = str(cm.get("canonical_sha256") or "").strip()
    cfb_actual = hashlib.sha256(cfb_csv.read_bytes()).hexdigest()
    if cfb_expected and cfb_actual != cfb_expected:
        raise SystemExit(
            "REFUSING TO PUSH: CFB Picker mirror metadata/CSV hash mismatch.\n"
            f"  metadata: {cfb_expected[:12]}\n  csv:      {cfb_actual[:12]}\n"
            "Publish the verified cache with scripts/scrape_cfbpicker_current.py --from-cache."
        )
    print(
        "Verified CFB Picker mirror: "
        f"season={cm.get('season')} week={cm.get('week')} rows={cm.get('rows')} hash={cfb_actual[:12]}"
    )
else:
    print("CFB Picker mirror metadata not present; current boards will use PredictionTracker only.")
PYMIRROR

echo "Preflight: verifying v3.5.41 Patrick preset + CFB Picker API transport..."
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
    'run_rolling_line_validation',
    'rolling_line_current_aggregate_table',
    'rolling_line_paired_table',
    'run_forward_stability',
    'forward_stability_aggregate_table',
    'forward_stability_finalists_table',
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
    raise SystemExit("REFUSING TO PUSH: stale v3.5.41 refresh code detected:\n  " + "\n  ".join(refresh_missing))
line_path = Path("strategy_lab/line_movement.py")
if not line_path.exists():
    raise SystemExit("REFUSING TO PUSH: strategy_lab/line_movement.py is missing")
line_text = line_path.read_text(encoding="utf-8")
line_missing = [x for x in [
    "fixed_bet_repricing", "bet_set_overlap", "signal_migration_detail",
    "opening_line_qc", "clean_line_history_for_analysis",
    "classify_open_line_anomalies", "run_line_specific_pipelines",
    "cross_reference", "fixed_repricing", "model_selection_comparison",
    "run_rolling_line_selection_validation", "build_rolling_validation_folds",
    "paired_updated", "current_line_blocks",
    "run_forward_stability_validation", "FORWARD_STABILITY_METHODS",
    "stable_score", "recency_score"
] if x not in line_text]
for x in ["line_active_strategy_status", '"discovery_periods": tuple(search_periods)', 'result["search_periods"]', "analysis_line_history()"]:
    if x not in s:
        line_missing.append(x)
if line_missing:
    raise SystemExit("REFUSING TO PUSH: stale v3.5.41 line-movement module detected:\n  " + "\n  ".join(line_missing))
fs_start = s.find("async def forward_stability_task(")
fs_end = s.find("@reactive.effect", fs_start)
fs_body = s[fs_start:fs_end] if fs_start >= 0 and fs_end > fs_start else ""
if "input." in fs_body:
    raise SystemExit("REFUSING TO PUSH: forward_stability_task reads Shiny reactive input inside Extended Task")
for token in ["min_discovery: int, min_games: int", "min_discovery_periods=int(min_discovery)", "min_games_per_period=int(min_games)"]:
    if token not in s:
        raise SystemExit(f"REFUSING TO PUSH: v3.5.41 Extended Task hotfix marker missing: {token}")
cfb_files = {
    "scripts/scrape_cfbpicker_history_api.py": ["TableauViz", 'activeSheet, "Year "', "picker_items_from_objects"],
    "scripts/cfbpicker_tooltip_legacy.py": ["click_and_read_tooltip_response", "collect_header_rows"],
    "scripts/scrape_cfbpicker_history.py": ["cfbpicker_overlap_audit.csv", "cfbpicker_orientation_audit.csv", "existing_keys", "tableau_embedding_api_tooltip"],
    "scripts/scrape_cfbpicker_current_api.py": [
        "requestedComplete", "parse_current_tooltip",
        "collectable_pickers_", "strategy_selected",
        "cfbpicker_live_model_mapping_", "return [None]",
        "const completeVerified=!hasCompleteRequest||",
        "master_slate_match", "resolve_labeled_team_side",
        'record["row_errors"]', r"Games\s*:\s*None",
    ],
    "scripts/scrape_cfbpicker_current.py": ["MIRROR_META_URL", "cfbpicker_current_long.csv", "scrape_cfbpicker_current_api.py", "--from-cache"],
    "scripts/audit_cfbpicker_current_integration.py": ["incremental_after_pt_first_dedup", "overlap_rows_pt_preferred"],
}
for path, tokens in cfb_files.items():
    pp = Path(path)
    if not pp.exists():
        raise SystemExit(f"REFUSING TO PUSH: missing CFB Picker integration file {path}")
    text = pp.read_text(encoding="utf-8")
    for token in tokens:
        if token not in text:
            raise SystemExit(f"REFUSING TO PUSH: stale CFB Picker integration marker missing in {path}: {token}")
if "load_cfbpicker_current(root, season=int(season), week=int(week))" not in current_week:
    raise SystemExit("REFUSING TO PUSH: current CFB Picker cache is not season/week filtered")
for token in [
    'source_order = {"predictiontracker": 0, "cfbpicker": 1}',
    "refresh_cfbpicker: bool = True",
    "completed_game_keys_from_history",
    "cfbpicker_rows_excluded_off_slate",
    "latest verified PredictionTracker board is the live game-universe",
]:
    if token not in current_week:
        raise SystemExit(f"REFUSING TO PUSH: current CFB Picker merge marker missing: {token}")
for token in ["include_cfbpicker=True", "refresh_cfbpicker=False", "current_cfbpicker_model_map"]:
    if token not in s:
        raise SystemExit(f"REFUSING TO PUSH: app does not consume the deployed CFB Picker cache: {token}")
cfb_refresh_helper = Path("refresh_cfbpicker_local_and_push.sh").read_text(encoding="utf-8")
if "refresh_predictiontracker_mirror.py" not in cfb_refresh_helper:
    raise SystemExit("REFUSING TO PUSH: CFB Picker refresh does not refresh PredictionTracker first")
print("Verified: Top 35 | Wilson | min 25 | sizes 3-6 | ATS rank | 50 finalists | Jaccard 0.50 | exposure audit | line movement | rolling OOS | nested forward stability | CFB Picker live-slate filter + PT-first de-dup | strict PT/CFB mirrors")
PYVERIFY

git add .
if ! git diff --cached --quiet; then
  git commit -m "Deploy NCAAF Consensus Lab v3.5.41"
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
