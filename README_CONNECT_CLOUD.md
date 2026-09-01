# NCAAF Consensus Lab v3.5.41 — Posit Connect Cloud bundle

This directory was generated from:

    /Users/mgelbach/Downloads/ncaaf-consensus

Primary file for Connect Cloud: `app.py`
Python version: 3.11 or 3.12 recommended.

The deployment is public-session safe: strategy selections and alternate-line overrides are kept in each Shiny session.
PredictionTracker currently blocks Posit Connect Cloud worker IPs. The deployed app therefore falls back to a strict season/week-tagged GitHub mirror refreshed from the user's Mac.
Use `./refresh_predictiontracker_local_and_push.sh SEASON WEEK` whenever PredictionTracker changes; the helper also persists unique source snapshots in Git.

Page 5 includes the repeated chronological and nested forward-stability experiments. v3.5.41 integrates Andrew Percival's CFB Picker via the Tableau Embedding API + L# tooltip transport proven on 2023–2025. The latest verified PredictionTracker board is now authoritative for live game membership; CFB Picker expands model coverage only on those games, so cumulative/stale Tableau rows cannot re-enter Pages 2 or 4. Matchups already graded in the current season are also excluded defensively. PredictionTracker remains preferred when the same canonical model/game exists in both sources. Live-only models are logged prospectively but remain ineligible for historically selected strategies. The slow Tableau collection remains a Mac-side weekly refresh rather than an in-app task. v3.5.41 also bundles the smoke-test picker files, refuses missing explicit --picker-file paths instead of silently scraping all models, and extends the wrapper watchdog for full 43-model runs.

## CFB Picker historical enrichment

The Tableau CSV/workbook route is not the production transport. First collect history locally through the proven Embedding API + L# tooltip path, then import the resulting cache:

    python scripts/scrape_cfbpicker_history_api.py       --root "$HOME/Downloads/ncaaf-consensus"       --year 2026 --discover-only --headed --system-chrome

    python scripts/scrape_cfbpicker_history_api.py       --root "$HOME/Downloads/ncaaf-consensus"       --year 2025 --weeks 1 --pickers "CFB Geek"       --headed --system-chrome --strict --force

After collecting the desired seasons with `--all-models`, audit and import the cache with `scripts/scrape_cfbpicker_history.py`. The importer maps overlapping systems onto existing canonical IDs, appends only missing model/game observations, and writes overlap/orientation/unmatched diagnostics under `data/derived/`.

For the live slate, refresh locally with:

    ./refresh_cfbpicker_local_and_push.sh 2026 2


## Local smoke test

    cd "/Users/mgelbach/Downloads/ncaaf-connect-cloud"
    python -m pip install -r requirements.txt
    python -m shiny run --reload --launch-browser app.py

## GitHub repository

The deployment target is:

    https://github.com/manulbets-web/ncaaf-consensus-lab

From this generated directory, run:

    ./push_to_github.sh

The helper initializes Git if needed, commits the deployment bundle, points `origin` at the repository above, and pushes `main` without force-pushing.

## Connect Cloud

Publish from GitHub using repository `manulbets-web/ncaaf-consensus-lab`, branch `main`, and primary file `app.py`.
