# NCAAF Consensus Lab v3.6.2 — Manual Cohort + Market Shelf

This directory was generated from:

    /Users/mgelbach/Downloads/ncaaf-consensus

Primary file for Connect Cloud: `app.py`
Python version: 3.11 or 3.12 recommended.

The deployment is public-session safe: strategy selections and alternate-line overrides are kept in each Shiny session.
PredictionTracker currently blocks Posit Connect Cloud worker IPs. The deployed app therefore falls back to a strict season/week-tagged GitHub mirror refreshed from the user's Mac.
Use `./refresh_predictiontracker_local_and_push.sh SEASON WEEK` whenever PredictionTracker changes; the helper also persists unique source snapshots in Git.

The production workflow is now Cohort → Current Slate → Market Shelf → Forecast. A hand-curated Patrick Core cohort is the default; the Assisted Cohort tool provides a constrained alternative that ranks individual models and greedily removes highly correlated near-duplicates without enumerating arbitrary subsets. The paid historical NCAAF Odds API archive is bundled into the deployed repo and is directly browsable on the Market Shelf page. Full-game ML, main/alternate spreads, totals, and team/alternate-team totals are retained. The previous combination/META, market-signal, formal chronological, ablation, and line-movement engines remain available under Research/Legacy tabs as diagnostic evidence rather than the primary production workflow. Andrew Percival's CFB Picker integration is retained with the latest verified PredictionTracker board authoritative for live game membership and PredictionTracker-first same-model/game de-duplication.

## CFB Picker historical enrichment

The Tableau CSV/workbook route is not the production transport. First collect history locally through the proven Embedding API + L# tooltip path, then import the resulting cache:

    python scripts/scrape_cfbpicker_history_api.py       --root "$HOME/Downloads/ncaaf-consensus"       --year 2026 --discover-only --headed --system-chrome

    python scripts/scrape_cfbpicker_history_api.py       --root "$HOME/Downloads/ncaaf-consensus"       --year 2025 --weeks 1 --pickers "CFB Geek"       --headed --system-chrome --strict --force

After collecting the desired seasons with `--all-models`, audit and import the cache with `scripts/scrape_cfbpicker_history.py`. The importer maps overlapping systems onto existing canonical IDs, appends only missing model/game observations, and writes overlap/orientation/unmatched diagnostics under `data/derived/`.

For the live slate, refresh locally with:

    ./refresh_cfbpicker_local_and_push.sh 2026 2


## Paid NCAAF Odds API archive

v3.6.2 intentionally publishes the collected NCAAF historical sportsbook data with the website. The builder stores `ncaaf_rich_quotes.csv.gz` in `data/odds/` so the paid archive stays below GitHub's single-file limit. If the earlier paid harvest has `flat_quotes/` but no consolidated CSV, the builder reconstructs and gzip-compresses the NCAAF CSV locally with no API calls or credits. It refuses to complete a production build only when neither form of the paid archive is available.

Automatic discovery checks the harvester's normal locations under the source project and its parent directory. To specify it explicitly:

    python prepare_connect_cloud_repo.py       --root "$HOME/Downloads/ncaaf-consensus"       --out "$HOME/Downloads/ncaaf-connect-cloud"       --odds-archive "$HOME/Downloads/odds_archive"       --force

The deployed Market Shelf page exposes full-game moneyline, main/alternate spreads, totals, and team/alternate-team totals from the archive. Moneyline is handled as the ±0.5 endpoint of the margin ladder.

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
