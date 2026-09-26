# NCAAF Consensus Lab v3.6.8 — Manual Cohort + Market Shelf

## v3.6.8 Patrick preferred strategy

## Corrected two-stage combination search

v3.6.8 fixes the v3.6.7 ordering error that reduced the candidate pool to ~15 models before the exact combination search. The preferred recipe now keeps the full ATS/high-volume candidate pool for Stage 1, evaluates every 3–6 model subset, then uses only discovery results to form a ~15-model compatibility core based on repeated participation in the strongest retained combinations plus discovery-only edge-correlation redundancy. A second exact 3–6 search confirms combinations within that core. The 15-week holdout remains untouched until the core and confirmation finalists are frozen.

For 35 eligible candidates, Stage 1 contains 2,006,697 exact 3–6 model combinations; a full 15-model core adds up to 9,828 confirmation combinations.

The one-click Patrick recipe emphasizes volume, chronological validation, and redundancy control without prematurely shrinking the search universe. It first considers up to 43 currently-posting models, ranked by discovery ATS with at least 60 discovery bets per model. The latest 15 usable completed weeks are reserved as untouched holdout. Every 3–6 model combination in the broad eligible discovery pool is then evaluated at the 0.75-SD anchor with at least 250 discovery bets. Models are ranked by repeated participation in the strongest discovery combinations; discovery-only model-minus-market edge correlation then removes obvious near-duplicates to form a target core of about 15. A second exact 3–6 search within that core freezes the top 60 confirmation finalists for the existing finalist/META analysis, after which holdout performance is finally revealed.

The Legacy · META Picks page now exposes spread-regime diagnostics for the frozen META portfolio using bins 0–3.5, 4–7.5, 8–14.5, 15–21.5, 22–27.5, 28–34.5, and 35+. Discovery and holdout are shown separately, with forecast error and betting results by regime. The exact historical META ledger can also be filtered by spread regime. Current 22+ point lines are flagged as blowout-like for review, but are not automatically suppressed; this remains a diagnostic until prospective evidence supports a restriction.

v3.6.8 reframes the production workflow around a fixed, interpretable cohort rather than exhaustive combination search.

## Production workflow

1. **Use Patrick Core** with one click. The preset is resolved from the exact original hand-curated model names; unavailable models are shown explicitly instead of replaced by fuzzy guesses.
2. **Edit a custom manual cohort** in a staged editor, then explicitly apply it. The active cohort is shown globally at the top of every page.
3. Optionally use Assisted Cohort selection. Assisted selection ranks individual models and greedily removes near-duplicates above a selected historical edge-correlation ceiling. It does **not** enumerate arbitrary model subsets.
4. Review cohort stability by season, leave-one-out contribution, and 0.90 edge-correlation families. These are diagnostics for manual curation, not another subset optimizer.
5. Refresh the PredictionTracker-defined current slate, augmented by current CFB Picker models on those games, then use the game explorer to see every posted model with the active cohort highlighted.
6. Explore the bundled paid Odds API historical market shelf, including a one-decision-per-game best-expression analysis across ML, spreads, and team totals.

## Bundled Odds API data

The Connect Cloud build now requires `ncaaf_rich_quotes.csv.gz` and copies it into:

    data/odds/ncaaf_rich_quotes.csv.gz

The builder searches the source project, its parent `odds_archive/` directory, `ODDS_ARCHIVE_DIR`, or a path supplied with `--odds-archive`. If `ncaaf_rich_quotes.csv.gz` is absent but the paid `flat_quotes/` tree exists, it reconstructs the NCAAF consolidated CSV locally with no network/API calls. A v3.6.8 production build fails only if neither the consolidated file nor usable NCAAF flat quotes can be bundled.

The deployed Market Shelf includes a direct archive browser plus cohort-pricing views for these full-game markets:

- moneyline (`h2h`)
- main spread
- alternate spreads
- game total / alternate total (context for a margin-only cohort)
- team total
- alternate team total

Moneyline is treated as the ±0.5 endpoint of the margin ladder for display and comparison.

## Historical pricing logic

For spread and moneyline offers, the active fixed cohort is rebuilt on each historical game. The probability of covering each sportsbook rung is estimated from the cohort's margin errors on **earlier games only**. No future games are used to calibrate a 2025 offer.

For team totals, the sportsbook game total anchors expected combined scoring. The active cohort home-margin forecast redistributes that total between the two teams:

    home_mean = (market_total + cohort_home_margin) / 2
    away_mean = (market_total - cohort_home_margin) / 2

Team-score residuals are then calibrated using **earlier matched Odds API games only**. Early-season team-total offers remain unpriced until the minimum prior residual sample is available.

The Market Shelf reports model probability, sportsbook implied probability, modeled EV at the actual American price, and the realized grade. Because many alternate rungs/books from one game are correlated, v3.6.8 also provides a **one-decision-per-game** analysis: it selects the highest modeled-EV offer without consulting the outcome, both across all priceable families and within ML/spread/team-total separately. The full offer-level summary remains descriptive context.

## Retained research backend

The v3.5.43–v3.5.45 formal walk-forward, A–E ablation, market-anchored Top-N analysis, exact combination search, and META portfolio tools are retained under Research/Legacy tabs. They no longer define the primary weekly production workflow.

## Chronological current slate

v3.6.8 enriches the PredictionTracker-authoritative current board with kickoff timestamps from ESPN's public college-football scoreboard feed. The schedule lookup is date-window based, not provider-week based, so the app does not inherit another site's week-number convention. Team names are normalized only for schedule matching; PredictionTracker still controls game membership, home/away orientation, and market line. A schedule miss leaves the game visible with `Date TBD`.

The current board, active-cohort forecast table, and game explorer are sorted by kickoff time and display Eastern Time. The local PredictionTracker refresh helper and the Connect builder both refresh/cache `data/current/current_game_schedule.csv` when possible.

## v3.6.8 exact historical cohort bet audit

The Cohort page now exposes the individual historical spread decisions behind the aggregate ATS/ROI figures. The audit always follows the currently active Patrick Core, custom manual, or assisted cohort and uses a transparent fixed rule: mean cohort home-margin forecast versus the archived main spread, with execution when `|edge| / cohort SD >= k`. Users can choose historical seasons, `k`, and the minimum number of selected cohort models that must have posted for a game.

Each qualifying row shows the exact matchup, bet side and line, market spread, cohort fair spread, edge, cohort SD, edge/SD, number of selected models actually available, final score when present in the bundled PredictionTracker history, realized margin versus the bet line, W/L/P outcome, -110 flat-risk unit result, the exact models used, and the archived line source. The table is filterable and downloadable as CSV. This is an audit surface only; it does not search for or optimize a cohort or threshold.


## v3.6.8 exact META bet audit

The Legacy · META Picks page now exposes the complete historical betting ledger for the active META portfolio. It uses the already-selected finalist combinations, overlap communities, and discovery-selected META k; it does not re-optimize those choices from outcomes. Every qualifying game includes the archived market, META fair spread, recommended side/line, standardized edge, active communities/combinations, final score, W/L/P result, and flat-risk -110 unit contribution. Discovery and Holdout are labeled separately, and the combined view is explicitly descriptive rather than fully out-of-sample. The exact ledger can be downloaded as CSV.
