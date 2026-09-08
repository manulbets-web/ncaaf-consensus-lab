# NCAAF Consensus Lab v3.6.5 — Manual Cohort + Market Shelf

v3.6.5 reframes the production workflow around a fixed, interpretable cohort rather than exhaustive combination search.

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

The builder searches the source project, its parent `odds_archive/` directory, `ODDS_ARCHIVE_DIR`, or a path supplied with `--odds-archive`. If `ncaaf_rich_quotes.csv.gz` is absent but the paid `flat_quotes/` tree exists, it reconstructs the NCAAF consolidated CSV locally with no network/API calls. A v3.6.5 production build fails only if neither the consolidated file nor usable NCAAF flat quotes can be bundled.

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

The Market Shelf reports model probability, sportsbook implied probability, modeled EV at the actual American price, and the realized grade. Because many alternate rungs/books from one game are correlated, v3.6.5 also provides a **one-decision-per-game** analysis: it selects the highest modeled-EV offer without consulting the outcome, both across all priceable families and within ML/spread/team-total separately. The full offer-level summary remains descriptive context.

## Retained research backend

The v3.5.43–v3.5.45 formal walk-forward, A–E ablation, market-anchored Top-N analysis, exact combination search, and META portfolio tools are retained under Research/Legacy tabs. They no longer define the primary weekly production workflow.

## Chronological current slate

v3.6.5 enriches the PredictionTracker-authoritative current board with kickoff timestamps from ESPN's public college-football scoreboard feed. The schedule lookup is date-window based, not provider-week based, so the app does not inherit another site's week-number convention. Team names are normalized only for schedule matching; PredictionTracker still controls game membership, home/away orientation, and market line. A schedule miss leaves the game visible with `Date TBD`.

The current board, active-cohort forecast table, and game explorer are sorted by kickoff time and display Eastern Time. The local PredictionTracker refresh helper and the Connect builder both refresh/cache `data/current/current_game_schedule.csv` when possible.

## v3.6.5 exact historical cohort bet audit

The Cohort page now exposes the individual historical spread decisions behind the aggregate ATS/ROI figures. The audit always follows the currently active Patrick Core, custom manual, or assisted cohort and uses a transparent fixed rule: mean cohort home-margin forecast versus the archived main spread, with execution when `|edge| / cohort SD >= k`. Users can choose historical seasons, `k`, and the minimum number of selected cohort models that must have posted for a game.

Each qualifying row shows the exact matchup, bet side and line, market spread, cohort fair spread, edge, cohort SD, edge/SD, number of selected models actually available, final score when present in the bundled PredictionTracker history, realized margin versus the bet line, W/L/P outcome, -110 flat-risk unit result, the exact models used, and the archived line source. The table is filterable and downloadable as CSV. This is an audit surface only; it does not search for or optimize a cohort or threshold.
