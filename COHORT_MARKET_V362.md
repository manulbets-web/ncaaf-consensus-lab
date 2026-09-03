# NCAAF Consensus Lab v3.6.2 — Manual Cohort + Market Shelf

v3.6.2 reframes the production workflow around a fixed, interpretable cohort rather than exhaustive combination search.

## Production workflow

1. Choose a manual cohort (Patrick Core is initialized from the original hand-curated model list where current canonical mappings exist).
2. Optionally use Assisted Cohort selection. Assisted selection ranks individual models and greedily removes near-duplicates above a selected historical edge-correlation ceiling. It does **not** enumerate arbitrary model subsets.
3. Refresh the PredictionTracker-defined current slate, augmented by current CFB Picker models on those games.
4. Inspect the active cohort's mean/median home-margin forecast, SD, number of models posting, model leans, and raw disagreement with the current market line.
5. Explore the bundled paid Odds API historical market shelf.

## Bundled Odds API data

The Connect Cloud build now requires `ncaaf_rich_quotes.csv.gz` and copies it into:

    data/odds/ncaaf_rich_quotes.csv

The builder searches the source project, its parent `odds_archive/` directory, `ODDS_ARCHIVE_DIR`, or a path supplied with `--odds-archive`. If `ncaaf_rich_quotes.csv.gz` is absent but the paid `flat_quotes/` tree exists, it reconstructs the NCAAF consolidated CSV locally with no network/API calls. A v3.6.2 production build fails only if neither the consolidated file nor usable NCAAF flat quotes can be bundled.

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

The Market Shelf reports model probability, sportsbook implied probability, modeled EV at the actual American price, and the realized grade. The offer-level historical summary is explicitly descriptive because multiple books and alternate rungs from the same game are correlated observations.

## Retained research backend

The v3.5.43–v3.5.45 formal walk-forward, A–E ablation, market-anchored Top-N analysis, exact combination search, and META portfolio tools are retained under Research/Legacy tabs. They no longer define the primary weekly production workflow.
