# v3.5.45 — Market-Anchored Signal + 2025 Live Forensics

This release adds a prior-only walk-forward test of whether public model projections contain **incremental information beyond the market line**.

## Market-anchored model

For every historical OOS block, line reference, ranking method, and Top-N pool, the discovery window fits:

`Actual margin = alpha + beta * Market + gamma * (Consensus - Market)`

The fitted coefficients are frozen before the untouched OOS block. Positive `gamma` means model disagreement contains incremental information, but values below 1 imply shrinkage toward the market.

## Pool-size grid

Top 3, 5, 8, 10, 15, 20, 25, 35, and all eligible historically posting models.

Ranking is recomputed using prior data only by:

- Wilson ATS lower bound
- margin MAE
- incremental-to-market residual correlation with sample-size shrinkage

Outputs include OOS MAE/RMSE improvement versus a market-only regression, residual correlation, gamma stability, and downstream ATS/ROI at frozen adjusted-edge cutoffs measured in points.

## 2025 reference log

`data/reference/live_bets_2025_reference.csv` is a cleaned transcription of the supplied 2025 betting sheet. It is descriptive only. Spread bets are fuzzy-matched to unique 2025 OOS games by team and spread; moneylines and ambiguous matches are excluded from spread-model forensics. No live-bet outcome is used to fit or rank the market-signal model.
