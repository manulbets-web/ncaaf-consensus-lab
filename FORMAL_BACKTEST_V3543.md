# v3.5.43 formal chronological validation

Production UI adds a compact Validation tab while retaining the v3.5.42 streamlined workflow.

- PredictionTracker Updated/final is the execution reference.
- Historical model availability is reconstructed from models posting in the first OOS week.
- Candidate ranking: prior-data Wilson lower bound, Top 35, minimum 25 bets.
- Exact combination sizes: 3–6; Top 50 retained by prior-data ATS.
- Search-anchor grid: 0.00–2.00 SD in 0.25 increments.
- Every anchor gets an independent finalist portfolio inside every untouched OOS block.
- Adaptive anchor uses only earlier OOS blocks: highest pooled Wilson lower bound with >=50 prior bets; 0.75 fallback.
- Existing overlap-community and diversified META threshold tuning remain discovery-only.
- Staking: flat 1u risk and risk to win 1u.
- Outputs: fixed-anchor surface, adaptive path, cumulative OOS equity, drawdown, and META standardized-edge calibration.
- `scripts/run_formal_backtest.py` caches results under `data/derived/formal_backtest_*`.
