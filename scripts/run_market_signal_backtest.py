#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in [str(HERE / "strategy_lab"), str(HERE)]:
    if p not in sys.path: sys.path.insert(0, p)

from engine import load_strategy_data
from committee import load_predictiontracker_line_history
from market_signal import (
    TOP_N_GRID, run_market_signal_walkforward, save_market_signal_outputs,
    load_live_bets_reference, match_live_spread_bets, live_forensics_summary,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run market-anchored Top-N walk-forward signal backtest and optional 2025 live-process forensics.")
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument("--min-prior-weeks", type=int, default=24)
    ap.add_argument("--min-model-bets", type=int, default=25)
    ap.add_argument("--live-bets", type=Path, default=None, help="Optional cleaned CSV with Date, Bet, Win/Loss columns.")
    ap.add_argument("--forensics-top-n", type=int, default=10)
    ap.add_argument("--forensics-ranking", choices=["wilson","mae","incremental"], default="wilson")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    data, _, _ = load_strategy_data(root)
    lines = load_predictiontracker_line_history(root, data)
    def progress(done, total, label):
        print(f"[{100*done/total:6.2f}%] {label}", flush=True)
    result = run_market_signal_walkforward(
        data, lines, oos_blocks=args.blocks, oos_block_size=args.block_size,
        min_discovery_periods=args.min_prior_weeks, min_model_bets=args.min_model_bets,
        progress_callback=progress,
    )
    live_path = args.live_bets
    if live_path is None:
        candidate = root / "data" / "reference" / "live_bets_2025_reference.csv"
        if candidate.exists(): live_path = candidate
    if live_path is not None and live_path.exists():
        live = load_live_bets_reference(live_path)
        matched, unmatched = match_live_spread_bets(
            live, result["oos_predictions"], ranking_method=args.forensics_ranking,
            top_n=args.forensics_top_n, line_reference="Close (PT Updated/final)",
        )
        result["live_matches"] = matched; result["live_unmatched"] = unmatched; result["live_forensics"] = live_forensics_summary(matched)
    written = save_market_signal_outputs(result, root)
    print("\nMarket-anchored signal backtest complete")
    for k, v in written.items(): print(f"  {k}: {v}")
    fs = result["forecast_summary"].copy()
    close = fs[fs["line_reference"].astype(str).eq("Close (PT Updated/final)")].sort_values(["ranking_method","mae_improvement"], ascending=[True,False])
    print("\nPT Updated/final — forecast summary")
    print(close.to_string(index=False))
    if "live_matches" in result:
        print(f"\n2025 live spread matches: {len(result['live_matches'])} matched; {len(result['live_unmatched'])} unmatched")
        print(result["live_forensics"].to_string(index=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
