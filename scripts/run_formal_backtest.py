#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LAB = HERE / "strategy_lab"
for p in [str(LAB), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import load_strategy_data
from committee import load_predictiontracker_line_history
from formal_backtest import run_formal_walkforward_backtest, save_formal_backtest_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the formal chronological NCAAF strategy backtest.")
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--blocks", type=int, default=8, help="Number of non-overlapping OOS blocks")
    ap.add_argument("--block-size", type=int, default=6, help="Usable weeks per OOS block; use 1 for strict weekly reselection")
    ap.add_argument("--min-prior-weeks", type=int, default=24)
    ap.add_argument("--min-games-week", type=int, default=10)
    ap.add_argument("--pool-n", type=int, default=35)
    ap.add_argument("--max-combinations", type=int, default=10_000_000)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    data, registry, _ = load_strategy_data(root)
    if registry is not None and len(registry):
        model_name_map = dict(zip(registry["canonical_model_id"].astype(str), registry["model_name"].astype(str)))
    else:
        z = data[["canonical_model_id", "model_name"]].drop_duplicates("canonical_model_id")
        model_name_map = dict(zip(z["canonical_model_id"].astype(str), z["model_name"].astype(str)))
    line_history = load_predictiontracker_line_history(root, data)

    def progress(done, total, label):
        pct = 100.0 * done / total if total else 0.0
        print(f"[{pct:6.2f}%] {label}", flush=True)

    result = run_formal_walkforward_backtest(
        data, line_history, model_name_map,
        oos_blocks=args.blocks, oos_block_size=args.block_size,
        min_discovery_periods=args.min_prior_weeks, min_games_per_period=args.min_games_week,
        pool_n=args.pool_n, max_combinations=args.max_combinations,
        progress_callback=progress,
    )
    written = save_formal_backtest_outputs(result, root)
    print("\nFormal backtest complete")
    for key, path in written.items():
        print(f"  {key}: {path}")
    fixed = result.get("fixed_anchor_surface")
    adaptive = result.get("adaptive_summary")
    if fixed is not None and len(fixed):
        print("\nFixed-anchor OOS surface")
        print(fixed.to_string(index=False))
    if adaptive is not None and len(adaptive):
        print("\nAdaptive-anchor OOS")
        print(adaptive.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
