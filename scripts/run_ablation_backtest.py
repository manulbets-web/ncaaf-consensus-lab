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
from ablation_backtest import run_ablation_walkforward_backtest, save_ablation_backtest_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the formal A-E chronological model-ablation backtest.")
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument("--min-prior-weeks", type=int, default=24)
    ap.add_argument("--min-games-week", type=int, default=10)
    ap.add_argument("--pool-n", type=int, default=35)
    ap.add_argument("--combo-search-anchor", type=float, default=0.75)
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

    result = run_ablation_walkforward_backtest(
        data, line_history, model_name_map,
        oos_blocks=args.blocks, oos_block_size=args.block_size,
        min_discovery_periods=args.min_prior_weeks, min_games_per_period=args.min_games_week,
        pool_n=args.pool_n, combo_search_anchor=args.combo_search_anchor,
        max_combinations=args.max_combinations, progress_callback=progress,
    )
    written = save_ablation_backtest_outputs(result, root)
    print("\nAblation backtest complete")
    for key, path in written.items():
        print(f"  {key}: {path}")
    print("\nA-E architecture summary (k=0)")
    print(result["architecture_summary"].to_string(index=False))
    print("\nFrozen threshold surface — Strategy E / PT Updated-final")
    ts = result["threshold_surface"]
    q = ts[(ts["architecture"].astype(str) == "E") & (ts["line_reference"].astype(str) == "Close (PT Updated/final)")].copy()
    print(q.to_string(index=False))
    print("\nThreshold monotonicity checks")
    print(result["threshold_monotonicity"].to_string(index=False))
    print("\nQC complementary-side check")
    print(result["qc_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
