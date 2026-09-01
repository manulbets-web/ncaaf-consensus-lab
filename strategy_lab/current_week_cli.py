#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine import load_strategy_data
from current_week import refresh_and_build_current_week


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument(
        "--refresh-cfbpicker", action="store_true",
        help="Explicitly run the slow Tableau collector; otherwise use the season/week cache.",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    config_path = root / "data/strategy/current_week_selection.json"
    if not config_path.exists():
        raise SystemExit(
            "No saved current-week selection. Save one from the "
            "Strategy Lab page first."
        )
    saved = json.loads(config_path.read_text())

    season = int(args.season or saved["season"])
    week = int(args.week if args.week is not None else saved["week"])
    k = float(saved.get("primary_k", 1.50))
    min_n = int(saved.get("min_available_models", 4))

    combos = []
    for i, c in enumerate(saved.get("combinations") or [], start=1):
        ids = [str(x) for x in c.get("model_ids", []) if str(x)]
        if ids:
            combos.append({"rank": int(c.get("rank", i)), "model_ids": ids})
    if not combos:
        ids = list(map(str, saved.get("model_ids", [])))
        if ids:
            combos = [{"rank": 1, "model_ids": ids}]
    if not combos:
        raise SystemExit("Saved selection contains no usable model combination.")

    data, registry, _ = load_strategy_data(root)
    if len(registry):
        name_map = dict(
            zip(
                registry["canonical_model_id"].astype(str),
                registry["model_name"].astype(str),
            )
        )
    else:
        name_map = dict(
            zip(
                data["canonical_model_id"].astype(str),
                data["model_name"].astype(str),
            )
        )

    portfolio_boards = []
    portfolio_plays = []
    for i, combo in enumerate(combos):
        ids = combo["model_ids"]
        result = refresh_and_build_current_week(
            root,
            data,
            ids,
            name_map,
            season=season,
            week=week,
            primary_k=k,
            min_available_models=min(min_n, len(ids)),
            refresh=(not args.cached_only and i == 0),
            include_cfbpicker=True,
            refresh_cfbpicker=bool(args.refresh_cfbpicker),
            # Preserve legacy outputs for a single set; portfolio gets its own files below.
            write_outputs=(len(combos) == 1),
        )
        board = result["board"].copy()
        plays = result["qualifying"].copy()
        board["combo_rank"] = combo["rank"]
        board["combo_models"] = "|".join(ids)
        plays["combo_rank"] = combo["rank"]
        plays["combo_models"] = "|".join(ids)
        portfolio_boards.append(board)
        portfolio_plays.append(plays)

        print("\n" + "=" * 72)
        print(f"COMBO RANK {combo['rank']} · {len(ids)} MODELS · k={k:.2f}")
        print("=" * 72)
        print(", ".join(name_map.get(x, x) for x in ids))
        print(f"games={len(board)} qualifying={len(plays)}")
        if len(plays):
            cols = [
                "away", "home", "market_home_margin",
                "consensus_home_margin", "model_sd", "edge_home",
                "signal_sd", "available_models", "bet_side",
            ]
            print(plays[cols].to_string(index=False))
        else:
            print("No games currently clear this combination's signal threshold.")

    if len(combos) > 1:
        output = root / "outputs/current_week"
        output.mkdir(parents=True, exist_ok=True)
        all_board = pd.concat(portfolio_boards, ignore_index=True, sort=False)
        all_plays = pd.concat(portfolio_plays, ignore_index=True, sort=False)
        board_path = output / "current_combo_portfolio_board.csv"
        plays_path = output / "current_combo_portfolio_qualifying.csv"
        all_board.to_csv(board_path, index=False)
        all_plays.to_csv(plays_path, index=False)
        print("\nPortfolio outputs:")
        print(board_path)
        print(plays_path)
    else:
        print("\nOutputs:")
        print(root / "outputs/current_week/current_consensus_board.csv")
        print(root / "outputs/current_week/current_qualifying_plays.csv")


if __name__ == "__main__":
    main()
