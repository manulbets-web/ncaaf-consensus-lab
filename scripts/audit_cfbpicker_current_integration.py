#!/usr/bin/env python3
"""Audit the incremental current-week coverage added by CFB Picker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "strategy_lab"
sys.path.insert(0, str(LAB))

from current_week import (  # noqa: E402
    build_current_source_mapping,
    game_key_from_names,
    load_cfbpicker_current,
    load_predictiontracker_current,
    predictiontracker_master_slate,
    completed_game_keys_from_history,
)
from engine import load_strategy_data  # noqa: E402


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["game_join_key"] = pd.Series(dtype=str)
        out["model_game_key"] = pd.Series(dtype=str)
        return out
    out["game_join_key"] = [
        game_key_from_names(a, h)
        for a, h in zip(out["away"], out["home"])
    ]
    out["model_game_key"] = (
        out["game_join_key"].astype(str) + "||"
        + out["canonical_model_id"].astype(str)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    history, registry, _ = load_strategy_data(root)
    historical_ids = set(history["canonical_model_id"].dropna().astype(str))
    name_map = dict(zip(
        history["canonical_model_id"].astype(str),
        history["model_name"].astype(str),
    ))
    if len(registry):
        name_map.update(dict(zip(
            registry["canonical_model_id"].astype(str),
            registry["model_name"].astype(str),
        )))

    cfb = add_keys(load_cfbpicker_current(
        root, season=int(args.season), week=int(args.week)
    ))
    if cfb.empty:
        raise SystemExit("No matching CFB Picker current cache was found.")
    name_map.update(dict(zip(
        cfb["canonical_model_id"].astype(str), cfb["model_name"].astype(str)
    )))
    all_ids = list(dict.fromkeys([
        *sorted(historical_ids), *cfb["canonical_model_id"].astype(str).tolist()
    ]))
    mapping = build_current_source_mapping(history, all_ids, name_map)
    pt, _ = load_predictiontracker_current(root, mapping)
    pt = add_keys(pt)

    raw_cfb_rows = int(cfb["model_game_key"].nunique())
    raw_cfb_games = int(cfb["game_join_key"].nunique())
    master = predictiontracker_master_slate(root)
    completed_keys = completed_game_keys_from_history(history, season=int(args.season))
    if len(master) and completed_keys:
        master = master[~master["game_join_key"].astype(str).isin(completed_keys)].copy()
    live_keys = set(master["game_join_key"].astype(str)) if len(master) else set()

    if completed_keys:
        cfb = cfb[~cfb["game_join_key"].astype(str).isin(completed_keys)].copy()
        pt = pt[~pt["game_join_key"].astype(str).isin(completed_keys)].copy()
    if live_keys:
        cfb = cfb[cfb["game_join_key"].astype(str).isin(live_keys)].copy()
        pt = pt[pt["game_join_key"].astype(str).isin(live_keys)].copy()

    pt_keys = set(pt["model_game_key"].astype(str)) if len(pt) else set()
    cfb["predictiontracker_overlap"] = cfb["model_game_key"].astype(str).isin(pt_keys)
    cfb["incremental_after_pt_first_dedup"] = ~cfb["predictiontracker_overlap"]
    cfb["historically_eligible_model"] = cfb["canonical_model_id"].astype(str).isin(
        historical_ids
    )

    by_model = (
        cfb.groupby(["canonical_model_id", "model_name"], as_index=False)
        .agg(
            cfb_rows=("model_game_key", "nunique"),
            pt_overlap_rows=("predictiontracker_overlap", "sum"),
            incremental_rows=("incremental_after_pt_first_dedup", "sum"),
            historically_eligible=("historically_eligible_model", "max"),
        )
        .sort_values(["incremental_rows", "cfb_rows", "model_name"], ascending=[False, False, True])
    )

    summary = {
        "season": int(args.season),
        "week": int(args.week),
        "raw_cfbpicker_rows": raw_cfb_rows,
        "raw_cfbpicker_games": raw_cfb_games,
        "cfbpicker_rows": int(cfb["model_game_key"].nunique()),
        "cfbpicker_models": int(cfb["canonical_model_id"].nunique()),
        "cfbpicker_games": int(cfb["game_join_key"].nunique()),
        "cfbpicker_rows_excluded_off_live_slate": int(raw_cfb_rows - cfb["model_game_key"].nunique()),
        "cfbpicker_games_excluded_off_live_slate": int(raw_cfb_games - cfb["game_join_key"].nunique()),
        "live_predictiontracker_games": int(len(master)),
        "graded_current_season_games_known": int(len(completed_keys)),
        "predictiontracker_rows": int(pt["model_game_key"].nunique()) if len(pt) else 0,
        "overlap_rows_pt_preferred": int(cfb["predictiontracker_overlap"].sum()),
        "incremental_cfbpicker_rows": int(cfb["incremental_after_pt_first_dedup"].sum()),
        "incremental_historically_eligible_rows": int((
            cfb["incremental_after_pt_first_dedup"]
            & cfb["historically_eligible_model"]
        ).sum()),
        "live_only_rows_logged_not_selected": int((
            ~cfb["historically_eligible_model"]
        ).sum()),
        "live_only_models_logged_not_selected": int(
            cfb.loc[~cfb["historically_eligible_model"], "canonical_model_id"].nunique()
        ),
    }

    derived = root / "data/derived"
    derived.mkdir(parents=True, exist_ok=True)
    json_path = derived / f"cfbpicker_current_integration_audit_{args.season}_week_{args.week:02d}.json"
    csv_path = derived / f"cfbpicker_current_incremental_by_model_{args.season}_week_{args.week:02d}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_model.to_csv(csv_path, index=False)
    print(json.dumps({
        **summary,
        "summary_output": str(json_path.relative_to(root)),
        "by_model_output": str(csv_path.relative_to(root)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
