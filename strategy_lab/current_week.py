from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


CFB_NAME_ALIASES = {
    "Sagarin Predictor": "Sagarin: Predictor",
    "Sagarin Golden Mean": "Sagarin: Golden",
    "Sagarin Ratings": "Sagarin",
    "Sagarin Recent": "Sagarin: Recent",
    "TeamRankings.com": "TeamRankings",
    "David Harville": "Harville",
    "Slate Fluker": "Slate Index",
}


# Connect Cloud is currently blocked by PredictionTracker (HTTP 403 from the
# cloud worker IP range).  A locally refreshed GitHub mirror is therefore the
# production fallback.  Cache-busting is applied only to GitHub, not to
# PredictionTracker itself.
PT_MIRROR_BASE = "https://raw.githubusercontent.com/manulbets-web/ncaaf-consensus-lab/main"
PT_MIRROR_CSV_URL = f"{PT_MIRROR_BASE}/data/current/ncaapredictions.csv"
PT_MIRROR_META_URL = f"{PT_MIRROR_BASE}/data/current/predictiontracker_mirror_status.json"



def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_json_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def slug_team(value):
    s = str(value or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = {
        "uconn": "connecticut",
        "ucf": "central florida",
        "usc": "southern california",
        "ole miss": "mississippi",
        "miami fl": "miami",
        "miami florida": "miami",
        "nc state": "north carolina state",
        "app state": "appalachian state",
    }
    return aliases.get(s, s)


def game_key_from_names(away, home):
    return f"{slug_team(away)}__{slug_team(home)}"


def _source_column(history):
    for c in ["selected_source", "source"]:
        if c in history.columns:
            return c
    return None


def _source_model_columns(history):
    candidates = [
        "source_model_name",
        "source_model_key",
        "source_model_id",
        "model_source_name",
        "source_key",
        "raw_model_name",
        "raw_model_key",
        "raw_column",
        "model_column",
    ]
    return [c for c in candidates if c in history.columns]


PT_CANONICAL_ALIASES = {
    # Canonical cross-source IDs that do not retain the raw PT "line*" key.
    "sagarin": ["linesagr", "linesag"],
    "sagarin_predictor": ["linesagpred"],
    "sagarin_golden_mean": ["linesaggm"],
    "sagarin_recent": ["linesagr"],
    "massey": ["linemassey"],
    "teamrankings": ["lineteamrank"],
    "moore_ratings": ["linemoore"],
    "congrove": ["linecong", "linecons"],
    "harville": ["lineharville"],
    "keeper": ["linekeeper"],
    "espn_fpi": ["linefpi", "lineespn"],
    "dokter_entropy": ["linedktr", "linedokter"],
    "fei": ["linefei"],
    "piratings": ["linepirate", "linepiratings"],
    "piratings_bias": ["linepiratebias", "linepiratingsbias"],
    "piratings_mean": ["linepiratemean", "linepiratingsmean"],
}

PT_DISPLAY_ALIASES = {
    "Big 200": ["linebig200"],
    "Sagarin Golden Mean": ["linesaggm"],
    "Sagarin Predictor": ["linesagpred"],
    "Sagarin Ratings": ["linesagr", "linesag"],
    "Sagarin Recent": ["linesagr"],
    "Talisman Red": ["linetalis"],
    "Howell": ["linehow"],
    "Laz Index": ["linelaz"],
    "DP Dwiggins": ["linedwig"],
    "Massey Ratings": ["linemassey"],
    "TeamRankings.com": ["lineteamrank"],
    "TeamRankings": ["lineteamrank"],
    "Moore Power Ratings": ["linemoore"],
    "Moore Ratings": ["linemoore"],
    "Congrove Computer Rankings": ["linecong", "linecons"],
    "Congrove": ["linecong", "linecons"],
    "Keeper": ["linekeeper"],
    "Donchess Inference": ["linedonchess"],
    "Daniel Curry Index": ["linecurry"],
    "Beck Elo": ["lineelo"],
    "PerformanZ Ratings": ["linepfz"],
}


def build_current_source_mapping(
    history: pd.DataFrame,
    selected_ids: Iterable[str],
    model_name_map: dict[str, str],
) -> pd.DataFrame:
    ids = list(dict.fromkeys(map(str, selected_ids)))
    source_col = _source_column(history)
    source_model_cols = _source_model_columns(history)

    rows = []
    for mid in ids:
        z = history[
            history["canonical_model_id"].astype(str).eq(mid)
        ].copy()

        pt_candidates = []
        cfb_candidates = []

        if source_col and source_model_cols and len(z):
            for source, target in [
                ("predictiontracker", pt_candidates),
                ("cfbpicker", cfb_candidates),
            ]:
                zs = z[
                    z[source_col].astype(str).str.lower().eq(source)
                ]
                for source_model_col in source_model_cols:
                    vals = (
                        zs[source_model_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )
                    target.extend(
                        [x for x in vals.unique().tolist() if x]
                    )

        model_name = model_name_map.get(mid, mid)

        # Explicit aliases for canonical cross-source models plus the old
        # PredictionTracker line-key naming convention.
        pt_candidates.extend(PT_CANONICAL_ALIASES.get(mid, []))
        pt_candidates.extend(PT_DISPLAY_ALIASES.get(model_name, []))

        if mid.startswith("linept_"):
            suffix = mid[len("linept_"):]
            pt_candidates.extend(
                [f"line{suffix}", mid.replace("linept_", "line", 1)]
            )

        picker_fallback = CFB_NAME_ALIASES.get(
            model_name, model_name
        )
        cfb_candidates.append(picker_fallback)

        def uniq(xs):
            return list(dict.fromkeys(str(x) for x in xs if str(x)))

        rows.append({
            "canonical_model_id": mid,
            "model_name": model_name,
            "pt_candidates": "|".join(uniq(pt_candidates)),
            "picker_name": uniq(cfb_candidates)[0],
            "cfb_candidates": "|".join(uniq(cfb_candidates)),
        })

    return pd.DataFrame(rows)


def _norm_model_key(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def resolve_pt_column(columns, row):
    model_cols = [
        c for c in columns
        if str(c).lower().startswith("line")
        and str(c).lower() != "line"
    ]
    norm_map = {_norm_model_key(c): c for c in model_cols}

    candidates = [
        x for x in str(row.pt_candidates or "").split("|") if x
    ]
    candidates.extend([
        row.canonical_model_id,
        row.model_name,
        "line" + _norm_model_key(row.model_name),
    ])

    for cand in candidates:
        key = _norm_model_key(cand)
        if key in norm_map:
            return norm_map[key]

    # Conservative fuzzy suffix match.
    model_slug = _norm_model_key(row.model_name)
    hits = [
        c for c in model_cols
        if model_slug
        and (
            model_slug in _norm_model_key(c)
            or _norm_model_key(c).replace("line", "", 1)
            in model_slug
        )
    ]
    if len(hits) == 1:
        return hits[0]
    return None



def load_predictiontracker_wide(root: Path) -> pd.DataFrame:
    path = root / "data/current/ncaapredictions.csv"
    if not path.exists():
        return pd.DataFrame()
    wide = pd.read_csv(path, low_memory=False)
    wide.columns = [str(c).strip().lower() for c in wide.columns]
    return wide


def predictiontracker_master_slate(root: Path) -> pd.DataFrame:
    wide = load_predictiontracker_wide(root)
    if wide.empty or not {"road", "home", "line"}.issubset(wide.columns):
        return pd.DataFrame()
    keep = [c for c in ["road", "home", "lineopen", "line", "linemidweek"] if c in wide.columns]
    out = wide[keep].copy()
    out["road"] = out["road"].astype(str)
    out["home"] = out["home"].astype(str)
    out["market_home_margin"] = pd.to_numeric(out["line"], errors="coerce")
    out["opening_home_margin"] = pd.to_numeric(out.get("lineopen"), errors="coerce")
    out["midweek_home_margin"] = pd.to_numeric(out.get("linemidweek"), errors="coerce")
    out["line_move_from_open"] = out["market_home_margin"] - out["opening_home_margin"]
    out["game_join_key"] = [
        game_key_from_names(a, h)
        for a, h in zip(out["road"], out["home"])
    ]
    return (
        out.rename(columns={"road": "away"})
        [[
            "away", "home", "opening_home_margin", "market_home_margin",
            "midweek_home_margin", "line_move_from_open", "game_join_key"
        ]]
        .drop_duplicates("game_join_key")
        .reset_index(drop=True)
    )


def predictiontracker_live_model_columns(root: Path) -> pd.DataFrame:
    wide = load_predictiontracker_wide(root)
    if wide.empty:
        return pd.DataFrame()
    rows = []
    excluded = {
        "line", "lineopen", "linemidweek", "lineca",
        "lineavg", "linemedian", "linecons", "linestd",
    }
    for col in wide.columns:
        low = str(col).lower()
        if not low.startswith("line") or low in excluded:
            continue
        v = pd.to_numeric(wide[col], errors="coerce")
        n = int(v.notna().sum())
        rows.append({
            "predictiontracker_column": col,
            "nonmissing_predictions": n,
            "currently_posted": bool(n > 0),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["currently_posted", "nonmissing_predictions", "predictiontracker_column"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def load_predictiontracker_current(
    root: Path,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = load_predictiontracker_wide(root)
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not {"road", "home", "line"}.issubset(wide.columns):
        raise ValueError(
            "PredictionTracker current CSV lacks road/home/line."
        )

    long_rows = []
    map_rows = []
    for row in mapping.itertuples(index=False):
        col = resolve_pt_column(wide.columns, row)
        map_rows.append({
            "canonical_model_id": row.canonical_model_id,
            "model_name": row.model_name,
            "pt_candidates": row.pt_candidates,
            "predictiontracker_column": col,
            "mapped": bool(col is not None),
            "current_nonmissing_predictions": (
                int(pd.to_numeric(wide[col], errors="coerce").notna().sum())
                if col is not None else 0
            ),
        })
        if col is None:
            continue

        line_cols = ["road", "home", "lineopen", "line", "linemidweek", col]
        for c in ["lineopen", "linemidweek"]:
            if c not in wide.columns:
                wide[c] = np.nan
        for g in wide[line_cols].itertuples(index=False, name=None):
            road, home, lineopen, line, linemidweek, pred = g
            market = pd.to_numeric(
                pd.Series([line]), errors="coerce"
            ).iloc[0]
            prediction = pd.to_numeric(
                pd.Series([pred]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(prediction):
                continue
            long_rows.append({
                "away": str(road),
                "home": str(home),
                "market_home_margin": (
                    float(market) if np.isfinite(market) else np.nan
                ),
                "opening_home_margin": pd.to_numeric(pd.Series([lineopen]), errors="coerce").iloc[0],
                "midweek_home_margin": pd.to_numeric(pd.Series([linemidweek]), errors="coerce").iloc[0],
                "canonical_model_id": row.canonical_model_id,
                "model_name": row.model_name,
                "prediction_home_margin": float(prediction),
                "source": "predictiontracker",
                "source_model_name": col,
            })

    return pd.DataFrame(long_rows), pd.DataFrame(map_rows)


def load_cfbpicker_current(
    root: Path,
    *,
    season: int | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    path = root / "data/current/cfbpicker_current_long.csv"
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        return x
    if season is not None and "season" in x.columns:
        x = x[pd.to_numeric(x["season"], errors="coerce").eq(int(season))].copy()
    if week is not None and "week" in x.columns:
        x = x[pd.to_numeric(x["week"], errors="coerce").eq(int(week))].copy()
    if x.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "away": x["away"].astype(str),
        "home": x["home"].astype(str),
        "market_home_margin": pd.to_numeric(
            x.get("market_home_margin_close"), errors="coerce"
        ),
        "canonical_model_id": x[
            "canonical_model_id"
        ].astype(str),
        "model_name": x["model_name"].astype(str),
        "prediction_home_margin": pd.to_numeric(
            x["prediction_home_margin"], errors="coerce"
        ),
        "source": "cfbpicker",
        "source_model_name": x["picker"].astype(str),
    })
    return out.dropna(subset=["prediction_home_margin"])


def current_cfbpicker_model_map(
    root: str | Path,
    *,
    season: int,
    week: int,
) -> dict[str, str]:
    """Canonical live-model names available in the verified CFB Picker cache."""
    current = load_cfbpicker_current(
        Path(root), season=int(season), week=int(week)
    )
    if current.empty:
        return {}
    pairs = current[["canonical_model_id", "model_name"]].drop_duplicates(
        "canonical_model_id", keep="last"
    )
    return dict(zip(
        pairs["canonical_model_id"].astype(str),
        pairs["model_name"].astype(str),
    ))


def build_current_board_from_cached_sources(
    root: str | Path,
    history: pd.DataFrame,
    selected_ids: Iterable[str],
    model_name_map: dict[str, str],
    *,
    season: int,
    week: int,
    primary_k: float = 1.50,
    min_available_models: int = 4,
    include_cfbpicker: bool = True,
    write_outputs: bool = True,
) -> dict:
    root = Path(root)
    selected = list(dict.fromkeys(map(str, selected_ids)))
    mapping = build_current_source_mapping(
        history, selected, model_name_map
    )

    pt, pt_map = load_predictiontracker_current(root, mapping)
    cfb = load_cfbpicker_current(root, season=int(season), week=int(week)) if include_cfbpicker else pd.DataFrame()
    pt_master = predictiontracker_master_slate(root)
    pt_live = predictiontracker_live_model_columns(root)

    if len(cfb):
        cfb = cfb[
            cfb["canonical_model_id"].astype(str).isin(selected)
        ].copy()

    combined = pd.concat(
        [pt, cfb], ignore_index=True, sort=False
    )

    output = root / "outputs/current_week"
    if write_outputs:
        output.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(
            output / "current_source_mapping.csv", index=False
        )
        if len(pt_map):
            pt_map.to_csv(
                output / "current_predictiontracker_mapping.csv",
                index=False,
            )
        if len(pt_live):
            pt_live.to_csv(
                output / "current_predictiontracker_live_models.csv",
                index=False,
            )

    if len(combined):
        combined["game_join_key"] = [
            game_key_from_names(a, h)
            for a, h in zip(combined["away"], combined["home"])
        ]

        # PredictionTracker preferred when the SAME canonical model/game exists
        # in both sources. Source is provenance only, never a weight.
        source_order = {"predictiontracker": 0, "cfbpicker": 1}
        combined["_source_order"] = combined["source"].map(
            source_order
        ).fillna(9)
        combined = (
            combined.sort_values(
                ["game_join_key", "canonical_model_id", "_source_order"]
            )
            .drop_duplicates(
                ["game_join_key", "canonical_model_id"],
                keep="first",
            )
            .drop(columns="_source_order")
        )

    # Master game universe is NOT derived from selected-model availability.
    # PredictionTracker road/home/line defines the current slate even when zero
    # selected models have posted. Add CFB-only games if that source is usable.
    master_frames = []
    if len(pt_master):
        master_frames.append(
            pt_master.assign(master_source="PredictionTracker")
        )
    if len(cfb):
        cfb_master = cfb[["away", "home", "market_home_margin"]].copy()
        cfb_master["game_join_key"] = [
            game_key_from_names(a, h)
            for a, h in zip(cfb_master["away"], cfb_master["home"])
        ]
        cfb_master["master_source"] = "CFB Picker"
        master_frames.append(cfb_master)

    if master_frames:
        master = pd.concat(master_frames, ignore_index=True, sort=False)
        # Prefer PT master line for overlapping games.
        master["_ord"] = master["master_source"].map(
            {"PredictionTracker": 0, "CFB Picker": 1}
        ).fillna(9)
        master = (
            master.sort_values(["game_join_key", "_ord"])
            .drop_duplicates("game_join_key", keep="first")
            .drop(columns="_ord")
            .reset_index(drop=True)
        )
    elif len(combined):
        master = (
            combined[
                ["away", "home", "market_home_margin", "game_join_key"]
            ]
            .drop_duplicates("game_join_key")
            .assign(master_source="selected-model sources")
        )
    else:
        master = pd.DataFrame()

    board_rows = []
    pred_rows = []

    if len(master):
        for mr in master.itertuples(index=False):
            game_key = str(mr.game_join_key)
            away = str(mr.away)
            home = str(mr.home)

            if len(combined):
                g = combined[
                    combined["game_join_key"].astype(str).eq(game_key)
                ].copy()
            else:
                g = pd.DataFrame()

            # Market preference remains PT current line, then CFB close.
            market = pd.to_numeric(
                pd.Series([getattr(mr, "market_home_margin", np.nan)]),
                errors="coerce",
            ).iloc[0]
            market_source = str(
                getattr(mr, "master_source", "current source")
            )

            if len(g):
                pt_market = pd.to_numeric(
                    g.loc[
                        g["source"].eq("predictiontracker"),
                        "market_home_margin",
                    ],
                    errors="coerce",
                ).dropna()
                if len(pt_market):
                    market = float(pt_market.iloc[0])
                    market_source = "PredictionTracker current line"
                else:
                    any_market = pd.to_numeric(
                        g["market_home_margin"], errors="coerce"
                    ).dropna()
                    if len(any_market):
                        market = float(any_market.iloc[0])
                        market_source = "CFB Picker close"

                vals_all = pd.to_numeric(
                    g["prediction_home_margin"], errors="coerce"
                )
                valid = g.loc[vals_all.notna()].copy()
            else:
                valid = pd.DataFrame(
                    columns=[
                        "canonical_model_id", "model_name",
                        "prediction_home_margin", "source",
                        "source_model_name",
                    ]
                )

            vals = pd.to_numeric(
                valid["prediction_home_margin"], errors="coerce"
            ) if len(valid) else pd.Series(dtype=float)

            available = (
                valid["canonical_model_id"].astype(str).tolist()
                if len(valid) else []
            )
            missing = [
                m for m in selected if m not in set(available)
            ]

            n = len(vals)
            consensus = float(vals.mean()) if n else np.nan
            sd = float(vals.std(ddof=1)) if n >= 2 else np.nan
            edge = (
                consensus - float(market)
                if np.isfinite(consensus) and np.isfinite(market)
                else np.nan
            )
            if np.isfinite(edge) and np.isfinite(sd):
                signal = (
                    abs(edge) / sd
                    if sd > 1e-12
                    else (np.inf if abs(edge) > 1e-12 else 0.0)
                )
            else:
                signal = np.nan
            qualifies = bool(
                n >= int(min_available_models)
                and (np.isfinite(signal) or np.isinf(signal))
                and signal >= float(primary_k)
            )

            if np.isfinite(edge):
                side = home if edge > 0 else away
            else:
                side = ""

            board_rows.append({
                "season": int(season),
                "week": int(week),
                "away": away,
                "home": home,
                "opening_home_margin": pd.to_numeric(
                    pd.Series([getattr(mr, "opening_home_margin", np.nan)]), errors="coerce"
                ).iloc[0],
                "market_home_margin": (
                    float(market) if np.isfinite(market) else np.nan
                ),
                "midweek_home_margin": pd.to_numeric(
                    pd.Series([getattr(mr, "midweek_home_margin", np.nan)]), errors="coerce"
                ).iloc[0],
                "line_move_from_open": pd.to_numeric(
                    pd.Series([getattr(mr, "line_move_from_open", np.nan)]), errors="coerce"
                ).iloc[0],
                "market_source": market_source,
                "consensus_home_margin": consensus,
                "model_sd": sd,
                "edge_home": edge,
                "signal_sd": signal,
                "selected_models": len(selected),
                "available_models": n,
                "qualifies": qualifies,
                "bet_side": side if qualifies else "",
                "availability_state": (
                    "SCORABLE"
                    if n >= int(min_available_models)
                    else (
                        "PARTIAL"
                        if n > 0
                        else "NO_SELECTED_MODELS_POSTED"
                    )
                ),
                "missing_models": "|".join(
                    model_name_map.get(m, m) for m in missing
                ),
                "available_model_names": "|".join(
                    valid["model_name"].astype(str).tolist()
                ) if len(valid) else "",
                "game_join_key": game_key,
            })

            for row in valid.itertuples(index=False):
                pred_rows.append({
                    "season": int(season),
                    "week": int(week),
                    "away": away,
                    "home": home,
                    "game_join_key": game_key,
                    "opening_home_margin": pd.to_numeric(
                        pd.Series([getattr(mr, "opening_home_margin", np.nan)]), errors="coerce"
                    ).iloc[0],
                    "market_home_margin": (
                        float(market) if np.isfinite(market) else np.nan
                    ),
                    "midweek_home_margin": pd.to_numeric(
                        pd.Series([getattr(mr, "midweek_home_margin", np.nan)]), errors="coerce"
                    ).iloc[0],
                    "line_move_from_open": pd.to_numeric(
                        pd.Series([getattr(mr, "line_move_from_open", np.nan)]), errors="coerce"
                    ).iloc[0],
                    "canonical_model_id": row.canonical_model_id,
                    "model_name": row.model_name,
                    "prediction_home_margin": row.prediction_home_margin,
                    "source": row.source,
                    "source_model_name": row.source_model_name,
                })

    board = pd.DataFrame(board_rows)
    if len(board):
        board = board.sort_values(
            ["qualifies", "available_models", "signal_sd"],
            ascending=[False, False, False],
            na_position="last",
        )
    predictions = pd.DataFrame(pred_rows)
    qualifying = (
        board[board["qualifies"]].copy()
        if len(board) else pd.DataFrame()
    )

    # Per-selected-model availability with a diagnosis, not just a count.
    pt_map_lookup = {}
    if len(pt_map):
        for r in pt_map.itertuples(index=False):
            pt_map_lookup[str(r.canonical_model_id)] = {
                "column": getattr(r, "predictiontracker_column", None),
                "mapped": bool(getattr(r, "mapped", False)),
                "n": int(
                    getattr(r, "current_nonmissing_predictions", 0) or 0
                ),
            }

    availability_rows = []
    for mid in selected:
        z = (
            predictions[
                predictions["canonical_model_id"].astype(str).eq(mid)
            ]
            if len(predictions)
            else pd.DataFrame()
        )
        pt_info = pt_map_lookup.get(mid, {})
        cfb_n = 0
        if len(cfb):
            cfb_n = int(
                cfb[
                    cfb["canonical_model_id"].astype(str).eq(mid)
                ][["away", "home"]]
                .drop_duplicates()
                .shape[0]
            )

        total_n = int(
            z[["away", "home"]].drop_duplicates().shape[0]
        ) if len(z) else 0

        if total_n > 0:
            state = "POSTED"
        elif pt_info.get("mapped") and int(pt_info.get("n", 0)) == 0:
            state = "PT_COLUMN_EXISTS_BUT_NOT_POSTED"
        elif not pt_info.get("mapped") and cfb_n == 0:
            state = "CFBP_REQUIRED_OR_NOT_POSTED"
        else:
            state = "UNAVAILABLE"

        availability_rows.append({
            "canonical_model_id": mid,
            "model_name": model_name_map.get(mid, mid),
            "games_available": total_n,
            "predictiontracker_column": pt_info.get("column"),
            "pt_nonmissing_predictions": int(pt_info.get("n", 0)),
            "cfbpicker_games": cfb_n,
            "availability_state": state,
            "sources_seen": (
                "|".join(sorted(z["source"].dropna().astype(str).unique()))
                if len(z) else ""
            ),
        })

    availability = pd.DataFrame(availability_rows)
    if len(availability):
        availability = availability.sort_values(
            ["games_available", "model_name"],
            ascending=[False, True],
        )

    if write_outputs:
        board.to_csv(
            output / "current_consensus_board.csv", index=False
        )
        qualifying.to_csv(
            output / "current_qualifying_plays.csv", index=False
        )
        predictions.to_csv(
            output / "current_model_predictions_long.csv", index=False
        )
        availability.to_csv(
            output / "current_model_availability.csv", index=False
        )

    return {
        "board": board,
        "qualifying": qualifying,
        "predictions": predictions,
        "availability": availability,
        "source_mapping": mapping,
        "pt_mapping": pt_map,
        "pt_live_models": pt_live,
        "usable_source_rows": int(len(combined)),
        "master_games": int(len(master)),
    }


def _run(cmd, cwd, timeout_seconds=None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return {
            "command": " ".join(map(str, cmd)),
            "returncode": int(proc.returncode),
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
            "stdout": proc.stdout[-6000:],
            "stderr": proc.stderr[-6000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "command": " ".join(map(str, cmd)),
            "returncode": 124,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": str(stdout)[-6000:],
            "stderr": (
                str(stderr)[-5000:]
                + f"\nTimed out after {timeout_seconds} seconds."
            ),
        }



def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download_raw_github(url: str, timeout: int = 45) -> bytes:
    """Fetch a raw-GitHub mirror with cache busting on GitHub only."""
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}_refresh={utc_stamp()}"
    response = requests.get(
        full,
        timeout=timeout,
        headers={
            "User-Agent": "ncaaf-consensus-lab/3.5.20",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Accept": "*/*",
        },
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"Empty GitHub mirror response: {url}")
    return response.content


def _refresh_predictiontracker_from_github_mirror(
    root: Path,
    *,
    season: int,
    week: int,
) -> dict:
    """Load a locally refreshed PT mirror from GitHub and validate its provenance.

    This is deliberately strict.  A mirror is accepted only when its metadata
    explicitly says it belongs to the season/week currently selected in the app
    and the canonical CSV hash agrees with the metadata.  Therefore a prior-week
    GitHub file cannot silently masquerade as a successful live refresh.
    """
    meta_bytes = _download_raw_github(PT_MIRROR_META_URL)
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse PT mirror metadata: {exc}") from exc

    mirror_season = int(meta.get("season", -1))
    mirror_week = int(meta.get("week", -1))
    if mirror_season != int(season) or mirror_week != int(week):
        raise RuntimeError(
            "PredictionTracker direct fetch is blocked by Connect Cloud, and the "
            f"GitHub mirror is season {mirror_season} week {mirror_week}, not "
            f"season {int(season)} week {int(week)}. On your Mac run: "
            f"./refresh_predictiontracker_local_and_push.sh {int(season)} {int(week)}"
        )

    csv_bytes = _download_raw_github(PT_MIRROR_CSV_URL)
    frame = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"home", "road", "line"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "GitHub PT mirror is missing required columns: " + ", ".join(missing)
        )
    model_columns = [
        c for c in frame.columns
        if c.startswith("line") and c not in {"line", "lineopen", "linemidweek", "lineca", "lineavg", "linemedian", "linecons", "linestd"}
    ]
    if len(model_columns) < 3:
        raise RuntimeError(
            f"GitHub PT mirror has only {len(model_columns)} model columns; refusing it."
        )

    canonical = frame.to_csv(index=False, na_rep="").encode("utf-8")
    canonical_sha = _sha256_bytes(canonical)
    expected_sha = str(meta.get("canonical_sha256") or "").strip()
    if expected_sha and canonical_sha != expected_sha:
        raise RuntimeError(
            "GitHub PT mirror CSV hash does not match mirror metadata; refusing it."
        )

    production = root / "data/current/ncaapredictions.csv"
    production.parent.mkdir(parents=True, exist_ok=True)
    old_bytes = production.read_bytes() if production.exists() else None
    changed = old_bytes != canonical
    production.write_bytes(canonical)

    raw_path = root / "data/raw/predictiontracker/current_ncaapredictions_from_github.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(csv_bytes)

    record = {
        "name": "current_predictions",
        "url": PT_MIRROR_CSV_URL,
        "status": "ok",
        "fetched_at_utc": utc_now(),
        "source_fetched_at_utc": meta.get("fetched_at_utc"),
        "http_status": 200,
        "bytes": len(csv_bytes),
        "sha256": _sha256_bytes(csv_bytes),
        "canonical_sha256": canonical_sha,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "changed": bool(changed),
        "production_path": "data/current/ncaapredictions.csv",
        "snapshot_path": meta.get("snapshot_path"),
        "published_update": meta.get("published_update"),
        "transport": "github_mirror",
        "mirror_commit_hint": meta.get("git_commit"),
        "message": json.dumps({
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "model_columns": int(len(model_columns)),
            "mirror_validation": "ok",
            "mirror_season": mirror_season,
            "mirror_week": mirror_week,
            "source_fetched_at_utc": meta.get("fetched_at_utc"),
            "page_validation": meta.get("page_validation") or {},
        }, sort_keys=True),
    }

    manifest = {
        "overall_status": "ok",
        "transport": "github_mirror",
        "records": [record],
    }
    status_path = root / "data/derived/predictiontracker_source_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return record


def refresh_current_sources(
    root: str | Path,
    history: pd.DataFrame,
    selected_ids: Iterable[str],
    model_name_map: dict[str, str],
    *,
    season: int,
    week: int,
    include_cfbpicker: bool = True,
    refresh_cfbpicker: bool = True,
) -> dict:
    root = Path(root).resolve()
    selected = list(dict.fromkeys(map(str, selected_ids)))
    mapping = build_current_source_mapping(
        history, selected, model_name_map
    )

    current_dir = root / "data/current"
    current_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = current_dir / "cfbpicker_selected_mapping.csv"
    mapping[
        ["canonical_model_id", "model_name", "picker_name"]
    ].to_csv(mapping_file, index=False)

    status = {
        "refreshed_at_utc": utc_now(),
        "season": int(season),
        "week": int(week),
        "selected_models": len(selected),
        "predictiontracker": None,
        "cfbpicker": None,
    }

    pt_script = root / "scripts/scrape_predictiontracker.py"
    if pt_script.exists():
        direct = _run(
            [
                sys.executable,
                str(pt_script),
                "--root",
                str(root),
                "--skip-results",
                "--skip-archives",
                "--strict",
            ],
            root,
            timeout_seconds=120,
        )
    else:
        direct = {
            "returncode": 127,
            "stderr": "scripts/scrape_predictiontracker.py is missing.",
        }

    status["predictiontracker_direct"] = direct

    source_manifest = _safe_json_load(
        root / "data/derived/predictiontracker_source_status.json"
    )
    current_records = [
        r for r in source_manifest.get("records", [])
        if r.get("name") == "current_predictions"
    ]
    direct_record = current_records[-1] if current_records else None

    if int(direct.get("returncode", 1)) == 0 and (direct_record or {}).get("status") == "ok":
        # Local development can still reach PredictionTracker directly.
        status["predictiontracker"] = direct
        status["predictiontracker_source_record"] = direct_record
        status["predictiontracker_transport"] = "direct"
        status["predictiontracker_source_manifest_status"] = source_manifest.get("overall_status")
    else:
        # Posit Connect Cloud receives HTTP 403 from PredictionTracker.  Do not
        # keep retrying headers or accept cached rows: load the explicit
        # season/week-tagged GitHub mirror produced by the user's local Mac.
        try:
            mirror_record = _refresh_predictiontracker_from_github_mirror(
                root, season=int(season), week=int(week)
            )
            status["predictiontracker"] = {
                "returncode": 0,
                "stdout": "Direct PredictionTracker access blocked; verified GitHub mirror loaded.",
                "stderr": str(direct.get("stderr") or ""),
                "fallback": "github_mirror",
            }
            status["predictiontracker_source_record"] = mirror_record
            status["predictiontracker_transport"] = "github_mirror"
            status["predictiontracker_source_manifest_status"] = "ok"
        except Exception as mirror_exc:
            status["predictiontracker"] = {
                "returncode": 1,
                "stdout": str(direct.get("stdout") or ""),
                "stderr": (
                    str(direct.get("stderr") or "")
                    + "\nGitHub mirror fallback failed: " + str(mirror_exc)
                ),
                "fallback": "github_mirror_failed",
            }
            status["predictiontracker_source_record"] = direct_record
            status["predictiontracker_transport"] = "failed"
            status["predictiontracker_source_manifest_status"] = "error"

    cfb_script = root / "scripts/scrape_cfbpicker_current.py"
    if not include_cfbpicker:
        status["cfbpicker"] = {
            "returncode": 0,
            "disabled": True,
            "stderr": "CFB Picker intentionally disabled for this refresh.",
        }
    elif not refresh_cfbpicker:
        cached_cfb = load_cfbpicker_current(
            root, season=int(season), week=int(week)
        )
        status["cfbpicker"] = {
            "returncode": 0 if len(cached_cfb) else 1,
            "refreshed": False,
            "transport": "verified_local_or_deployed_cache",
            "rows": int(len(cached_cfb)),
            "models": int(cached_cfb["canonical_model_id"].nunique())
            if len(cached_cfb) else 0,
            "stderr": "" if len(cached_cfb) else (
                "No season/week-matched CFB Picker cache was available. "
                "Run the Mac-side CFB Picker refresh and redeploy the mirror."
            ),
        }
    elif cfb_script.exists():
        status["cfbpicker"] = _run(
            [
                sys.executable,
                str(cfb_script),
                "--root",
                str(root),
                "--season",
                str(int(season)),
                "--week",
                str(int(week)),
                "--mapping-file",
                str(mapping_file),
                "--strict",
                "--quick",
            ],
            root,
            timeout_seconds=900,
        )
    else:
        status["cfbpicker"] = {
            "returncode": 127,
            "stderr": (
                "scripts/scrape_cfbpicker_current.py is missing."
            ),
        }

    path = root / "data/derived/current_week_refresh_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2))
    return status


def _snapshot_index_path(root: Path) -> Path:
    return root / "data/snapshots/predictiontracker/prospective_index.csv"


def save_prospective_current_week_snapshot(
    root: str | Path,
    result: dict,
    refresh_status: dict | None,
    *,
    season: int,
    week: int,
) -> dict:
    """Persist each unique observed PT board for later early-line/CLV research.

    The source SHA is the deduplication key.  Re-clicking Refresh without a
    PredictionTracker change therefore does not create fake extra observations.
    """
    root = Path(root)
    refresh_status = refresh_status or {}
    record = refresh_status.get("predictiontracker_source_record") or {}
    sha = str(record.get("canonical_sha256") or record.get("sha256") or "").strip()
    observed_at = str(record.get("fetched_at_utc") or utc_now())
    stamp = re.sub(r"[^0-9TZ]", "", observed_at.replace("+00:00", "Z"))[:16]
    if not stamp:
        stamp = utc_stamp()
    sha12 = sha[:12] if sha else "nohash"

    base = (
        root / "data/snapshots/predictiontracker/prospective"
        / f"season_{int(season)}" / f"week_{int(week):02d}"
    )
    base.mkdir(parents=True, exist_ok=True)
    prefix = f"{stamp}_{sha12}"

    index_path = _snapshot_index_path(root)
    if index_path.exists():
        try:
            existing = pd.read_csv(index_path, dtype=str)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    already = False
    if len(existing) and sha and "source_sha256" in existing.columns:
        already = bool(existing["source_sha256"].fillna("").astype(str).eq(sha).any())

    board = result.get("board", pd.DataFrame()).copy()
    predictions = result.get("predictions", pd.DataFrame()).copy()
    availability = result.get("availability", pd.DataFrame()).copy()
    wide = load_predictiontracker_wide(root)

    # Make each exported row independently joinable after many weekly downloads.
    # The raw PT wide CSV is left untouched; derived prospective tables carry
    # explicit observation/provenance columns.
    for frame in (board, predictions, availability):
        frame.insert(0, "observed_at_utc", observed_at)
        frame.insert(1, "source_sha256", sha)
        frame.insert(2, "pt_published_update", record.get("published_update"))

    paths = {
        "board": base / f"{prefix}_board.csv",
        "predictions": base / f"{prefix}_predictions_long.csv",
        "availability": base / f"{prefix}_availability.csv",
        "predictiontracker_wide": base / f"{prefix}_predictiontracker_wide.csv",
        "metadata": base / f"{prefix}_metadata.json",
    }

    if not already:
        board.to_csv(paths["board"], index=False)
        predictions.to_csv(paths["predictions"], index=False)
        availability.to_csv(paths["availability"], index=False)
        wide.to_csv(paths["predictiontracker_wide"], index=False)

        meta = {
            "observed_at_utc": observed_at,
            "season": int(season),
            "week": int(week),
            "source_sha256": sha or None,
            "predictiontracker_published_update": record.get("published_update"),
            "source_changed": record.get("changed"),
            "source_rows": record.get("rows"),
            "source_columns": record.get("columns"),
            "source_message": record.get("message"),
            "games": int(len(board)),
            "mapped_predictions": int(len(predictions)),
            "canonical_models_posting": int(
                predictions["canonical_model_id"].astype(str).nunique()
            ) if len(predictions) and "canonical_model_id" in predictions.columns else 0,
            "live_pt_model_columns": int(len(result.get("pt_live_models", pd.DataFrame()))),
            "files": {k: str(v.relative_to(root)) for k, v in paths.items() if k != "metadata"},
        }
        paths["metadata"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

        row = pd.DataFrame([{
            "observed_at_utc": observed_at,
            "season": int(season),
            "week": int(week),
            "source_sha256": sha,
            "published_update": record.get("published_update"),
            "source_changed": record.get("changed"),
            "games": int(len(board)),
            "mapped_predictions": int(len(predictions)),
            "canonical_models_posting": meta["canonical_models_posting"],
            "snapshot_prefix": str((base / prefix).relative_to(root)),
        }])
        if len(existing):
            new_index = pd.concat([existing, row], ignore_index=True, sort=False)
        else:
            new_index = row
        index_path.parent.mkdir(parents=True, exist_ok=True)
        new_index.to_csv(index_path, index=False)
    else:
        # Find the prior snapshot prefix for UI provenance.
        prior = existing[existing["source_sha256"].fillna("").astype(str).eq(sha)]
        if len(prior) and "snapshot_prefix" in prior.columns:
            prefix_value = str(prior.iloc[-1]["snapshot_prefix"])
        else:
            prefix_value = str((base / prefix).relative_to(root))
        return {
            "saved": False,
            "duplicate_source_sha": True,
            "source_sha256": sha,
            "observed_at_utc": observed_at,
            "snapshot_prefix": prefix_value,
            "index_path": str(index_path.relative_to(root)),
        }

    return {
        "saved": True,
        "duplicate_source_sha": False,
        "source_sha256": sha,
        "observed_at_utc": observed_at,
        "snapshot_prefix": str((base / prefix).relative_to(root)),
        "index_path": str(index_path.relative_to(root)),
        "paths": {k: str(v.relative_to(root)) for k, v in paths.items()},
    }


def refresh_and_build_current_week(
    root: str | Path,
    history: pd.DataFrame,
    selected_ids: Iterable[str],
    model_name_map: dict[str, str],
    *,
    season: int,
    week: int,
    primary_k: float = 1.50,
    min_available_models: int = 4,
    refresh: bool = True,
    include_cfbpicker: bool = True,
    refresh_cfbpicker: bool = True,
    write_outputs: bool = True,
) -> dict:
    root = Path(root)
    refresh_status = None
    if refresh:
        refresh_status = refresh_current_sources(
            root,
            history,
            selected_ids,
            model_name_map,
            season=season,
            week=week,
            include_cfbpicker=include_cfbpicker,
            refresh_cfbpicker=refresh_cfbpicker,
        )

        # CRITICAL: never report a successful refresh while silently rebuilding
        # from an old cached ncaapredictions.csv.  The scraper runs --strict; a
        # nonzero code or non-ok current source record aborts the Shiny task.
        pt_run = (refresh_status or {}).get("predictiontracker") or {}
        pt_record = (refresh_status or {}).get("predictiontracker_source_record") or {}
        if int(pt_run.get("returncode", 1)) != 0:
            detail = str(pt_run.get("stderr") or pt_run.get("stdout") or "")[-1800:]
            raise RuntimeError(
                "PredictionTracker refresh failed; cached prior-week rows were NOT used. "
                + detail
            )
        if pt_record.get("status") != "ok":
            raise RuntimeError(
                "PredictionTracker refresh did not produce a verified current source record; "
                "cached rows were NOT used. " + str(pt_record.get("message") or "")
            )

    result = build_current_board_from_cached_sources(
        root,
        history,
        selected_ids,
        model_name_map,
        season=season,
        week=week,
        primary_k=primary_k,
        min_available_models=min_available_models,
        include_cfbpicker=include_cfbpicker,
        write_outputs=write_outputs,
    )
    result["refresh_status"] = refresh_status
    if refresh:
        result["prospective_snapshot"] = save_prospective_current_week_snapshot(
            root, result, refresh_status, season=season, week=week
        )
    else:
        result["prospective_snapshot"] = None
    return result


def save_current_selection(
    root: str | Path,
    selected_ids: Iterable[str],
    model_name_map: dict[str, str],
    *,
    season: int,
    week: int,
    primary_k: float,
    min_available_models: int,
    combinations: list[dict] | None = None,
):
    root = Path(root)
    ids = list(dict.fromkeys(map(str, selected_ids)))
    payload = {
        "saved_at_utc": utc_now(),
        "season": int(season),
        "week": int(week),
        "primary_k": float(primary_k),
        "min_available_models": int(min_available_models),
        "model_ids": ids,
        "model_names": [model_name_map.get(x, x) for x in ids],
    }
    if combinations:
        clean = []
        for c in combinations:
            mids = [str(x) for x in c.get("model_ids", []) if str(x)]
            if not mids:
                continue
            row = {
                "rank": int(c.get("rank", len(clean) + 1)),
                "model_ids": mids,
                "model_names": [model_name_map.get(x, x) for x in mids],
            }
            if c.get("k") is not None:
                row["k"] = float(c.get("k"))
            if c.get("community") is not None:
                row["community"] = int(c.get("community"))
            clean.append(row)
        if clean:
            payload["combinations"] = clean
    path = root / "data/strategy/current_week_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
