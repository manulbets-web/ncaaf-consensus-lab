#!/usr/bin/env python3
"""Import historical CFB Picker Tableau predictions into the canonical model file.

This importer is deliberately conservative:
- PredictionTracker remains the game/market/outcome source of record.
- Duplicate model/game observations are NOT appended.
- Overlapping predictions are written to an audit file.
- New CFB Picker models are appended only on games matched to the canonical slate.
- Ambiguous prediction sign is inferred from PredictionTracker overlap or market
  correlation and otherwise withheld rather than silently flipping spreads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfbpicker_tableau import compact, canonical_picker_name, slug_team


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_slug(value: str) -> str:
    s = compact(value)
    return s[:60] or "model"


def model_aliases(name: str) -> set[str]:
    c = canonical_picker_name(name)
    base = {compact(name), compact(c)}
    aliases = {
        "FEI": ["FEI"],
        "SP+": ["SP+", "SP Plus", "SPPlus", "Bill Connelly SP+"],
        "FPI": ["FPI", "ESPN FPI"],
        "KFord": ["KFord", "KFord Ratings", "Kelley Ford", "Kelley Ford Ratings"],
        "Massey Ratings": ["Massey", "Massey Ratings"],
        "Harville": ["Harville", "David Harville"],
        "TeamRankings": ["TeamRankings", "TeamRankings.com", "Team Rankings"],
        "Sagarin: Predictor": ["Sagarin Predictor", "Sagarin: Predictor", "Sagarin Pred"],
        "Sagarin: Golden": ["Sagarin Golden Mean", "Sagarin: Golden", "Sagarin Golden"],
        "Sagarin": ["Sagarin", "Sagarin Ratings"],
        "Sagarin: Recent": ["Sagarin Recent", "Sagarin: Recent"],
    }
    for x in aliases.get(c, []):
        base.add(compact(x))
    return {x for x in base if x}


def build_existing_model_lookup(canonical: pd.DataFrame, registry: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    counts = canonical["canonical_model_id"].astype(str).value_counts().to_dict()
    candidates: dict[str, set[str]] = {}
    display: dict[str, str] = {}

    def add(mid, value):
        if pd.isna(value):
            return
        mid = str(mid)
        value = str(value).strip()
        if not value:
            return
        display.setdefault(mid, value if value else mid)
        for a in model_aliases(value):
            candidates.setdefault(a, set()).add(mid)

    if "model_name" in canonical:
        for row in canonical[["canonical_model_id", "model_name"]].drop_duplicates().itertuples(index=False):
            add(row.canonical_model_id, row.model_name)
    for col in ["source_model_name", "raw_model_name", "model_source_name", "source_model_key"]:
        if col in canonical:
            for row in canonical[["canonical_model_id", col]].dropna().drop_duplicates().itertuples(index=False):
                add(row[0], row[1])
    if len(registry) and {"canonical_model_id", "model_name"}.issubset(registry.columns):
        for row in registry[["canonical_model_id", "model_name"]].drop_duplicates().itertuples(index=False):
            add(row.canonical_model_id, row.model_name)

    resolved = {}
    for alias, ids in candidates.items():
        resolved[alias] = sorted(ids, key=lambda x: (-counts.get(x, 0), x))[0]
    # Prefer canonical registry display name when possible.
    if len(registry) and {"canonical_model_id", "model_name"}.issubset(registry.columns):
        display.update(dict(zip(registry["canonical_model_id"].astype(str), registry["model_name"].astype(str))))
    return resolved, display


def map_picker_models(parsed: pd.DataFrame, canonical: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup, display = build_existing_model_lookup(canonical, registry)
    existing_ids = set(canonical["canonical_model_id"].astype(str))
    mapping_rows = []
    used_new = set()
    for picker in sorted(parsed["picker"].dropna().astype(str).unique()):
        aliases = model_aliases(picker)
        hits = [lookup[a] for a in aliases if a in lookup]
        if hits:
            # Most frequent canonical observation wins if aliases hit multiple IDs.
            vc = canonical["canonical_model_id"].astype(str).value_counts()
            mid = sorted(set(hits), key=lambda x: (-int(vc.get(x, 0)), x))[0]
            is_new = False
            reason = "matched_existing_alias"
            model_name = display.get(mid, canonical_picker_name(picker))
        else:
            base = f"cfbpicker_{safe_slug(canonical_picker_name(picker))}"
            mid = base
            i = 2
            while mid in existing_ids or mid in used_new:
                mid = f"{base}_{i}"
                i += 1
            used_new.add(mid)
            is_new = True
            reason = "new_cfbpicker_model"
            model_name = canonical_picker_name(picker)
        mapping_rows.append({
            "picker": picker,
            "canonical_model_id": mid,
            "model_name": model_name,
            "is_new_canonical_model": is_new,
            "mapping_reason": reason,
        })
    mapping = pd.DataFrame(mapping_rows)
    if len(mapping):
        parsed = parsed.merge(mapping, on="picker", how="left")
    return parsed, mapping


def _canonical_game_table(canonical: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["game_key", "season", "week", "road", "home"] if c in canonical.columns]
    if not {"game_key", "season", "week"}.issubset(cols) or "road" not in cols or "home" not in cols:
        raise ValueError("Canonical predictions need game_key, season, week, road, and home for CFB Picker matching.")
    source_col = "selected_source" if "selected_source" in canonical.columns else ("source" if "source" in canonical.columns else None)
    x = canonical.copy()
    if source_col:
        x["_pt"] = x[source_col].astype(str).str.lower().eq("predictiontracker").astype(int)
    else:
        x["_pt"] = 0
    games = (
        x.sort_values(["_pt"], ascending=False)
        .drop_duplicates("game_key")[["game_key", "season", "week", "road", "home"]]
        .copy()
    )
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games["road_slug"] = games["road"].map(slug_team)
    games["home_slug"] = games["home"].map(slug_team)
    return games


def match_games(parsed: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = _canonical_game_table(canonical)
    # Strict season/week index first.
    strict = {}
    loose = {}
    for r in games.itertuples(index=False):
        if pd.notna(r.season) and pd.notna(r.week):
            strict.setdefault((int(r.season), int(r.week), r.road_slug, r.home_slug), []).append(r.game_key)
            strict.setdefault((int(r.season), int(r.week), r.home_slug, r.road_slug), []).append((r.game_key, "reversed"))
        if pd.notna(r.season):
            loose.setdefault((int(r.season), r.road_slug, r.home_slug), []).append(r.game_key)
            loose.setdefault((int(r.season), r.home_slug, r.road_slug), []).append((r.game_key, "reversed"))

    rows = []
    unmatched = []
    for idx, r in parsed.iterrows():
        season = pd.to_numeric(pd.Series([r.get("season")]), errors="coerce").iloc[0]
        week = pd.to_numeric(pd.Series([r.get("week")]), errors="coerce").iloc[0]
        away = r.get("away_slug") or slug_team(r.get("away"))
        home = r.get("home_slug") or slug_team(r.get("home"))
        hits = []
        if pd.notna(season) and pd.notna(week):
            hits = strict.get((int(season), int(week), away, home), [])
        match_type = "strict"
        if not hits and pd.notna(season):
            hits = loose.get((int(season), away, home), [])
            # Loose match is accepted only if unique to avoid bowl/week mismatches.
            match_type = "season_teams"
        normalized = []
        for h in hits:
            if isinstance(h, tuple):
                normalized.append((h[0], h[1]))
            else:
                normalized.append((h, "direct"))
        # de-dupe same game reached through aliases.
        normalized = list(dict.fromkeys(normalized))
        if len(normalized) == 1:
            gk, orientation = normalized[0]
            d = r.to_dict()
            d.update({
                "game_key": gk,
                "game_match_orientation": orientation,
                "game_match_type": match_type,
                "game_match_sign": -1.0 if orientation == "reversed" else 1.0,
            })
            rows.append(d)
        else:
            unmatched.append({
                "source_row": int(idx), "season": r.get("season"), "week": r.get("week"),
                "away": r.get("away"), "home": r.get("home"), "picker": r.get("picker"),
                "candidate_matches": len(normalized), "reason": "ambiguous" if normalized else "no_match",
            })
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def infer_prediction_sign(matched: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matched.empty:
        return matched, pd.DataFrame()
    # Existing prediction lookup in true home-margin coordinates.
    existing = canonical[["game_key", "canonical_model_id", "prediction_home_margin"]].copy()
    existing["canonical_model_id"] = existing["canonical_model_id"].astype(str)
    existing["prediction_home_margin"] = pd.to_numeric(existing["prediction_home_margin"], errors="coerce")
    existing = existing.dropna(subset=["prediction_home_margin"]).drop_duplicates(["game_key", "canonical_model_id"])
    z = matched.merge(existing, on=["game_key", "canonical_model_id"], how="left", suffixes=("", "_existing"))
    market_ref = canonical[["game_key", "market_home_margin"]].copy()
    market_ref["market_home_margin"] = pd.to_numeric(market_ref["market_home_margin"], errors="coerce")
    market_ref = market_ref.groupby("game_key", as_index=False)["market_home_margin"].median().rename(columns={"market_home_margin": "market_home_margin_canonical"})
    z = z.merge(market_ref, on="game_key", how="left")
    z["raw_home_oriented"] = pd.to_numeric(z["prediction_raw"], errors="coerce") * pd.to_numeric(z["game_match_sign"], errors="coerce")

    sign_rows = []
    signs = {}
    # Infer by source table and picker because Tableau can mix calculations from different sheets.
    for key, g in z.groupby(["source_table", "picker"], dropna=False):
        source_table, picker = key
        certainty = set(g["orientation_certainty"].dropna().astype(str))
        if certainty == {"home_margin"} or ("home_margin" in certainty and "infer_from_overlap" not in certainty):
            sign, method, n_overlap, direct_mad, neg_mad, market_corr = 1.0, "explicit_home_margin", 0, np.nan, np.nan, np.nan
        else:
            ov = g.dropna(subset=["prediction_home_margin", "raw_home_oriented"])
            n_overlap = len(ov)
            direct_mad = float(np.nanmedian(np.abs(ov["raw_home_oriented"] - ov["prediction_home_margin"]))) if n_overlap else np.nan
            neg_mad = float(np.nanmedian(np.abs(-ov["raw_home_oriented"] - ov["prediction_home_margin"]))) if n_overlap else np.nan
            market_corr = np.nan
            if n_overlap >= 5 and np.isfinite(direct_mad) and np.isfinite(neg_mad):
                sign = 1.0 if direct_mad <= neg_mad else -1.0
                method = "existing_model_overlap"
            else:
                gg = g.copy()
                # Use the canonical PredictionTracker home-margin market reference,
                # not the Tableau spread field whose sign convention may differ.
                m = pd.to_numeric(gg.get("market_home_margin_canonical"), errors="coerce")
                p = pd.to_numeric(gg["raw_home_oriented"], errors="coerce")
                valid = m.notna() & p.notna()
                if int(valid.sum()) >= 12 and m[valid].std() > 0 and p[valid].std() > 0:
                    market_corr = float(np.corrcoef(p[valid], m[valid])[0, 1])
                    sign = 1.0 if market_corr >= 0 else -1.0
                    method = "canonical_market_direction_correlation"
                else:
                    sign = np.nan
                    method = "unverified"
        signs[(source_table, picker)] = sign
        sign_rows.append({
            "source_table": source_table, "picker": picker, "sign": sign, "method": method,
            "overlap_n": int(n_overlap), "direct_mad": direct_mad, "negative_mad": neg_mad,
            "market_correlation": market_corr,
        })
    z["prediction_sign"] = [signs.get((r.source_table, r.picker), np.nan) for r in z.itertuples(index=False)]
    z["prediction_home_margin_cfb"] = z["raw_home_oriented"] * z["prediction_sign"]
    return z, pd.DataFrame(sign_rows)


def append_to_canonical(matched: pd.DataFrame, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matched.empty:
        return canonical, pd.DataFrame(), pd.DataFrame()
    verified = matched.dropna(subset=["prediction_home_margin_cfb"]).copy()
    existing_keys = set(zip(canonical["game_key"].astype(str), canonical["canonical_model_id"].astype(str)))
    templates = canonical.drop_duplicates("game_key").set_index("game_key")
    existing_pred = (
        canonical[["game_key", "canonical_model_id", "prediction_home_margin"]]
        .drop_duplicates(["game_key", "canonical_model_id"])
        .rename(columns={"prediction_home_margin": "prediction_home_margin_existing"})
    )
    audit_rows = []
    append_rows = []
    for r in verified.itertuples(index=False):
        key = (str(r.game_key), str(r.canonical_model_id))
        if key in existing_keys:
            old = existing_pred[(existing_pred["game_key"].astype(str) == key[0]) & (existing_pred["canonical_model_id"].astype(str) == key[1])]
            oldv = pd.to_numeric(old["prediction_home_margin_existing"], errors="coerce").dropna()
            oldv = float(oldv.iloc[0]) if len(oldv) else np.nan
            audit_rows.append({
                "game_key": r.game_key, "season": r.season, "week": r.week, "away": r.away, "home": r.home,
                "canonical_model_id": r.canonical_model_id, "model_name": r.model_name, "picker": r.picker,
                "prediction_home_margin_existing": oldv,
                "prediction_home_margin_cfbpicker": float(r.prediction_home_margin_cfb),
                "difference_cfb_minus_existing": float(r.prediction_home_margin_cfb - oldv) if np.isfinite(oldv) else np.nan,
                "source_table": r.source_table,
            })
            continue
        if r.game_key not in templates.index:
            continue
        d = templates.loc[r.game_key].to_dict()
        d["game_key"] = r.game_key
        d["canonical_model_id"] = str(r.canonical_model_id)
        d["model_name"] = str(r.model_name)
        d["prediction_home_margin"] = float(r.prediction_home_margin_cfb)
        # PT/template remains the source of record for market/outcome/game metadata.
        d["source"] = "cfbpicker"
        d["selected_source"] = "cfbpicker"
        d["source_model_name"] = str(r.picker)
        d["source_table"] = str(r.source_table)
        d["source_season"] = r.season
        d["source_week"] = r.week
        d["cfbpicker_game_match_type"] = str(r.game_match_type)
        d["cfbpicker_game_match_orientation"] = str(r.game_match_orientation)
        append_rows.append(d)
        existing_keys.add(key)
    appended = pd.DataFrame(append_rows)
    combined = pd.concat([canonical, appended], ignore_index=True, sort=False) if len(appended) else canonical.copy()
    return combined, pd.DataFrame(audit_rows), appended


def update_registry(registry: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return registry
    if registry.empty:
        registry = pd.DataFrame(columns=["canonical_model_id", "model_name"])
    reg = registry.copy()
    for row in mapping[mapping["is_new_canonical_model"] == True].itertuples(index=False):  # noqa: E712
        if "canonical_model_id" in reg and str(row.canonical_model_id) in set(reg["canonical_model_id"].astype(str)):
            continue
        d = {c: np.nan for c in reg.columns}
        d["canonical_model_id"] = row.canonical_model_id
        d["model_name"] = row.model_name
        if "source" in d:
            d["source"] = "cfbpicker"
        reg = pd.concat([reg, pd.DataFrame([d])], ignore_index=True)
    return reg


def rebuild_pairwise(canonical: pd.DataFrame, min_shared: int = 30) -> pd.DataFrame:
    x = canonical.copy()
    for c in ["prediction_home_margin", "market_home_margin", "pair_orientation"]:
        x[c] = pd.to_numeric(x.get(c), errors="coerce")
    orient = x["pair_orientation"].fillna(1.0)
    x["edge"] = (x["prediction_home_margin"] - x["market_home_margin"]) * orient
    pivot = x.pivot_table(index="game_key", columns="canonical_model_id", values="edge", aggfunc="first")
    models = list(map(str, pivot.columns))
    rows = []
    for i, a in enumerate(models):
        for b in models[i+1:]:
            z = pivot[[a, b]].dropna()
            n = len(z)
            corr = float(z[a].corr(z[b])) if n >= min_shared and z[a].std() > 0 and z[b].std() > 0 else np.nan
            rows.append({"model_a": a, "model_b": b, "shared_games": int(n), "edge_correlation": corr})
    return pd.DataFrame(rows)


def load_embedding_api_history(root: Path, season_start: int, season_end: int) -> tuple[pd.DataFrame, dict]:
    """Load the proven Embedding-API tooltip history into the modern importer schema."""
    path = root / "data" / "cfbpicker" / "cfbpicker_history_long.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing API history cache: {path}. Run scripts/scrape_cfbpicker_history_api.py first."
        )
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        return pd.DataFrame(), {"transport": "tableau_embedding_api_tooltip", "source": str(path), "rows": 0}
    required = {"year", "week", "picker", "away", "home", "prediction_home_margin"}
    missing = sorted(required - set(x.columns))
    if missing:
        raise ValueError(f"CFB Picker API history is missing required columns: {missing}")
    z = pd.DataFrame({
        "season": pd.to_numeric(x["year"], errors="coerce"),
        "week": pd.to_numeric(x["week"], errors="coerce"),
        "away": x["away"].astype(str).str.strip(),
        "home": x["home"].astype(str).str.strip(),
        "picker": x["picker"].astype(str).str.strip(),
        "prediction_raw": pd.to_numeric(x["prediction_home_margin"], errors="coerce"),
        "market_home_margin_close": pd.to_numeric(x.get("market_home_margin_close"), errors="coerce"),
        "actual_home_margin": pd.to_numeric(x.get("actual_home_margin"), errors="coerce"),
        "orientation_certainty": "home_margin",
        "source_table": "tableau_embedding_api_tooltip",
    })
    z = z[(z["season"] >= int(season_start)) & (z["season"] <= int(season_end))].copy()
    z = z.dropna(subset=["season", "week", "away", "home", "picker", "prediction_raw"])
    z["away_slug"] = z["away"].map(slug_team)
    z["home_slug"] = z["home"].map(slug_team)
    z = z.drop_duplicates(["season", "week", "away_slug", "home_slug", "picker"], keep="last")
    meta = {
        "transport": "tableau_embedding_api_tooltip",
        "source": str(path),
        "rows": int(len(z)),
        "seasons": sorted(pd.to_numeric(z["season"], errors="coerce").dropna().astype(int).unique().tolist()),
        "pickers": sorted(z["picker"].dropna().astype(str).unique().tolist()),
    }
    return z.reset_index(drop=True), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--season-start", type=int, default=2021)
    ap.add_argument("--season-end", type=int, default=2025)
    # Retained for CLI compatibility with v3.5.31; API collection is now a separate
    # resumable stage using scrape_cfbpicker_history_api.py.
    ap.add_argument("--discover-season", type=int, default=2026)
    ap.add_argument("--discover-week", type=int, default=2)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    derived = root / "data" / "derived"
    pred_path = derived / "model_game_predictions.csv"
    reg_path = derived / "model_registry.csv"
    pair_path = derived / "model_pairwise_metrics.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing canonical predictions: {pred_path}")
    canonical = pd.read_csv(pred_path, low_memory=False)
    registry = pd.read_csv(reg_path, low_memory=False) if reg_path.exists() else pd.DataFrame()
    before = len(canonical)

    try:
        parsed, diagnostics = load_embedding_api_history(root, args.season_start, args.season_end)
    except Exception as exc:
        print(json.dumps({
            "status": "api_history_missing_or_invalid",
            "message": f"{type(exc).__name__}: {exc}",
            "next": "Run scripts/scrape_cfbpicker_history_api.py for the desired seasons, then rerun this importer.",
        }, indent=2))
        return 2
    if parsed.empty:
        print(json.dumps({
            "status": "no_historical_rows",
            "message": f"Embedding-API cache had no rows in requested seasons {args.season_start}-{args.season_end}.",
            "source": diagnostics.get("source"),
        }, indent=2))
        return 2
    parsed, mapping = map_picker_models(parsed, canonical, registry)
    matched, unmatched = match_games(parsed, canonical)
    oriented, sign_audit = infer_prediction_sign(matched, canonical)
    combined, overlap, appended = append_to_canonical(oriented, canonical)
    updated_registry = update_registry(registry, mapping)

    derived.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(derived / "cfbpicker_model_mapping.csv", index=False)
    unmatched.to_csv(derived / "cfbpicker_unmatched_games.csv", index=False)
    sign_audit.to_csv(derived / "cfbpicker_orientation_audit.csv", index=False)
    overlap.to_csv(derived / "cfbpicker_overlap_audit.csv", index=False)
    overlap_by_model = pd.DataFrame()
    if len(overlap):
        oo = overlap.copy()
        oo["abs_difference"] = pd.to_numeric(oo["difference_cfb_minus_existing"], errors="coerce").abs()
        rows = []
        for (mid, mname), g in oo.groupby(["canonical_model_id", "model_name"], dropna=False):
            a = pd.to_numeric(g["prediction_home_margin_existing"], errors="coerce")
            b = pd.to_numeric(g["prediction_home_margin_cfbpicker"], errors="coerce")
            valid = a.notna() & b.notna()
            corr = float(a[valid].corr(b[valid])) if int(valid.sum()) >= 3 and a[valid].std() > 0 and b[valid].std() > 0 else np.nan
            diff = (b - a).abs()
            rows.append({
                "canonical_model_id": mid, "model_name": mname, "overlap_games": int(valid.sum()),
                "mean_abs_difference": float(diff[valid].mean()) if int(valid.sum()) else np.nan,
                "median_abs_difference": float(diff[valid].median()) if int(valid.sum()) else np.nan,
                "pct_exact_within_0_01": float((diff[valid] <= 0.01).mean() * 100.0) if int(valid.sum()) else np.nan,
                "prediction_correlation": corr,
            })
        overlap_by_model = pd.DataFrame(rows).sort_values(["overlap_games", "model_name"], ascending=[False, True])
        overlap_by_model.to_csv(derived / "cfbpicker_overlap_by_model.csv", index=False)

    coverage_rows = []
    if len(mapping):
        for mr in mapping.itertuples(index=False):
            psub = parsed[parsed["picker"].astype(str).eq(str(mr.picker))]
            msub = matched[matched["picker"].astype(str).eq(str(mr.picker))] if len(matched) else pd.DataFrame()
            osub = overlap[overlap["picker"].astype(str).eq(str(mr.picker))] if len(overlap) else pd.DataFrame()
            asub = appended[appended.get("source_model_name", pd.Series(index=appended.index, dtype=object)).astype(str).eq(str(mr.picker))] if len(appended) else pd.DataFrame()
            coverage_rows.append({
                "picker": mr.picker, "canonical_model_id": mr.canonical_model_id, "model_name": mr.model_name,
                "is_new_canonical_model": bool(mr.is_new_canonical_model),
                "parsed_rows": int(len(psub)), "matched_rows": int(len(msub)),
                "overlap_rows": int(len(osub)), "appended_rows": int(len(asub)),
            })
    coverage = pd.DataFrame(coverage_rows)
    if len(coverage):
        coverage.to_csv(derived / "cfbpicker_model_coverage.csv", index=False)

    if len(appended):
        appended.to_csv(derived / "cfbpicker_rows_appended.csv", index=False)

    summary = {
        "created_at_utc": utc_now(),
        "season_start": args.season_start,
        "season_end": args.season_end,
        "api_prediction_rows_loaded": int(len(parsed)),
        "cfbpicker_transport": diagnostics.get("transport"),
        "discovered_picker_names": diagnostics.get("pickers", []),
        "mapping_rows": int(len(mapping)),
        "new_canonical_models": int(mapping["is_new_canonical_model"].sum()) if len(mapping) else 0,
        "matched_rows": int(len(matched)),
        "unmatched_rows": int(len(unmatched)),
        "orientation_verified_rows": int(oriented["prediction_home_margin_cfb"].notna().sum()) if len(oriented) else 0,
        "orientation_unverified_rows": int(oriented["prediction_home_margin_cfb"].isna().sum()) if len(oriented) else 0,
        "canonical_rows_before": int(before),
        "canonical_rows_after": int(len(combined)),
        "rows_appended": int(len(appended)),
        "overlap_rows_not_duplicated": int(len(overlap)),
        "overlap_models": int(overlap["canonical_model_id"].nunique()) if len(overlap) else 0,
        "dry_run": bool(args.dry_run),
    }
    (derived / "cfbpicker_import_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.dry_run:
        combined.to_csv(pred_path, index=False)
        updated_registry.to_csv(reg_path, index=False)
        pairwise = rebuild_pairwise(combined)
        pairwise.to_csv(pair_path, index=False)

    print(json.dumps(summary, indent=2))
    if len(sign_audit):
        print("\nOrientation audit:")
        print(sign_audit.to_string(index=False, max_rows=40))
    if len(mapping):
        print("\nModel mapping:")
        print(mapping.to_string(index=False, max_rows=80))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
