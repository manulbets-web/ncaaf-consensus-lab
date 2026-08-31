from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable
import math

import numpy as np
import pandas as pd

from committee import (
    _matrix_and_meta,
    _combo_forecast_arrays,
    _meta_game_frame,
    _signal,
    _wilson_lower,
    _unit_result,
    analyze_finalist_portfolio,
)
from streamlined_engine import (
    CombinationSearchConfig,
    brute_force_combination_search,
    individual_model_performance,
)


LINE_REFS = (
    ("Open", "open_margin"),
    ("Midweek", "midweek_margin"),
    ("Close (PT Updated/final)", "close_margin"),
)
LINE_REF_MAP = dict(LINE_REFS)
SHORT_REF = {
    "Open": "open",
    "Midweek": "midweek",
    "Close (PT Updated/final)": "close",
}


def _period_mask(data: pd.DataFrame, periods: Iterable[tuple[int, int]]) -> pd.Series:
    wanted = set((int(y), int(w)) for y, w in periods)
    yy = pd.to_numeric(data.get("season"), errors="coerce")
    ww = pd.to_numeric(data.get("week"), errors="coerce")
    return pd.Series([
        (int(y), int(w)) in wanted if pd.notna(y) and pd.notna(w) else False
        for y, w in zip(yy, ww)
    ], index=data.index)


def data_for_line_reference(data: pd.DataFrame, line_history: pd.DataFrame, line_col: str) -> pd.DataFrame:
    """Return a copy of canonical data with market_margin replaced by one PT line field.

    Only games that map to the requested PT line receive a market margin. Downstream
    routines already require finite market/actual values, so unmatched games are
    naturally excluded without contaminating another line reference.
    """
    if line_col not in {c for _, c in LINE_REFS}:
        raise ValueError(f"Unknown line reference column: {line_col}")
    if data is None or data.empty or line_history is None or line_history.empty:
        return pd.DataFrame(columns=list(data.columns) if isinstance(data, pd.DataFrame) else [])
    lh = line_history[["game_key", line_col]].drop_duplicates("game_key").copy()
    lh[line_col] = pd.to_numeric(lh[line_col], errors="coerce")
    out = data.copy().merge(lh, on="game_key", how="left")
    out["market_margin_original"] = pd.to_numeric(out.get("market_margin"), errors="coerce")
    out["market_margin"] = pd.to_numeric(out[line_col], errors="coerce")
    out["line_reference_market"] = line_col
    return out


def individual_line_reference_by_period(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    discovery_periods: Iterable[tuple[int, int]],
    holdout_periods: Iterable[tuple[int, int]],
    *,
    standard_price: int = -110,
) -> pd.DataFrame:
    rows = []
    for period_name, periods in (("Discovery", tuple(discovery_periods)), ("Holdout", tuple(holdout_periods))):
        if not periods:
            continue
        for label, col in LINE_REFS:
            ref = data_for_line_reference(data, line_history, col)
            if ref.empty:
                continue
            q = ref.loc[_period_mask(ref, periods)].copy()
            hist = individual_model_performance(q, standard_price=standard_price).get("overall", pd.DataFrame()).copy()
            if hist.empty:
                continue
            hist["period"] = period_name
            hist["line_reference"] = label
            rows.append(hist)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _fixed_outcome_stats(side: np.ndarray, actual: np.ndarray, grade_market: np.ndarray, selected: np.ndarray, *, standard_price=-110) -> dict:
    side = np.asarray(side, dtype=float)
    actual = np.asarray(actual, dtype=float)
    grade_market = np.asarray(grade_market, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    cover = actual - grade_market
    valid = selected & np.isfinite(side) & (np.abs(side) > 0) & np.isfinite(cover)
    push = valid & (np.abs(cover) <= 1e-12)
    graded = valid & ~push
    win = graded & ((side * cover) > 0)
    loss = graded & ((side * cover) < 0)
    wins = int(win.sum()); losses = int(loss.sum()); pushes = int(push.sum())
    bets = wins + losses
    units = float(_unit_result(win, loss, standard_price).sum())
    return {
        "selected_n": int(selected.sum()),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": wins / bets if bets else np.nan,
        "units": units,
        "roi": units / bets if bets else np.nan,
        "wilson_low": _wilson_lower(wins, bets),
    }


def _entity_forecast_frames(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    combinations: list[dict],
    discovery_periods,
    holdout_periods,
    combo_k_selected: pd.DataFrame,
    meta_selected: pd.DataFrame,
    *,
    min_available_models=3,
    min_meta_communities=2,
):
    """Yield one forecast frame per period/entity, independent of market reference."""
    kmap = {}
    if isinstance(combo_k_selected, pd.DataFrame) and len(combo_k_selected):
        for r in combo_k_selected.itertuples(index=False):
            try:
                kmap[int(getattr(r, "portfolio_combo"))] = float(getattr(r, "selected_k"))
            except Exception:
                pass
    meta_k = 0.75
    if isinstance(meta_selected, pd.DataFrame) and len(meta_selected):
        q = meta_selected[meta_selected["method"].astype(str).eq("Diversified META")]
        if len(q):
            v = pd.to_numeric(pd.Series([q.iloc[0].get("selected_k")]), errors="coerce").iloc[0]
            if np.isfinite(v):
                meta_k = float(v)

    union = list(dict.fromkeys(mid for c in combinations for mid in map(str, c.get("model_ids", []))))
    lh_index = line_history.drop_duplicates("game_key").set_index("game_key")
    frames = []
    for period_name, periods in (("Discovery", tuple(discovery_periods)), ("Holdout", tuple(holdout_periods))):
        if not periods:
            continue
        pred, meta = _matrix_and_meta(data, union, periods)
        if pred.empty:
            continue
        lh = lh_index.reindex(pred.index)
        actual = pd.to_numeric(meta["actual_margin"], errors="coerce")
        for i, c in enumerate(combinations, start=1):
            count, mean, sd = _combo_forecast_arrays(pred, c.get("model_ids", []), min_available_models)
            k = float(kmap.get(i, c.get("k", 0.75)))
            f = pd.DataFrame({
                "game_key": pred.index.astype(str),
                "period": period_name,
                "entity": f"C{i}",
                "search_rank": int(c.get("rank", i)),
                "community": int(c.get("community", i)),
                "selected_k": k,
                "forecast": mean,
                "forecast_sd": sd,
                "actual_margin": actual.reindex(pred.index).to_numpy(float),
                "active_gate": np.isfinite(mean),
            })
            for _, col in LINE_REFS:
                f[col] = pd.to_numeric(lh[col], errors="coerce").to_numpy(float)
            frames.append(f)

        mf = _meta_game_frame(
            data, combinations, periods,
            min_available_models=min_available_models,
            diversified=True,
        )
        if len(mf):
            mfi = mf.set_index("game_key")
            lh2 = lh_index.reindex(mfi.index)
            f = pd.DataFrame({
                "game_key": mfi.index.astype(str),
                "period": period_name,
                "entity": "Diversified META",
                "search_rank": np.nan,
                "community": np.nan,
                "selected_k": meta_k,
                "forecast": pd.to_numeric(mfi["meta_mean"], errors="coerce").to_numpy(float),
                "forecast_sd": pd.to_numeric(mfi["meta_sd"], errors="coerce").to_numpy(float),
                "actual_margin": pd.to_numeric(mfi["actual_margin"], errors="coerce").to_numpy(float),
                "active_gate": pd.to_numeric(mfi["active_units"], errors="coerce").fillna(0).to_numpy(float) >= int(min_meta_communities),
            })
            for _, col in LINE_REFS:
                f[col] = pd.to_numeric(lh2[col], errors="coerce").to_numpy(float)
            frames.append(f)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_selection_detail(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    combinations: list[dict],
    discovery_periods,
    holdout_periods,
    combo_k_selected: pd.DataFrame,
    meta_selected: pd.DataFrame,
    *,
    min_available_models=3,
    min_meta_communities=2,
) -> pd.DataFrame:
    base = _entity_forecast_frames(
        data, line_history, combinations, discovery_periods, holdout_periods,
        combo_k_selected, meta_selected,
        min_available_models=min_available_models,
        min_meta_communities=min_meta_communities,
    )
    if base.empty:
        return base
    rows = []
    for label, col in LINE_REFS:
        q = base.copy()
        market = pd.to_numeric(q[col], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(q["forecast"], errors="coerce").to_numpy(float)
        sd = pd.to_numeric(q["forecast_sd"], errors="coerce").to_numpy(float)
        k = pd.to_numeric(q["selected_k"], errors="coerce").to_numpy(float)
        edge = pred - market
        sig = _signal(edge, sd)
        gate = q["active_gate"].fillna(False).astype(bool).to_numpy()
        selected = gate & np.isfinite(sig) & np.isfinite(k) & (sig >= k) & np.isfinite(edge) & (np.abs(edge) > 1e-12)
        q["selection_reference"] = label
        q["selection_market"] = market
        q["edge"] = edge
        q["signal"] = sig
        q["selected"] = selected
        q["side"] = np.where(selected, np.sign(edge), 0.0)
        rows.append(q)
    return pd.concat(rows, ignore_index=True, sort=False)


def fixed_bet_repricing(selection_detail: pd.DataFrame, *, standard_price=-110) -> pd.DataFrame:
    """Freeze game+side selection at one reference, then re-grade at every price."""
    if selection_detail is None or selection_detail.empty:
        return pd.DataFrame()
    rows = []
    keys = ["period", "entity", "selection_reference", "selected_k"]
    for key, g in selection_detail.groupby(keys, dropna=False):
        period, entity, selection_ref, k = key
        selected = g["selected"].fillna(False).astype(bool).to_numpy()
        side = pd.to_numeric(g["side"], errors="coerce").to_numpy(float)
        actual = pd.to_numeric(g["actual_margin"], errors="coerce").to_numpy(float)
        sel_market = pd.to_numeric(g["selection_market"], errors="coerce").to_numpy(float)
        for grade_label, grade_col in LINE_REFS:
            grade_market = pd.to_numeric(g[grade_col], errors="coerce").to_numpy(float)
            st = _fixed_outcome_stats(side, actual, grade_market, selected, standard_price=standard_price)
            both = selected & np.isfinite(sel_market) & np.isfinite(grade_market)
            signed_move = side * (grade_market - sel_market)
            st.update({
                "period": period,
                "entity": entity,
                "selected_k": k,
                "selection_reference": selection_ref,
                "grade_reference": grade_label,
                "mean_signed_market_move": float(np.nanmean(signed_move[both])) if both.any() else np.nan,
                "median_signed_market_move": float(np.nanmedian(signed_move[both])) if both.any() else np.nan,
            })
            rows.append(st)
    return pd.DataFrame(rows)


def bet_set_overlap(selection_detail: pd.DataFrame) -> pd.DataFrame:
    if selection_detail is None or selection_detail.empty:
        return pd.DataFrame()
    labels = [x[0] for x in LINE_REFS]
    pairs = [(labels[0], labels[1]), (labels[0], labels[2]), (labels[1], labels[2])]
    rows = []
    for (period, entity), g in selection_detail.groupby(["period", "entity"], dropna=False):
        sets = {}
        gamesets = {}
        sides = {}
        for label in labels:
            q = g[(g["selection_reference"].astype(str) == label) & g["selected"].fillna(False)].copy()
            pairs_set = set((str(r.game_key), int(np.sign(float(r.side)))) for r in q.itertuples(index=False) if np.isfinite(float(r.side)) and abs(float(r.side)) > 0)
            sets[label] = pairs_set
            gamesets[label] = set(x[0] for x in pairs_set)
            sides[label] = {x[0]: x[1] for x in pairs_set}
        for a, b in pairs:
            union = sets[a] | sets[b]
            inter = sets[a] & sets[b]
            common_games = gamesets[a] & gamesets[b]
            flips = sum(1 for game in common_games if sides[a].get(game) != sides[b].get(game))
            rows.append({
                "period": period,
                "entity": entity,
                "reference_a": a,
                "reference_b": b,
                "bets_a": len(sets[a]),
                "bets_b": len(sets[b]),
                "same_side_common": len(inter),
                "common_games": len(common_games),
                "side_flips": flips,
                "jaccard": len(inter) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def signal_migration_detail(selection_detail: pd.DataFrame) -> pd.DataFrame:
    if selection_detail is None or selection_detail.empty:
        return pd.DataFrame()
    id_cols = ["period", "entity", "game_key", "selected_k"]
    base_cols = id_cols + ["forecast", "forecast_sd", "actual_margin", "open_margin", "midweek_margin", "close_margin"]
    base = selection_detail.sort_values("selection_reference").drop_duplicates(id_cols)[base_cols].copy()
    for label, short in (("Open", "open"), ("Midweek", "midweek"), ("Close (PT Updated/final)", "close")):
        q = selection_detail[selection_detail["selection_reference"].astype(str).eq(label)][id_cols + ["edge", "signal", "selected", "side"]].copy()
        q = q.rename(columns={"edge": f"{short}_edge", "signal": f"{short}_signal", "selected": f"{short}_bet", "side": f"{short}_side"})
        base = base.merge(q, on=id_cols, how="left")
    for c in ["open_bet", "midweek_bet", "close_bet"]:
        base[c] = base[c].fillna(False).astype(bool)

    def cls(r):
        bits = (bool(r.open_bet), bool(r.midweek_bet), bool(r.close_bet))
        return {
            (True, False, False): "Open only",
            (True, True, False): "Open + Midweek",
            (True, False, True): "Open + Close",
            (False, True, False): "Midweek only",
            (False, True, True): "Midweek + Close",
            (False, False, True): "Close only",
            (True, True, True): "All three",
            (False, False, False): "None",
        }[bits]
    base["migration_class"] = base.apply(cls, axis=1)

    actual = pd.to_numeric(base["actual_margin"], errors="coerce")
    for short, market_col in (("open", "open_margin"), ("midweek", "midweek_margin"), ("close", "close_margin")):
        side = pd.to_numeric(base[f"{short}_side"], errors="coerce")
        cover = actual - pd.to_numeric(base[market_col], errors="coerce")
        selected = base[f"{short}_bet"]
        base[f"{short}_outcome"] = np.where(~selected | cover.isna(), np.nan, np.where(cover.abs() <= 1e-12, 0.0, np.where(side * cover > 0, 1.0, -1.0)))

    open_side = pd.to_numeric(base["open_side"], errors="coerce")
    mid_side = pd.to_numeric(base["midweek_side"], errors="coerce")
    open_move = pd.to_numeric(base["close_margin"], errors="coerce") - pd.to_numeric(base["open_margin"], errors="coerce")
    mid_move = pd.to_numeric(base["close_margin"], errors="coerce") - pd.to_numeric(base["midweek_margin"], errors="coerce")
    base["open_to_close_move"] = open_move
    base["open_clv_pts"] = np.where(base["open_bet"], open_side * open_move, np.nan)
    base["midweek_clv_pts"] = np.where(base["midweek_bet"], mid_side * mid_move, np.nan)
    base["abs_open_close_move"] = open_move.abs()

    sel_sides = base[["open_side", "midweek_side", "close_side"]].where(base[["open_bet", "midweek_bet", "close_bet"]].to_numpy())
    def any_flip(row):
        vals = [int(np.sign(v)) for v in row if pd.notna(v) and abs(float(v)) > 0]
        return len(set(vals)) > 1
    base["side_flip_any"] = sel_sides.apply(any_flip, axis=1)
    return base


def _outcome_rate(s: pd.Series) -> tuple[int, int, int, int, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    wins = int((x > 0).sum()); losses = int((x < 0).sum()); pushes = int((x == 0).sum())
    bets = wins + losses
    return bets, wins, losses, pushes, wins / bets if bets else np.nan


def signal_migration_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame()
    rows = []
    q = detail[detail["migration_class"].astype(str).ne("None")].copy()
    for (period, entity, cls), g in q.groupby(["period", "entity", "migration_class"], dropna=False):
        row = {"period": period, "entity": entity, "migration_class": cls, "games": int(len(g))}
        for short in ("open", "midweek", "close"):
            bets, wins, losses, pushes, ats = _outcome_rate(g[f"{short}_outcome"])
            row.update({f"{short}_bets": bets, f"{short}_ats_pct": ats})
        clv = pd.to_numeric(g["open_clv_pts"], errors="coerce").dropna()
        row["mean_open_clv_pts"] = float(clv.mean()) if len(clv) else np.nan
        row["median_open_clv_pts"] = float(clv.median()) if len(clv) else np.nan
        row["positive_open_clv_pct"] = float((clv > 0).mean()) if len(clv) else np.nan
        row["side_flip_games"] = int(g["side_flip_any"].fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def opening_line_qc(line_history: pd.DataFrame, data: pd.DataFrame, *, move_threshold: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if line_history is None or line_history.empty:
        return pd.DataFrame(), pd.DataFrame()
    lh = line_history.drop_duplicates("game_key").copy()
    # Attach human-readable matchup and actual margin if available.
    meta_cols = [c for c in ["game_key", "season", "week", "road", "away", "home", "actual_margin"] if c in data.columns]
    if meta_cols:
        meta = data[meta_cols].sort_values([c for c in ["season", "week", "game_key"] if c in meta_cols]).drop_duplicates("game_key")
        # season/week already exist in line history; keep one copy.
        merge_cols = [c for c in meta_cols if c not in {"season", "week"}]
        lh = lh.merge(meta[merge_cols], on="game_key", how="left")
    away_col = "road" if "road" in lh.columns else ("away" if "away" in lh.columns else None)
    if away_col:
        lh["game"] = lh[away_col].astype(str) + " @ " + lh.get("home", "").astype(str)
    else:
        lh["game"] = lh["game_key"].astype(str)
    for c in ["open_margin", "midweek_margin", "close_margin", "actual_margin"]:
        if c in lh.columns:
            lh[c] = pd.to_numeric(lh[c], errors="coerce")
    lh["open_to_midweek"] = lh["midweek_margin"] - lh["open_margin"]
    lh["midweek_to_close"] = lh["close_margin"] - lh["midweek_margin"]
    lh["open_to_close"] = lh["close_margin"] - lh["open_margin"]
    lh["abs_open_close_move"] = lh["open_to_close"].abs()
    flags = []
    for r in lh.itertuples(index=False):
        f = []
        o = getattr(r, "open_margin", np.nan); m = getattr(r, "midweek_margin", np.nan); c = getattr(r, "close_margin", np.nan)
        if not np.isfinite(o): f.append("missing open")
        if not np.isfinite(m): f.append("missing midweek")
        if not np.isfinite(c): f.append("missing close")
        if np.isfinite(o) and np.isfinite(c) and abs(c-o) >= float(move_threshold): f.append(f"open→close ≥{float(move_threshold):g}")
        if np.isfinite(o) and abs(o) >= 50: f.append("|open| ≥50")
        if np.isfinite(o) and np.isfinite(c) and np.sign(o) != np.sign(c) and abs(c-o) >= 7: f.append("large favorite flip")
        flags.append("; ".join(f))
    lh["qc_flags"] = flags
    lh["flagged"] = lh["qc_flags"].astype(str).str.len() > 0

    summary_rows = []
    actual = pd.to_numeric(lh.get("actual_margin"), errors="coerce") if "actual_margin" in lh.columns else pd.Series(np.nan, index=lh.index)
    for label, col in LINE_REFS:
        x = pd.to_numeric(lh[col], errors="coerce")
        valid = x.notna()
        scored = valid & actual.notna()
        summary_rows.append({
            "line_reference": label,
            "games_available": int(valid.sum()),
            "availability_pct": float(valid.mean()) if len(valid) else np.nan,
            "market_mae": float((x[scored]-actual[scored]).abs().mean()) if scored.any() else np.nan,
            "median_abs_error": float((x[scored]-actual[scored]).abs().median()) if scored.any() else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary["flagged_games"] = int(lh["flagged"].sum())
    summary["move_threshold"] = float(move_threshold)
    lh = lh.sort_values(["flagged", "abs_open_close_move"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return summary, lh


def _rank_candidates_for_reference(
    data_ref: pd.DataFrame,
    discovery_periods,
    live_ids: Iterable[str],
    *,
    pool_n=35,
    pool_min_bets=25,
    pool_metric="wilson",
    standard_price=-110,
) -> pd.DataFrame:
    q = data_ref.loc[_period_mask(data_ref, discovery_periods)].copy()
    hist = individual_model_performance(q, standard_price=standard_price).get("overall", pd.DataFrame()).copy()
    if hist.empty:
        return hist
    live = set(map(str, live_ids))
    hist["canonical_model_id"] = hist["canonical_model_id"].astype(str)
    hist = hist[hist["canonical_model_id"].isin(live)].copy()
    hist = hist[pd.to_numeric(hist["bets"], errors="coerce").fillna(0) >= int(pool_min_bets)].copy()
    if pool_metric == "mae":
        hist = hist.sort_values(["mae", "bets", "wilson_low"], ascending=[True, False, False], na_position="last")
    else:
        col = {"ats":"ats_pct", "roi":"roi", "wilson":"wilson_low"}.get(pool_metric, "wilson_low")
        hist = hist.sort_values([col, "bets", "wilson_low"], ascending=[False, False, False], na_position="last")
    hist = hist.head(int(pool_n)).reset_index(drop=True)
    hist["pool_rank"] = np.arange(1, len(hist)+1)
    return hist


def run_line_specific_pipelines(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    live_ids: Iterable[str],
    model_name_map: dict[str, str],
    discovery_periods,
    holdout_periods,
    *,
    pool_n=35,
    pool_min_bets=25,
    min_size=3,
    max_size=6,
    search_k=0.75,
    min_available_models=3,
    min_search_bets=50,
    finalists=50,
    overlap_threshold=0.50,
    min_meta_communities=2,
    thresholds=(0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00),
    max_combinations=10_000_000,
    standard_price=-110,
    progress_callback: Callable[[int,int,str],None] | None = None,
) -> dict:
    """Build three genuinely independent discovery pipelines, one per line reference."""
    results = {}
    summary_rows = []
    candidate_rows = []
    finalist_rows = []
    live_ids = list(dict.fromkeys(map(str, live_ids)))

    # Candidate pool sizes may differ slightly by reference; estimate total work
    # after rankings are known, then report cumulative combination progress.
    prepared = []
    total_work = 0
    for label, col in LINE_REFS:
        ref_data = data_for_line_reference(data, line_history, col)
        ranked = _rank_candidates_for_reference(
            ref_data, discovery_periods, live_ids,
            pool_n=pool_n, pool_min_bets=pool_min_bets,
            pool_metric="wilson", standard_price=standard_price,
        )
        ids = ranked["canonical_model_id"].astype(str).tolist() if len(ranked) else []
        hi = min(int(max_size), len(ids))
        if len(ids) >= int(min_size):
            from streamlined_engine import combination_count
            nwork = combination_count(len(ids), int(min_size), hi)
        else:
            nwork = 0
        prepared.append((label, col, ref_data, ranked, ids, hi, nwork))
        total_work += nwork
    done_before = 0

    for label, col, ref_data, ranked, ids, hi, nwork in prepared:
        if len(ranked):
            tmp = ranked.copy()
            tmp["line_reference"] = label
            candidate_rows.append(tmp)
        if len(ids) < int(min_size):
            summary_rows.append({"line_reference": label, "status": "Too few eligible models", "candidate_models": len(ids)})
            done_before += nwork
            continue
        seasons = tuple(sorted(set(y for y, _ in tuple(discovery_periods) + tuple(holdout_periods))))
        cfg = CombinationSearchConfig(
            search_seasons=seasons,
            validation_seasons=seasons,
            search_periods=tuple(discovery_periods),
            validation_periods=tuple(holdout_periods),
            min_size=int(min_size), max_size=int(hi), primary_k=float(search_k),
            min_available_models=int(min_available_models), min_search_bets=int(min_search_bets),
            min_seasons_represented=1, min_distinct_weeks=1, ranking_metric="ats",
            standard_price=int(standard_price), chunk_size=512, top_n=int(finalists),
            max_combinations=int(max_combinations),
        )
        base_done = done_before
        def progress(done, total, msg, label=label, base_done=base_done):
            if progress_callback is not None:
                progress_callback(base_done + int(done), max(1,total_work), f"{label}: {msg}")
        search = brute_force_combination_search(ref_data, ids, model_name_map, cfg, progress_callback=progress)
        top = search.get("top", pd.DataFrame()).head(int(finalists)).copy()
        combos = []
        for j, r in top.reset_index(drop=True).iterrows():
            mids = [x for x in str(r.get("model_ids", "")).split("|") if x]
            if not mids and isinstance(r.get("model_ids"), (list,tuple)):
                mids = list(map(str,r.get("model_ids")))
            combos.append({"rank": int(r.get("search_rank", j+1)), "model_ids": mids})
        analysis = analyze_finalist_portfolio(
            ref_data, combos, discovery_periods, holdout_periods,
            min_available_models=int(min_available_models), thresholds=thresholds,
            combo_min_bets=int(min_search_bets), meta_min_bets=int(min_search_bets),
            overlap_threshold=float(overlap_threshold), min_meta_communities=int(min_meta_communities),
            standard_price=int(standard_price), line_history=None,
        )
        combos = analysis.get("combinations", combos)
        meta_summary = analysis.get("meta_summary", pd.DataFrame())
        row = {
            "line_reference": label,
            "status": "ok",
            "candidate_models": len(ids),
            "evaluated_combinations": int(search.get("evaluated_combinations", 0)),
            "eligible_combinations": int(search.get("eligible_combinations", 0)),
            "finalists": len(combos),
            "communities": int(analysis.get("overlap_summary", {}).get("communities", 0)),
            "max_effective_model_weight": float(analysis.get("overlap_summary", {}).get("max_effective_model_weight", np.nan)),
        }
        if isinstance(meta_summary, pd.DataFrame) and len(meta_summary):
            for period in ("Discovery", "Holdout"):
                q = meta_summary[(meta_summary["method"].astype(str)=="Diversified META") & (meta_summary["period"].astype(str)==period)]
                if len(q):
                    rr = q.iloc[0]
                    p = period.lower()
                    row.update({
                        "meta_k": float(rr.get("selected_k", np.nan)),
                        f"{p}_bets": int(rr.get("bets",0)),
                        f"{p}_ats_pct": float(rr.get("ats_pct",np.nan)),
                        f"{p}_roi": float(rr.get("roi",np.nan)),
                        f"{p}_wilson_low": float(rr.get("wilson_low",np.nan)),
                    })
        summary_rows.append(row)
        if len(top):
            t = top.copy()
            t["line_reference"] = label
            finalist_rows.append(t)
        results[label] = {"ranked": ranked, "combinations": combos}
        done_before += nwork
        if progress_callback is not None:
            progress_callback(done_before, max(1,total_work), f"{label}: complete")

    candidate_table = pd.concat(candidate_rows, ignore_index=True, sort=False) if candidate_rows else pd.DataFrame()
    finalist_table = pd.concat(finalist_rows, ignore_index=True, sort=False) if finalist_rows else pd.DataFrame()

    # Cross-reference candidate-pool overlap.
    overlap_rows = []
    labels = [x[0] for x in LINE_REFS]
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            aset = set(results.get(a, {}).get("ranked", pd.DataFrame()).get("canonical_model_id", pd.Series(dtype=str)).astype(str))
            bset = set(results.get(b, {}).get("ranked", pd.DataFrame()).get("canonical_model_id", pd.Series(dtype=str)).astype(str))
            union = aset | bset
            overlap_rows.append({
                "reference_a": a, "reference_b": b,
                "candidate_a": len(aset), "candidate_b": len(bset),
                "shared_candidates": len(aset & bset),
                "candidate_jaccard": len(aset & bset)/len(union) if union else np.nan,
            })
    candidate_overlap = pd.DataFrame(overlap_rows)

    # Exact finalist-membership overlap + nearest-set similarity across pipelines.
    final_overlap_rows = []
    def combo_sets(label):
        return [frozenset(map(str,c.get("model_ids",[]))) for c in results.get(label,{}).get("combinations",[])]
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            aa = combo_sets(a); bb = combo_sets(b)
            sa=set(aa); sb=set(bb); union=sa|sb
            nearest=[]
            for x in aa:
                best=0.0
                for y in bb:
                    u=x|y
                    if u: best=max(best,len(x&y)/len(u))
                nearest.append(best)
            final_overlap_rows.append({
                "reference_a":a,"reference_b":b,"finalists_a":len(aa),"finalists_b":len(bb),
                "exact_shared_finalists":len(sa&sb),
                "exact_finalist_jaccard":len(sa&sb)/len(union) if union else np.nan,
                "mean_nearest_combo_jaccard":float(np.mean(nearest)) if nearest else np.nan,
            })
    finalist_overlap = pd.DataFrame(final_overlap_rows)
    return {
        "summary": pd.DataFrame(summary_rows),
        "candidate_table": candidate_table,
        "candidate_overlap": candidate_overlap,
        "finalist_table": finalist_table,
        "finalist_overlap": finalist_overlap,
        "results": results,
        "total_work": total_work,
    }
