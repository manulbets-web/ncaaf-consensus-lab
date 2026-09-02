from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable
import json
import math

import numpy as np
import pandas as pd

from committee import analyze_finalist_portfolio, _wilson_lower
from line_movement import (
    _period_mask,
    _rank_candidates_for_reference,
    data_for_line_reference,
)
from streamlined_engine import CombinationSearchConfig, combination_count, _make_combo_matrix


FORMAL_ANCHOR_GRID = tuple(np.round(np.arange(0.0, 2.01, 0.25), 2))
FORMAL_META_K_GRID = tuple(np.round(np.arange(0.25, 2.01, 0.25), 2))
FORMAL_FALLBACK_ANCHOR = 0.75


def _build_formal_folds(
    close_data: pd.DataFrame,
    *,
    period_scope: Iterable[tuple[int, int]] | None = None,
    block_size: int = 6,
    n_folds: int = 8,
    min_discovery_periods: int = 24,
    min_games_per_period: int = 10,
    min_available_models: int = 3,
) -> tuple[list[dict], pd.DataFrame]:
    """Create close-line OOS folds without requiring Open/Midweek availability."""
    if close_data is None or close_data.empty:
        return [], pd.DataFrame()
    z = close_data.copy()
    for c in ["season", "week", "prediction_margin", "market_margin", "actual_margin"]:
        z[c] = pd.to_numeric(z.get(c), errors="coerce")
    z = z[
        z["season"].notna() & z["week"].notna()
        & z["prediction_margin"].notna() & z["market_margin"].notna() & z["actual_margin"].notna()
    ].copy()
    if period_scope:
        wanted = set((int(y), int(w)) for y, w in period_scope)
        z = z[[((int(y), int(w)) in wanted) for y, w in zip(z["season"], z["week"])]]
    if z.empty:
        return [], pd.DataFrame()
    per_game = (
        z.groupby(["season", "week", "game_key"], as_index=False)
        .agg(available_models=("canonical_model_id", "nunique"))
    )
    per_game["scorable"] = pd.to_numeric(per_game["available_models"], errors="coerce").fillna(0).ge(int(min_available_models))
    coverage = (
        per_game.groupby(["season", "week"], as_index=False)
        .agg(games=("game_key", "nunique"), scorable_games=("scorable", "sum"))
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    coverage["season"] = coverage["season"].astype(int)
    coverage["week"] = coverage["week"].astype(int)
    coverage["usable"] = coverage["scorable_games"] >= int(min_games_per_period)
    usable = [(int(r.season), int(r.week)) for r in coverage.itertuples(index=False) if bool(r.usable)]
    if period_scope:
        scope = sorted(set((int(y), int(w)) for y, w in period_scope))
    else:
        scope = sorted(set((int(y), int(w)) for y, w in zip(z["season"], z["week"])))
    block_size = max(1, int(block_size)); n_folds = max(1, int(n_folds)); min_discovery_periods = max(1, int(min_discovery_periods))
    max_folds = max(0, (len(usable) - min_discovery_periods) // block_size)
    take = min(n_folds, max_folds)
    if take <= 0:
        return [], coverage
    first_idx = len(usable) - take * block_size
    folds = []
    for j in range(take):
        start = first_idx + j * block_size
        validation = tuple(usable[start:start + block_size])
        if len(validation) < block_size:
            continue
        first_val = validation[0]
        discovery = tuple(p for p in scope if p < first_val)
        if len(discovery) < min_discovery_periods:
            continue
        folds.append({
            "fold": len(folds) + 1,
            "discovery_periods": discovery, "validation_periods": validation,
            "discovery_start": discovery[0], "discovery_end": discovery[-1],
            "validation_start": validation[0], "validation_end": validation[-1],
        })
    return folds, coverage


def _wilson_vec(wins, bets, z: float = 1.96) -> np.ndarray:
    wins = np.asarray(wins, dtype=float)
    bets = np.asarray(bets, dtype=float)
    out = np.full_like(bets, np.nan, dtype=float)
    ok = bets > 0
    if not np.any(ok):
        return out
    n = bets[ok]
    p = wins[ok] / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    adj = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    out[ok] = (center - adj) / denom
    return out


def _wilson_interval(wins: int, bets: int, z: float = 1.96) -> tuple[float, float]:
    if bets <= 0:
        return np.nan, np.nan
    p = wins / bets
    denom = 1.0 + z * z / bets
    center = (p + z * z / (2.0 * bets)) / denom
    half = z * math.sqrt((p * (1.0 - p) / bets) + z * z / (4.0 * bets * bets)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _ranking_columns(metric: str) -> tuple[list[str], list[bool]]:
    metric = str(metric).lower()
    if metric == "wilson":
        return ["wilson_low", "bets", "ats_pct"], [False, False, False]
    if metric == "roi":
        return ["roi", "bets", "wilson_low"], [False, False, False]
    return ["ats_pct", "bets", "wilson_low"], [False, False, False]


def _trim(frame: pd.DataFrame, metric: str, limit: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols, asc = _ranking_columns(metric)
    out = frame.sort_values(cols, ascending=asc, na_position="last", kind="mergesort")
    if limit > 0 and len(out) > int(limit):
        out = out.head(int(limit))
    return out.reset_index(drop=True)


def _posting_models_for_period(data: pd.DataFrame, period: tuple[int, int]) -> list[str]:
    if data is None or data.empty:
        return []
    q = data.loc[_period_mask(data, [period])].copy()
    if q.empty:
        return []
    pred = pd.to_numeric(q.get("prediction_margin"), errors="coerce")
    q = q[pred.notna()].copy()
    if q.empty:
        return []
    return sorted(q["canonical_model_id"].astype(str).dropna().unique().tolist())


def multi_anchor_combination_search(
    data: pd.DataFrame,
    candidate_ids: Iterable[str],
    model_name_map: dict[str, str],
    *,
    discovery_periods: Iterable[tuple[int, int]],
    anchors: Iterable[float] = FORMAL_ANCHOR_GRID,
    min_size: int = 3,
    max_size: int = 6,
    min_available_models: int = 3,
    min_search_bets: int = 50,
    finalists: int = 50,
    ranking_metric: str = "ats",
    standard_price: int = -110,
    chunk_size: int = 512,
    max_combinations: int = 10_000_000,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Exact combination screening for a full anchor grid in one pass.

    Mean/SD/signal are computed once per candidate set; every predeclared anchor
    is then graded against that same signal matrix. This is mathematically the
    same search as running one exact search per anchor, but avoids recomputing
    the expensive combination forecasts nine times.
    """
    candidate_ids = list(dict.fromkeys(map(str, candidate_ids)))
    anchors = tuple(sorted(set(float(k) for k in anchors)))
    if len(candidate_ids) < int(min_size):
        raise ValueError("Candidate pool is smaller than the minimum combination size.")
    hi = min(int(max_size), len(candidate_ids))
    total = combination_count(len(candidate_ids), int(min_size), hi)
    if total <= 0:
        raise ValueError("No combinations to evaluate.")
    if total > int(max_combinations):
        raise ValueError(
            f"Formal exact search contains {total:,} combinations, above the configured "
            f"safety limit of {int(max_combinations):,}."
        )

    seasons = tuple(sorted(set(int(y) for y, _ in discovery_periods)))
    matrix = _make_combo_matrix(data, candidate_ids, seasons, tuple(discovery_periods))
    if matrix is None:
        raise ValueError("No discovery games available for formal combination search.")

    pred0 = matrix.get("pred0")
    predsq = matrix.get("predsq")
    available = matrix.get("available")
    if pred0 is None or predsq is None or available is None:
        p = matrix["pred"]
        finite = np.isfinite(p)
        pred0 = np.where(finite, p, 0.0).astype(np.float32, copy=False)
        predsq = pred0 * pred0
        available = finite.astype(np.uint8)
    market = np.asarray(matrix["market"], dtype=np.float32)
    cover = np.asarray(matrix["cover"], dtype=np.float32)

    retain_n = max(int(finalists), 100)
    leaderboards = {k: pd.DataFrame() for k in anchors}
    eligible_counts = {k: 0 for k in anchors}
    done = 0
    win_units = 100.0 / abs(float(standard_price)) if float(standard_price) < 0 else float(standard_price) / 100.0

    for size in range(int(min_size), hi + 1):
        batch: list[tuple[int, ...]] = []

        def consume(items: list[tuple[int, ...]]):
            nonlocal done
            if not items:
                return
            idx = np.asarray(items, dtype=int)
            count = available[:, idx].sum(axis=2, dtype=np.int16)
            sums = pred0[:, idx].sum(axis=2, dtype=np.float32)
            sqs = predsq[:, idx].sum(axis=2, dtype=np.float32)
            mean = np.divide(
                sums, count,
                out=np.full(sums.shape, np.nan, dtype=np.float32),
                where=count > 0,
            )
            numerator = sqs - np.divide(
                sums * sums, count,
                out=np.zeros(sums.shape, dtype=np.float32),
                where=count > 0,
            )
            var = np.divide(
                numerator, count - 1,
                out=np.full(sums.shape, np.nan, dtype=np.float32),
                where=count > 1,
            )
            np.maximum(var, 0.0, out=var)
            sd = np.sqrt(var, dtype=np.float32)
            edge = mean - market[:, None]
            signal = np.divide(
                np.abs(edge), sd,
                out=np.full(edge.shape, np.nan, dtype=np.float32),
                where=sd > 1e-6,
            )
            zero_sd = np.isfinite(sd) & (sd <= 1e-6)
            signal[zero_sd & (np.abs(edge) > 1e-6)] = np.inf
            signal[zero_sd & (np.abs(edge) <= 1e-6)] = 0.0
            base_gate = (
                (count >= int(min_available_models))
                & (np.isfinite(signal) | np.isinf(signal))
                & (np.abs(edge) > 1e-6)
            )

            for k in anchors:
                valid = base_gate & (signal >= float(k))
                push = valid & (np.abs(cover[:, None]) < 1e-6)
                graded = valid & ~push
                win = graded & ((edge * cover[:, None]) > 0)
                loss = graded & ~win
                wins = win.sum(axis=0).astype(int)
                losses = loss.sum(axis=0).astype(int)
                pushes = push.sum(axis=0).astype(int)
                bets = wins + losses
                ats = np.divide(
                    wins, bets,
                    out=np.full(len(bets), np.nan, dtype=float),
                    where=bets > 0,
                )
                units = wins * win_units - losses
                roi = np.divide(
                    units, bets,
                    out=np.full(len(bets), np.nan, dtype=float),
                    where=bets > 0,
                )
                metrics = pd.DataFrame({
                    "combo_size": int(size),
                    "bets": bets,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "ats_pct": ats,
                    "units": units,
                    "roi": roi,
                    "wilson_low": _wilson_vec(wins, bets),
                    "_combo_tuple": list(items),
                })
                eligible = metrics[pd.to_numeric(metrics["bets"], errors="coerce").ge(int(min_search_bets))].copy()
                eligible_counts[k] += int(len(eligible))
                if len(eligible):
                    leaderboards[k] = _trim(
                        pd.concat([leaderboards[k], eligible], ignore_index=True),
                        ranking_metric,
                        retain_n,
                    )

            done += len(items)
            if progress_callback is not None:
                best_bits = []
                for k in anchors:
                    lb = leaderboards[k]
                    if len(lb):
                        best_bits.append(f"{k:g}:{100*float(lb.iloc[0]['ats_pct']):.1f}%")
                best_txt = " · ".join(best_bits[:4])
                if len(best_bits) > 4:
                    best_txt += " · …"
                progress_callback(done, total, f"set size {size} · {done:,}/{total:,}" + (f" · {best_txt}" if best_txt else ""))

        for combo in combinations(range(len(candidate_ids)), size):
            batch.append(combo)
            if len(batch) >= int(chunk_size):
                consume(batch)
                batch = []
        consume(batch)

    out = {}
    for k in anchors:
        lb = _trim(leaderboards[k], ranking_metric, int(finalists)).copy()
        if len(lb):
            lb["search_rank"] = np.arange(1, len(lb) + 1)
            lb["model_ids"] = lb["_combo_tuple"].apply(
                lambda c: "|".join(candidate_ids[i] for i in c)
            )
            lb["model_names"] = lb["_combo_tuple"].apply(
                lambda c: " | ".join(model_name_map.get(candidate_ids[i], candidate_ids[i]) for i in c)
            )
        out[float(k)] = lb
    return {
        "by_anchor": out,
        "eligible_counts": eligible_counts,
        "evaluated_combinations": int(done),
        "total_combinations": int(total),
        "candidate_ids": candidate_ids,
    }


def _top_to_combos(top: pd.DataFrame) -> list[dict]:
    combos = []
    if top is None or top.empty:
        return combos
    for j, r in top.reset_index(drop=True).iterrows():
        mids = [x for x in str(r.get("model_ids", "")).split("|") if x]
        combos.append({"rank": int(r.get("search_rank", j + 1)), "model_ids": mids})
    return combos


def _extract_meta_rows(
    analysis: dict,
    *,
    fold: int,
    search_anchor: float,
    standard_price: int,
    min_meta_communities: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ms = analysis.get("meta_summary", pd.DataFrame())
    q = ms[(ms["method"].astype(str) == "Diversified META") & (ms["period"].astype(str) == "Holdout")] if isinstance(ms, pd.DataFrame) and len(ms) else pd.DataFrame()
    if q.empty:
        summary = {
            "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "ats_pct": np.nan, "units_flat": 0.0, "roi_flat": np.nan,
            "units_win1": 0.0, "roi_win1": np.nan, "meta_k": np.nan,
        }
        return pd.DataFrame(), pd.DataFrame(), summary
    row = q.iloc[0]
    meta_k = float(pd.to_numeric(pd.Series([row.get("selected_k")]), errors="coerce").iloc[0])
    frame = analysis.get("meta_frames", {}).get(("Diversified META", "holdout"), pd.DataFrame()).copy()
    if frame is None or frame.empty or not np.isfinite(meta_k):
        return pd.DataFrame(), pd.DataFrame(), {
            "bets": int(row.get("bets", 0) or 0), "wins": int(row.get("wins", 0) or 0),
            "losses": int(row.get("losses", 0) or 0), "pushes": int(row.get("pushes", 0) or 0),
            "ats_pct": float(row.get("ats_pct", np.nan)), "units_flat": float(row.get("units", 0.0) or 0.0),
            "roi_flat": float(row.get("roi", np.nan)), "units_win1": np.nan, "roi_win1": np.nan,
            "meta_k": meta_k,
        }

    edge = pd.to_numeric(frame.get("meta_edge"), errors="coerce").to_numpy(float)
    sig = pd.to_numeric(frame.get("meta_signal"), errors="coerce").to_numpy(float)
    cover = pd.to_numeric(frame.get("cover"), errors="coerce").to_numpy(float)
    active = pd.to_numeric(frame.get("active_units"), errors="coerce").fillna(0).to_numpy(float) >= int(min_meta_communities)
    directional = active & np.isfinite(edge) & np.isfinite(sig) & np.isfinite(cover) & (np.abs(edge) > 1e-12)
    selected = directional & (sig >= float(meta_k))
    side = np.sign(edge)
    signed_cover = side * cover
    win = selected & (signed_cover > 1e-12)
    loss = selected & (signed_cover < -1e-12)
    push = selected & (np.abs(signed_cover) <= 1e-12)

    price = float(standard_price)
    flat_win = 100.0 / abs(price) if price < 0 else price / 100.0
    win1_risk = abs(price) / 100.0 if price < 0 else 100.0 / price
    units_flat = np.where(win, flat_win, np.where(loss, -1.0, 0.0))
    units_win1 = np.where(win, 1.0, np.where(loss, -win1_risk, 0.0))

    keep = [c for c in ["game_key", "season", "week", "market_margin", "actual_margin", "meta_mean", "meta_sd", "meta_edge", "meta_signal", "active_units"] if c in frame.columns]
    all_games = frame.loc[directional, keep].copy()
    ix_all = np.flatnonzero(directional)
    all_games["fold"] = int(fold)
    all_games["search_anchor"] = float(search_anchor)
    all_games["meta_k"] = float(meta_k)
    all_games["side"] = side[ix_all]
    all_games["signed_cover"] = signed_cover[ix_all]
    all_games["directional_win"] = signed_cover[ix_all] > 1e-12
    all_games["directional_loss"] = signed_cover[ix_all] < -1e-12
    all_games["directional_push"] = np.abs(signed_cover[ix_all]) <= 1e-12

    bets = frame.loc[selected, keep].copy()
    ix = np.flatnonzero(selected)
    bets["fold"] = int(fold)
    bets["search_anchor"] = float(search_anchor)
    bets["meta_k"] = float(meta_k)
    bets["side"] = side[ix]
    bets["win"] = win[ix]
    bets["loss"] = loss[ix]
    bets["push"] = push[ix]
    bets["units_flat"] = units_flat[ix]
    bets["units_win1"] = units_win1[ix]
    decisive_selected = (win | loss)[ix]
    bets["risk_flat"] = np.where(decisive_selected, 1.0, 0.0)
    bets["risk_win1"] = np.where(decisive_selected, win1_risk, 0.0)

    wins = int(win.sum()); losses = int(loss.sum()); pushes = int(push.sum()); n_bets = wins + losses
    summary = {
        "bets": n_bets, "wins": wins, "losses": losses, "pushes": pushes,
        "ats_pct": wins / n_bets if n_bets else np.nan,
        "units_flat": float(units_flat.sum()),
        "roi_flat": float(units_flat.sum() / n_bets) if n_bets else np.nan,
        "units_win1": float(units_win1.sum()),
        "roi_win1": float(units_win1.sum() / (n_bets * win1_risk)) if n_bets else np.nan,
        "meta_k": float(meta_k),
    }
    return bets, all_games, summary


def _max_drawdown(units: Iterable[float]) -> float:
    u = pd.to_numeric(pd.Series(list(units)), errors="coerce").fillna(0).to_numpy(float)
    if len(u) == 0:
        return np.nan
    equity = np.cumsum(u)
    path = np.r_[0.0, equity]
    peaks = np.maximum.accumulate(path)
    return float(np.min(path - peaks))


def _longest_losing_streak(rows: pd.DataFrame) -> int:
    if rows is None or rows.empty or "loss" not in rows.columns:
        return 0
    best = cur = 0
    for v in rows["loss"].fillna(False).astype(bool).tolist():
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _aggregate_rows(rows: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    groups = [(None, rows)] if group_col is None else list(rows.groupby(group_col, sort=True))
    out = []
    for key, q in groups:
        wins = int(pd.to_numeric(q.get("wins"), errors="coerce").fillna(0).sum()) if "wins" in q.columns else int(q.get("win", pd.Series(False, index=q.index)).fillna(False).sum())
        losses = int(pd.to_numeric(q.get("losses"), errors="coerce").fillna(0).sum()) if "losses" in q.columns else int(q.get("loss", pd.Series(False, index=q.index)).fillna(False).sum())
        pushes = int(pd.to_numeric(q.get("pushes"), errors="coerce").fillna(0).sum()) if "pushes" in q.columns else int(q.get("push", pd.Series(False, index=q.index)).fillna(False).sum())
        bets = wins + losses
        units_flat = float(pd.to_numeric(q.get("units_flat"), errors="coerce").fillna(0).sum())
        units_win1 = float(pd.to_numeric(q.get("units_win1"), errors="coerce").fillna(0).sum())
        risk_flat = float(pd.to_numeric(q.get("risk_flat", pd.Series(1.0, index=q.index)), errors="coerce").fillna(0).sum())
        risk_win1 = float(pd.to_numeric(q.get("risk_win1"), errors="coerce").fillna(0).sum()) if "risk_win1" in q.columns else np.nan
        wl, wu = _wilson_interval(wins, bets)
        rec = {
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "ats_pct": wins / bets if bets else np.nan,
            "wilson_low": wl, "wilson_high": wu,
            "units_flat": units_flat, "roi_flat": units_flat / risk_flat if risk_flat > 0 else np.nan,
            "units_win1": units_win1, "roi_win1": units_win1 / risk_win1 if np.isfinite(risk_win1) and risk_win1 > 0 else np.nan,
        }
        if group_col is not None:
            rec[group_col] = key
        out.append(rec)
    return pd.DataFrame(out)


def _aggregate_fixed(fold_results: pd.DataFrame, bet_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_results is None or fold_results.empty:
        return pd.DataFrame()
    rows = []
    for k, q in fold_results.groupby("search_anchor", sort=True):
        b = bet_rows[np.isclose(pd.to_numeric(bet_rows["search_anchor"], errors="coerce"), float(k))].copy() if isinstance(bet_rows, pd.DataFrame) and len(bet_rows) else pd.DataFrame()
        agg = _aggregate_rows(b)
        a = agg.iloc[0].to_dict() if len(agg) else {
            "bets": 0, "wins": 0, "losses": 0, "pushes": 0, "ats_pct": np.nan,
            "wilson_low": np.nan, "wilson_high": np.nan, "units_flat": 0.0, "roi_flat": np.nan,
            "units_win1": 0.0, "roi_win1": np.nan,
        }
        a.update({
            "search_anchor": float(k),
            "blocks": int(q["fold"].nunique()),
            "profitable_blocks_flat": int((pd.to_numeric(q["units_flat"], errors="coerce") > 0).sum()),
            "median_block_roi_flat": float(pd.to_numeric(q["roi_flat"], errors="coerce").median()),
            "mean_meta_k": float(pd.to_numeric(q["meta_k"], errors="coerce").mean()),
        })
        if len(b):
            b = b.sort_values(["season", "week", "game_key"], kind="mergesort")
            a["max_drawdown_flat"] = _max_drawdown(b["units_flat"])
            a["max_drawdown_win1"] = _max_drawdown(b["units_win1"])
            a["longest_losing_streak"] = _longest_losing_streak(b)
        else:
            a["max_drawdown_flat"] = np.nan; a["max_drawdown_win1"] = np.nan; a["longest_losing_streak"] = 0
        rows.append(a)
    return pd.DataFrame(rows).sort_values("search_anchor").reset_index(drop=True)


def _adaptive_path(fold_results: pd.DataFrame, *, min_prior_bets: int = 50, fallback_anchor: float = FORMAL_FALLBACK_ANCHOR) -> pd.DataFrame:
    if fold_results is None or fold_results.empty:
        return pd.DataFrame()
    out = []
    folds = sorted(pd.to_numeric(fold_results["fold"], errors="coerce").dropna().astype(int).unique().tolist())
    anchors = sorted(pd.to_numeric(fold_results["search_anchor"], errors="coerce").dropna().astype(float).unique().tolist())
    for fold in folds:
        prior = fold_results[pd.to_numeric(fold_results["fold"], errors="coerce") < int(fold)].copy()
        selected = float(fallback_anchor)
        reason = "predeclared fallback"
        best_score = np.nan
        prior_bets = 0
        if len(prior):
            cand = []
            for k in anchors:
                q = prior[np.isclose(pd.to_numeric(prior["search_anchor"], errors="coerce"), float(k))]
                wins = int(pd.to_numeric(q.get("wins"), errors="coerce").fillna(0).sum())
                losses = int(pd.to_numeric(q.get("losses"), errors="coerce").fillna(0).sum())
                bets = wins + losses
                if bets < int(min_prior_bets):
                    continue
                wil = _wilson_lower(wins, bets)
                cand.append((float(wil), int(bets), -abs(float(k) - float(fallback_anchor)), -float(k), float(k)))
            if cand:
                cand.sort(reverse=True)
                best_score, prior_bets, _, _, selected = cand[0]
                reason = "best prior OOS Wilson LB"
        qnow = fold_results[(pd.to_numeric(fold_results["fold"], errors="coerce") == int(fold)) & np.isclose(pd.to_numeric(fold_results["search_anchor"], errors="coerce"), float(selected))]
        if qnow.empty:
            # Grid/floating fallback protection.
            qfold = fold_results[pd.to_numeric(fold_results["fold"], errors="coerce") == int(fold)].copy()
            if qfold.empty:
                continue
            qfold["_dist"] = (pd.to_numeric(qfold["search_anchor"], errors="coerce") - float(selected)).abs()
            rr = qfold.sort_values("_dist").iloc[0]
            selected = float(rr["search_anchor"])
        else:
            rr = qnow.iloc[0]
        rec = rr.to_dict()
        rec.update({
            "selected_anchor": float(selected),
            "anchor_selection_reason": reason,
            "prior_anchor_wilson_low": float(best_score) if np.isfinite(best_score) else np.nan,
            "prior_anchor_bets": int(prior_bets),
        })
        out.append(rec)
    return pd.DataFrame(out)


def _edge_calibration(all_games: pd.DataFrame) -> pd.DataFrame:
    if all_games is None or all_games.empty:
        return pd.DataFrame()
    q = all_games.copy()
    sig = pd.to_numeric(q.get("meta_signal"), errors="coerce")
    bins = [-np.inf, 0.50, 0.75, 1.00, 1.25, np.inf]
    labels = ["<0.50", "0.50–0.75", "0.75–1.00", "1.00–1.25", "≥1.25"]
    q["edge_bin"] = pd.cut(sig, bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        z = q[q["edge_bin"].astype(str) == label].copy()
        if z.empty:
            rows.append({"edge_bin": label, "games": 0, "wins": 0, "losses": 0, "pushes": 0, "ats_pct": np.nan, "wilson_low": np.nan, "wilson_high": np.nan})
            continue
        wins = int(z["directional_win"].fillna(False).sum())
        losses = int(z["directional_loss"].fillna(False).sum())
        pushes = int(z["directional_push"].fillna(False).sum())
        bets = wins + losses
        wl, wu = _wilson_interval(wins, bets)
        rows.append({"edge_bin": label, "games": len(z), "wins": wins, "losses": losses, "pushes": pushes, "ats_pct": wins / bets if bets else np.nan, "wilson_low": wl, "wilson_high": wu})
    return pd.DataFrame(rows)


def run_formal_walkforward_backtest(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    model_name_map: dict[str, str],
    *,
    period_scope: Iterable[tuple[int, int]] | None = None,
    anchors: Iterable[float] = FORMAL_ANCHOR_GRID,
    oos_blocks: int = 8,
    oos_block_size: int = 6,
    min_discovery_periods: int = 24,
    min_games_per_period: int = 10,
    pool_n: int = 35,
    pool_min_bets: int = 25,
    min_size: int = 3,
    max_size: int = 6,
    min_available_models: int = 3,
    min_search_bets: int = 50,
    finalists: int = 50,
    overlap_threshold: float = 0.50,
    min_meta_communities: int = 2,
    meta_thresholds: Iterable[float] = FORMAL_META_K_GRID,
    max_combinations: int = 10_000_000,
    standard_price: int = -110,
    adaptive_min_prior_bets: int = 50,
    fallback_anchor: float = FORMAL_FALLBACK_ANCHOR,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Production-equivalent chronological validation of the current recipe.

    Each OOS block is untouched. Candidate availability is determined from the
    first OOS week, rankings use only earlier graded data, and one exact
    multi-anchor combination pass supplies a fully independent finalist search
    for every predeclared anchor. Adaptive anchor selection for block t uses
    only fixed-anchor OOS results from blocks < t.
    """
    anchors = tuple(sorted(set(float(k) for k in anchors)))
    close_data = data_for_line_reference(data, line_history, "close_margin")
    if close_data.empty:
        raise ValueError("No PT Updated/final line history is available for formal backtesting.")
    folds, coverage = _build_formal_folds(
        close_data, period_scope=period_scope,
        block_size=int(oos_block_size), n_folds=int(oos_blocks),
        min_discovery_periods=int(min_discovery_periods),
        min_games_per_period=int(min_games_per_period),
        min_available_models=int(min_available_models),
    )
    if not folds:
        raise ValueError("No formal OOS folds could be formed with the requested chronology settings.")

    fold_rows = []
    bet_frames = []
    all_game_frames = []
    chronology = []
    total_units = len(folds) * 1000

    for fi, fold in enumerate(folds, start=1):
        base = (fi - 1) * 1000
        def prog(local: int, label: str):
            if progress_callback is not None:
                progress_callback(base + int(local), total_units, f"OOS {fi}/{len(folds)} · {label}")

        first_oos = tuple(fold["validation_periods"])[0]
        posting_ids = _posting_models_for_period(data, first_oos)
        prog(10, f"historical posting universe {len(posting_ids)} models")
        ranked = _rank_candidates_for_reference(
            close_data, fold["discovery_periods"], posting_ids,
            pool_n=int(pool_n), pool_min_bets=int(pool_min_bets),
            pool_metric="wilson", standard_price=int(standard_price),
        )
        ids = ranked["canonical_model_id"].astype(str).tolist() if len(ranked) else []
        if len(ids) < int(min_size):
            chronology.append({
                "fold": fi, "status": "too few eligible models", "posting_models": len(posting_ids),
                "candidate_models": len(ids), "oos_start": f"{first_oos[0]} W{first_oos[1]}",
            })
            continue
        hi = min(int(max_size), len(ids))
        nwork = combination_count(len(ids), int(min_size), hi)
        prog(25, f"exact multi-anchor search · {len(ids)} models · {nwork:,} sets")

        def search_progress(done: int, total: int, label: str):
            frac = float(done) / float(total) if total else 0.0
            prog(25 + int(650 * max(0.0, min(1.0, frac))), label)

        search = multi_anchor_combination_search(
            close_data, ids, model_name_map,
            discovery_periods=fold["discovery_periods"], anchors=anchors,
            min_size=int(min_size), max_size=int(hi), min_available_models=int(min_available_models),
            min_search_bets=int(min_search_bets), finalists=int(finalists), ranking_metric="ats",
            standard_price=int(standard_price), chunk_size=512, max_combinations=int(max_combinations),
            progress_callback=search_progress,
        )

        usable_anchor_n = 0
        for ai, k in enumerate(anchors, start=1):
            top = search["by_anchor"].get(float(k), pd.DataFrame()).head(int(finalists)).copy()
            combos = _top_to_combos(top)
            if len(combos) < 2:
                continue
            usable_anchor_n += 1
            prog(690 + int(280 * ai / max(1, len(anchors))), f"anchor {k:g} · finalist META")
            analysis = analyze_finalist_portfolio(
                close_data, combos, fold["discovery_periods"], fold["validation_periods"],
                min_available_models=int(min_available_models), thresholds=tuple(meta_thresholds),
                combo_min_bets=int(min_search_bets), meta_min_bets=int(min_search_bets),
                overlap_threshold=float(overlap_threshold), min_meta_communities=int(min_meta_communities),
                standard_price=int(standard_price), line_history=None,
            )
            bets, all_games, st = _extract_meta_rows(
                analysis, fold=fi, search_anchor=float(k), standard_price=int(standard_price),
                min_meta_communities=int(min_meta_communities),
            )
            fold_rows.append({
                "fold": fi,
                "discovery_start": f"{fold['discovery_start'][0]} W{fold['discovery_start'][1]}",
                "discovery_end": f"{fold['discovery_end'][0]} W{fold['discovery_end'][1]}",
                "oos_start": f"{fold['validation_start'][0]} W{fold['validation_start'][1]}",
                "oos_end": f"{fold['validation_end'][0]} W{fold['validation_end'][1]}",
                "search_anchor": float(k), "posting_models": len(posting_ids), "candidate_models": len(ids),
                "evaluated_combinations": int(search.get("evaluated_combinations", 0)),
                "eligible_combinations": int(search.get("eligible_counts", {}).get(float(k), 0)),
                "finalists": len(combos),
                "communities": int(analysis.get("overlap_summary", {}).get("communities", 0)),
                **st,
            })
            if len(bets): bet_frames.append(bets)
            if len(all_games): all_game_frames.append(all_games)

        chronology.append({
            "fold": fi, "status": "ok" if usable_anchor_n else "no usable anchor pipelines",
            "discovery_periods": len(fold["discovery_periods"]),
            "discovery_start": f"{fold['discovery_start'][0]} W{fold['discovery_start'][1]}",
            "discovery_end": f"{fold['discovery_end'][0]} W{fold['discovery_end'][1]}",
            "oos_start": f"{fold['validation_start'][0]} W{fold['validation_start'][1]}",
            "oos_end": f"{fold['validation_end'][0]} W{fold['validation_end'][1]}",
            "oos_periods": len(fold["validation_periods"]), "posting_models": len(posting_ids),
            "candidate_models": len(ids), "candidate_combinations": int(nwork), "anchors": usable_anchor_n,
        })
        prog(1000, "complete")

    fold_df = pd.DataFrame(fold_rows)
    bet_df = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    all_games_df = pd.concat(all_game_frames, ignore_index=True, sort=False) if all_game_frames else pd.DataFrame()
    if fold_df.empty:
        raise RuntimeError("Formal walk-forward backtest produced no scorable OOS results.")

    fixed = _aggregate_fixed(fold_df, bet_df)
    adaptive = _adaptive_path(
        fold_df, min_prior_bets=int(adaptive_min_prior_bets), fallback_anchor=float(fallback_anchor)
    )
    if len(adaptive):
        chosen_pairs = {(int(r.fold), float(r.selected_anchor)) for r in adaptive.itertuples(index=False)}
        if len(bet_df):
            keep = [((int(f), float(k)) in chosen_pairs) for f, k in zip(bet_df["fold"], bet_df["search_anchor"])]
            adaptive_bets = bet_df.loc[keep].copy()
        else:
            adaptive_bets = pd.DataFrame()
        if len(all_games_df):
            keep = [((int(f), float(k)) in chosen_pairs) for f, k in zip(all_games_df["fold"], all_games_df["search_anchor"])]
            adaptive_all_games = all_games_df.loc[keep].copy()
        else:
            adaptive_all_games = pd.DataFrame()
    else:
        adaptive_bets = pd.DataFrame(); adaptive_all_games = pd.DataFrame()

    adaptive_summary = _aggregate_rows(adaptive_bets)
    if len(adaptive_summary):
        if len(adaptive_bets):
            z = adaptive_bets.sort_values(["season", "week", "game_key"], kind="mergesort")
            adaptive_summary["max_drawdown_flat"] = _max_drawdown(z["units_flat"])
            adaptive_summary["max_drawdown_win1"] = _max_drawdown(z["units_win1"])
            adaptive_summary["longest_losing_streak"] = _longest_losing_streak(z)
        adaptive_summary["blocks"] = int(adaptive["fold"].nunique())
        adaptive_summary["profitable_blocks_flat"] = int((pd.to_numeric(adaptive["units_flat"], errors="coerce") > 0).sum())
    edge_cal = _edge_calibration(adaptive_all_games)

    return {
        "fixed_anchor_surface": fixed,
        "fold_results": fold_df,
        "adaptive_path": adaptive,
        "adaptive_summary": adaptive_summary,
        "bet_rows": bet_df,
        "adaptive_bet_rows": adaptive_bets,
        "adaptive_all_games": adaptive_all_games,
        "edge_calibration": edge_cal,
        "chronology": pd.DataFrame(chronology),
        "coverage": coverage,
        "anchors": anchors,
        "oos_blocks_completed": int(fold_df["fold"].nunique()),
        "oos_block_size": int(oos_block_size),
        "standard_price": int(standard_price),
        "adaptive_rule": f"highest pooled prior-OOS Wilson lower bound with >= {int(adaptive_min_prior_bets)} bets; fallback {float(fallback_anchor):g}",
        "selection_spec": {
            "candidate_rank": "prior-data Wilson lower bound",
            "candidate_pool_n": int(pool_n),
            "candidate_min_bets": int(pool_min_bets),
            "combination_sizes": [int(min_size), int(max_size)],
            "combination_rank": "prior-data ATS",
            "finalists": int(finalists),
            "search_anchor_grid": list(anchors),
            "meta_k_tuning": "discovery-only stable threshold selection",
            "execution_line": "PredictionTracker Updated/final",
            "historical_availability": "models posting in first OOS week",
            "staking": ["flat 1u risk", "risk to win 1u"],
        },
    }


def save_formal_backtest_outputs(result: dict, root: str | Path) -> dict[str, str]:
    root = Path(root)
    outdir = root / "data" / "derived"
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "fixed_anchor_surface": "formal_backtest_anchor_surface.csv",
        "fold_results": "formal_backtest_fold_results.csv",
        "adaptive_path": "formal_backtest_adaptive_path.csv",
        "adaptive_summary": "formal_backtest_adaptive_summary.csv",
        "adaptive_bet_rows": "formal_backtest_adaptive_bets.csv",
        "edge_calibration": "formal_backtest_edge_calibration.csv",
        "chronology": "formal_backtest_chronology.csv",
    }
    written: dict[str, str] = {}
    for key, name in mapping.items():
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            path = outdir / name
            frame.to_csv(path, index=False)
            written[key] = str(path.relative_to(root))
    meta = {
        "oos_blocks_completed": int(result.get("oos_blocks_completed", 0)),
        "oos_block_size": int(result.get("oos_block_size", 0)),
        "standard_price": int(result.get("standard_price", -110)),
        "anchors": list(result.get("anchors", FORMAL_ANCHOR_GRID)),
        "adaptive_rule": result.get("adaptive_rule"),
        "selection_spec": result.get("selection_spec", {}),
        "files": written,
    }
    path = outdir / "formal_backtest_status.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    written["status"] = str(path.relative_to(root))
    return written


def load_formal_backtest_outputs(root: str | Path) -> dict:
    root = Path(root)
    outdir = root / "data" / "derived"
    names = {
        "fixed_anchor_surface": "formal_backtest_anchor_surface.csv",
        "fold_results": "formal_backtest_fold_results.csv",
        "adaptive_path": "formal_backtest_adaptive_path.csv",
        "adaptive_summary": "formal_backtest_adaptive_summary.csv",
        "adaptive_bet_rows": "formal_backtest_adaptive_bets.csv",
        "edge_calibration": "formal_backtest_edge_calibration.csv",
        "chronology": "formal_backtest_chronology.csv",
    }
    result = {}
    for key, name in names.items():
        path = outdir / name
        if path.exists():
            try:
                result[key] = pd.read_csv(path, low_memory=False)
            except Exception:
                result[key] = pd.DataFrame()
        else:
            result[key] = pd.DataFrame()
    status = outdir / "formal_backtest_status.json"
    if status.exists():
        try:
            meta = json.loads(status.read_text(encoding="utf-8"))
            result.update(meta)
        except Exception:
            pass
    if not any(isinstance(result.get(k), pd.DataFrame) and len(result.get(k)) for k in names):
        return {}
    return result
