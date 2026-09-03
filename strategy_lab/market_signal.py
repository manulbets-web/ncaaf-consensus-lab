from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import math
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from committee import _matrix_and_meta
from formal_backtest import _build_formal_folds, _posting_models_for_period, _wilson_interval, _max_drawdown, _longest_losing_streak
from line_movement import LINE_REFS, _period_mask, clean_line_history_for_analysis, data_for_line_reference, _rank_candidates_for_reference

TOP_N_GRID = (3, 5, 8, 10, 15, 20, 25, 35, 999)
TOP_N_LABELS = {999: "All"}
RANKING_METHODS = ("wilson", "mae", "incremental")
BET_EDGE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _consensus_frame(data_ref: pd.DataFrame, model_ids: Iterable[str], periods: Iterable[tuple[int, int]], *, min_available: int = 2) -> pd.DataFrame:
    ids = list(dict.fromkeys(map(str, model_ids)))
    pred, meta = _matrix_and_meta(data_ref, ids, periods)
    if pred.empty or not ids:
        return pd.DataFrame()
    x = pred.reindex(columns=[m for m in ids if m in pred.columns]).to_numpy(dtype=float)
    if x.size == 0:
        return pd.DataFrame()
    finite = np.isfinite(x)
    count = finite.sum(axis=1)
    mean = np.divide(np.nansum(x, axis=1), count, out=np.full(len(pred), np.nan), where=count > 0)
    out = pd.DataFrame({
        "game_key": pred.index.astype(str),
        "season": pd.to_numeric(meta["season"], errors="coerce").to_numpy(),
        "week": pd.to_numeric(meta["week"], errors="coerce").to_numpy(),
        "market_margin": pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy(float),
        "actual_margin": pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(float),
        "consensus_margin": mean,
        "models_available": count,
    })
    for c in ["road", "home"]:
        if c in meta.columns:
            out[c] = meta[c].astype(str).to_numpy()
    out["model_residual"] = out["consensus_margin"] - out["market_margin"]
    out["actual_residual"] = out["actual_margin"] - out["market_margin"]
    need = max(1, min(int(min_available), len(ids)))
    good = (
        out["models_available"].ge(need)
        & np.isfinite(out["market_margin"])
        & np.isfinite(out["actual_margin"])
        & np.isfinite(out["consensus_margin"])
    )
    return out.loc[good].reset_index(drop=True)


def _fit_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(good.sum()) < X.shape[1] + 8:
        return np.full(X.shape[1], np.nan)
    return np.linalg.lstsq(X[good], y[good], rcond=None)[0]


def _fit_market_models(train: pd.DataFrame) -> dict:
    if train is None or train.empty:
        return {}
    y = pd.to_numeric(train["actual_margin"], errors="coerce").to_numpy(float)
    m = pd.to_numeric(train["market_margin"], errors="coerce").to_numpy(float)
    r = pd.to_numeric(train["model_residual"], errors="coerce").to_numpy(float)
    b0 = _fit_ols(y, np.column_stack([np.ones(len(train)), m]))
    b1 = _fit_ols(y, np.column_stack([np.ones(len(train)), m, r]))
    return {
        "market_intercept": float(b0[0]) if len(b0) and np.isfinite(b0[0]) else np.nan,
        "market_beta": float(b0[1]) if len(b0) > 1 and np.isfinite(b0[1]) else np.nan,
        "signal_intercept": float(b1[0]) if len(b1) and np.isfinite(b1[0]) else np.nan,
        "signal_market_beta": float(b1[1]) if len(b1) > 1 and np.isfinite(b1[1]) else np.nan,
        "gamma": float(b1[2]) if len(b1) > 2 and np.isfinite(b1[2]) else np.nan,
        "train_games": int(len(train)),
    }


def _score_incremental_models(data_ref: pd.DataFrame, discovery_periods, live_ids: Iterable[str], *, min_bets: int = 25) -> pd.DataFrame:
    q = data_ref.loc[_period_mask(data_ref, discovery_periods)].copy()
    live = set(map(str, live_ids))
    q = q[q["canonical_model_id"].astype(str).isin(live)].copy()
    rows = []
    for mid, g in q.groupby("canonical_model_id", sort=False):
        pred = pd.to_numeric(g["prediction_margin"], errors="coerce")
        market = pd.to_numeric(g["market_margin"], errors="coerce")
        actual = pd.to_numeric(g["actual_margin"], errors="coerce")
        ok = pred.notna() & market.notna() & actual.notna()
        if int(ok.sum()) < int(min_bets):
            continue
        x = (pred[ok] - market[ok]).to_numpy(float)
        y = (actual[ok] - market[ok]).to_numpy(float)
        if np.nanstd(x) <= 1e-9 or np.nanstd(y) <= 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        denom = float(np.dot(x, x))
        gamma = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
        # Positive, repeatable residual association is the target. The modest
        # sample-size shrinkage prevents tiny-history models from dominating.
        shrink = math.sqrt(len(x) / (len(x) + 50.0))
        score = corr * shrink
        rows.append({
            "canonical_model_id": str(mid), "bets": int(len(x)),
            "incremental_corr": corr, "incremental_gamma": gamma,
            "incremental_score": score,
            "residual_mae": float(np.mean(np.abs(y - x))),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["incremental_score", "bets"], ascending=[False, False]).reset_index(drop=True)


def _rank_models(data_ref: pd.DataFrame, discovery_periods, live_ids: Iterable[str], *, method: str, min_bets: int, standard_price: int = -110) -> pd.DataFrame:
    method = str(method).lower()
    if method == "incremental":
        out = _score_incremental_models(data_ref, discovery_periods, live_ids, min_bets=min_bets)
        if len(out):
            out["pool_rank"] = np.arange(1, len(out) + 1)
        return out
    metric = "mae" if method == "mae" else "wilson"
    return _rank_candidates_for_reference(
        data_ref, discovery_periods, live_ids,
        pool_n=10_000, pool_min_bets=int(min_bets), pool_metric=metric,
        standard_price=int(standard_price),
    )


def _evaluate_oos(test: pd.DataFrame, fit: dict) -> pd.DataFrame:
    if test is None or test.empty or not fit:
        return pd.DataFrame()
    q = test.copy()
    m = pd.to_numeric(q["market_margin"], errors="coerce").to_numpy(float)
    r = pd.to_numeric(q["model_residual"], errors="coerce").to_numpy(float)
    q["pred_market_only"] = fit["market_intercept"] + fit["market_beta"] * m
    q["pred_market_signal"] = fit["signal_intercept"] + fit["signal_market_beta"] * m + fit["gamma"] * r
    q["adjusted_edge"] = q["pred_market_signal"] - q["market_margin"]
    q["market_only_edge"] = q["pred_market_only"] - q["market_margin"]
    return q


def _forecast_summary(preds: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    rows = []
    keys = ["ranking_method", "top_n", "top_n_label", "line_reference"]
    for vals, q in preds.groupby(keys, dropna=False, sort=True):
        rank, n, nlab, line = vals
        y = pd.to_numeric(q["actual_margin"], errors="coerce").to_numpy(float)
        pm = pd.to_numeric(q["pred_market_only"], errors="coerce").to_numpy(float)
        ps = pd.to_numeric(q["pred_market_signal"], errors="coerce").to_numpy(float)
        mr = pd.to_numeric(q["model_residual"], errors="coerce").to_numpy(float)
        ar = pd.to_numeric(q["actual_residual"], errors="coerce").to_numpy(float)
        good = np.isfinite(y) & np.isfinite(pm) & np.isfinite(ps)
        if not np.any(good):
            continue
        market_mae = float(np.mean(np.abs(y[good] - pm[good])))
        signal_mae = float(np.mean(np.abs(y[good] - ps[good])))
        market_rmse = float(np.sqrt(np.mean(np.square(y[good] - pm[good]))))
        signal_rmse = float(np.sqrt(np.mean(np.square(y[good] - ps[good]))))
        rg = np.isfinite(mr) & np.isfinite(ar)
        corr = float(np.corrcoef(mr[rg], ar[rg])[0, 1]) if int(rg.sum()) >= 3 and np.std(mr[rg]) > 1e-9 and np.std(ar[rg]) > 1e-9 else np.nan
        rows.append({
            "ranking_method": rank, "top_n": int(n), "top_n_label": nlab, "line_reference": line,
            "oos_games": int(good.sum()),
            "market_mae": market_mae, "signal_mae": signal_mae, "mae_improvement": market_mae - signal_mae,
            "market_rmse": market_rmse, "signal_rmse": signal_rmse, "rmse_improvement": market_rmse - signal_rmse,
            "residual_corr": corr,
            "mean_gamma": float(pd.to_numeric(q["gamma"], errors="coerce").mean()),
            "median_gamma": float(pd.to_numeric(q["gamma"], errors="coerce").median()),
            "positive_gamma_folds": int((pd.to_numeric(q[["fold", "gamma"]].drop_duplicates()["gamma"], errors="coerce") > 0).sum()),
            "folds": int(q["fold"].nunique()),
        })
    return pd.DataFrame(rows)


def _betting_summary(preds: pd.DataFrame, *, edge_grid: Iterable[float], standard_price: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preds is None or preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    price = float(standard_price)
    win_unit = 100.0 / abs(price) if price < 0 else price / 100.0
    rows, bets_out = [], []
    keys = ["ranking_method", "top_n", "top_n_label", "line_reference"]
    for vals, q0 in preds.groupby(keys, dropna=False, sort=True):
        rank, n, nlab, line = vals
        q0 = q0.copy()
        edge = pd.to_numeric(q0["adjusted_edge"], errors="coerce").to_numpy(float)
        cover = (pd.to_numeric(q0["actual_margin"], errors="coerce") - pd.to_numeric(q0["market_margin"], errors="coerce")).to_numpy(float)
        for cutoff in edge_grid:
            selected = np.isfinite(edge) & np.isfinite(cover) & (np.abs(edge) >= float(cutoff)) & (np.abs(edge) > 1e-12)
            signed = np.sign(edge) * cover
            w = selected & (signed > 1e-12); l = selected & (signed < -1e-12); p = selected & (np.abs(signed) <= 1e-12)
            wins, losses, pushes = int(w.sum()), int(l.sum()), int(p.sum()); bets = wins + losses
            units = wins * win_unit - losses
            wl, wu = _wilson_interval(wins, bets)
            profitable_folds = 0
            for _, gg in q0.groupby("fold"):
                ge = pd.to_numeric(gg["adjusted_edge"], errors="coerce").to_numpy(float)
                gc = (pd.to_numeric(gg["actual_margin"], errors="coerce") - pd.to_numeric(gg["market_margin"], errors="coerce")).to_numpy(float)
                gs = np.isfinite(ge) & np.isfinite(gc) & (np.abs(ge) >= float(cutoff)) & (np.abs(ge) > 1e-12)
                gmargin = np.sign(ge) * gc
                gu = np.where(gs & (gmargin > 1e-12), win_unit, np.where(gs & (gmargin < -1e-12), -1.0, 0.0))
                profitable_folds += int(float(np.sum(gu)) > 0)
            rows.append({
                "ranking_method": rank, "top_n": int(n), "top_n_label": nlab, "line_reference": line,
                "edge_cutoff_points": float(cutoff), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
                "ats_pct": wins / bets if bets else np.nan, "wilson_low": wl, "wilson_high": wu,
                "units_flat": units, "roi_flat": units / bets if bets else np.nan,
                "profitable_folds": profitable_folds,
            })
            if np.any(selected):
                b = q0.loc[selected].copy(); ix = np.flatnonzero(selected)
                b["edge_cutoff_points"] = float(cutoff); b["win"] = w[ix]; b["loss"] = l[ix]; b["push"] = p[ix]
                b["cover_margin"] = signed[ix]
                b["units_flat"] = np.where(w[ix], win_unit, np.where(l[ix], -1.0, 0.0))
                bets_out.append(b)
    return pd.DataFrame(rows), (pd.concat(bets_out, ignore_index=True, sort=False) if bets_out else pd.DataFrame())


def run_market_signal_walkforward(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    *,
    period_scope: Iterable[tuple[int, int]] | None = None,
    top_n_grid: Iterable[int] = TOP_N_GRID,
    ranking_methods: Iterable[str] = RANKING_METHODS,
    edge_grid: Iterable[float] = BET_EDGE_GRID,
    oos_blocks: int = 8,
    oos_block_size: int = 6,
    min_discovery_periods: int = 24,
    min_games_per_period: int = 10,
    min_model_bets: int = 25,
    min_available_models: int = 2,
    standard_price: int = -110,
    common_line_games: bool = True,
    progress_callback=None,
) -> dict:
    clean = clean_line_history_for_analysis(line_history, move_threshold=10.0, exclude_suspect_open=True)
    if clean is None or clean.empty:
        raise ValueError("Historical PredictionTracker line history is unavailable.")
    common_keys = set()
    if common_line_games:
        z = clean.copy()
        for c in ["open_margin", "midweek_margin", "close_margin"]:
            z[c] = pd.to_numeric(z.get(c), errors="coerce")
        common_keys = set(z.loc[z[["open_margin", "midweek_margin", "close_margin"]].notna().all(axis=1), "game_key"].astype(str))
    close = data_for_line_reference(data, clean, "close_margin")
    if common_line_games:
        close = close[close["game_key"].astype(str).isin(common_keys)].copy()
    folds, coverage = _build_formal_folds(
        close, period_scope=period_scope, block_size=int(oos_block_size), n_folds=int(oos_blocks),
        min_discovery_periods=int(min_discovery_periods), min_games_per_period=int(min_games_per_period),
        min_available_models=max(1, int(min_available_models)),
    )
    if not folds:
        raise ValueError("No market-signal OOS folds could be formed.")

    top_n_grid = tuple(dict.fromkeys(int(x) for x in top_n_grid))
    ranking_methods = tuple(dict.fromkeys(str(x).lower() for x in ranking_methods))
    predictions, coef_rows, rank_rows, chronology = [], [], [], []
    total = len(folds) * len(LINE_REFS) * len(ranking_methods)
    step = 0
    for fi, fold in enumerate(folds, start=1):
        first_oos = tuple(fold["validation_periods"])[0]
        posting = _posting_models_for_period(data, first_oos)
        chronology.append({
            "fold": fi,
            "discovery_start": f"{fold['discovery_start'][0]} W{fold['discovery_start'][1]}",
            "discovery_end": f"{fold['discovery_end'][0]} W{fold['discovery_end'][1]}",
            "oos_start": f"{fold['validation_start'][0]} W{fold['validation_start'][1]}",
            "oos_end": f"{fold['validation_end'][0]} W{fold['validation_end'][1]}",
            "posting_models": len(posting),
        })
        for line_label, line_col in LINE_REFS:
            ref = data_for_line_reference(data, clean, line_col)
            if common_line_games:
                ref = ref[ref["game_key"].astype(str).isin(common_keys)].copy()
            for method in ranking_methods:
                ranked = _rank_models(ref, fold["discovery_periods"], posting, method=method, min_bets=int(min_model_bets), standard_price=int(standard_price))
                ids_all = ranked["canonical_model_id"].astype(str).tolist() if len(ranked) else []
                if len(ranked):
                    rr = ranked.copy(); rr["fold"] = fi; rr["line_reference"] = line_label; rr["ranking_method"] = method
                    rank_rows.append(rr)
                for n in top_n_grid:
                    ids = ids_all if int(n) >= 999 else ids_all[:int(n)]
                    if not ids:
                        continue
                    need = min(max(1, int(min_available_models)), len(ids))
                    train = _consensus_frame(ref, ids, fold["discovery_periods"], min_available=need)
                    test = _consensus_frame(ref, ids, fold["validation_periods"], min_available=need)
                    fit = _fit_market_models(train)
                    oos = _evaluate_oos(test, fit)
                    if oos.empty or not fit or not np.isfinite(fit.get("gamma", np.nan)):
                        continue
                    nlab = TOP_N_LABELS.get(int(n), str(int(n)))
                    oos["fold"] = fi; oos["ranking_method"] = method; oos["top_n"] = int(n); oos["top_n_label"] = nlab; oos["line_reference"] = line_label
                    oos["gamma"] = fit["gamma"]; oos["signal_market_beta"] = fit["signal_market_beta"]; oos["signal_intercept"] = fit["signal_intercept"]
                    predictions.append(oos)
                    coef_rows.append({
                        "fold": fi, "ranking_method": method, "top_n": int(n), "top_n_label": nlab, "line_reference": line_label,
                        **fit, "selected_models": len(ids),
                    })
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total, f"OOS {fi}/{len(folds)} · {line_label} · {method}")
    pred = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    if pred.empty:
        raise RuntimeError("Market-anchored walk-forward produced no OOS predictions.")
    forecast = _forecast_summary(pred)
    betting, bet_rows = _betting_summary(pred, edge_grid=edge_grid, standard_price=int(standard_price))

    # Non-parametric response: does actual market error move in the direction of model-market disagreement?
    edges = [-np.inf, -7, -4, -2, 0, 2, 4, 7, np.inf]
    labels = ["≤-7", "-7 to -4", "-4 to -2", "-2 to 0", "0 to 2", "2 to 4", "4 to 7", "≥7"]
    cal_rows = []
    for keys, q in pred.groupby(["ranking_method", "top_n", "top_n_label", "line_reference"], sort=True):
        qq = q.copy(); qq["residual_bin"] = pd.cut(pd.to_numeric(qq["model_residual"], errors="coerce"), bins=edges, labels=labels, include_lowest=True, right=False)
        for b, g in qq.groupby("residual_bin", observed=True):
            ar = pd.to_numeric(g["actual_residual"], errors="coerce")
            mr = pd.to_numeric(g["model_residual"], errors="coerce")
            cal_rows.append({
                "ranking_method": keys[0], "top_n": int(keys[1]), "top_n_label": keys[2], "line_reference": keys[3],
                "model_residual_bin": str(b), "games": int(ar.notna().sum()),
                "mean_model_residual": float(mr.mean()), "mean_actual_residual": float(ar.mean()), "median_actual_residual": float(ar.median()),
            })
    return {
        "forecast_summary": forecast,
        "betting_summary": betting,
        "fold_coefficients": pd.DataFrame(coef_rows),
        "oos_predictions": pred,
        "bet_rows": bet_rows,
        "residual_calibration": pd.DataFrame(cal_rows),
        "model_rankings": pd.concat(rank_rows, ignore_index=True, sort=False) if rank_rows else pd.DataFrame(),
        "chronology": pd.DataFrame(chronology),
        "coverage": coverage,
        "oos_blocks_completed": int(pred["fold"].nunique()),
        "oos_block_size": int(oos_block_size),
        "standard_price": int(standard_price),
        "top_n_grid": list(top_n_grid),
        "ranking_methods": list(ranking_methods),
        "edge_grid": list(edge_grid),
        "common_line_games": bool(common_line_games),
        "common_line_game_count": len(common_keys),
    }


def _normalize_team(s: str) -> str:
    s = str(s or "").lower().replace("&", " and ")
    aliases = {
        "uva": "virginia", "virginia tech": "virginia tech", "va tech": "virginia tech",
        "jax state": "jacksonville state", "jax st": "jacksonville state", "jacksonville st": "jacksonville state",
        "miss st": "mississippi state", "mississippi st": "mississippi state",
        "isu": "iowa state", "cmu": "central michigan", "odu": "old dominion", "uconn": "connecticut",
        "asu": "arizona state", "sdsu": "san diego state", "ecu": "east carolina", "smu": "smu",
        "unc": "north carolina", "lsu": "lsu", "usc": "usc", "byu": "byu", "unlv": "unlv", "utep": "utep",
        "ucf": "ucf", "tcu": "tcu",
    }
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(st|state)\.?\b", " state ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return aliases.get(s, s)


def parse_live_bet_label(label: str) -> dict:
    raw = str(label or "").strip()
    is_ml = bool(re.search(r"\bML\b", raw, flags=re.I))
    m = re.search(r"\s([+-]\d+(?:\.5)?)\s*$", raw)
    spread = float(m.group(1)) if m else np.nan
    team = raw[:m.start()].strip() if m else re.sub(r"\s+ML\s*$", "", raw, flags=re.I).strip()
    return {"bet_team": team, "bet_team_norm": _normalize_team(team), "bet_spread": spread, "is_moneyline": is_ml}


def load_live_bets_reference(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    q = pd.read_csv(p)
    parsed = pd.DataFrame([parse_live_bet_label(x) for x in q.get("Bet", pd.Series(dtype=str))])
    q = pd.concat([q.reset_index(drop=True), parsed], axis=1)
    if "Date" in q.columns:
        q["Date"] = q["Date"].replace("", np.nan).ffill()
    return q


def match_live_spread_bets(live_bets: pd.DataFrame, oos_predictions: pd.DataFrame, *, ranking_method: str = "wilson", top_n: int = 10, line_reference: str = "Close (PT Updated/final)", team_similarity_min: float = 0.72, spread_tolerance: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if live_bets is None or live_bets.empty or oos_predictions is None or oos_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = oos_predictions[
        (oos_predictions["ranking_method"].astype(str) == str(ranking_method))
        & (pd.to_numeric(oos_predictions["top_n"], errors="coerce") == int(top_n))
        & (oos_predictions["line_reference"].astype(str) == str(line_reference))
        & (pd.to_numeric(oos_predictions["season"], errors="coerce") == 2025)
    ].copy()
    p = p.drop_duplicates("game_key")
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()
    p["road_norm"] = p.get("road", "").map(_normalize_team)
    p["home_norm"] = p.get("home", "").map(_normalize_team)
    matches, unmatched = [], []
    for i, r in live_bets.reset_index(drop=True).iterrows():
        if bool(r.get("is_moneyline", False)) or not np.isfinite(pd.to_numeric(r.get("bet_spread"), errors="coerce")):
            continue
        team = str(r.get("bet_team_norm", "")); spread = float(r["bet_spread"])
        candidates = []
        for _, g in p.iterrows():
            sr = SequenceMatcher(None, team, str(g["road_norm"])).ratio()
            sh = SequenceMatcher(None, team, str(g["home_norm"])).ratio()
            is_home = sh >= sr; sim = max(sr, sh)
            if sim < float(team_similarity_min):
                continue
            market = float(g["market_margin"])
            implied_selected_spread = -market if is_home else market
            dist = abs(implied_selected_spread - spread)
            score = sim - 0.035 * min(dist, 10.0)
            candidates.append((score, dist, sim, is_home, implied_selected_spread, g))
        candidates.sort(key=lambda z: (z[0], -z[1]), reverse=True)
        if not candidates or candidates[0][1] > float(spread_tolerance):
            unmatched.append({**r.to_dict(), "match_reason": "no unique team/spread match"})
            continue
        best = candidates[0]
        if len(candidates) > 1 and abs(best[0] - candidates[1][0]) < 0.02:
            unmatched.append({**r.to_dict(), "match_reason": "ambiguous team/spread match"})
            continue
        g = best[5]
        selected_home = bool(best[3])
        model_edge_selected = float(g["model_residual"]) * (1.0 if selected_home else -1.0)
        adjusted_edge_selected = float(g["adjusted_edge"]) * (1.0 if selected_home else -1.0)
        matches.append({
            **r.to_dict(), "game_key": g["game_key"], "season": int(g["season"]), "week": int(g["week"]),
            "road": g.get("road", ""), "home": g.get("home", ""), "match_similarity": best[2],
            "market_selected_spread": best[4], "spread_difference": best[1],
            "market_home_margin": g["market_margin"], "actual_home_margin": g["actual_margin"],
            "consensus_home_margin": g["consensus_margin"], "model_residual_home": g["model_residual"],
            "adjusted_home_margin": g["pred_market_signal"], "adjusted_edge_home": g["adjusted_edge"],
            "model_edge_selected_side": model_edge_selected, "adjusted_edge_selected_side": adjusted_edge_selected,
            "models_available": g.get("models_available", np.nan), "gamma": g.get("gamma", np.nan),
        })
    return pd.DataFrame(matches), pd.DataFrame(unmatched)


def live_forensics_summary(matches: pd.DataFrame) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame()
    q = matches.copy()
    q["abs_spread"] = pd.to_numeric(q["bet_spread"], errors="coerce").abs()
    q["won"] = q.get("Win/Loss", "").astype(str).str.upper().eq("W")
    q["lost"] = q.get("Win/Loss", "").astype(str).str.upper().eq("L")
    q["model_agrees"] = pd.to_numeric(q["model_edge_selected_side"], errors="coerce") > 0
    q["adjusted_agrees"] = pd.to_numeric(q["adjusted_edge_selected_side"], errors="coerce") > 0
    bins = [-np.inf, 3.5, 7, 10, 14, 17, 21, 28, np.inf]
    labels = ["≤3.5", "4–7", "7.5–10", "10.5–14", "14.5–17", "17.5–21", "21.5–28", ">28"]
    q["spread_bucket"] = pd.cut(q["abs_spread"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for b, g in q.groupby("spread_bucket", observed=True):
        graded = g[g["won"] | g["lost"]]
        rows.append({
            "spread_bucket": str(b), "matched_bets": len(g), "graded_bets": len(graded),
            "live_win_pct": float(graded["won"].mean()) if len(graded) else np.nan,
            "raw_model_agreement_pct": float(g["model_agrees"].mean()),
            "market_anchored_agreement_pct": float(g["adjusted_agrees"].mean()),
            "mean_abs_model_edge": float(pd.to_numeric(g["model_edge_selected_side"], errors="coerce").abs().mean()),
        })
    return pd.DataFrame(rows)


def save_market_signal_outputs(result: dict, root: str | Path) -> dict[str, str]:
    root = Path(root); outdir = root / "data" / "derived"; outdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "forecast_summary": "market_signal_forecast_summary.csv",
        "betting_summary": "market_signal_betting_summary.csv",
        "fold_coefficients": "market_signal_fold_coefficients.csv",
        "oos_predictions": "market_signal_oos_predictions.csv",
        "residual_calibration": "market_signal_residual_calibration.csv",
        "model_rankings": "market_signal_model_rankings.csv",
        "chronology": "market_signal_chronology.csv",
        "live_matches": "market_signal_live_matches.csv",
        "live_unmatched": "market_signal_live_unmatched.csv",
        "live_forensics": "market_signal_live_forensics.csv",
    }
    written = {}
    for key, name in mapping.items():
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            p = outdir / name; frame.to_csv(p, index=False); written[key] = str(p.relative_to(root))
    status = {
        "version": "v3.5.45-market-anchored-signal",
        "oos_blocks_completed": int(result.get("oos_blocks_completed", 0)),
        "oos_block_size": int(result.get("oos_block_size", 0)),
        "standard_price": int(result.get("standard_price", -110)),
        "top_n_grid": result.get("top_n_grid", list(TOP_N_GRID)),
        "ranking_methods": result.get("ranking_methods", list(RANKING_METHODS)),
        "edge_grid": result.get("edge_grid", list(BET_EDGE_GRID)),
        "common_line_game_count": int(result.get("common_line_game_count", 0)),
        "outputs": written,
    }
    sp = outdir / "market_signal_status.json"; sp.write_text(json.dumps(status, indent=2), encoding="utf-8"); written["status"] = str(sp.relative_to(root))
    return written


def load_market_signal_outputs(root: str | Path) -> dict:
    root = Path(root); sp = root / "data" / "derived" / "market_signal_status.json"
    if not sp.exists(): return {}
    try: status = json.loads(sp.read_text(encoding="utf-8"))
    except Exception: return {}
    out = dict(status)
    for key, rel in (status.get("outputs") or {}).items():
        p = root / rel
        if p.exists() and p.suffix.lower() == ".csv":
            try: out[key] = pd.read_csv(p)
            except Exception: pass
    return out
