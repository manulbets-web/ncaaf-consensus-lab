from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import json
import math

import numpy as np
import pandas as pd

from committee import (
    _matrix_and_meta,
    _combo_forecast_arrays,
    _signal,
    analyze_finalist_portfolio,
)
from formal_backtest import (
    FORMAL_ANCHOR_GRID,
    FORMAL_META_K_GRID,
    _build_formal_folds,
    _posting_models_for_period,
    _wilson_interval,
    _max_drawdown,
    _longest_losing_streak,
)
from line_movement import (
    LINE_REFS,
    _period_mask,
    _rank_candidates_for_reference,
    clean_line_history_for_analysis,
    data_for_line_reference,
)
from streamlined_engine import CombinationSearchConfig, brute_force_combination_search, combination_count


ABLATION_ARCHITECTURES = (
    ("A", "All-model consensus"),
    ("B", "Prior-ranked Top-N consensus"),
    ("C", "Best prior single model"),
    ("D", "Top-N diversified model consensus"),
    ("E", "Exact combinations → Top-50 → META"),
)
ABLATION_ARCH_MAP = dict(ABLATION_ARCHITECTURES)
PURE_THRESHOLD_GRID = FORMAL_ANCHOR_GRID
DEFAULT_COMBO_SEARCH_ANCHOR = 0.75
DEFAULT_DUPLICATE_CORR = 0.90


def _common_line_game_keys(line_history: pd.DataFrame) -> set[str]:
    if line_history is None or line_history.empty:
        return set()
    q = line_history.copy()
    for c in ["open_margin", "midweek_margin", "close_margin"]:
        q[c] = pd.to_numeric(q.get(c), errors="coerce")
    good = q[["open_margin", "midweek_margin", "close_margin"]].notna().all(axis=1)
    return set(q.loc[good, "game_key"].astype(str))


def _decorate_frame(frame: pd.DataFrame, meta: pd.DataFrame, *, architecture: str, line_reference: str, gate: np.ndarray) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy().reset_index(drop=True)
    if "game_key" not in out.columns:
        out["game_key"] = meta.index.astype(str).to_numpy()
    if "season" not in out.columns:
        out["season"] = pd.to_numeric(meta["season"], errors="coerce").to_numpy()
    if "week" not in out.columns:
        out["week"] = pd.to_numeric(meta["week"], errors="coerce").to_numpy()
    if "market_margin" not in out.columns:
        out["market_margin"] = pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy()
    if "actual_margin" not in out.columns:
        out["actual_margin"] = pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy()
    if "cover" not in out.columns:
        out["cover"] = pd.to_numeric(out["actual_margin"], errors="coerce") - pd.to_numeric(out["market_margin"], errors="coerce")
    out["architecture"] = str(architecture)
    out["architecture_name"] = ABLATION_ARCH_MAP.get(str(architecture), str(architecture))
    out["line_reference"] = str(line_reference)
    out["gate"] = np.asarray(gate, dtype=bool)
    return out


def _mean_consensus_frame(
    data_ref: pd.DataFrame,
    model_ids: Iterable[str],
    periods: Iterable[tuple[int, int]],
    *,
    architecture: str,
    line_reference: str,
    min_available_models: int,
) -> pd.DataFrame:
    ids = list(dict.fromkeys(map(str, model_ids)))
    pred, meta = _matrix_and_meta(data_ref, ids, periods)
    if pred.empty:
        return pd.DataFrame()
    count, mean, sd = _combo_forecast_arrays(pred, ids, min_available_models=max(1, int(min_available_models)))
    market = pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy(float)
    actual = pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(float)
    edge = mean - market
    sig = _signal(edge, sd)
    # A one-model architecture has zero dispersion by definition; any non-zero
    # disagreement is treated as infinite standardized edge, so the pure-k sweep
    # does not manufacture an uncertainty estimate that does not exist.
    if len(ids) == 1:
        finite = np.isfinite(edge) & (np.abs(edge) > 1e-12)
        sig[finite] = np.inf
        sd[finite] = 0.0
    gate = (count >= max(1, int(min_available_models))) & np.isfinite(mean) & np.isfinite(market) & np.isfinite(actual)
    frame = pd.DataFrame({
        "game_key": pred.index.astype(str),
        "season": pd.to_numeric(meta["season"], errors="coerce").to_numpy(),
        "week": pd.to_numeric(meta["week"], errors="coerce").to_numpy(),
        "market_margin": market,
        "actual_margin": actual,
        "cover": actual - market,
        "forecast_mean": mean,
        "forecast_sd": sd,
        "edge": edge,
        "signal": sig,
        "active_units": count,
    })
    for c in ["road", "home"]:
        if c in meta.columns:
            frame[c] = meta[c].astype(str).to_numpy()
    return _decorate_frame(frame, meta, architecture=architecture, line_reference=line_reference, gate=gate)


def _correlation_communities(
    data_ref: pd.DataFrame,
    model_ids: Iterable[str],
    discovery_periods: Iterable[tuple[int, int]],
    *,
    corr_threshold: float = DEFAULT_DUPLICATE_CORR,
    min_overlap: int = 20,
) -> list[list[str]]:
    ids = list(dict.fromkeys(map(str, model_ids)))
    if not ids:
        return []
    pred, _ = _matrix_and_meta(data_ref, ids, discovery_periods)
    if pred.empty or len(ids) == 1:
        return [[x] for x in ids]
    arr = pred.reindex(columns=ids)
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(ids)):
        xi = pd.to_numeric(arr.iloc[:, i], errors="coerce")
        for j in range(i + 1, len(ids)):
            xj = pd.to_numeric(arr.iloc[:, j], errors="coerce")
            ok = xi.notna() & xj.notna()
            if int(ok.sum()) < int(min_overlap):
                continue
            cor = xi[ok].corr(xj[ok])
            if pd.notna(cor) and float(cor) >= float(corr_threshold):
                union(i, j)
    groups: dict[int, list[str]] = {}
    for i, mid in enumerate(ids):
        groups.setdefault(find(i), []).append(mid)
    return list(groups.values())


def _diversified_model_frame(
    data_ref: pd.DataFrame,
    model_ids: Iterable[str],
    discovery_periods: Iterable[tuple[int, int]],
    validation_periods: Iterable[tuple[int, int]],
    *,
    line_reference: str,
    corr_threshold: float = DEFAULT_DUPLICATE_CORR,
) -> tuple[pd.DataFrame, list[list[str]]]:
    ids = list(dict.fromkeys(map(str, model_ids)))
    communities = _correlation_communities(
        data_ref, ids, discovery_periods,
        corr_threshold=float(corr_threshold), min_overlap=20,
    )
    pred, meta = _matrix_and_meta(data_ref, ids, validation_periods)
    if pred.empty:
        return pd.DataFrame(), communities
    unit_means = []
    unit_vars = []
    for grp in communities:
        cols = [m for m in grp if m in pred.columns]
        if not cols:
            continue
        x = pred[cols].to_numpy(dtype=float)
        finite = np.isfinite(x)
        n = finite.sum(axis=1)
        sums = np.nansum(x, axis=1)
        mean = np.divide(sums, n, out=np.full(len(pred), np.nan), where=n > 0)
        sd = np.full(len(pred), np.nan)
        if len(cols) >= 2:
            sq = np.nansum(np.square(x), axis=1)
            var = np.divide(
                sq - np.divide(np.square(sums), n, out=np.zeros(len(pred)), where=n > 0),
                n - 1, out=np.full(len(pred), np.nan), where=n > 1,
            )
            sd = np.sqrt(np.maximum(var, 0.0))
        # Uncertainty of each community mean; singleton communities contribute
        # zero within-community variance rather than being discarded.
        var_mean = np.where(n > 1, np.square(sd) / np.maximum(n, 1), 0.0)
        mean[n <= 0] = np.nan
        var_mean[n <= 0] = np.nan
        unit_means.append(mean)
        unit_vars.append(var_mean)
    if not unit_means:
        return pd.DataFrame(), communities
    means = np.vstack(unit_means).T
    vars_ = np.vstack(unit_vars).T
    active = np.isfinite(means).sum(axis=1)
    meta_mean = np.nanmean(means, axis=1)
    within = np.nanmean(vars_, axis=1)
    between = np.full(len(pred), 0.0)
    for i in range(len(pred)):
        vals = means[i, np.isfinite(means[i])]
        between[i] = float(np.var(vals, ddof=1)) if len(vals) >= 2 else 0.0
    total_var = within + between
    meta_sd = np.sqrt(np.maximum(total_var, 0.0))
    market = pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy(float)
    actual = pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(float)
    edge = meta_mean - market
    sig = _signal(edge, meta_sd)
    gate = (active >= 1) & np.isfinite(meta_mean) & np.isfinite(market) & np.isfinite(actual)
    frame = pd.DataFrame({
        "game_key": pred.index.astype(str),
        "season": pd.to_numeric(meta["season"], errors="coerce").to_numpy(),
        "week": pd.to_numeric(meta["week"], errors="coerce").to_numpy(),
        "market_margin": market,
        "actual_margin": actual,
        "cover": actual - market,
        "forecast_mean": meta_mean,
        "forecast_sd": meta_sd,
        "edge": edge,
        "signal": sig,
        "active_units": active,
    })
    for c in ["road", "home"]:
        if c in meta.columns:
            frame[c] = meta[c].astype(str).to_numpy()
    return _decorate_frame(frame, meta, architecture="D", line_reference=line_reference, gate=gate), communities


def _exact_meta_frame(
    data_ref: pd.DataFrame,
    ranked_ids: list[str],
    model_name_map: dict[str, str],
    discovery_periods: Iterable[tuple[int, int]],
    validation_periods: Iterable[tuple[int, int]],
    *,
    line_reference: str,
    search_anchor: float = DEFAULT_COMBO_SEARCH_ANCHOR,
    min_size: int = 3,
    max_size: int = 6,
    min_available_models: int = 3,
    min_search_bets: int = 50,
    finalists: int = 50,
    overlap_threshold: float = 0.50,
    min_meta_communities: int = 2,
    meta_thresholds: Iterable[float] = FORMAL_META_K_GRID,
    standard_price: int = -110,
    max_combinations: int = 10_000_000,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    ids = list(dict.fromkeys(map(str, ranked_ids)))
    hi = min(int(max_size), len(ids))
    if len(ids) < int(min_size):
        return pd.DataFrame(), {"status": "too few candidate models"}
    seasons = tuple(sorted(set(y for y, _ in tuple(discovery_periods) + tuple(validation_periods))))
    cfg = CombinationSearchConfig(
        search_seasons=seasons,
        validation_seasons=seasons,
        search_periods=tuple(discovery_periods),
        validation_periods=tuple(validation_periods),
        min_size=int(min_size), max_size=int(hi), primary_k=float(search_anchor),
        min_available_models=int(min_available_models), min_search_bets=int(min_search_bets),
        min_seasons_represented=1, min_distinct_weeks=1, ranking_metric="ats",
        standard_price=int(standard_price), chunk_size=512, top_n=int(finalists),
        max_combinations=int(max_combinations),
    )
    search = brute_force_combination_search(
        data_ref, ids, model_name_map, cfg, progress_callback=progress_callback
    )
    top = search.get("top", pd.DataFrame()).head(int(finalists)).copy()
    combos = []
    for j, r in top.reset_index(drop=True).iterrows():
        raw = r.get("model_ids", "")
        mids = [x for x in str(raw).split("|") if x]
        combos.append({"rank": int(r.get("search_rank", j + 1)), "model_ids": mids})
    if len(combos) < 2:
        return pd.DataFrame(), {"status": "too few finalists", "search": search}
    analysis = analyze_finalist_portfolio(
        data_ref, combos, discovery_periods, validation_periods,
        min_available_models=int(min_available_models), thresholds=tuple(meta_thresholds),
        combo_min_bets=int(min_search_bets), meta_min_bets=int(min_search_bets),
        overlap_threshold=float(overlap_threshold), min_meta_communities=int(min_meta_communities),
        standard_price=int(standard_price), line_history=None,
    )
    frame = analysis.get("meta_frames", {}).get(("Diversified META", "holdout"), pd.DataFrame()).copy()
    if frame is None or frame.empty:
        return pd.DataFrame(), {"status": "no META frame", "search": search, "analysis": analysis}
    frame = frame.rename(columns={"meta_mean": "forecast_mean", "meta_sd": "forecast_sd", "meta_edge": "edge", "meta_signal": "signal"})
    gate = pd.to_numeric(frame.get("active_units"), errors="coerce").fillna(0).to_numpy(float) >= int(min_meta_communities)
    # Add matchup labels for QC display.
    meta = (
        data_ref.loc[_period_mask(data_ref, validation_periods), [c for c in ["game_key", "road", "home"] if c in data_ref.columns]]
        .drop_duplicates("game_key")
    )
    if len(meta):
        frame = frame.merge(meta, on="game_key", how="left")
    frame = _decorate_frame(frame, frame.set_index("game_key"), architecture="E", line_reference=line_reference, gate=gate)
    info = {
        "status": "ok",
        "evaluated_combinations": int(search.get("evaluated_combinations", 0)),
        "eligible_combinations": int(search.get("eligible_combinations", 0)),
        "finalists": len(combos),
        "communities": int(analysis.get("overlap_summary", {}).get("communities", 0)),
        "search_anchor": float(search_anchor),
    }
    return frame, info


def _threshold_rows(frame: pd.DataFrame, *, fold: int, thresholds: Iterable[float], standard_price: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = frame.copy()
    edge = pd.to_numeric(q.get("edge"), errors="coerce").to_numpy(float)
    sig = pd.to_numeric(q.get("signal"), errors="coerce").to_numpy(float)
    cover = pd.to_numeric(q.get("cover"), errors="coerce").to_numpy(float)
    gate = q.get("gate", pd.Series(True, index=q.index)).fillna(False).astype(bool).to_numpy()
    price = float(standard_price)
    flat_win = 100.0 / abs(price) if price < 0 else price / 100.0
    win1_risk = abs(price) / 100.0 if price < 0 else 100.0 / price
    rows = []
    bet_frames = []
    for k in thresholds:
        directional = gate & np.isfinite(edge) & np.isfinite(cover) & (np.abs(edge) > 1e-12)
        selected = directional & ((np.isfinite(sig) | np.isinf(sig)) & (sig >= float(k)))
        signed = np.sign(edge) * cover
        win = selected & (signed > 1e-12)
        loss = selected & (signed < -1e-12)
        push = selected & (np.abs(signed) <= 1e-12)
        wins = int(win.sum()); losses = int(loss.sum()); pushes = int(push.sum()); bets = wins + losses
        units_flat = wins * flat_win - losses
        units_win1 = wins - losses * win1_risk
        risk_flat = float(bets)
        risk_win1 = float(bets) * win1_risk
        wl, wu = _wilson_interval(wins, bets)
        rows.append({
            "fold": int(fold), "architecture": str(q["architecture"].iloc[0]),
            "architecture_name": str(q["architecture_name"].iloc[0]),
            "line_reference": str(q["line_reference"].iloc[0]), "threshold": float(k),
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "ats_pct": wins / bets if bets else np.nan, "wilson_low": wl, "wilson_high": wu,
            "units_flat": units_flat, "roi_flat": units_flat / risk_flat if risk_flat else np.nan,
            "units_win1": units_win1, "roi_win1": units_win1 / risk_win1 if risk_win1 else np.nan,
        })
        if np.any(selected):
            b = q.loc[selected].copy()
            ix = np.flatnonzero(selected)
            b["fold"] = int(fold); b["threshold"] = float(k)
            b["side"] = np.sign(edge[ix])
            b["win"] = win[ix]; b["loss"] = loss[ix]; b["push"] = push[ix]
            b["cover_margin"] = signed[ix]
            b["units_flat"] = np.where(win[ix], flat_win, np.where(loss[ix], -1.0, 0.0))
            b["units_win1"] = np.where(win[ix], 1.0, np.where(loss[ix], -win1_risk, 0.0))
            bet_frames.append(b)
    return pd.DataFrame(rows), (pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame())


def _aggregate_thresholds(fold_stats: pd.DataFrame, bet_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_stats is None or fold_stats.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["architecture", "architecture_name", "line_reference", "threshold"]
    for keys, q in fold_stats.groupby(group_cols, sort=True, dropna=False):
        arch, name, line, k = keys
        wins = int(pd.to_numeric(q["wins"], errors="coerce").fillna(0).sum())
        losses = int(pd.to_numeric(q["losses"], errors="coerce").fillna(0).sum())
        pushes = int(pd.to_numeric(q["pushes"], errors="coerce").fillna(0).sum())
        bets = wins + losses
        units_flat = float(pd.to_numeric(q["units_flat"], errors="coerce").fillna(0).sum())
        units_win1 = float(pd.to_numeric(q["units_win1"], errors="coerce").fillna(0).sum())
        risk_win1 = bets * 1.1
        wl, wu = _wilson_interval(wins, bets)
        br = bet_rows[
            (bet_rows["architecture"].astype(str) == str(arch))
            & (bet_rows["line_reference"].astype(str) == str(line))
            & np.isclose(pd.to_numeric(bet_rows["threshold"], errors="coerce"), float(k))
        ].copy() if isinstance(bet_rows, pd.DataFrame) and len(bet_rows) else pd.DataFrame()
        if len(br):
            br = br.sort_values(["season", "week", "game_key"], kind="mergesort")
        rows.append({
            "architecture": arch, "architecture_name": name, "line_reference": line, "threshold": float(k),
            "blocks": int(q["fold"].nunique()), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "ats_pct": wins / bets if bets else np.nan, "wilson_low": wl, "wilson_high": wu,
            "units_flat": units_flat, "roi_flat": units_flat / bets if bets else np.nan,
            "units_win1": units_win1, "roi_win1": units_win1 / risk_win1 if risk_win1 else np.nan,
            "profitable_blocks": int((pd.to_numeric(q["units_flat"], errors="coerce") > 0).sum()),
            "max_drawdown_flat": _max_drawdown(br["units_flat"]) if len(br) else np.nan,
            "max_drawdown_win1": _max_drawdown(br["units_win1"]) if len(br) else np.nan,
            "longest_losing_streak": _longest_losing_streak(br) if len(br) else 0,
        })
    return pd.DataFrame(rows).sort_values(["line_reference", "architecture", "threshold"]).reset_index(drop=True)


def _qc_from_bets(bet_rows: pd.DataFrame, *, architecture: str = "E", line_reference: str = "Close (PT Updated/final)", threshold: float = 0.75, n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bet_rows is None or bet_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = bet_rows[
        (bet_rows["architecture"].astype(str) == str(architecture))
        & (bet_rows["line_reference"].astype(str) == str(line_reference))
        & np.isclose(pd.to_numeric(bet_rows["threshold"], errors="coerce"), float(threshold))
    ].copy()
    if q.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = q.sort_values(["season", "week", "game_key"], kind="mergesort").reset_index(drop=True)
    wins = int(q["win"].fillna(False).sum()); losses = int(q["loss"].fillna(False).sum()); pushes = int(q["push"].fillna(False).sum())
    bets = wins + losses
    owins, olosses = losses, wins
    summary = pd.DataFrame([{
        "architecture": architecture, "line_reference": line_reference, "threshold": float(threshold),
        "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
        "ats_pct": wins / bets if bets else np.nan,
        "opposite_wins": owins, "opposite_losses": olosses,
        "opposite_ats_pct": owins / bets if bets else np.nan,
        "ats_plus_opposite": (wins + owins) / bets if bets else np.nan,
        "orientation_check": "PASS" if bets and (wins + owins == bets) else "CHECK",
    }])
    take = min(int(n), len(q))
    if take < len(q):
        idx = np.unique(np.linspace(0, len(q) - 1, num=take, dtype=int))
        sample = q.iloc[idx].copy()
    else:
        sample = q.copy()
    sample["game"] = np.where(
        sample.get("road", pd.Series("", index=sample.index)).astype(str).str.len().gt(0),
        sample.get("road", "").astype(str) + " @ " + sample.get("home", "").astype(str),
        sample["game_key"].astype(str),
    )
    sample["selected_side"] = np.where(pd.to_numeric(sample["side"], errors="coerce") > 0, sample.get("home", "Home"), sample.get("road", "Away"))
    sample["result"] = np.where(sample["win"], "W", np.where(sample["loss"], "L", "P"))
    keep = [
        "fold", "season", "week", "game", "selected_side", "forecast_mean", "forecast_sd",
        "market_margin", "actual_margin", "edge", "signal", "cover_margin", "result",
    ]
    return summary, sample[[c for c in keep if c in sample.columns]]


def run_ablation_walkforward_backtest(
    data: pd.DataFrame,
    line_history: pd.DataFrame,
    model_name_map: dict[str, str],
    *,
    period_scope: Iterable[tuple[int, int]] | None = None,
    thresholds: Iterable[float] = PURE_THRESHOLD_GRID,
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
    combo_search_anchor: float = DEFAULT_COMBO_SEARCH_ANCHOR,
    duplicate_corr_threshold: float = DEFAULT_DUPLICATE_CORR,
    standard_price: int = -110,
    max_combinations: int = 10_000_000,
    common_line_games: bool = True,
    qc_threshold: float = 0.75,
    qc_n: int = 100,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Ablation-style chronological validation on identical future blocks.

    Architecture table uses threshold k=0 to isolate forecast/selection quality.
    The threshold surface then freezes each block's architecture and changes only
    the execution gate.  E uses the predeclared 0.75 combination-search anchor;
    its *execution* threshold is swept independently from 0.00 to 2.00.
    """
    thresholds = tuple(sorted(set(float(k) for k in thresholds)))
    clean_lines = clean_line_history_for_analysis(line_history, move_threshold=10.0, exclude_suspect_open=True)
    if clean_lines is None or clean_lines.empty:
        raise ValueError("Historical PredictionTracker Open/Midweek/Updated line history is unavailable.")
    common_keys = _common_line_game_keys(clean_lines) if common_line_games else set()
    close_data = data_for_line_reference(data, clean_lines, "close_margin")
    if common_line_games:
        close_data = close_data[close_data["game_key"].astype(str).isin(common_keys)].copy()
    folds, coverage = _build_formal_folds(
        close_data, period_scope=period_scope, block_size=int(oos_block_size), n_folds=int(oos_blocks),
        min_discovery_periods=int(min_discovery_periods), min_games_per_period=int(min_games_per_period),
        min_available_models=int(min_available_models),
    )
    if not folds:
        raise ValueError("No ablation OOS folds could be formed with the requested chronology settings.")

    fold_stats = []
    bet_frames = []
    chronology = []
    detail_rows = []
    total_steps = len(folds) * len(LINE_REFS) * 1000

    for fi, fold in enumerate(folds, start=1):
        first_oos = tuple(fold["validation_periods"])[0]
        posting_ids = _posting_models_for_period(data, first_oos)
        for li, (line_label, line_col) in enumerate(LINE_REFS, start=1):
            base = ((fi - 1) * len(LINE_REFS) + (li - 1)) * 1000
            def prog(local: int, label: str):
                if progress_callback is not None:
                    progress_callback(base + int(local), total_steps, f"OOS {fi}/{len(folds)} · {line_label} · {label}")
            ref_data = data_for_line_reference(data, clean_lines, line_col)
            if common_line_games:
                ref_data = ref_data[ref_data["game_key"].astype(str).isin(common_keys)].copy()
            ranked = _rank_candidates_for_reference(
                ref_data, fold["discovery_periods"], posting_ids,
                pool_n=int(pool_n), pool_min_bets=int(pool_min_bets), pool_metric="wilson", standard_price=int(standard_price),
            )
            ranked_ids = ranked["canonical_model_id"].astype(str).tolist() if len(ranked) else []
            prog(20, f"posting={len(posting_ids)} · ranked={len(ranked_ids)}")
            frames: dict[str, pd.DataFrame] = {}
            frames["A"] = _mean_consensus_frame(
                ref_data, posting_ids, fold["validation_periods"], architecture="A", line_reference=line_label,
                min_available_models=int(min_available_models),
            )
            frames["B"] = _mean_consensus_frame(
                ref_data, ranked_ids, fold["validation_periods"], architecture="B", line_reference=line_label,
                min_available_models=int(min_available_models),
            ) if ranked_ids else pd.DataFrame()
            frames["C"] = _mean_consensus_frame(
                ref_data, ranked_ids[:1], fold["validation_periods"], architecture="C", line_reference=line_label,
                min_available_models=1,
            ) if ranked_ids else pd.DataFrame()
            dframe, communities = _diversified_model_frame(
                ref_data, ranked_ids, fold["discovery_periods"], fold["validation_periods"],
                line_reference=line_label, corr_threshold=float(duplicate_corr_threshold),
            ) if ranked_ids else (pd.DataFrame(), [])
            frames["D"] = dframe
            prog(100, f"A-D ready · D communities={len(communities)}")

            e_info = {"status": "skipped"}
            if len(ranked_ids) >= int(min_size):
                nwork = combination_count(len(ranked_ids), int(min_size), min(int(max_size), len(ranked_ids)))
                def e_progress(done: int, total: int, label: str):
                    frac = float(done) / float(total) if total else 0.0
                    prog(120 + int(760 * max(0.0, min(1.0, frac))), f"E exact search · {label}")
                eframe, e_info = _exact_meta_frame(
                    ref_data, ranked_ids, model_name_map, fold["discovery_periods"], fold["validation_periods"],
                    line_reference=line_label, search_anchor=float(combo_search_anchor),
                    min_size=int(min_size), max_size=int(max_size), min_available_models=int(min_available_models),
                    min_search_bets=int(min_search_bets), finalists=int(finalists), overlap_threshold=float(overlap_threshold),
                    min_meta_communities=int(min_meta_communities), meta_thresholds=tuple(meta_thresholds),
                    standard_price=int(standard_price), max_combinations=int(max_combinations), progress_callback=e_progress,
                )
                frames["E"] = eframe
                e_info["candidate_combinations"] = int(nwork)
            else:
                frames["E"] = pd.DataFrame()

            for arch, _name in ABLATION_ARCHITECTURES:
                frame = frames.get(arch, pd.DataFrame())
                st, br = _threshold_rows(frame, fold=fi, thresholds=thresholds, standard_price=int(standard_price))
                if len(st):
                    st["oos_start"] = f"{fold['validation_start'][0]} W{fold['validation_start'][1]}"
                    st["oos_end"] = f"{fold['validation_end'][0]} W{fold['validation_end'][1]}"
                    fold_stats.append(st)
                if len(br):
                    br["oos_start"] = f"{fold['validation_start'][0]} W{fold['validation_start'][1]}"
                    br["oos_end"] = f"{fold['validation_end'][0]} W{fold['validation_end'][1]}"
                    bet_frames.append(br)
            detail_rows.append({
                "fold": fi, "line_reference": line_label,
                "oos_start": f"{fold['validation_start'][0]} W{fold['validation_start'][1]}",
                "oos_end": f"{fold['validation_end'][0]} W{fold['validation_end'][1]}",
                "posting_models": len(posting_ids), "ranked_models": len(ranked_ids),
                "D_communities": len(communities),
                "E_status": e_info.get("status", ""), "E_finalists": e_info.get("finalists", 0),
                "E_communities": e_info.get("communities", 0),
                "E_evaluated_combinations": e_info.get("evaluated_combinations", 0),
            })
            prog(1000, "complete")
        chronology.append({
            "fold": fi, "discovery_start": f"{fold['discovery_start'][0]} W{fold['discovery_start'][1]}",
            "discovery_end": f"{fold['discovery_end'][0]} W{fold['discovery_end'][1]}",
            "oos_start": f"{fold['validation_start'][0]} W{fold['validation_start'][1]}",
            "oos_end": f"{fold['validation_end'][0]} W{fold['validation_end'][1]}",
            "oos_periods": len(fold["validation_periods"]), "posting_models": len(posting_ids),
        })

    fold_df = pd.concat(fold_stats, ignore_index=True, sort=False) if fold_stats else pd.DataFrame()
    bet_df = pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame()
    surface = _aggregate_thresholds(fold_df, bet_df)
    if surface.empty:
        raise RuntimeError("Ablation walk-forward backtest produced no scorable OOS results.")
    headline = surface[np.isclose(pd.to_numeric(surface["threshold"], errors="coerce"), 0.0)].copy()
    qc_summary, qc_sample = _qc_from_bets(
        bet_df, architecture="E", line_reference="Close (PT Updated/final)", threshold=float(qc_threshold), n=int(qc_n)
    )
    monotonic_rows = []
    for (arch, line), q in surface.groupby(["architecture", "line_reference"], sort=True):
        q = q.sort_values("threshold")
        bets = pd.to_numeric(q["bets"], errors="coerce").fillna(0).to_numpy(float)
        monotonic_rows.append({
            "architecture": arch, "line_reference": line,
            "bet_count_monotone_nonincreasing": bool(np.all(np.diff(bets) <= 1e-9)),
            "bets_k0": int(bets[0]) if len(bets) else 0,
            "bets_k2": int(bets[-1]) if len(bets) else 0,
        })
    return {
        "architecture_summary": headline.reset_index(drop=True),
        "threshold_surface": surface,
        "fold_threshold_results": fold_df,
        "bet_rows": bet_df,
        "qc_summary": qc_summary,
        "qc_sample": qc_sample,
        "threshold_monotonicity": pd.DataFrame(monotonic_rows),
        "chronology": pd.DataFrame(chronology),
        "fold_detail": pd.DataFrame(detail_rows),
        "coverage": coverage,
        "oos_blocks_completed": int(pd.to_numeric(fold_df["fold"], errors="coerce").nunique()),
        "oos_block_size": int(oos_block_size),
        "standard_price": int(standard_price),
        "common_line_games": bool(common_line_games),
        "common_line_game_count": len(common_keys),
        "combo_search_anchor": float(combo_search_anchor),
        "duplicate_corr_threshold": float(duplicate_corr_threshold),
        "threshold_grid": thresholds,
        "selection_spec": {
            "A": "all historically posting models; equal-weight mean",
            "B": f"Top {int(pool_n)} prior Wilson models with >= {int(pool_min_bets)} prior bets; equal-weight mean",
            "C": "single highest prior Wilson model",
            "D": f"same Top {int(pool_n)} pool; >= {float(duplicate_corr_threshold):.2f} discovery prediction-correlation components collapsed, then communities equal-weighted",
            "E": f"current exact {int(min_size)}-{int(max_size)} model search at fixed {float(combo_search_anchor):g} SD search anchor -> Top {int(finalists)} -> overlap communities -> diversified META",
            "headline_execution": "k=0 directional disagreement only",
            "pure_threshold_sweep": "architecture frozen inside each OOS block; only execution k changes; bet counts must be non-increasing",
            "line_policy": "same OOS chronology and common verified Open/Midweek/Updated games" if common_line_games else "same OOS chronology; each line uses its available games",
        },
    }


def save_ablation_backtest_outputs(result: dict, root: str | Path) -> dict[str, str]:
    root = Path(root)
    outdir = root / "data" / "derived"
    outdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "architecture_summary": "ablation_backtest_architecture_summary.csv",
        "threshold_surface": "ablation_backtest_threshold_surface.csv",
        "fold_threshold_results": "ablation_backtest_fold_threshold_results.csv",
        "qc_summary": "ablation_backtest_qc_summary.csv",
        "qc_sample": "ablation_backtest_qc_sample.csv",
        "threshold_monotonicity": "ablation_backtest_threshold_monotonicity.csv",
        "chronology": "ablation_backtest_chronology.csv",
        "fold_detail": "ablation_backtest_fold_detail.csv",
    }
    written = {}
    for key, name in mapping.items():
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame):
            path = outdir / name
            frame.to_csv(path, index=False)
            written[key] = str(path.relative_to(root))
    status = {
        "version": "v3.5.44-ablation-validation",
        "oos_blocks_completed": int(result.get("oos_blocks_completed", 0)),
        "oos_block_size": int(result.get("oos_block_size", 0)),
        "standard_price": int(result.get("standard_price", -110)),
        "common_line_games": bool(result.get("common_line_games", True)),
        "common_line_game_count": int(result.get("common_line_game_count", 0)),
        "combo_search_anchor": float(result.get("combo_search_anchor", DEFAULT_COMBO_SEARCH_ANCHOR)),
        "duplicate_corr_threshold": float(result.get("duplicate_corr_threshold", DEFAULT_DUPLICATE_CORR)),
        "threshold_grid": list(result.get("threshold_grid", PURE_THRESHOLD_GRID)),
        "selection_spec": result.get("selection_spec", {}),
        "outputs": written,
    }
    sp = outdir / "ablation_backtest_status.json"
    sp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    written["status"] = str(sp.relative_to(root))
    return written


def load_ablation_backtest_outputs(root: str | Path) -> dict:
    root = Path(root)
    outdir = root / "data" / "derived"
    sp = outdir / "ablation_backtest_status.json"
    if not sp.exists():
        return {}
    try:
        status = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result = dict(status)
    for key, rel in (status.get("outputs") or {}).items():
        if key == "status":
            continue
        p = root / rel
        if p.exists() and p.suffix.lower() == ".csv":
            try:
                result[key] = pd.read_csv(p)
            except Exception:
                pass
    return result
