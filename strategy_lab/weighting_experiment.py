from __future__ import annotations

"""Within-ensemble reliability weighting experiment for frozen finalist combinations.

All reliability weights are fit on discovery data only and then frozen for holdout.
The experiment compares:
  * Equal weights (production baseline)
  * Reliability weights (inverse shrunk MSE, moderately capped)
  * Shrunk reliability weights (50% reliability + 50% equal, tighter cap)

Each combo/method receives its own discovery-only stable k.  The same frozen
weights and k are evaluated on holdout.  The diversified META layer is also
rebuilt/backtested under each internal weighting method.
"""

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd

from committee import (
    DEFAULT_K_GRID,
    _matrix_and_meta,
    _signal,
    _threshold_stats,
    _choose_stable_k,
)


WEIGHT_METHOD_LABELS = {
    "equal": "Equal",
    "reliability": "Reliability",
    "shrunk_reliability": "Shrunk reliability",
}


@dataclass(frozen=True)
class WeightingConfig:
    # Shrink noisy model MSE estimates toward the combo-wide median MSE.
    mse_history_shrink_k: float = 50.0
    # Reliability method: avoid one model becoming effectively the whole combo.
    reliability_min_relative: float = 0.25
    reliability_max_relative: float = 4.00
    # Shrunk reliability method: deliberately modest departure from equal.
    shrunk_equal_fraction: float = 0.50
    shrunk_min_relative: float = 0.50
    shrunk_max_relative: float = 2.00


def _normalize(weights: pd.Series, models: list[str]) -> pd.Series:
    w = pd.to_numeric(weights.reindex(models), errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(1.0 / max(len(models), 1), index=models, dtype=float)
    return w / total


def _cap_relative(weights: pd.Series, models: list[str], lo: float, hi: float) -> pd.Series:
    if not models:
        return pd.Series(dtype=float)
    w = _normalize(weights, models)
    equal = 1.0 / len(models)
    rel = (w / equal).clip(lower=float(lo), upper=float(hi))
    return _normalize(rel * equal, models)


def fit_combo_weights(
    pred_discovery: pd.DataFrame,
    meta_discovery: pd.DataFrame,
    model_ids: Iterable[str],
    *,
    config: WeightingConfig = WeightingConfig(),
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Fit all three weighting schemes from discovery outcomes only."""
    models = [str(m) for m in model_ids if str(m) in pred_discovery.columns]
    if not models:
        return {}, pd.DataFrame()

    target = pd.to_numeric(meta_discovery["actual_margin"], errors="coerce")
    err = pred_discovery[models].subtract(target, axis=0)
    sq = err.pow(2)
    mse = sq.mean(axis=0, skipna=True)
    n = sq.count(axis=0).astype(float)

    finite_mse = pd.to_numeric(mse, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    global_mse = float(finite_mse.median()) if len(finite_mse) else 225.0
    if not np.isfinite(global_mse) or global_mse <= 1e-8:
        global_mse = 225.0

    raw_mse = pd.to_numeric(mse, errors="coerce").fillna(global_mse).clip(lower=1e-6)
    n = pd.to_numeric(n, errors="coerce").fillna(0.0).clip(lower=0.0)
    shrink = n / (n + float(config.mse_history_shrink_k))
    stable_mse = shrink * raw_mse + (1.0 - shrink) * global_mse
    inv = 1.0 / stable_mse.clip(lower=1e-6)

    equal = pd.Series(1.0 / len(models), index=models, dtype=float)
    reliability = _cap_relative(
        inv, models,
        config.reliability_min_relative,
        config.reliability_max_relative,
    )
    s = float(np.clip(config.shrunk_equal_fraction, 0.0, 1.0))
    shrunk = (1.0 - s) * reliability + s * equal
    shrunk = _cap_relative(
        shrunk, models,
        config.shrunk_min_relative,
        config.shrunk_max_relative,
    )

    methods = {
        "equal": equal,
        "reliability": reliability,
        "shrunk_reliability": shrunk,
    }
    rows = []
    mae = err.abs().mean(axis=0, skipna=True)
    bias = err.mean(axis=0, skipna=True)
    for model in models:
        row = {
            "canonical_model_id": model,
            "discovery_n": int(n.get(model, 0.0)),
            "discovery_mae": float(mae.get(model, np.nan)),
            "discovery_rmse": float(math.sqrt(raw_mse.get(model, np.nan))) if np.isfinite(raw_mse.get(model, np.nan)) else np.nan,
            "discovery_bias": float(bias.get(model, np.nan)),
            "stable_mse": float(stable_mse.get(model, np.nan)),
        }
        for key, w in methods.items():
            row[f"weight_{key}"] = float(w.get(model, np.nan))
            row[f"relative_weight_{key}"] = float(w.get(model, np.nan) * len(models))
        rows.append(row)
    return methods, pd.DataFrame(rows)


def weighted_combo_forecast_arrays(
    pred: pd.DataFrame,
    model_ids: Iterable[str],
    base_weights: pd.Series,
    min_available_models: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return count, weighted mean, weighted sample SD, and effective N.

    Weights are renormalized across models actually available for each game.
    The variance denominator 1-sum(w^2) makes equal weights reduce exactly to
    the ordinary sample variance (ddof=1).
    """
    ids = [str(x) for x in model_ids if str(x) in pred.columns]
    n_games = len(pred)
    empty = (
        np.zeros(n_games, dtype=np.int16),
        np.full(n_games, np.nan),
        np.full(n_games, np.nan),
        np.full(n_games, np.nan),
    )
    if not ids:
        return empty
    arr = pred[ids].to_numpy(dtype=float)
    finite = np.isfinite(arr)
    count = finite.sum(axis=1).astype(np.int16)
    raw_w = pd.to_numeric(base_weights.reindex(ids), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_w = np.clip(raw_w, 0.0, np.inf)
    if raw_w.sum() <= 0:
        raw_w[:] = 1.0

    wmat = finite * raw_w.reshape(1, -1)
    wsum = wmat.sum(axis=1)
    normw = np.divide(wmat, wsum[:, None], out=np.zeros_like(wmat, dtype=float), where=wsum[:, None] > 0)
    safe_arr = np.where(finite, arr, 0.0)
    mean = np.sum(normw * safe_arr, axis=1)
    mean[wsum <= 0] = np.nan

    centered = np.where(finite, arr - mean[:, None], 0.0)
    numerator = np.sum(normw * np.square(centered), axis=1)
    sum_w2 = np.sum(np.square(normw), axis=1)
    denom = 1.0 - sum_w2
    var = np.divide(numerator, denom, out=np.full(n_games, np.nan), where=denom > 1e-12)
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    effective_n = np.divide(1.0, sum_w2, out=np.full(n_games, np.nan), where=sum_w2 > 1e-12)

    required = min(int(min_available_models), max(1, len(ids)))
    scorable = count >= required
    mean[~scorable] = np.nan
    sd[~scorable] = np.nan
    effective_n[~scorable] = np.nan
    return count, mean, sd, effective_n


def _forecast_accuracy(mean: np.ndarray, actual: np.ndarray) -> dict:
    ok = np.isfinite(mean) & np.isfinite(actual)
    if not ok.any():
        return {"scorable_games": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    err = mean[ok] - actual[ok]
    return {
        "scorable_games": int(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
    }


def _evaluate_combo_method(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    model_ids: list[str],
    weights: pd.Series,
    k: float,
    min_available_models: int,
    standard_price: int,
) -> tuple[dict, dict]:
    count, mean, sd, effective_n = weighted_combo_forecast_arrays(
        pred, model_ids, weights, min_available_models
    )
    market = pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(dtype=float)
    cover = actual - market
    edge = mean - market
    sig = _signal(edge, sd)
    stats = _threshold_stats(edge, sig, cover, float(k), standard_price=standard_price)
    acc = _forecast_accuracy(mean, actual)
    finite_eff = effective_n[np.isfinite(effective_n)]
    acc["mean_effective_n"] = float(np.mean(finite_eff)) if len(finite_eff) else np.nan
    acc["mean_available_n"] = float(np.mean(count[np.isfinite(mean)])) if np.isfinite(mean).any() else np.nan
    return {**stats, **acc}, {
        "mean": mean, "sd": sd, "signal": sig, "edge": edge,
        "count": count, "effective_n": effective_n,
    }


def _tune_combo_method(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    model_ids: list[str],
    weights: pd.Series,
    thresholds: Iterable[float],
    min_available_models: int,
    min_bets: int,
    standard_price: int,
) -> tuple[float, pd.DataFrame]:
    count, mean, sd, effective_n = weighted_combo_forecast_arrays(
        pred, model_ids, weights, min_available_models
    )
    market = pd.to_numeric(meta["market_margin"], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(dtype=float)
    edge = mean - market
    sig = _signal(edge, sd)
    cover = actual - market
    rows = []
    acc = _forecast_accuracy(mean, actual)
    for k in thresholds:
        rows.append({
            **_threshold_stats(edge, sig, cover, float(k), standard_price=standard_price),
            **acc,
        })
    profile = pd.DataFrame(rows)
    choice = _choose_stable_k(profile, min_bets=min_bets)
    selected = float(choice.get("selected_k", np.nan))
    if not np.isfinite(selected):
        selected = 0.50
    profile = choice.get("profile", profile)
    profile["selected"] = np.isclose(pd.to_numeric(profile["k"], errors="coerce"), selected)
    return selected, profile


def _meta_frame_from_forecasts(
    meta: pd.DataFrame,
    combo_forecasts: list[dict],
    *,
    min_meta_communities: int,
) -> pd.DataFrame:
    rows = []
    for gi, game_key in enumerate(meta.index):
        active = []
        for f in combo_forecasts:
            mu = f["mean"][gi]
            sd = f["sd"][gi]
            if np.isfinite(mu) and np.isfinite(sd):
                active.append((int(f["community"]), int(f["combo"]), float(mu), float(sd)))
        if not active:
            continue
        units = []
        for community in sorted({x[0] for x in active}):
            z = [x for x in active if x[0] == community]
            means = np.array([x[2] for x in z], dtype=float)
            sds = np.array([x[3] for x in z], dtype=float)
            cmean = float(np.mean(means))
            within = float(np.mean(np.square(sds)))
            between = float(np.var(means, ddof=1)) if len(means) >= 2 else 0.0
            units.append((community, cmean, within + between))
        unit_means = np.array([x[1] for x in units], dtype=float)
        unit_vars = np.array([x[2] for x in units], dtype=float)
        meta_mean = float(np.mean(unit_means))
        within = float(np.mean(unit_vars))
        between = float(np.var(unit_means, ddof=1)) if len(unit_means) >= 2 else 0.0
        total_var = max(0.0, within + between)
        meta_sd = float(math.sqrt(total_var))
        m = meta.loc[game_key]
        market = float(m["market_margin"]); actual = float(m["actual_margin"])
        edge = meta_mean - market
        signal = _signal(np.array([edge]), np.array([meta_sd]))[0]
        rows.append({
            "game_key": game_key,
            "season": int(m["season"]), "week": int(m["week"]),
            "market_margin": market, "actual_margin": actual,
            "cover": actual - market,
            "meta_mean": meta_mean, "meta_sd": meta_sd,
            "meta_edge": edge, "meta_signal": signal,
            "active_communities": len(units), "active_combos": len(active),
            "eligible_communities": len(units) >= int(min_meta_communities),
        })
    return pd.DataFrame(rows)


def _meta_profile(frame: pd.DataFrame, thresholds: Iterable[float], min_communities: int, standard_price: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    edge = pd.to_numeric(frame["meta_edge"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(frame["meta_signal"], errors="coerce").to_numpy(dtype=float)
    cover = pd.to_numeric(frame["cover"], errors="coerce").to_numpy(dtype=float)
    gate = pd.to_numeric(frame["active_communities"], errors="coerce").fillna(0).to_numpy(dtype=float) >= int(min_communities)
    actual = pd.to_numeric(frame["actual_margin"], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(frame["meta_mean"], errors="coerce").to_numpy(dtype=float)
    acc = _forecast_accuracy(mean, actual)
    return pd.DataFrame([
        {**_threshold_stats(edge, sig, cover, float(k), standard_price=standard_price, extra_gate=gate), **acc}
        for k in thresholds
    ])


def analyze_weighting_experiment(
    data: pd.DataFrame,
    combinations: list[dict],
    discovery_periods: Iterable[tuple[int, int]],
    holdout_periods: Iterable[tuple[int, int]],
    *,
    min_available_models: int = 4,
    thresholds: Iterable[float] = DEFAULT_K_GRID,
    combo_min_bets: int = 50,
    meta_min_bets: int = 30,
    min_meta_communities: int = 2,
    standard_price: int = -110,
    config: WeightingConfig = WeightingConfig(),
) -> dict:
    """Run frozen-finalist weighting experiment without changing production picks."""
    discovery_periods = tuple(discovery_periods)
    holdout_periods = tuple(holdout_periods)
    combos = [dict(c) for c in combinations if c.get("model_ids")]
    if not combos or not discovery_periods:
        return {}

    union = list(dict.fromkeys(mid for c in combos for mid in map(str, c.get("model_ids", []))))
    disc_pred, disc_meta = _matrix_and_meta(data, union, discovery_periods)
    hold_pred, hold_meta = _matrix_and_meta(data, union, holdout_periods) if holdout_periods else (pd.DataFrame(), pd.DataFrame())
    if disc_pred.empty:
        return {}

    fitted: dict[tuple[int, str], pd.Series] = {}
    weight_rows = []
    combo_rows = []
    profile_rows = []

    for i, combo in enumerate(combos, start=1):
        ids = [str(x) for x in combo.get("model_ids", []) if str(x) in disc_pred.columns]
        methods, diagnostics = fit_combo_weights(disc_pred, disc_meta, ids, config=config)
        if diagnostics.empty:
            continue
        diagnostics["portfolio_combo"] = i
        diagnostics["search_rank"] = int(combo.get("rank", i))
        diagnostics["community"] = int(combo.get("community", i))
        weight_rows.append(diagnostics)

        for method, weights in methods.items():
            fitted[(i, method)] = weights
            selected_k, profile = _tune_combo_method(
                disc_pred, disc_meta, ids, weights, thresholds,
                min_available_models, combo_min_bets, standard_price,
            )
            profile["portfolio_combo"] = i
            profile["search_rank"] = int(combo.get("rank", i))
            profile["community"] = int(combo.get("community", i))
            profile["weight_method"] = method
            profile["weight_method_label"] = WEIGHT_METHOD_LABELS[method]
            profile_rows.append(profile)

            disc_stats, _ = _evaluate_combo_method(
                disc_pred, disc_meta, ids, weights, selected_k,
                min_available_models, standard_price,
            )
            if not hold_pred.empty:
                hold_stats, _ = _evaluate_combo_method(
                    hold_pred, hold_meta, ids, weights, selected_k,
                    min_available_models, standard_price,
                )
            else:
                hold_stats = {k: np.nan for k in ["ats_pct", "roi", "wilson_low", "mae", "rmse", "bias", "mean_effective_n", "mean_available_n"]}
                hold_stats.update({"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "scorable_games": 0, "units": 0.0})

            combo_rows.append({
                "portfolio_combo": i,
                "search_rank": int(combo.get("rank", i)),
                "community": int(combo.get("community", i)),
                "combo_size": len(ids),
                "weight_method": method,
                "weight_method_label": WEIGHT_METHOD_LABELS[method],
                "selected_k": selected_k,
                **{f"discovery_{k}": v for k, v in disc_stats.items() if k != "k"},
                **{f"holdout_{k}": v for k, v in hold_stats.items() if k != "k"},
            })

    combo_summary = pd.DataFrame(combo_rows)
    if len(combo_summary):
        equal = combo_summary[combo_summary["weight_method"].eq("equal")][
            ["portfolio_combo", "holdout_mae", "holdout_rmse", "holdout_ats_pct", "holdout_roi"]
        ].rename(columns={
            "holdout_mae": "equal_holdout_mae", "holdout_rmse": "equal_holdout_rmse",
            "holdout_ats_pct": "equal_holdout_ats_pct", "holdout_roi": "equal_holdout_roi",
        })
        combo_summary = combo_summary.merge(equal, on="portfolio_combo", how="left")
        combo_summary["holdout_mae_delta_vs_equal"] = combo_summary["holdout_mae"] - combo_summary["equal_holdout_mae"]
        combo_summary["holdout_rmse_delta_vs_equal"] = combo_summary["holdout_rmse"] - combo_summary["equal_holdout_rmse"]
        combo_summary["holdout_ats_delta_vs_equal"] = combo_summary["holdout_ats_pct"] - combo_summary["equal_holdout_ats_pct"]
        combo_summary["holdout_roi_delta_vs_equal"] = combo_summary["holdout_roi"] - combo_summary["equal_holdout_roi"]

    # Diversified META under each internal weighting method. Community membership
    # remains fixed because it depends only on model membership, not outcomes.
    meta_rows = []
    meta_profiles = []
    meta_frames = {}
    for method in WEIGHT_METHOD_LABELS:
        disc_forecasts = []
        hold_forecasts = []
        for i, combo in enumerate(combos, start=1):
            ids = [str(x) for x in combo.get("model_ids", []) if str(x) in disc_pred.columns]
            weights = fitted.get((i, method))
            if weights is None:
                continue
            _, dmean, dsd, _ = weighted_combo_forecast_arrays(disc_pred, ids, weights, min_available_models)
            disc_forecasts.append({"combo": i, "community": int(combo.get("community", i)), "mean": dmean, "sd": dsd})
            if not hold_pred.empty:
                _, hmean, hsd, _ = weighted_combo_forecast_arrays(hold_pred, ids, weights, min_available_models)
                hold_forecasts.append({"combo": i, "community": int(combo.get("community", i)), "mean": hmean, "sd": hsd})

        dframe = _meta_frame_from_forecasts(disc_meta, disc_forecasts, min_meta_communities=min_meta_communities)
        hframe = _meta_frame_from_forecasts(hold_meta, hold_forecasts, min_meta_communities=min_meta_communities) if not hold_meta.empty else pd.DataFrame()
        meta_frames[(method, "discovery")] = dframe
        meta_frames[(method, "holdout")] = hframe
        prof = _meta_profile(dframe, thresholds, min_meta_communities, standard_price)
        if len(prof):
            choice = _choose_stable_k(prof, min_bets=meta_min_bets)
            mk = float(choice.get("selected_k", np.nan))
            prof = choice.get("profile", prof)
        else:
            mk = np.nan
        if not np.isfinite(mk):
            mk = 0.50
        if len(prof):
            prof["weight_method"] = method
            prof["weight_method_label"] = WEIGHT_METHOD_LABELS[method]
            prof["selected"] = np.isclose(pd.to_numeric(prof["k"], errors="coerce"), mk)
            meta_profiles.append(prof)

        for period_name, frame in (("Discovery", dframe), ("Holdout", hframe)):
            if frame is None or frame.empty:
                stats = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "ats_pct": np.nan, "units": 0.0, "roi": np.nan, "wilson_low": np.nan}
                acc = {"scorable_games": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
            else:
                edge = frame["meta_edge"].to_numpy(dtype=float)
                sig = frame["meta_signal"].to_numpy(dtype=float)
                cover = frame["cover"].to_numpy(dtype=float)
                gate = frame["active_communities"].to_numpy(dtype=float) >= int(min_meta_communities)
                stats = _threshold_stats(edge, sig, cover, mk, standard_price=standard_price, extra_gate=gate)
                acc = _forecast_accuracy(frame["meta_mean"].to_numpy(dtype=float), frame["actual_margin"].to_numpy(dtype=float))
            meta_rows.append({
                "weight_method": method,
                "weight_method_label": WEIGHT_METHOD_LABELS[method],
                "period": period_name,
                "selected_k": mk,
                **stats,
                **acc,
            })

    meta_summary = pd.DataFrame(meta_rows)
    if len(meta_summary):
        eq_hold = meta_summary[(meta_summary["weight_method"].eq("equal")) & (meta_summary["period"].eq("Holdout"))]
        if len(eq_hold):
            eq = eq_hold.iloc[0]
            for metric in ["mae", "rmse", "ats_pct", "roi"]:
                meta_summary[f"delta_{metric}_vs_equal_holdout"] = np.where(
                    meta_summary["period"].eq("Holdout"),
                    pd.to_numeric(meta_summary[metric], errors="coerce") - float(eq.get(metric, np.nan)),
                    np.nan,
                )

    # Compact method-level diagnostics across finalist combos. Lower error delta is better.
    method_rows = []
    if len(combo_summary):
        for method, g in combo_summary.groupby("weight_method", sort=False):
            h_rmse = pd.to_numeric(g["holdout_rmse"], errors="coerce")
            d_rmse = pd.to_numeric(g["holdout_rmse_delta_vs_equal"], errors="coerce")
            d_mae = pd.to_numeric(g["holdout_mae_delta_vs_equal"], errors="coerce")
            method_rows.append({
                "weight_method": method,
                "weight_method_label": WEIGHT_METHOD_LABELS.get(method, method),
                "combos_evaluated": int(h_rmse.notna().sum()),
                "median_holdout_rmse": float(h_rmse.median()) if h_rmse.notna().any() else np.nan,
                "median_holdout_rmse_delta_vs_equal": float(d_rmse.median()) if d_rmse.notna().any() else np.nan,
                "median_holdout_mae_delta_vs_equal": float(d_mae.median()) if d_mae.notna().any() else np.nan,
                "combos_rmse_improved": int((d_rmse < 0).sum()),
                "combos_mae_improved": int((d_mae < 0).sum()),
            })

    return {
        "config": config,
        "combo_summary": combo_summary,
        "combo_k_profiles": pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame(),
        "model_weights": pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(),
        "method_summary": pd.DataFrame(method_rows),
        "meta_summary": meta_summary,
        "meta_k_profiles": pd.concat(meta_profiles, ignore_index=True) if meta_profiles else pd.DataFrame(),
        "meta_frames": meta_frames,
        "production_method": "equal",
        "experimental_only": True,
    }
