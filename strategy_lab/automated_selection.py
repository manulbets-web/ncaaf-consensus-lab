from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine import (
    BacktestConfig,
    METHOD_LABELS,
    _adjust_predictions,
    _cluster_assignments,
    _effective_n,
    _edge_correlation,
    _fit_market_lambda,
    _fit_ridge,
    _inverse_mse_weights,
    _one_method_prediction,
    _regularized_error_covariance,
)


R_MAD_CONSTANT = 1.4826
AUTO_K_GRID = (0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.10)
AVAILABILITY_GRID = (3, 4, 5)

SELECTOR_LABELS = {
    "accuracy": "Accuracy: low historical MSE",
    "market_value": "Market value: improves on market MAE",
    "edge_skill": "Edge skill: market-error slope + ATS",
    "stable_edge_skill": "Stable edge skill: persistent market-relative skill",
    "top20_quality_baseline": "Baseline: Top 20 old quality score",
}

WEEKLY_RULE_LABELS = {
    "frozen": "Frozen preseason pool",
    "top_half": "Weekly top half",
    "top_third": "Weekly top third",
}


@dataclass
class AutomatedSelectionConfig:
    # Qualification and coverage are intentionally separate. A season-level
    # candidate pool can be six-plus models while a game can still be evaluated
    # with whichever 3/4/5 members are available.
    min_preseason_model_history: int = 100
    min_weekly_model_history: int = 100
    min_season_model_games: int = 50
    min_qualified_pool: int = 6
    auto_pool_floor: int = 6
    auto_pool_cap: int = 20

    primary_min_available: int = 4
    availability_grid: tuple = AVAILABILITY_GRID

    beta_alpha: float = 250.0
    beta_beta: float = 250.0
    ats_exponent: float = 4.0
    slope_shrink_k: float = 200.0
    value_shrink_k: float = 200.0

    redundancy_threshold: float = 0.85
    redundancy_min_shared: int = 100

    selector_rules: tuple = (
        "accuracy",
        "market_value",
        "edge_skill",
        "stable_edge_skill",
    )
    weekly_rules: tuple = ("frozen",)
    k_grid: tuple = AUTO_K_GRID

    minimum_prior_seasons: int = 1
    stable_min_prior_seasons: int = 2

    # ATS is calculated at several edge sizes. The 2-point threshold is the
    # primary ingredient in the edge-skill selector; the 1- and 3-point rows
    # remain visible for audit/sensitivity.
    edge_ats_thresholds: tuple = (1.0, 2.0, 3.0)
    primary_edge_ats_threshold: float = 2.0


def _unit_result(result: str, price: int = -110) -> float:
    if result == "win":
        if price < 0:
            return 100.0 / abs(float(price))
        return float(price) / 100.0
    if result == "loss":
        return -1.0
    return 0.0


def _grade(edge: float, cover: float, price: int) -> tuple[str, float]:
    if not np.isfinite(edge) or abs(edge) < 1e-12:
        return "no_bet", 0.0
    if not np.isfinite(cover):
        return "no_bet", 0.0
    if abs(cover) < 1e-12:
        return "push", 0.0
    result = "win" if edge * cover > 0 else "loss"
    return result, _unit_result(result, price)


def _r_mad(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    med = float(np.median(x))
    return float(R_MAD_CONSTANT * np.median(np.abs(x - med)))


def _matrix_and_meta(
    data: pd.DataFrame,
    model_ids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = list(dict.fromkeys(map(str, model_ids)))
    d = data[data["canonical_model_id"].astype(str).isin(models)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    pred = (
        d.pivot_table(
            index="game_key",
            columns="canonical_model_id",
            values="prediction_margin",
            aggfunc="first",
        )
        .reindex(columns=models)
    )

    meta_cols = [
        "game_key", "season", "week", "market_margin", "actual_margin",
        "period_key", "team_a_id", "team_b_id", "road", "home",
        "market_reference_source", "market_snapshot_label",
    ]
    meta = (
        d[[c for c in meta_cols if c in d.columns]]
        .drop_duplicates("game_key")
        .set_index("game_key")
        .reindex(pred.index)
    )
    if "period_key" not in meta.columns:
        meta["period_key"] = list(
            zip(
                pd.to_numeric(meta["season"], errors="coerce"),
                pd.to_numeric(meta["week"], errors="coerce"),
            )
        )

    valid = (
        pd.to_numeric(meta["season"], errors="coerce").notna()
        & pd.to_numeric(meta["week"], errors="coerce").notna()
        & pd.to_numeric(meta["market_margin"], errors="coerce").notna()
        & pd.to_numeric(meta["actual_margin"], errors="coerce").notna()
    )
    pred = pred.loc[valid]
    meta = meta.loc[valid].copy()
    meta["season"] = meta["season"].astype(int)
    meta["week"] = meta["week"].astype(int)

    order = (
        meta.reset_index()
        .sort_values(["season", "week", "game_key"])["game_key"]
        .tolist()
    )
    return pred.reindex(order), meta.reindex(order)


def _ats_stats(
    edge: pd.Series,
    cover: pd.Series,
    threshold: float,
    alpha: float,
    beta: float,
) -> dict:
    edge = pd.to_numeric(edge, errors="coerce")
    cover = pd.to_numeric(cover, errors="coerce")
    decision = (
        edge.notna()
        & cover.notna()
        & edge.abs().ge(float(threshold))
        & cover.ne(0)
    )
    wins = int(((edge * cover > 0) & decision).sum())
    losses = int(((edge * cover < 0) & decision).sum())
    n = wins + losses
    raw = wins / n if n else np.nan
    denom = n + float(alpha) + float(beta)
    posterior = (
        (wins + float(alpha)) / denom
        if denom > 0 else np.nan
    )
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "pct": raw,
        "posterior": posterior,
    }


def _edge_slope(edge: pd.Series, cover: pd.Series) -> tuple[float, float, int]:
    x = pd.to_numeric(edge, errors="coerce")
    y = pd.to_numeric(cover, errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok]
    y = y[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    vx = float(x.var(ddof=1))
    if not np.isfinite(vx) or vx <= 1e-12:
        return np.nan, np.nan, n
    cov = float(np.cov(x.to_numpy(dtype=float), y.to_numpy(dtype=float), ddof=1)[0, 1])
    slope = cov / vx
    corr = float(x.corr(y)) if n >= 3 else np.nan
    return float(slope), corr, n


def _model_selection_metrics(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[str],
    config: AutomatedSelectionConfig,
    *,
    min_history: int,
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()

    actual = pd.to_numeric(meta["actual_margin"], errors="coerce")
    market = pd.to_numeric(meta["market_margin"], errors="coerce")
    cover = actual - market
    seasons = sorted(meta["season"].unique().tolist())

    rows = []
    for model in models:
        p = pd.to_numeric(pred[model], errors="coerce")
        ok = p.notna() & actual.notna() & market.notna()
        n = int(ok.sum())

        if n:
            model_err = p[ok] - actual[ok]
            market_err = market[ok] - actual[ok]
            model_mse = float(np.mean(np.square(model_err)))
            model_mae = float(np.mean(np.abs(model_err)))
            market_mse = float(np.mean(np.square(market_err)))
            market_mae = float(np.mean(np.abs(market_err)))
            delta_mae = market_mae - model_mae
            delta_mse = market_mse - model_mse
            bias = float(model_err.mean())
        else:
            model_mse = model_mae = market_mse = market_mae = np.nan
            delta_mae = delta_mse = bias = np.nan

        edge = p - market
        slope, edge_corr, slope_n = _edge_slope(edge, cover)
        slope_shrink = slope_n / (slope_n + float(config.slope_shrink_k)) if slope_n else 0.0
        slope_shrunk = slope * slope_shrink if np.isfinite(slope) else np.nan

        value_shrink = n / (n + float(config.value_shrink_k)) if n else 0.0
        delta_mae_shrunk = delta_mae * value_shrink if np.isfinite(delta_mae) else np.nan
        delta_mse_shrunk = delta_mse * value_shrink if np.isfinite(delta_mse) else np.nan

        ats_values = {}
        for threshold in config.edge_ats_thresholds:
            stats = _ats_stats(
                edge,
                cover,
                float(threshold),
                float(config.beta_alpha),
                float(config.beta_beta),
            )
            suffix = str(float(threshold)).rstrip("0").rstrip(".").replace(".", "p")
            ats_values[f"ats{suffix}_n"] = stats["n"]
            ats_values[f"ats{suffix}_wins"] = stats["wins"]
            ats_values[f"ats{suffix}_losses"] = stats["losses"]
            ats_values[f"ats{suffix}_pct"] = stats["pct"]
            ats_values[f"ats{suffix}_posterior"] = stats["posterior"]

        # Season-level persistence. A season counts only when the model has a
        # reasonable number of observations in that season.
        season_rows = []
        for season in seasons:
            keys = meta.index[meta["season"].eq(season)]
            if len(keys) == 0:
                continue
            ps = p.reindex(keys)
            as_ = actual.reindex(keys)
            ms = market.reindex(keys)
            os = ps.notna() & as_.notna() & ms.notna()
            season_n = int(os.sum())
            if season_n < int(config.min_season_model_games):
                continue

            e_s = ps - ms
            c_s = as_ - ms
            slope_s, _, _ = _edge_slope(e_s, c_s)
            mae_s = float(np.mean(np.abs(ps[os] - as_[os])))
            market_mae_s = float(np.mean(np.abs(ms[os] - as_[os])))
            delta_s = market_mae_s - mae_s
            ats2 = _ats_stats(
                e_s,
                c_s,
                float(config.primary_edge_ats_threshold),
                0.0,
                0.0,
            )
            season_rows.append({
                "season": int(season),
                "n": season_n,
                "edge_slope": slope_s,
                "delta_mae": delta_s,
                "ats2_pct": ats2["pct"],
                "ats2_n": ats2["n"],
            })

        season_df = pd.DataFrame(season_rows)
        prior_seasons = int(len(season_df))
        if prior_seasons:
            positive_slope_seasons = int((season_df["edge_slope"] > 0).sum())
            positive_delta_seasons = int((season_df["delta_mae"] > 0).sum())
            ats_above_50_seasons = int((season_df["ats2_pct"] > 0.5).sum())
            slope_fraction = positive_slope_seasons / prior_seasons
            delta_fraction = positive_delta_seasons / prior_seasons
            ats_fraction = ats_above_50_seasons / prior_seasons
            stability_fraction = float(np.mean([slope_fraction, delta_fraction, ats_fraction]))
            worst_slope = float(season_df["edge_slope"].min())
            median_slope = float(season_df["edge_slope"].median())
            worst_delta = float(season_df["delta_mae"].min())
        else:
            positive_slope_seasons = positive_delta_seasons = ats_above_50_seasons = 0
            slope_fraction = delta_fraction = ats_fraction = stability_fraction = np.nan
            worst_slope = median_slope = worst_delta = np.nan

        primary_suffix = str(float(config.primary_edge_ats_threshold)).rstrip("0").rstrip(".").replace(".", "p")
        primary_post = ats_values.get(f"ats{primary_suffix}_posterior", np.nan)

        rows.append({
            "canonical_model_id": str(model),
            "history_n": n,
            "model_mse": model_mse,
            "model_mae": model_mae,
            "market_mse_matched": market_mse,
            "market_mae_matched": market_mae,
            "delta_mse": delta_mse,
            "delta_mae": delta_mae,
            "delta_mse_shrunk": delta_mse_shrunk,
            "delta_mae_shrunk": delta_mae_shrunk,
            "bias": bias,
            "edge_slope": slope,
            "edge_slope_shrunk": slope_shrunk,
            "edge_correlation": edge_corr,
            "edge_slope_n": slope_n,
            "prior_seasons_with_data": prior_seasons,
            "positive_edge_slope_seasons": positive_slope_seasons,
            "positive_delta_mae_seasons": positive_delta_seasons,
            "ats2_above_50_seasons": ats_above_50_seasons,
            "positive_edge_slope_fraction": slope_fraction,
            "positive_delta_mae_fraction": delta_fraction,
            "ats2_above_50_fraction": ats_fraction,
            "stability_fraction": stability_fraction,
            "worst_season_edge_slope": worst_slope,
            "median_season_edge_slope": median_slope,
            "worst_season_delta_mae": worst_delta,
            "eligible_history": bool(n >= int(min_history)),
            **ats_values,
        })

    out = pd.DataFrame(rows)
    eligible = out["eligible_history"] & out["model_mse"].notna() & out["model_mse"].gt(0)
    med_mse = float(out.loc[eligible, "model_mse"].median()) if eligible.any() else np.nan
    out["median_eligible_mse"] = med_mse
    out["accuracy_score"] = np.where(
        eligible & np.isfinite(med_mse) & (med_mse > 0),
        np.sqrt(med_mse / out["model_mse"]),
        np.nan,
    )

    primary_suffix = str(float(config.primary_edge_ats_threshold)).rstrip("0").rstrip(".").replace(".", "p")
    primary_post_col = f"ats{primary_suffix}_posterior"
    primary_post = pd.to_numeric(out[primary_post_col], errors="coerce")
    out["edge_skill_score"] = (
        out["edge_slope_shrunk"]
        * np.exp(float(config.ats_exponent) * (primary_post - 0.5))
    )
    out["stable_edge_skill_score"] = (
        out["edge_skill_score"]
        * pd.to_numeric(out["stability_fraction"], errors="coerce")
    )

    # Preserve the v2.0 quality score as a comparator only.
    out["old_quality_score"] = (
        out["accuracy_score"]
        * np.exp(float(config.ats_exponent) * (primary_post - 0.5))
    )
    return out


def _selector_definition(
    metrics: pd.DataFrame,
    selector: str,
    config: AutomatedSelectionConfig,
) -> tuple[pd.Series, str]:
    eligible = metrics["eligible_history"].fillna(False)
    if selector == "accuracy":
        passing = (
            eligible
            & metrics["model_mse"].le(metrics["median_eligible_mse"])
        )
        return passing, "accuracy_score"

    if selector == "market_value":
        passing = eligible & metrics["delta_mae_shrunk"].gt(0)
        return passing, "delta_mae_shrunk"

    if selector == "edge_skill":
        post = metrics["ats2_posterior"] if "ats2_posterior" in metrics.columns else np.nan
        passing = (
            eligible
            & metrics["edge_slope_shrunk"].gt(0)
            & pd.to_numeric(post, errors="coerce").gt(0.5)
        )
        return passing, "edge_skill_score"

    if selector == "stable_edge_skill":
        post = metrics["ats2_posterior"] if "ats2_posterior" in metrics.columns else np.nan
        passing = (
            eligible
            & metrics["prior_seasons_with_data"].ge(int(config.stable_min_prior_seasons))
            & metrics["edge_slope_shrunk"].gt(0)
            & pd.to_numeric(post, errors="coerce").gt(0.5)
            & metrics["stability_fraction"].ge(2.0 / 3.0)
        )
        return passing, "stable_edge_skill_score"

    if selector == "top20_quality_baseline":
        passing = eligible.copy()
        return passing, "old_quality_score"

    raise ValueError(f"Unknown selector: {selector}")


def _prior_edge_correlation(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    model_ids: list[str],
    min_shared: int,
) -> pd.DataFrame:
    if not model_ids:
        return pd.DataFrame()
    edge = pred.reindex(columns=model_ids).subtract(meta["market_margin"], axis=0)
    corr = edge.corr(min_periods=int(min_shared)).fillna(0.0).clip(-1.0, 1.0)
    corr = corr.reindex(index=model_ids, columns=model_ids).fillna(0.0)
    arr = corr.to_numpy(dtype=float, copy=True)
    if len(arr):
        np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=model_ids, columns=model_ids)


def _greedy_diversity_select(
    metrics: pd.DataFrame,
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    selector: str,
    config: AutomatedSelectionConfig,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Select only models that actually pass the philosophy's criterion.

    A crucial v2.1 rule: the pool floor NEVER promotes criterion-failing models.
    If fewer than `min_qualified_pool` models pass, that selector/season has no
    valid pool. The only floor-fill allowed is to re-admit a *qualifying* model
    that was initially skipped for redundancy.
    """
    passing, score_col = _selector_definition(metrics, selector, config)
    ranked = metrics[
        metrics["eligible_history"].fillna(False)
        & pd.to_numeric(metrics[score_col], errors="coerce").notna()
    ].copy()
    ranked["criterion_pass"] = passing.reindex(ranked.index).fillna(False)
    ranked = ranked.sort_values(
        ["criterion_pass", score_col, "history_n"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    if ranked.empty:
        return [], pd.DataFrame(), {
            "criterion_pass_n": 0,
            "eligible_models": 0,
            "floor_applied": False,
            "cap_applied": False,
            "redundancy_skipped": 0,
            "selected_n": 0,
            "score_col": score_col,
            "pool_status": "no eligible models",
        }

    eligible_ids = ranked["canonical_model_id"].astype(str).tolist()
    corr = _prior_edge_correlation(
        pred,
        meta,
        eligible_ids,
        int(config.redundancy_min_shared),
    )

    pass_ranked = ranked[ranked["criterion_pass"]].copy()
    criterion_pass_n = int(len(pass_ranked))
    hard_min = int(config.min_qualified_pool)
    desired_floor = min(
        max(hard_min, int(config.auto_pool_floor)),
        criterion_pass_n,
    ) if criterion_pass_n else 0
    cap_n = min(int(config.auto_pool_cap), criterion_pass_n)

    def max_positive_corr(model_id: str, chosen: list[str]) -> tuple[float, str]:
        if not chosen or corr.empty:
            return 0.0, ""
        vals = corr.loc[model_id, chosen]
        if vals.empty:
            return 0.0, ""
        blocker = str(vals.idxmax())
        return float(vals.max()), blocker

    selected: list[str] = []
    blocked_qualifying: list[tuple] = []

    # Only criterion-passing models participate in pool construction.
    for rank_idx, row in pass_ranked.iterrows():
        model_id = str(row["canonical_model_id"])
        max_corr, blocker = max_positive_corr(model_id, selected)
        if selected and max_corr >= float(config.redundancy_threshold):
            blocked_qualifying.append((rank_idx, model_id, max_corr, blocker))
            continue
        selected.append(model_id)
        if len(selected) >= cap_n:
            break

    # If redundancy alone drove a sufficiently large qualifying set below the
    # desired floor, re-admit the least-redundant QUALIFYING models only.
    floor_applied = False
    if criterion_pass_n >= hard_min and len(selected) < desired_floor:
        floor_applied = True
        remaining = []
        for rank_idx, row in pass_ranked.iterrows():
            model_id = str(row["canonical_model_id"])
            if model_id in selected:
                continue
            max_corr, blocker = max_positive_corr(model_id, selected)
            remaining.append((max_corr, rank_idx, model_id, blocker))
        remaining.sort(key=lambda x: (x[0], x[1]))
        for max_corr, rank_idx, model_id, blocker in remaining:
            selected.append(model_id)
            if len(selected) >= desired_floor:
                break

    # Hard rule: do not manufacture a "good-model" pool from failing models.
    if criterion_pass_n < hard_min:
        selected = []
        pool_status = f"insufficient qualifying models ({criterion_pass_n} < {hard_min})"
    elif len(selected) < hard_min:
        # This should be rare because the redundancy floor re-admits qualifying
        # models, but keep the guard explicit.
        selected = []
        pool_status = "insufficient pool after redundancy"
    else:
        pool_status = "valid"

    selected_set = set(selected)
    audit = ranked.copy()
    audit["selected"] = audit["canonical_model_id"].astype(str).isin(selected_set)
    order_map = {m: i + 1 for i, m in enumerate(selected)}
    audit["selection_order"] = audit["canonical_model_id"].astype(str).map(order_map)
    audit["score_col"] = score_col
    audit["selector_score"] = pd.to_numeric(audit[score_col], errors="coerce")
    audit["max_corr_to_selected_pool"] = np.nan
    audit["redundancy_blocker"] = ""
    audit["redundancy_blocked"] = False
    for i, row in audit.iterrows():
        mid = str(row["canonical_model_id"])
        others = [m for m in selected if m != mid]
        max_corr, blocker = max_positive_corr(mid, others)
        audit.at[i, "max_corr_to_selected_pool"] = max_corr
        audit.at[i, "redundancy_blocker"] = blocker
        audit.at[i, "redundancy_blocked"] = (
            (not bool(row["selected"]))
            and bool(row["criterion_pass"])
            and max_corr >= float(config.redundancy_threshold)
        )

    cap_applied = criterion_pass_n > int(config.auto_pool_cap)
    diag = {
        "criterion_pass_n": criterion_pass_n,
        "eligible_models": int(len(ranked)),
        "floor_applied": bool(floor_applied),
        "cap_applied": bool(cap_applied),
        "redundancy_skipped": int(audit["redundancy_blocked"].sum()),
        "selected_n": int(len(selected)),
        "score_col": score_col,
        "pool_status": pool_status,
    }
    return selected, audit, diag


def _weekly_rank_select(
    metrics: pd.DataFrame,
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    frozen_ids: list[str],
    selector: str,
    target_n: int,
    config: AutomatedSelectionConfig,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Rank the already-qualified frozen pool without reapplying a hard pass rule."""
    frozen_set = set(map(str, frozen_ids))
    m = metrics[
        metrics["canonical_model_id"].astype(str).isin(frozen_set)
        & metrics["eligible_history"].fillna(False)
    ].copy()
    if m.empty:
        return [], pd.DataFrame(), {
            "weekly_eligible_models": 0,
            "weekly_pool_n": 0,
            "weekly_floor_applied": False,
            "weekly_redundancy_skipped": 0,
        }

    _, score_col = _selector_definition(m, selector, config)
    m = m[pd.to_numeric(m[score_col], errors="coerce").notna()].copy()
    m = m.sort_values(
        [score_col, "history_n"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    if m.empty:
        return [], pd.DataFrame(), {
            "weekly_eligible_models": 0,
            "weekly_pool_n": 0,
            "weekly_floor_applied": False,
            "weekly_redundancy_skipped": 0,
        }

    target_n = min(max(int(config.min_qualified_pool), int(target_n)), len(m))
    ids = m["canonical_model_id"].astype(str).tolist()
    corr = _prior_edge_correlation(
        pred.reindex(columns=ids),
        meta,
        ids,
        int(config.redundancy_min_shared),
    )

    selected = []
    blocked = []
    for rank_idx, row in m.iterrows():
        mid = str(row["canonical_model_id"])
        if not selected:
            max_corr, blocker = 0.0, ""
        else:
            vals = corr.loc[mid, selected]
            max_corr = float(vals.max()) if len(vals) else 0.0
            blocker = str(vals.idxmax()) if len(vals) else ""
        if selected and max_corr >= float(config.redundancy_threshold):
            blocked.append((max_corr, rank_idx, mid, blocker))
            continue
        selected.append(mid)
        if len(selected) >= target_n:
            break

    floor_applied = len(selected) < target_n
    if floor_applied:
        # All candidates are already members of the preseason-qualified pool,
        # so it is acceptable to re-admit redundant members to hit the weekly
        # target. Choose the least redundant first.
        remaining = []
        for rank_idx, row in m.iterrows():
            mid = str(row["canonical_model_id"])
            if mid in selected:
                continue
            if selected:
                vals = corr.loc[mid, selected]
                max_corr = float(vals.max()) if len(vals) else 0.0
                blocker = str(vals.idxmax()) if len(vals) else ""
            else:
                max_corr, blocker = 0.0, ""
            remaining.append((max_corr, rank_idx, mid, blocker))
        remaining.sort(key=lambda x: (x[0], x[1]))
        for max_corr, rank_idx, mid, blocker in remaining:
            selected.append(mid)
            if len(selected) >= target_n:
                break

    selected_set = set(selected)
    audit = m.copy()
    audit["criterion_pass"] = True
    audit["selected"] = audit["canonical_model_id"].astype(str).isin(selected_set)
    order_map = {mid: i + 1 for i, mid in enumerate(selected)}
    audit["selection_order"] = audit["canonical_model_id"].astype(str).map(order_map)
    audit["score_col"] = score_col
    audit["selector_score"] = pd.to_numeric(audit[score_col], errors="coerce")
    audit["max_corr_to_selected_pool"] = np.nan
    audit["redundancy_blocker"] = ""
    audit["redundancy_blocked"] = False
    for i, row in audit.iterrows():
        mid = str(row["canonical_model_id"])
        others = [x for x in selected if x != mid]
        if others:
            vals = corr.loc[mid, others]
            max_corr = float(vals.max()) if len(vals) else 0.0
            blocker = str(vals.idxmax()) if len(vals) else ""
        else:
            max_corr, blocker = 0.0, ""
        audit.at[i, "max_corr_to_selected_pool"] = max_corr
        audit.at[i, "redundancy_blocker"] = blocker
        audit.at[i, "redundancy_blocked"] = (
            (not bool(row["selected"]))
            and max_corr >= float(config.redundancy_threshold)
        )

    return selected, audit, {
        "weekly_eligible_models": int(len(m)),
        "weekly_pool_n": int(len(selected)),
        "weekly_floor_applied": bool(floor_applied),
        "weekly_redundancy_skipped": int(audit["redundancy_blocked"].sum()),
    }

def _weekly_refine(
    metrics: pd.DataFrame,
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    frozen_ids: list[str],
    selector: str,
    weekly_rule: str,
    config: AutomatedSelectionConfig,
) -> tuple[list[str], pd.DataFrame, dict]:
    if weekly_rule == "frozen":
        audit = metrics[
            metrics["canonical_model_id"].astype(str).isin(set(frozen_ids))
        ].copy()
        audit["selected"] = True
        audit["criterion_pass"] = True
        audit["selection_order"] = range(1, len(audit) + 1)
        audit["selector_score"] = np.nan
        audit["score_col"] = "frozen"
        audit["max_corr_to_selected_pool"] = np.nan
        audit["redundancy_blocker"] = ""
        audit["redundancy_blocked"] = False
        return list(frozen_ids), audit, {
            "weekly_eligible_models": len(frozen_ids),
            "weekly_pool_n": len(frozen_ids),
            "weekly_floor_applied": False,
            "weekly_redundancy_skipped": 0,
        }

    if weekly_rule == "top_half":
        target = max(int(config.min_qualified_pool), int(math.ceil(len(frozen_ids) * 0.50)))
    elif weekly_rule == "top_third":
        target = max(int(config.min_qualified_pool), int(math.ceil(len(frozen_ids) / 3.0)))
    else:
        raise ValueError(f"Unknown weekly rule: {weekly_rule}")

    return _weekly_rank_select(
        metrics,
        pred,
        meta,
        frozen_ids,
        selector,
        target,
        config,
    )

def _bias_adjustment(
    train_pred_raw: pd.DataFrame,
    train_meta: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
) -> pd.Series:
    if not config.bias_correction:
        return pd.Series(0.0, index=models, dtype=float)

    err = train_pred_raw.subtract(train_meta["actual_margin"], axis=0)
    means = err.mean(axis=0, skipna=True).reindex(models).fillna(0.0)
    counts = err.count(axis=0).reindex(models).fillna(0.0)
    shrink = counts / (counts + float(config.bias_shrink_k))
    return means * shrink


def _fit_state(
    train_pred_raw: pd.DataFrame,
    train_meta: pd.DataFrame,
    models: list[str],
    methods: list[str],
    config: BacktestConfig,
) -> tuple[pd.Series, dict]:
    bias = _bias_adjustment(train_pred_raw, train_meta, models, config)
    train_adj = _adjust_predictions(train_pred_raw, bias)

    needs_inverse = "inverse_mse" in methods
    needs_cluster = "cluster_balanced" in methods
    needs_lambda = "market_shrinkage" in methods
    needs_ridge = "market_ridge" in methods
    needs_bayes = "bayesian_market" in methods

    corr = _edge_correlation(
        train_adj, train_meta, config.cluster_min_shared
    ).reindex(index=models, columns=models).fillna(0.0)
    arr = corr.to_numpy(dtype=float, copy=True)
    if len(arr):
        np.fill_diagonal(arr, 1.0)
    corr = pd.DataFrame(arr, index=models, columns=models)

    if needs_inverse:
        inverse_w = _inverse_mse_weights(
            train_adj, train_meta, models, config
        )
    else:
        inverse_w = pd.Series(
            1.0 / max(len(models), 1), index=models, dtype=float
        )

    if needs_cluster:
        clusters, _ = _cluster_assignments(
            train_adj, train_meta, models, config
        )
    else:
        clusters = {m: i + 1 for i, m in enumerate(models)}

    market_lambda = (
        _fit_market_lambda(train_adj, train_meta, config)
        if needs_lambda else np.nan
    )

    if needs_ridge:
        ridge_fit, ridge_alpha = _fit_ridge(
            train_adj, train_meta, models, config
        )
    else:
        ridge_fit, ridge_alpha = None, np.nan

    if needs_bayes:
        bayes_cov = _regularized_error_covariance(
            train_adj, train_meta, models, config
        )
        market_error = (
            train_meta["actual_margin"] - train_meta["market_margin"]
        )
        tau2 = float(market_error.var(ddof=1))
        if not np.isfinite(tau2) or tau2 <= 1e-6:
            tau2 = 225.0
    else:
        bayes_cov, tau2 = None, np.nan

    state = {
        "inverse_mse_weights": inverse_w,
        "cluster_assignments": clusters,
        "corr": corr,
        "market_lambda": market_lambda,
        "ridge_fit": ridge_fit,
        "ridge_alpha": ridge_alpha,
        "bayes_cov": bayes_cov,
        "tau2": tau2,
        "bayes_cache": {},
    }
    return bias, state


def _bayesian_prediction_cached(
    row: pd.Series,
    market: float,
    state: dict,
) -> tuple[float, dict]:
    cov = state.get("bayes_cov")
    tau2 = float(state.get("tau2", np.nan))
    available = list(row.dropna().index.astype(str))
    if cov is None or len(available) == 0:
        return np.nan, {}
    if not np.isfinite(tau2) or tau2 <= 0:
        return np.nan, {}

    key = tuple(available)
    cached = state["bayes_cache"].get(key)
    if cached is None:
        sub = cov.loc[available, available].to_numpy(dtype=float)
        ones = np.ones(len(available))
        inv = np.linalg.pinv(sub, hermitian=True)
        precision = 1.0 / tau2 + float(ones @ inv @ ones)
        post_var = 1.0 / precision
        cached = (inv, ones, post_var)
        state["bayes_cache"][key] = cached

    inv, ones, post_var = cached
    d = (row.loc[available] - market).to_numpy(dtype=float)
    post_edge = post_var * float(ones @ inv @ d)
    posterior_sd = float(math.sqrt(max(post_var, 0.0)))
    extra = {"posterior_sd": posterior_sd}
    if posterior_sd > 0:
        extra["home_cover_probability"] = float(
            norm.cdf(post_edge / posterior_sd)
        )
    return float(market + post_edge), extra


def _method_prediction(
    method: str,
    row: pd.Series,
    market: float,
    models: list[str],
    config: BacktestConfig,
    state: dict,
) -> tuple[float, dict]:
    if method == "bayesian_market":
        return _bayesian_prediction_cached(row, market, state)
    return _one_method_prediction(
        method, row, market, models, config, state
    )


def _summary_stats(g: pd.DataFrame) -> dict:
    if g.empty:
        return {
            "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "ats_pct": np.nan, "units": 0.0, "roi": np.nan,
            "mean_preseason_pool": np.nan,
            "mean_weekly_pool": np.nan,
            "mean_available_models": np.nan,
            "mean_effective_n": np.nan,
            "mean_abs_edge": np.nan,
            "consensus_mae": np.nan,
            "market_mae": np.nan,
        }

    wins = int((g["ats_result"] == "win").sum())
    losses = int((g["ats_result"] == "loss").sum())
    pushes = int((g["ats_result"] == "push").sum())
    bets = wins + losses
    units = float(g["unit_result"].sum())
    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": wins / bets if bets else np.nan,
        "units": units,
        "roi": units / bets if bets else np.nan,
        "mean_preseason_pool": float(g["preseason_pool_n"].mean()),
        "mean_weekly_pool": float(g["weekly_pool_n"].mean()),
        "mean_available_models": float(g["available_models"].mean()),
        "mean_effective_n": float(g["effective_n"].mean()),
        "mean_abs_edge": float(g["absolute_edge"].mean()),
        "consensus_mae": float(
            np.mean(np.abs(g["consensus_margin"] - g["actual_margin"]))
        ),
        "market_mae": float(
            np.mean(np.abs(g["market_margin"] - g["actual_margin"]))
        ),
    }


def _threshold_summary(
    detail: pd.DataFrame,
    k_grid: tuple,
    availability_grid: tuple,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "selector", "selector_name",
        "weekly_rule", "weekly_rule_name",
        "method", "method_name",
    ]
    for keys, g in detail.groupby(group_cols, sort=False):
        for min_available in availability_grid:
            g_av = g[g["available_models"].ge(int(min_available))]
            for denominator, signal_col in [
                ("sd", "signal_sd"),
                ("rmad", "signal_rmad"),
            ]:
                for k in k_grid:
                    z = g_av[
                        pd.to_numeric(g_av[signal_col], errors="coerce").ge(float(k))
                    ].copy()
                    (
                        selector, selector_name,
                        weekly_rule, weekly_name,
                        method, method_name,
                    ) = keys
                    rows.append({
                        "selector": selector,
                        "selector_name": selector_name,
                        "weekly_rule": weekly_rule,
                        "weekly_rule_name": weekly_name,
                        "method": method,
                        "method_name": method_name,
                        "min_available_n": int(min_available),
                        "denominator": denominator,
                        "k": float(k),
                        **_summary_stats(z),
                    })
    return pd.DataFrame(rows)


def _threshold_by_season(
    detail: pd.DataFrame,
    k_grid: tuple,
    availability_grid: tuple,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "selector", "selector_name",
        "weekly_rule", "weekly_rule_name",
        "method", "method_name", "season",
    ]
    for keys, g in detail.groupby(group_cols, sort=False):
        for min_available in availability_grid:
            g_av = g[g["available_models"].ge(int(min_available))]
            for denominator, signal_col in [
                ("sd", "signal_sd"),
                ("rmad", "signal_rmad"),
            ]:
                for k in k_grid:
                    z = g_av[
                        pd.to_numeric(g_av[signal_col], errors="coerce").ge(float(k))
                    ].copy()
                    (
                        selector, selector_name,
                        weekly_rule, weekly_name,
                        method, method_name, season,
                    ) = keys
                    rows.append({
                        "selector": selector,
                        "selector_name": selector_name,
                        "weekly_rule": weekly_rule,
                        "weekly_rule_name": weekly_name,
                        "method": method,
                        "method_name": method_name,
                        "season": int(season),
                        "min_available_n": int(min_available),
                        "denominator": denominator,
                        "k": float(k),
                        **_summary_stats(z),
                    })
    return pd.DataFrame(rows)


def _stability_table(
    summary: pd.DataFrame,
    by_season: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty or by_season.empty:
        return pd.DataFrame()

    key = [
        "selector", "selector_name",
        "weekly_rule", "weekly_rule_name",
        "method", "method_name", "min_available_n", "denominator", "k",
    ]
    rows = []
    for keys, g in by_season.groupby(key, sort=False):
        overall = summary.copy()
        mask = pd.Series(True, index=overall.index)
        for col, value in zip(key, keys):
            mask &= overall[col].eq(value)
        o = overall[mask]
        if o.empty:
            continue
        o = o.iloc[0]
        season_rois = pd.to_numeric(g["roi"], errors="coerce")
        season_bets = pd.to_numeric(g["bets"], errors="coerce")
        valid = season_bets.gt(0) & season_rois.notna()
        rois = season_rois[valid]
        rows.append({
            **dict(zip(key, keys)),
            "overall_bets": int(o["bets"]),
            "overall_ats_pct": float(o["ats_pct"]) if pd.notna(o["ats_pct"]) else np.nan,
            "overall_units": float(o["units"]),
            "overall_roi": float(o["roi"]) if pd.notna(o["roi"]) else np.nan,
            "seasons_with_bets": int(valid.sum()),
            "positive_roi_seasons": int((rois > 0).sum()),
            "median_season_roi": float(rois.median()) if len(rois) else np.nan,
            "worst_season_roi": float(rois.min()) if len(rois) else np.nan,
            "best_season_roi": float(rois.max()) if len(rois) else np.nan,
            "min_season_bets": int(season_bets[valid].min()) if valid.any() else 0,
        })
    d = pd.DataFrame(rows)
    if not d.empty:
        d = d.sort_values(
            [
                "positive_roi_seasons",
                "worst_season_roi",
                "overall_roi",
                "overall_bets",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        ).reset_index(drop=True)
    return d


def _availability_sensitivity(
    summary: pd.DataFrame,
    primary_min_available: int,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    # Keep a compact region around the historical threshold so the effect of
    # N>=3/4/5 can be read without thousands of rows.
    return summary[
        summary["k"].isin([0.70, 0.80, 0.90, 0.95])
        & summary["denominator"].isin(["sd", "rmad"])
    ].copy().sort_values(
        ["selector_name", "weekly_rule_name", "method_name", "denominator", "k", "min_available_n"],
        kind="stable",
    ).reset_index(drop=True)


def _selector_isolation(
    summary: pd.DataFrame,
    primary_min_available: int,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    # Equal mean provides the cleanest comparison of selector philosophies.
    return summary[
        summary["weekly_rule"].eq("frozen")
        & summary["method"].eq("mean")
        & summary["min_available_n"].eq(int(primary_min_available))
        & summary["k"].isin([0.70, 0.80, 0.90, 0.95])
    ].copy().sort_values(
        ["denominator", "k", "selector_name"],
        kind="stable",
    ).reset_index(drop=True)


def run_automated_selection_lab(
    data: pd.DataFrame,
    candidate_models: Iterable[str],
    methods: Iterable[str],
    backtest_config: BacktestConfig,
    auto_config: AutomatedSelectionConfig | None = None,
) -> dict[str, pd.DataFrame | dict]:
    auto = auto_config or AutomatedSelectionConfig()
    candidates = list(dict.fromkeys(map(str, candidate_models)))
    methods = [m for m in methods if m in METHOD_LABELS]

    empty_result = {
        "auto_selection_detail": pd.DataFrame(),
        "auto_selection_summary": pd.DataFrame(),
        "auto_selection_by_season": pd.DataFrame(),
        "auto_selection_stability": pd.DataFrame(),
        "auto_selection_focus": pd.DataFrame(),
        "auto_selector_isolation": pd.DataFrame(),
        "auto_availability_sensitivity": pd.DataFrame(),
        "auto_preseason_pool_audit": pd.DataFrame(),
        "auto_weekly_pool_audit": pd.DataFrame(),
        "auto_model_quality_audit": pd.DataFrame(),
        "auto_redundancy_audit": pd.DataFrame(),
        "auto_selection_status": {},
    }
    if not candidates or not methods:
        return empty_result

    pred_all, meta_all = _matrix_and_meta(data, candidates)
    if pred_all.empty:
        return empty_result

    model_names = (
        data[["canonical_model_id", "model_name"]]
        .drop_duplicates("canonical_model_id")
        .assign(canonical_model_id=lambda x: x["canonical_model_id"].astype(str))
        .set_index("canonical_model_id")["model_name"]
        .astype(str).to_dict()
    )

    seasons = sorted(meta_all["season"].unique().tolist())
    season_arr = meta_all["season"].to_numpy(dtype=int)
    week_arr = meta_all["week"].to_numpy(dtype=int)

    preseason_pools: dict[tuple[int, str], list[str]] = {}
    target_seasons = set()
    preseason_audit_rows = []
    quality_audit_rows = []
    redundancy_audit_rows = []

    for target_season in seasons:
        prior_seasons = [s for s in seasons if s < target_season]
        if len(prior_seasons) < int(auto.minimum_prior_seasons):
            continue

        train_keys = meta_all.index[meta_all["season"].lt(target_season)]
        if len(train_keys) < int(backtest_config.warmup_games):
            continue

        train_pred = pred_all.loc[train_keys]
        train_meta = meta_all.loc[train_keys]
        metrics = _model_selection_metrics(
            train_pred,
            train_meta,
            candidates,
            auto,
            min_history=int(auto.min_preseason_model_history),
        )
        if metrics.empty:
            continue
        metrics["model_name"] = metrics["canonical_model_id"].map(model_names).fillna(metrics["canonical_model_id"])
        metrics["target_season"] = int(target_season)
        metrics["stage"] = "preseason"

        for selector in auto.selector_rules:
            if selector == "stable_edge_skill" and len(prior_seasons) < int(auto.stable_min_prior_seasons):
                preseason_audit_rows.append({
                    "target_season": int(target_season),
                    "selector": selector,
                    "selector_name": SELECTOR_LABELS.get(selector, selector),
                    "prior_seasons": "|".join(map(str, prior_seasons)),
                    "prior_games": int(len(train_keys)),
                    "eligible_models": 0,
                    "criterion_pass_n": 0,
                    "preseason_pool_n": 0,
                    "pool_status": f"requires {int(auto.stable_min_prior_seasons)} prior seasons",
                    "floor_applied": False,
                    "cap_applied": False,
                    "redundancy_skipped": 0,
                    "score_col": "stable_edge_skill_score",
                    "model_ids": "",
                    "model_names": "",
                })
                continue

            pool, audit, diag = _greedy_diversity_select(
                metrics,
                train_pred,
                train_meta,
                selector,
                auto,
            )

            preseason_audit_rows.append({
                "target_season": int(target_season),
                "selector": selector,
                "selector_name": SELECTOR_LABELS.get(selector, selector),
                "prior_seasons": "|".join(map(str, prior_seasons)),
                "prior_games": int(len(train_keys)),
                "eligible_models": int(diag["eligible_models"]),
                "criterion_pass_n": int(diag["criterion_pass_n"]),
                "preseason_pool_n": int(len(pool)),
                "pool_status": str(diag["pool_status"]),
                "floor_applied": bool(diag["floor_applied"]),
                "cap_applied": bool(diag["cap_applied"]),
                "redundancy_skipped": int(diag["redundancy_skipped"]),
                "score_col": str(diag["score_col"]),
                "model_ids": "|".join(pool),
                "model_names": "|".join(model_names.get(m, m) for m in pool),
            })

            audit = audit.copy()
            audit["target_season"] = int(target_season)
            audit["selector"] = selector
            audit["selector_name"] = SELECTOR_LABELS.get(selector, selector)
            audit["model_name"] = audit["canonical_model_id"].map(model_names).fillna(audit["canonical_model_id"])
            audit["stage"] = "preseason"
            quality_audit_rows.extend(audit.to_dict("records"))
            redundancy_audit_rows.extend(
                audit[
                    [
                        "target_season", "selector", "selector_name",
                        "canonical_model_id", "model_name", "criterion_pass",
                        "selected", "selection_order", "selector_score",
                        "max_corr_to_selected_pool", "redundancy_blocker",
                        "redundancy_blocked",
                    ]
                ].to_dict("records")
            )

            if len(pool) < int(auto.min_qualified_pool):
                continue
            preseason_pools[(target_season, selector)] = pool
            target_seasons.add(int(target_season))

    detail_rows = []
    weekly_audit_rows = []

    for target_season in sorted(target_seasons):
        target_weeks = sorted(
            meta_all.loc[meta_all["season"].eq(target_season), "week"].unique().tolist()
        )
        for week in target_weeks:
            if (
                int(week) < int(backtest_config.evaluation_week_min)
                or int(week) > int(backtest_config.evaluation_week_max)
            ):
                continue

            train_mask = (season_arr < target_season) | (
                (season_arr == target_season) & (week_arr < week)
            )
            test_mask = (season_arr == target_season) & (week_arr == week)
            train_keys = meta_all.index[train_mask]
            test_keys = meta_all.index[test_mask]
            if (
                len(train_keys) < int(backtest_config.warmup_games)
                or len(test_keys) == 0
            ):
                continue

            state_cache = {}

            for selector in auto.selector_rules:
                frozen = preseason_pools.get((target_season, selector), [])
                if len(frozen) < int(auto.min_qualified_pool):
                    continue

                weekly_train_pred = pred_all.loc[train_keys, frozen]
                weekly_train_meta = meta_all.loc[train_keys]
                weekly_metrics = _model_selection_metrics(
                    weekly_train_pred,
                    weekly_train_meta,
                    frozen,
                    auto,
                    min_history=int(auto.min_weekly_model_history),
                )
                weekly_metrics["model_name"] = weekly_metrics["canonical_model_id"].map(model_names).fillna(weekly_metrics["canonical_model_id"])

                for weekly_rule in auto.weekly_rules:
                    selected, weekly_audit, wdiag = _weekly_refine(
                        weekly_metrics,
                        weekly_train_pred,
                        weekly_train_meta,
                        frozen,
                        selector,
                        weekly_rule,
                        auto,
                    )
                    if len(selected) < int(auto.min_qualified_pool):
                        continue

                    selected_tuple = tuple(selected)
                    if selected_tuple not in state_cache:
                        train_raw = pred_all.loc[train_keys, selected]
                        bias, state = _fit_state(
                            train_raw,
                            meta_all.loc[train_keys],
                            selected,
                            methods,
                            backtest_config,
                        )
                        state_cache[selected_tuple] = (bias, state)

                    bias, state = state_cache[selected_tuple]
                    test_raw = pred_all.loc[test_keys, selected]
                    test_adj = _adjust_predictions(test_raw, bias)

                    weekly_audit_rows.append({
                        "season": int(target_season),
                        "week": int(week),
                        "selector": selector,
                        "selector_name": SELECTOR_LABELS.get(selector, selector),
                        "weekly_rule": weekly_rule,
                        "weekly_rule_name": WEEKLY_RULE_LABELS.get(weekly_rule, weekly_rule),
                        "training_games": int(len(train_keys)),
                        "preseason_pool_n": int(len(frozen)),
                        "weekly_eligible_models": int(wdiag["weekly_eligible_models"]),
                        "weekly_pool_n": int(len(selected)),
                        "weekly_floor_applied": bool(wdiag["weekly_floor_applied"]),
                        "weekly_redundancy_skipped": int(wdiag["weekly_redundancy_skipped"]),
                        "model_ids": "|".join(selected),
                        "model_names": "|".join(model_names.get(m, m) for m in selected),
                    })

                    # Generate predictions for any game with at least the minimum
                    # of the 3/4/5 availability sensitivity grid. The summary
                    # layer then evaluates each N threshold separately.
                    detail_floor = max(2, min(map(int, auto.availability_grid)))

                    for game_key, row in test_adj.iterrows():
                        available = row.dropna()
                        if len(available) < detail_floor:
                            continue

                        market = float(meta_all.loc[game_key, "market_margin"])
                        actual = float(meta_all.loc[game_key, "actual_margin"])
                        sd = float(available.std(ddof=1))
                        rmad = _r_mad(available)
                        if not np.isfinite(sd) or sd <= 0:
                            continue

                        eff_n = _effective_n(
                            state["corr"],
                            list(available.index.astype(str)),
                        )

                        for method in methods:
                            consensus, extra = _method_prediction(
                                method,
                                row,
                                market,
                                selected,
                                backtest_config,
                                state,
                            )
                            if not np.isfinite(consensus):
                                continue

                            edge = float(consensus - market)
                            cover = float(actual - market)
                            result, units = _grade(
                                edge, cover, backtest_config.standard_price
                            )

                            detail_rows.append({
                                "season": int(target_season),
                                "week": int(week),
                                "game_key": str(game_key),
                                "selector": selector,
                                "selector_name": SELECTOR_LABELS.get(selector, selector),
                                "weekly_rule": weekly_rule,
                                "weekly_rule_name": WEEKLY_RULE_LABELS.get(weekly_rule, weekly_rule),
                                "method": method,
                                "method_name": METHOD_LABELS.get(method, method),
                                "preseason_pool_n": int(len(frozen)),
                                "weekly_pool_n": int(len(selected)),
                                "available_models": int(len(available)),
                                "effective_n": float(eff_n),
                                "model_sd": sd,
                                "model_rmad": rmad,
                                "consensus_margin": float(consensus),
                                "market_margin": market,
                                "actual_margin": actual,
                                "edge": edge,
                                "absolute_edge": abs(edge),
                                "signal_sd": abs(edge) / sd,
                                "signal_rmad": (
                                    abs(edge) / rmad
                                    if np.isfinite(rmad) and rmad > 0
                                    else np.nan
                                ),
                                "actual_cover_margin": cover,
                                "ats_result": result,
                                "unit_result": units,
                                "market_lambda": float(state.get("market_lambda", np.nan)),
                                "ridge_alpha": float(state.get("ridge_alpha", np.nan)),
                                "posterior_sd": float(extra.get("posterior_sd", np.nan)),
                            })

    detail = pd.DataFrame(detail_rows)
    summary = _threshold_summary(detail, auto.k_grid, auto.availability_grid)
    by_season = _threshold_by_season(detail, auto.k_grid, auto.availability_grid)
    stability = _stability_table(summary, by_season)

    focus = pd.DataFrame()
    if not stability.empty:
        focus = stability[
            stability["min_available_n"].eq(int(auto.primary_min_available))
            & stability["k"].isin([0.70, 0.80, 0.90, 0.95])
            & stability["overall_bets"].ge(40)
        ].copy().reset_index(drop=True)

    selector_isolation = _selector_isolation(summary, int(auto.primary_min_available))
    availability_sensitivity = _availability_sensitivity(summary, int(auto.primary_min_available))

    status = {
        "candidate_models": int(len(candidates)),
        "methods": tuple(methods),
        "target_seasons": tuple(sorted(map(int, target_seasons))),
        "selector_rules": tuple(auto.selector_rules),
        "weekly_rules": tuple(auto.weekly_rules),
        "min_preseason_model_history": int(auto.min_preseason_model_history),
        "min_weekly_model_history": int(auto.min_weekly_model_history),
        "min_qualified_pool": int(auto.min_qualified_pool),
        "primary_min_available": int(auto.primary_min_available),
        "availability_grid": tuple(map(int, auto.availability_grid)),
        "auto_pool_floor": int(auto.auto_pool_floor),
        "auto_pool_cap": int(auto.auto_pool_cap),
        "redundancy_threshold": float(auto.redundancy_threshold),
        "redundancy_metric": "positive market-edge correlation on prior data",
        "edge_skill_definition": (
            "edge=(model-market), market_error=(actual-market); "
            "slope=Cov(edge, market_error)/Var(edge), shrunk toward 0; "
            "ATS posterior evaluated at |edge|>=2"
        ),
        "stable_definition": (
            "at least 2 prior seasons with sufficient games; persistent positive "
            "edge slope, positive market-vs-model MAE delta, and ATS>50% frequency"
        ),
        "leakage_guard": (
            "Preseason pools use seasons strictly before target season; weekly "
            "refinement and method fitting use completed games only."
        ),
    }

    return {
        "auto_selection_detail": detail,
        "auto_selection_summary": summary,
        "auto_selection_by_season": by_season,
        "auto_selection_stability": stability,
        "auto_selection_focus": focus,
        "auto_selector_isolation": selector_isolation,
        "auto_availability_sensitivity": availability_sensitivity,
        "auto_preseason_pool_audit": pd.DataFrame(preseason_audit_rows),
        "auto_weekly_pool_audit": pd.DataFrame(weekly_audit_rows),
        "auto_model_quality_audit": pd.DataFrame(quality_audit_rows),
        "auto_redundancy_audit": pd.DataFrame(redundancy_audit_rows),
        "auto_selection_status": status,
    }
