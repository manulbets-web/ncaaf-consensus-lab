from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import math
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.linear_model import Ridge, RidgeCV


METHOD_LABELS = {
    "mean": "Equal Mean",
    "median": "Median",
    "robust_hybrid": "Robust Hybrid",
    "inverse_mse": "Inverse-MSE Weighted",
    "cluster_balanced": "Cluster-Balanced",
    "market_shrinkage": "Learned Market Shrinkage",
    "market_ridge": "Market-Anchored Ridge",
    "bayesian_market": "Bayesian Market Prior",
}


@dataclass
class BacktestConfig:
    min_models: int = 3
    min_fraction: float = 0.0
    warmup_games: int = 300
    bias_correction: bool = False
    bias_shrink_k: float = 200.0

    winsor_fraction: float = 0.10
    hybrid_median_weight: float = 0.50

    inverse_mse_equal_shrink: float = 0.35
    inverse_mse_min_relative_weight: float = 0.50
    inverse_mse_max_relative_weight: float = 2.00

    cluster_threshold: float = 0.85
    cluster_min_shared: int = 50

    market_shrink_base: str = "robust_hybrid"
    lambda_min: float = 0.0
    lambda_max: float = 1.50

    ridge_mode: str = "time_cv"  # time_cv or fixed
    ridge_alpha: float = 10.0
    ridge_alphas: tuple = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

    bayes_cov_shrinkage: float = 0.35
    bayes_min_model_history: int = 30

    standard_price: int = -110

    # Threshold diagnostics / legacy strategy.
    threshold_sd_scalars: tuple = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
    threshold_raw_edges: tuple = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    # Reproduction / audit controls. Training remains expanding-window; these
    # filters control which historical weeks are evaluated as bets/predictions.
    evaluation_week_min: int = 5
    evaluation_week_max: int = 16

    # available_case = use whichever selected models have a prediction for the
    # game, subject only to min_models (and an optional min_fraction floor).
    # fixed_core is retained only as an advanced audit sensitivity mode.
    availability_mode: str = "available_case"
    core_required_fraction: float = 1.0

    audit_sample_seed: int = 20260811


def _mode_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    mode = s.mode()
    return float(mode.iloc[0]) if len(mode) else float(s.iloc[0])


def _first_nonempty(series: pd.Series):
    for value in series:
        if pd.notna(value) and str(value) != "":
            return value
    return np.nan


def load_strategy_data(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    derived = root / "data" / "derived"
    predictions_path = derived / "model_game_predictions.csv"
    registry_path = derived / "model_registry.csv"
    pairwise_path = derived / "model_pairwise_metrics.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Run scripts/build_model_comparison.py first."
        )

    x = pd.read_csv(predictions_path, low_memory=False)
    registry = pd.read_csv(registry_path, low_memory=False) if registry_path.exists() else pd.DataFrame()
    pairwise = pd.read_csv(pairwise_path, low_memory=False) if pairwise_path.exists() else pd.DataFrame()

    required = {
        "season", "week", "game_key", "canonical_model_id", "model_name",
        "prediction_home_margin", "market_home_margin", "actual_home_margin",
        "pair_orientation",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise ValueError(f"Canonical prediction file missing columns: {missing}")

    for col in [
        "season", "week", "prediction_home_margin",
        "market_home_margin", "actual_home_margin", "pair_orientation",
    ]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    orientation = x["pair_orientation"].fillna(1.0)
    x["prediction_margin"] = x["prediction_home_margin"] * orientation
    x["market_margin_row"] = x["market_home_margin"] * orientation
    x["actual_margin_row"] = x["actual_home_margin"] * orientation

    # For each canonical game use one market/outcome/chronology definition.
    # Prediction Tracker is preferred when available because the canonical
    # registry uses it as the historical source of record for cross-source IDs.
    source_col = "selected_source" if "selected_source" in x.columns else "source"
    if source_col not in x.columns:
        x[source_col] = ""

    game_rows = []
    for game_key, g in x.groupby("game_key", sort=False):
        pt = g[g[source_col].astype(str).eq("predictiontracker")]
        ref = pt if len(pt) else g
        season = int(pd.to_numeric(ref["season"], errors="coerce").dropna().mode().iloc[0])
        week = int(pd.to_numeric(ref["week"], errors="coerce").dropna().mode().iloc[0])
        market = float(pd.to_numeric(ref["market_margin_row"], errors="coerce").dropna().median())
        actual = float(pd.to_numeric(ref["actual_margin_row"], errors="coerce").dropna().median())

        market_reference_source = (
            "predictiontracker" if len(pt) else
            str(_first_nonempty(ref[source_col]))
        )
        if market_reference_source == "cfbpicker":
            market_snapshot_label = "CFB Picker close"
        elif market_reference_source == "predictiontracker":
            market_snapshot_label = "PredictionTracker archive line"
        else:
            market_snapshot_label = f"{market_reference_source} market_home_margin"

        row = {
            "game_key": game_key,
            "season": season,
            "week": week,
            "market_margin": market,
            "actual_margin": actual,
            "team_a_id": _first_nonempty(g["team_a_id"]) if "team_a_id" in g else "",
            "team_b_id": _first_nonempty(g["team_b_id"]) if "team_b_id" in g else "",
            "road": _first_nonempty(ref["road"]) if "road" in ref else "",
            "home": _first_nonempty(ref["home"]) if "home" in ref else "",
            "market_reference_source": market_reference_source,
            "market_snapshot_label": market_snapshot_label,
        }
        game_rows.append(row)

    games = pd.DataFrame(game_rows)
    games["period_key"] = list(zip(games["season"].astype(int), games["week"].astype(int)))

    # Replace row-specific market/outcome with the common game-level reference.
    x = x.merge(
        games[[
            "game_key", "season", "week", "market_margin", "actual_margin",
            "period_key", "team_a_id", "team_b_id", "road", "home",
            "market_reference_source", "market_snapshot_label",
        ]],
        on="game_key",
        how="left",
        suffixes=("_source", ""),
    )
    # Preserve source week/season for audit while using canonical chronology.
    if "season_source" in x.columns:
        x["source_season"] = x["season_source"]
    if "week_source" in x.columns:
        x["source_week"] = x["week_source"]

    return x, registry, pairwise


def selection_diagnostics(
    data: pd.DataFrame,
    selected_models: Iterable[str],
    pairwise: pd.DataFrame | None = None,
) -> dict:
    models = list(dict.fromkeys(map(str, selected_models)))
    d = data[data["canonical_model_id"].astype(str).isin(models)].copy()
    n_models = len(models)
    if n_models == 0:
        return {
            "selected_models": 0, "games": 0, "complete_games": 0,
            "mean_models_per_game": 0.0, "mean_edge_correlation": np.nan,
            "max_edge_correlation": np.nan, "effective_model_count": 0.0,
        }

    coverage = d.groupby("game_key")["canonical_model_id"].nunique()
    result = {
        "selected_models": n_models,
        "games": int(coverage.size),
        "complete_games": int((coverage == n_models).sum()),
        "mean_models_per_game": float(coverage.mean()) if len(coverage) else 0.0,
        "mean_edge_correlation": np.nan,
        "max_edge_correlation": np.nan,
        "effective_model_count": float(n_models),
    }

    if pairwise is not None and len(pairwise):
        p = pairwise[
            pairwise["model_a"].astype(str).isin(models)
            & pairwise["model_b"].astype(str).isin(models)
        ].copy()
        corr = pd.to_numeric(p.get("edge_correlation"), errors="coerce").dropna()
        if len(corr):
            result["mean_edge_correlation"] = float(corr.mean())
            result["max_edge_correlation"] = float(corr.max())

        # Descriptive full-history effective count. Backtest calculations use
        # only prior history and are recomputed period by period.
        mat = pd.DataFrame(np.eye(n_models), index=models, columns=models)
        for row in p.itertuples(index=False):
            value = getattr(row, "edge_correlation", np.nan)
            if pd.notna(value):
                mat.loc[str(row.model_a), str(row.model_b)] = float(value)
                mat.loc[str(row.model_b), str(row.model_a)] = float(value)
        denom = np.abs(mat.to_numpy()).sum()
        if denom > 0:
            result["effective_model_count"] = float(
                np.clip(n_models * n_models / denom, 1.0, n_models)
            )
    return result


def _biases(train: pd.DataFrame, models: list[str], config: BacktestConfig) -> pd.Series:
    if not config.bias_correction:
        return pd.Series(0.0, index=models, dtype=float)

    tmp = train.copy()
    tmp["err"] = tmp["prediction_margin"] - tmp["actual_margin"]
    stats = tmp.groupby("canonical_model_id")["err"].agg(["mean", "count"])
    out = pd.Series(0.0, index=models, dtype=float)
    for m in models:
        if m in stats.index:
            n = float(stats.loc[m, "count"])
            shrink = n / (n + float(config.bias_shrink_k))
            out[m] = float(stats.loc[m, "mean"]) * shrink
    return out


def _wide(train: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = train.pivot_table(
        index="game_key", columns="canonical_model_id",
        values="prediction_margin", aggfunc="first"
    ).reindex(columns=models)
    meta = (
        train[["game_key", "season", "week", "market_margin", "actual_margin", "period_key"]]
        .drop_duplicates("game_key")
        .set_index("game_key")
        .reindex(pred.index)
    )
    return pred, meta


def _adjust_predictions(pred: pd.DataFrame, bias: pd.Series) -> pd.DataFrame:
    return pred.subtract(bias.reindex(pred.columns).fillna(0.0), axis=1)


def _winsorized_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if len(values) < 3 or fraction <= 0:
        return float(np.mean(values))
    lo = float(np.quantile(values, fraction))
    hi = float(np.quantile(values, 1.0 - fraction))
    return float(np.mean(np.clip(values, lo, hi)))


def _base_consensus(values: np.ndarray, method: str, config: BacktestConfig) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "robust_hybrid":
        med = float(np.median(values))
        win = _winsorized_mean(values, config.winsor_fraction)
        w = float(config.hybrid_median_weight)
        return w * med + (1.0 - w) * win
    raise ValueError(f"Unsupported base consensus: {method}")


def _eligibility(values: pd.Series, n_selected: int, config: BacktestConfig) -> bool:
    """Primary game eligibility for the selected ensemble.

    Available-case is the reproduction default: every selected model that has a
    prediction contributes, subject only to a literal minimum model count. The
    selected-pool fraction is optional and defaults to zero. Effective-N is
    descriptive only and is never an eligibility gate.
    """
    n = int(values.notna().sum())
    fraction = n / max(n_selected, 1)

    mode = str(config.availability_mode)
    if mode == "fixed_core":
        required_fraction = float(
            np.clip(config.core_required_fraction, 0.0, 1.0)
        )
        required_n = int(math.ceil(required_fraction * n_selected))
        required_n = max(required_n, int(config.min_models))
        return n >= required_n and fraction >= required_fraction

    if mode not in {"available_case", "flexible"}:
        raise ValueError(f"Unknown availability mode: {mode}")

    optional_fraction = float(np.clip(config.min_fraction, 0.0, 1.0))
    return (
        n >= int(config.min_models)
        and (optional_fraction <= 0.0 or fraction >= optional_fraction)
    )


def _inverse_mse_weights(
    train_pred_adj: pd.DataFrame,
    train_meta: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
) -> pd.Series:
    target = train_meta["actual_margin"]
    errors = train_pred_adj.subtract(target, axis=0)
    mse = errors.pow(2).mean(axis=0, skipna=True)
    global_mse = float(np.nanmedian(mse.to_numpy(dtype=float)))
    if not np.isfinite(global_mse) or global_mse <= 0:
        global_mse = 225.0
    mse = mse.fillna(global_mse).clip(lower=1e-6)
    raw = 1.0 / mse
    raw = raw / raw.sum()

    equal = pd.Series(1.0 / len(models), index=models)
    s = float(np.clip(config.inverse_mse_equal_shrink, 0, 1))
    weights = (1 - s) * raw.reindex(models).fillna(0.0) + s * equal

    relative = weights / equal
    relative = relative.clip(
        lower=float(config.inverse_mse_min_relative_weight),
        upper=float(config.inverse_mse_max_relative_weight),
    )
    weights = relative * equal
    weights = weights / weights.sum()
    return weights


def _edge_correlation(
    pred_adj: pd.DataFrame, meta: pd.DataFrame, min_shared: int
) -> pd.DataFrame:
    edge = pred_adj.subtract(meta["market_margin"], axis=0)
    corr = edge.corr(min_periods=int(min_shared))
    corr = corr.fillna(0.0).clip(-1, 1)

    # pandas 3 copy-on-write can expose DataFrame.values as read-only.
    # Always mutate an explicit writable NumPy copy.
    corr_array = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_array, 1.0)
    return pd.DataFrame(
        corr_array,
        index=corr.index,
        columns=corr.columns,
    )


def _cluster_assignments(
    pred_adj: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
) -> tuple[dict[str, int], pd.DataFrame]:
    if len(models) <= 1:
        return {m: 1 for m in models}, pd.DataFrame(np.eye(len(models)), index=models, columns=models)

    corr = _edge_correlation(
        pred_adj, meta, config.cluster_min_shared
    ).reindex(index=models, columns=models).fillna(0.0)

    corr_array = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_array, 1.0)
    corr = pd.DataFrame(
        corr_array,
        index=models,
        columns=models,
    )

    distance = 1.0 - corr
    arr = distance.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    try:
        Z = linkage(squareform(arr, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - float(config.cluster_threshold), criterion="distance")
        assignments = dict(zip(models, map(int, labels)))
    except Exception:
        assignments = {m: i + 1 for i, m in enumerate(models)}
    return assignments, corr


def _effective_n(corr: pd.DataFrame, available: list[str]) -> float:
    if len(available) <= 1:
        return float(len(available))
    sub = corr.reindex(
        index=available, columns=available
    ).fillna(0.0)

    sub_array = sub.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(sub_array, 1.0)
    denom = np.abs(sub_array).sum()
    if denom <= 0:
        return float(len(available))
    return float(np.clip(len(available) ** 2 / denom, 1.0, len(available)))


def _fit_market_lambda(
    pred_adj: pd.DataFrame,
    meta: pd.DataFrame,
    config: BacktestConfig,
) -> float:
    """Fit market-shrinkage lambda with vectorized row consensus."""
    arr = pred_adj.to_numpy(dtype=float)
    if arr.size == 0:
        return 1.0

    counts = np.isfinite(arr).sum(axis=1)
    if str(config.availability_mode) == "fixed_core":
        availability_fraction = float(
            np.clip(config.core_required_fraction, 0.0, 1.0)
        )
    else:
        availability_fraction = float(
            np.clip(config.min_fraction, 0.0, 1.0)
        )

    required = max(
        int(config.min_models),
        int(math.ceil(availability_fraction * arr.shape[1])),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if config.market_shrink_base == "mean":
            base = np.nanmean(arr, axis=1)
        elif config.market_shrink_base == "median":
            base = np.nanmedian(arr, axis=1)
        elif config.market_shrink_base == "robust_hybrid":
            med = np.nanmedian(arr, axis=1)
            if config.winsor_fraction <= 0:
                win = np.nanmean(arr, axis=1)
            else:
                lo = np.nanquantile(
                    arr,
                    config.winsor_fraction,
                    axis=1,
                )
                hi = np.nanquantile(
                    arr,
                    1.0 - config.winsor_fraction,
                    axis=1,
                )
                clipped = np.minimum(
                    np.maximum(arr, lo[:, None]),
                    hi[:, None],
                )
                win = np.nanmean(clipped, axis=1)
            w = float(config.hybrid_median_weight)
            base = w * med + (1.0 - w) * win
        else:
            raise ValueError(
                f"Unsupported market shrink base: "
                f"{config.market_shrink_base}"
            )

    market = meta["market_margin"].to_numpy(dtype=float)
    actual = meta["actual_margin"].to_numpy(dtype=float)
    keep = (
        (counts >= required)
        & np.isfinite(base)
        & np.isfinite(market)
        & np.isfinite(actual)
    )

    if keep.sum() < 30:
        return 1.0

    x = base[keep] - market[keep]
    y = actual[keep] - market[keep]
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return 1.0

    lam = float(np.dot(x, y) / denom)
    return float(
        np.clip(lam, config.lambda_min, config.lambda_max)
    )


def _period_cv_splits(
    meta: pd.DataFrame,
    min_train_periods: int = 4,
    n_splits: int = 4,
):
    # Keep period keys as Python tuples. np.array(list_of_tuples,
    # dtype=object) can still become a 2-D array, whose rows are
    # unhashable ndarrays.
    periods = [tuple(p) for p in meta["period_key"].tolist()]
    unique = list(dict.fromkeys(periods))

    if len(unique) < min_train_periods + 2:
        return None

    validation_starts = np.linspace(
        min_train_periods,
        len(unique) - 1,
        num=min(
            n_splits,
            len(unique) - min_train_periods,
        ),
        dtype=int,
    )

    splits = []
    for start_idx in sorted(set(map(int, validation_starts))):
        train_periods = set(unique[:start_idx])
        valid_period = unique[start_idx]

        tr = np.asarray(
            [
                i
                for i, p in enumerate(periods)
                if p in train_periods
            ],
            dtype=int,
        )
        va = np.asarray(
            [
                i
                for i, p in enumerate(periods)
                if p == valid_period
            ],
            dtype=int,
        )

        if len(tr) >= 50 and len(va) > 0:
            splits.append((tr, va))

    return splits if len(splits) >= 2 else None


def _fit_ridge(
    pred_adj: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
):
    X = pred_adj.subtract(meta["market_margin"], axis=0).fillna(0.0).reindex(columns=models)
    y = meta["actual_margin"] - meta["market_margin"]
    keep = y.notna()
    X = X.loc[keep]
    y = y.loc[keep]
    m = meta.loc[keep]
    if len(X) < 100:
        return None, np.nan

    if config.ridge_mode == "time_cv":
        splits = _period_cv_splits(m)
        if splits:
            fit = RidgeCV(
                alphas=np.asarray(config.ridge_alphas, dtype=float),
                fit_intercept=True,
                scoring="neg_mean_squared_error",
                cv=splits,
            ).fit(X.to_numpy(), y.to_numpy())
            return fit, float(fit.alpha_)

    fit = Ridge(alpha=float(config.ridge_alpha), fit_intercept=True).fit(X.to_numpy(), y.to_numpy())
    return fit, float(config.ridge_alpha)


def _regularized_error_covariance(
    pred_adj: pd.DataFrame,
    meta: pd.DataFrame,
    models: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    errors = pred_adj.subtract(meta["actual_margin"], axis=0)
    cov = errors.cov(min_periods=int(config.bayes_min_model_history)).reindex(index=models, columns=models)

    variances = errors.var(axis=0, skipna=True).reindex(models)
    fallback_var = float(np.nanmedian(variances.to_numpy(dtype=float)))
    if not np.isfinite(fallback_var) or fallback_var <= 1e-6:
        fallback_var = 225.0
    variances = variances.fillna(fallback_var).clip(lower=1.0)

    cov = cov.fillna(0.0)
    for m in models:
        cov.loc[m, m] = float(variances.loc[m])

    rho = float(np.clip(config.bayes_cov_shrinkage, 0.0, 1.0))
    diag = np.diag(np.diag(cov.to_numpy(dtype=float)))
    arr = (1.0 - rho) * cov.to_numpy(dtype=float) + rho * diag
    arr = (arr + arr.T) / 2.0

    # Eigenvalue floor guarantees a stable positive-definite covariance.
    vals, vecs = np.linalg.eigh(arr)
    floor = max(float(np.nanmedian(np.diag(arr))) * 1e-4, 1e-4)
    vals = np.clip(vals, floor, None)
    arr = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(arr, index=models, columns=models)


def _weighted_available(values: pd.Series, weights: pd.Series) -> float:
    keep = values.notna() & weights.reindex(values.index).notna()
    vals = values[keep]
    w = weights.reindex(vals.index)
    if len(vals) == 0 or float(w.sum()) <= 0:
        return np.nan
    w = w / w.sum()
    return float(np.dot(vals.to_numpy(dtype=float), w.to_numpy(dtype=float)))


def _cluster_balanced(values: pd.Series, assignments: dict[str, int]) -> float:
    vals = values.dropna()
    if len(vals) == 0:
        return np.nan
    clusters: dict[int, list[float]] = {}
    for m, value in vals.items():
        c = int(assignments.get(str(m), hash(str(m)) % 1_000_000))
        clusters.setdefault(c, []).append(float(value))
    cluster_means = [np.mean(v) for v in clusters.values()]
    return float(np.mean(cluster_means))


def _model_distribution_stats(values: pd.Series) -> dict:
    vals = values.dropna().astype(float)
    if len(vals) == 0:
        return {
            "model_mean": np.nan,
            "model_median": np.nan,
            "model_sd": np.nan,
            "model_mad": np.nan,
        }
    mean = float(vals.mean())
    median = float(vals.median())
    sd = float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan
    mad = float(np.median(np.abs(vals - median)))
    return {
        "model_mean": mean,
        "model_median": median,
        "model_sd": sd,
        "model_mad": mad,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator):
        return np.nan
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def _agreement_and_dispersion(values: pd.Series, market: float, consensus: float):
    vals = values.dropna().astype(float)
    if len(vals) == 0:
        return np.nan, np.nan
    med = float(vals.median())
    dispersion = float(np.median(np.abs(vals - med)))
    edge = consensus - market
    if abs(edge) < 1e-12:
        agreement = float(np.mean(np.isclose(vals - market, 0.0)))
    else:
        agreement = float(np.mean(np.sign(vals - market) == np.sign(edge)))
    return agreement, dispersion


def _unit_result(result: str, price: int = -110) -> float:
    if result == "win":
        if price < 0:
            return 100.0 / abs(price)
        return price / 100.0
    if result == "loss":
        return -1.0
    return 0.0


def _one_method_prediction(
    method: str,
    values: pd.Series,
    market: float,
    models: list[str],
    config: BacktestConfig,
    state: dict,
) -> tuple[float, dict]:
    vals = values.reindex(models)
    available = list(vals.dropna().index.astype(str))
    info: dict = {}

    if method in ("mean", "median", "robust_hybrid"):
        consensus = _base_consensus(vals.dropna().to_numpy(), method, config)

    elif method == "inverse_mse":
        consensus = _weighted_available(vals, state["inverse_mse_weights"])

    elif method == "cluster_balanced":
        consensus = _cluster_balanced(vals, state["cluster_assignments"])

    elif method == "market_shrinkage":
        base = _base_consensus(vals.dropna().to_numpy(), config.market_shrink_base, config)
        lam = float(state["market_lambda"])
        consensus = market + lam * (base - market)
        info["lambda"] = lam

    elif method == "market_ridge":
        fit = state.get("ridge_fit")
        if fit is None:
            return np.nan, info
        x = (vals - market).fillna(0.0).reindex(models).to_numpy(dtype=float).reshape(1, -1)
        edge = float(fit.predict(x)[0])
        consensus = market + edge
        info["ridge_alpha"] = float(state.get("ridge_alpha", np.nan))

    elif method == "bayesian_market":
        cov = state["bayes_cov"]
        tau2 = float(state["tau2"])
        if len(available) == 0 or not np.isfinite(tau2) or tau2 <= 0:
            return np.nan, info
        sub = cov.loc[available, available].to_numpy(dtype=float)
        d = (vals.loc[available] - market).to_numpy(dtype=float)
        ones = np.ones(len(available))
        try:
            inv = np.linalg.pinv(sub, hermitian=True)
            precision = 1.0 / tau2 + float(ones @ inv @ ones)
            post_var = 1.0 / precision
            post_mean_edge = post_var * float(ones @ inv @ d)
            consensus = market + post_mean_edge
            info["posterior_sd"] = float(math.sqrt(max(post_var, 0.0)))
            if info["posterior_sd"] > 0:
                info["home_cover_probability"] = float(
                    norm.cdf(post_mean_edge / info["posterior_sd"])
                )
        except Exception:
            return np.nan, info
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(consensus), info


def run_walkforward_backtest(
    data: pd.DataFrame,
    selected_models: Iterable[str],
    methods: Iterable[str],
    config: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Week-level expanding-window backtest.

    v1.1 performance changes:
    - Pivot the selected-model prediction matrix once, not once per week.
    - Fit only the learned components required by selected methods.
    - Cache Bayesian matrix inverses by within-week availability pattern.
    """
    config = config or BacktestConfig()
    models = list(dict.fromkeys(map(str, selected_models)))
    methods = [m for m in methods if m in METHOD_LABELS]

    if not models:
        raise ValueError("Select at least one model.")
    if not methods:
        raise ValueError("Select at least one consensus method.")
    if config.min_models > len(models):
        raise ValueError("Minimum models cannot exceed the number selected.")

    d = data[data["canonical_model_id"].astype(str).isin(models)].copy()
    d = d.dropna(
        subset=[
            "game_key",
            "season",
            "week",
            "prediction_margin",
            "market_margin",
            "actual_margin",
        ]
    )
    if d.empty:
        raise ValueError("No historical predictions exist for the selected models.")

    # Build a compact source/provenance lookup once. Avoid repeatedly scanning
    # the full long table inside the game/model audit loop.
    audit_meta_lookup = {}
    audit_cols = [
        c for c in [
            "game_key", "canonical_model_id", "model_name",
            "selected_source", "source",
        ] if c in d.columns
    ]
    for rec in d[audit_cols].drop_duplicates(
        ["game_key", "canonical_model_id"]
    ).itertuples(index=False):
        rd = rec._asdict()
        audit_meta_lookup[(
            str(rd.get("game_key", "")),
            str(rd.get("canonical_model_id", "")),
        )] = {
            "model_name": str(rd.get("model_name", rd.get("canonical_model_id", ""))),
            "selected_source": str(rd.get("selected_source", "")),
            "source_coverage": str(rd.get("source", "")),
        }

    # Build the selected-model matrix ONCE.
    pred_all = (
        d.pivot_table(
            index="game_key",
            columns="canonical_model_id",
            values="prediction_margin",
            aggfunc="first",
        )
        .reindex(columns=models)
    )

    meta_all = (
        d[
            [
                "game_key",
                "season",
                "week",
                "market_margin",
                "actual_margin",
                "period_key",
                "team_a_id",
                "team_b_id",
                "road",
                "home",
                "market_reference_source",
                "market_snapshot_label",
            ]
        ]
        .drop_duplicates("game_key")
        .set_index("game_key")
        .reindex(pred_all.index)
    )

    valid_meta = (
        meta_all["season"].notna()
        & meta_all["week"].notna()
        & meta_all["market_margin"].notna()
        & meta_all["actual_margin"].notna()
    )
    pred_all = pred_all.loc[valid_meta]
    meta_all = meta_all.loc[valid_meta].copy()

    meta_all["season"] = meta_all["season"].astype(int)
    meta_all["week"] = meta_all["week"].astype(int)

    order = (
        meta_all.reset_index()
        .sort_values(["season", "week", "game_key"])
        ["game_key"]
        .tolist()
    )
    pred_all = pred_all.reindex(order)
    meta_all = meta_all.reindex(order)

    periods = list(
        dict.fromkeys(
            zip(
                meta_all["season"].astype(int),
                meta_all["week"].astype(int),
            )
        )
    )

    needs_inverse = "inverse_mse" in methods
    needs_cluster = "cluster_balanced" in methods
    needs_lambda = "market_shrinkage" in methods
    needs_ridge = "market_ridge" in methods
    needs_bayes = "bayesian_market" in methods

    # We retain effective-N on every result, so prior-history edge correlation
    # is still estimated weekly. Cluster linkage itself is only run when needed.
    details = []
    parameter_rows = []
    audit_model_rows = []

    season_arr = meta_all["season"].to_numpy()
    week_arr = meta_all["week"].to_numpy()

    for season, week in periods:
        # Evaluate only the requested week window, while retaining every prior
        # completed game in the expanding training history. This reproduces the
        # old "start betting at week 5" behavior without throwing away early
        # games as training evidence.
        if (
            int(week) < int(config.evaluation_week_min)
            or int(week) > int(config.evaluation_week_max)
        ):
            continue

        train_mask = (season_arr < season) | (
            (season_arr == season) & (week_arr < week)
        )
        test_mask = (season_arr == season) & (week_arr == week)

        train_keys = meta_all.index[train_mask]
        test_keys = meta_all.index[test_mask]

        train_games = int(len(train_keys))
        if train_games < int(config.warmup_games) or len(test_keys) == 0:
            continue

        train_pred_raw = pred_all.loc[train_keys]
        train_meta = meta_all.loc[train_keys]
        test_pred_raw = pred_all.loc[test_keys]
        test_meta = meta_all.loc[test_keys]

        if config.bias_correction:
            err = train_pred_raw.subtract(
                train_meta["actual_margin"], axis=0
            )
            means = err.mean(axis=0, skipna=True).reindex(models).fillna(0.0)
            counts = err.count(axis=0).reindex(models).fillna(0.0)
            shrink = counts / (counts + float(config.bias_shrink_k))
            bias = means * shrink
        else:
            bias = pd.Series(0.0, index=models, dtype=float)

        train_pred_adj = _adjust_predictions(train_pred_raw, bias)
        test_pred_adj = _adjust_predictions(test_pred_raw, bias)

        # Always estimate correlation because effective N is an output
        # diagnostic. This is substantially cheaper than the old full state fit.
        corr = _edge_correlation(
            train_pred_adj,
            train_meta,
            config.cluster_min_shared,
        ).reindex(
            index=models,
            columns=models,
        ).fillna(0.0)

        corr_array = corr.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr_array, 1.0)
        corr = pd.DataFrame(
            corr_array,
            index=models,
            columns=models,
        )

        if needs_inverse:
            inverse_w = _inverse_mse_weights(
                train_pred_adj, train_meta, models, config
            )
        else:
            inverse_w = pd.Series(
                1.0 / len(models), index=models, dtype=float
            )

        if needs_cluster:
            clusters, _ = _cluster_assignments(
                train_pred_adj, train_meta, models, config
            )
        else:
            clusters = {m: i + 1 for i, m in enumerate(models)}

        market_lambda = (
            _fit_market_lambda(train_pred_adj, train_meta, config)
            if needs_lambda
            else np.nan
        )

        if needs_ridge:
            ridge_fit, ridge_alpha = _fit_ridge(
                train_pred_adj, train_meta, models, config
            )
        else:
            ridge_fit, ridge_alpha = None, np.nan

        if needs_bayes:
            bayes_cov = _regularized_error_covariance(
                train_pred_adj, train_meta, models, config
            )
            market_error = (
                train_meta["actual_margin"] - train_meta["market_margin"]
            )
            tau2 = float(market_error.var(ddof=1))
            if not np.isfinite(tau2) or tau2 <= 1e-6:
                tau2 = 225.0
        else:
            bayes_cov = None
            tau2 = np.nan

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

        parameter_rows.append(
            {
                "season": season,
                "week": week,
                "training_games": train_games,
                "market_lambda": market_lambda,
                "ridge_alpha": ridge_alpha,
                "bayes_tau2": tau2,
                "correlation_clusters": (
                    len(set(clusters.values()))
                    if needs_cluster
                    else np.nan
                ),
            }
        )

        for game_key, row in test_pred_adj.iterrows():
            if not _eligibility(row, len(models), config):
                continue

            market = float(test_meta.loc[game_key, "market_margin"])
            actual = float(test_meta.loc[game_key, "actual_margin"])
            n_available = int(row.notna().sum())
            available = list(row.dropna().index.astype(str))
            eff_n = _effective_n(corr, available)
            missing = [m for m in models if m not in set(available)]
            availability_signature = "|".join(available)
            missing_signature = "|".join(missing)

            team_a = str(test_meta.loc[game_key, "team_a_id"])
            team_b = str(test_meta.loc[game_key, "team_b_id"])
            market_reference_source = str(
                test_meta.loc[game_key, "market_reference_source"]
            )
            market_snapshot_label = str(
                test_meta.loc[game_key, "market_snapshot_label"]
            )

            raw_row = test_pred_raw.loc[game_key]
            for model_id in available:
                model_meta = audit_meta_lookup.get(
                    (str(game_key), str(model_id)), {}
                )
                selected_source = model_meta.get("selected_source", "")
                source_coverage = model_meta.get("source_coverage", "")
                model_name = model_meta.get("model_name", model_id)

                adjusted_prediction = float(row.loc[model_id])
                raw_prediction = float(raw_row.loc[model_id])
                audit_model_rows.append({
                    "season": season,
                    "week": week,
                    "game_key": game_key,
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "canonical_model_id": model_id,
                    "model_name": model_name,
                    "raw_prediction_margin_team_b": raw_prediction,
                    "adjusted_prediction_margin_team_b": adjusted_prediction,
                    "market_margin_team_b": market,
                    "model_edge_team_b": adjusted_prediction - market,
                    "actual_margin_team_b": actual,
                    "selected_source": selected_source,
                    "source_coverage": source_coverage,
                    "market_reference_source": market_reference_source,
                    "market_snapshot_label": market_snapshot_label,
                })

            for method in methods:
                # Bayesian path is handled here so repeated availability
                # patterns can reuse the same covariance inverse.
                if method == "bayesian_market":
                    if bayes_cov is None or len(available) == 0:
                        continue

                    availability_key = tuple(available)
                    cached = state["bayes_cache"].get(availability_key)

                    if cached is None:
                        sub = bayes_cov.loc[
                            available, available
                        ].to_numpy(dtype=float)
                        ones = np.ones(len(available))
                        inv = np.linalg.pinv(sub, hermitian=True)
                        precision = (
                            1.0 / float(tau2)
                            + float(ones @ inv @ ones)
                        )
                        post_var = 1.0 / precision
                        cached = (inv, ones, post_var)
                        state["bayes_cache"][availability_key] = cached

                    inv, ones, post_var = cached
                    model_edges = (
                        row.loc[available] - market
                    ).to_numpy(dtype=float)
                    post_mean_edge = (
                        post_var * float(ones @ inv @ model_edges)
                    )
                    consensus = market + post_mean_edge
                    extra = {
                        "posterior_sd": float(
                            math.sqrt(max(post_var, 0.0))
                        )
                    }
                    if extra["posterior_sd"] > 0:
                        extra["home_cover_probability"] = float(
                            norm.cdf(
                                post_mean_edge
                                / extra["posterior_sd"]
                            )
                        )
                else:
                    consensus, extra = _one_method_prediction(
                        method,
                        row,
                        market,
                        models,
                        config,
                        state,
                    )

                if not np.isfinite(consensus):
                    continue

                edge = float(consensus - market)
                cover_margin = float(actual - market)

                if abs(edge) < 1e-12:
                    result = "no_bet"
                elif abs(cover_margin) < 1e-12:
                    result = "push"
                elif edge * cover_margin > 0:
                    result = "win"
                else:
                    result = "loss"

                agreement, dispersion = _agreement_and_dispersion(
                    row, market, consensus
                )
                dist = _model_distribution_stats(row)

                details.append(
                    {
                        "method": method,
                        "method_name": METHOD_LABELS[method],
                        "season": season,
                        "week": week,
                        "game_key": game_key,
                        "market_margin": market,
                        "actual_margin": actual,
                        "consensus_margin": float(consensus),
                        "consensus_edge": edge,
                        "absolute_edge": abs(edge),
                        "prediction_error": float(consensus - actual),
                        "absolute_error": abs(float(consensus - actual)),
                        "squared_error": float(
                            (consensus - actual) ** 2
                        ),
                        "market_error": market - actual,
                        "market_absolute_error": abs(market - actual),
                        "market_squared_error": (
                            market - actual
                        ) ** 2,
                        "actual_cover_margin": cover_margin,
                        "ats_result": result,
                        "unit_result": _unit_result(
                            result, config.standard_price
                        ),
                        "selected_model_count": len(models),
                        "available_model_count": n_available,
                        "available_fraction": (
                            n_available / len(models)
                        ),
                        "availability_mode": str(config.availability_mode),
                        "core_required_fraction": float(config.core_required_fraction),
                        "availability_signature": availability_signature,
                        "missing_model_count": len(missing),
                        "missing_model_ids": missing_signature,
                        "team_a_id": team_a,
                        "team_b_id": team_b,
                        "positive_margin_side": team_b,
                        "negative_margin_side": team_a,
                        "selected_side": (
                            team_b if edge > 0 else
                            team_a if edge < 0 else
                            "no_bet"
                        ),
                        "market_reference_source": market_reference_source,
                        "market_snapshot_label": market_snapshot_label,
                        "agreement": agreement,
                        "dispersion_mad": dispersion,
                        "model_mean": dist["model_mean"],
                        "model_median": dist["model_median"],
                        "model_sd": dist["model_sd"],
                        "model_mad": dist["model_mad"],
                        "edge_over_sd": _safe_ratio(abs(edge), dist["model_sd"]),
                        "edge_over_mad": _safe_ratio(abs(edge), dist["model_mad"]),
                        "effective_n": eff_n,
                        "training_games": train_games,
                        **extra,
                    }
                )

    detail = pd.DataFrame(details)
    params = pd.DataFrame(parameter_rows)
    audit_models = pd.DataFrame(audit_model_rows)

    if detail.empty:
        return {
            "detail": detail,
            "summary": pd.DataFrame(),
            "common_summary": pd.DataFrame(),
            "edge_table": pd.DataFrame(),
            "sd_threshold_table": pd.DataFrame(),
            "hybrid_threshold_table": pd.DataFrame(),
            "season_threshold_table": pd.DataFrame(),
            "season_table": pd.DataFrame(),
            "direction_audit": pd.DataFrame(),
            "direction_audit_by_season": pd.DataFrame(),
            "availability_summary": pd.DataFrame(),
            "min_model_sensitivity": pd.DataFrame(),
            "threshold_by_availability_bin": pd.DataFrame(),
            "available_model_counts": pd.DataFrame(),
            "orientation_audit_games": pd.DataFrame(),
            "orientation_audit_models": audit_models,
            "parameters": params,
            "config": asdict(config),
        }

    summary = summarize_backtest(detail)
    common_summary = summarize_common_games(detail, methods)
    edge_table = summarize_edges(
        detail, thresholds=tuple(config.threshold_raw_edges)
    )
    sd_threshold_table = summarize_sd_thresholds(
        detail,
        sd_scalars=tuple(config.threshold_sd_scalars),
        raw_edges=(0.0,),
    )
    hybrid_threshold_table = summarize_sd_thresholds(
        detail,
        sd_scalars=tuple(config.threshold_sd_scalars),
        raw_edges=tuple(config.threshold_raw_edges),
    )
    season_threshold_table = summarize_sd_thresholds_by_season(
        detail,
        sd_scalars=tuple(config.threshold_sd_scalars),
        raw_edges=(0.0, 2.0, 3.0),
    )
    direction_audit = summarize_follow_fade(
        detail,
        sd_scalars=tuple(config.threshold_sd_scalars),
        raw_edges=(0.0, 2.0, 3.0),
        price=config.standard_price,
    )
    direction_audit_by_season = summarize_follow_fade_by_season(
        detail,
        sd_scalars=tuple(config.threshold_sd_scalars),
        raw_edges=(0.0, 2.0, 3.0),
        price=config.standard_price,
    )
    availability_summary = summarize_availability(detail)
    sensitivity_counts = tuple(sorted({
        int(config.min_models),
        *[n for n in (3, 5, 7, 9, 11) if n >= int(config.min_models)],
    }))
    min_model_sensitivity = summarize_min_model_sensitivity(
        detail, min_counts=sensitivity_counts, sd_scalars=(0.7, 0.9, 1.1)
    )
    threshold_by_availability_bin = summarize_threshold_by_availability_bin(
        detail, sd_scalars=(0.7, 0.9, 1.1)
    )
    available_model_counts = summarize_available_model_counts(detail)
    orientation_audit_games = build_orientation_audit_games(detail)
    season_table = summarize_by_season(detail)

    return {
        "detail": detail,
        "summary": summary,
        "common_summary": common_summary,
        "edge_table": edge_table,
        "sd_threshold_table": sd_threshold_table,
        "hybrid_threshold_table": hybrid_threshold_table,
        "season_threshold_table": season_threshold_table,
        "direction_audit": direction_audit,
        "direction_audit_by_season": direction_audit_by_season,
        "availability_summary": availability_summary,
        "min_model_sensitivity": min_model_sensitivity,
        "threshold_by_availability_bin": threshold_by_availability_bin,
        "available_model_counts": available_model_counts,
        "orientation_audit_games": orientation_audit_games,
        "orientation_audit_models": audit_models,
        "season_table": season_table,
        "parameters": params,
        "config": asdict(config),
    }


def _summary_row(g: pd.DataFrame) -> dict:
    decisions = g[g["ats_result"].isin(["win", "loss"])]
    wins = int((g["ats_result"] == "win").sum())
    losses = int((g["ats_result"] == "loss").sum())
    pushes = int((g["ats_result"] == "push").sum())
    return {
        "games": int(g["game_key"].nunique()),
        "predictions": int(len(g)),
        "mae": float(g["absolute_error"].mean()),
        "rmse": float(math.sqrt(g["squared_error"].mean())),
        "market_mae": float(g["market_absolute_error"].mean()),
        "market_rmse": float(math.sqrt(g["market_squared_error"].mean())),
        "delta_mae_vs_market": float(g["absolute_error"].mean() - g["market_absolute_error"].mean()),
        "delta_rmse_vs_market": float(
            math.sqrt(g["squared_error"].mean()) - math.sqrt(g["market_squared_error"].mean())
        ),
        "bets": wins + losses,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,
        "ats_pct": wins / (wins + losses) if wins + losses else np.nan,
        "units": float(g["unit_result"].sum()),
        "roi": float(g["unit_result"].sum() / len(decisions)) if len(decisions) else np.nan,
        "mean_edge": float(g["absolute_edge"].mean()),
        "mean_available_models": float(g["available_model_count"].mean()),
        "mean_effective_n": float(g["effective_n"].mean()),
    }


def summarize_backtest(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in detail.groupby("method", sort=False):
        rows.append({
            "method": method,
            "method_name": METHOD_LABELS.get(method, method),
            **_summary_row(g),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def summarize_common_games(detail: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    methods = list(methods)
    game_method = detail[["game_key", "method"]].drop_duplicates()
    counts = game_method.groupby("game_key")["method"].nunique()
    common = set(counts[counts == len(methods)].index)
    d = detail[detail["game_key"].isin(common)]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for method, g in d.groupby("method", sort=False):
        rows.append({
            "method": method,
            "method_name": METHOD_LABELS.get(method, method),
            **_summary_row(g),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def summarize_edges(detail: pd.DataFrame, thresholds=(0, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    rows = []
    for method, mg in detail.groupby("method", sort=False):
        for threshold in thresholds:
            g = mg[mg["absolute_edge"] >= float(threshold)]
            if g.empty:
                continue
            row = {
                "method": method,
                "method_name": METHOD_LABELS.get(method, method),
                "minimum_edge": float(threshold),
                **_summary_row(g),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_sd_thresholds(
    detail: pd.DataFrame,
    sd_scalars=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    raw_edges=(0.0,),
) -> pd.DataFrame:
    rows = []
    for method, mg in detail.groupby("method", sort=False):
        for raw_edge in raw_edges:
            for scalar in sd_scalars:
                keep = (
                    mg["absolute_edge"].ge(float(raw_edge))
                    & mg["edge_over_sd"].ge(float(scalar))
                )
                g = mg[keep].copy()
                if g.empty:
                    continue
                rows.append({
                    "method": method,
                    "method_name": METHOD_LABELS.get(method, method),
                    "raw_edge_threshold": float(raw_edge),
                    "sd_scalar": float(scalar),
                    "rule": f"|edge| >= {float(raw_edge):g} and |edge|/SD >= {float(scalar):g}",
                    **_summary_row(g),
                })
    return pd.DataFrame(rows)


def summarize_sd_thresholds_by_season(
    detail: pd.DataFrame,
    sd_scalars=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    raw_edges=(0.0,),
) -> pd.DataFrame:
    rows = []
    for (method, season), mg in detail.groupby(["method", "season"], sort=False):
        for raw_edge in raw_edges:
            for scalar in sd_scalars:
                keep = (
                    mg["absolute_edge"].ge(float(raw_edge))
                    & mg["edge_over_sd"].ge(float(scalar))
                )
                g = mg[keep].copy()
                if g.empty:
                    continue
                rows.append({
                    "method": method,
                    "method_name": METHOD_LABELS.get(method, method),
                    "season": int(season),
                    "raw_edge_threshold": float(raw_edge),
                    "sd_scalar": float(scalar),
                    **_summary_row(g),
                })
    return pd.DataFrame(rows)


def _price_win_return(price: int) -> float:
    return 100.0 / abs(price) if price < 0 else price / 100.0


def _follow_fade_row(g: pd.DataFrame, price: int) -> dict:
    wins = int((g["ats_result"] == "win").sum())
    losses = int((g["ats_result"] == "loss").sum())
    pushes = int((g["ats_result"] == "push").sum())
    bets = wins + losses
    win_return = _price_win_return(price)

    follow_units = wins * win_return - losses
    fade_units = losses * win_return - wins

    return {
        "bets": bets,
        "pushes": pushes,
        "follow_wins": wins,
        "follow_losses": losses,
        "follow_ats_pct": wins / bets if bets else np.nan,
        "follow_units": follow_units,
        "follow_roi": follow_units / bets if bets else np.nan,
        "fade_wins": losses,
        "fade_losses": wins,
        "fade_ats_pct": losses / bets if bets else np.nan,
        "fade_units": fade_units,
        "fade_roi": fade_units / bets if bets else np.nan,
    }


def summarize_follow_fade(
    detail: pd.DataFrame,
    sd_scalars=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    raw_edges=(0.0, 2.0, 3.0),
    price: int = -110,
) -> pd.DataFrame:
    rows = []
    for method, mg in detail.groupby("method", sort=False):
        for raw_edge in raw_edges:
            for scalar in sd_scalars:
                keep = (
                    mg["absolute_edge"].ge(float(raw_edge))
                    & mg["edge_over_sd"].ge(float(scalar))
                )
                g = mg[keep]
                if g.empty:
                    continue
                rows.append({
                    "method": method,
                    "method_name": METHOD_LABELS.get(method, method),
                    "raw_edge_threshold": float(raw_edge),
                    "sd_scalar": float(scalar),
                    "threshold_label": (
                        f"|edge| >= {float(raw_edge):g} AND "
                        f"|edge|/SD >= {float(scalar):g}"
                    ),
                    **_follow_fade_row(g, price),
                })
    return pd.DataFrame(rows)


def summarize_follow_fade_by_season(
    detail: pd.DataFrame,
    sd_scalars=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    raw_edges=(0.0, 2.0, 3.0),
    price: int = -110,
) -> pd.DataFrame:
    rows = []
    for (method, season), mg in detail.groupby(
        ["method", "season"], sort=False
    ):
        for raw_edge in raw_edges:
            for scalar in sd_scalars:
                keep = (
                    mg["absolute_edge"].ge(float(raw_edge))
                    & mg["edge_over_sd"].ge(float(scalar))
                )
                g = mg[keep]
                if g.empty:
                    continue
                rows.append({
                    "method": method,
                    "method_name": METHOD_LABELS.get(method, method),
                    "season": int(season),
                    "raw_edge_threshold": float(raw_edge),
                    "sd_scalar": float(scalar),
                    **_follow_fade_row(g, price),
                })
    return pd.DataFrame(rows)


def summarize_availability(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    games = detail.drop_duplicates("game_key").copy()
    rows = []
    for signature, g in games.groupby("availability_signature", sort=False):
        rows.append({
            "availability_signature": signature,
            "games": int(g["game_key"].nunique()),
            "seasons": ",".join(map(str, sorted(g["season"].astype(int).unique()))),
            "available_model_count": int(g["available_model_count"].iloc[0]),
            "available_fraction": float(g["available_fraction"].iloc[0]),
            "missing_model_count": int(g["missing_model_count"].iloc[0]),
            "missing_model_ids": str(g["missing_model_ids"].iloc[0]),
            "mean_model_sd": float(g["model_sd"].mean()),
            "mean_effective_n": float(g["effective_n"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["games", "available_model_count"], ascending=[False, False]
    ).reset_index(drop=True)


def _availability_bin(count: int) -> str:
    count = int(count)
    if count <= 5:
        return "3-5"
    if count <= 8:
        return "6-8"
    if count <= 11:
        return "9-11"
    return "12+"


def summarize_min_model_sensitivity(
    detail: pd.DataFrame,
    min_counts=(3, 5, 7, 9, 11),
    sd_scalars=(0.7, 0.9, 1.1),
    raw_edges=(0.0,),
) -> pd.DataFrame:
    """Legacy threshold performance while varying literal available-model N."""
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for method, mg in detail.groupby("method", sort=False):
        for min_n in min_counts:
            n_filtered = mg[mg["available_model_count"] >= int(min_n)]
            if n_filtered.empty:
                continue
            for raw_edge in raw_edges:
                for scalar in sd_scalars:
                    g = n_filtered[
                        n_filtered["absolute_edge"].ge(float(raw_edge))
                        & n_filtered["edge_over_sd"].ge(float(scalar))
                    ]
                    if g.empty:
                        continue
                    rows.append({
                        "method": method,
                        "method_name": METHOD_LABELS.get(method, method),
                        "minimum_available_models": int(min_n),
                        "raw_edge_threshold": float(raw_edge),
                        "sd_scalar": float(scalar),
                        **_summary_row(g),
                    })
    return pd.DataFrame(rows)


def summarize_threshold_by_availability_bin(
    detail: pd.DataFrame,
    sd_scalars=(0.7, 0.9, 1.1),
    raw_edges=(0.0,),
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    d["availability_bin"] = d["available_model_count"].map(_availability_bin)
    bin_order = {"3-5": 0, "6-8": 1, "9-11": 2, "12+": 3}
    rows = []
    for (method, availability_bin), mg in d.groupby(
        ["method", "availability_bin"], sort=False
    ):
        for raw_edge in raw_edges:
            for scalar in sd_scalars:
                g = mg[
                    mg["absolute_edge"].ge(float(raw_edge))
                    & mg["edge_over_sd"].ge(float(scalar))
                ]
                if g.empty:
                    continue
                rows.append({
                    "method": method,
                    "method_name": METHOD_LABELS.get(method, method),
                    "availability_bin": str(availability_bin),
                    "raw_edge_threshold": float(raw_edge),
                    "sd_scalar": float(scalar),
                    **_summary_row(g),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_bin_order"] = out["availability_bin"].map(bin_order)
        out = out.sort_values(
            ["method", "sd_scalar", "raw_edge_threshold", "_bin_order"]
        ).drop(columns="_bin_order").reset_index(drop=True)
    return out


def summarize_available_model_counts(detail: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the literal number of model predictions per game."""
    if detail.empty:
        return pd.DataFrame()
    games = detail.sort_values(["game_key", "method"]).drop_duplicates("game_key")
    total_games = max(int(games["game_key"].nunique()), 1)
    rows = []
    for n, g in games.groupby("available_model_count", sort=True):
        rows.append({
            "available_model_count": int(n),
            "games": int(g["game_key"].nunique()),
            "share_of_games": float(g["game_key"].nunique() / total_games),
            "mean_model_sd": float(g["model_sd"].mean()),
            "mean_effective_n_diagnostic": float(g["effective_n"].mean()),
            "seasons": ",".join(map(str, sorted(g["season"].astype(int).unique()))),
        })
    return pd.DataFrame(rows)


def build_orientation_audit_games(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    # Equal Mean is the primary legacy reproduction method; fall back to the
    # first available method when mean was not requested.
    if (detail["method"] == "mean").any():
        d = detail[detail["method"] == "mean"].copy()
    else:
        first_method = str(detail["method"].iloc[0])
        d = detail[detail["method"] == first_method].copy()

    d = d.drop_duplicates("game_key").copy()
    d["follow_side"] = np.where(
        d["consensus_edge"] > 0,
        d["team_b_id"],
        np.where(d["consensus_edge"] < 0, d["team_a_id"], "no_bet"),
    )
    d["fade_side"] = np.where(
        d["consensus_edge"] > 0,
        d["team_a_id"],
        np.where(d["consensus_edge"] < 0, d["team_b_id"], "no_bet"),
    )

    cols = [
        "season", "week", "game_key", "team_a_id", "team_b_id",
        "market_margin", "market_reference_source", "market_snapshot_label",
        "model_mean", "model_sd", "consensus_margin", "consensus_edge",
        "edge_over_sd", "actual_margin", "actual_cover_margin",
        "follow_side", "fade_side", "ats_result", "available_model_count",
        "selected_model_count", "available_fraction", "missing_model_count",
        "missing_model_ids", "availability_signature",
    ]
    return d[[c for c in cols if c in d.columns]].sort_values(
        ["season", "week", "game_key"]
    ).reset_index(drop=True)


def summarize_by_season(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, season), g in detail.groupby(["method", "season"], sort=False):
        rows.append({
            "method": method,
            "method_name": METHOD_LABELS.get(method, method),
            "season": int(season),
            **_summary_row(g),
        })
    return pd.DataFrame(rows)


def save_backtest(result: dict, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in [
        "detail", "summary", "common_summary", "edge_table",
        "sd_threshold_table", "hybrid_threshold_table",
        "season_threshold_table", "direction_audit",
        "direction_audit_by_season", "availability_summary",
        "min_model_sensitivity", "threshold_by_availability_bin",
        "available_model_counts", "orientation_audit_games",
        "orientation_audit_models",
        "season_table", "parameters",
    ]:
        frame = result.get(name)
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(directory / f"{name}.csv", index=False)
    pd.Series(result.get("config", {})).to_json(
        directory / "config.json", indent=2
    )
