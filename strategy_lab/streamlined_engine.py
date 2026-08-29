from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from automated_selection import (
    AutomatedSelectionConfig,
    _matrix_and_meta,
    _model_selection_metrics,
)


HIGH_PRIORITY_NAMES = [
    "Slate Fluker",
    "Room44 Bets",
    "McIllece Sports",
    "OddsTrader AI",
    "David Sasser",
    "Dokter Entropy",
    "Sagarin Predictor",
    "Power Rank",
    "Sagarin Ratings",
    "JP+",
    "Matt Grissom",
    "Sagarin Golden Mean",
]

SECONDARY_NAMES = [
    "DRatings",
    "Donchess Inference",
    "CFB Geek",
    "KFord",
    "PEARatings",
    "Sagarin Recent",
    "Big 200",
    "Laz Index",
]

POOL_LABELS = {
    "high": "High priority",
    "secondary": "Secondary",
    "other": "All other models",
}


@dataclass
class StreamlinedBacktestConfig:
    selection_mode: str = "walkforward"
    target_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025)
    lookback_seasons: int = 2
    min_history_games: int = 100
    preseason_pool_size: int = 20
    weekly_core_size: int = 7
    min_available_models: int = 4
    evaluation_week_min: int = 5
    evaluation_week_max: int = 16
    thresholds: tuple[float, ...] = (1.25, 1.50, 1.75, 2.00)
    standard_price: int = -110
    beta_alpha: float = 250.0
    beta_beta: float = 250.0
    ats_exponent: float = 4.0
    edge_ats_threshold: float = 2.0


def resolve_model_groups(models: pd.DataFrame) -> dict:
    """Resolve the display-name presets against the current canonical registry."""
    x = models[["canonical_model_id", "model_name"]].drop_duplicates().copy()
    x["canonical_model_id"] = x["canonical_model_id"].astype(str)
    x["model_name"] = x["model_name"].astype(str)

    by_name = {
        name.casefold(): mid
        for mid, name in zip(x["canonical_model_id"], x["model_name"])
    }

    high = [by_name[n.casefold()] for n in HIGH_PRIORITY_NAMES if n.casefold() in by_name]
    secondary = [
        by_name[n.casefold()]
        for n in SECONDARY_NAMES
        if n.casefold() in by_name and by_name[n.casefold()] not in high
    ]
    reserved = set(high) | set(secondary)
    other = x.loc[~x["canonical_model_id"].isin(reserved), "canonical_model_id"].tolist()

    return {
        "high": high,
        "secondary": secondary,
        "other": other,
    }


def make_model_choices(models: pd.DataFrame, ids: Iterable[str]) -> dict[str, str]:
    ids = list(dict.fromkeys(map(str, ids)))
    d = models[models["canonical_model_id"].astype(str).isin(ids)].copy()
    name_map = dict(
        zip(
            d["canonical_model_id"].astype(str),
            d["model_name"].astype(str),
        )
    )
    return {
        mid: name_map.get(mid, mid)
        for mid in ids
        if mid in name_map
    }


def _auto_config(config: StreamlinedBacktestConfig) -> AutomatedSelectionConfig:
    return AutomatedSelectionConfig(
        min_preseason_model_history=int(config.min_history_games),
        min_weekly_model_history=int(config.min_history_games),
        min_season_model_games=50,
        min_qualified_pool=1,
        auto_pool_floor=1,
        auto_pool_cap=max(1, int(config.preseason_pool_size)),
        primary_min_available=int(config.min_available_models),
        beta_alpha=float(config.beta_alpha),
        beta_beta=float(config.beta_beta),
        ats_exponent=float(config.ats_exponent),
        primary_edge_ats_threshold=float(config.edge_ats_threshold),
        edge_ats_thresholds=(1.0, float(config.edge_ats_threshold), 3.0),
    )


def _training_slice(
    data: pd.DataFrame,
    target_season: int,
    lookback_seasons: int,
    before_week: int | None = None,
) -> pd.DataFrame:
    low = int(target_season) - int(lookback_seasons)
    prior = data[
        (pd.to_numeric(data["season"], errors="coerce") >= low)
        & (pd.to_numeric(data["season"], errors="coerce") < int(target_season))
    ].copy()

    if before_week is None:
        return prior

    current = data[
        (pd.to_numeric(data["season"], errors="coerce") == int(target_season))
        & (pd.to_numeric(data["week"], errors="coerce") < int(before_week))
    ].copy()
    return pd.concat([prior, current], ignore_index=True)


def rank_models(
    data: pd.DataFrame,
    candidate_ids: Iterable[str],
    config: StreamlinedBacktestConfig,
    *,
    target_season: int,
    before_week: int | None = None,
) -> pd.DataFrame:
    candidates = list(dict.fromkeys(map(str, candidate_ids)))
    if not candidates:
        return pd.DataFrame()

    train = _training_slice(
        data,
        int(target_season),
        int(config.lookback_seasons),
        before_week=before_week,
    )
    if train.empty:
        return pd.DataFrame()

    pred, meta = _matrix_and_meta(train, candidates)
    if pred.empty:
        return pd.DataFrame()

    metrics = _model_selection_metrics(
        pred,
        meta,
        candidates,
        _auto_config(config),
        min_history=int(config.min_history_games),
    )
    if metrics.empty:
        return metrics

    name_map = (
        data[["canonical_model_id", "model_name"]]
        .drop_duplicates("canonical_model_id")
        .assign(canonical_model_id=lambda z: z["canonical_model_id"].astype(str))
        .set_index("canonical_model_id")["model_name"]
        .astype(str)
        .to_dict()
    )
    metrics["model_name"] = metrics["canonical_model_id"].astype(str).map(name_map)
    metrics["selection_score"] = pd.to_numeric(
        metrics["old_quality_score"], errors="coerce"
    )
    metrics = metrics[
        metrics["eligible_history"].fillna(False)
        & metrics["selection_score"].notna()
    ].copy()
    metrics = metrics.sort_values(
        ["selection_score", "model_mse", "history_n"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    metrics["quality_rank"] = np.arange(1, len(metrics) + 1)
    return metrics


def load_current_ranking(
    project_root,
    data: pd.DataFrame,
    models: pd.DataFrame,
    config: StreamlinedBacktestConfig | None = None,
) -> pd.DataFrame:
    """Prefer the fast R selector output; calculate a fallback if absent."""
    from pathlib import Path

    root = Path(project_root)
    p = root / "outputs" / "fast_model_selector" / "tables" / "model_ranking_2026.csv"
    if p.exists():
        x = pd.read_csv(p, low_memory=False)
        x["canonical_model_id"] = x["canonical_model_id"].astype(str)
        return x

    if config is None:
        config = StreamlinedBacktestConfig()

    ids = models["canonical_model_id"].astype(str).tolist()
    return rank_models(
        data,
        ids,
        config,
        target_season=2026,
        before_week=None,
    )


def model_scorecard(
    models: pd.DataFrame,
    ranking: pd.DataFrame,
    groups: dict,
    selected_ids: Iterable[str],
) -> pd.DataFrame:
    selected = set(map(str, selected_ids))
    x = models[["canonical_model_id", "model_name"]].drop_duplicates().copy()
    x["canonical_model_id"] = x["canonical_model_id"].astype(str)

    group_map = {}
    for group, ids in groups.items():
        for mid in ids:
            group_map[str(mid)] = POOL_LABELS[group]

    x["pool_group"] = x["canonical_model_id"].map(group_map).fillna("All other models")
    x["selected"] = x["canonical_model_id"].isin(selected)

    if ranking is not None and len(ranking):
        keep = [
            c
            for c in [
                "canonical_model_id",
                "quality_rank",
                "selector_score",
                "selection_score",
                "model_mse",
                "model_mae",
                "history_n",
                "coverage_fraction",
                "rank_2024",
                "rank_2025",
                "seasons_top20",
                "seasons_top10",
            ]
            if c in ranking.columns
        ]
        r = ranking[keep].copy()
        r["canonical_model_id"] = r["canonical_model_id"].astype(str)
        x = x.merge(r, on="canonical_model_id", how="left")

    sort_group = pd.Categorical(
        x["pool_group"],
        categories=["High priority", "Secondary", "All other models"],
        ordered=True,
    )
    x["_pool_sort"] = sort_group
    sort_cols = ["_pool_sort"]
    if "quality_rank" in x:
        sort_cols.append("quality_rank")
    else:
        sort_cols.append("model_name")
    x = x.sort_values(sort_cols, na_position="last").drop(columns="_pool_sort")
    return x.reset_index(drop=True)


def _unit_result(result: str, price: int) -> float:
    if result == "win":
        return 100.0 / abs(float(price)) if price < 0 else float(price) / 100.0
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


def _summarize_units(results: pd.Series, units: pd.Series) -> dict:
    r = results.astype(str)
    wins = int((r == "win").sum())
    losses = int((r == "loss").sum())
    pushes = int((r == "push").sum())
    bets = wins + losses
    unit_sum = float(pd.to_numeric(units, errors="coerce").fillna(0).sum())
    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": wins / bets if bets else np.nan,
        "units": unit_sum,
        "roi": unit_sum / bets if bets else np.nan,
    }


def _game_predictions(
    week_data: pd.DataFrame,
    selected_models: list[str],
    config: StreamlinedBacktestConfig,
    *,
    season: int,
    week: int,
    pool_label: str,
) -> pd.DataFrame:
    d = week_data[
        week_data["canonical_model_id"].astype(str).isin(selected_models)
    ].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for game_key, g in d.groupby("game_key", sort=False):
        g = g.drop_duplicates("canonical_model_id")
        vals = pd.to_numeric(g["prediction_margin"], errors="coerce").dropna()
        if len(vals) < int(config.min_available_models):
            continue

        dispersion = float(vals.std(ddof=1))
        if not np.isfinite(dispersion):
            continue

        consensus = float(vals.mean())
        market = float(pd.to_numeric(g["market_margin"], errors="coerce").dropna().iloc[0])
        actual = float(pd.to_numeric(g["actual_margin"], errors="coerce").dropna().iloc[0])
        edge = consensus - market
        cover = actual - market
        result, units = _grade(edge, cover, int(config.standard_price))

        road = str(g["road"].dropna().iloc[0]) if "road" in g and g["road"].notna().any() else ""
        home = str(g["home"].dropna().iloc[0]) if "home" in g and g["home"].notna().any() else ""

        rows.append({
            "season": int(season),
            "week": int(week),
            "game_key": str(game_key),
            "road": road,
            "home": home,
            "pool_label": pool_label,
            "selected_pool_n": len(selected_models),
            "available_models": int(len(vals)),
            "model_ids_available": "|".join(
                g.loc[
                    pd.to_numeric(g["prediction_margin"], errors="coerce").notna(),
                    "canonical_model_id",
                ].astype(str).tolist()
            ),
            "consensus_margin": consensus,
            "market_margin": market,
            "actual_margin": actual,
            "edge": edge,
            "model_sd": dispersion,
            "signal_sd": (
                abs(edge) / dispersion
                if dispersion > 1e-12
                else (np.inf if abs(edge) > 1e-12 else 0.0)
            ),
            "ats_result": result,
            "unit_result": units,
        })

    return pd.DataFrame(rows)


def _season_candidates(
    data: pd.DataFrame,
    selected_ids: list[str],
    config: StreamlinedBacktestConfig,
    target_season: int,
) -> tuple[list[str], pd.DataFrame]:
    if config.selection_mode in {"exact", "weekly"}:
        return list(selected_ids), pd.DataFrame()

    ranking = rank_models(
        data,
        selected_ids,
        config,
        target_season=int(target_season),
    )
    n = min(int(config.preseason_pool_size), len(ranking))
    ids = ranking.head(n)["canonical_model_id"].astype(str).tolist()
    return ids, ranking


def _weekly_models(
    data: pd.DataFrame,
    frozen_ids: list[str],
    config: StreamlinedBacktestConfig,
    season: int,
    week: int,
) -> tuple[list[str], pd.DataFrame]:
    if config.selection_mode == "exact":
        return list(frozen_ids), pd.DataFrame()

    ranking = rank_models(
        data,
        frozen_ids,
        config,
        target_season=int(season),
        before_week=int(week),
    )
    n = min(int(config.weekly_core_size), len(ranking))
    ids = ranking.head(n)["canonical_model_id"].astype(str).tolist()
    return ids, ranking


def run_streamlined_backtest(
    data: pd.DataFrame,
    selected_ids: Iterable[str],
    config: StreamlinedBacktestConfig,
) -> dict:
    selected_ids = list(dict.fromkeys(map(str, selected_ids)))
    if not selected_ids:
        raise ValueError("Select at least one model.")

    seasons = tuple(sorted(set(map(int, config.target_seasons))))
    details = []
    preseason_rows = []
    weekly_rows = []

    name_map = (
        data[["canonical_model_id", "model_name"]]
        .drop_duplicates("canonical_model_id")
        .assign(canonical_model_id=lambda z: z["canonical_model_id"].astype(str))
        .set_index("canonical_model_id")["model_name"]
        .astype(str)
        .to_dict()
    )

    for season in seasons:
        frozen, pre_rank = _season_candidates(
            data, selected_ids, config, int(season)
        )
        if not frozen:
            continue

        preseason_rows.append({
            "season": int(season),
            "selection_mode": config.selection_mode,
            "candidate_n": len(selected_ids),
            "frozen_pool_n": len(frozen),
            "model_ids": "|".join(frozen),
            "model_names": "|".join(name_map.get(mid, mid) for mid in frozen),
        })

        season_weeks = (
            pd.to_numeric(
                data.loc[pd.to_numeric(data["season"], errors="coerce").eq(season), "week"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        weeks = [
            w for w in sorted(season_weeks)
            if int(config.evaluation_week_min) <= w <= int(config.evaluation_week_max)
        ]

        for week in weeks:
            active, week_rank = _weekly_models(
                data, frozen, config, int(season), int(week)
            )
            if len(active) < int(config.min_available_models):
                continue

            weekly_rows.append({
                "season": int(season),
                "week": int(week),
                "selection_mode": config.selection_mode,
                "frozen_pool_n": len(frozen),
                "weekly_pool_n": len(active),
                "model_ids": "|".join(active),
                "model_names": "|".join(name_map.get(mid, mid) for mid in active),
            })

            week_data = data[
                pd.to_numeric(data["season"], errors="coerce").eq(season)
                & pd.to_numeric(data["week"], errors="coerce").eq(week)
            ]
            game_rows = _game_predictions(
                week_data,
                active,
                config,
                season=int(season),
                week=int(week),
                pool_label=config.selection_mode,
            )
            if len(game_rows):
                details.append(game_rows)

    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    preseason = pd.DataFrame(preseason_rows)
    weekly = pd.DataFrame(weekly_rows)

    summary_rows = []
    season_rows = []
    thresholds = sorted(set(map(float, config.thresholds)))

    if len(detail):
        for k in thresholds:
            q = detail[
                pd.to_numeric(detail["signal_sd"], errors="coerce").ge(k)
            ].copy()
            s = _summarize_units(q["ats_result"], q["unit_result"])
            summary_rows.append({"k": k, **s})

            for season in seasons:
                qs = q[q["season"].eq(season)]
                ss = _summarize_units(qs["ats_result"], qs["unit_result"])
                season_rows.append({"season": season, "k": k, **ss})

    summary = pd.DataFrame(summary_rows)
    by_season = pd.DataFrame(season_rows)

    frequency_rows = []
    if len(weekly):
        for _, row in weekly.iterrows():
            ids = [x for x in str(row["model_ids"]).split("|") if x]
            for mid in ids:
                frequency_rows.append({
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "canonical_model_id": mid,
                    "model_name": name_map.get(mid, mid),
                })
    frequency = pd.DataFrame(frequency_rows)
    if len(frequency):
        frequency = (
            frequency.groupby(
                ["canonical_model_id", "model_name"], as_index=False
            )
            .agg(
                weeks_selected=("week", "size"),
                seasons_selected=("season", "nunique"),
            )
            .sort_values(
                ["weeks_selected", "seasons_selected"],
                ascending=False,
            )
        )

    return {
        "detail": detail,
        "summary": summary,
        "by_season": by_season,
        "preseason": preseason,
        "weekly": weekly,
        "frequency": frequency,
        "config": config,
    }


# ---------------------------------------------------------------------------
# v3.1: Historical benchmark + marginal contribution analysis
# ---------------------------------------------------------------------------

@dataclass
class ContributionConfig:
    mode: str = "exact"  # exact or walkforward
    primary_k: float = 1.50
    add_scope: str = "secondary"  # secondary, all_unselected, none
    max_add_candidates: int = 100
    minimum_common_bets: int = 10


def _primary_threshold_frame(result: dict, k: float) -> pd.DataFrame:
    d = result.get("detail", pd.DataFrame())
    if d is None or d.empty:
        return pd.DataFrame()
    sig = pd.to_numeric(d["signal_sd"], errors="coerce")
    return d[sig.ge(float(k))].copy()


def _summary_from_frame(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "ats_pct": np.nan,
            "units": 0.0,
            "roi": np.nan,
        }
    return _summarize_units(frame["ats_result"], frame["unit_result"])


def _common_game_effect(
    baseline_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
) -> dict:
    """
    Compare performance on games selected by BOTH strategies.

    A common game is keyed by season/week/game_key. The side may differ if
    removing/adding a model flips the consensus edge. That is intentional:
    the common-game delta reflects the practical change in the bet decision
    on the same underlying game.
    """
    if baseline_frame.empty or variant_frame.empty:
        return {
            "common_bets": 0,
            "baseline_common_ats": np.nan,
            "variant_common_ats": np.nan,
            "delta_common_ats_pp": np.nan,
            "baseline_common_roi": np.nan,
            "variant_common_roi": np.nan,
            "delta_common_roi_pp": np.nan,
            "side_flip_rate": np.nan,
        }

    keys = ["season", "week", "game_key"]
    b = baseline_frame.copy()
    v = variant_frame.copy()

    b["_edge_sign"] = np.sign(pd.to_numeric(b["edge"], errors="coerce"))
    v["_edge_sign"] = np.sign(pd.to_numeric(v["edge"], errors="coerce"))

    b = b.drop_duplicates(keys)
    v = v.drop_duplicates(keys)

    common = b.merge(
        v,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_variant"),
    )
    if common.empty:
        return {
            "common_bets": 0,
            "baseline_common_ats": np.nan,
            "variant_common_ats": np.nan,
            "delta_common_ats_pp": np.nan,
            "baseline_common_roi": np.nan,
            "variant_common_roi": np.nan,
            "delta_common_roi_pp": np.nan,
            "side_flip_rate": np.nan,
        }

    bs = _summarize_units(
        common["ats_result_baseline"],
        common["unit_result_baseline"],
    )
    vs = _summarize_units(
        common["ats_result_variant"],
        common["unit_result_variant"],
    )

    return {
        "common_bets": int(len(common)),
        "baseline_common_ats": bs["ats_pct"],
        "variant_common_ats": vs["ats_pct"],
        "delta_common_ats_pp": (
            100.0 * (vs["ats_pct"] - bs["ats_pct"])
            if np.isfinite(vs["ats_pct"]) and np.isfinite(bs["ats_pct"])
            else np.nan
        ),
        "baseline_common_roi": bs["roi"],
        "variant_common_roi": vs["roi"],
        "delta_common_roi_pp": (
            100.0 * (vs["roi"] - bs["roi"])
            if np.isfinite(vs["roi"]) and np.isfinite(bs["roi"])
            else np.nan
        ),
        "side_flip_rate": float(
            (
                common["_edge_sign_baseline"]
                != common["_edge_sign_variant"]
            ).mean()
        ),
    }


def _season_delta_count(
    baseline_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
) -> dict:
    seasons = sorted(
        set(pd.to_numeric(baseline_frame.get("season"), errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(variant_frame.get("season"), errors="coerce").dropna().astype(int))
    )
    improved = 0
    worsened = 0
    tied = 0
    rows = []

    for season in seasons:
        b = baseline_frame[
            pd.to_numeric(baseline_frame["season"], errors="coerce").eq(season)
        ]
        v = variant_frame[
            pd.to_numeric(variant_frame["season"], errors="coerce").eq(season)
        ]
        bs = _summary_from_frame(b)
        vs = _summary_from_frame(v)

        delta = (
            vs["roi"] - bs["roi"]
            if np.isfinite(vs["roi"]) and np.isfinite(bs["roi"])
            else np.nan
        )
        if np.isfinite(delta):
            if delta > 1e-12:
                improved += 1
            elif delta < -1e-12:
                worsened += 1
            else:
                tied += 1
        rows.append({
            "season": season,
            "baseline_bets": bs["bets"],
            "variant_bets": vs["bets"],
            "baseline_roi": bs["roi"],
            "variant_roi": vs["roi"],
            "delta_roi": delta,
        })

    return {
        "seasons_improved": improved,
        "seasons_worsened": worsened,
        "seasons_tied": tied,
        "season_rows": rows,
    }


def run_historical_benchmark(
    data: pd.DataFrame,
    all_model_ids: Iterable[str],
    config: StreamlinedBacktestConfig,
) -> dict:
    """
    True historical benchmark:
      all canonical models as candidate universe
      -> prior-data Top N
      -> weekly core
      -> Mean / SD
    """
    cfg = StreamlinedBacktestConfig(
        selection_mode="walkforward",
        target_seasons=tuple(config.target_seasons),
        lookback_seasons=int(config.lookback_seasons),
        min_history_games=int(config.min_history_games),
        preseason_pool_size=int(config.preseason_pool_size),
        weekly_core_size=int(config.weekly_core_size),
        min_available_models=int(config.min_available_models),
        evaluation_week_min=int(config.evaluation_week_min),
        evaluation_week_max=int(config.evaluation_week_max),
        thresholds=tuple(config.thresholds),
        standard_price=int(config.standard_price),
        beta_alpha=float(config.beta_alpha),
        beta_beta=float(config.beta_beta),
        ats_exponent=float(config.ats_exponent),
        edge_ats_threshold=float(config.edge_ats_threshold),
    )
    return run_streamlined_backtest(data, all_model_ids, cfg)


def _analysis_config(
    base: StreamlinedBacktestConfig,
    contribution: ContributionConfig,
) -> StreamlinedBacktestConfig:
    mode = "exact" if contribution.mode == "exact" else "walkforward"
    return StreamlinedBacktestConfig(
        selection_mode=mode,
        target_seasons=tuple(base.target_seasons),
        lookback_seasons=int(base.lookback_seasons),
        min_history_games=int(base.min_history_games),
        preseason_pool_size=int(base.preseason_pool_size),
        weekly_core_size=int(base.weekly_core_size),
        min_available_models=int(base.min_available_models),
        evaluation_week_min=int(base.evaluation_week_min),
        evaluation_week_max=int(base.evaluation_week_max),
        thresholds=(float(contribution.primary_k),),
        standard_price=int(base.standard_price),
        beta_alpha=float(base.beta_alpha),
        beta_beta=float(base.beta_beta),
        ats_exponent=float(base.ats_exponent),
        edge_ats_threshold=float(base.edge_ats_threshold),
    )


def _variant_row(
    *,
    kind: str,
    model_id: str,
    model_name: str,
    baseline_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    baseline_summary: dict,
    minimum_common_bets: int,
) -> dict:
    vs = _summary_from_frame(variant_frame)
    common = _common_game_effect(baseline_frame, variant_frame)
    season = _season_delta_count(baseline_frame, variant_frame)

    delta_roi_pp = (
        100.0 * (vs["roi"] - baseline_summary["roi"])
        if np.isfinite(vs["roi"]) and np.isfinite(baseline_summary["roi"])
        else np.nan
    )
    delta_ats_pp = (
        100.0 * (vs["ats_pct"] - baseline_summary["ats_pct"])
        if np.isfinite(vs["ats_pct"]) and np.isfinite(baseline_summary["ats_pct"])
        else np.nan
    )

    # Contribution is positive when the model appears helpful:
    # - removal hurts -> positive contribution
    # - addition helps -> positive contribution
    direction = -1.0 if kind == "remove" else 1.0

    contribution_roi_pp = direction * delta_roi_pp if np.isfinite(delta_roi_pp) else np.nan
    contribution_ats_pp = direction * delta_ats_pp if np.isfinite(delta_ats_pp) else np.nan

    # For common games, positive contribution means baseline (with the model)
    # is better for removal tests, and variant (with the added model) is better
    # for addition tests.
    common_contrib_roi_pp = (
        direction * common["delta_common_roi_pp"]
        if np.isfinite(common["delta_common_roi_pp"])
        else np.nan
    )
    common_contrib_ats_pp = (
        direction * common["delta_common_ats_pp"]
        if np.isfinite(common["delta_common_ats_pp"])
        else np.nan
    )

    robust_common = common["common_bets"] >= int(minimum_common_bets)

    # A compact review score only for sorting. It is deliberately transparent:
    # common-game effect gets more weight than raw selection-set changes.
    parts = []
    if np.isfinite(common_contrib_roi_pp) and robust_common:
        parts.append(0.45 * common_contrib_roi_pp)
    if np.isfinite(common_contrib_ats_pp) and robust_common:
        parts.append(0.25 * common_contrib_ats_pp)
    if np.isfinite(contribution_roi_pp):
        parts.append(0.20 * contribution_roi_pp)
    if np.isfinite(contribution_ats_pp):
        parts.append(0.10 * contribution_ats_pp)
    review_score = float(np.sum(parts)) if parts else np.nan

    if np.isfinite(review_score):
        if review_score >= 1.0:
            signal = "HELPFUL"
        elif review_score <= -1.0:
            signal = "HARMFUL"
        else:
            signal = "NEUTRAL"
    else:
        signal = "INSUFFICIENT"

    return {
        "analysis_type": "leave_one_out" if kind == "remove" else "add_one_in",
        "model_id": model_id,
        "model_name": model_name,
        "signal": signal,
        "review_score": review_score,
        "baseline_bets": baseline_summary["bets"],
        "variant_bets": vs["bets"],
        "delta_bets": int(vs["bets"] - baseline_summary["bets"]),
        "baseline_ats_pct": baseline_summary["ats_pct"],
        "variant_ats_pct": vs["ats_pct"],
        "contribution_ats_pp": contribution_ats_pp,
        "baseline_roi": baseline_summary["roi"],
        "variant_roi": vs["roi"],
        "contribution_roi_pp": contribution_roi_pp,
        "common_bets": common["common_bets"],
        "common_contribution_ats_pp": common_contrib_ats_pp,
        "common_contribution_roi_pp": common_contrib_roi_pp,
        "side_flip_rate": common["side_flip_rate"],
        "seasons_variant_better": (
            season["seasons_worsened"]
            if kind == "remove"
            else season["seasons_improved"]
        ),
        "seasons_variant_worse": (
            season["seasons_improved"]
            if kind == "remove"
            else season["seasons_worsened"]
        ),
        "robust_common_sample": robust_common,
    }


def run_contribution_analysis(
    data: pd.DataFrame,
    selected_ids: Iterable[str],
    candidate_add_ids: Iterable[str],
    model_name_map: dict[str, str],
    base_config: StreamlinedBacktestConfig,
    contribution_config: ContributionConfig,
    *,
    progress_callback=None,
) -> dict:
    selected = list(dict.fromkeys(map(str, selected_ids)))
    candidate_add = [
        x
        for x in list(dict.fromkeys(map(str, candidate_add_ids)))
        if x not in set(selected)
    ][: int(contribution_config.max_add_candidates)]

    if len(selected) < int(base_config.min_available_models):
        raise ValueError(
            "Selected pool is smaller than the minimum available-model gate."
        )

    cfg = _analysis_config(base_config, contribution_config)

    total = 1 + len(selected) + len(candidate_add)
    step = 0

    def ping(label):
        nonlocal step
        step += 1
        if progress_callback is not None:
            progress_callback(step, total, label)

    ping("Baseline")
    baseline_result = run_streamlined_backtest(data, selected, cfg)
    baseline_frame = _primary_threshold_frame(
        baseline_result, contribution_config.primary_k
    )
    baseline_summary = _summary_from_frame(baseline_frame)

    rows = []

    for mid in selected:
        remaining = [x for x in selected if x != mid]
        if len(remaining) < int(base_config.min_available_models):
            continue
        ping(f"Remove {model_name_map.get(mid, mid)}")
        result = run_streamlined_backtest(data, remaining, cfg)
        frame = _primary_threshold_frame(
            result, contribution_config.primary_k
        )
        rows.append(
            _variant_row(
                kind="remove",
                model_id=mid,
                model_name=model_name_map.get(mid, mid),
                baseline_frame=baseline_frame,
                variant_frame=frame,
                baseline_summary=baseline_summary,
                minimum_common_bets=contribution_config.minimum_common_bets,
            )
        )

    for mid in candidate_add:
        ping(f"Add {model_name_map.get(mid, mid)}")
        result = run_streamlined_backtest(
            data, selected + [mid], cfg
        )
        frame = _primary_threshold_frame(
            result, contribution_config.primary_k
        )
        rows.append(
            _variant_row(
                kind="add",
                model_id=mid,
                model_name=model_name_map.get(mid, mid),
                baseline_frame=baseline_frame,
                variant_frame=frame,
                baseline_summary=baseline_summary,
                minimum_common_bets=contribution_config.minimum_common_bets,
            )
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["review_score", "common_bets"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    return {
        "baseline_result": baseline_result,
        "baseline_frame": baseline_frame,
        "baseline_summary": baseline_summary,
        "contributions": table,
        "config": contribution_config,
    }


# ---------------------------------------------------------------------------
# v3.2: coverage-aware / fixed-game contribution analysis
# ---------------------------------------------------------------------------

def _ids_contain(value, model_id: str) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(model_id) in {x for x in str(value).split("|") if x}


def _fixed_baseline_forecast_effect(
    baseline_frame: pd.DataFrame,
    variant_detail: pd.DataFrame,
    *,
    kind: str,
    model_id: str,
    primary_k: float,
) -> dict:
    """
    Hold the BASELINE BET SET fixed and ask whether the variant consensus was
    more or less accurate on those exact games.

    This avoids the central flaw of binary common-game ATS:
    if the side does not flip, ATS is necessarily identical even when the
    forecast itself meaningfully improves or deteriorates.
    """
    empty = {
        "baseline_bet_games": int(len(baseline_frame)) if baseline_frame is not None else 0,
        "fixed_game_n": 0,
        "fixed_game_fraction": np.nan,
        "model_presence_on_baseline_bets": 0,
        "model_presence_fraction": np.nan,
        "baseline_forecast_mae": np.nan,
        "variant_forecast_mae": np.nan,
        "forecast_mae_contribution_points": np.nan,
        "baseline_forecast_rmse": np.nan,
        "variant_forecast_rmse": np.nan,
        "forecast_rmse_contribution_points": np.nan,
        "baseline_edge_mae": np.nan,
        "variant_edge_mae": np.nan,
        "threshold_retention_rate": np.nan,
        "direction_flip_rate_fixed": np.nan,
    }

    if baseline_frame is None or baseline_frame.empty:
        return empty
    if variant_detail is None or variant_detail.empty:
        return empty

    keys = ["season", "week", "game_key"]
    b = baseline_frame.drop_duplicates(keys).copy()
    v = variant_detail.drop_duplicates(keys).copy()

    keep_b = keys + [
        "consensus_margin",
        "market_margin",
        "actual_margin",
        "edge",
        "signal_sd",
        "model_ids_available",
    ]
    keep_v = keys + [
        "consensus_margin",
        "market_margin",
        "actual_margin",
        "edge",
        "signal_sd",
        "model_ids_available",
    ]
    keep_b = [c for c in keep_b if c in b.columns]
    keep_v = [c for c in keep_v if c in v.columns]

    m = b[keep_b].merge(
        v[keep_v],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_variant"),
    )
    if m.empty:
        return empty

    actual = pd.to_numeric(m["actual_margin_baseline"], errors="coerce")
    cb = pd.to_numeric(m["consensus_margin_baseline"], errors="coerce")
    cv = pd.to_numeric(m["consensus_margin_variant"], errors="coerce")

    valid = actual.notna() & cb.notna() & cv.notna()
    mm = m.loc[valid].copy()
    if mm.empty:
        return empty

    actual = pd.to_numeric(mm["actual_margin_baseline"], errors="coerce")
    cb = pd.to_numeric(mm["consensus_margin_baseline"], errors="coerce")
    cv = pd.to_numeric(mm["consensus_margin_variant"], errors="coerce")

    eb = cb - actual
    ev = cv - actual
    mae_b = float(np.mean(np.abs(eb)))
    mae_v = float(np.mean(np.abs(ev)))
    rmse_b = float(np.sqrt(np.mean(np.square(eb))))
    rmse_v = float(np.sqrt(np.mean(np.square(ev))))

    # Positive always means the MODEL is helpful.
    # Removal: variant error - baseline error.
    # Addition: baseline error - variant error.
    if kind == "remove":
        mae_contrib = mae_v - mae_b
        rmse_contrib = rmse_v - rmse_b
    else:
        mae_contrib = mae_b - mae_v
        rmse_contrib = rmse_b - rmse_v

    edge_b = pd.to_numeric(mm["edge_baseline"], errors="coerce")
    edge_v = pd.to_numeric(mm["edge_variant"], errors="coerce")
    market = pd.to_numeric(mm["market_margin_baseline"], errors="coerce")
    realized_cover = actual - market

    # Edge prediction error equals consensus margin error, but keep the
    # explicit market-relative form because it is easier to interpret.
    edge_mae_b = float(np.mean(np.abs(edge_b - realized_cover)))
    edge_mae_v = float(np.mean(np.abs(edge_v - realized_cover)))

    sig_v = pd.to_numeric(mm["signal_sd_variant"], errors="coerce")
    threshold_retention = float(sig_v.ge(float(primary_k)).mean())

    sign_b = np.sign(edge_b)
    sign_v = np.sign(edge_v)
    flip_rate = float((sign_b != sign_v).mean())

    if kind == "remove":
        source_presence = b.get("model_ids_available", pd.Series(index=b.index, dtype=object))
    else:
        source_presence = v.get("model_ids_available", pd.Series(index=v.index, dtype=object))

    # Presence is measured on baseline-bet game keys.
    presence_df = pd.DataFrame({**{k: b[k] for k in keys}})
    presence_df["present"] = source_presence.apply(
        lambda x: _ids_contain(x, model_id)
    )
    presence_join = b[keys].merge(
        presence_df,
        on=keys,
        how="left",
    )
    presence_n = int(presence_join["present"].fillna(False).sum())

    n_base = int(len(b))
    n_fixed = int(len(mm))

    return {
        "baseline_bet_games": n_base,
        "fixed_game_n": n_fixed,
        "fixed_game_fraction": n_fixed / n_base if n_base else np.nan,
        "model_presence_on_baseline_bets": presence_n,
        "model_presence_fraction": presence_n / n_base if n_base else np.nan,
        "baseline_forecast_mae": mae_b,
        "variant_forecast_mae": mae_v,
        "forecast_mae_contribution_points": mae_contrib,
        "baseline_forecast_rmse": rmse_b,
        "variant_forecast_rmse": rmse_v,
        "forecast_rmse_contribution_points": rmse_contrib,
        "baseline_edge_mae": edge_mae_b,
        "variant_edge_mae": edge_mae_v,
        "threshold_retention_rate": threshold_retention,
        "direction_flip_rate_fixed": flip_rate,
    }


def _contribution_season_counts(
    baseline_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    *,
    kind: str,
) -> dict:
    seasons = sorted(
        set(pd.to_numeric(baseline_frame.get("season"), errors="coerce").dropna().astype(int))
        | set(pd.to_numeric(variant_frame.get("season"), errors="coerce").dropna().astype(int))
    )

    positive = 0
    negative = 0
    neutral = 0

    for season in seasons:
        b = baseline_frame[
            pd.to_numeric(baseline_frame["season"], errors="coerce").eq(season)
        ]
        v = variant_frame[
            pd.to_numeric(variant_frame["season"], errors="coerce").eq(season)
        ]
        bs = _summary_from_frame(b)
        vs = _summary_from_frame(v)

        if not (np.isfinite(bs["roi"]) and np.isfinite(vs["roi"])):
            continue

        raw_delta = 100.0 * (vs["roi"] - bs["roi"])
        contribution = -raw_delta if kind == "remove" else raw_delta

        if contribution > 1e-9:
            positive += 1
        elif contribution < -1e-9:
            negative += 1
        else:
            neutral += 1

    return {
        "positive_contribution_seasons": positive,
        "negative_contribution_seasons": negative,
        "neutral_contribution_seasons": neutral,
    }


def _coverage_aware_signal(
    contribution_roi_pp: float,
    contribution_ats_pp: float,
    forecast_mae_contribution_points: float,
    fixed_game_n: int,
    minimum_fixed_games: int,
    positive_seasons: int,
    negative_seasons: int,
) -> str:
    """
    Transparent directional label, not an optimization target.

    Require operational and fixed-game forecast evidence to agree before
    calling a model helpful/harmful. Otherwise label it MIXED/NEUTRAL.
    """
    enough_fixed = int(fixed_game_n) >= int(minimum_fixed_games)
    roi = contribution_roi_pp if np.isfinite(contribution_roi_pp) else 0.0
    ats = contribution_ats_pp if np.isfinite(contribution_ats_pp) else 0.0
    mae = (
        forecast_mae_contribution_points
        if np.isfinite(forecast_mae_contribution_points)
        else 0.0
    )

    op_positive = roi >= 1.0 and ats > 0
    op_negative = roi <= -1.0 and ats < 0

    forecast_positive = enough_fixed and mae >= 0.05
    forecast_negative = enough_fixed and mae <= -0.05

    season_positive = positive_seasons >= negative_seasons
    season_negative = negative_seasons >= positive_seasons

    if op_positive and forecast_positive and season_positive:
        return "HELPFUL"
    if op_negative and forecast_negative and season_negative:
        return "HARMFUL"

    if (op_positive and forecast_negative) or (op_negative and forecast_positive):
        return "MIXED"

    return "NEUTRAL"


# Preserve the v3.1 implementation name for debugging/reference.
run_contribution_analysis_v3_1 = run_contribution_analysis


def run_contribution_analysis(
    data: pd.DataFrame,
    selected_ids: Iterable[str],
    candidate_add_ids: Iterable[str],
    model_name_map: dict[str, str],
    base_config: StreamlinedBacktestConfig,
    contribution_config: ContributionConfig,
    *,
    progress_callback=None,
) -> dict:
    selected = list(dict.fromkeys(map(str, selected_ids)))
    candidate_add = [
        x
        for x in list(dict.fromkeys(map(str, candidate_add_ids)))
        if x not in set(selected)
    ][: int(contribution_config.max_add_candidates)]

    if len(selected) < int(base_config.min_available_models):
        raise ValueError(
            "Selected pool is smaller than the minimum available-model gate."
        )

    cfg = _analysis_config(base_config, contribution_config)

    total = 1 + len(selected) + len(candidate_add)
    step = 0

    def ping(label):
        nonlocal step
        step += 1
        if progress_callback is not None:
            progress_callback(step, total, label)

    ping("Baseline")
    baseline_result = run_streamlined_backtest(data, selected, cfg)
    baseline_frame = _primary_threshold_frame(
        baseline_result, contribution_config.primary_k
    )
    baseline_summary = _summary_from_frame(baseline_frame)

    rows = []

    def build_row(kind, mid, result):
        variant_frame = _primary_threshold_frame(
            result, contribution_config.primary_k
        )
        variant_detail = result.get("detail", pd.DataFrame()).copy()
        vs = _summary_from_frame(variant_frame)

        raw_roi_delta_pp = (
            100.0 * (vs["roi"] - baseline_summary["roi"])
            if np.isfinite(vs["roi"]) and np.isfinite(baseline_summary["roi"])
            else np.nan
        )
        raw_ats_delta_pp = (
            100.0 * (vs["ats_pct"] - baseline_summary["ats_pct"])
            if np.isfinite(vs["ats_pct"]) and np.isfinite(baseline_summary["ats_pct"])
            else np.nan
        )

        if kind == "remove":
            contribution_roi_pp = -raw_roi_delta_pp if np.isfinite(raw_roi_delta_pp) else np.nan
            contribution_ats_pp = -raw_ats_delta_pp if np.isfinite(raw_ats_delta_pp) else np.nan
        else:
            contribution_roi_pp = raw_roi_delta_pp
            contribution_ats_pp = raw_ats_delta_pp

        fixed = _fixed_baseline_forecast_effect(
            baseline_frame,
            variant_detail,
            kind=kind,
            model_id=mid,
            primary_k=contribution_config.primary_k,
        )
        seasons = _contribution_season_counts(
            baseline_frame,
            variant_frame,
            kind=kind,
        )

        signal = _coverage_aware_signal(
            contribution_roi_pp,
            contribution_ats_pp,
            fixed["forecast_mae_contribution_points"],
            fixed["fixed_game_n"],
            contribution_config.minimum_common_bets,
            seasons["positive_contribution_seasons"],
            seasons["negative_contribution_seasons"],
        )

        # Coverage burden is an explicit diagnostic rather than hidden in score.
        coverage_loss = (
            fixed["baseline_bet_games"] - fixed["fixed_game_n"]
        )

        return {
            "analysis_type": "leave_one_out" if kind == "remove" else "add_one_in",
            "model_id": mid,
            "model_name": model_name_map.get(mid, mid),
            "signal": signal,

            # Operational strategy effect.
            "baseline_bets": baseline_summary["bets"],
            "variant_bets": vs["bets"],
            "delta_bets": int(vs["bets"] - baseline_summary["bets"]),
            "baseline_ats_pct": baseline_summary["ats_pct"],
            "variant_ats_pct": vs["ats_pct"],
            "operational_contribution_ats_pp": contribution_ats_pp,
            "baseline_roi": baseline_summary["roi"],
            "variant_roi": vs["roi"],
            "operational_contribution_roi_pp": contribution_roi_pp,

            # Fixed-game forecast effect.
            **fixed,
            "coverage_lost_baseline_bets": int(coverage_loss),
            **seasons,
        }

    for mid in selected:
        remaining = [x for x in selected if x != mid]
        if len(remaining) < int(base_config.min_available_models):
            continue
        ping(f"Remove {model_name_map.get(mid, mid)}")
        result = run_streamlined_backtest(data, remaining, cfg)
        rows.append(build_row("remove", mid, result))

    for mid in candidate_add:
        ping(f"Add {model_name_map.get(mid, mid)}")
        result = run_streamlined_backtest(
            data, selected + [mid], cfg
        )
        rows.append(build_row("add", mid, result))

    table = pd.DataFrame(rows)
    if not table.empty:
        signal_order = pd.Categorical(
            table["signal"],
            categories=["HELPFUL", "MIXED", "NEUTRAL", "HARMFUL"],
            ordered=True,
        )
        table["_signal_order"] = signal_order
        table = table.sort_values(
            [
                "_signal_order",
                "operational_contribution_roi_pp",
                "forecast_mae_contribution_points",
                "fixed_game_n",
            ],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop(columns="_signal_order").reset_index(drop=True)

    return {
        "baseline_result": baseline_result,
        "baseline_frame": baseline_frame,
        "baseline_summary": baseline_summary,
        "contributions": table,
        "config": contribution_config,
    }


# ---------------------------------------------------------------------------
# v3.3: exact brute-force model-combination search
# ---------------------------------------------------------------------------

from itertools import combinations
import math


@dataclass
class CombinationSearchConfig:
    search_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025)
    validation_seasons: tuple[int, ...] = ()
    # Optional finer-grained chronology filters. When populated they take
    # precedence over the season-only filters above. This lets the UI hold out
    # recent weeks rather than forcing an entire season into validation.
    search_periods: tuple[tuple[int, int], ...] = ()
    validation_periods: tuple[tuple[int, int], ...] = ()
    min_size: int = 4
    max_size: int = 8
    primary_k: float = 1.50
    min_available_models: int = 4
    min_search_bets: int = 50
    min_positive_seasons: int = 1
    ranking_metric: str = "ats"  # ats, wilson, roi
    standard_price: int = -110
    chunk_size: int = 256
    top_n: int = 100
    max_combinations: int = 10000000


def combination_count(n_models: int, min_size: int, max_size: int) -> int:
    lo = max(1, int(min_size))
    hi = min(int(max_size), int(n_models))
    if hi < lo:
        return 0
    return int(sum(math.comb(int(n_models), r) for r in range(lo, hi + 1)))


def _wilson_lower_vec(wins, bets, z=1.96):
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


def _make_combo_matrix(
    data: pd.DataFrame,
    candidate_ids: list[str],
    seasons: tuple[int, ...],
):
    z = data[
        pd.to_numeric(data["season"], errors="coerce")
        .isin(list(map(int, seasons)))
        & data["canonical_model_id"].astype(str).isin(candidate_ids)
    ].copy()

    if z.empty:
        return None

    z["canonical_model_id"] = z["canonical_model_id"].astype(str)
    z["prediction_margin"] = pd.to_numeric(
        z["prediction_margin"], errors="coerce"
    )
    z["market_margin"] = pd.to_numeric(
        z["market_margin"], errors="coerce"
    )
    z["actual_margin"] = pd.to_numeric(
        z["actual_margin"], errors="coerce"
    )
    z["season"] = pd.to_numeric(z["season"], errors="coerce").astype(int)
    z["week"] = pd.to_numeric(z["week"], errors="coerce").astype(int)

    game_meta = (
        z[
            [
                "game_key",
                "season",
                "week",
                "market_margin",
                "actual_margin",
            ]
        ]
        .drop_duplicates("game_key")
        .dropna(subset=["market_margin", "actual_margin"])
        .reset_index(drop=True)
    )
    game_order = game_meta["game_key"].astype(str).tolist()
    game_index = {g: i for i, g in enumerate(game_order)}
    model_index = {m: j for j, m in enumerate(candidate_ids)}

    p = np.full(
        (len(game_order), len(candidate_ids)),
        np.nan,
        dtype=np.float64,
    )

    for row in z[
        ["game_key", "canonical_model_id", "prediction_margin"]
    ].itertuples(index=False):
        gi = game_index.get(str(row.game_key))
        mj = model_index.get(str(row.canonical_model_id))
        if gi is None or mj is None:
            continue
        val = float(row.prediction_margin) if pd.notna(row.prediction_margin) else np.nan
        if np.isfinite(val):
            p[gi, mj] = val

    market = pd.to_numeric(
        game_meta["market_margin"], errors="coerce"
    ).to_numpy(dtype=float)
    actual = pd.to_numeric(
        game_meta["actual_margin"], errors="coerce"
    ).to_numpy(dtype=float)
    cover = actual - market
    season = game_meta["season"].to_numpy(dtype=int)

    return {
        "pred": p,
        "market": market,
        "actual": actual,
        "cover": cover,
        "season": season,
        "game_meta": game_meta,
    }


def _evaluate_combo_indices(
    matrix,
    combo_idx: np.ndarray,
    combo_size: int,
    config: CombinationSearchConfig,
):
    """
    Vectorized evaluation of a CHUNK of combinations of one common size.

    combo_idx shape: [n_combos, combo_size]
    """
    if matrix is None or len(combo_idx) == 0:
        return pd.DataFrame()

    p = matrix["pred"]
    market = matrix["market"]
    cover = matrix["cover"]
    seasons = matrix["season"]

    # [games, combos, r]
    vals = p[:, combo_idx]
    finite = np.isfinite(vals)
    count = finite.sum(axis=2)

    sums = np.nansum(vals, axis=2)
    sqs = np.nansum(vals * vals, axis=2)

    mean = np.divide(
        sums,
        count,
        out=np.full_like(sums, np.nan, dtype=float),
        where=count > 0,
    )

    # unbiased sample variance = (sum(x^2) - sum(x)^2/n)/(n-1)
    numerator = sqs - np.divide(
        sums * sums,
        count,
        out=np.zeros_like(sums, dtype=float),
        where=count > 0,
    )
    var = np.divide(
        numerator,
        count - 1,
        out=np.full_like(sums, np.nan, dtype=float),
        where=count > 1,
    )
    # numerical roundoff can produce tiny negative values.
    var[var < 0] = 0
    sd = np.sqrt(var)

    edge = mean - market[:, None]
    signal = np.divide(
        np.abs(edge),
        sd,
        out=np.full_like(edge, np.nan, dtype=float),
        where=sd > 1e-12,
    )
    zero_sd = np.isfinite(sd) & (sd <= 1e-12)
    signal[zero_sd & (np.abs(edge) > 1e-12)] = np.inf
    signal[zero_sd & (np.abs(edge) <= 1e-12)] = 0.0

    valid = (
        (count >= int(config.min_available_models))
        & (np.isfinite(signal) | np.isinf(signal))
        & (signal >= float(config.primary_k))
    )

    push = valid & (np.abs(cover[:, None]) < 1e-12)
    graded = valid & ~push
    win = graded & ((edge * cover[:, None]) > 0)
    loss = graded & ~win

    wins = win.sum(axis=0).astype(int)
    losses = loss.sum(axis=0).astype(int)
    pushes = push.sum(axis=0).astype(int)
    bets = wins + losses
    ats = np.divide(
        wins,
        bets,
        out=np.full(len(bets), np.nan, dtype=float),
        where=bets > 0,
    )
    win_units = 100.0 / abs(float(config.standard_price))
    units = wins * win_units - losses
    roi = np.divide(
        units,
        bets,
        out=np.full(len(bets), np.nan, dtype=float),
        where=bets > 0,
    )
    wilson = _wilson_lower_vec(wins, bets)

    out = pd.DataFrame({
        "combo_size": int(combo_size),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": ats,
        "units": units,
        "roi": roi,
        "wilson_low": wilson,
    })

    # Season-level stability on the exact same combination.
    unique_seasons = sorted(set(map(int, seasons.tolist())))
    positive = np.zeros(len(combo_idx), dtype=int)
    negative = np.zeros(len(combo_idx), dtype=int)
    seasons_with_bets = np.zeros(len(combo_idx), dtype=int)
    worst_ats = np.full(len(combo_idx), np.nan, dtype=float)

    season_atss = []
    for yy in unique_seasons:
        sm = seasons == yy
        if not np.any(sm):
            continue
        wy = win[sm].sum(axis=0)
        ly = loss[sm].sum(axis=0)
        by = wy + ly
        ay = np.divide(
            wy,
            by,
            out=np.full(len(by), np.nan, dtype=float),
            where=by > 0,
        )
        uy = wy * win_units - ly
        ry = np.divide(
            uy,
            by,
            out=np.full(len(by), np.nan, dtype=float),
            where=by > 0,
        )
        seasons_with_bets += (by > 0).astype(int)
        positive += (ry > 0).astype(int)
        negative += (ry < 0).astype(int)
        season_atss.append(ay)

    if season_atss:
        stack = np.vstack(season_atss)
        finite_any = np.isfinite(stack).any(axis=0)
        if np.any(finite_any):
            worst_ats[finite_any] = np.nanmin(
                stack[:, finite_any], axis=0
            )

    out["seasons_with_bets"] = seasons_with_bets
    out["positive_seasons"] = positive
    out["negative_seasons"] = negative
    out["worst_season_ats"] = worst_ats

    return out


def _evaluate_specific_combos(
    matrix,
    combos: list[tuple[int, ...]],
    config: CombinationSearchConfig,
):
    if matrix is None or not combos:
        return pd.DataFrame()

    rows = []
    # Group by size for vectorized evaluation.
    by_size = {}
    for combo in combos:
        by_size.setdefault(len(combo), []).append(combo)

    for size, group in sorted(by_size.items()):
        for start in range(0, len(group), int(config.chunk_size)):
            chunk = group[start : start + int(config.chunk_size)]
            idx = np.asarray(chunk, dtype=int)
            metrics = _evaluate_combo_indices(
                matrix,
                idx,
                int(size),
                config,
            )
            metrics["_combo_tuple"] = chunk
            rows.append(metrics)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def brute_force_combination_search(
    data: pd.DataFrame,
    candidate_ids: Iterable[str],
    model_name_map: dict[str, str],
    config: CombinationSearchConfig,
    *,
    progress_callback=None,
) -> dict:
    candidate_ids = list(dict.fromkeys(map(str, candidate_ids)))
    n = len(candidate_ids)

    if n < int(config.min_size):
        raise ValueError("Candidate pool is smaller than the minimum combination size.")

    total = combination_count(n, config.min_size, config.max_size)
    if total <= 0:
        raise ValueError("No combinations to evaluate.")
    if total > int(config.max_combinations):
        raise ValueError(
            f"This search contains {total:,} combinations, above the "
            f"configured safety limit of {int(config.max_combinations):,}. "
            "Narrow the combination-size range or candidate universe."
        )

    search_matrix = _make_combo_matrix(
        data,
        candidate_ids,
        tuple(config.search_seasons),
    )
    if search_matrix is None:
        raise ValueError("No historical games found for the selected search seasons.")

    val_matrix = None
    if tuple(config.validation_seasons):
        val_matrix = _make_combo_matrix(
            data,
            candidate_ids,
            tuple(config.validation_seasons),
        )

    all_rows = []
    done = 0

    for size in range(int(config.min_size), int(config.max_size) + 1):
        if size > n:
            break
        group_count = math.comb(n, size)

        iterator = combinations(range(n), size)
        batch = []

        for combo in iterator:
            batch.append(combo)
            if len(batch) >= int(config.chunk_size):
                idx = np.asarray(batch, dtype=int)
                metrics = _evaluate_combo_indices(
                    search_matrix,
                    idx,
                    size,
                    config,
                )
                metrics["_combo_tuple"] = batch
                all_rows.append(metrics)
                done += len(batch)
                if progress_callback is not None:
                    progress_callback(
                        done,
                        total,
                        f"size {size}: {done:,}/{total:,}",
                    )
                batch = []

        if batch:
            idx = np.asarray(batch, dtype=int)
            metrics = _evaluate_combo_indices(
                search_matrix,
                idx,
                size,
                config,
            )
            metrics["_combo_tuple"] = batch
            all_rows.append(metrics)
            done += len(batch)
            if progress_callback is not None:
                progress_callback(
                    done,
                    total,
                    f"size {size}: {done:,}/{total:,}",
                )

    results = pd.concat(all_rows, ignore_index=True)
    if results.empty:
        return {
            "results": results,
            "top": results,
            "size_summary": pd.DataFrame(),
            "config": config,
            "total_combinations": total,
        }

    # Eligibility = confidence gates, NOT optimization objective.
    eligible = results[
        (pd.to_numeric(results["bets"], errors="coerce") >= int(config.min_search_bets))
        & (
            pd.to_numeric(results["positive_seasons"], errors="coerce")
            >= int(config.min_positive_seasons)
        )
    ].copy()

    metric = str(config.ranking_metric)
    if metric == "wilson":
        sort_col = "wilson_low"
    elif metric == "roi":
        sort_col = "roi"
    else:
        sort_col = "ats_pct"

    eligible = eligible.sort_values(
        [sort_col, "bets", "wilson_low"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    eligible["search_rank"] = np.arange(1, len(eligible) + 1)

    def names_for(combo):
        return " | ".join(
            model_name_map.get(candidate_ids[i], candidate_ids[i])
            for i in combo
        )

    def ids_for(combo):
        return "|".join(candidate_ids[i] for i in combo)

    eligible["model_names"] = eligible["_combo_tuple"].apply(names_for)
    eligible["model_ids"] = eligible["_combo_tuple"].apply(ids_for)

    # Validation is NEVER used to sort. Only evaluate top search combinations
    # so validation remains cheap and clearly secondary.
    top = eligible.head(int(config.top_n)).copy()

    if val_matrix is not None and len(top):
        val_combos = top["_combo_tuple"].tolist()
        val_metrics = _evaluate_specific_combos(
            val_matrix,
            val_combos,
            config,
        )
        if len(val_metrics):
            val_metrics = val_metrics[
                [
                    "_combo_tuple",
                    "bets",
                    "wins",
                    "losses",
                    "pushes",
                    "ats_pct",
                    "units",
                    "roi",
                    "wilson_low",
                    "positive_seasons",
                    "negative_seasons",
                    "worst_season_ats",
                ]
            ].copy()
            rename = {
                c: f"validation_{c}"
                for c in val_metrics.columns
                if c != "_combo_tuple"
            }
            val_metrics = val_metrics.rename(columns=rename)
            # tuple keys are awkward in merge; create stable string key.
            top["_combo_key"] = top["_combo_tuple"].apply(
                lambda x: ",".join(map(str, x))
            )
            val_metrics["_combo_key"] = val_metrics["_combo_tuple"].apply(
                lambda x: ",".join(map(str, x))
            )
            top = top.merge(
                val_metrics.drop(columns="_combo_tuple"),
                on="_combo_key",
                how="left",
            ).drop(columns="_combo_key")

    # Best eligible combination at each size.
    if len(eligible):
        size_summary = (
            eligible.sort_values(
                ["combo_size", sort_col, "bets"],
                ascending=[True, False, False],
            )
            .groupby("combo_size", as_index=False)
            .first()
        )
    else:
        size_summary = pd.DataFrame()

    display_cols = [
        "search_rank",
        "combo_size",
        "model_names",
        "bets",
        "wins",
        "losses",
        "ats_pct",
        "roi",
        "wilson_low",
        "positive_seasons",
        "negative_seasons",
        "worst_season_ats",
    ]
    display_cols += [
        c for c in [
            "validation_bets",
            "validation_wins",
            "validation_losses",
            "validation_ats_pct",
            "validation_roi",
            "validation_wilson_low",
            "validation_positive_seasons",
            "validation_negative_seasons",
            "validation_worst_season_ats",
        ]
        if c in top.columns
    ]
    display_cols += ["model_ids"]
    display_cols = [c for c in display_cols if c in top.columns]

    return {
        "results": eligible,
        "top": top[display_cols].copy(),
        "size_summary": size_summary[
            [c for c in display_cols if c in size_summary.columns]
        ].copy(),
        "config": config,
        "total_combinations": total,
        "eligible_combinations": int(len(eligible)),
        "candidate_ids": candidate_ids,
    }


# ===========================================================================
# v3.4 — chronological combination validation + confidence-only coverage gates
# ===========================================================================

@dataclass
class CombinationSearchConfig:
    search_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025)
    validation_seasons: tuple[int, ...] = ()
    # Optional finer-grained chronology filters. When populated they take
    # precedence over the season-only filters above. This lets the UI hold out
    # recent weeks rather than forcing an entire season into validation.
    search_periods: tuple[tuple[int, int], ...] = ()
    validation_periods: tuple[tuple[int, int], ...] = ()
    min_size: int = 4
    max_size: int = 8
    primary_k: float = 1.50
    min_available_models: int = 4

    # Confidence / eligibility gates only. They are never part of the
    # optimization score.
    min_search_bets: int = 50
    min_seasons_represented: int = 1
    min_distinct_weeks: int = 1

    ranking_metric: str = "ats"  # ats, wilson, roi
    standard_price: int = -110
    chunk_size: int = 256
    top_n: int = 100
    max_combinations: int = 10000000


@dataclass
class ChronologicalCombinationConfig:
    season: int = 2025
    folds: tuple[tuple[int, int, int, int], ...] = (
        (5, 8, 9, 10),
        (5, 10, 11, 12),
        (5, 12, 13, 14),
        (5, 14, 15, 16),
    )
    top_n_per_fold: int = 25
    min_train_bets: int = 15
    min_train_weeks: int = 2


def _make_combo_matrix(
    data: pd.DataFrame,
    candidate_ids: list[str],
    seasons: tuple[int, ...],
    periods: tuple[tuple[int, int], ...] = (),
):
    season_num = pd.to_numeric(data["season"], errors="coerce")
    week_num = pd.to_numeric(data["week"], errors="coerce")
    if periods:
        wanted = set((int(y), int(w)) for y, w in periods)
        period_mask = pd.Series(
            [
                (int(y), int(w)) in wanted if pd.notna(y) and pd.notna(w) else False
                for y, w in zip(season_num, week_num)
            ],
            index=data.index,
        )
    else:
        period_mask = season_num.isin(list(map(int, seasons)))

    z = data[
        period_mask
        & data["canonical_model_id"].astype(str).isin(candidate_ids)
    ].copy()

    if z.empty:
        return None

    z["canonical_model_id"] = z["canonical_model_id"].astype(str)
    z["prediction_margin"] = pd.to_numeric(
        z["prediction_margin"], errors="coerce"
    )
    z["market_margin"] = pd.to_numeric(
        z["market_margin"], errors="coerce"
    )
    z["actual_margin"] = pd.to_numeric(
        z["actual_margin"], errors="coerce"
    )
    z["season"] = pd.to_numeric(z["season"], errors="coerce").astype(int)
    z["week"] = pd.to_numeric(z["week"], errors="coerce").astype(int)

    game_meta = (
        z[["game_key", "season", "week", "market_margin", "actual_margin"]]
        .drop_duplicates("game_key")
        .dropna(subset=["market_margin", "actual_margin"])
        .reset_index(drop=True)
    )
    game_order = game_meta["game_key"].astype(str).tolist()
    game_index = {g: i for i, g in enumerate(game_order)}
    model_index = {m: j for j, m in enumerate(candidate_ids)}

    # float32 is more than adequate for point spreads and halves the bandwidth
    # / temporary-memory cost of the million-combination search.
    p = np.full(
        (len(game_order), len(candidate_ids)),
        np.nan,
        dtype=np.float32,
    )

    for row in z[["game_key", "canonical_model_id", "prediction_margin"]].itertuples(index=False):
        gi = game_index.get(str(row.game_key))
        mj = model_index.get(str(row.canonical_model_id))
        if gi is None or mj is None:
            continue
        val = float(row.prediction_margin) if pd.notna(row.prediction_margin) else np.nan
        if np.isfinite(val):
            p[gi, mj] = val

    market = pd.to_numeric(game_meta["market_margin"], errors="coerce").to_numpy(dtype=np.float32)
    actual = pd.to_numeric(game_meta["actual_margin"], errors="coerce").to_numpy(dtype=np.float32)
    cover = actual - market
    season = game_meta["season"].to_numpy(dtype=int)
    week = game_meta["week"].to_numpy(dtype=int)

    finite = np.isfinite(p)
    pred0 = np.where(finite, p, 0.0).astype(np.float32, copy=False)

    return {
        "pred": p,
        "pred0": pred0,
        "predsq": pred0 * pred0,
        "available": finite.astype(np.uint8),
        "market": market,
        "actual": actual,
        "cover": cover,
        "season": season,
        "week": week,
        "game_meta": game_meta,
    }


def _evaluate_combo_indices_fast(
    matrix,
    combo_idx: np.ndarray,
    combo_size: int,
    config: CombinationSearchConfig,
):
    """Fast exact evaluator used during exhaustive screening.

    It computes only the statistics needed to rank/filter the universe. Rich
    season/week diagnostics are recomputed exactly for the bounded finalist
    leaderboard after the exhaustive pass.
    """
    if matrix is None or len(combo_idx) == 0:
        return pd.DataFrame()

    pred0 = matrix.get("pred0")
    predsq = matrix.get("predsq")
    available = matrix.get("available")
    if pred0 is None or predsq is None or available is None:
        p = matrix["pred"]
        finite = np.isfinite(p)
        pred0 = np.where(finite, p, 0.0).astype(np.float32, copy=False)
        predsq = pred0 * pred0
        available = finite.astype(np.uint8)

    market = matrix["market"]
    cover = matrix["cover"]

    # Precomputing zero-filled predictions/squares/availability makes this
    # materially faster than repeated np.isfinite + np.nansum on every chunk.
    count = available[:, combo_idx].sum(axis=2, dtype=np.int16)
    sums = pred0[:, combo_idx].sum(axis=2, dtype=np.float32)
    sqs = predsq[:, combo_idx].sum(axis=2, dtype=np.float32)

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

    valid = (
        (count >= int(config.min_available_models))
        & (np.isfinite(signal) | np.isinf(signal))
        & (signal >= float(config.primary_k))
    )
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
    win_units = 100.0 / abs(float(config.standard_price))
    units = wins * win_units - losses
    roi = np.divide(
        units, bets,
        out=np.full(len(bets), np.nan, dtype=float),
        where=bets > 0,
    )

    return pd.DataFrame({
        "combo_size": int(combo_size),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": ats,
        "units": units,
        "roi": roi,
        "wilson_low": _wilson_lower_vec(wins, bets),
    })


def _evaluate_combo_indices(
    matrix,
    combo_idx: np.ndarray,
    combo_size: int,
    config: CombinationSearchConfig,
):
    if matrix is None or len(combo_idx) == 0:
        return pd.DataFrame()

    p = matrix["pred"]
    market = matrix["market"]
    cover = matrix["cover"]
    seasons = matrix["season"]
    weeks = matrix["week"]

    pred0 = matrix.get("pred0")
    predsq = matrix.get("predsq")
    available = matrix.get("available")
    if pred0 is not None and predsq is not None and available is not None:
        count = available[:, combo_idx].sum(axis=2, dtype=np.int16)
        sums = pred0[:, combo_idx].sum(axis=2, dtype=np.float32)
        sqs = predsq[:, combo_idx].sum(axis=2, dtype=np.float32)
    else:
        vals = p[:, combo_idx]  # [games, combos, combo_size]
        finite = np.isfinite(vals)
        count = finite.sum(axis=2)
        sums = np.nansum(vals, axis=2)
        sqs = np.nansum(vals * vals, axis=2)

    mean = np.divide(
        sums,
        count,
        out=np.full_like(sums, np.nan, dtype=float),
        where=count > 0,
    )

    numerator = sqs - np.divide(
        sums * sums,
        count,
        out=np.zeros_like(sums, dtype=float),
        where=count > 0,
    )
    var = np.divide(
        numerator,
        count - 1,
        out=np.full_like(sums, np.nan, dtype=float),
        where=count > 1,
    )
    var[var < 0] = 0
    sd = np.sqrt(var)

    edge = mean - market[:, None]
    signal = np.divide(
        np.abs(edge),
        sd,
        out=np.full_like(edge, np.nan, dtype=float),
        where=sd > 1e-12,
    )
    zero_sd = np.isfinite(sd) & (sd <= 1e-12)
    signal[zero_sd & (np.abs(edge) > 1e-12)] = np.inf
    signal[zero_sd & (np.abs(edge) <= 1e-12)] = 0.0

    valid = (
        (count >= int(config.min_available_models))
        & (np.isfinite(signal) | np.isinf(signal))
        & (signal >= float(config.primary_k))
    )

    push = valid & (np.abs(cover[:, None]) < 1e-12)
    graded = valid & ~push
    win = graded & ((edge * cover[:, None]) > 0)
    loss = graded & ~win

    wins = win.sum(axis=0).astype(int)
    losses = loss.sum(axis=0).astype(int)
    pushes = push.sum(axis=0).astype(int)
    bets = wins + losses
    ats = np.divide(
        wins,
        bets,
        out=np.full(len(bets), np.nan, dtype=float),
        where=bets > 0,
    )
    win_units = 100.0 / abs(float(config.standard_price))
    units = wins * win_units - losses
    roi = np.divide(
        units,
        bets,
        out=np.full(len(bets), np.nan, dtype=float),
        where=bets > 0,
    )
    wilson = _wilson_lower_vec(wins, bets)

    out = pd.DataFrame({
        "combo_size": int(combo_size),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": ats,
        "units": units,
        "roi": roi,
        "wilson_low": wilson,
    })

    unique_seasons = sorted(set(map(int, seasons.tolist())))
    positive = np.zeros(len(combo_idx), dtype=int)
    negative = np.zeros(len(combo_idx), dtype=int)
    seasons_with_bets = np.zeros(len(combo_idx), dtype=int)
    worst_ats = np.full(len(combo_idx), np.nan, dtype=float)
    season_atss = []

    for yy in unique_seasons:
        sm = seasons == yy
        wy = win[sm].sum(axis=0)
        ly = loss[sm].sum(axis=0)
        by = wy + ly
        ay = np.divide(
            wy,
            by,
            out=np.full(len(by), np.nan, dtype=float),
            where=by > 0,
        )
        uy = wy * win_units - ly
        ry = np.divide(
            uy,
            by,
            out=np.full(len(by), np.nan, dtype=float),
            where=by > 0,
        )
        seasons_with_bets += (by > 0).astype(int)
        positive += (ry > 0).astype(int)
        negative += (ry < 0).astype(int)
        season_atss.append(ay)

    if season_atss:
        stack = np.vstack(season_atss)
        finite_any = np.isfinite(stack).any(axis=0)
        if np.any(finite_any):
            worst_ats[finite_any] = np.nanmin(
                stack[:, finite_any], axis=0
            )

    # Distinct chronology blocks with at least one graded bet. Use
    # (season, week), not week number alone.
    period_keys = list(zip(seasons.tolist(), weeks.tolist()))
    unique_periods = list(dict.fromkeys(period_keys))
    distinct_weeks = np.zeros(len(combo_idx), dtype=int)
    for yy, ww in unique_periods:
        pm = (seasons == int(yy)) & (weeks == int(ww))
        distinct_weeks += graded[pm].any(axis=0).astype(int)

    out["seasons_with_bets"] = seasons_with_bets
    out["distinct_weeks"] = distinct_weeks
    out["positive_seasons"] = positive
    out["negative_seasons"] = negative
    out["worst_season_ats"] = worst_ats
    return out


def _combo_sort_column(config: CombinationSearchConfig) -> str:
    metric = str(config.ranking_metric)
    if metric == "wilson":
        return "wilson_low"
    if metric == "roi":
        return "roi"
    return "ats_pct"


def _combination_frequency_tables(
    top: pd.DataFrame,
    candidate_ids: list[str],
    model_name_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top is None or top.empty:
        return pd.DataFrame(), pd.DataFrame()

    model_rows = []
    pair_rows = []
    denom = len(top)

    for _, row in top.iterrows():
        combo = tuple(row["_combo_tuple"])
        rank = int(row.get("search_rank", 0) or 0)
        ats = float(row.get("ats_pct", np.nan))
        roi = float(row.get("roi", np.nan))

        ids = [candidate_ids[i] for i in combo]
        for mid in ids:
            model_rows.append({
                "canonical_model_id": mid,
                "model_name": model_name_map.get(mid, mid),
                "search_rank": rank,
                "ats_pct": ats,
                "roi": roi,
            })
        for a, b in combinations(sorted(ids), 2):
            pair_rows.append({
                "model_a": a,
                "model_b": b,
                "model_a_name": model_name_map.get(a, a),
                "model_b_name": model_name_map.get(b, b),
                "search_rank": rank,
                "ats_pct": ats,
                "roi": roi,
            })

    md = pd.DataFrame(model_rows)
    pdx = pd.DataFrame(pair_rows)

    if len(md):
        model_freq = (
            md.groupby(
                ["canonical_model_id", "model_name"],
                as_index=False,
            )
            .agg(
                top_combinations=("search_rank", "size"),
                best_rank=("search_rank", "min"),
                mean_rank=("search_rank", "mean"),
                mean_combo_ats=("ats_pct", "mean"),
                mean_combo_roi=("roi", "mean"),
            )
        )
        model_freq["top_frequency"] = (
            model_freq["top_combinations"] / float(denom)
        )
        model_freq = model_freq.sort_values(
            ["top_frequency", "best_rank"],
            ascending=[False, True],
        ).reset_index(drop=True)
    else:
        model_freq = pd.DataFrame()

    if len(pdx):
        pair_freq = (
            pdx.groupby(
                ["model_a", "model_b", "model_a_name", "model_b_name"],
                as_index=False,
            )
            .agg(
                top_combinations=("search_rank", "size"),
                best_rank=("search_rank", "min"),
                mean_rank=("search_rank", "mean"),
                mean_combo_ats=("ats_pct", "mean"),
                mean_combo_roi=("roi", "mean"),
            )
        )
        pair_freq["top_frequency"] = (
            pair_freq["top_combinations"] / float(denom)
        )
        pair_freq = pair_freq.sort_values(
            ["top_frequency", "best_rank"],
            ascending=[False, True],
        ).reset_index(drop=True)
    else:
        pair_freq = pd.DataFrame()

    return model_freq, pair_freq


def _eligible_combo_rows(
    metrics: pd.DataFrame,
    config: CombinationSearchConfig,
) -> pd.DataFrame:
    """Apply confidence-only gates to one evaluated combination batch."""
    if metrics is None or metrics.empty:
        return pd.DataFrame()

    return metrics[
        (
            pd.to_numeric(metrics["bets"], errors="coerce")
            >= int(config.min_search_bets)
        )
        & (
            pd.to_numeric(
                metrics["seasons_with_bets"], errors="coerce"
            )
            >= int(config.min_seasons_represented)
        )
        & (
            pd.to_numeric(
                metrics["distinct_weeks"], errors="coerce"
            )
            >= int(config.min_distinct_weeks)
        )
    ].copy()


def _trim_combo_leaderboard(
    frame: pd.DataFrame,
    config: CombinationSearchConfig,
    limit: int,
) -> pd.DataFrame:
    """Return the exact best `limit` rows under the configured ranking rule."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    sort_col = _combo_sort_column(config)
    out = frame.sort_values(
        [sort_col, "bets", "wilson_low"],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    )
    if int(limit) > 0 and len(out) > int(limit):
        out = out.head(int(limit))
    return out.reset_index(drop=True)


def brute_force_combination_search(
    data: pd.DataFrame,
    candidate_ids: Iterable[str],
    model_name_map: dict[str, str],
    config: CombinationSearchConfig,
    *,
    progress_callback=None,
) -> dict:
    """Exact combination search with bounded memory.

    v3.4.4 keeps the statistical search exhaustive but no longer stores a row
    for every evaluated subset. Each chunk is evaluated exactly, eligibility
    is counted exactly, and only the leading rows needed by the UI / committee
    layer plus the best row for each pool size are retained.

    This changes memory scaling from O(number of combinations) to essentially
    O(chunk size + retained leaderboard size) without changing the ranking
    objective or confidence gates.
    """
    candidate_ids = list(dict.fromkeys(map(str, candidate_ids)))
    n = len(candidate_ids)

    if n < int(config.min_size):
        raise ValueError(
            "Candidate pool is smaller than the minimum combination size."
        )

    total = combination_count(
        n, config.min_size, config.max_size
    )
    if total <= 0:
        raise ValueError("No combinations to evaluate.")
    if total > int(config.max_combinations):
        raise ValueError(
            f"This exact search contains {total:,} combinations, above the "
            f"configured safety limit of "
            f"{int(config.max_combinations):,}."
        )

    search_matrix = _make_combo_matrix(
        data,
        candidate_ids,
        tuple(config.search_seasons),
        tuple(config.search_periods),
    )
    if search_matrix is None:
        raise ValueError(
            "No historical games found for the selected search seasons."
        )

    val_matrix = None
    if tuple(config.validation_seasons) or tuple(config.validation_periods):
        val_matrix = _make_combo_matrix(
            data,
            candidate_ids,
            tuple(config.validation_seasons),
            tuple(config.validation_periods),
        )

    # Keep more than the visible top-N so callers that historically inspected
    # result["results"] still receive a useful leaderboard, while memory stays
    # bounded even for million-combination searches.
    retain_n = max(int(config.top_n), 500)
    leaderboard = pd.DataFrame()
    size_best: dict[int, pd.DataFrame] = {}
    eligible_count = 0
    done = 0

    for size in range(
        int(config.min_size),
        int(config.max_size) + 1,
    ):
        if size > n:
            break
        iterator = combinations(range(n), size)
        batch: list[tuple[int, ...]] = []

        def consume_batch(items: list[tuple[int, ...]]):
            nonlocal leaderboard, eligible_count, done
            if not items:
                return

            idx = np.asarray(items, dtype=int)
            fast_gate = (
                int(config.min_seasons_represented) <= 1
                and int(config.min_distinct_weeks) <= 1
            )
            metrics = (
                _evaluate_combo_indices_fast(search_matrix, idx, size, config)
                if fast_gate
                else _evaluate_combo_indices(search_matrix, idx, size, config)
            )
            if fast_gate:
                # With >= min_search_bets >= 1, these confidence gates are
                # automatically satisfied; rich chronology is recomputed for
                # finalists after the exact pass.
                metrics["seasons_with_bets"] = (metrics["bets"] > 0).astype(int)
                metrics["distinct_weeks"] = (metrics["bets"] > 0).astype(int)
                metrics["positive_seasons"] = np.nan
                metrics["negative_seasons"] = np.nan
                metrics["worst_season_ats"] = np.nan
            metrics["_combo_tuple"] = list(items)

            eligible_batch = _eligible_combo_rows(metrics, config)
            eligible_count += int(len(eligible_batch))

            if len(eligible_batch):
                batch_best = _trim_combo_leaderboard(
                    eligible_batch, config, 1
                )
                prior = size_best.get(int(size))
                if prior is None or prior.empty:
                    size_best[int(size)] = batch_best
                else:
                    size_best[int(size)] = _trim_combo_leaderboard(
                        pd.concat(
                            [prior, batch_best],
                            ignore_index=True,
                        ),
                        config,
                        1,
                    )

                leaderboard = _trim_combo_leaderboard(
                    pd.concat(
                        [leaderboard, eligible_batch],
                        ignore_index=True,
                    ),
                    config,
                    retain_n,
                )

            done += len(items)
            if progress_callback is not None:
                best_txt = "no eligible set yet"
                if leaderboard is not None and len(leaderboard):
                    best = leaderboard.iloc[0]
                    metric_col = _combo_sort_column(config)
                    metric_val = pd.to_numeric(pd.Series([best.get(metric_col)]), errors="coerce").iloc[0]
                    if np.isfinite(metric_val):
                        suffix = "%" if metric_col in {"ats_pct", "roi", "wilson_low"} else ""
                        metric_disp = f"{100.0*float(metric_val):.1f}{suffix}" if suffix else f"{float(metric_val):.3f}"
                        best_txt = f"best {metric_col} {metric_disp} ({int(best.get('bets', 0))} bets)"
                progress_callback(
                    done, total,
                    f"set size {size} · {done:,}/{total:,} · "
                    f"eligible {eligible_count:,} · {best_txt}",
                )

        for combo in iterator:
            batch.append(combo)
            if len(batch) >= int(config.chunk_size):
                consume_batch(batch)
                batch = []

        consume_batch(batch)

    eligible = _trim_combo_leaderboard(
        leaderboard, config, retain_n
    )

    if eligible.empty:
        empty = pd.DataFrame()
        return {
            "results": empty,
            "top": empty,
            "top_internal": empty,
            "size_summary": empty,
            "model_frequency": empty,
            "pair_frequency": empty,
            "config": config,
            "total_combinations": total,
            "evaluated_combinations": done,
            "eligible_combinations": int(eligible_count),
            "retained_combinations": 0,
            "results_truncated": False,
            "search_mode": "exact_streaming",
            "candidate_ids": candidate_ids,
        }

    eligible["search_rank"] = np.arange(1, len(eligible) + 1)

    def names_for(combo):
        return " | ".join(
            model_name_map.get(candidate_ids[i], candidate_ids[i])
            for i in combo
        )

    def ids_for(combo):
        return "|".join(candidate_ids[i] for i in combo)

    eligible["model_names"] = eligible["_combo_tuple"].apply(names_for)
    eligible["model_ids"] = eligible["_combo_tuple"].apply(ids_for)

    # The exhaustive pass intentionally skips expensive per-season/per-week
    # diagnostics. Recompute those exactly only for the bounded retained
    # leaderboard, then preserve the original exact ranking.
    if len(eligible):
        detailed = _evaluate_specific_combos(
            search_matrix, eligible["_combo_tuple"].tolist(), config
        )
        if len(detailed):
            detailed = detailed.copy()
            detailed["_combo_key_tmp"] = detailed["_combo_tuple"].apply(tuple)
            detail_map = detailed.set_index("_combo_key_tmp")
            for col in [
                "seasons_with_bets", "distinct_weeks", "positive_seasons",
                "negative_seasons", "worst_season_ats",
            ]:
                if col in detail_map.columns:
                    eligible[col] = [
                        detail_map.at[tuple(c), col]
                        if tuple(c) in detail_map.index else np.nan
                        for c in eligible["_combo_tuple"]
                    ]

    top_internal = eligible.head(int(config.top_n)).copy()

    if val_matrix is not None and len(top_internal):
        val_combos = top_internal["_combo_tuple"].tolist()
        val_metrics = _evaluate_specific_combos(
            val_matrix, val_combos, config
        )
        if len(val_metrics):
            val_keep = [
                "_combo_tuple",
                "bets",
                "wins",
                "losses",
                "pushes",
                "ats_pct",
                "units",
                "roi",
                "wilson_low",
                "seasons_with_bets",
                "distinct_weeks",
                "positive_seasons",
                "negative_seasons",
                "worst_season_ats",
            ]
            val_metrics = val_metrics[
                [c for c in val_keep if c in val_metrics.columns]
            ].copy()
            rename = {
                c: f"validation_{c}"
                for c in val_metrics.columns
                if c != "_combo_tuple"
            }
            val_metrics = val_metrics.rename(columns=rename)
            top_internal["_combo_key"] = (
                top_internal["_combo_tuple"]
                .apply(lambda x: ",".join(map(str, x)))
            )
            val_metrics["_combo_key"] = (
                val_metrics["_combo_tuple"]
                .apply(lambda x: ",".join(map(str, x)))
            )
            top_internal = top_internal.merge(
                val_metrics.drop(columns="_combo_tuple"),
                on="_combo_key",
                how="left",
            ).drop(columns="_combo_key")

    model_freq, pair_freq = _combination_frequency_tables(
        eligible.head(int(config.top_n)).copy(),
        candidate_ids,
        model_name_map,
    )

    if size_best:
        size_summary = pd.concat(
            [size_best[k] for k in sorted(size_best)],
            ignore_index=True,
        )
        size_summary["model_names"] = size_summary["_combo_tuple"].apply(
            names_for
        )
        size_summary["model_ids"] = size_summary["_combo_tuple"].apply(
            ids_for
        )
        size_summary["size_rank"] = 1
    else:
        size_summary = pd.DataFrame()

    display_cols = [
        "search_rank",
        "combo_size",
        "model_names",
        "bets",
        "wins",
        "losses",
        "ats_pct",
        "roi",
        "wilson_low",
        "seasons_with_bets",
        "distinct_weeks",
        "positive_seasons",
        "negative_seasons",
        "worst_season_ats",
    ]
    display_cols += [
        c for c in top_internal.columns
        if c.startswith("validation_")
    ]
    display_cols += ["model_ids"]
    display_cols = [
        c for c in display_cols if c in top_internal.columns
    ]

    size_display_cols = [
        "combo_size",
        "size_rank",
        "model_names",
        "bets",
        "wins",
        "losses",
        "ats_pct",
        "roi",
        "wilson_low",
        "seasons_with_bets",
        "distinct_weeks",
        "positive_seasons",
        "negative_seasons",
        "worst_season_ats",
        "model_ids",
    ]

    return {
        # Backward-compatible key, intentionally bounded in v3.4.4.
        "results": eligible.copy(),
        "top": top_internal[display_cols].copy(),
        "top_internal": top_internal,
        "size_summary": size_summary[
            [c for c in size_display_cols if c in size_summary.columns]
        ].copy(),
        "model_frequency": model_freq,
        "pair_frequency": pair_freq,
        "config": config,
        "total_combinations": total,
        "evaluated_combinations": done,
        "eligible_combinations": int(eligible_count),
        "retained_combinations": int(len(eligible)),
        "results_truncated": bool(eligible_count > len(eligible)),
        "search_mode": "exact_streaming",
        "candidate_ids": candidate_ids,
    }

def _filter_season_weeks(
    data: pd.DataFrame,
    season: int,
    week_start: int,
    week_end: int,
) -> pd.DataFrame:
    ss = pd.to_numeric(data["season"], errors="coerce")
    ww = pd.to_numeric(data["week"], errors="coerce")
    return data[
        ss.eq(int(season))
        & ww.ge(int(week_start))
        & ww.le(int(week_end))
    ].copy()


def _aggregate_frequency_across_folds(
    fold_top_rows: pd.DataFrame,
    candidate_ids: list[str],
    model_name_map: dict[str, str],
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fold_top_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    model_rows = []
    pair_rows = []
    combo_rows = []

    for row in fold_top_rows.itertuples(index=False):
        ids = [
            x for x in str(getattr(row, "model_ids")).split("|")
            if x
        ]
        fold_id = str(getattr(row, "fold_id"))
        rank = int(getattr(row, "search_rank"))
        combo_key = "|".join(sorted(ids))
        combo_rows.append({
            "combo_key": combo_key,
            "model_names": " | ".join(
                model_name_map.get(x, x) for x in sorted(ids)
            ),
            "fold_id": fold_id,
            "search_rank": rank,
        })
        for mid in ids:
            model_rows.append({
                "canonical_model_id": mid,
                "model_name": model_name_map.get(mid, mid),
                "fold_id": fold_id,
                "search_rank": rank,
                "winner": rank == 1,
            })
        for a, b in combinations(sorted(ids), 2):
            pair_rows.append({
                "model_a": a,
                "model_b": b,
                "model_a_name": model_name_map.get(a, a),
                "model_b_name": model_name_map.get(b, b),
                "fold_id": fold_id,
                "search_rank": rank,
                "winner": rank == 1,
            })

    m = pd.DataFrame(model_rows)
    p = pd.DataFrame(pair_rows)
    c = pd.DataFrame(combo_rows)

    mf = (
        m.groupby(["canonical_model_id", "model_name"], as_index=False)
        .agg(
            topk_appearances=("search_rank", "size"),
            folds_appeared=("fold_id", "nunique"),
            winner_appearances=("winner", "sum"),
            best_fold_rank=("search_rank", "min"),
            mean_fold_rank=("search_rank", "mean"),
        )
    )
    mf["fold_coverage"] = mf["folds_appeared"] / float(max(n_folds, 1))
    mf["winner_frequency"] = (
        mf["winner_appearances"] / float(max(n_folds, 1))
    )
    mf = mf.sort_values(
        ["winner_frequency", "fold_coverage", "topk_appearances"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pf = (
        p.groupby(
            ["model_a", "model_b", "model_a_name", "model_b_name"],
            as_index=False,
        )
        .agg(
            topk_appearances=("search_rank", "size"),
            folds_appeared=("fold_id", "nunique"),
            winner_appearances=("winner", "sum"),
            best_fold_rank=("search_rank", "min"),
            mean_fold_rank=("search_rank", "mean"),
        )
    )
    pf["fold_coverage"] = pf["folds_appeared"] / float(max(n_folds, 1))
    pf["winner_frequency"] = (
        pf["winner_appearances"] / float(max(n_folds, 1))
    )
    pf = pf.sort_values(
        ["winner_frequency", "fold_coverage", "topk_appearances"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    cf = (
        c.groupby(["combo_key", "model_names"], as_index=False)
        .agg(
            topk_appearances=("search_rank", "size"),
            folds_appeared=("fold_id", "nunique"),
            best_fold_rank=("search_rank", "min"),
            mean_fold_rank=("search_rank", "mean"),
        )
        .sort_values(
            ["folds_appeared", "topk_appearances", "best_fold_rank"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    return mf, pf, cf


def run_chronological_combination_validation(
    data: pd.DataFrame,
    candidate_ids: Iterable[str],
    model_name_map: dict[str, str],
    base_config: CombinationSearchConfig,
    chrono_config: ChronologicalCombinationConfig,
) -> dict:
    candidate_ids = list(dict.fromkeys(map(str, candidate_ids)))
    fold_top_frames = []
    fold_summary_rows = []
    winner_validation_rows = []

    for idx, fold in enumerate(chrono_config.folds, start=1):
        train_start, train_end, test_start, test_end = map(int, fold)
        fold_id = (
            f"{chrono_config.season}:"
            f"W{train_start}-{train_end}->W{test_start}-{test_end}"
        )

        train = _filter_season_weeks(
            data,
            chrono_config.season,
            train_start,
            train_end,
        )
        test = _filter_season_weeks(
            data,
            chrono_config.season,
            test_start,
            test_end,
        )

        fold_cfg = CombinationSearchConfig(
            search_seasons=(int(chrono_config.season),),
            validation_seasons=(),
            min_size=int(base_config.min_size),
            max_size=int(base_config.max_size),
            primary_k=float(base_config.primary_k),
            min_available_models=int(base_config.min_available_models),
            min_search_bets=int(chrono_config.min_train_bets),
            min_seasons_represented=1,
            min_distinct_weeks=int(chrono_config.min_train_weeks),
            ranking_metric=str(base_config.ranking_metric),
            standard_price=int(base_config.standard_price),
            chunk_size=int(base_config.chunk_size),
            top_n=int(chrono_config.top_n_per_fold),
            max_combinations=int(base_config.max_combinations),
        )

        search = brute_force_combination_search(
            train,
            candidate_ids,
            model_name_map,
            fold_cfg,
        )

        top_internal = search.get("top_internal", pd.DataFrame()).copy()
        if top_internal.empty:
            fold_summary_rows.append({
                "fold_id": fold_id,
                "train_weeks": f"{train_start}-{train_end}",
                "test_weeks": f"{test_start}-{test_end}",
                "eligible_combinations": 0,
                "winner_models": "",
                "search_bets": 0,
                "search_ats": np.nan,
                "validation_bets": 0,
                "validation_ats": np.nan,
                "validation_roi": np.nan,
            })
            continue

        top_internal = top_internal.head(
            int(chrono_config.top_n_per_fold)
        ).copy()
        top_internal["fold_id"] = fold_id
        top_internal["train_weeks"] = f"{train_start}-{train_end}"
        top_internal["test_weeks"] = f"{test_start}-{test_end}"

        test_matrix = _make_combo_matrix(
            test,
            candidate_ids,
            (int(chrono_config.season),),
        )
        if test_matrix is not None:
            val = _evaluate_specific_combos(
                test_matrix,
                top_internal["_combo_tuple"].tolist(),
                fold_cfg,
            )
        else:
            val = pd.DataFrame()

        if len(val):
            top_internal["_combo_key"] = (
                top_internal["_combo_tuple"]
                .apply(lambda x: ",".join(map(str, x)))
            )
            val["_combo_key"] = (
                val["_combo_tuple"]
                .apply(lambda x: ",".join(map(str, x)))
            )
            val_keep = [
                "_combo_key",
                "bets",
                "wins",
                "losses",
                "pushes",
                "ats_pct",
                "units",
                "roi",
                "wilson_low",
                "distinct_weeks",
            ]
            val = val[[c for c in val_keep if c in val.columns]]
            val = val.rename(
                columns={
                    c: f"validation_{c}"
                    for c in val.columns
                    if c != "_combo_key"
                }
            )
            top_internal = top_internal.merge(
                val, on="_combo_key", how="left"
            ).drop(columns="_combo_key")

        fold_top_frames.append(top_internal)

        winner = top_internal.iloc[0]
        fold_summary_rows.append({
            "fold_id": fold_id,
            "train_weeks": f"{train_start}-{train_end}",
            "test_weeks": f"{test_start}-{test_end}",
            "eligible_combinations": int(
                search.get("eligible_combinations", 0)
            ),
            "winner_models": winner["model_names"],
            "search_bets": int(winner["bets"]),
            "search_ats": float(winner["ats_pct"]),
            "search_roi": float(winner["roi"]),
            "validation_bets": int(
                winner.get("validation_bets", 0)
                if pd.notna(winner.get("validation_bets", np.nan))
                else 0
            ),
            "validation_wins": int(
                winner.get("validation_wins", 0)
                if pd.notna(winner.get("validation_wins", np.nan))
                else 0
            ),
            "validation_losses": int(
                winner.get("validation_losses", 0)
                if pd.notna(winner.get("validation_losses", np.nan))
                else 0
            ),
            "validation_ats": winner.get(
                "validation_ats_pct", np.nan
            ),
            "validation_roi": winner.get(
                "validation_roi", np.nan
            ),
        })

        winner_validation_rows.append({
            "fold_id": fold_id,
            "bets": int(
                winner.get("validation_bets", 0)
                if pd.notna(winner.get("validation_bets", np.nan))
                else 0
            ),
            "wins": int(
                winner.get("validation_wins", 0)
                if pd.notna(winner.get("validation_wins", np.nan))
                else 0
            ),
            "losses": int(
                winner.get("validation_losses", 0)
                if pd.notna(winner.get("validation_losses", np.nan))
                else 0
            ),
        })

    fold_top = (
        pd.concat(fold_top_frames, ignore_index=True)
        if fold_top_frames
        else pd.DataFrame()
    )
    fold_summary = pd.DataFrame(fold_summary_rows)

    mf, pf, cf = _aggregate_frequency_across_folds(
        fold_top,
        candidate_ids,
        model_name_map,
        len(chrono_config.folds),
    )

    winner_oos = pd.DataFrame(winner_validation_rows)
    if len(winner_oos):
        bets = int(winner_oos["bets"].sum())
        wins = int(winner_oos["wins"].sum())
        losses = int(winner_oos["losses"].sum())
        win_units = 100.0 / abs(float(base_config.standard_price))
        units = wins * win_units - losses
        pooled = pd.DataFrame([{
            "folds": len(winner_oos),
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "ats_pct": wins / bets if bets else np.nan,
            "units": units,
            "roi": units / bets if bets else np.nan,
            "wilson_low": (
                float(_wilson_lower_vec([wins], [bets])[0])
                if bets else np.nan
            ),
        }])
    else:
        pooled = pd.DataFrame()

    display_fold_cols = [
        "fold_id",
        "search_rank",
        "combo_size",
        "model_names",
        "bets",
        "ats_pct",
        "roi",
        "wilson_low",
        "distinct_weeks",
        "validation_bets",
        "validation_wins",
        "validation_losses",
        "validation_ats_pct",
        "validation_roi",
        "validation_wilson_low",
        "model_ids",
    ]
    if len(fold_top):
        fold_top_display = fold_top[
            [c for c in display_fold_cols if c in fold_top.columns]
        ].copy()
    else:
        fold_top_display = pd.DataFrame()

    return {
        "fold_summary": fold_summary,
        "fold_top": fold_top_display,
        "model_frequency": mf,
        "pair_frequency": pf,
        "combination_frequency": cf,
        "pooled_fold_winner_oos": pooled,
        "config": chrono_config,
    }

# ===========================================================================
# v3.5 — streamlined four-page app helpers
# ===========================================================================

def individual_model_performance(
    data: pd.DataFrame,
    *,
    standard_price: int = -110,
) -> dict:
    """Summarize each canonical model as a standalone spread forecaster/bettor.

    Every non-zero model-vs-market edge is treated as that model's ATS side.
    This is descriptive history for Page 1; it is not used as an outcome gate
    inside combination discovery.
    """
    if data is None or data.empty:
        return {"overall": pd.DataFrame(), "by_season": pd.DataFrame()}

    cols = [
        "season", "week", "game_key", "canonical_model_id", "model_name",
        "prediction_margin", "market_margin", "actual_margin",
    ]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Historical data missing columns: {missing}")

    x = data[cols].copy()
    x["canonical_model_id"] = x["canonical_model_id"].astype(str)
    x["model_name"] = x["model_name"].astype(str)
    for c in ["season", "week", "prediction_margin", "market_margin", "actual_margin"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Canonical data should already be one row/model/game, but protect against
    # source-level duplicates so Page 1 cannot double-count a game.
    x = x.sort_values(["season", "week", "game_key"]).drop_duplicates(
        ["canonical_model_id", "game_key"], keep="first"
    )
    x["edge"] = x["prediction_margin"] - x["market_margin"]
    x["cover"] = x["actual_margin"] - x["market_margin"]
    x["forecast_error"] = x["prediction_margin"] - x["actual_margin"]

    valid = (
        np.isfinite(x["edge"])
        & np.isfinite(x["cover"])
        & np.isfinite(x["prediction_margin"])
    )
    x = x.loc[valid].copy()
    x["push"] = x["cover"].abs() < 1e-12
    x["bet"] = x["edge"].abs() >= 1e-12
    x["graded"] = x["bet"] & ~x["push"]
    x["win"] = x["graded"] & ((x["edge"] * x["cover"]) > 0)
    x["loss"] = x["graded"] & ~x["win"]

    win_units = 100.0 / abs(float(standard_price)) if standard_price < 0 else float(standard_price) / 100.0
    x["unit_result"] = np.where(x["win"], win_units, np.where(x["loss"], -1.0, 0.0))

    def summarize(group_cols):
        g = (
            x.groupby(group_cols, dropna=False)
            .agg(
                predictions=("game_key", "size"),
                bets=("graded", "sum"),
                wins=("win", "sum"),
                losses=("loss", "sum"),
                pushes=("push", "sum"),
                units=("unit_result", "sum"),
                mae=("forecast_error", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
                bias=("forecast_error", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce")))),
            )
            .reset_index()
        )
        for c in ["bets", "wins", "losses", "pushes", "predictions"]:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(int)
        g["ats_pct"] = np.divide(
            g["wins"], g["bets"],
            out=np.full(len(g), np.nan, dtype=float),
            where=g["bets"].to_numpy() > 0,
        )
        g["roi"] = np.divide(
            g["units"], g["bets"],
            out=np.full(len(g), np.nan, dtype=float),
            where=g["bets"].to_numpy() > 0,
        )
        # 95% Wilson lower bound for standalone ATS record.
        n = g["bets"].to_numpy(dtype=float)
        p = np.divide(
            g["wins"].to_numpy(dtype=float), n,
            out=np.full(len(g), np.nan, dtype=float),
            where=n > 0,
        )
        z = 1.959963984540054
        denom = 1.0 + z * z / np.where(n > 0, n, 1.0)
        center = p + z * z / (2.0 * np.where(n > 0, n, 1.0))
        adj = z * np.sqrt(
            np.divide(
                p * (1.0 - p), np.where(n > 0, n, 1.0)
            ) + z * z / (4.0 * np.where(n > 0, n, 1.0) ** 2)
        )
        low = (center - adj) / denom
        low[n <= 0] = np.nan
        g["wilson_low"] = low
        return g

    overall = summarize(["canonical_model_id", "model_name"])
    seasons = (
        x.groupby(["canonical_model_id", "model_name"])["season"]
        .nunique()
        .rename("seasons")
        .reset_index()
    )
    overall = overall.merge(
        seasons, on=["canonical_model_id", "model_name"], how="left"
    )
    overall = overall.sort_values(
        ["wilson_low", "bets", "ats_pct"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    overall["rank"] = np.arange(1, len(overall) + 1)

    by_season = summarize(
        ["season", "canonical_model_id", "model_name"]
    ).sort_values(
        ["season", "ats_pct", "bets"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return {"overall": overall, "by_season": by_season}



def combination_spread_scale_performance(
    data: pd.DataFrame,
    search_result: dict,
    *,
    discovery_periods: Iterable[tuple[int, int]] = (),
    validation_periods: Iterable[tuple[int, int]] = (),
    top_n: int = 25,
    buckets: Iterable[tuple[str, float, float | None]] = (
        ("0–3.5", 0.0, 3.5),
        ("4–7.5", 3.5, 7.5),
        ("8–14.5", 7.5, 14.5),
        ("15–21.5", 14.5, 21.5),
        ("22+", 21.5, None),
    ),
) -> pd.DataFrame:
    """Evaluate frozen finalist combinations by absolute market-spread scale.

    This is intentionally a diagnostic, not another optimization dimension.
    Finalists remain ranked by the configured discovery objective; the table
    shows whether their historical signal is concentrated in small, medium,
    or very large point spreads at the already-chosen primary k threshold.
    """
    if not search_result:
        return pd.DataFrame()

    top = search_result.get("top_internal", pd.DataFrame())
    candidate_ids = list(map(str, search_result.get("candidate_ids", [])))
    base: CombinationSearchConfig = search_result.get("config")
    if top is None or top.empty or not candidate_ids or base is None:
        return pd.DataFrame()

    frozen = top.head(int(top_n)).copy()
    period_sets = [
        ("Discovery", tuple((int(y), int(w)) for y, w in discovery_periods)),
        ("Holdout", tuple((int(y), int(w)) for y, w in validation_periods)),
    ]
    bucket_specs = list(buckets)
    rows = []
    win_units = 100.0 / abs(float(base.standard_price))

    for period_label, periods in period_sets:
        if not periods:
            continue
        seasons = tuple(sorted({int(y) for y, _ in periods}))
        matrix = _make_combo_matrix(data, candidate_ids, seasons, periods)
        if matrix is None:
            continue

        p = matrix["pred"]
        market = np.asarray(matrix["market"], dtype=float)
        cover = np.asarray(matrix["cover"], dtype=float)
        abs_market = np.abs(market)

        for _, meta in frozen.iterrows():
            combo = tuple(meta.get("_combo_tuple", ()))
            if not combo:
                continue
            vals = p[:, np.asarray(combo, dtype=int)]
            finite = np.isfinite(vals)
            count = finite.sum(axis=1)
            vals0 = np.where(finite, vals, 0.0)
            sums = vals0.sum(axis=1, dtype=float)
            sqs = (vals0 * vals0).sum(axis=1, dtype=float)
            mean = np.divide(
                sums, count,
                out=np.full(len(count), np.nan, dtype=float),
                where=count > 0,
            )
            numerator = sqs - np.divide(
                sums * sums, count,
                out=np.zeros(len(count), dtype=float),
                where=count > 0,
            )
            var = np.divide(
                numerator, count - 1,
                out=np.full(len(count), np.nan, dtype=float),
                where=count > 1,
            )
            var[var < 0] = 0.0
            sd = np.sqrt(var)
            edge = mean - market
            signal = np.divide(
                np.abs(edge), sd,
                out=np.full(len(edge), np.nan, dtype=float),
                where=sd > 1e-12,
            )
            zero_sd = np.isfinite(sd) & (sd <= 1e-12)
            signal[zero_sd & (np.abs(edge) > 1e-12)] = np.inf
            signal[zero_sd & (np.abs(edge) <= 1e-12)] = 0.0

            scorable = (
                (count >= int(base.min_available_models))
                & (np.isfinite(signal) | np.isinf(signal))
            )
            qualifies = scorable & (signal >= float(base.primary_k))
            pushes = qualifies & (np.abs(cover) < 1e-12)
            graded = qualifies & ~pushes
            wins = graded & ((edge * cover) > 0)
            losses = graded & ~wins

            for bucket_order, (label, lower, upper) in enumerate(bucket_specs, start=1):
                if upper is None:
                    in_bucket = abs_market > float(lower)
                elif float(lower) <= 0:
                    in_bucket = abs_market <= float(upper)
                else:
                    in_bucket = (abs_market > float(lower)) & (abs_market <= float(upper))

                scorable_n = int((scorable & in_bucket).sum())
                win_n = int((wins & in_bucket).sum())
                loss_n = int((losses & in_bucket).sum())
                push_n = int((pushes & in_bucket).sum())
                bet_n = win_n + loss_n
                ats = win_n / bet_n if bet_n else np.nan
                units = win_n * win_units - loss_n
                roi = units / bet_n if bet_n else np.nan
                wilson = _wilson_lower_vec([win_n], [bet_n])[0] if bet_n else np.nan
                bet_mask = qualifies & in_bucket
                mean_abs_edge = float(np.nanmean(np.abs(edge[bet_mask]))) if np.any(bet_mask) else np.nan
                mean_abs_market = float(np.nanmean(abs_market[in_bucket])) if np.any(in_bucket) else np.nan

                rows.append({
                    "search_rank": int(meta.get("search_rank", len(rows) + 1)),
                    "model_names": meta.get("model_names", ""),
                    "combo_size": int(meta.get("combo_size", len(combo))),
                    "period": period_label,
                    "bucket_order": int(bucket_order),
                    "line_bucket": label,
                    "scorable_games": scorable_n,
                    "bets": bet_n,
                    "wins": win_n,
                    "losses": loss_n,
                    "pushes": push_n,
                    "bet_rate": (bet_n / scorable_n) if scorable_n else np.nan,
                    "ats_pct": ats,
                    "roi": roi,
                    "wilson_low": wilson,
                    "mean_abs_edge": mean_abs_edge,
                    "mean_abs_market": mean_abs_market,
                    "k": float(base.primary_k),
                })

    return pd.DataFrame(rows)

def combination_threshold_robustness(
    data: pd.DataFrame,
    search_result: dict,
    *,
    seasons: Iterable[int] | None = None,
    periods: Iterable[tuple[int, int]] | None = None,
    thresholds: Iterable[float] = (
        0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00
    ),
    top_n: int = 25,
) -> dict:
    """Re-evaluate already-discovered finalists across a small k grid.

    Importantly, this does *not* rerank the exhaustive universe independently
    at every k. It is a robustness diagnostic on frozen finalists, reducing
    threshold-selection overfit while exposing low-k behavior down to 0.25 SD.
    """
    if not search_result:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame()}

    top = search_result.get("top_internal", pd.DataFrame())
    if top is None or top.empty:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame()}

    candidate_ids = list(map(str, search_result.get("candidate_ids", [])))
    base: CombinationSearchConfig = search_result.get("config")
    if not candidate_ids or base is None:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame()}

    if seasons is None:
        seasons = tuple(sorted(set(base.search_seasons) | set(base.validation_seasons)))
    seasons = tuple(sorted(set(map(int, seasons))))
    if periods is None:
        periods = tuple(
            sorted(set(tuple(x) for x in base.search_periods) | set(tuple(x) for x in base.validation_periods))
        )
    periods = tuple((int(y), int(w)) for y, w in periods)
    matrix = _make_combo_matrix(data, candidate_ids, seasons, periods)
    if matrix is None:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame()}

    frozen = top.head(int(top_n)).copy()
    combos = frozen["_combo_tuple"].tolist()
    # _evaluate_specific_combos already returns combo_size; keep only identity
    # columns here to avoid merge suffixes such as combo_size_x/combo_size_y.
    meta = frozen[[c for c in ["search_rank", "model_names", "model_ids"] if c in frozen.columns]].copy()
    meta["_combo_key"] = frozen["_combo_tuple"].apply(lambda x: ",".join(map(str, x)))

    rows = []
    for k in sorted(set(float(v) for v in thresholds)):
        cfg = CombinationSearchConfig(
            search_seasons=seasons,
            validation_seasons=(),
            search_periods=periods,
            validation_periods=(),
            min_size=int(base.min_size),
            max_size=int(base.max_size),
            primary_k=float(k),
            min_available_models=int(base.min_available_models),
            min_search_bets=0,
            min_seasons_represented=1,
            min_distinct_weeks=1,
            ranking_metric=str(base.ranking_metric),
            standard_price=int(base.standard_price),
            chunk_size=int(base.chunk_size),
            top_n=int(top_n),
            max_combinations=int(base.max_combinations),
        )
        m = _evaluate_specific_combos(matrix, combos, cfg)
        if m.empty:
            continue
        m["_combo_key"] = m["_combo_tuple"].apply(lambda x: ",".join(map(str, x)))
        m = m.merge(meta, on="_combo_key", how="left")
        m["k"] = float(k)
        rows.append(m)

    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if detail.empty:
        return {"detail": detail, "summary": pd.DataFrame()}

    summary = (
        detail.groupby(["search_rank", "model_names", "model_ids", "combo_size"], as_index=False)
        .agg(
            thresholds_tested=("k", "nunique"),
            min_bets=("bets", "min"),
            max_bets=("bets", "max"),
            mean_ats=("ats_pct", "mean"),
            min_ats=("ats_pct", "min"),
            max_ats=("ats_pct", "max"),
            mean_roi=("roi", "mean"),
            min_roi=("roi", "min"),
            max_roi=("roi", "max"),
            profitable_thresholds=("roi", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .sort_values(["profitable_thresholds", "mean_ats", "min_bets"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return {"detail": detail, "summary": summary}
