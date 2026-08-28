from __future__ import annotations

from collections import Counter
from typing import Iterable
import math

import numpy as np
import pandas as pd


DEFAULT_K_GRID = tuple(np.round(np.arange(0.25, 2.01, 0.25), 2))


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return np.nan
    p = float(wins) / float(n)
    den = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    adj = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (center - adj) / den


def _unit_result(win: np.ndarray, loss: np.ndarray, price: int = -110) -> np.ndarray:
    out = np.zeros(len(win), dtype=float)
    if price < 0:
        win_unit = 100.0 / abs(float(price))
    else:
        win_unit = float(price) / 100.0
    out[win] = win_unit
    out[loss] = -1.0
    return out


def _period_mask(df: pd.DataFrame, periods: Iterable[tuple[int, int]]) -> pd.Series:
    wanted = {(int(y), int(w)) for y, w in periods}
    if not wanted:
        return pd.Series(False, index=df.index)
    return pd.Series(
        [(int(y), int(w)) in wanted for y, w in zip(df["season"], df["week"])],
        index=df.index,
    )


def _matrix_and_meta(
    data: pd.DataFrame,
    model_ids: Iterable[str],
    periods: Iterable[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mids = list(dict.fromkeys(map(str, model_ids)))
    if not mids:
        return pd.DataFrame(), pd.DataFrame()
    d = data[data["canonical_model_id"].astype(str).isin(mids)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d[d["season"].notna() & d["week"].notna()].copy()
    d["season"] = d["season"].astype(int)
    d["week"] = d["week"].astype(int)
    d = d[_period_mask(d, periods)]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    pred = (
        d.pivot_table(
            index="game_key",
            columns="canonical_model_id",
            values="prediction_margin",
            aggfunc="first",
        )
        .reindex(columns=mids)
    )
    meta_cols = [
        "game_key", "season", "week", "market_margin", "actual_margin",
        "team_a_id", "team_b_id", "road", "home",
    ]
    meta = (
        d[[c for c in meta_cols if c in d.columns]]
        .drop_duplicates("game_key")
        .set_index("game_key")
        .reindex(pred.index)
    )
    market = pd.to_numeric(meta.get("market_margin"), errors="coerce")
    actual = pd.to_numeric(meta.get("actual_margin"), errors="coerce")
    valid = market.notna() & actual.notna()
    pred = pred.loc[valid]
    meta = meta.loc[valid].copy()
    if meta.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta["market_margin"] = pd.to_numeric(meta["market_margin"], errors="coerce")
    meta["actual_margin"] = pd.to_numeric(meta["actual_margin"], errors="coerce")
    order = (
        meta.reset_index()
        .sort_values(["season", "week", "game_key"])["game_key"]
        .tolist()
    )
    return pred.reindex(order), meta.reindex(order)


def _combo_forecast_arrays(
    pred: pd.DataFrame,
    model_ids: Iterable[str],
    min_available_models: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [str(x) for x in model_ids if str(x) in pred.columns]
    n_games = len(pred)
    if not ids:
        return (
            np.zeros(n_games, dtype=np.int16),
            np.full(n_games, np.nan),
            np.full(n_games, np.nan),
        )
    arr = pred[ids].to_numpy(dtype=float)
    available = np.isfinite(arr)
    count = available.sum(axis=1).astype(np.int16)
    sums = np.nansum(arr, axis=1)
    mean = np.divide(
        sums,
        count,
        out=np.full(n_games, np.nan, dtype=float),
        where=count > 0,
    )
    sd = np.full(n_games, np.nan, dtype=float)
    enough2 = count >= 2
    if enough2.any():
        sq = np.nansum(np.square(arr), axis=1)
        var = np.divide(
            sq - np.divide(np.square(sums), count, out=np.zeros(n_games), where=count > 0),
            count - 1,
            out=np.full(n_games, np.nan),
            where=count > 1,
        )
        var = np.maximum(var, 0.0)
        sd[enough2] = np.sqrt(var[enough2])
    scorable = count >= min(int(min_available_models), max(1, len(ids)))
    mean[~scorable] = np.nan
    sd[~scorable] = np.nan
    return count, mean, sd


def _signal(edge: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.full(len(edge), np.nan, dtype=float)
    finite = np.isfinite(edge) & np.isfinite(sd)
    pos = finite & (sd > 1e-12)
    out[pos] = np.abs(edge[pos]) / sd[pos]
    zero = finite & ~pos
    out[zero & (np.abs(edge) > 1e-12)] = np.inf
    out[zero & (np.abs(edge) <= 1e-12)] = 0.0
    return out


def _threshold_stats(
    edge: np.ndarray,
    signal: np.ndarray,
    cover: np.ndarray,
    k: float,
    *,
    standard_price: int = -110,
    extra_gate: np.ndarray | None = None,
) -> dict:
    decision = (np.isfinite(signal) | np.isinf(signal)) & (signal >= float(k))
    if extra_gate is not None:
        decision &= extra_gate.astype(bool)
    valid = decision & np.isfinite(edge) & np.isfinite(cover) & (np.abs(cover) > 1e-12)
    win = valid & ((edge * cover) > 0)
    loss = valid & ((edge * cover) < 0)
    push = decision & np.isfinite(cover) & (np.abs(cover) <= 1e-12)
    wins = int(win.sum())
    losses = int(loss.sum())
    pushes = int(push.sum())
    bets = wins + losses
    units_vec = _unit_result(win, loss, standard_price)
    units = float(units_vec.sum())
    return {
        "k": float(k),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": (wins / bets) if bets else np.nan,
        "units": units,
        "roi": (units / bets) if bets else np.nan,
        "wilson_low": _wilson_lower(wins, bets),
    }


def _choose_stable_k(profile: pd.DataFrame, min_bets: int) -> dict:
    if profile is None or profile.empty:
        return {"selected_k": np.nan, "selection_score": np.nan}
    p = profile.sort_values("k").reset_index(drop=True).copy()
    p["eligible_k"] = pd.to_numeric(p["bets"], errors="coerce").fillna(0).ge(int(min_bets))
    floors = []
    means = []
    for i in range(len(p)):
        lo = max(0, i - 1)
        hi = min(len(p), i + 2)
        window = p.iloc[lo:hi]
        vals = pd.to_numeric(window.loc[window["eligible_k"], "wilson_low"], errors="coerce").dropna()
        floors.append(float(vals.min()) if len(vals) >= 2 else np.nan)
        means.append(float(vals.mean()) if len(vals) >= 2 else np.nan)
    p["neighbor_wilson_floor"] = floors
    p["neighbor_wilson_mean"] = means
    cand = p[p["eligible_k"] & p["neighbor_wilson_floor"].notna()].copy()
    if cand.empty:
        cand = p[p["eligible_k"]].copy()
        if cand.empty:
            cand = p[p["bets"].gt(0)].copy()
        if cand.empty:
            return {"selected_k": np.nan, "selection_score": np.nan, "profile": p}
        cand["neighbor_wilson_floor"] = cand["wilson_low"]
        cand["neighbor_wilson_mean"] = cand["wilson_low"]
    cand["distance_from_anchor"] = (pd.to_numeric(cand["k"]) - 0.50).abs()
    cand = cand.sort_values(
        ["neighbor_wilson_floor", "neighbor_wilson_mean", "wilson_low", "bets", "distance_from_anchor", "k"],
        ascending=[False, False, False, False, True, True],
        na_position="last",
    )
    row = cand.iloc[0]
    return {
        "selected_k": float(row["k"]),
        "selection_score": float(row["neighbor_wilson_floor"]),
        "profile": p,
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa = set(map(str, a)); bb = set(map(str, b))
    u = aa | bb
    return float(len(aa & bb) / len(u)) if u else 0.0


def assign_overlap_communities(
    combinations: list[dict],
    threshold: float = 0.60,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict]:
    """Greedy rank-ordered communities using the best-ranked member as representative.

    This avoids graph chaining: a combo joins a community only when it overlaps
    the community's highest-ranked representative by at least `threshold`.
    """
    combos = []
    for i, c in enumerate(combinations, start=1):
        ids = list(dict.fromkeys(map(str, c.get("model_ids", []))))
        if ids:
            cc = dict(c)
            cc["rank"] = int(cc.get("rank", i))
            cc["model_ids"] = ids
            cc["portfolio_combo"] = i
            combos.append(cc)
    combos.sort(key=lambda x: (int(x.get("rank", 10**9)), int(x.get("portfolio_combo", 10**9))))

    communities: list[dict] = []
    for c in combos:
        scores = [jaccard(c["model_ids"], q["representative_ids"]) for q in communities]
        if scores and max(scores) >= float(threshold):
            idx = int(np.argmax(scores))
            community = communities[idx]
            c["community"] = int(community["community"])
            c["jaccard_to_representative"] = float(scores[idx])
            community["members"].append(c)
        else:
            cid = len(communities) + 1
            c["community"] = cid
            c["jaccard_to_representative"] = 1.0
            communities.append({
                "community": cid,
                "representative_rank": int(c["rank"]),
                "representative_ids": list(c["model_ids"]),
                "members": [c],
            })

    # Restore user's C1/C2 order, not rank-sort order.
    combos.sort(key=lambda x: int(x.get("portfolio_combo", 10**9)))

    rows = []
    for c in combos:
        rows.append({
            "combo": f"C{int(c['portfolio_combo'])}",
            "portfolio_combo": int(c["portfolio_combo"]),
            "search_rank": int(c["rank"]),
            "community": int(c["community"]),
            "combo_size": len(c["model_ids"]),
            "jaccard_to_representative": float(c["jaccard_to_representative"]),
            "model_ids": "|".join(c["model_ids"]),
        })
    community_table = pd.DataFrame(rows)

    pair_rows = []
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            pair_rows.append({
                "combo_a": f"C{int(combos[i]['portfolio_combo'])}",
                "combo_b": f"C{int(combos[j]['portfolio_combo'])}",
                "jaccard": jaccard(combos[i]["model_ids"], combos[j]["model_ids"]),
                "same_community": int(combos[i]["community"]) == int(combos[j]["community"]),
            })
    pairwise = pd.DataFrame(pair_rows)
    counts = Counter(mid for c in combos for mid in c["model_ids"])
    model_freq = pd.DataFrame(
        [{"canonical_model_id": mid, "combo_count": n, "combo_share": n / len(combos)} for mid, n in counts.items()]
    ).sort_values(["combo_count", "canonical_model_id"], ascending=[False, True]) if counts else pd.DataFrame()
    summary = {
        "raw_combos": len(combos),
        "communities": len(communities),
        "unique_models": len(counts),
        "mean_pairwise_jaccard": float(pairwise["jaccard"].mean()) if len(pairwise) else np.nan,
        "max_pairwise_jaccard": float(pairwise["jaccard"].max()) if len(pairwise) else np.nan,
        "overlap_threshold": float(threshold),
    }
    return combos, community_table, pairwise, {"summary": summary, "model_frequency": model_freq}


def tune_combination_thresholds(
    data: pd.DataFrame,
    combinations: list[dict],
    periods: Iterable[tuple[int, int]],
    *,
    min_available_models: int = 4,
    thresholds: Iterable[float] = DEFAULT_K_GRID,
    min_bets: int = 50,
    standard_price: int = -110,
) -> dict:
    combos = [dict(c) for c in combinations]
    union = list(dict.fromkeys(mid for c in combos for mid in map(str, c.get("model_ids", []))))
    pred, meta = _matrix_and_meta(data, union, periods)
    if pred.empty:
        return {"combinations": combos, "detail": pd.DataFrame(), "selected": pd.DataFrame()}
    market = meta["market_margin"].to_numpy(dtype=float)
    cover = meta["actual_margin"].to_numpy(dtype=float) - market
    detail_rows = []
    selected_rows = []
    for i, c in enumerate(combos, start=1):
        _, mean, sd = _combo_forecast_arrays(pred, c.get("model_ids", []), min_available_models)
        edge = mean - market
        sig = _signal(edge, sd)
        prof = pd.DataFrame([
            {
                **_threshold_stats(edge, sig, cover, float(k), standard_price=standard_price),
                "portfolio_combo": i,
                "search_rank": int(c.get("rank", i)),
            }
            for k in thresholds
        ])
        choice = _choose_stable_k(prof, min_bets=min_bets)
        chosen = float(choice.get("selected_k", np.nan))
        if not np.isfinite(chosen):
            chosen = float(c.get("k", 0.50))
        c["k"] = chosen
        prof = choice.get("profile", prof)
        prof["portfolio_combo"] = i
        prof["search_rank"] = int(c.get("rank", i))
        prof["selected"] = np.isclose(pd.to_numeric(prof["k"], errors="coerce"), chosen)
        detail_rows.append(prof)
        rr = prof[prof["selected"]].iloc[0] if prof["selected"].any() else prof.iloc[0]
        selected_rows.append({
            "portfolio_combo": i,
            "search_rank": int(c.get("rank", i)),
            "selected_k": chosen,
            "discovery_bets": int(rr.get("bets", 0)),
            "discovery_ats_pct": rr.get("ats_pct", np.nan),
            "discovery_roi": rr.get("roi", np.nan),
            "discovery_wilson_low": rr.get("wilson_low", np.nan),
            "neighbor_wilson_floor": rr.get("neighbor_wilson_floor", np.nan),
        })
    return {
        "combinations": combos,
        "detail": pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(),
        "selected": pd.DataFrame(selected_rows),
    }


def evaluate_selected_combo_thresholds(
    data: pd.DataFrame,
    combinations: list[dict],
    periods: Iterable[tuple[int, int]],
    *,
    min_available_models: int = 4,
    standard_price: int = -110,
) -> pd.DataFrame:
    periods = tuple(periods)
    if not periods or not combinations:
        return pd.DataFrame()
    union = list(dict.fromkeys(mid for c in combinations for mid in map(str, c.get("model_ids", []))))
    pred, meta = _matrix_and_meta(data, union, periods)
    if pred.empty:
        return pd.DataFrame()
    market = meta["market_margin"].to_numpy(dtype=float)
    cover = meta["actual_margin"].to_numpy(dtype=float) - market
    rows = []
    for i, c in enumerate(combinations, start=1):
        _, mean, sd = _combo_forecast_arrays(pred, c.get("model_ids", []), min_available_models)
        edge = mean - market
        sig = _signal(edge, sd)
        k = float(c.get("k", 0.50))
        st = _threshold_stats(edge, sig, cover, k, standard_price=standard_price)
        rows.append({
            "portfolio_combo": i,
            "search_rank": int(c.get("rank", i)),
            "holdout_k": k,
            "holdout_bets": int(st["bets"]),
            "holdout_wins": int(st["wins"]),
            "holdout_losses": int(st["losses"]),
            "holdout_ats_pct": st["ats_pct"],
            "holdout_roi": st["roi"],
            "holdout_wilson_low": st["wilson_low"],
        })
    return pd.DataFrame(rows)


def _meta_game_frame(
    data: pd.DataFrame,
    combinations: list[dict],
    periods: Iterable[tuple[int, int]],
    *,
    min_available_models: int,
    diversified: bool,
) -> pd.DataFrame:
    union = list(dict.fromkeys(mid for c in combinations for mid in map(str, c.get("model_ids", []))))
    pred, meta = _matrix_and_meta(data, union, periods)
    if pred.empty:
        return pd.DataFrame()
    forecasts = []
    for i, c in enumerate(combinations, start=1):
        _, mean, sd = _combo_forecast_arrays(pred, c.get("model_ids", []), min_available_models)
        forecasts.append({
            "combo": i,
            "community": int(c.get("community", i)),
            "mean": mean,
            "sd": sd,
        })
    rows = []
    for gi, game_key in enumerate(pred.index):
        active = []
        for f in forecasts:
            mu = float(f["mean"][gi]) if np.isfinite(f["mean"][gi]) else np.nan
            sd = float(f["sd"][gi]) if np.isfinite(f["sd"][gi]) else np.nan
            if np.isfinite(mu) and np.isfinite(sd):
                active.append((f["community"], f["combo"], mu, sd))
        if not active:
            continue

        if diversified:
            units = []
            for community in sorted({x[0] for x in active}):
                z = [x for x in active if x[0] == community]
                means = np.array([x[2] for x in z], dtype=float)
                sds = np.array([x[3] for x in z], dtype=float)
                cmean = float(np.mean(means))
                within = float(np.mean(np.square(sds))) if len(sds) else np.nan
                between = float(np.var(means, ddof=1)) if len(means) >= 2 else 0.0
                cvar = within + between if np.isfinite(within) else np.nan
                units.append((community, cmean, cvar))
            unit_means = np.array([x[1] for x in units], dtype=float)
            unit_vars = np.array([x[2] for x in units], dtype=float)
            n_units = len(units)
        else:
            unit_means = np.array([x[2] for x in active], dtype=float)
            unit_vars = np.square(np.array([x[3] for x in active], dtype=float))
            n_units = len(active)

        meta_mean = float(np.mean(unit_means)) if len(unit_means) else np.nan
        within = float(np.mean(unit_vars)) if len(unit_vars) else np.nan
        between = float(np.var(unit_means, ddof=1)) if len(unit_means) >= 2 else 0.0
        total_var = within + between if np.isfinite(within) else np.nan
        meta_sd = float(math.sqrt(max(0.0, total_var))) if np.isfinite(total_var) else np.nan
        m = meta.loc[game_key]
        market = float(m["market_margin"])
        actual = float(m["actual_margin"])
        edge = meta_mean - market if np.isfinite(meta_mean) else np.nan
        sig = _signal(np.array([edge]), np.array([meta_sd]))[0]
        rows.append({
            "game_key": game_key,
            "season": int(m["season"]),
            "week": int(m["week"]),
            "market_margin": market,
            "actual_margin": actual,
            "cover": actual - market,
            "meta_mean": meta_mean,
            "meta_sd": meta_sd,
            "meta_edge": edge,
            "meta_signal": sig,
            "active_units": n_units,
            "active_combos": len(active),
            "method": "Diversified META" if diversified else "Naive META",
        })
    return pd.DataFrame(rows)


def _profile_meta(
    frame: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    min_active_units: int,
    standard_price: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    edge = pd.to_numeric(frame["meta_edge"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(frame["meta_signal"], errors="coerce").to_numpy(dtype=float)
    cover = pd.to_numeric(frame["cover"], errors="coerce").to_numpy(dtype=float)
    gate = pd.to_numeric(frame["active_units"], errors="coerce").fillna(0).to_numpy(dtype=float) >= int(min_active_units)
    return pd.DataFrame([
        {**_threshold_stats(edge, sig, cover, k, standard_price=standard_price, extra_gate=gate),
         "scorable_games": int(gate.sum())}
        for k in thresholds
    ])


def analyze_finalist_portfolio(
    data: pd.DataFrame,
    combinations: list[dict],
    discovery_periods: Iterable[tuple[int, int]],
    holdout_periods: Iterable[tuple[int, int]],
    *,
    min_available_models: int = 4,
    thresholds: Iterable[float] = DEFAULT_K_GRID,
    combo_min_bets: int = 50,
    meta_min_bets: int = 30,
    overlap_threshold: float = 0.60,
    min_meta_communities: int = 2,
    standard_price: int = -110,
) -> dict:
    # 1) Let every frozen finalist select a stable discovery-only k.
    tuned = tune_combination_thresholds(
        data, combinations, discovery_periods,
        min_available_models=min_available_models,
        thresholds=thresholds,
        min_bets=combo_min_bets,
        standard_price=standard_price,
    )
    # 2) Annotate overlap communities after tuning; membership is model-only.
    combos, community_table, pairwise, overlap_extra = assign_overlap_communities(
        tuned["combinations"], threshold=overlap_threshold
    )
    # carry community ids back into selected-k table
    selected_k = tuned["selected"].copy()
    if len(selected_k) and len(community_table):
        selected_k = selected_k.merge(
            community_table[["portfolio_combo", "community"]],
            on="portfolio_combo", how="left"
        )
    holdout_combo = evaluate_selected_combo_thresholds(
        data, combos, holdout_periods,
        min_available_models=min_available_models,
        standard_price=standard_price,
    )
    if len(selected_k) and len(holdout_combo):
        selected_k = selected_k.merge(
            holdout_combo.drop(columns=["search_rank"], errors="ignore"),
            on="portfolio_combo", how="left"
        )

    discovery_profiles = []
    holdout_summaries = []
    meta_frames = {}
    selected_meta = []
    for diversified in (False, True):
        method = "Diversified META" if diversified else "Naive META"
        disc_frame = _meta_game_frame(
            data, combos, discovery_periods,
            min_available_models=min_available_models,
            diversified=diversified,
        )
        val_frame = _meta_game_frame(
            data, combos, holdout_periods,
            min_available_models=min_available_models,
            diversified=diversified,
        ) if tuple(holdout_periods) else pd.DataFrame()
        meta_frames[(method, "discovery")] = disc_frame
        meta_frames[(method, "holdout")] = val_frame

        min_units = min_meta_communities if diversified else 2
        prof = _profile_meta(
            disc_frame, thresholds,
            min_active_units=min_units,
            standard_price=standard_price,
        )
        if len(prof):
            prof["method"] = method
            discovery_profiles.append(prof)
            choice = _choose_stable_k(prof, min_bets=meta_min_bets)
            k = float(choice.get("selected_k", np.nan))
        else:
            k = np.nan
        if not np.isfinite(k):
            k = 0.50
        selected_meta.append({"method": method, "selected_k": k})

        for period_name, frame in (("Discovery", disc_frame), ("Holdout", val_frame)):
            if frame is None or frame.empty:
                row = {"k": k, "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                       "ats_pct": np.nan, "units": 0.0, "roi": np.nan, "wilson_low": np.nan}
                scorable = 0
            else:
                edge = pd.to_numeric(frame["meta_edge"], errors="coerce").to_numpy(dtype=float)
                sig = pd.to_numeric(frame["meta_signal"], errors="coerce").to_numpy(dtype=float)
                cover = pd.to_numeric(frame["cover"], errors="coerce").to_numpy(dtype=float)
                gate = pd.to_numeric(frame["active_units"], errors="coerce").fillna(0).to_numpy(dtype=float) >= min_units
                row = _threshold_stats(edge, sig, cover, k, standard_price=standard_price, extra_gate=gate)
                scorable = int(gate.sum())
            holdout_summaries.append({
                "method": method,
                "period": period_name,
                "selected_k": k,
                "scorable_games": scorable,
                **{kk: vv for kk, vv in row.items() if kk != "k"},
            })

    overlap_summary = overlap_extra["summary"]
    return {
        "combinations": combos,
        "overlap_summary": overlap_summary,
        "community_table": community_table,
        "pairwise": pairwise,
        "model_frequency": overlap_extra["model_frequency"],
        "combo_k_detail": tuned["detail"],
        "combo_k_selected": selected_k,
        "meta_discovery_grid": pd.concat(discovery_profiles, ignore_index=True) if discovery_profiles else pd.DataFrame(),
        "meta_summary": pd.DataFrame(holdout_summaries),
        "meta_selected": pd.DataFrame(selected_meta),
        "meta_frames": meta_frames,
        "overlap_threshold": float(overlap_threshold),
        "min_meta_communities": int(min_meta_communities),
    }
