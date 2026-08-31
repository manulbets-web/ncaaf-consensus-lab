from __future__ import annotations

from collections import Counter
from typing import Iterable
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


DEFAULT_K_GRID = tuple(np.round(np.arange(0.25, 2.01, 0.25), 2))


META_SPREAD_BUCKETS = (
    ("0–3.5", 0.0, 3.5),
    ("4–7.5", 3.5, 7.5),
    ("8–14.5", 7.5, 14.5),
    ("15–21.5", 14.5, 21.5),
    ("22–27.5", 21.5, 27.5),
    ("28–34.5", 27.5, 34.5),
    ("35+", 34.5, None),
)


def meta_spread_bucket_label(value: float) -> str:
    """Return the configured absolute-market-spread regime label."""
    try:
        x = abs(float(value))
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    for label, lower, upper in META_SPREAD_BUCKETS:
        if upper is None:
            if x > float(lower):
                return label
        elif float(lower) <= 0:
            if x <= float(upper):
                return label
        elif x > float(lower) and x <= float(upper):
            return label
    return ""


def summarize_meta_spread_regimes(
    frame: pd.DataFrame,
    selected_k: float,
    *,
    period: str,
    min_active_units: int = 2,
    standard_price: int = -110,
    buckets=META_SPREAD_BUCKETS,
) -> pd.DataFrame:
    """Summarize a frozen META strategy by absolute market-spread regime.

    Forecast-error metrics use all scorable games in each regime. Betting
    metrics use only the frozen META k and therefore do not optimize by spread
    bucket. Favorite/underdog splits describe the side actually bet.
    """
    cols = [
        "period", "bucket_order", "line_bucket", "selected_k",
        "scorable_games", "forecast_mae", "market_mae",
        "delta_mae_vs_market", "forecast_bias", "bets", "wins", "losses",
        "pushes", "ats_pct", "roi", "wilson_low", "mean_abs_edge",
        "favorite_bets", "favorite_ats_pct", "underdog_bets",
        "underdog_ats_pct",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)

    d = frame.copy()
    market = pd.to_numeric(d.get("market_margin"), errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(d.get("actual_margin"), errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(d.get("meta_mean"), errors="coerce").to_numpy(dtype=float)
    edge = pd.to_numeric(d.get("meta_edge"), errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(d.get("meta_signal"), errors="coerce").to_numpy(dtype=float)
    active = pd.to_numeric(d.get("active_units"), errors="coerce").fillna(0).to_numpy(dtype=float)
    cover = actual - market
    abs_market = np.abs(market)

    gate = (
        (active >= int(min_active_units))
        & np.isfinite(market) & np.isfinite(actual) & np.isfinite(mean)
        & (np.isfinite(signal) | np.isinf(signal))
    )
    qualifies = gate & (signal >= float(selected_k))
    pushes = qualifies & (np.abs(cover) < 1e-12)
    graded = qualifies & ~pushes
    wins = graded & ((edge * cover) > 0)
    losses = graded & ~wins
    favorite_side = qualifies & ((edge * market) > 0)
    underdog_side = qualifies & ((edge * market) < 0)
    win_units = 100.0 / abs(float(standard_price)) if standard_price < 0 else float(standard_price) / 100.0

    rows = []
    for order, (label, lower, upper) in enumerate(tuple(buckets), start=1):
        if upper is None:
            in_bucket = abs_market > float(lower)
        elif float(lower) <= 0:
            in_bucket = abs_market <= float(upper)
        else:
            in_bucket = (abs_market > float(lower)) & (abs_market <= float(upper))
        sc = gate & in_bucket
        b = qualifies & in_bucket
        g = graded & in_bucket
        w = wins & in_bucket
        l = losses & in_bucket
        p = pushes & in_bucket
        fav_g = g & favorite_side
        dog_g = g & underdog_side
        fav_w = w & favorite_side
        dog_w = w & underdog_side
        n_sc = int(sc.sum())
        n_bets = int(g.sum())
        n_w = int(w.sum())
        n_l = int(l.sum())
        n_p = int(p.sum())
        units = n_w * win_units - n_l
        ats = n_w / n_bets if n_bets else np.nan
        roi = units / n_bets if n_bets else np.nan
        forecast_mae = float(np.nanmean(np.abs(mean[sc] - actual[sc]))) if n_sc else np.nan
        market_mae = float(np.nanmean(np.abs(market[sc] - actual[sc]))) if n_sc else np.nan
        bias = float(np.nanmean(mean[sc] - actual[sc])) if n_sc else np.nan
        mean_abs_edge = float(np.nanmean(np.abs(edge[b]))) if np.any(b) else np.nan
        fav_n = int(fav_g.sum())
        dog_n = int(dog_g.sum())
        rows.append({
            "period": str(period),
            "bucket_order": order,
            "line_bucket": label,
            "selected_k": float(selected_k),
            "scorable_games": n_sc,
            "forecast_mae": forecast_mae,
            "market_mae": market_mae,
            "delta_mae_vs_market": forecast_mae - market_mae if np.isfinite(forecast_mae) and np.isfinite(market_mae) else np.nan,
            "forecast_bias": bias,
            "bets": n_bets,
            "wins": n_w,
            "losses": n_l,
            "pushes": n_p,
            "ats_pct": ats,
            "roi": roi,
            "wilson_low": _wilson_lower(n_w, n_bets) if n_bets else np.nan,
            "mean_abs_edge": mean_abs_edge,
            "favorite_bets": fav_n,
            "favorite_ats_pct": (int(fav_w.sum()) / fav_n) if fav_n else np.nan,
            "underdog_bets": dog_n,
            "underdog_ats_pct": (int(dog_w.sum()) / dog_n) if dog_n else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


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
    """Choose a discovery-only k from a genuinely centered stable plateau.

    The previous selector allowed endpoint thresholds (especially k=2.00) to
    win using only a one-sided two-point neighborhood. That could turn a
    modest improvement in historical ATS into a very restrictive current-week
    rule. The revised selector requires both adjacent k values to be eligible,
    scores the centered three-k neighborhood by its worst Wilson lower bound,
    and—when robust performance is within 1 percentage point of the best
    plateau—prefers the threshold with more betting support, then the lower k.

    Endpoint thresholds remain visible in the diagnostic profile but are only
    used as a fallback when no centered plateau can be formed.
    """
    if profile is None or profile.empty:
        return {"selected_k": np.nan, "selection_score": np.nan}
    p = profile.sort_values("k").reset_index(drop=True).copy()
    p["eligible_k"] = pd.to_numeric(p["bets"], errors="coerce").fillna(0).ge(int(min_bets))
    floors = np.full(len(p), np.nan, dtype=float)
    means = np.full(len(p), np.nan, dtype=float)
    centered = np.zeros(len(p), dtype=bool)
    # A stable threshold needs an eligible k immediately on both sides.
    for i in range(1, len(p) - 1):
        window = p.iloc[i-1:i+2]
        if bool(window["eligible_k"].all()):
            vals = pd.to_numeric(window["wilson_low"], errors="coerce")
            if vals.notna().all():
                floors[i] = float(vals.min())
                means[i] = float(vals.mean())
                centered[i] = True
    p["centered_plateau"] = centered
    p["neighbor_wilson_floor"] = floors
    p["neighbor_wilson_mean"] = means

    cand = p[p["eligible_k"] & p["centered_plateau"] & p["neighbor_wilson_floor"].notna()].copy()
    if len(cand):
        best_floor = float(pd.to_numeric(cand["neighbor_wilson_floor"], errors="coerce").max())
        # Treat <=1 percentage point of robust Wilson performance as essentially
        # tied, then prefer more observations / a less restrictive threshold.
        cand["near_best"] = pd.to_numeric(cand["neighbor_wilson_floor"], errors="coerce") >= (best_floor - 0.01)
        near = cand[cand["near_best"]].copy()
        near["distance_from_anchor"] = (pd.to_numeric(near["k"], errors="coerce") - 0.75).abs()
        near = near.sort_values(
            ["bets", "k", "neighbor_wilson_floor", "neighbor_wilson_mean", "distance_from_anchor"],
            ascending=[False, True, False, False, True],
            na_position="last",
        )
        row = near.iloc[0]
    else:
        # Sparse profiles cannot establish a centered plateau. Preserve a
        # deterministic fallback but favor support and proximity to the 0.75
        # search anchor instead of selecting an extreme endpoint on hit rate.
        cand = p[p["eligible_k"]].copy()
        if cand.empty:
            cand = p[p["bets"].gt(0)].copy()
        if cand.empty:
            return {"selected_k": np.nan, "selection_score": np.nan, "profile": p}
        cand["neighbor_wilson_floor"] = cand["wilson_low"]
        cand["neighbor_wilson_mean"] = cand["wilson_low"]
        cand["distance_from_anchor"] = (pd.to_numeric(cand["k"], errors="coerce") - 0.75).abs()
        cand = cand.sort_values(
            ["bets", "distance_from_anchor", "wilson_low", "k"],
            ascending=[False, True, False, True],
            na_position="last",
        )
        row = cand.iloc[0]

    return {
        "selected_k": float(row["k"]),
        "selection_score": float(row.get("neighbor_wilson_floor", np.nan)),
        "profile": p,
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa = set(map(str, a)); bb = set(map(str, b))
    u = aa | bb
    return float(len(aa & bb) / len(u)) if u else 0.0


def assign_overlap_communities(
    combinations: list[dict],
    threshold: float = 0.50,
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
            # A representative is not meaningfully "100% overlapping" with
            # another finalist; keep this missing and label its role explicitly.
            c["jaccard_to_representative"] = np.nan
            communities.append({
                "community": cid,
                "representative_rank": int(c["rank"]),
                "representative_ids": list(c["model_ids"]),
                "members": [c],
            })

    # Restore user's C1/C2 order, not rank-sort order.
    combos.sort(key=lambda x: int(x.get("portfolio_combo", 10**9)))

    # Pairwise similarities are also used to show each finalist's nearest *other*
    # relative. This avoids the misleading self-Jaccard=100% display.
    pair_rows = []
    nearest = {}
    for i in range(len(combos)):
        best = None
        for j in range(len(combos)):
            if i == j:
                continue
            jac = jaccard(combos[i]["model_ids"], combos[j]["model_ids"])
            shared_ids = sorted(set(combos[i]["model_ids"]) & set(combos[j]["model_ids"]))
            candidate = (jac, -int(combos[j].get("rank", 10**9)), int(combos[j]["portfolio_combo"]), shared_ids)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is not None:
            nearest[int(combos[i]["portfolio_combo"])] = {
                "nearest_other_jaccard": float(best[0]),
                "nearest_other_combo": f"C{int(best[2])}",
                "nearest_shared_ids": "|".join(best[3]),
                "nearest_shared_n": len(best[3]),
            }
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            pair_rows.append({
                "combo_a": f"C{int(combos[i]['portfolio_combo'])}",
                "combo_b": f"C{int(combos[j]['portfolio_combo'])}",
                "jaccard": jaccard(combos[i]["model_ids"], combos[j]["model_ids"]),
                "shared_models": len(set(combos[i]["model_ids"]) & set(combos[j]["model_ids"])),
                "same_community": int(combos[i]["community"]) == int(combos[j]["community"]),
            })
    pairwise = pd.DataFrame(pair_rows)

    rows = []
    rep_by_community = {int(q["community"]): int(q["members"][0]["portfolio_combo"]) for q in communities}
    for c in combos:
        pid = int(c["portfolio_combo"])
        ninfo = nearest.get(pid, {})
        is_rep = rep_by_community.get(int(c["community"])) == pid
        rows.append({
            "combo": f"C{pid}",
            "portfolio_combo": pid,
            "search_rank": int(c["rank"]),
            "community": int(c["community"]),
            "combo_size": len(c["model_ids"]),
            "role": "Representative" if is_rep else "Member",
            "jaccard_to_representative": float(c["jaccard_to_representative"]) if np.isfinite(c.get("jaccard_to_representative", np.nan)) else np.nan,
            "nearest_other_combo": ninfo.get("nearest_other_combo", ""),
            "nearest_other_jaccard": ninfo.get("nearest_other_jaccard", np.nan),
            "nearest_shared_n": ninfo.get("nearest_shared_n", 0),
            "nearest_shared_ids": ninfo.get("nearest_shared_ids", ""),
            "model_ids": "|".join(c["model_ids"]),
        })
    community_table = pd.DataFrame(rows)

    # Structural effective META exposure when every model is present:
    # equal community weight -> equal combo weight within community -> equal model
    # weight within combo. This is the exact nominal influence implied by the
    # diversified META hierarchy before game-specific missingness.
    counts = Counter(mid for c in combos for mid in c["model_ids"])
    community_sets = {}
    structural_weight = Counter()
    G = max(1, len(communities))
    for q in communities:
        members = q["members"]
        nc = max(1, len(members))
        for c in members:
            nm = max(1, len(c["model_ids"]))
            w = 1.0 / G / nc / nm
            for mid in c["model_ids"]:
                structural_weight[mid] += w
                community_sets.setdefault(mid, set()).add(int(q["community"]))
    model_freq = pd.DataFrame([
        {
            "canonical_model_id": mid,
            "combo_count": int(n),
            "combo_share": n / len(combos),
            "community_count": len(community_sets.get(mid, set())),
            "community_share": len(community_sets.get(mid, set())) / G,
            "effective_meta_weight": float(structural_weight.get(mid, 0.0)),
        }
        for mid, n in counts.items()
    ]).sort_values(["effective_meta_weight", "combo_count", "canonical_model_id"], ascending=[False, False, True]) if counts else pd.DataFrame()
    summary = {
        "raw_combos": len(combos),
        "communities": len(communities),
        "unique_models": len(counts),
        "mean_pairwise_jaccard": float(pairwise["jaccard"].mean()) if len(pairwise) else np.nan,
        "max_pairwise_jaccard": float(pairwise["jaccard"].max()) if len(pairwise) else np.nan,
        "overlap_threshold": float(threshold),
        "max_effective_model_weight": float(model_freq["effective_meta_weight"].max()) if len(model_freq) else np.nan,
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
        count, mean, sd = _combo_forecast_arrays(pred, c.get("model_ids", []), min_available_models)
        forecasts.append({
            "combo": i,
            "community": int(c.get("community", i)),
            "count": count,
            "mean": mean,
            "sd": sd,
        })
    rows = []
    for gi, game_key in enumerate(pred.index):
        active = []
        for f in forecasts:
            mu = float(f["mean"][gi]) if np.isfinite(f["mean"][gi]) else np.nan
            sd = float(f["sd"][gi]) if np.isfinite(f["sd"][gi]) else np.nan
            n = int(f["count"][gi]) if np.isfinite(f["count"][gi]) else 0
            if np.isfinite(mu) and np.isfinite(sd) and n > 0:
                active.append((f["community"], f["combo"], mu, sd, n))
        if not active:
            continue

        # META is an ensemble of ensemble *means*. The old implementation
        # reused each combo's raw model SD at full strength, then added
        # between-community dispersion. That double-counted within-ensemble
        # disagreement and made the final consensus much more restrictive than
        # the underlying C1/C2/... decisions. Here the within-unit component is
        # the uncertainty of the ensemble mean (SD / sqrt(n)), while independent
        # community disagreement remains on the full spread scale.
        if diversified:
            units = []
            raw_unit_vars = []
            for community in sorted({x[0] for x in active}):
                z = [x for x in active if x[0] == community]
                means = np.array([x[2] for x in z], dtype=float)
                sds = np.array([x[3] for x in z], dtype=float)
                counts = np.array([max(1, int(x[4])) for x in z], dtype=float)
                cmean = float(np.mean(means))
                mean_uncertainty = float(np.mean(np.square(sds) / counts)) if len(sds) else np.nan
                between_combo = float(np.var(means, ddof=1)) if len(means) >= 2 else 0.0
                cvar_mean = mean_uncertainty + between_combo if np.isfinite(mean_uncertainty) else np.nan
                raw_within = float(np.mean(np.square(sds))) if len(sds) else np.nan
                raw_cvar = raw_within + between_combo if np.isfinite(raw_within) else np.nan
                units.append((community, cmean, cvar_mean))
                raw_unit_vars.append(raw_cvar)
            unit_means = np.array([x[1] for x in units], dtype=float)
            unit_vars = np.array([x[2] for x in units], dtype=float)
            raw_unit_vars = np.array(raw_unit_vars, dtype=float)
            n_units = len(units)
        else:
            unit_means = np.array([x[2] for x in active], dtype=float)
            unit_vars = np.array([np.square(x[3]) / max(1, int(x[4])) for x in active], dtype=float)
            raw_unit_vars = np.array([np.square(x[3]) for x in active], dtype=float)
            n_units = len(active)

        meta_mean = float(np.mean(unit_means)) if len(unit_means) else np.nan
        within = float(np.mean(unit_vars)) if len(unit_vars) else np.nan
        between = float(np.var(unit_means, ddof=1)) if len(unit_means) >= 2 else 0.0
        consensus_var = within + between if np.isfinite(within) else np.nan
        meta_sd = float(math.sqrt(max(0.0, consensus_var))) if np.isfinite(consensus_var) else np.nan
        raw_within = float(np.nanmean(raw_unit_vars)) if len(raw_unit_vars) else np.nan
        raw_total_var = raw_within + between if np.isfinite(raw_within) else np.nan
        raw_total_sd = float(math.sqrt(max(0.0, raw_total_var))) if np.isfinite(raw_total_var) else np.nan
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
            "meta_consensus_sd": meta_sd,
            "meta_raw_total_sd": raw_total_sd,
            "meta_between_community_sd": float(math.sqrt(max(0.0, between))),
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
    overlap_threshold: float = 0.50,
    min_meta_communities: int = 2,
    standard_price: int = -110,
    line_history: pd.DataFrame | None = None,
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

    # 4) Diagnose the frozen final META strategy by absolute market-spread
    # regime. This is descriptive only: the same discovery-selected META k is
    # used in every bucket, so the tail analysis cannot retrospectively choose
    # a more favorable threshold.
    selected_meta_df = pd.DataFrame(selected_meta)
    diversified_k = np.nan
    if len(selected_meta_df):
        qk = selected_meta_df[selected_meta_df["method"].astype(str).eq("Diversified META")]
        if len(qk):
            diversified_k = pd.to_numeric(pd.Series([qk.iloc[0].get("selected_k")]), errors="coerce").iloc[0]
    if not np.isfinite(diversified_k):
        diversified_k = 0.50
    spread_rows = []
    for period_name, frame_key in (("Discovery", "discovery"), ("Holdout", "holdout")):
        frame = meta_frames.get(("Diversified META", frame_key), pd.DataFrame())
        q = summarize_meta_spread_regimes(
            frame, float(diversified_k), period=period_name,
            min_active_units=int(min_meta_communities),
            standard_price=standard_price,
        )
        if len(q):
            spread_rows.append(q)
    meta_spread_scale = pd.concat(spread_rows, ignore_index=True) if spread_rows else pd.DataFrame()

    overlap_summary = overlap_extra["summary"]

    # Sensitivity check requested for the new 0.50 rule: compare the exact same
    # frozen finalists under 0.50 and the previous 0.60 community definition.
    # Each threshold gets its own discovery-only META k; holdout remains untouched.
    sensitivity_rows = []
    for ov in sorted({0.50, 0.60, float(overlap_threshold)}):
        oc, _, _, oe = assign_overlap_communities(tuned["combinations"], threshold=float(ov))
        disc = _meta_game_frame(data, oc, discovery_periods, min_available_models=min_available_models, diversified=True)
        val = _meta_game_frame(data, oc, holdout_periods, min_available_models=min_available_models, diversified=True) if tuple(holdout_periods) else pd.DataFrame()
        prof = _profile_meta(disc, thresholds, min_active_units=min_meta_communities, standard_price=standard_price)
        choice = _choose_stable_k(prof, min_bets=meta_min_bets) if len(prof) else {"selected_k": np.nan}
        sk = float(choice.get("selected_k", np.nan))
        if not np.isfinite(sk): sk = 0.50
        row = {
            "overlap_threshold": float(ov),
            "live_rule": bool(abs(float(ov)-float(overlap_threshold)) < 1e-12),
            "communities": int(oe["summary"].get("communities",0)),
            "max_effective_model_weight": float(oe["summary"].get("max_effective_model_weight",np.nan)),
            "selected_k": sk,
        }
        for label, frame in (("discovery", disc), ("holdout", val)):
            if frame is None or frame.empty:
                st={"bets":0,"ats_pct":np.nan,"roi":np.nan,"wilson_low":np.nan}
            else:
                edge=pd.to_numeric(frame["meta_edge"],errors="coerce").to_numpy(float)
                sig=pd.to_numeric(frame["meta_signal"],errors="coerce").to_numpy(float)
                cover=pd.to_numeric(frame["cover"],errors="coerce").to_numpy(float)
                gate=pd.to_numeric(frame["active_units"],errors="coerce").fillna(0).to_numpy(float)>=int(min_meta_communities)
                st=_threshold_stats(edge,sig,cover,sk,standard_price=standard_price,extra_gate=gate)
            row.update({f"{label}_bets":st.get("bets",0),f"{label}_ats_pct":st.get("ats_pct",np.nan),f"{label}_roi":st.get("roi",np.nan),f"{label}_wilson_low":st.get("wilson_low",np.nan)})
        sensitivity_rows.append(row)
    overlap_sensitivity = pd.DataFrame(sensitivity_rows)

    line_reference_performance = pd.DataFrame()
    if isinstance(line_history, pd.DataFrame) and not line_history.empty:
        line_reference_performance = consortium_line_reference_performance(
            data, line_history, combos, discovery_periods, holdout_periods,
            selected_k, selected_meta_df,
            min_available_models=min_available_models,
            min_meta_communities=min_meta_communities,
            standard_price=standard_price,
        )

    return {
        "combinations": combos,
        "overlap_summary": overlap_summary,
        "community_table": community_table,
        "pairwise": pairwise,
        "model_frequency": overlap_extra["model_frequency"],
        "overlap_sensitivity": overlap_sensitivity,
        "line_reference_performance": line_reference_performance,
        "combo_k_detail": tuned["detail"],
        "combo_k_selected": selected_k,
        "meta_discovery_grid": pd.concat(discovery_profiles, ignore_index=True) if discovery_profiles else pd.DataFrame(),
        "meta_summary": pd.DataFrame(holdout_summaries),
        "meta_selected": selected_meta_df,
        "meta_frames": meta_frames,
        "meta_spread_scale": meta_spread_scale,
        "overlap_threshold": float(overlap_threshold),
        "min_meta_communities": int(min_meta_communities),
        "discovery_periods": tuple((int(y), int(w)) for y, w in discovery_periods),
        "holdout_periods": tuple((int(y), int(w)) for y, w in holdout_periods),
    }


# ===========================================================================
# v3.5.21 — PredictionTracker opening / midweek / updated-final line analysis
# ===========================================================================

def _team_slug(value) -> str:
    x = str(value or "").lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def load_predictiontracker_line_history(root: str | Path, data: pd.DataFrame) -> pd.DataFrame:
    """Map PT's lineopen/linemidweek/line fields onto canonical game orientation.

    `line` is the archive's final Updated field. We expose it as the close proxy
    but label it explicitly in the UI; it is not claimed to be a sportsbook
    timestamped closing snapshot.
    """
    root = Path(root)
    paths = sorted((root / "data" / "historical").glob("ncaa*.csv"))
    if not paths or data is None or data.empty:
        return pd.DataFrame()

    raw_frames = []
    for path in paths:
        try:
            z = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        z.columns = [str(c).strip().lower() for c in z.columns]
        if not {"road", "home", "week"}.issubset(z.columns):
            continue
        if "season" not in z.columns:
            m = re.search(r"ncaa(\d{4})", path.name.lower())
            if not m:
                continue
            z["season"] = int(m.group(1))
        for c in ["season", "week", "lineopen", "linemidweek", "line"]:
            if c in z.columns:
                z[c] = pd.to_numeric(z[c], errors="coerce")
        z["road_slug"] = z["road"].map(_team_slug)
        z["home_slug"] = z["home"].map(_team_slug)
        raw_frames.append(z)
    if not raw_frames:
        return pd.DataFrame()
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)

    d = data.copy()
    for c in ["season", "week", "pair_orientation"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["road_slug"] = d.get("road", "").map(_team_slug)
    d["home_slug"] = d.get("home", "").map(_team_slug)
    source_col = "selected_source" if "selected_source" in d.columns else ("source" if "source" in d.columns else None)
    if source_col:
        pt = d[d[source_col].astype(str).eq("predictiontracker")].copy()
        if pt.empty:
            pt = d.copy()
    else:
        pt = d.copy()
    gm = (
        pt.sort_values(["season", "week", "game_key"])
        .drop_duplicates("game_key")
        [[c for c in ["game_key", "season", "week", "road_slug", "home_slug", "pair_orientation", "actual_margin"] if c in pt.columns]]
    )
    if "pair_orientation" not in gm.columns:
        gm["pair_orientation"] = 1.0
    gm["pair_orientation"] = pd.to_numeric(gm["pair_orientation"], errors="coerce").fillna(1.0)
    key = ["season", "week", "road_slug", "home_slug"]
    raw = raw.merge(gm, on=key, how="inner")
    if raw.empty:
        return pd.DataFrame()
    out = raw.drop_duplicates("game_key").copy()
    orient = pd.to_numeric(out["pair_orientation"], errors="coerce").fillna(1.0)

    # Preserve the literal PredictionTracker values as well as the canonical
    # orientation used by the modeling engine.  The raw home-margin fields are
    # useful for auditing archive anomalies because the public PT convention is
    # expressed relative to the listed home team, while the canonical engine can
    # reverse a matchup through pair_orientation.
    out["pt_road"] = out.get("road", pd.Series("", index=out.index)).astype(str)
    out["pt_home"] = out.get("home", pd.Series("", index=out.index)).astype(str)
    out["pair_orientation"] = orient
    for src, dst, raw_dst in [
        ("lineopen", "open_margin", "open_home_margin_raw"),
        ("linemidweek", "midweek_margin", "midweek_home_margin_raw"),
        ("line", "close_margin", "close_home_margin_raw"),
    ]:
        vals = pd.to_numeric(out[src], errors="coerce") if src in out.columns else pd.Series(np.nan, index=out.index)
        out[raw_dst] = vals
        out[dst] = vals * orient
    keep = [
        "game_key", "season", "week", "pt_road", "pt_home", "pair_orientation",
        "open_home_margin_raw", "midweek_home_margin_raw", "close_home_margin_raw",
        "open_margin", "midweek_margin", "close_margin",
    ]
    return out[keep].reset_index(drop=True)


def _line_reference_summary(edge, cover, prediction, actual, market, *, standard_price=-110, k=None, signal=None, gate=None):
    edge = np.asarray(edge, dtype=float); cover = np.asarray(cover, dtype=float)
    prediction = np.asarray(prediction, dtype=float); actual = np.asarray(actual, dtype=float); market = np.asarray(market, dtype=float)
    if k is None:
        decision = np.isfinite(edge) & (np.abs(edge) >= 1e-12)
        valid = decision & np.isfinite(cover) & (np.abs(cover) > 1e-12)
        win = valid & ((edge * cover) > 0); loss = valid & ((edge * cover) < 0)
        push = decision & np.isfinite(cover) & (np.abs(cover) <= 1e-12)
        wins, losses, pushes = int(win.sum()), int(loss.sum()), int(push.sum())
        units = float(_unit_result(win, loss, standard_price).sum())
        bets = wins + losses
        stats = {"bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
                 "ats_pct": wins / bets if bets else np.nan, "units": units,
                 "roi": units / bets if bets else np.nan, "wilson_low": _wilson_lower(wins, bets)}
    else:
        stats = _threshold_stats(edge, np.asarray(signal, dtype=float), cover, float(k), standard_price=standard_price, extra_gate=gate)
    fmask = np.isfinite(prediction) & np.isfinite(actual)
    mmask = np.isfinite(market) & np.isfinite(actual)
    emask = np.isfinite(edge)
    model_mae = float(np.mean(np.abs(prediction[fmask] - actual[fmask]))) if fmask.any() else np.nan
    market_mae = float(np.mean(np.abs(market[mmask] - actual[mmask]))) if mmask.any() else np.nan
    stats.update({
        "scorable_games": int((fmask & np.isfinite(market)).sum()),
        "model_mae": model_mae,
        "market_mae": market_mae,
        "delta_mae_vs_market": model_mae - market_mae if np.isfinite(model_mae) and np.isfinite(market_mae) else np.nan,
        "mean_abs_edge": float(np.mean(np.abs(edge[emask]))) if emask.any() else np.nan,
    })
    return stats


def individual_model_line_reference_performance(data: pd.DataFrame, line_history: pd.DataFrame, *, standard_price=-110) -> pd.DataFrame:
    if data is None or data.empty or line_history is None or line_history.empty:
        return pd.DataFrame()
    cols = ["game_key", "canonical_model_id", "model_name", "prediction_margin", "actual_margin"]
    d = data[[c for c in cols if c in data.columns]].drop_duplicates(["game_key", "canonical_model_id"]).copy()
    d = d.merge(line_history, on="game_key", how="inner")
    if d.empty:
        return pd.DataFrame()
    rows=[]
    refs=[("Open", "open_margin"), ("Midweek", "midweek_margin"), ("Close (PT Updated/final)", "close_margin")]
    for (mid, name), g in d.groupby(["canonical_model_id", "model_name"], dropna=False):
        pred=pd.to_numeric(g["prediction_margin"], errors="coerce").to_numpy(float)
        actual=pd.to_numeric(g["actual_margin"], errors="coerce").to_numpy(float)
        for label,col in refs:
            market=pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            edge=pred-market; cover=actual-market
            st=_line_reference_summary(edge, cover, pred, actual, market, standard_price=standard_price)
            rows.append({"canonical_model_id":str(mid), "model_name":str(name), "line_reference":label, **st})
    return pd.DataFrame(rows)


def consortium_line_reference_performance(
    data: pd.DataFrame, line_history: pd.DataFrame, combinations: list[dict],
    discovery_periods, holdout_periods, combo_k_selected: pd.DataFrame, meta_selected: pd.DataFrame,
    *, min_available_models=3, min_meta_communities=2, standard_price=-110,
) -> pd.DataFrame:
    if not combinations or line_history is None or line_history.empty:
        return pd.DataFrame()
    kmap={}
    if isinstance(combo_k_selected, pd.DataFrame) and len(combo_k_selected):
        for r in combo_k_selected.itertuples(index=False):
            try: kmap[int(getattr(r,"portfolio_combo"))]=float(getattr(r,"selected_k"))
            except Exception: pass
    meta_k=0.75
    if isinstance(meta_selected, pd.DataFrame) and len(meta_selected):
        q=meta_selected[meta_selected["method"].astype(str).eq("Diversified META")]
        if len(q):
            v=pd.to_numeric(pd.Series([q.iloc[0].get("selected_k")]), errors="coerce").iloc[0]
            if np.isfinite(v): meta_k=float(v)
    refs=[("Open", "open_margin"), ("Midweek", "midweek_margin"), ("Close (PT Updated/final)", "close_margin")]
    rows=[]
    union=list(dict.fromkeys(mid for c in combinations for mid in map(str,c.get("model_ids",[]))))
    for period_name, periods in [("Discovery", tuple(discovery_periods)), ("Holdout", tuple(holdout_periods))]:
        if not periods: continue
        pred, meta=_matrix_and_meta(data, union, periods)
        if pred.empty: continue
        lh=line_history.set_index("game_key").reindex(pred.index)
        actual=pd.to_numeric(meta["actual_margin"], errors="coerce").to_numpy(float)
        for i,c in enumerate(combinations,start=1):
            count, mean, sd=_combo_forecast_arrays(pred,c.get("model_ids",[]),min_available_models)
            k=float(kmap.get(i,c.get("k",0.75) if np.isfinite(pd.to_numeric(pd.Series([c.get("k",np.nan)]), errors="coerce").iloc[0]) else 0.75))
            for label,col in refs:
                market=pd.to_numeric(lh[col], errors="coerce").to_numpy(float)
                edge=mean-market; cover=actual-market; sig=_signal(edge,sd)
                st=_line_reference_summary(edge,cover,mean,actual,market,standard_price=standard_price,k=k,signal=sig)
                rows.append({"period":period_name,"entity":f"C{i}","search_rank":int(c.get("rank",i)),"community":int(c.get("community",i)),"selected_k":k,"line_reference":label,**st})
        mf=_meta_game_frame(data,combinations,periods,min_available_models=min_available_models,diversified=True)
        if len(mf):
            mfi=mf.set_index("game_key")
            lh2=line_history.set_index("game_key").reindex(mfi.index)
            mean=pd.to_numeric(mfi["meta_mean"],errors="coerce").to_numpy(float)
            sd=pd.to_numeric(mfi["meta_sd"],errors="coerce").to_numpy(float)
            actual2=pd.to_numeric(mfi["actual_margin"],errors="coerce").to_numpy(float)
            gate=pd.to_numeric(mfi["active_units"],errors="coerce").fillna(0).to_numpy(float)>=int(min_meta_communities)
            for label,col in refs:
                market=pd.to_numeric(lh2[col],errors="coerce").to_numpy(float)
                edge=mean-market; cover=actual2-market; sig=_signal(edge,sd)
                st=_line_reference_summary(edge,cover,mean,actual2,market,standard_price=standard_price,k=meta_k,signal=sig,gate=gate)
                rows.append({"period":period_name,"entity":"Diversified META","search_rank":np.nan,"community":np.nan,"selected_k":meta_k,"line_reference":label,**st})
    return pd.DataFrame(rows)
