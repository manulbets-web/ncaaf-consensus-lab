from __future__ import annotations

import numpy as np
import pandas as pd


def rerank_confirmation_finalists(
    result: dict,
    *,
    mode: str,
    finalists: int,
    min_holdout_bets: int,
) -> dict:
    """Rerank a bounded core-search leaderboard after holdout scoring.

    The caller is responsible for constructing the candidate/core universe using
    discovery data only.  When ``mode`` uses holdout metrics, the holdout is
    validation/tuning data rather than an untouched final test.
    """
    if not result:
        return result
    d = result.get("top_internal", pd.DataFrame())
    if d is None or d.empty:
        return result

    d = d.copy()
    if "search_rank" in d.columns:
        d["discovery_rank"] = pd.to_numeric(d["search_rank"], errors="coerce").astype("Int64")
    else:
        d["discovery_rank"] = pd.Series(np.arange(1, len(d) + 1), dtype="Int64")

    for col in [
        "wins", "bets", "ats_pct", "roi", "wilson_low",
        "validation_wins", "validation_bets", "validation_ats_pct",
        "validation_roi", "validation_wilson_low",
    ]:
        if col not in d.columns:
            d[col] = np.nan

    pooled_wins = pd.to_numeric(d["wins"], errors="coerce").fillna(0) + pd.to_numeric(
        d["validation_wins"], errors="coerce"
    ).fillna(0)
    pooled_bets = pd.to_numeric(d["bets"], errors="coerce").fillna(0) + pd.to_numeric(
        d["validation_bets"], errors="coerce"
    ).fillna(0)
    d["combined_pooled_ats"] = np.divide(
        pooled_wins,
        pooled_bets,
        out=np.full(len(d), np.nan, dtype=float),
        where=pooled_bets.to_numpy(dtype=float) > 0,
    )

    mode = str(mode or "discovery_only")
    if mode != "discovery_only" and (
        d["validation_bets"].isna().all() or d["validation_ats_pct"].isna().all()
    ):
        mode = "discovery_only"

    if mode == "discovery_only":
        d["final_rank_score"] = pd.to_numeric(d["ats_pct"], errors="coerce")
        ranked = d.sort_values(
            ["discovery_rank"], ascending=[True], na_position="last", kind="mergesort"
        )
        ranking_label = "Discovery only"
    else:
        hold_bets = pd.to_numeric(d["validation_bets"], errors="coerce")
        eligible = hold_bets.ge(int(min_holdout_bets))
        if mode == "balanced_wilson":
            disc = pd.to_numeric(d["wilson_low"], errors="coerce")
            hold = pd.to_numeric(d["validation_wilson_low"], errors="coerce")
            ranking_label = "50/50 discovery + holdout Wilson LB"
        else:
            mode = "balanced_ats"
            disc = pd.to_numeric(d["ats_pct"], errors="coerce")
            hold = pd.to_numeric(d["validation_ats_pct"], errors="coerce")
            ranking_label = "50/50 discovery + holdout ATS"

        d["final_rank_score"] = 0.5 * disc + 0.5 * hold
        eligible = eligible & d["final_rank_score"].notna()
        d.loc[~eligible, "final_rank_score"] = np.nan
        ranked = d.loc[eligible].sort_values(
            ["final_rank_score", "validation_bets", "validation_ats_pct", "ats_pct", "bets"],
            ascending=[False, False, False, False, False],
            na_position="last",
            kind="mergesort",
        )
        if ranked.empty:
            raise ValueError(
                f"No confirmation combinations had at least {int(min_holdout_bets)} holdout bets "
                "for the combined finalist rank. Lower the minimum holdout-bet gate or use "
                "discovery-only ranking."
            )

    ranked = ranked.head(max(0, int(finalists))).copy().reset_index(drop=True)
    ranked["search_rank"] = np.arange(1, len(ranked) + 1)
    ranked["final_rank"] = ranked["search_rank"]
    ranked["final_rank_mode"] = ranking_label

    display_cols = [
        "search_rank", "discovery_rank", "combo_size", "model_names",
        "bets", "wins", "losses", "ats_pct", "roi", "wilson_low",
        "validation_bets", "validation_wins", "validation_losses",
        "validation_ats_pct", "validation_roi", "validation_wilson_low",
        "final_rank_score", "combined_pooled_ats", "final_rank_mode", "model_ids",
    ]
    result = dict(result)
    result["top_internal"] = ranked.copy()
    result["top"] = ranked[[c for c in display_cols if c in ranked.columns]].copy()
    result["finalist_ranking_mode"] = mode
    result["finalist_ranking_label"] = ranking_label
    result["min_holdout_bets_for_rank"] = int(min_holdout_bets)
    result["validation_candidates_scored"] = int(len(d))
    return result
