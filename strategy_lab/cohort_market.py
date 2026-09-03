from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


FULL_GAME_MARKETS = (
    "h2h", "spreads", "alternate_spreads", "totals", "alternate_totals",
    "team_totals", "alternate_team_totals",
)
LEGACY_COHORT_TOKENS = (
    "big200", "saggm", "sagpred", "talis", "how", "laz", "sag", "sagr", "dwig",
    "cfbgeek", "cfbprofesor", "sportsratings", "metricsconsensus", "massey", "mcllece",
    "grissom", "sorenson", "keeper", "sasser", "lineSP",
)


def _slug(x: object) -> str:
    s = str(x or "").lower().strip()
    aliases = {
        "southern california": "usc", "usc trojans": "usc",
        "jacksonville state": "jacksonvillestate", "jacksonville st": "jacksonvillestate",
        "miami ohio": "miamioh", "miami oh": "miamioh",
        "connecticut": "uconn", "uconn huskies": "uconn",
        "north carolina": "unc", "north carolina tar heels": "unc",
        "central michigan": "centralmich", "central mich": "centralmich",
        "old dominion": "olddominion", "west virginia": "westvirginia",
        "san jose state": "sanjosestate", "fresno state": "fresnostate",
        "mississippi state": "mississippistate", "arizona state": "arizonastate",
        "san diego state": "sandiegostate", "east carolina": "eastcarolina",
    }
    if s in aliases:
        return aliases[s]
    s = s.replace("&", "and")
    s = re.sub(r"\b(university|college|fighting|state university)\b", "", s)
    s = re.sub(r"\b(st\.?|state)\b", "state", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def american_to_decimal(price: float) -> float:
    p = float(price)
    if not np.isfinite(p) or p == 0:
        return np.nan
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / abs(p))


def american_to_implied(price: float) -> float:
    d = american_to_decimal(price)
    return 1.0 / d if np.isfinite(d) and d > 1 else np.nan


def _wilson_lower(wins: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return np.nan
    phat = wins / n
    den = 1 + z * z / n
    center = phat + z * z / (2 * n)
    adj = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return float((center - adj) / den)


def model_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["prediction_margin"] = pd.to_numeric(d.get("prediction_margin"), errors="coerce")
    d["market_margin"] = pd.to_numeric(d.get("market_margin"), errors="coerce")
    d["actual_margin"] = pd.to_numeric(d.get("actual_margin"), errors="coerce")
    d = d.drop_duplicates(["game_key", "canonical_model_id"])
    rows = []
    for (mid, name), g in d.groupby(["canonical_model_id", "model_name"], dropna=False):
        p = g["prediction_margin"].to_numpy(float)
        m = g["market_margin"].to_numpy(float)
        a = g["actual_margin"].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(a) & np.isfinite(m)
        if not ok.any():
            continue
        p, m, a = p[ok], m[ok], a[ok]
        edge = p - m
        cover = a - m
        bet = np.abs(edge) > 1e-12
        graded = bet & (np.abs(cover) > 1e-12)
        wins = int(np.sum(graded & (edge * cover > 0)))
        losses = int(np.sum(graded & (edge * cover < 0)))
        n = wins + losses
        rows.append({
            "canonical_model_id": str(mid), "model_name": str(name),
            "games": int(len(a)), "bets": n, "wins": wins, "losses": losses,
            "ats_pct": wins / n if n else np.nan,
            "wilson_low": _wilson_lower(wins, n),
            "mae": float(np.mean(np.abs(p - a))),
            "market_mae": float(np.mean(np.abs(m - a))),
            "delta_mae_vs_market": float(np.mean(np.abs(p - a)) - np.mean(np.abs(m - a))),
            "mean_abs_edge": float(np.mean(np.abs(edge))),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mae_pct"] = out["mae"].rank(pct=True, ascending=False)
    out["wilson_pct"] = out["wilson_low"].rank(pct=True, ascending=True)
    out["balanced_score"] = 0.55 * out["mae_pct"] + 0.45 * out["wilson_pct"]
    return out.sort_values(["balanced_score", "games"], ascending=[False, False]).reset_index(drop=True)


def _corr_lookup(data: pd.DataFrame, pairwise: pd.DataFrame | None = None) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    if pairwise is not None and len(pairwise) and {"model_a", "model_b", "edge_correlation"}.issubset(pairwise.columns):
        for r in pairwise[["model_a", "model_b", "edge_correlation"]].itertuples(index=False):
            v = pd.to_numeric(pd.Series([r.edge_correlation]), errors="coerce").iloc[0]
            if np.isfinite(v):
                a, b = str(r.model_a), str(r.model_b)
                lookup[(a, b)] = float(v)
                lookup[(b, a)] = float(v)
        return lookup
    z = data[["game_key", "canonical_model_id", "prediction_margin", "market_margin"]].drop_duplicates(["game_key", "canonical_model_id"]).copy()
    z["edge"] = pd.to_numeric(z["prediction_margin"], errors="coerce") - pd.to_numeric(z["market_margin"], errors="coerce")
    wide = z.pivot(index="game_key", columns="canonical_model_id", values="edge")
    corr = wide.corr(min_periods=30)
    for a in corr.columns:
        for b in corr.columns:
            v = corr.loc[a, b]
            if np.isfinite(v):
                lookup[(str(a), str(b))] = float(v)
    return lookup


def assisted_cohort(
    data: pd.DataFrame,
    pairwise: pd.DataFrame | None = None,
    *, method: str = "balanced", max_models: int = 12,
    min_bets: int = 100, correlation_ceiling: float = 0.90,
) -> tuple[list[str], pd.DataFrame]:
    q = model_quality_table(data)
    if q.empty:
        return [], q
    q = q[pd.to_numeric(q["bets"], errors="coerce") >= int(min_bets)].copy()
    if method == "mae":
        q = q.sort_values(["mae", "wilson_low"], ascending=[True, False])
    elif method == "wilson":
        q = q.sort_values(["wilson_low", "mae"], ascending=[False, True])
    else:
        q = q.sort_values(["balanced_score", "mae"], ascending=[False, True])
    corr = _corr_lookup(data, pairwise)
    selected: list[str] = []
    audit = []
    for r in q.itertuples(index=False):
        mid = str(r.canonical_model_id)
        blocker = None
        blocker_corr = np.nan
        for keep in selected:
            c = corr.get((mid, keep), np.nan)
            if np.isfinite(c) and abs(c) >= float(correlation_ceiling):
                blocker, blocker_corr = keep, float(c)
                break
        accepted = blocker is None and len(selected) < int(max_models)
        if accepted:
            selected.append(mid)
        audit.append({
            "canonical_model_id": mid, "model_name": str(r.model_name),
            "accepted": accepted, "blocked_by": blocker or "",
            "blocking_corr": blocker_corr,
            "bets": int(r.bets), "ats_pct": float(r.ats_pct),
            "wilson_low": float(r.wilson_low), "mae": float(r.mae),
            "delta_mae_vs_market": float(r.delta_mae_vs_market),
            "balanced_score": float(r.balanced_score),
        })
        if len(selected) >= int(max_models):
            # Keep a little audit context after filling the cohort, but do not scan thousands of rows.
            if len(audit) >= int(max_models) + 20:
                break
    return selected, pd.DataFrame(audit)


def resolve_legacy_cohort(models: pd.DataFrame, fallback: Iterable[str] = ()) -> list[str]:
    if models is None or models.empty:
        return list(fallback)
    rows = []
    for r in models[["canonical_model_id", "model_name"]].drop_duplicates().itertuples(index=False):
        blob = re.sub(r"[^a-z0-9]+", "", f"{r.canonical_model_id} {r.model_name}".lower())
        rows.append((str(r.canonical_model_id), blob))
    out = []
    for token in LEGACY_COHORT_TOKENS:
        t = re.sub(r"[^a-z0-9]+", "", token.lower().replace("line", ""))
        matches = [mid for mid, blob in rows if t and t in blob]
        if matches:
            out.append(matches[0])
    return list(dict.fromkeys(out)) or list(fallback)


def current_cohort_summary(board: pd.DataFrame, predictions: pd.DataFrame, model_ids: Iterable[str], *, min_models: int = 3) -> pd.DataFrame:
    ids = set(map(str, model_ids))
    if board is None or board.empty or predictions is None or predictions.empty or not ids:
        return pd.DataFrame()
    p = predictions[predictions["canonical_model_id"].astype(str).isin(ids)].copy()
    if p.empty:
        return pd.DataFrame()
    p["prediction_home_margin"] = pd.to_numeric(p["prediction_home_margin"], errors="coerce")
    rows = []
    blookup = board.set_index("game_join_key") if "game_join_key" in board.columns else pd.DataFrame()
    for key, g in p.groupby("game_join_key", sort=False):
        vals = g["prediction_home_margin"].dropna().to_numpy(float)
        if len(vals) < int(min_models):
            continue
        br = blookup.loc[key] if len(blookup) and key in blookup.index else None
        market = float(pd.to_numeric(pd.Series([br.get("market_home_margin", np.nan) if br is not None else np.nan]), errors="coerce").iloc[0])
        home = str(g["home"].iloc[0]); away = str(g["away"].iloc[0])
        mean = float(np.mean(vals)); med = float(np.median(vals)); sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        rows.append({
            "game_join_key": key, "away": away, "home": home,
            "market_home_margin": market, "cohort_mean": mean, "cohort_median": med,
            "cohort_sd": sd, "cohort_n": int(len(vals)),
            "home_lean": int(np.sum(vals > market)) if np.isfinite(market) else np.nan,
            "away_lean": int(np.sum(vals < market)) if np.isfinite(market) else np.nan,
            "raw_edge_home": mean - market if np.isfinite(market) else np.nan,
            "model_names": " | ".join(g["model_name"].astype(str).tolist()),
        })
    return pd.DataFrame(rows).sort_values("raw_edge_home", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def locate_odds_archive(root: str | Path) -> Path | None:
    root = Path(root)
    env = Path(str(__import__("os").environ.get("NCAAF_ODDS_ARCHIVE", "")).strip()).expanduser() if str(__import__("os").environ.get("NCAAF_ODDS_ARCHIVE", "")).strip() else None
    candidates = [
        root / "data" / "odds" / "ncaaf_rich_quotes.csv.gz",
        root / "data" / "odds" / "ncaaf_rich_quotes.csv",
        root / "odds_archive" / "derived" / "ncaaf_rich_quotes.csv.gz",
        root / "odds_archive" / "derived" / "ncaaf_rich_quotes.csv",
        root.parent / "odds_archive" / "derived" / "ncaaf_rich_quotes.csv.gz",
        root.parent / "odds_archive" / "derived" / "ncaaf_rich_quotes.csv",
        root / "data" / "derived" / "ncaaf_rich_quotes.csv.gz",
        root / "data" / "derived" / "ncaaf_rich_quotes.csv",
    ]
    if env is not None:
        if env.is_dir():
            candidates[0:0] = [env / "derived" / "ncaaf_rich_quotes.csv.gz", env / "derived" / "ncaaf_rich_quotes.csv"]
        else:
            candidates.insert(0, env)
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_odds_archive(root: str | Path) -> pd.DataFrame:
    path = locate_odds_archive(root)
    if path is None:
        return pd.DataFrame()
    use = [
        "event_id", "sport_key", "commence_time", "home_team", "away_team",
        "snapshot_returned", "tier", "bookmaker_key", "bookmaker_title",
        "market_key", "outcome_name", "outcome_description", "point", "price_american",
    ]
    df = pd.read_csv(path, low_memory=False)
    cols = [c for c in use if c in df.columns]
    df = df[cols].copy()
    if "sport_key" in df.columns:
        df = df[df["sport_key"].astype(str).eq("americanfootball_ncaaf")]
    df = df[df["market_key"].astype(str).isin(FULL_GAME_MARKETS)].copy()
    df["point"] = pd.to_numeric(df.get("point"), errors="coerce")
    df["price_american"] = pd.to_numeric(df.get("price_american"), errors="coerce")
    df["commence_time"] = pd.to_datetime(df.get("commence_time"), errors="coerce", utc=True)
    df["snapshot_returned"] = pd.to_datetime(df.get("snapshot_returned"), errors="coerce", utc=True)
    df["home_slug"] = df["home_team"].map(_slug)
    df["away_slug"] = df["away_team"].map(_slug)
    # CFB season: Jan/Feb postseason belongs to prior fall season.
    y = df["commence_time"].dt.year
    mo = df["commence_time"].dt.month
    df["season"] = np.where(mo <= 2, y - 1, y)
    return df.reset_index(drop=True)


def odds_archive_coverage(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame()
    g = quotes.groupby("market_key", dropna=False).agg(
        events=("event_id", "nunique"), books=("bookmaker_key", "nunique"), quotes=("event_id", "size"),
        first_game=("commence_time", "min"), last_game=("commence_time", "max"),
    ).reset_index()
    return g.sort_values(["events", "quotes"], ascending=False).reset_index(drop=True)


def _history_cohort_games(data: pd.DataFrame, model_ids: Iterable[str], *, min_models: int = 3) -> pd.DataFrame:
    ids = set(map(str, model_ids))
    d = data[data["canonical_model_id"].astype(str).isin(ids)].copy()
    if d.empty:
        return pd.DataFrame()
    # Use literal listed-home orientation for sportsbook matching.
    for c in ["prediction_home_margin", "market_home_margin", "actual_home_margin", "season", "week"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["away_slug"] = d.get("road", "").map(_slug)
    d["home_slug"] = d.get("home", "").map(_slug)
    rows = []
    for key, g in d.groupby("game_key", sort=False):
        vals = g.drop_duplicates("canonical_model_id")["prediction_home_margin"].dropna().to_numpy(float)
        if len(vals) < int(min_models):
            continue
        ref = g.iloc[0]
        actual = pd.to_numeric(pd.Series([ref.get("actual_home_margin", np.nan)]), errors="coerce").iloc[0]
        market = pd.to_numeric(pd.Series([ref.get("market_home_margin", np.nan)]), errors="coerce").iloc[0]
        rows.append({
            "game_key": key, "season": int(ref["season"]), "week": int(ref["week"]),
            "road": str(ref.get("road", "")), "home": str(ref.get("home", "")),
            "away_slug": str(ref.get("away_slug", "")), "home_slug": str(ref.get("home_slug", "")),
            "cohort_home_margin": float(np.mean(vals)), "cohort_median_home_margin": float(np.median(vals)),
            "cohort_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "cohort_n": int(len(vals)), "market_home_margin": float(market) if np.isfinite(market) else np.nan,
            "actual_home_margin": float(actual) if np.isfinite(actual) else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["margin_residual"] = out["actual_home_margin"] - out["cohort_home_margin"]
    return out


def load_pt_scores(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted((root / "data" / "historical").glob("ncaa*.csv")):
        if path.name == "ncaa_history.csv":
            continue
        m = re.search(r"ncaa(\d{4})", path.name.lower())
        try:
            z = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        z.columns = [str(c).strip().lower() for c in z.columns]
        if not {"road", "home", "week"}.issubset(z.columns):
            continue
        if "season" not in z.columns:
            if not m:
                continue
            z["season"] = int(m.group(1))
        if "hscore" not in z.columns or "vscore" not in z.columns:
            continue
        z["season"] = pd.to_numeric(z["season"], errors="coerce")
        z["week"] = pd.to_numeric(z["week"], errors="coerce")
        z["hscore"] = pd.to_numeric(z["hscore"], errors="coerce")
        z["vscore"] = pd.to_numeric(z["vscore"], errors="coerce")
        z["home_slug"] = z["home"].map(_slug); z["away_slug"] = z["road"].map(_slug)
        rows.append(z[["season", "week", "road", "home", "away_slug", "home_slug", "vscore", "hscore"]])
    return pd.concat(rows, ignore_index=True, sort=False).drop_duplicates(["season", "week", "away_slug", "home_slug"]) if rows else pd.DataFrame()


def _event_total_anchor(g: pd.DataFrame) -> float:
    main = pd.to_numeric(g.loc[g["market_key"].eq("totals"), "point"], errors="coerce").dropna()
    if len(main):
        return float(main.median())
    alt = pd.to_numeric(g.loc[g["market_key"].eq("alternate_totals"), "point"], errors="coerce").dropna()
    return float(alt.median()) if len(alt) else np.nan


def _empirical_prob(resid: np.ndarray, threshold: float, *, greater: bool) -> float:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return np.nan
    hits = np.sum(r > threshold) if greater else np.sum(r < threshold)
    # Jeffreys-style half-count smoothing prevents exact 0/1 on sparse tails.
    return float((hits + 0.5) / (n + 1.0))


def _grade_offer(row: pd.Series) -> str:
    fam = str(row.get("family", ""))
    actual_margin = pd.to_numeric(pd.Series([row.get("actual_home_margin", np.nan)]), errors="coerce").iloc[0]
    if fam in {"ML", "Spread"} and np.isfinite(actual_margin):
        side = row.get("team_side")
        point = pd.to_numeric(pd.Series([row.get("point", np.nan)]), errors="coerce").iloc[0]
        if fam == "ML":
            value = actual_margin if side == "home" else -actual_margin
        else:
            value = actual_margin + point if side == "home" else -actual_margin + point
        return "W" if value > 0 else "L" if value < 0 else "P"
    if fam == "Team Total":
        actual_score = pd.to_numeric(pd.Series([row.get("actual_team_score", np.nan)]), errors="coerce").iloc[0]
        point = pd.to_numeric(pd.Series([row.get("point", np.nan)]), errors="coerce").iloc[0]
        if not (np.isfinite(actual_score) and np.isfinite(point)):
            return ""
        side = str(row.get("outcome_name", ""))
        value = actual_score - point
        if abs(value) < 1e-12:
            return "P"
        return "W" if (side == "Over" and value > 0) or (side == "Under" and value < 0) else "L"
    return ""


def price_historical_market_shelf(
    data: pd.DataFrame, quotes: pd.DataFrame, model_ids: Iterable[str], *, root: str | Path,
    min_models: int = 3, min_margin_calibration: int = 100, min_team_total_calibration: int = 40,
) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame()
    games = _history_cohort_games(data, model_ids, min_models=min_models)
    if games.empty:
        return pd.DataFrame()
    scores = load_pt_scores(root)
    if len(scores):
        games = games.merge(scores[["season", "week", "away_slug", "home_slug", "vscore", "hscore"]], on=["season", "week", "away_slug", "home_slug"], how="left")
    q = quotes.copy()
    full = q[q["market_key"].isin(FULL_GAME_MARKETS)].copy()
    event_rows = []
    for event_id, eg in full.groupby("event_id", sort=False):
        season = pd.to_numeric(eg["season"], errors="coerce").dropna()
        if not len(season):
            continue
        season = int(season.iloc[0])
        away_slug = str(eg["away_slug"].iloc[0]); home_slug = str(eg["home_slug"].iloc[0])
        match = games[(games["season"].eq(season)) & games["away_slug"].eq(away_slug) & games["home_slug"].eq(home_slug)]
        if len(match) != 1:
            continue
        gr = match.iloc[0]
        total_anchor = _event_total_anchor(eg)
        home_mu = (total_anchor + float(gr.cohort_home_margin)) / 2.0 if np.isfinite(total_anchor) else np.nan
        away_mu = (total_anchor - float(gr.cohort_home_margin)) / 2.0 if np.isfinite(total_anchor) else np.nan
        event_rows.append({
            "event_id": event_id, "season": season, "week": int(gr.week), "game_key": gr.game_key,
            "away_slug": away_slug, "home_slug": home_slug, "away": gr.road, "home": gr.home,
            "cohort_home_margin": float(gr.cohort_home_margin), "cohort_sd": float(gr.cohort_sd), "cohort_n": int(gr.cohort_n),
            "market_home_margin": float(gr.market_home_margin) if np.isfinite(gr.market_home_margin) else np.nan,
            "actual_home_margin": float(gr.actual_home_margin) if np.isfinite(gr.actual_home_margin) else np.nan,
            "margin_residual": float(gr.margin_residual) if np.isfinite(gr.margin_residual) else np.nan,
            "market_total_anchor": total_anchor, "cohort_home_score_mu": home_mu, "cohort_away_score_mu": away_mu,
            "hscore": float(gr.get("hscore", np.nan)) if np.isfinite(pd.to_numeric(pd.Series([gr.get("hscore", np.nan)]), errors="coerce").iloc[0]) else np.nan,
            "vscore": float(gr.get("vscore", np.nan)) if np.isfinite(pd.to_numeric(pd.Series([gr.get("vscore", np.nan)]), errors="coerce").iloc[0]) else np.nan,
            "commence_time": eg["commence_time"].dropna().min() if eg["commence_time"].notna().any() else pd.NaT,
        })
    events = pd.DataFrame(event_rows)
    if events.empty:
        return pd.DataFrame()
    events = events.sort_values(["season", "week", "commence_time", "event_id"]).reset_index(drop=True)
    events["home_score_residual"] = events["hscore"] - events["cohort_home_score_mu"]
    events["away_score_residual"] = events["vscore"] - events["cohort_away_score_mu"]
    event_map = events.set_index("event_id")
    out_rows = []
    for er in events.itertuples(index=False):
        prior_margin = games[(games["season"] < er.season) | ((games["season"] == er.season) & (games["week"] < er.week))]["margin_residual"].to_numpy(float)
        prior_events = events[(events["season"] < er.season) | ((events["season"] == er.season) & (events["week"] < er.week))]
        home_score_resid = prior_events["home_score_residual"].to_numpy(float)
        away_score_resid = prior_events["away_score_residual"].to_numpy(float)
        eg = full[full["event_id"].astype(str).eq(str(er.event_id))]
        for qr in eg.itertuples(index=False):
            mk = str(qr.market_key)
            family = ""
            team_side = ""
            model_prob = np.nan
            equivalent_line = np.nan
            actual_team_score = np.nan
            derived_team_mean = np.nan
            point = pd.to_numeric(pd.Series([getattr(qr, "point", np.nan)]), errors="coerce").iloc[0]
            outcome_name = str(getattr(qr, "outcome_name", ""))
            outcome_desc = str(getattr(qr, "outcome_description", ""))
            out_slug = _slug(outcome_name)
            desc_slug = _slug(outcome_desc)
            if mk == "h2h":
                family = "ML"
                if out_slug == er.home_slug:
                    team_side = "home"; equivalent_line = -0.5
                    threshold = -er.cohort_home_margin
                    if np.isfinite(prior_margin).sum() >= int(min_margin_calibration):
                        model_prob = _empirical_prob(prior_margin, threshold, greater=True)
                elif out_slug == er.away_slug:
                    team_side = "away"; equivalent_line = +0.5
                    threshold = -er.cohort_home_margin
                    if np.isfinite(prior_margin).sum() >= int(min_margin_calibration):
                        model_prob = _empirical_prob(prior_margin, threshold, greater=False)
                else:
                    continue
            elif mk in {"spreads", "alternate_spreads"}:
                family = "Spread"
                if not np.isfinite(point):
                    continue
                if out_slug == er.home_slug:
                    team_side = "home"; equivalent_line = point
                    threshold = -point - er.cohort_home_margin
                    if np.isfinite(prior_margin).sum() >= int(min_margin_calibration):
                        model_prob = _empirical_prob(prior_margin, threshold, greater=True)
                elif out_slug == er.away_slug:
                    team_side = "away"; equivalent_line = point
                    threshold = point - er.cohort_home_margin
                    if np.isfinite(prior_margin).sum() >= int(min_margin_calibration):
                        model_prob = _empirical_prob(prior_margin, threshold, greater=False)
                else:
                    continue
            elif mk in {"team_totals", "alternate_team_totals"}:
                family = "Team Total"
                if not np.isfinite(point) or outcome_name not in {"Over", "Under"}:
                    continue
                if desc_slug == er.home_slug:
                    team_side = "home"; derived_team_mean = er.cohort_home_score_mu; actual_team_score = er.hscore
                    resid = home_score_resid
                elif desc_slug == er.away_slug:
                    team_side = "away"; derived_team_mean = er.cohort_away_score_mu; actual_team_score = er.vscore
                    resid = away_score_resid
                else:
                    continue
                threshold = point - derived_team_mean
                if np.isfinite(resid).sum() >= int(min_team_total_calibration):
                    model_prob = _empirical_prob(resid, threshold, greater=(outcome_name == "Over"))
            elif mk in {"totals", "alternate_totals"}:
                # Margin-only cohort does not independently move the game-total mean.
                family = "Game Total"
            else:
                continue
            price = pd.to_numeric(pd.Series([getattr(qr, "price_american", np.nan)]), errors="coerce").iloc[0]
            implied = american_to_implied(price) if np.isfinite(price) else np.nan
            dec = american_to_decimal(price) if np.isfinite(price) else np.nan
            ev = model_prob * dec - 1.0 if np.isfinite(model_prob) and np.isfinite(dec) else np.nan
            row = {
                "event_id": er.event_id, "season": er.season, "week": er.week,
                "game": f"{er.away} @ {er.home}", "away": er.away, "home": er.home,
                "commence_time": er.commence_time, "book": getattr(qr, "bookmaker_title", getattr(qr, "bookmaker_key", "")),
                "book_key": getattr(qr, "bookmaker_key", ""), "market_key": mk, "family": family,
                "outcome_name": outcome_name, "outcome_description": outcome_desc,
                "team_side": team_side, "point": point, "equivalent_spread": equivalent_line,
                "price_american": price, "implied_prob": implied, "model_prob": model_prob, "ev": ev,
                "cohort_home_margin": er.cohort_home_margin, "cohort_sd": er.cohort_sd, "cohort_n": er.cohort_n,
                "market_home_margin": er.market_home_margin, "market_total_anchor": er.market_total_anchor,
                "derived_team_mean": derived_team_mean, "actual_home_margin": er.actual_home_margin,
                "actual_team_score": actual_team_score,
                "margin_calibration_n": int(np.isfinite(prior_margin).sum()),
                "team_total_calibration_n": int(np.isfinite(resid).sum()) if family == "Team Total" else np.nan,
            }
            row["grade"] = _grade_offer(pd.Series(row))
            out_rows.append(row)
    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out["edge_prob"] = out["model_prob"] - out["implied_prob"]
    out["is_alternate"] = out["market_key"].astype(str).str.startswith("alternate_")
    return out.sort_values(["ev", "season", "week"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def shelf_backtest_summary(offers: pd.DataFrame, *, ev_cutoff: float = 0.0) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame()
    d = offers.copy()
    d["ev"] = pd.to_numeric(d["ev"], errors="coerce")
    d = d[np.isfinite(d["ev"]) & (d["ev"] >= float(ev_cutoff)) & d["grade"].isin(["W", "L", "P"])].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for family, g in d.groupby("family"):
        wins = int((g["grade"] == "W").sum()); losses = int((g["grade"] == "L").sum()); pushes = int((g["grade"] == "P").sum())
        units = 0.0
        for r in g.itertuples(index=False):
            if r.grade == "P":
                continue
            dec = american_to_decimal(r.price_american)
            units += (dec - 1.0) if r.grade == "W" else -1.0
        n = wins + losses
        rows.append({
            "family": family, "offers": int(len(g)), "graded": n, "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": wins / n if n else np.nan, "units_flat_risk": units,
            "roi_flat_risk": units / n if n else np.nan, "mean_model_ev": float(g["ev"].mean()),
        })
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)
