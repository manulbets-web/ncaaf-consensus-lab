from __future__ import annotations

import math
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def _finite(values) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def _spread_label(away: str, home: str, home_margin: float) -> str:
    if home_margin is None or not np.isfinite(float(home_margin)):
        return "—"
    m = float(home_margin)
    if abs(m) < 0.05:
        return "Pick'em"
    return f"{home} -{abs(m):.1f}" if m > 0 else f"{away} -{abs(m):.1f}"


def _x_limits(individual: pd.DataFrame, combos: pd.DataFrame, meta_mean, meta_sd, market, pt_market):
    vals: list[float] = []
    if individual is not None and len(individual):
        vals.extend(_finite(individual.get("prediction_home_margin", [])).tolist())
    if combos is not None and len(combos):
        mu = pd.to_numeric(combos.get("consensus_home_margin"), errors="coerce")
        sd = pd.to_numeric(combos.get("model_sd"), errors="coerce")
        vals.extend(mu[np.isfinite(mu)].astype(float).tolist())
        both = np.isfinite(mu) & np.isfinite(sd)
        vals.extend((mu[both] - sd[both]).astype(float).tolist())
        vals.extend((mu[both] + sd[both]).astype(float).tolist())
    for v in [meta_mean, market, pt_market]:
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
    if meta_mean is not None and meta_sd is not None and np.isfinite(float(meta_mean)) and np.isfinite(float(meta_sd)):
        vals.extend([float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd)])
    if not vals:
        return -10.0, 10.0
    lo, hi = min(vals), max(vals)
    span = max(4.0, hi - lo)
    pad = max(2.0, span * 0.12)
    return lo - pad, hi + pad


def hierarchy_plot(
    individual: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    away: str,
    home: str,
    selected_model_ids: Iterable[str] = (),
    meta_mean: float = np.nan,
    meta_sd: float = np.nan,
    market_home_margin: float = np.nan,
    pt_market_home_margin: float = np.nan,
    k: float = 0.5,
):
    """Three-level plot: individual models -> finalist combinations -> meta forecast."""
    ind = individual.copy() if individual is not None else pd.DataFrame()
    cmb = combos.copy() if combos is not None else pd.DataFrame()
    selected = {str(x) for x in selected_model_ids}

    if len(ind):
        ind["prediction_home_margin"] = pd.to_numeric(ind["prediction_home_margin"], errors="coerce")
        ind = ind[np.isfinite(ind["prediction_home_margin"])].drop_duplicates("canonical_model_id", keep="first")
        ind = ind.sort_values("prediction_home_margin").reset_index(drop=True)
    if len(cmb):
        cmb["portfolio_combo"] = pd.to_numeric(cmb["portfolio_combo"], errors="coerce")
        cmb["consensus_home_margin"] = pd.to_numeric(cmb["consensus_home_margin"], errors="coerce")
        cmb["model_sd"] = pd.to_numeric(cmb["model_sd"], errors="coerce")
        cmb = cmb.sort_values("portfolio_combo").drop_duplicates("portfolio_combo", keep="first")

    n_combo = len(cmb)
    height = max(7.0, 4.7 + 0.38 * n_combo)
    fig, ax = plt.subplots(figsize=(13.5, height))

    # Market reference lines.
    if np.isfinite(market_home_margin):
        ax.axvline(float(market_home_margin), color="#d62728", lw=2.4, zorder=1,
                   label=f"Line used: {_spread_label(away, home, float(market_home_margin))}")
    if np.isfinite(pt_market_home_margin) and (
        not np.isfinite(market_home_margin) or abs(float(pt_market_home_margin) - float(market_home_margin)) > 1e-9
    ):
        ax.axvline(float(pt_market_home_margin), color="#d62728", lw=1.5, ls="--", alpha=0.55, zorder=1,
                   label=f"PT line: {_spread_label(away, home, float(pt_market_home_margin))}")

    # All currently available individual projections on a single lightly-jittered tier.
    if len(ind):
        x = ind["prediction_home_margin"].to_numpy(float)
        mids = ind["canonical_model_id"].astype(str)
        used = mids.isin(selected).to_numpy()
        # deterministic jitter so identical projections remain visible
        jitter = ((np.arange(len(ind)) % 7) - 3) * 0.045
        y = np.full(len(ind), 0.0) + jitter
        if (~used).any():
            ax.scatter(x[~used], y[~used], s=34, color="#a8b3bc", alpha=0.78, edgecolor="white", linewidth=0.4,
                       zorder=3, label="Other current models")
        if used.any():
            ax.scatter(x[used], y[used], s=46, color="#1976d2", alpha=0.95, edgecolor="white", linewidth=0.5,
                       zorder=4, label="Models used by finalist portfolio")

    # One row per finalist combination. Error bar is +/- the SD among models inside that combo.
    yticks = [0.0]
    ylabels = ["Individual models"]
    for j, row in enumerate(cmb.itertuples(index=False), start=1):
        y = float(j)
        mu = float(getattr(row, "consensus_home_margin", np.nan))
        sd = float(getattr(row, "model_sd", np.nan))
        cnum = int(getattr(row, "portfolio_combo", j))
        qualifies = bool(getattr(row, "qualifies", False))
        edge = float(getattr(row, "edge_home", np.nan))
        if np.isfinite(mu):
            if np.isfinite(sd):
                ax.errorbar(mu, y, xerr=sd, fmt="o", ms=6.5, capsize=3.5,
                            color="#245a8d", ecolor="#6f94b5", elinewidth=2.0, zorder=5)
            else:
                ax.scatter([mu], [y], s=48, color="#245a8d", zorder=5)
            decision = "PASS"
            if qualifies and np.isfinite(edge):
                decision = f"BET {home if edge > 0 else away}"
            ax.text(1.005, y, decision, transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=8.5, color="#4d5963")
        yticks.append(y)
        community = getattr(row, "community", None)
        ylabels.append(f"C{cnum} · G{int(community)}" if community is not None and pd.notna(community) else f"C{cnum}")

    # Equal-weight portfolio of combination means.
    meta_y = float(n_combo + 1.65)
    if np.isfinite(meta_mean):
        if np.isfinite(meta_sd):
            ax.errorbar(float(meta_mean), meta_y, xerr=float(meta_sd), fmt="D", ms=9, capsize=5,
                        color="#16823b", ecolor="#58a66d", elinewidth=3.0, zorder=6,
                        label="Diversified META mean ± total SD")
        else:
            ax.scatter([float(meta_mean)], [meta_y], marker="D", s=75, color="#16823b", zorder=6,
                       label="Diversified META mean")
    yticks.append(meta_y)
    ylabels.append("META")

    xmin, xmax = _x_limits(ind, cmb, meta_mean, meta_sd, market_home_margin, pt_market_home_margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, meta_y + 0.75)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(axis="x", alpha=0.16)
    ax.grid(axis="y", alpha=0.08)
    ax.axvline(0, color="#606b75", lw=0.9, alpha=0.35)
    ax.set_xlabel(f"Projected home margin (negative = {away} favored; positive = {home} favored)")
    fig.suptitle(f"{away} @ {home} — forecast hierarchy", x=0.125, ha="left", fontsize=15, weight="bold", y=0.985)
    fig.text(
        0.125, 0.955,
        f"Combo bars show mean ± 1 within-combo SD. G labels are overlap communities. META gives each community equal influence and combines within- + between-community dispersion. META k = {float(k):.2f} SD.",
        fontsize=9.5, color="#5c6770", va="top",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Preserve first occurrence of each label.
        seen = set(); hh = []; ll = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); hh.append(h); ll.append(l)
        ax.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, -0.075), fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout(rect=(0.02, 0.10, 0.92, 0.92))
    return fig


def distribution_plot(
    individual: pd.DataFrame,
    combos: pd.DataFrame,
    *,
    away: str,
    home: str,
    selected_model_ids: Iterable[str] = (),
    meta_mean: float = np.nan,
    meta_sd: float = np.nan,
    market_home_margin: float = np.nan,
    pt_market_home_margin: float = np.nan,
    k: float = 0.5,
):
    """Legacy-inspired KDE/rug plot with individual, combo, and meta layers."""
    ind = individual.copy() if individual is not None else pd.DataFrame()
    cmb = combos.copy() if combos is not None else pd.DataFrame()
    selected = {str(x) for x in selected_model_ids}

    if len(ind):
        ind["prediction_home_margin"] = pd.to_numeric(ind["prediction_home_margin"], errors="coerce")
        ind = ind[np.isfinite(ind["prediction_home_margin"])].drop_duplicates("canonical_model_id", keep="first")
    if len(cmb):
        cmb["portfolio_combo"] = pd.to_numeric(cmb["portfolio_combo"], errors="coerce")
        cmb["consensus_home_margin"] = pd.to_numeric(cmb["consensus_home_margin"], errors="coerce")
        cmb = cmb[np.isfinite(cmb["consensus_home_margin"])].sort_values("portfolio_combo")

    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    xmin, xmax = _x_limits(ind, cmb, meta_mean, meta_sd, market_home_margin, pt_market_home_margin)
    xs = _finite(ind.get("prediction_home_margin", [])) if len(ind) else np.array([])
    peak = 1.0

    if len(xs) >= 3 and np.std(xs, ddof=1) > 1e-8 and len(np.unique(np.round(xs, 8))) >= 3:
        grid = np.linspace(xmin, xmax, 500)
        try:
            dens = gaussian_kde(xs)(grid)
            peak = max(float(np.nanmax(dens)), 1e-6)
            ax.fill_between(grid, 0, dens, color="#90caf9", alpha=0.72, zorder=1)
            ax.plot(grid, dens, color="#37474f", lw=1.4, zorder=2)
        except Exception:
            peak = 1.0
    elif len(xs):
        peak = 1.0

    # Rug: all models grey, strategy-used models blue.
    if len(ind):
        mids = ind["canonical_model_id"].astype(str)
        used = mids.isin(selected).to_numpy()
        vals = ind["prediction_home_margin"].to_numpy(float)
        rug_h = peak * 0.12
        if (~used).any():
            ax.vlines(vals[~used], 0, rug_h, color="#9eabb4", lw=2, alpha=0.72, zorder=3)
        if used.any():
            ax.vlines(vals[used], 0, rug_h * 1.18, color="#1976d2", lw=2.6, alpha=0.95, zorder=4)

    # Combo means get their own row beneath the density baseline.
    combo_y = -peak * 0.075
    if len(cmb):
        for row in cmb.itertuples(index=False):
            mu = float(getattr(row, "consensus_home_margin", np.nan))
            cnum = int(getattr(row, "portfolio_combo", 0))
            ax.scatter([mu], [combo_y], s=38, color="#245a8d", zorder=5)
            ax.text(mu, combo_y - peak * 0.025, f"C{cnum}", ha="center", va="top", fontsize=7.5, rotation=90, color="#245a8d")

    if np.isfinite(meta_mean):
        ax.axvline(float(meta_mean), color="#16823b", ls="--", lw=2.6, zorder=6,
                   label=f"META: {_spread_label(away, home, float(meta_mean))}")
        if np.isfinite(meta_sd):
            ax.axvspan(float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd),
                       color="#58a66d", alpha=0.13, zorder=0, label="Diversified META ± 1 total SD")
    if np.isfinite(market_home_margin):
        ax.axvline(float(market_home_margin), color="#d62728", lw=2.6, zorder=6,
                   label=f"Line used: {_spread_label(away, home, float(market_home_margin))}")
    if np.isfinite(pt_market_home_margin) and (
        not np.isfinite(market_home_margin) or abs(float(pt_market_home_margin) - float(market_home_margin)) > 1e-9
    ):
        ax.axvline(float(pt_market_home_margin), color="#d62728", lw=1.4, ls=":", alpha=0.65, zorder=5,
                   label=f"PT line: {_spread_label(away, home, float(pt_market_home_margin))}")

    ax.axvline(0, color="#606b75", lw=0.8, alpha=0.3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(combo_y - peak * 0.12, peak * 1.13)
    ax.set_yticks([])
    ax.set_xlabel(f"Projected home margin (negative = {away} favored; positive = {home} favored)")
    fig.suptitle(f"{away} @ {home} — forecast distribution", x=0.08, ha="left", fontsize=15, weight="bold", y=0.985)
    fig.text(
        0.08, 0.955,
        "Density/rug = individual current models; blue rug = models used by finalists; C1–C12 = combination means; green = diversity-adjusted META; red = market.",
        fontsize=9.5, color="#5c6770", va="top",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set(); hh = []; ll = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); hh.append(h); ll.append(l)
        ax.legend(hh, ll, loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    return fig


def build_forecast_plot(style: str, individual: pd.DataFrame, combos: pd.DataFrame, **kwargs):
    if str(style).lower() == "distribution":
        return distribution_plot(individual, combos, **kwargs)
    return hierarchy_plot(individual, combos, **kwargs)
