from __future__ import annotations

from collections import defaultdict
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


def _prepare_frames(individual: pd.DataFrame, combos: pd.DataFrame):
    ind = individual.copy() if individual is not None else pd.DataFrame()
    cmb = combos.copy() if combos is not None else pd.DataFrame()

    if len(ind):
        ind["canonical_model_id"] = ind["canonical_model_id"].astype(str)
        if "model_name" not in ind.columns:
            ind["model_name"] = ind["canonical_model_id"]
        ind["model_name"] = ind["model_name"].astype(str)
        ind["prediction_home_margin"] = pd.to_numeric(ind["prediction_home_margin"], errors="coerce")
        ind = (
            ind[np.isfinite(ind["prediction_home_margin"])]
            .drop_duplicates("canonical_model_id", keep="first")
            .sort_values(["prediction_home_margin", "model_name"], ascending=[False, True])
            .reset_index(drop=True)
        )

    if len(cmb):
        cmb["portfolio_combo"] = pd.to_numeric(cmb["portfolio_combo"], errors="coerce")
        cmb["consensus_home_margin"] = pd.to_numeric(cmb["consensus_home_margin"], errors="coerce")
        cmb["model_sd"] = pd.to_numeric(cmb["model_sd"], errors="coerce")
        if "primary_k" in cmb.columns:
            cmb["primary_k"] = pd.to_numeric(cmb["primary_k"], errors="coerce")
        cmb = (
            cmb.sort_values("portfolio_combo")
            .drop_duplicates("portfolio_combo", keep="first")
            .reset_index(drop=True)
        )
    return ind, cmb


def _combo_membership_map(cmb: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    if cmb is None or cmb.empty or "combo_model_ids" not in cmb.columns:
        return dict(out)
    for row in cmb.itertuples(index=False):
        try:
            cnum = int(getattr(row, "portfolio_combo"))
        except Exception:
            continue
        raw = str(getattr(row, "combo_model_ids", "") or "")
        for mid in [x for x in raw.split("|") if x]:
            if cnum not in out[mid]:
                out[mid].append(cnum)
    return dict(out)


def _x_limits(individual: pd.DataFrame, combos: pd.DataFrame, meta_mean, meta_sd, market, pt_market, meta_k=1.0):
    vals: list[float] = []
    if individual is not None and len(individual):
        vals.extend(_finite(individual.get("prediction_home_margin", [])).tolist())
    if combos is not None and len(combos):
        mu = pd.to_numeric(combos.get("consensus_home_margin"), errors="coerce")
        sd = pd.to_numeric(combos.get("model_sd"), errors="coerce")
        kval = pd.to_numeric(combos.get("primary_k", 1.0), errors="coerce") if "primary_k" in combos.columns else pd.Series(1.0, index=combos.index)
        both = np.isfinite(mu) & np.isfinite(sd)
        vals.extend(mu[np.isfinite(mu)].astype(float).tolist())
        vals.extend((mu[both] - sd[both]).astype(float).tolist())
        vals.extend((mu[both] + sd[both]).astype(float).tolist())
        decision = both & np.isfinite(kval)
        vals.extend((mu[decision] - kval[decision] * sd[decision]).astype(float).tolist())
        vals.extend((mu[decision] + kval[decision] * sd[decision]).astype(float).tolist())
    for v in [meta_mean, market, pt_market]:
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
    if meta_mean is not None and meta_sd is not None and np.isfinite(float(meta_mean)) and np.isfinite(float(meta_sd)):
        vals.extend([float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd)])
        if meta_k is not None and np.isfinite(float(meta_k)):
            vals.extend([
                float(meta_mean) - float(meta_k) * float(meta_sd),
                float(meta_mean) + float(meta_k) * float(meta_sd),
            ])
    if not vals:
        return -10.0, 10.0
    lo, hi = min(vals), max(vals)
    span = max(4.0, hi - lo)
    pad = max(2.0, span * 0.10)
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
    """Full ladder: one labeled row per current model -> one row per ensemble -> META."""
    ind, cmb = _prepare_frames(individual, combos)
    selected = {str(x) for x in selected_model_ids}
    memberships = _combo_membership_map(cmb)
    n_model = len(ind)
    n_combo = len(cmb)
    n_total = n_model + n_combo + 1

    # Intentionally tall: the point of this view is to show every model legibly.
    height = max(10.0, 3.8 + 0.28 * n_total)
    fig, ax = plt.subplots(figsize=(15.5, height))

    # Market reference lines span every layer.
    if np.isfinite(market_home_margin):
        ax.axvline(
            float(market_home_margin), color="#d62728", lw=2.4, zorder=1,
            label=f"Line used: {_spread_label(away, home, float(market_home_margin))}",
        )
    if np.isfinite(pt_market_home_margin) and (
        not np.isfinite(market_home_margin)
        or abs(float(pt_market_home_margin) - float(market_home_margin)) > 1e-9
    ):
        ax.axvline(
            float(pt_market_home_margin), color="#d62728", lw=1.4, ls="--", alpha=0.55, zorder=1,
            label=f"PT line: {_spread_label(away, home, float(pt_market_home_margin))}",
        )
    ax.axvline(0, color="#606b75", lw=0.9, alpha=0.30, zorder=0)

    y_positions: list[float] = []
    y_labels: list[str] = []
    used_tick_indices: list[int] = []
    y = 0.0

    # ------------------------------------------------------------------
    # Layer 1: every currently available individual model, one row each.
    # ------------------------------------------------------------------
    for idx, row in ind.iterrows():
        mid = str(row["canonical_model_id"])
        name = str(row.get("model_name", mid))
        x = float(row["prediction_home_margin"])
        used_by = memberships.get(mid, [])
        is_used = mid in selected or bool(used_by)
        color = "#1976d2" if is_used else "#9aa7b1"
        size = 54 if is_used else 40
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.55, zorder=4)
        usage = f" · used by {len(used_by)}/{n_combo} ensembles" if used_by else " · not in finalists"
        ax.text(
            1.005, y, f"{_spread_label(away, home, x)}{usage}",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=8.0, color="#47525c",
        )
        y_positions.append(y)
        y_labels.append(name)
        if is_used:
            used_tick_indices.append(len(y_labels) - 1)
        y += 1.0

    # Section gap and separator.
    if n_model:
        separator_y = y - 0.15
        ax.axhline(separator_y, color="#75828c", lw=0.9, alpha=0.30)
        y += 0.85

    # ------------------------------------------------------------------
    # Layer 2: every finalist ensemble.
    # Thin line = +/-1 SD. Thick translucent line = +/-k*SD decision band.
    # ------------------------------------------------------------------
    for row in cmb.itertuples(index=False):
        mu = float(getattr(row, "consensus_home_margin", np.nan))
        sd = float(getattr(row, "model_sd", np.nan))
        cnum = int(getattr(row, "portfolio_combo", 0))
        ck = float(getattr(row, "primary_k", k)) if pd.notna(getattr(row, "primary_k", k)) else float(k)
        community = getattr(row, "community", None)
        qualifies = bool(getattr(row, "qualifies", False))
        edge = float(getattr(row, "edge_home", np.nan))
        available = getattr(row, "available_models", None)
        combo_size = getattr(row, "combo_size", None)

        if np.isfinite(mu):
            if np.isfinite(sd):
                ax.hlines(y, mu - ck * sd, mu + ck * sd, color="#245a8d", lw=6.5, alpha=0.23, zorder=2)
                ax.hlines(y, mu - sd, mu + sd, color="#6f94b5", lw=2.1, zorder=3)
                ax.vlines([mu - sd, mu + sd], y - 0.11, y + 0.11, color="#6f94b5", lw=1.5, zorder=3)
            ax.scatter([mu], [y], s=61, color="#245a8d", edgecolor="white", linewidth=0.5, zorder=5)

        decision = "PASS"
        if qualifies and np.isfinite(edge):
            decision = f"BET {home if edge > 0 else away}"
        avail_text = ""
        if available is not None and pd.notna(available) and combo_size is not None and pd.notna(combo_size):
            avail_text = f" · {int(available)}/{int(combo_size)} models"
        spread_text = _spread_label(away, home, mu) if np.isfinite(mu) else "—"
        sd_text = f" ± {sd:.2f} SD" if np.isfinite(sd) else ""
        ax.text(
            1.005, y, f"{spread_text}{sd_text} · k={ck:.2f}{avail_text} · {decision}",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=8.0, color="#47525c",
        )
        y_positions.append(y)
        if community is not None and pd.notna(community):
            y_labels.append(f"C{cnum} · G{int(community)} · k={ck:.2f}")
        else:
            y_labels.append(f"C{cnum} · k={ck:.2f}")
        y += 1.0

    if n_combo:
        ax.axhline(y - 0.15, color="#75828c", lw=0.9, alpha=0.30)
        y += 0.85

    # ------------------------------------------------------------------
    # Layer 3: diversified META.
    # ------------------------------------------------------------------
    meta_y = y
    if np.isfinite(meta_mean):
        if np.isfinite(meta_sd):
            ax.hlines(meta_y, float(meta_mean) - float(k) * float(meta_sd), float(meta_mean) + float(k) * float(meta_sd),
                      color="#16823b", lw=8.0, alpha=0.22, zorder=2)
            ax.hlines(meta_y, float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd),
                      color="#58a66d", lw=2.8, zorder=3)
            ax.vlines([float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd)],
                      meta_y - 0.14, meta_y + 0.14, color="#58a66d", lw=1.8, zorder=3)
        ax.scatter([float(meta_mean)], [meta_y], marker="D", s=92, color="#16823b", edgecolor="white", linewidth=0.6,
                   zorder=6, label="Diversified META")
        meta_text = _spread_label(away, home, float(meta_mean))
        if np.isfinite(meta_sd):
            meta_text += f" ± {float(meta_sd):.2f} total SD"
        meta_text += f" · k={float(k):.2f}"
        ax.text(1.005, meta_y, meta_text, transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=8.5, color="#355b3d", weight="bold")
    y_positions.append(meta_y)
    y_labels.append("META")

    xmin, xmax = _x_limits(ind, cmb, meta_mean, meta_sd, market_home_margin, pt_market_home_margin, meta_k=k)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.75, meta_y + 0.75)
    ax.invert_yaxis()  # models first, ensembles next, META last when read top-to-bottom
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.3)
    for i, tick in enumerate(ax.get_yticklabels()):
        if i in used_tick_indices:
            tick.set_weight("bold")
            tick.set_color("#155ca2")
        if i >= n_model and i < n_model + n_combo:
            tick.set_weight("bold")
        if i == len(y_labels) - 1:
            tick.set_weight("bold")
            tick.set_color("#16823b")

    # Section labels on the left margin.
    if n_model:
        ax.text(-0.19, -0.35, "INDIVIDUAL MODELS", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=9.0, weight="bold", color="#5c6770", clip_on=False)
    if n_combo:
        combo_start = float(n_model) + 0.85
        ax.text(-0.19, combo_start - 0.35, "FINALIST ENSEMBLES", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=9.0, weight="bold", color="#5c6770", clip_on=False)
    ax.text(-0.19, meta_y - 0.35, "FINAL CONSENSUS", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=9.0, weight="bold", color="#5c6770", clip_on=False)

    ax.grid(axis="x", alpha=0.16)
    ax.grid(axis="y", alpha=0.05)
    ax.set_xlabel(f"Projected home margin (negative = {away} favored; positive = {home} favored)")
    fig.suptitle(f"{away} @ {home} — complete forecast hierarchy", x=0.12, ha="left", fontsize=15, weight="bold", y=0.995)
    fig.text(
        0.12, 0.974,
        "Every current model is shown, including models outside the finalists. Ensemble thin bars = ±1 within-ensemble SD; thick bars = each ensemble's frozen ±k×SD decision band. META uses overlap communities.",
        fontsize=9.4, color="#5c6770", va="top",
    )

    # Minimal legend; detailed values are printed on each row.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set(); hh = []; ll = []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab); hh.append(h); ll.append(lab)
        ax.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, -0.035), fontsize=8.2, frameon=False, ncol=3)

    # Extra left margin for model names and right margin for row annotations.
    fig.subplots_adjust(left=0.27, right=0.76, top=0.94, bottom=0.07)
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
    """Expanded legacy-inspired density view. All current models remain in the density/rug."""
    ind, cmb = _prepare_frames(individual, combos)
    selected = {str(x) for x in selected_model_ids}
    memberships = _combo_membership_map(cmb)

    fig, ax = plt.subplots(figsize=(15.5, 9.3))
    xmin, xmax = _x_limits(ind, cmb, meta_mean, meta_sd, market_home_margin, pt_market_home_margin, meta_k=k)
    xs = _finite(ind.get("prediction_home_margin", [])) if len(ind) else np.array([])
    peak = 1.0

    if len(xs) >= 3 and np.std(xs, ddof=1) > 1e-8 and len(np.unique(np.round(xs, 8))) >= 3:
        grid = np.linspace(xmin, xmax, 600)
        try:
            dens = gaussian_kde(xs)(grid)
            peak = max(float(np.nanmax(dens)), 1e-6)
            ax.fill_between(grid, 0, dens, color="#90caf9", alpha=0.66, zorder=1)
            ax.plot(grid, dens, color="#37474f", lw=1.4, zorder=2)
        except Exception:
            peak = 1.0

    if len(ind):
        mids = ind["canonical_model_id"].astype(str)
        used = mids.isin(selected).to_numpy()
        vals = ind["prediction_home_margin"].to_numpy(float)
        rug_h = peak * 0.13
        if (~used).any():
            ax.vlines(vals[~used], 0, rug_h, color="#9eabb4", lw=2, alpha=0.70, zorder=3)
        if used.any():
            ax.vlines(vals[used], 0, rug_h * 1.20, color="#1976d2", lw=2.8, alpha=0.95, zorder=4)

        # Label all models along the top edge. Alternating heights reduce collisions.
        label_base = peak * 1.02
        for i, row in ind.sort_values("prediction_home_margin").reset_index(drop=True).iterrows():
            x = float(row["prediction_home_margin"])
            mid = str(row["canonical_model_id"])
            name = str(row.get("model_name", mid))
            y = label_base + (i % 3) * peak * 0.045
            ax.text(x, y, name, rotation=72, ha="left", va="bottom", fontsize=6.6,
                    color="#155ca2" if memberships.get(mid) else "#6f7b84")

    # Ensemble means are spread across multiple tiers below zero to keep labels legible.
    if len(cmb):
        base = -peak * 0.085
        step = peak * 0.052
        for i, row in enumerate(cmb.itertuples(index=False)):
            mu = float(getattr(row, "consensus_home_margin", np.nan))
            sd = float(getattr(row, "model_sd", np.nan))
            ck = float(getattr(row, "primary_k", k)) if pd.notna(getattr(row, "primary_k", k)) else float(k)
            cnum = int(getattr(row, "portfolio_combo", 0))
            tier = i % 3
            cy = base - tier * step
            if np.isfinite(mu):
                if np.isfinite(sd):
                    ax.hlines(cy, mu - ck * sd, mu + ck * sd, color="#245a8d", lw=4.0, alpha=0.22, zorder=4)
                ax.scatter([mu], [cy], s=44, color="#245a8d", zorder=5)
                ax.text(mu, cy - peak * 0.018, f"C{cnum}", ha="center", va="top", fontsize=7.2, color="#245a8d")

    if np.isfinite(meta_mean):
        ax.axvline(float(meta_mean), color="#16823b", ls="--", lw=2.6, zorder=6,
                   label=f"META: {_spread_label(away, home, float(meta_mean))}")
        if np.isfinite(meta_sd):
            ax.axvspan(float(meta_mean) - float(meta_sd), float(meta_mean) + float(meta_sd),
                       color="#58a66d", alpha=0.11, zorder=0, label="META ± 1 total SD")
            ax.axvspan(float(meta_mean) - float(k) * float(meta_sd), float(meta_mean) + float(k) * float(meta_sd),
                       color="#16823b", alpha=0.06, zorder=0, label="META ± k×SD decision band")
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
    ax.set_ylim(-peak * 0.29, peak * 1.23)
    ax.set_yticks([])
    ax.set_xlabel(f"Projected home margin (negative = {away} favored; positive = {home} favored)")
    fig.suptitle(f"{away} @ {home} — all-model forecast distribution", x=0.07, ha="left", fontsize=15, weight="bold", y=0.985)
    fig.text(
        0.07, 0.958,
        "Every available current model contributes to the density/rug and is labeled above it. C1–C12 are ensemble means; green is the diversity-adjusted META; red is the market.",
        fontsize=9.5, color="#5c6770", va="top",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set(); hh = []; ll = []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab); hh.append(h); ll.append(lab)
        ax.legend(hh, ll, loc="upper left", fontsize=8.2, frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    return fig


def build_forecast_plot(style: str, individual: pd.DataFrame, combos: pd.DataFrame, **kwargs):
    if str(style).lower() == "distribution":
        return distribution_plot(individual, combos, **kwargs)
    return hierarchy_plot(individual, combos, **kwargs)
