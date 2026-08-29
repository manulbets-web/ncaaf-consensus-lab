from __future__ import annotations

from pathlib import Path
import os
import asyncio
import json
import threading
import time

import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

from engine import load_strategy_data
from forecast_plots import build_forecast_plot
from committee import analyze_finalist_portfolio, meta_spread_bucket_label
from current_week import (
    refresh_and_build_current_week,
    build_current_board_from_cached_sources,
    save_current_selection,
)
from streamlined_engine import (
    StreamlinedBacktestConfig,
    CombinationSearchConfig,
    combination_count,
    brute_force_combination_search,
    combination_threshold_robustness,
    combination_spread_scale_performance,
    individual_model_performance,
    load_current_ranking,
    run_streamlined_backtest,
)


# ---------------------------------------------------------------------------
# Data + constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_MODE = os.environ.get("NCAAF_CLOUD_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
DATA, REGISTRY, PAIRWISE = load_strategy_data(PROJECT_ROOT)

if REGISTRY.empty:
    MODELS = (
        DATA[["canonical_model_id", "model_name"]]
        .drop_duplicates("canonical_model_id")
        .sort_values("model_name")
        .reset_index(drop=True)
    )
else:
    MODELS = (
        REGISTRY[["canonical_model_id", "model_name"]]
        .drop_duplicates("canonical_model_id")
        .sort_values("model_name")
        .reset_index(drop=True)
    )

MODELS["canonical_model_id"] = MODELS["canonical_model_id"].astype(str)
MODELS["model_name"] = MODELS["model_name"].astype(str)
MODEL_NAME_MAP = dict(zip(MODELS["canonical_model_id"], MODELS["model_name"]))
ALL_MODEL_IDS = MODELS["canonical_model_id"].astype(str).tolist()
ALL_MODEL_CHOICES = {
    mid: MODEL_NAME_MAP.get(mid, mid)
    for mid in ALL_MODEL_IDS
}

# Historical discovery/holdout periods must be graded periods. DATA may also
# contain current/future rows with season/week populated but no final outcome.
_hist_actual = pd.to_numeric(DATA.get("actual_margin"), errors="coerce")
_hist_market = pd.to_numeric(DATA.get("market_margin"), errors="coerce")
_HISTORICAL_GRADED_MASK = _hist_actual.notna() & _hist_market.notna()
_HISTORICAL_GRADED_DATA = DATA.loc[_HISTORICAL_GRADED_MASK].copy()

HISTORICAL_SEASONS = sorted(
    pd.to_numeric(_HISTORICAL_GRADED_DATA["season"], errors="coerce")
    .dropna().astype(int).unique().tolist()
)
SEASON_CHOICES = {str(y): str(y) for y in HISTORICAL_SEASONS}

_period_frame = _HISTORICAL_GRADED_DATA[["season", "week"]].copy()
_period_frame["season"] = pd.to_numeric(_period_frame["season"], errors="coerce")
_period_frame["week"] = pd.to_numeric(_period_frame["week"], errors="coerce")
_period_frame = _period_frame.dropna().drop_duplicates()
HISTORICAL_PERIODS = tuple(
    sorted((int(y), int(w)) for y, w in _period_frame[["season", "week"]].itertuples(index=False, name=None))
)
if len(HISTORICAL_SEASONS) >= 2:
    DEFAULT_SEARCH_SEASONS = HISTORICAL_SEASONS[:-1]
    DEFAULT_VALIDATION_SEASONS = [HISTORICAL_SEASONS[-1]]
else:
    DEFAULT_SEARCH_SEASONS = HISTORICAL_SEASONS
    DEFAULT_VALIDATION_SEASONS = []

K_GRID = tuple(np.round(np.arange(0.25, 2.01, 0.25), 2))
K_CHOICES = {f"{k:.2f}": f"{k:.2f} SD" for k in K_GRID}
DEFAULT_K = 0.75

# One-click Page 4 recipe. Other screening gates intentionally use the
# established automatic-search defaults so the recommendation changes only
# the settings Patrick explicitly standardized.
PATRICK_HOLDOUT_WEEKS = 6
PATRICK_MIN_SIZE = 4
PATRICK_MAX_SIZE = 7
PATRICK_K = 0.75
PATRICK_FINALISTS = 19

# Exact streaming search remains memory-bounded because only each batch and a
# bounded leaderboard are retained. The UI exposes a user-controlled safety cap
# so multi-million subset spaces can be explored deliberately.
EXACT_SEARCH_DEFAULT_MAX = 10_000_000
EXACT_SEARCH_HARD_MAX = 50_000_000
PATRICK_POOL_N = 26
PATRICK_POOL_METRIC = "wilson"
PATRICK_POOL_MIN_BETS = 25
PATRICK_MIN_AVAILABLE = 4
PATRICK_MIN_SEARCH_BETS = 50
PATRICK_RANK_METRIC = "wilson"
PATRICK_OVERLAP_THRESHOLD = 0.60
PATRICK_META_MIN_COMMUNITIES = 2
PATRICK_HOLDOUT_MIN_SCORABLE_GAMES = 10

# Current ranking is used only as a convenient candidate-pool preset, never as
# an outcome gate inside the combination search.
try:
    CURRENT_RANKING = load_current_ranking(
        PROJECT_ROOT, DATA, MODELS, StreamlinedBacktestConfig()
    )
except Exception:
    CURRENT_RANKING = pd.DataFrame()

if len(CURRENT_RANKING) and "quality_rank" in CURRENT_RANKING.columns:
    DEFAULT_AUTO_IDS = (
        CURRENT_RANKING.sort_values("quality_rank")
        .head(20)["canonical_model_id"]
        .astype(str).tolist()
    )
else:
    DEFAULT_AUTO_IDS = ALL_MODEL_IDS[:20]

DEFAULT_MANUAL_IDS = DEFAULT_AUTO_IDS[: min(10, len(DEFAULT_AUTO_IDS))]

INDIVIDUAL_HISTORY = individual_model_performance(DATA, standard_price=-110)


def _saved_strategy() -> dict:
    path = PROJECT_ROOT / "data/strategy/current_week_selection.json"
    empty = {
        "source": "none",
        "label": "No strategy selected yet",
        "model_ids": [],
        "combinations": [],
        "primary_k": DEFAULT_K,
        "min_available_models": 4,
    }
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text())
        k = float(payload.get("primary_k", DEFAULT_K))
        raw_combos = payload.get("combinations") or []
        combos = []
        for i, c in enumerate(raw_combos, start=1):
            mids = [
                str(x) for x in c.get("model_ids", [])
                if str(x) in MODEL_NAME_MAP
            ]
            if mids:
                row = {
                    "rank": int(c.get("rank", i)),
                    "model_ids": mids,
                }
                if c.get("k") is not None:
                    row["k"] = float(c.get("k"))
                if c.get("community") is not None:
                    row["community"] = int(c.get("community"))
                combos.append(row)

        # Backward compatibility with v3.5.0/v3.5.1 single-set saves.
        ids = [
            str(x) for x in payload.get("model_ids", [])
            if str(x) in MODEL_NAME_MAP
        ]
        if not combos and ids:
            combos = [{"rank": 1, "model_ids": ids}]
        if not combos:
            raise ValueError("saved strategy has no recognized models")

        union_ids = list(dict.fromkeys(
            mid for c in combos for mid in c["model_ids"]
        ))
        n = int(payload.get("min_available_models", min(4, max(len(c["model_ids"]) for c in combos))))
        label = (
            f"Saved portfolio: {len(combos)} combinations @ {k:.2f} SD"
            if len(combos) > 1
            else f"Saved strategy: {len(union_ids)} models @ {k:.2f} SD"
        )
        return {
            "source": "saved",
            "label": label,
            "model_ids": union_ids,
            "combinations": combos,
            "primary_k": k,
            "min_available_models": max(1, n),
        }
    except Exception:
        return empty


SAVED_STRATEGY = _saved_strategy() if not CLOUD_MODE else {
    "source": "none",
    "label": "No strategy selected yet",
    "model_ids": [],
    "combinations": [],
    "primary_k": DEFAULT_K,
    "min_available_models": 4,
}


def _pct_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = 100.0 * pd.to_numeric(out[c], errors="coerce")
    return out


def _spread_label(away: str, home: str, home_margin) -> str:
    m = pd.to_numeric(pd.Series([home_margin]), errors="coerce").iloc[0]
    if not np.isfinite(m):
        return "—"
    if abs(float(m)) < 0.05:
        return "Pick'em"
    if m > 0:
        return f"{home} -{abs(float(m)):.1f}"
    return f"{away} -{abs(float(m)):.1f}"


def _style_css():
    return ui.tags.style(
        """
        .app-title { margin-bottom: 0.15rem; }
        .app-subtitle { color: #65707a; margin-top: 0; margin-bottom: 1rem; }
        .muted { color: #65707a; font-size: 0.92rem; }
        .strategy-banner { padding: .75rem 1rem; border: 1px solid #d9dee3;
                           border-radius: .5rem; margin-bottom: 1rem; }
        .compact-card .card-body { padding-top: .7rem; }
        .search-progress-track { width: 100%; height: 12px; background: #e9ecef;
                                 border-radius: 999px; overflow: hidden; margin: .45rem 0 .35rem 0; }
        .search-progress-fill { height: 100%; background: #0d6efd; transition: width .25s ease; }
        """
    )


# ---------------------------------------------------------------------------
# Four-page UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    _style_css(),
    ui.h2("NCAAF Consensus Lab", class_="app-title"),
    ui.p(
        "Past performance → current board → strategy discovery → upcoming picks → forecast plots. "
        "CFB Picker is intentionally inactive until its 2026 feed is available.",
        class_="app-subtitle",
    ),
    ui.navset_card_tab(
        # ------------------------------------------------------------------
        # Instructions
        # ------------------------------------------------------------------
        ui.nav_panel(
            "How to Use",
            ui.card(
                ui.card_header("Quick start"),
                ui.tags.ol(
                    ui.tags.li(ui.strong("Refresh the current slate on Page 2."), " This loads PredictionTracker's current games, market lines, and model projections."),
                    ui.tags.li(ui.strong("Review Page 1 if you want context on individual models."), " Historical ATS, ROI, Wilson lower bound, forecast error, and season-by-season history are shown there."),
                    ui.tags.li(ui.strong("Build a strategy on Page 3."), " Automatic screening ranks only models that are posting this week, searches promising model combinations on historical discovery data, and evaluates frozen finalists on a recent chronological holdout."),
                    ui.tags.li(ui.strong("Choose one or more finalists on Page 3."), " The selected combinations become C1, C2, C3, and so on for the current session."),
                    ui.tags.li(ui.strong("Apply them on Page 4."), " Each combination independently forms a mean projected spread, an SD across its component models, and a BET/PASS decision. The page also summarizes agreement across combinations."),
                    ui.tags.li(ui.strong("Patrick\'s one-click recipe on Page 4."), " After Page 2 is refreshed, it screens the current-week eligible pool using 6 sufficiently covered held-out weeks, set sizes 4–7, k = 0.75 SD, automatically applies the top 19 combinations, tunes each finalist's k on discovery data, and builds the overlap-adjusted META backtest."),
                    ui.tags.li(ui.strong("Visualize the hierarchy on Page 5."), " Pick any current game to see every mapped model projection, the selected finalist-combination forecasts, and the final portfolio/meta estimate against the market line."),
                ),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("How the betting signal works"),
                    ui.p("For each selected combination, the app averages the available model projected spreads for a game and calculates the sample SD across those model projections."),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Expected spread"), ": mean of the combination's available model projections."),
                        ui.tags.li(ui.strong("Model SD"), ": disagreement among models inside that combination."),
                        ui.tags.li(ui.strong("Raw edge"), ": expected spread minus the market-implied margin, expressed in points."),
                        ui.tags.li(ui.strong("Signal (SD)"), ": absolute raw edge divided by model SD."),
                        ui.tags.li(ui.strong("k"), ": the required signal threshold. Smaller k values are less restrictive and produce more bets."),
                    ),
                    ui.p("A combination produces a BET only when enough of its models are available and the market lies outside that combination's mean ± k×SD decision boundary.", class_="muted"),
                ),
                ui.card(
                    ui.card_header("How to read the combination portfolio"),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("C1, C2, ..."), ": the finalist combinations you selected on Page 3. Search rank remains available in the detailed table, but the short labels are used throughout Page 4."),
                        ui.tags.li(ui.strong("Portfolio mean ± SD"), ": the old raw equal-weight summary across C1–C12, retained as a benchmark."),
                        ui.tags.li(ui.strong("Diversified META"), ": near-duplicate combinations are grouped by 0.60 Jaccard model overlap; each overlap community gets equal influence. META consensus SD combines uncertainty in the ensemble means with disagreement between independent overlap communities; raw within-model dispersion is not counted a second time at full strength."),
                        ui.tags.li(ui.strong("Automatic k"), ": each frozen finalist is evaluated across 0.25–2.00 SD on discovery data only. The app favors a stable neighboring-k plateau rather than the single prettiest threshold, then tests that frozen k on holdout."),
                        ui.tags.li(ui.strong("Mean direction"), ": how many combination means lie on each side of the current betting line."),
                        ui.tags.li(ui.strong("Bet direction"), ": how many combinations actually clear their k×SD threshold in each direction."),
                        ui.tags.li(ui.strong("PASS"), ": a combination has enough models to score the game, but the available line remains inside its decision boundary."),
                    ),
                ),
                col_widths=(6, 6),
            ),
            ui.card(
                ui.card_header("Line shopping / alternate lines"),
                ui.p(
                    "Page 4 lets you temporarily replace the PredictionTracker market line with a line you can actually bet. "
                    "Choose a game and enter the home team's point spread exactly as a sportsbook displays it: negative if the home team is favored and positive if the home team is the underdog. "
                    "For example, for NC State @ Virginia, enter -3.0 to test Virginia -3 instead of Virginia -4."
                ),
                ui.p(
                    "The override immediately re-scores every selected combination and updates the agreement counts. "
                    "It does not change the historical data, PredictionTracker's stored line, or anyone else's session.",
                    class_="muted",
                ),
            ),
            ui.card(
                ui.card_header("Recommended workflow for automatic screening"),
                ui.tags.ul(
                    ui.tags.li("Refresh Page 2 first so the candidate pool contains only models that are actually posting this week."),
                    ui.tags.li(ui.strong("Patrick\'s recommended recipe:"), " Top 26 current-week models by discovery Wilson lower bound (minimum 25 discovery bets/model), hold out the latest 6 completed weeks with adequate model coverage, screen set sizes 4–7 at k = 0.75 SD, rank by Wilson lower bound, freeze the top 19, then automatically tune each finalist's k across 0.25–2.00 SD on discovery only and build a diversity-adjusted META backtest."),
                    ui.tags.li("The Page 4 button runs that recipe end-to-end; Page 3 remains available when you want to inspect or alter the research settings."),
                    ui.tags.li("Treat the discovery ranking as model-combination discovery and the recent held-out weeks as the cleaner validation check."),
                    ui.tags.li("On Page 4, use the Absolute spread trust check to compare the final Diversified META strategy across small spreads through 35+, including META-vs-market MAE and separate favorite/underdog betting performance."),
                ),
                ui.p("CFB Picker is currently inactive and can be added back when its current-season feed is available.", class_="muted"),
            ),
            ui.p(
                "This app is a research and decision-support tool. Historical performance and model agreement do not guarantee future results.",
                class_="muted",
            ),
        ),

        # ------------------------------------------------------------------
        # Page 1
        # ------------------------------------------------------------------
        ui.nav_panel(
            "1 · Historical Performance",
            ui.p(
                "Standalone historical performance for every canonical model. "
                "A model's ATS side is the direction of its projected margin relative to the market line.",
                class_="muted",
            ),
            ui.layout_columns(
                ui.value_box("Historical games", ui.output_text("hist_games")),
                ui.value_box("Seasons", ui.output_text("hist_seasons")),
                ui.value_box("Models", ui.output_text("hist_models")),
                ui.value_box("Model predictions", ui.output_text("hist_predictions")),
                col_widths=(3, 3, 3, 3),
            ),
            ui.card(
                ui.card_header("All-model historical scorecard"),
                ui.output_data_frame("historical_model_table"),
            ),
            ui.card(
                ui.card_header("Season-by-season model performance"),
                ui.output_data_frame("historical_season_table"),
            ),
        ),

        # ------------------------------------------------------------------
        # Page 2
        # ------------------------------------------------------------------
        ui.nav_panel(
            "2 · Upcoming Games",
            ui.p(
                "Raw PredictionTracker board only. This page does not decide what to bet; "
                "it shows the current line and every model projection we can map.",
                class_="muted",
            ),
            ui.layout_columns(
                ui.input_numeric("current_season", "Season", 2026, min=2025, max=2030, step=1),
                ui.input_numeric("current_week", "Week", 1, min=0, max=22, step=1),
                ui.input_task_button(
                    "refresh_upcoming",
                    "Refresh PredictionTracker",
                    label_busy="Refreshing PredictionTracker…",
                    type="primary",
                ),
                col_widths=(3, 3, 6),
            ),
            ui.output_text("upcoming_status"),
            ui.layout_columns(
                ui.value_box("Games", ui.output_text("upcoming_games_n")),
                ui.value_box("Models posting", ui.output_text("upcoming_models_n")),
                ui.value_box("Mapped predictions", ui.output_text("upcoming_predictions_n")),
                ui.value_box("CFB Picker", "Inactive for now"),
                col_widths=(3, 3, 3, 3),
            ),
            ui.card(
                ui.card_header("Upcoming game board"),
                ui.output_data_frame("upcoming_board_table"),
            ),
            ui.card(
                ui.card_header("Raw model projection matrix"),
                ui.p(
                    "Values are projected home-team margins. Positive = home projected ahead; negative = away projected ahead.",
                    class_="muted",
                ),
                ui.output_data_frame("upcoming_matrix_table"),
            ),
        ),

        # ------------------------------------------------------------------
        # Page 3
        # ------------------------------------------------------------------
        ui.nav_panel(
            "3 · Strategy Lab",
            ui.p(
                "A strategy is quantitative: collective expected spread = mean(model spreads), "
                "uncertainty = sample SD(model spreads), and signal = |mean − market| / SD. "
                "A bet is recommended when signal ≥ k and enough selected models are available.",
                class_="muted",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Manual model set"),
                    ui.input_selectize(
                        "manual_models",
                        "Models",
                        choices=ALL_MODEL_CHOICES,
                        selected=DEFAULT_MANUAL_IDS,
                        multiple=True,
                        options={"plugins": ["remove_button"], "placeholder": "Choose models…"},
                    ),
                    ui.layout_columns(
                        ui.input_select(
                            "manual_k", "Decision threshold", choices=K_CHOICES,
                            selected=f"{DEFAULT_K:.2f}",
                        ),
                        ui.input_numeric(
                            "manual_min_available", "Minimum available models",
                            min(4, max(2, len(DEFAULT_MANUAL_IDS))), min=2, max=20, step=1,
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.input_checkbox_group(
                        "manual_seasons",
                        "Backtest seasons",
                        choices=SEASON_CHOICES,
                        selected=[str(y) for y in HISTORICAL_SEASONS],
                        inline=True,
                    ),
                    ui.input_task_button(
                        "run_manual",
                        "Backtest manual set",
                        label_busy="Backtesting…",
                        type="primary",
                    ),
                    ui.input_action_button(
                        "use_manual_strategy",
                        "Use this strategy for upcoming games",
                        class_="btn-sm",
                    ),
                    ui.output_text("manual_status"),
                ),
                ui.card(
                    ui.card_header("Automatic combination screening"),
                    ui.input_radio_buttons(
                        "auto_pool_mode",
                        "Candidate pool",
                        choices={
                            "top": "Top N models automatically",
                            "manual": "Choose candidates manually",
                        },
                        selected="top",
                        inline=True,
                    ),
                    ui.layout_columns(
                        ui.input_numeric("auto_pool_n", "Top N", 20, min=4, max=50, step=1),
                        ui.input_select(
                            "auto_pool_metric", "Rank individual models by",
                            choices={
                                "wilson": "Wilson lower bound",
                                "ats": "ATS %",
                                "roi": "ROI",
                                "mae": "Forecast MAE (lower is better)",
                            },
                            selected="wilson",
                        ),
                        ui.input_numeric(
                            "auto_pool_min_bets", "Min discovery bets/model",
                            25, min=0, max=500, step=5,
                        ),
                        col_widths=(3, 5, 4),
                    ),
                    ui.input_selectize(
                        "auto_models",
                        "Manual candidate list (used only in manual mode)",
                        choices=ALL_MODEL_CHOICES,
                        selected=DEFAULT_AUTO_IDS,
                        multiple=True,
                        options={"plugins": ["remove_button"], "placeholder": "Choose candidate universe…"},
                    ),
                    ui.input_checkbox_group(
                        "auto_history_seasons", "Historical seasons available to screening",
                        choices=SEASON_CHOICES,
                        selected=[str(y) for y in HISTORICAL_SEASONS],
                        inline=True,
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            "auto_holdout_weeks", "Hold out latest chronology weeks",
                            6, min=0, max=20, step=1,
                        ),
                        ui.input_numeric("auto_min_size", "Min set size", 4, min=2, max=20, step=1),
                        ui.input_numeric("auto_max_size", "Max set size", 10, min=2, max=20, step=1),
                        col_widths=(4, 4, 4),
                    ),
                    ui.p(
                        "The holdout uses completed/graded chronology weeks by season/week, not by whole season. "
                        "This lets newer models contribute discovery history and still receive a genuine recent holdout.",
                        class_="muted",
                    ),
                    ui.layout_columns(
                        ui.input_select(
                            "auto_k", "Search threshold", choices=K_CHOICES,
                            selected=f"{DEFAULT_K:.2f}",
                        ),
                        ui.input_numeric("auto_min_available", "Min available", 4, min=2, max=20, step=1),
                        ui.input_numeric("auto_min_bets", "Minimum discovery bets", 50, min=10, max=1000, step=10),
                        col_widths=(4, 4, 4),
                    ),
                    ui.layout_columns(
                        ui.input_select(
                            "auto_rank_metric", "Rank combinations by",
                            choices={"ats": "ATS %", "wilson": "Wilson lower bound", "roi": "ROI"},
                            selected="wilson",
                        ),
                        ui.input_numeric("auto_top_n", "Finalists retained", 25, min=5, max=100, step=5),
                        ui.input_numeric(
                            "auto_max_combinations_m",
                            "Exact-search safety cap (millions)",
                            EXACT_SEARCH_DEFAULT_MAX // 1_000_000,
                            min=1, max=EXACT_SEARCH_HARD_MAX // 1_000_000, step=1,
                        ),
                        col_widths=(4, 4, 4),
                    ),
                    ui.p(
                        "The search is exhaustive, not sampled. Multi-million searches stream in batches and retain only the bounded leaderboard. "
                        "Raise the safety cap deliberately for larger candidate pools; very large searches can take several minutes on Connect Cloud.",
                        class_="muted",
                    ),
                    ui.p(
                        "Automatic screening is restricted to models that are actually posting in the current PredictionTracker week. "
                        "Refresh Page 2 before running the search.",
                        class_="muted",
                    ),
                    ui.output_text("auto_pool_status"),
                    ui.output_data_frame("auto_candidate_table"),
                    ui.output_text("auto_combo_count"),
                    ui.input_task_button(
                        "run_auto",
                        "Find promising combinations",
                        label_busy="Searching combinations…",
                        type="primary",
                    ),
                    ui.output_ui("auto_progress_bar"),
                    ui.output_text("auto_status"),
                ),
                col_widths=(5, 7),
            ),
            ui.layout_columns(
                ui.value_box("Manual bets @ chosen k", ui.output_text("manual_primary_bets")),
                ui.value_box("Manual ATS", ui.output_text("manual_primary_ats")),
                ui.value_box("Manual ROI", ui.output_text("manual_primary_roi")),
                ui.value_box("Selected strategy", ui.output_text("strategy_short")),
                col_widths=(3, 3, 3, 3),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Manual threshold curve"),
                    ui.p("The full 0.25–2.00 SD grid is always shown.", class_="muted"),
                    ui.output_data_frame("manual_threshold_table"),
                ),
                ui.card(
                    ui.card_header("Manual season results at chosen k"),
                    ui.output_data_frame("manual_season_table"),
                ),
                col_widths=(6, 6),
            ),
            ui.card(
                ui.card_header("Promising combinations"),
                ui.p(
                    "The exhaustive search uses one chosen k on discovery weeks only. Finalists are then frozen and evaluated "
                    "on the recent chronological holdout; holdout results never rank the discovery search.",
                    class_="muted",
                ),
                ui.output_data_frame("auto_top_table"),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Inspect and select finalists"),
                    ui.input_numeric("auto_pick_rank", "Inspect search rank", 1, min=1, max=100, step=1),
                    ui.output_text_verbatim("auto_selected_models"),
                    ui.hr(),
                    ui.input_selectize(
                        "auto_portfolio_ranks",
                        "Finalist ranks for upcoming predictions",
                        choices={str(i): f"Rank {i}" for i in range(1, 101)},
                        selected=["1", "2", "3"],
                        multiple=True,
                        options={"plugins": ["remove_button"], "placeholder": "Choose one or more finalist ranks…"},
                    ),
                    ui.input_action_button(
                        "use_auto_strategy",
                        "Use selected finalist portfolio",
                        class_="btn-sm",
                    ),
                ),
                ui.card(
                    ui.card_header("Threshold robustness on holdout for chosen finalist"),
                    ui.output_data_frame("auto_threshold_detail"),
                ),
                col_widths=(5, 7),
            ),
            ui.card(
                ui.card_header("Finalist robustness summary"),
                ui.output_data_frame("auto_robust_summary"),
            ),
            ui.card(
                ui.card_header("Finalist performance by market-spread scale"),
                ui.p(
                    "Diagnostic only: finalists keep the same discovery ranking and chosen k. "
                    "This table shows whether a combination's historical performance changes between small spreads, moderate favorites, and very large favorites.",
                    class_="muted",
                ),
                ui.output_data_frame("auto_scale_table"),
            ),
        ),

        # ------------------------------------------------------------------
        # Page 4
        # ------------------------------------------------------------------
        ui.nav_panel(
            "4 · Upcoming Predictions",
            ui.div(ui.output_text("strategy_banner"), class_="strategy-banner"),
            ui.p(
                "Applies every finalist combination selected on Page 3 to the same cached PredictionTracker slate. "
                "Each combination independently calculates its collective expected spread, model SD, k×SD boundary, and BET/PASS decision.",
                class_="muted",
            ),
            ui.card(
                ui.card_header("Patrick's recommended settings"),
                ui.p(
                    "One-click current-week recipe: Top 26 currently posting models by discovery Wilson lower bound; "
                    "latest 6 completed weeks with adequate candidate-model coverage held out; combination sizes 4–7; 0.75 SD search anchor; "
                    "top 19 discovery-ranked combinations frozen, then each finalist gets an automatic stable k from 0.25–2.00 SD. "
                    "Near-duplicate combinations are collapsed into overlap communities for the final META estimate and backtest. Models inside every individual ensemble are equal-weighted.",
                    class_="muted",
                ),
                ui.input_action_button(
                    "run_patrick",
                    "Run Patrick's recommended settings",
                    class_="btn-primary",
                ),
                ui.output_ui("patrick_progress_bar"),
                ui.output_text("patrick_status"),
            ),
            ui.p("Or apply a portfolio you selected manually on Page 3:", class_="muted"),
            ui.input_task_button(
                "apply_strategy_current",
                "Apply selected finalist portfolio",
                label_busy="Scoring combinations…",
                type="primary",
            ),
            ui.output_text("strategy_current_status"),
            ui.layout_columns(
                ui.value_box("Selected combos", ui.output_text("strategy_combo_n")),
                ui.value_box("Unique models", ui.output_text("strategy_model_n")),
                ui.value_box("Required k", ui.output_text("strategy_k")),
                ui.value_box("Games scored", ui.output_text("strategy_games_n")),
                col_widths=(3, 3, 3, 3),
            ),
            ui.card(
                ui.card_header("Final consensus: overlap, automatic k, and META backtest"),
                ui.p(
                    "The top combinations are often close relatives. A 0.60 Jaccard overlap groups near-duplicates into communities so one core model family cannot masquerade as many independent votes. "
                    "Each finalist's k is selected from 0.25–2.00 SD using discovery data only; the six completed-week holdout remains untouched until those choices are frozen. "
                    "The diversified META forecast gives each overlap community equal influence and uses a consensus-uncertainty scale based on uncertainty of ensemble means plus between-community disagreement, avoiding the overly conservative double-counting of raw within-ensemble SD.",
                    class_="muted",
                ),
                ui.layout_columns(
                    ui.value_box("Independent communities", ui.output_text("committee_community_n")),
                    ui.value_box("Mean pairwise overlap", ui.output_text("committee_mean_overlap")),
                    ui.value_box("META k", ui.output_text("committee_meta_k")),
                    ui.value_box("Holdout META ATS", ui.output_text("committee_holdout_ats")),
                    col_widths=(3, 3, 3, 3),
                ),
                ui.p(ui.strong("Backtest split: "), ui.output_text("committee_holdout_window"), class_="muted"),
                ui.h5("Final META backtest"),
                ui.output_data_frame("committee_meta_backtest_table"),
                ui.h5("Automatic k by finalist"),
                ui.output_data_frame("committee_combo_k_table"),
                ui.h5("Overlap communities"),
                ui.output_data_frame("committee_overlap_table"),
            ),
            ui.card(
                ui.card_header("Absolute spread trust check"),
                ui.p(
                    "Large favorites and underdogs are a distinct forecasting regime. This diagnostic keeps the frozen Diversified META strategy unchanged, then asks how its forecast accuracy and betting performance vary with |market spread|. "
                    "The tail is split into 22–27.5, 28–34.5, and 35+ so a -38 game is not pooled with an ordinary -23 favorite. Negative ΔMAE vs market means META was closer to the final margin than the market on average.",
                    class_="muted",
                ),
                ui.h5("Current slate mapped to historical spread regimes"),
                ui.output_data_frame("committee_current_spread_context_table"),
                ui.h5("Diversified META performance by |market spread|"),
                ui.output_data_frame("committee_meta_spread_table"),
                ui.p(
                    "These rows are diagnostics, not new optimization rules. The same discovery-selected META k is used in every bucket. Favorite/underdog columns classify the side actually bet, which is especially important for very large spreads.",
                    class_="muted",
                ),
            ),
            ui.card(
                ui.card_header("Line shopping / alternate market"),
                ui.p(
                    "Test a sportsbook line without refreshing the slate or rerunning the combination search. "
                    "Enter the home team's spread exactly as displayed by the book: negative = home favorite, positive = home underdog. "
                    "Example: NC State @ Virginia → enter -3.0 to test Virginia -3.",
                    class_="muted",
                ),
                ui.layout_columns(
                    ui.input_select(
                        "line_override_game",
                        "Game",
                        choices={"": "Apply the portfolio first"},
                        selected="",
                    ),
                    ui.input_numeric(
                        "line_override_value",
                        "Available home-team spread",
                        0.0,
                        min=-80,
                        max=80,
                        step=0.5,
                    ),
                    col_widths=(8, 4),
                ),
                ui.output_text("line_override_prompt"),
                ui.div(
                    ui.input_action_button("apply_line_override", "Use this line", class_="btn-sm btn-primary"),
                    ui.input_action_button("reset_line_override", "Reset this game", class_="btn-sm"),
                    ui.input_action_button("clear_line_overrides", "Reset all lines", class_="btn-sm"),
                    style="display:flex; gap:.5rem; flex-wrap:wrap; margin:.4rem 0;",
                ),
                ui.output_text("line_override_status"),
                ui.output_text("line_override_game_result"),
            ),
            ui.card(
                ui.card_header("Combination agreement by game"),
                ui.p(
                    "Counts how many selected finalist combinations independently produce a bet in each direction. "
                    "Raw portfolio mean ± SD is retained for comparison; diversified META collapses near-duplicate combinations into equal-weight overlap communities. "
                    "Each C1/C2/etc. uses its own discovery-selected k when automatic threshold tuning is available.",
                    class_="muted",
                ),
                ui.output_data_frame("strategy_combo_summary_table"),
            ),
            ui.card(
                ui.card_header("Per-combination expected spreads"),
                ui.p(
                    "One row per game × finalist. Mean ± SD describes the model set itself; the k×SD interval is the actual decision boundary.",
                    class_="muted",
                ),
                ui.output_data_frame("strategy_combo_detail_table"),
            ),
            ui.card(
                ui.card_header("Selected-model projections by game"),
                ui.output_data_frame("strategy_model_predictions_table"),
            ),
        ),

        # ------------------------------------------------------------------
        # Page 5
        # ------------------------------------------------------------------
        ui.nav_panel(
            "5 · Forecast Plots",
            ui.p(
                "Visualizes the complete current forecast hierarchy for one game at a time: every mapped model posting for the game → "
                "every selected finalist ensemble → the diversity-adjusted META estimate. Models do not need to belong to a finalist to appear. Alternate lines entered on Page 4 are reflected automatically.",
                class_="muted",
            ),
            ui.layout_columns(
                ui.input_select(
                    "plot_game",
                    "Game",
                    choices={"": "Apply a portfolio on Page 4 first"},
                    selected="",
                ),
                ui.input_radio_buttons(
                    "plot_style",
                    "Plot style",
                    choices={
                        "hierarchy": "Forecast hierarchy",
                        "distribution": "Distribution / rug (legacy style)",
                    },
                    selected="hierarchy",
                    inline=True,
                ),
                col_widths=(7, 5),
            ),
            ui.output_text("forecast_plot_status"),
            ui.card(
                ui.card_header("Complete current-game forecast hierarchy"),
                ui.output_plot("forecast_plot", height="1500px"),
            ),
            ui.card(
                ui.card_header("What each layer means"),
                ui.tags.ul(
                    ui.tags.li(ui.strong("Individual models"), ": every mapped model currently posting for that game gets its own labeled row, including models not selected into C1–C12. Models used by at least one finalist are highlighted, with the number of finalist ensembles using each model shown at right."),
                    ui.tags.li(ui.strong("C1, C2, ..."), ": every finalist ensemble gets its own row. The thin interval is ±1 within-ensemble SD; the heavier interval is that finalist's actual ±k×SD decision band."),
                    ui.tags.li(ui.strong("META"), ": the diversity-adjusted final consensus. Near-duplicate finalist sets are collapsed into overlap communities, each community gets equal influence, and both ±1 consensus SD and the frozen META ±k×SD decision band are shown."),
                    ui.tags.li(ui.strong("Market"), ": the line currently being used on Page 4, including any session-only line-shopping override. If overridden, the original PredictionTracker line is also shown."),
                ),
                ui.p(
                    "C1–C12 retain their own BET/PASS rules. META now also has its own discovery-selected k and an independently backtested BET/PASS rule shown on Page 4.",
                    class_="muted",
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    strategy = reactive.Value(dict(SAVED_STRATEGY))
    # Session-only sportsbook line overrides. Values use the app's internal
    # market_home_margin convention (positive = home favored). They never
    # overwrite PredictionTracker's cached line or another visitor's session.
    line_overrides = reactive.Value({})
    auto_progress = {
        "done": 0, "total": 0, "label": "", "phase": "idle",
        "started": None, "updated": None,
    }
    auto_progress_lock = threading.Lock()
    patrick_state = reactive.Value({
        "phase": "idle",
        "message": "Refresh Page 2, then run the one-click recommended recipe.",
    })

    def set_auto_progress(**kwargs):
        with auto_progress_lock:
            auto_progress.update(kwargs)

    def get_auto_progress():
        with auto_progress_lock:
            return dict(auto_progress)

    # ------------------------------------------------------------------
    # Page 1: historical performance
    # ------------------------------------------------------------------
    @render.text
    def hist_games():
        return f"{DATA['game_key'].astype(str).nunique():,}"

    @render.text
    def hist_seasons():
        return f"{len(HISTORICAL_SEASONS):,}"

    @render.text
    def hist_models():
        return f"{len(MODELS):,}"

    @render.text
    def hist_predictions():
        return f"{len(DATA):,}"

    @render.data_frame
    def historical_model_table():
        d = INDIVIDUAL_HISTORY["overall"].copy()
        if d.empty:
            return render.DataGrid(pd.DataFrame())
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low"])
        rename = {
            "rank": "Rank",
            "model_name": "Model",
            "predictions": "Predictions",
            "bets": "Bets",
            "wins": "Wins",
            "losses": "Losses",
            "pushes": "Pushes",
            "ats_pct": "ATS %",
            "roi": "ROI %",
            "wilson_low": "Wilson LB %",
            "units": "Units",
            "mae": "MAE (pts)",
            "bias": "Bias (pts)",
            "seasons": "Seasons",
        }
        cols = [c for c in rename if c in d.columns]
        d = d[cols].rename(columns=rename)
        return render.DataGrid(d, filters=True, height="620px")

    @render.data_frame
    def historical_season_table():
        d = INDIVIDUAL_HISTORY["by_season"].copy()
        if d.empty:
            return render.DataGrid(pd.DataFrame())
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low"])
        rename = {
            "season": "Season",
            "model_name": "Model",
            "predictions": "Predictions",
            "bets": "Bets",
            "wins": "Wins",
            "losses": "Losses",
            "ats_pct": "ATS %",
            "roi": "ROI %",
            "wilson_low": "Wilson LB %",
            "units": "Units",
            "mae": "MAE (pts)",
            "bias": "Bias (pts)",
        }
        cols = [c for c in rename if c in d.columns]
        d = d[cols].rename(columns=rename)
        return render.DataGrid(d, filters=True, height="600px")

    # ------------------------------------------------------------------
    # Page 2: PredictionTracker raw upcoming board
    # ------------------------------------------------------------------
    @ui.bind_task_button(button_id="refresh_upcoming")
    @reactive.extended_task
    async def upcoming_task(season: int, week: int):
        def compute():
            return refresh_and_build_current_week(
                PROJECT_ROOT,
                DATA,
                ALL_MODEL_IDS,
                MODEL_NAME_MAP,
                season=int(season),
                week=int(week),
                primary_k=0.25,
                min_available_models=2,
                refresh=True,
                include_cfbpicker=False,
                write_outputs=False,
            )
        return await asyncio.to_thread(compute)

    @reactive.effect
    @reactive.event(input.refresh_upcoming)
    def start_upcoming():
        upcoming_task(int(input.current_season()), int(input.current_week()))

    def upcoming_result():
        if upcoming_task.status() != "success":
            return None
        return upcoming_task.result()

    def current_week_available_model_ids():
        """Canonical models with at least one actual PT projection for the active week."""
        r = upcoming_result()
        if r is None:
            return set(), False, "Refresh PredictionTracker on Page 2 first."
        b = r.get("board", pd.DataFrame())
        if len(b):
            sy = pd.to_numeric(b.get("season"), errors="coerce").dropna().astype(int).unique().tolist()
            wk = pd.to_numeric(b.get("week"), errors="coerce").dropna().astype(int).unique().tolist()
            if sy and wk and (sy[0] != int(input.current_season()) or wk[0] != int(input.current_week())):
                return set(), False, "Page 2 was refreshed for a different season/week; refresh it again."
        p = r.get("predictions", pd.DataFrame())
        if p.empty or "canonical_model_id" not in p.columns:
            return set(), True, "PredictionTracker loaded, but no mapped models have posted projections."
        ids = set(p["canonical_model_id"].dropna().astype(str))
        return ids, True, f"{len(ids)} canonical models are posting in the active PredictionTracker week."

    @render.text
    def upcoming_status():
        s = upcoming_task.status()
        if s == "initial":
            return "Refresh PredictionTracker to load the upcoming board."
        if s == "running":
            return "Refreshing PredictionTracker…"
        if s == "success":
            r = upcoming_result()
            live = len(r.get("pt_live_models", pd.DataFrame())) if r else 0
            return f"PredictionTracker loaded. {live} model columns are currently posting. CFB Picker was not queried."
        if s == "error":
            return "PredictionTracker refresh failed."
        return str(s)

    @reactive.effect
    def show_upcoming_error():
        if upcoming_task.status() == "error":
            try:
                upcoming_task.result()
            except Exception as exc:
                ui.notification_show(f"Upcoming-board error: {exc}", type="error", duration=12)

    @render.text
    def upcoming_games_n():
        r = upcoming_result()
        return "—" if r is None else f"{len(r.get('board', pd.DataFrame())):,}"

    @render.text
    def upcoming_models_n():
        r = upcoming_result()
        if r is None:
            return "—"
        p = r.get("predictions", pd.DataFrame())
        return f"{p['canonical_model_id'].astype(str).nunique():,}" if len(p) else "0"

    @render.text
    def upcoming_predictions_n():
        r = upcoming_result()
        if r is None:
            return "—"
        return f"{len(r.get('predictions', pd.DataFrame())):,}"

    @render.data_frame
    def upcoming_board_table():
        r = upcoming_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("board", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d["Game"] = d["away"].astype(str) + " @ " + d["home"].astype(str)
        d["Market spread"] = [
            _spread_label(a, h, m)
            for a, h, m in zip(d["away"], d["home"], d["market_home_margin"])
        ]
        d = d[["Game", "Market spread", "available_models", "market_source"]].rename(
            columns={"available_models": "Models posting", "market_source": "Line source"}
        )
        return render.DataGrid(d, filters=True, height="420px")

    @render.data_frame
    def upcoming_matrix_table():
        r = upcoming_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        p = r.get("predictions", pd.DataFrame()).copy()
        b = r.get("board", pd.DataFrame()).copy()
        if p.empty:
            return render.DataGrid(pd.DataFrame())
        p["Game"] = p["away"].astype(str) + " @ " + p["home"].astype(str)
        matrix = p.pivot_table(
            index="Game",
            columns="model_name",
            values="prediction_home_margin",
            aggfunc="first",
        ).reset_index()
        if len(b):
            b["Game"] = b["away"].astype(str) + " @ " + b["home"].astype(str)
            b["Market"] = [
                _spread_label(a, h, m)
                for a, h, m in zip(b["away"], b["home"], b["market_home_margin"])
            ]
            matrix = b[["Game", "Market"]].drop_duplicates("Game").merge(matrix, on="Game", how="left")
        return render.DataGrid(matrix, filters=True, height="650px")

    # ------------------------------------------------------------------
    # Page 3a: manual strategy
    # ------------------------------------------------------------------
    @ui.bind_task_button(button_id="run_manual")
    @reactive.extended_task
    async def manual_task(ids: list[str], seasons: tuple[int, ...], min_n: int):
        cfg = StreamlinedBacktestConfig(
            selection_mode="exact",
            target_seasons=seasons,
            min_available_models=int(min_n),
            evaluation_week_min=1,
            evaluation_week_max=30,
            thresholds=K_GRID,
            standard_price=-110,
        )

        def compute():
            return run_streamlined_backtest(DATA, ids, cfg)
        return await asyncio.to_thread(compute)

    @reactive.effect
    @reactive.event(input.run_manual)
    def start_manual():
        ids = list(input.manual_models() or [])
        seasons = tuple(sorted(int(x) for x in list(input.manual_seasons() or [])))
        min_n = int(input.manual_min_available())
        if not ids:
            ui.notification_show("Choose at least one manual model.", type="error")
            return
        if not seasons:
            ui.notification_show("Choose at least one backtest season.", type="error")
            return
        if min_n > len(ids):
            ui.notification_show("Minimum available models cannot exceed the selected set size.", type="error")
            return
        manual_task(ids, seasons, min_n)

    def manual_result():
        if manual_task.status() != "success":
            return None
        return manual_task.result()

    @render.text
    def manual_status():
        s = manual_task.status()
        if s == "initial":
            return "The backtest will report every threshold from 0.25 to 2.00 SD."
        if s == "running":
            return "Manual backtest running…"
        if s == "success":
            return "Manual backtest complete."
        if s == "error":
            return "Manual backtest failed."
        return str(s)

    @reactive.effect
    def show_manual_error():
        if manual_task.status() == "error":
            try:
                manual_task.result()
            except Exception as exc:
                ui.notification_show(f"Manual backtest error: {exc}", type="error", duration=12)

    def manual_primary_row():
        r = manual_result()
        if r is None:
            return None
        d = r.get("summary", pd.DataFrame())
        if d.empty:
            return None
        k = float(input.manual_k())
        q = d[np.isclose(pd.to_numeric(d["k"], errors="coerce"), k)]
        return None if q.empty else q.iloc[0]

    @render.text
    def manual_primary_bets():
        row = manual_primary_row()
        return "—" if row is None else f"{int(row['bets']):,}"

    @render.text
    def manual_primary_ats():
        row = manual_primary_row()
        return "—" if row is None or not np.isfinite(row["ats_pct"]) else f"{100*float(row['ats_pct']):.1f}%"

    @render.text
    def manual_primary_roi():
        row = manual_primary_row()
        return "—" if row is None or not np.isfinite(row["roi"]) else f"{100*float(row['roi']):+.1f}%"

    @render.data_frame
    def manual_threshold_table():
        r = manual_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("summary", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d = _pct_frame(d, ["ats_pct", "roi"])
        d = d.rename(columns={
            "k": "k (SD)", "bets": "Bets", "wins": "Wins", "losses": "Losses",
            "pushes": "Pushes", "ats_pct": "ATS %", "units": "Units", "roi": "ROI %",
        })
        return render.DataGrid(d, filters=False, height="380px")

    @render.data_frame
    def manual_season_table():
        r = manual_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("by_season", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        k = float(input.manual_k())
        d = d[np.isclose(pd.to_numeric(d["k"], errors="coerce"), k)].copy()
        d = _pct_frame(d, ["ats_pct", "roi"])
        d = d.rename(columns={
            "season": "Season", "k": "k (SD)", "bets": "Bets", "wins": "Wins",
            "losses": "Losses", "pushes": "Pushes", "ats_pct": "ATS %",
            "units": "Units", "roi": "ROI %",
        })
        return render.DataGrid(d, filters=False, height="380px")

    @reactive.effect
    @reactive.event(input.use_manual_strategy)
    def select_manual_strategy():
        ids = list(input.manual_models() or [])
        if not ids:
            ui.notification_show("Choose manual models first.", type="error")
            return
        min_n = int(input.manual_min_available())
        if min_n > len(ids):
            ui.notification_show("Minimum available models exceeds the selected set size.", type="error")
            return
        k = float(input.manual_k())
        combos = [{"rank": 1, "model_ids": ids}]
        state = {
            "source": "manual",
            "label": f"Manual set: {len(ids)} models @ {k:.2f} SD",
            "model_ids": ids,
            "combinations": combos,
            "primary_k": k,
            "min_available_models": min_n,
        }
        strategy.set(state)
        if not CLOUD_MODE:
            save_current_selection(
                PROJECT_ROOT, ids, MODEL_NAME_MAP,
                season=int(input.current_season()), week=int(input.current_week()),
                primary_k=k, min_available_models=min_n, combinations=combos,
            )
        ui.notification_show("Manual strategy selected for Page 4.", type="message")

    # ------------------------------------------------------------------
    # Page 3b: automatic combination discovery
    # ------------------------------------------------------------------
    @reactive.calc
    def selected_auto_periods():
        seasons = tuple(sorted(int(x) for x in list(input.auto_history_seasons() or [])))
        periods = tuple((y, w) for y, w in HISTORICAL_PERIODS if y in set(seasons))
        holdout_n = max(0, int(input.auto_holdout_weeks()))
        if holdout_n <= 0:
            return seasons, periods, periods, tuple()
        if holdout_n >= len(periods):
            return seasons, periods, tuple(), periods
        return seasons, periods, periods[:-holdout_n], periods[-holdout_n:]

    def period_subset(periods: tuple[tuple[int, int], ...]) -> pd.DataFrame:
        if not periods:
            return DATA.iloc[0:0].copy()
        wanted = set((int(y), int(w)) for y, w in periods)
        ss = pd.to_numeric(DATA["season"], errors="coerce")
        ww = pd.to_numeric(DATA["week"], errors="coerce")
        mask = pd.Series(
            [
                (int(y), int(w)) in wanted if pd.notna(y) and pd.notna(w) else False
                for y, w in zip(ss, ww)
            ],
            index=DATA.index,
        )
        return DATA.loc[mask].copy()

    def candidate_period_coverage(
        periods: tuple[tuple[int, int], ...],
        candidate_ids: list[str],
        min_available_models: int,
    ) -> pd.DataFrame:
        """Count graded games per chronology period with enough candidate forecasts.

        This is used by Patrick's one-click recipe so a nominal holdout cannot
        silently land on bowl/postseason weeks where the currently relevant
        model family has almost no simultaneous coverage.
        """
        if not periods or not candidate_ids:
            return pd.DataFrame(columns=["season", "week", "graded_games", "scorable_games"])
        wanted = set((int(y), int(w)) for y, w in periods)
        z = DATA.copy()
        yy = pd.to_numeric(z["season"], errors="coerce")
        ww = pd.to_numeric(z["week"], errors="coerce")
        pred = pd.to_numeric(z.get("prediction_margin"), errors="coerce")
        actual = pd.to_numeric(z.get("actual_margin"), errors="coerce")
        market = pd.to_numeric(z.get("market_margin"), errors="coerce")
        mask = pd.Series([
            (int(y), int(w)) in wanted if pd.notna(y) and pd.notna(w) else False
            for y, w in zip(yy, ww)
        ], index=z.index)
        mask &= z["canonical_model_id"].astype(str).isin(set(map(str, candidate_ids)))
        mask &= pred.notna() & actual.notna() & market.notna()
        z = z.loc[mask, ["game_key", "season", "week", "canonical_model_id"]].copy()
        if z.empty:
            return pd.DataFrame(columns=["season", "week", "graded_games", "scorable_games"])
        per_game = (
            z.groupby(["season", "week", "game_key"], as_index=False)["canonical_model_id"]
            .nunique()
            .rename(columns={"canonical_model_id": "available_models"})
        )
        per_game["scorable"] = per_game["available_models"] >= int(min_available_models)
        out = (
            per_game.groupby(["season", "week"], as_index=False)
            .agg(graded_games=("game_key", "nunique"), scorable_games=("scorable", "sum"))
        )
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype(int)
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype(int)
        return out.sort_values(["season", "week"]).reset_index(drop=True)

    def resolve_ranked_live_candidates(
        search_periods: tuple[tuple[int, int], ...],
        live_ids: set[str],
        *,
        pool_n: int,
        pool_metric: str,
        pool_min_bets: int,
    ):
        """Rank only current-week posting models using discovery history."""
        discovery_data = period_subset(search_periods)
        hist = individual_model_performance(discovery_data, standard_price=-110).get("overall", pd.DataFrame()).copy()
        if hist.empty:
            return [], hist
        hist["canonical_model_id"] = hist["canonical_model_id"].astype(str)
        hist = hist[hist["canonical_model_id"].isin(ALL_MODEL_IDS)].copy()
        hist = hist[hist["canonical_model_id"].isin(live_ids)].copy()
        hist = hist[pd.to_numeric(hist["bets"], errors="coerce").fillna(0) >= int(pool_min_bets)].copy()
        if str(pool_metric) == "mae":
            hist = hist.sort_values(["mae", "bets", "wilson_low"], ascending=[True, False, False], na_position="last")
        else:
            col = {"ats": "ats_pct", "roi": "roi", "wilson": "wilson_low"}.get(str(pool_metric), "wilson_low")
            hist = hist.sort_values([col, "bets", "wilson_low"], ascending=[False, False, False], na_position="last")
        hist = hist.head(max(1, int(pool_n))).reset_index(drop=True)
        ids = hist["canonical_model_id"].astype(str).tolist()
        return ids, hist

    def resolve_auto_candidates(search_periods: tuple[tuple[int, int], ...]):
        live_ids, live_ready, live_message = current_week_available_model_ids()
        if not live_ready:
            return [], pd.DataFrame(), {
                "live_ready": False, "live_count": 0, "message": live_message, "excluded": 0
            }

        mode = str(input.auto_pool_mode())
        if mode == "manual":
            requested = [str(x) for x in list(input.auto_models() or [])]
            ids = [x for x in requested if x in live_ids]
            return ids, pd.DataFrame(), {
                "live_ready": True, "live_count": len(live_ids), "message": live_message,
                "excluded": len(requested) - len(ids),
            }

        ids, hist = resolve_ranked_live_candidates(
            search_periods, live_ids,
            pool_n=int(input.auto_pool_n()),
            pool_metric=str(input.auto_pool_metric()),
            pool_min_bets=int(input.auto_pool_min_bets()),
        )
        return ids, hist, {
            "live_ready": True, "live_count": len(live_ids), "message": live_message,
            "excluded": 0,
        }

    @reactive.calc
    def auto_candidate_resolution():
        return resolve_auto_candidates(selected_auto_periods()[2])

    @render.text
    def auto_pool_status():
        seasons, all_periods, search_periods, val_periods = selected_auto_periods()
        ids, ranked, availability_meta = auto_candidate_resolution()
        if not seasons:
            return "Choose at least one historical season."
        if not search_periods:
            return "The requested holdout consumes the full historical window; reduce held-out weeks."
        def fmt_period(p):
            return f"{p[0]} W{p[1]}"
        discovery = f"{fmt_period(search_periods[0])}–{fmt_period(search_periods[-1])}"
        holdout = "none" if not val_periods else f"{fmt_period(val_periods[0])}–{fmt_period(val_periods[-1])}"
        if not availability_meta.get("live_ready", False):
            return availability_meta.get("message", "Refresh Page 2 before screening.")
        source = "automatic top-N" if str(input.auto_pool_mode()) == "top" else "manual"
        extra = ""
        if str(input.auto_pool_mode()) == "manual" and availability_meta.get("excluded", 0):
            extra = f" · {availability_meta['excluded']} manually requested models excluded because they are not posting this week"
        return (
            f"Resolved pool: {len(ids)} models ({source}) from {availability_meta.get('live_count', 0)} currently posting models "
            f"· discovery {discovery} · holdout {holdout}{extra}."
        )

    @render.data_frame
    def auto_candidate_table():
        _, _, search_periods, _ = selected_auto_periods()
        ids, ranked, _ = auto_candidate_resolution()
        if not ids:
            return render.DataGrid(pd.DataFrame())
        if ranked is None or ranked.empty:
            d = pd.DataFrame({
                "Model": [MODEL_NAME_MAP.get(mid, mid) for mid in ids],
                "Model ID": ids,
            })
            return render.DataGrid(d, filters=False, height="220px")
        d = ranked.copy()
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low"])
        d.insert(0, "Pool rank", np.arange(1, len(d) + 1))
        rename = {
            "model_name": "Model", "bets": "Discovery bets",
            "ats_pct": "ATS %", "roi": "ROI %",
            "wilson_low": "Wilson LB %", "mae": "MAE (pts)",
            "seasons": "Seasons represented",
        }
        cols = [
            c for c in ["Pool rank", "model_name", "bets", "ats_pct", "roi", "wilson_low", "mae", "seasons"]
            if c in d.columns
        ]
        # Pool rank was inserted with its display name already.
        if "Pool rank" not in cols:
            cols.insert(0, "Pool rank")
        return render.DataGrid(d[cols].rename(columns=rename), filters=False, height="250px")

    @render.text
    def auto_combo_count():
        _, _, search_periods, _ = selected_auto_periods()
        ids, _, _ = auto_candidate_resolution()
        if not ids:
            return "0 combinations"
        lo, hi = int(input.auto_min_size()), int(input.auto_max_size())
        if hi < lo:
            return "Maximum set size must be at least the minimum."
        total = combination_count(len(ids), lo, hi)
        cap = max(1_000_000, min(EXACT_SEARCH_HARD_MAX, int(float(input.auto_max_combinations_m()) * 1_000_000)))
        status = "within cap" if total <= cap else f"ABOVE {cap:,} cap"
        return (
            f"{total:,} exact combinations from {len(ids)} resolved candidates · "
            f"safety cap {cap:,} ({status})"
        )

    @ui.bind_task_button(button_id="run_auto")
    @reactive.extended_task
    async def auto_task(ids: list[str], config_values: dict, robustness_periods: tuple[tuple[int, int], ...]):
        cfg = CombinationSearchConfig(**config_values)

        def compute():
            last_console = [0.0]
            def progress(done, total, label):
                now = time.monotonic()
                set_auto_progress(
                    done=int(done), total=int(total), label=str(label),
                    phase="Exact combination search", updated=now,
                )
                # Also print periodic progress to the terminal running Shiny.
                if done >= total or now - last_console[0] >= 2.0:
                    pct = 100.0 * done / total if total else 0.0
                    print(f"[Strategy Lab] {done:,}/{total:,} ({pct:.1f}%) · {label}", flush=True)
                    last_console[0] = now

            result = brute_force_combination_search(
                DATA, ids, MODEL_NAME_MAP, cfg, progress_callback=progress
            )
            set_auto_progress(
                done=int(result.get("evaluated_combinations", 0)),
                total=int(result.get("total_combinations", 0)),
                label="Exact search complete; stress-testing frozen finalists across k values…",
                phase="Finalist holdout robustness",
                updated=time.monotonic(),
            )
            result["robustness"] = combination_threshold_robustness(
                DATA,
                result,
                seasons=tuple(config_values["search_seasons"]),
                periods=robustness_periods,
                thresholds=K_GRID,
                top_n=int(config_values["top_n"]),
            )
            result["scale_performance"] = combination_spread_scale_performance(
                DATA,
                result,
                discovery_periods=tuple(config_values.get("search_periods", ())),
                validation_periods=tuple(config_values.get("validation_periods", ())),
                top_n=int(config_values["top_n"]),
            )
            set_auto_progress(phase="Complete", label="Search and finalist validation complete.", updated=time.monotonic())
            return result
        return await asyncio.to_thread(compute)

    @reactive.effect
    @reactive.event(input.run_auto)
    def start_auto():
        seasons, all_periods, search_periods, val_periods = selected_auto_periods()
        ids, ranked, availability_meta = auto_candidate_resolution()
        lo, hi = int(input.auto_min_size()), int(input.auto_max_size())
        min_n = int(input.auto_min_available())

        if not seasons:
            ui.notification_show("Choose at least one historical season.", type="error")
            return
        if not availability_meta.get("live_ready", False):
            ui.notification_show("Refresh PredictionTracker on Page 2 before automatic combination screening.", type="error")
            return
        if not search_periods:
            ui.notification_show("Reduce held-out weeks; no discovery weeks remain.", type="error")
            return
        if not ids:
            ui.notification_show("No candidate models satisfy the current pool settings.", type="error")
            return
        if hi < lo:
            ui.notification_show("Maximum set size must be at least minimum set size.", type="error")
            return
        if lo < min_n:
            ui.notification_show("Minimum set size must be at least the minimum-available gate.", type="error")
            return
        if len(ids) < lo:
            ui.notification_show(f"Only {len(ids)} candidate models resolved; reduce minimum set size or broaden the pool.", type="error")
            return

        total = combination_count(len(ids), lo, hi)
        max_combinations = max(
            1_000_000,
            min(EXACT_SEARCH_HARD_MAX, int(float(input.auto_max_combinations_m()) * 1_000_000)),
        )
        if total > max_combinations:
            ui.notification_show(
                f"{total:,} combinations exceeds your {max_combinations:,} exact-search safety cap. "
                "Raise the cap or reduce Top N / the size range.",
                type="error", duration=12,
            )
            return

        values = {
            "search_seasons": seasons,
            "validation_seasons": seasons if val_periods else (),
            "search_periods": search_periods,
            "validation_periods": val_periods,
            "min_size": lo,
            "max_size": hi,
            "primary_k": float(input.auto_k()),
            "min_available_models": min_n,
            "min_search_bets": int(input.auto_min_bets()),
            "min_seasons_represented": 1,
            "min_distinct_weeks": 1,
            "ranking_metric": str(input.auto_rank_metric()),
            "standard_price": -110,
            "chunk_size": 512,
            "top_n": int(input.auto_top_n()),
            "max_combinations": max_combinations,
        }
        now = time.monotonic()
        set_auto_progress(
            done=0, total=total,
            label=f"Resolved {len(ids)} candidates; preparing discovery matrix…",
            phase="Preparing", started=now, updated=now,
        )
        print(
            f"[Strategy Lab] starting exact search: {len(ids)} candidates, "
            f"sizes {lo}–{hi}, {total:,} combinations, k={float(input.auto_k()):.2f}",
            flush=True,
        )
        if total > 5_000_000:
            ui.notification_show(
                f"Launching an exact {total:,}-combination search. Keep this session open; progress and ETA will update below.",
                type="message", duration=10,
            )
        auto_task(ids, values, val_periods if val_periods else search_periods)

    def auto_result():
        if auto_task.status() != "success":
            return None
        return auto_task.result()

    @render.ui
    def auto_progress_bar():
        s = auto_task.status()
        if s not in {"running", "success"}:
            return ui.div()
        if s == "running":
            reactive.invalidate_later(0.4)
        p = get_auto_progress()
        total = int(p.get("total") or 0)
        done = int(p.get("done") or 0)
        pct = 100.0 * done / total if total else 0.0
        if s == "success":
            pct = 100.0
        return ui.div(
            ui.div(class_="search-progress-fill", style=f"width: {max(0.0, min(100.0, pct)):.1f}%"),
            class_="search-progress-track",
        )

    @render.text
    def auto_status():
        s = auto_task.status()
        if s == "initial":
            return "Choose Top N (or a manual candidate pool), a holdout length, and a search threshold."
        if s == "running":
            reactive.invalidate_later(0.4)
            p = get_auto_progress()
            done = int(p.get("done") or 0)
            total = int(p.get("total") or 0)
            started = p.get("started")
            elapsed = max(0.0, time.monotonic() - started) if started else 0.0
            pct = 100.0 * done / total if total else 0.0
            rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 and total >= done else np.nan
            eta = f" · est. {remaining:.0f}s remaining" if np.isfinite(remaining) and remaining > 1 else ""
            return (
                f"{p.get('phase', 'Running')}: {done:,}/{total:,} ({pct:.1f}%) · "
                f"elapsed {elapsed:.1f}s{eta} · {p.get('label', '')}"
            )
        if s == "success":
            r = auto_result()
            return (
                f"Search complete: {int(r.get('evaluated_combinations', 0)):,} evaluated; "
                f"{int(r.get('eligible_combinations', 0)):,} met the discovery confidence gates. "
                "Finalists were then evaluated on the held-out recent weeks."
            )
        if s == "error":
            return "Automatic combination search failed."
        return str(s)

    @reactive.effect
    def show_auto_error():
        if auto_task.status() == "error":
            try:
                auto_task.result()
            except Exception as exc:
                ui.notification_show(f"Combination-search error: {exc}", type="error", duration=14)

    @reactive.effect
    @reactive.event(input.run_patrick)
    def start_patrick_recommended():
        if auto_task.status() == "running":
            ui.notification_show("A combination search is already running.", type="error")
            return
        live_ids, live_ready, _ = current_week_available_model_ids()
        if not live_ready:
            ui.notification_show("Refresh PredictionTracker on Page 2 first.", type="error")
            return
        if not live_ids:
            ui.notification_show("No mapped models are currently posting.", type="error")
            return

        seasons = tuple(HISTORICAL_SEASONS)
        periods = tuple((y, w) for y, w in HISTORICAL_PERIODS if y in set(seasons))
        if len(periods) <= PATRICK_HOLDOUT_WEEKS:
            ui.notification_show("Not enough historical weeks for the 6-week holdout.", type="error")
            return
        # First pass uses the ordinary chronological split only to obtain a
        # provisional candidate pool. Then choose the latest six *usable*
        # completed weeks for that pool and rerank candidates on discovery
        # data strictly before the holdout. Sparse later periods are excluded
        # rather than leaking back into discovery.
        provisional_search = periods[:-PATRICK_HOLDOUT_WEEKS]
        ids, _ = resolve_ranked_live_candidates(
            provisional_search, live_ids,
            pool_n=PATRICK_POOL_N,
            pool_metric=PATRICK_POOL_METRIC,
            pool_min_bets=PATRICK_POOL_MIN_BETS,
        )
        search_periods = provisional_search
        val_periods = periods[-PATRICK_HOLDOUT_WEEKS:]
        for _ in range(2):
            coverage = candidate_period_coverage(periods, ids, PATRICK_MIN_AVAILABLE)
            covered = [
                (int(r.season), int(r.week))
                for r in coverage.itertuples(index=False)
                if int(r.scorable_games) >= PATRICK_HOLDOUT_MIN_SCORABLE_GAMES
            ]
            if len(covered) < PATRICK_HOLDOUT_WEEKS:
                covered = [
                    (int(r.season), int(r.week))
                    for r in coverage.itertuples(index=False)
                    if int(r.scorable_games) > 0
                ]
            if len(covered) < PATRICK_HOLDOUT_WEEKS:
                break
            val_periods = tuple(covered[-PATRICK_HOLDOUT_WEEKS:])
            first_val = val_periods[0]
            search_periods = tuple(p for p in periods if p < first_val)
            ids, _ = resolve_ranked_live_candidates(
                search_periods, live_ids,
                pool_n=PATRICK_POOL_N,
                pool_metric=PATRICK_POOL_METRIC,
                pool_min_bets=PATRICK_POOL_MIN_BETS,
            )
        if len(ids) < PATRICK_MIN_SIZE:
            ui.notification_show(
                f"Only {len(ids)} eligible current-week models remain; at least {PATRICK_MIN_SIZE} are required.",
                type="error",
            )
            return

        hi = min(PATRICK_MAX_SIZE, len(ids))
        total = combination_count(len(ids), PATRICK_MIN_SIZE, hi)
        if total > EXACT_SEARCH_DEFAULT_MAX:
            ui.notification_show(
                f"Recommended search resolves to {total:,} combinations, above its {EXACT_SEARCH_DEFAULT_MAX:,} exact-search safeguard.",
                type="error", duration=12,
            )
            return

        # Synchronize Page 3 controls so the exact one-click recipe is visible
        # if a user later opens Strategy Lab to inspect the run.
        ui.update_select("auto_pool_mode", selected="top")
        ui.update_numeric("auto_pool_n", value=PATRICK_POOL_N)
        ui.update_select("auto_pool_metric", selected=PATRICK_POOL_METRIC)
        ui.update_numeric("auto_pool_min_bets", value=PATRICK_POOL_MIN_BETS)
        ui.update_numeric("auto_holdout_weeks", value=PATRICK_HOLDOUT_WEEKS)
        ui.update_numeric("auto_min_size", value=PATRICK_MIN_SIZE)
        ui.update_numeric("auto_max_size", value=PATRICK_MAX_SIZE)
        ui.update_select("auto_k", selected=f"{PATRICK_K:.2f}")
        ui.update_numeric("auto_min_available", value=PATRICK_MIN_AVAILABLE)
        ui.update_numeric("auto_min_bets", value=PATRICK_MIN_SEARCH_BETS)
        ui.update_select("auto_rank_metric", selected=PATRICK_RANK_METRIC)
        ui.update_numeric("auto_top_n", value=PATRICK_FINALISTS)

        values = {
            "search_seasons": seasons,
            "validation_seasons": seasons,
            "search_periods": search_periods,
            "validation_periods": val_periods,
            "min_size": PATRICK_MIN_SIZE,
            "max_size": hi,
            "primary_k": PATRICK_K,
            "min_available_models": PATRICK_MIN_AVAILABLE,
            "min_search_bets": PATRICK_MIN_SEARCH_BETS,
            "min_seasons_represented": 1,
            "min_distinct_weeks": 1,
            "ranking_metric": PATRICK_RANK_METRIC,
            "standard_price": -110,
            "chunk_size": 512,
            "top_n": PATRICK_FINALISTS,
            "max_combinations": EXACT_SEARCH_DEFAULT_MAX,
        }
        now = time.monotonic()
        set_auto_progress(
            done=0, total=total,
            label=f"Patrick recipe: {len(ids)} live candidates; preparing discovery matrix…",
            phase="Preparing Patrick's recommended search", started=now, updated=now,
        )
        holdout_label = f"{val_periods[0][0]} W{val_periods[0][1]}–{val_periods[-1][0]} W{val_periods[-1][1]}" if val_periods else "none"
        patrick_state.set({
            "phase": "searching",
            "message": f"Searching {total:,} exact combinations from {len(ids)} current-week candidates · holdout {holdout_label}…",
        })
        print(
            f"[Patrick recipe] starting exact search: {len(ids)} candidates, sizes "
            f"{PATRICK_MIN_SIZE}–{hi}, {total:,} combinations, k={PATRICK_K:.2f}",
            flush=True,
        )
        auto_task(ids, values, val_periods)

    @render.ui
    def patrick_progress_bar():
        state = patrick_state.get()
        if state.get("phase") != "searching":
            return ui.div()
        reactive.invalidate_later(0.4)
        p = get_auto_progress()
        total = int(p.get("total") or 0)
        done = int(p.get("done") or 0)
        pct = 100.0 * done / total if total else 0.0
        return ui.div(
            ui.div(class_="search-progress-fill", style=f"width: {max(0.0, min(100.0, pct)):.1f}%"),
            class_="search-progress-track",
        )

    @render.text
    def patrick_status():
        state = patrick_state.get()
        phase = str(state.get("phase", "idle"))
        if phase == "searching":
            reactive.invalidate_later(0.4)
            p = get_auto_progress()
            done = int(p.get("done") or 0)
            total = int(p.get("total") or 0)
            started = p.get("started")
            elapsed = max(0.0, time.monotonic() - started) if started else 0.0
            pct = 100.0 * done / total if total else 0.0
            rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 and total >= done else np.nan
            eta = f" · est. {remaining:.0f}s remaining" if np.isfinite(remaining) and remaining > 1 else ""
            return f"{done:,}/{total:,} ({pct:.1f}%) · elapsed {elapsed:.1f}s{eta} · {p.get('label', '')}"
        return str(state.get("message", ""))

    def auto_top_merged():
        r = auto_result()
        if r is None:
            return pd.DataFrame()
        d = r.get("top", pd.DataFrame()).copy()
        robust = r.get("robustness", {}).get("summary", pd.DataFrame()).copy()
        if len(d) and len(robust) and "search_rank" in d.columns and "search_rank" in robust.columns:
            keep = [
                c for c in [
                    "search_rank", "thresholds_tested", "min_bets", "max_bets",
                    "mean_ats", "min_ats", "mean_roi", "min_roi", "profitable_thresholds",
                ] if c in robust.columns
            ]
            d = d.merge(robust[keep], on="search_rank", how="left")
        return d

    @render.data_frame
    def auto_top_table():
        d = auto_top_merged()
        if d.empty:
            return render.DataGrid(d)
        d = _pct_frame(d, [
            "ats_pct", "roi", "wilson_low", "worst_season_ats",
            "validation_ats_pct", "validation_roi", "validation_wilson_low",
            "mean_ats", "min_ats", "mean_roi", "min_roi",
        ])
        rename = {
            "search_rank": "Rank", "combo_size": "N", "model_names": "Models",
            "bets": "Discovery bets", "ats_pct": "Discovery ATS %", "roi": "Discovery ROI %",
            "wilson_low": "Discovery Wilson LB %", "validation_bets": "Validation bets",
            "validation_ats_pct": "Validation ATS %", "validation_roi": "Validation ROI %",
            "validation_wilson_low": "Validation Wilson LB %",
            "thresholds_tested": "k values tested", "min_bets": "Min bets across k",
            "max_bets": "Max bets across k", "mean_ats": "Mean ATS across k %",
            "min_ats": "Worst ATS across k %", "mean_roi": "Mean ROI across k %",
            "min_roi": "Worst ROI across k %", "profitable_thresholds": "Profitable k values",
        }
        preferred = [
            "search_rank", "combo_size", "model_names", "bets", "ats_pct", "roi", "wilson_low",
            "validation_bets", "validation_ats_pct", "validation_roi", "validation_wilson_low",
            "profitable_thresholds", "mean_ats", "min_ats", "min_bets", "max_bets", "model_ids",
        ]
        cols = [c for c in preferred if c in d.columns]
        d = d[cols].rename(columns=rename)
        return render.DataGrid(d, filters=True, height="620px")

    def selected_auto_row():
        r = auto_result()
        if r is None:
            return None
        d = r.get("top", pd.DataFrame())
        if d.empty:
            return None
        rank = int(input.auto_pick_rank())
        if "search_rank" in d.columns:
            q = d[pd.to_numeric(d["search_rank"], errors="coerce").eq(rank)]
            if len(q):
                return q.iloc[0]
        idx = max(0, min(rank - 1, len(d) - 1))
        return d.iloc[idx]

    @render.text
    def auto_selected_models():
        row = selected_auto_row()
        if row is None:
            return "Run the search, then enter a finalist rank."
        return (
            f"Rank {int(row.get('search_rank', input.auto_pick_rank()))}\n"
            f"{row.get('model_names', '')}\n\n"
            f"Discovery: {int(row.get('bets', 0))} bets | "
            f"{100*float(row.get('ats_pct', np.nan)):.1f}% ATS | "
            f"{100*float(row.get('roi', np.nan)):+.1f}% ROI"
        )

    @render.data_frame
    def auto_threshold_detail():
        r = auto_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("robustness", {}).get("detail", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        rank = int(input.auto_pick_rank())
        d = d[pd.to_numeric(d["search_rank"], errors="coerce").eq(rank)].copy()
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low"])
        keep = [c for c in ["k", "bets", "wins", "losses", "ats_pct", "roi", "wilson_low"] if c in d.columns]
        d = d[keep].rename(columns={
            "k": "k (SD)", "bets": "Bets", "wins": "Wins", "losses": "Losses",
            "ats_pct": "ATS %", "roi": "ROI %", "wilson_low": "Wilson LB %",
        })
        return render.DataGrid(d, filters=False, height="360px")

    @render.data_frame
    def auto_robust_summary():
        r = auto_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("robustness", {}).get("summary", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d = _pct_frame(d, ["mean_ats", "min_ats", "max_ats", "mean_roi", "min_roi", "max_roi"])
        rename = {
            "search_rank": "Rank", "model_names": "Models", "combo_size": "N",
            "thresholds_tested": "k values", "min_bets": "Min bets", "max_bets": "Max bets",
            "mean_ats": "Mean ATS %", "min_ats": "Worst ATS %", "max_ats": "Best ATS %",
            "mean_roi": "Mean ROI %", "min_roi": "Worst ROI %", "max_roi": "Best ROI %",
            "profitable_thresholds": "Profitable k values",
        }
        preferred = [
            "search_rank", "combo_size", "model_names", "profitable_thresholds",
            "mean_ats", "min_ats", "mean_roi", "min_roi", "min_bets", "max_bets",
        ]
        cols = [c for c in preferred if c in d.columns]
        return render.DataGrid(d[cols].rename(columns=rename), filters=True, height="470px")

    @render.data_frame
    def auto_scale_table():
        r = auto_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("scale_performance", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d = _pct_frame(d, ["bet_rate", "ats_pct", "roi", "wilson_low"])
        period_order = pd.Categorical(d["period"], categories=["Discovery", "Holdout"], ordered=True)
        d = d.assign(_period_order=period_order).sort_values(["search_rank", "_period_order", "bucket_order"])
        keep = [
            "search_rank", "period", "line_bucket", "scorable_games", "bets", "bet_rate",
            "wins", "losses", "ats_pct", "roi", "wilson_low", "mean_abs_edge",
        ]
        d = d[[c for c in keep if c in d.columns]].rename(columns={
            "search_rank": "Rank", "period": "Period", "line_bucket": "|Market spread|",
            "scorable_games": "Scorable games", "bets": "Bets", "bet_rate": "Bet rate %",
            "wins": "Wins", "losses": "Losses", "ats_pct": "ATS %", "roi": "ROI %",
            "wilson_low": "Wilson LB %", "mean_abs_edge": "Mean raw edge (pts)",
        })
        return render.DataGrid(d, filters=True, height="520px")

    @reactive.effect
    @reactive.event(input.use_auto_strategy)
    def select_auto_strategy():
        r = auto_result()
        if r is None:
            ui.notification_show("Run automatic screening first.", type="error")
            return
        top = r.get("top", pd.DataFrame()).copy()
        if top.empty:
            ui.notification_show("The search returned no finalist combinations.", type="error")
            return

        raw_ranks = list(input.auto_portfolio_ranks() or [])
        ranks = []
        for x in raw_ranks:
            try:
                q = int(x)
            except Exception:
                continue
            if q not in ranks:
                ranks.append(q)
        if not ranks:
            ranks = [int(input.auto_pick_rank())]

        combos = []
        for rank in ranks:
            if "search_rank" in top.columns:
                q = top[pd.to_numeric(top["search_rank"], errors="coerce").eq(rank)]
            else:
                q = top.iloc[[rank - 1]] if 1 <= rank <= len(top) else pd.DataFrame()
            if q.empty:
                continue
            row = q.iloc[0]
            ids = [x for x in str(row.get("model_ids", "")).split("|") if x]
            if ids:
                combos.append({"rank": rank, "model_ids": ids})

        if not combos:
            ui.notification_show("None of the selected finalist ranks could be resolved.", type="error")
            return

        live_ids, live_ready, _ = current_week_available_model_ids()
        if not live_ready:
            ui.notification_show("Refresh PredictionTracker on Page 2 before selecting finalists.", type="error")
            return
        missing_now = sorted({
            mid for c in combos for mid in c["model_ids"] if mid not in live_ids
        })
        if missing_now:
            names = ", ".join(MODEL_NAME_MAP.get(x, x) for x in missing_now[:8])
            suffix = "…" if len(missing_now) > 8 else ""
            ui.notification_show(
                f"The current-week model availability changed ({names}{suffix}). Rerun combination screening for the refreshed week.",
                type="error", duration=12,
            )
            return

        k = float(input.auto_k())
        min_n = int(input.auto_min_available())
        # Freeze the model sets first. Then choose each finalist's k using discovery
        # data only and build the diversity-adjusted META backtest. Holdout data
        # are used only after those choices are frozen.
        _, _, discovery_periods, holdout_periods = selected_auto_periods()
        committee_analysis = analyze_finalist_portfolio(
            DATA, combos, discovery_periods, holdout_periods,
            min_available_models=min_n,
            thresholds=K_GRID,
            combo_min_bets=max(20, int(input.auto_min_bets())),
            meta_min_bets=max(30, int(input.auto_min_bets())),
            overlap_threshold=PATRICK_OVERLAP_THRESHOLD,
            min_meta_communities=PATRICK_META_MIN_COMMUNITIES,
            standard_price=-110,
        )
        combos = list(committee_analysis.get("combinations", combos))
        # Union is retained for display/backward compatibility; Page 4 scores each combo separately.
        union_ids = list(dict.fromkeys(mid for c in combos for mid in c["model_ids"]))
        state = {
            "source": "automatic_portfolio",
            "label": f"Finalist portfolio: {len(combos)} combos with discovery-selected k",
            "model_ids": union_ids,
            "combinations": combos,
            "primary_k": k,
            "min_available_models": min_n,
            "committee_analysis": committee_analysis,
            "discovery_periods": tuple(discovery_periods),
            "holdout_periods": tuple(holdout_periods),
        }
        strategy.set(state)
        if not CLOUD_MODE:
            save_current_selection(
                PROJECT_ROOT, union_ids, MODEL_NAME_MAP,
                season=int(input.current_season()), week=int(input.current_week()),
                primary_k=k, min_available_models=min_n, combinations=combos,
            )
        ui.notification_show(
            f"Selected {len(combos)} finalist combinations for Page 4.", type="message"
        )

    # ------------------------------------------------------------------
    # Strategy state shared by Pages 3 and 4
    # ------------------------------------------------------------------
    @render.text
    def strategy_short():
        s = strategy.get()
        combos = s.get("combinations", [])
        if not combos:
            return "None"
        return f"{len(combos)} combo{'s' if len(combos) != 1 else ''} @ {float(s['primary_k']):.2f} SD"

    @render.text
    def strategy_banner():
        s = strategy.get()
        combos = s.get("combinations", [])
        if not combos:
            return "No active strategy. Choose a manual set or one or more automatic finalists on Page 3."
        if len(combos) == 1:
            c = combos[0]
            names = ", ".join(MODEL_NAME_MAP.get(x, x) for x in c.get("model_ids", []))
            return (
                f"{s.get('label', 'Active strategy')} | Minimum available: {int(s.get('min_available_models', 1))}. "
                f"Models: {names}"
            )
        return (
            f"{s.get('label', 'Active finalist portfolio')} | "
            f"Minimum available within each combination: {int(s.get('min_available_models', 1))}."
        )

    @render.text
    def strategy_combo_n():
        return f"{len(strategy.get().get('combinations', [])):,}"

    @render.text
    def strategy_model_n():
        return f"{len(strategy.get().get('model_ids', [])):,}"

    @render.text
    def strategy_k():
        s = strategy.get()
        combos = list(s.get("combinations", []))
        if not combos:
            return "—"
        ks = [float(c.get("k", s.get("primary_k", DEFAULT_K))) for c in combos]
        if len(set(round(x, 6) for x in ks)) == 1:
            return f"{ks[0]:.2f} SD"
        return f"Auto {min(ks):.2f}–{max(ks):.2f} SD"

    # ------------------------------------------------------------------
    # Page 4: apply a finalist portfolio to the cached current board
    # ------------------------------------------------------------------
    @ui.bind_task_button(button_id="apply_strategy_current")
    @reactive.extended_task
    async def strategy_current_task(combinations: list[dict], season: int, week: int, k: float, min_n: int):
        def compute():
            detail_frames = []
            prediction_frames = []
            for i, combo in enumerate(combinations, start=1):
                combo_number = i
                rank = int(combo.get("rank", i))
                combo_k = float(combo.get("k", k))
                community = int(combo.get("community", i))
                ids = [str(x) for x in combo.get("model_ids", []) if str(x)]
                if not ids:
                    continue
                rr = build_current_board_from_cached_sources(
                    PROJECT_ROOT,
                    DATA,
                    ids,
                    MODEL_NAME_MAP,
                    season=int(season),
                    week=int(week),
                    primary_k=float(combo_k),
                    min_available_models=min(int(min_n), len(ids)),
                    include_cfbpicker=False,
                    write_outputs=False,
                )
                b = rr.get("board", pd.DataFrame()).copy()
                if len(b):
                    b["portfolio_combo"] = combo_number
                    b["combo_rank"] = rank
                    b["community"] = community
                    b["combo_size"] = len(ids)
                    b["combo_model_ids"] = "|".join(ids)
                    b["combo_models"] = ", ".join(MODEL_NAME_MAP.get(x, x) for x in ids)
                    b["primary_k"] = float(combo_k)
                    sd = pd.to_numeric(b["model_sd"], errors="coerce")
                    mu = pd.to_numeric(b["consensus_home_margin"], errors="coerce")
                    b["mean_minus_sd"] = mu - sd
                    b["mean_plus_sd"] = mu + sd
                    b["decision_lower"] = mu - float(combo_k) * sd
                    b["decision_upper"] = mu + float(combo_k) * sd
                    detail_frames.append(b)
                pp = rr.get("predictions", pd.DataFrame()).copy()
                if len(pp):
                    pp["portfolio_combo"] = combo_number
                    pp["combo_rank"] = rank
                    prediction_frames.append(pp)

            detail = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
            preds = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()

            # Collapse duplicated model/game rows while preserving which finalist ranks use that model.
            if len(preds):
                keys = [
                    "season", "week", "away", "home", "canonical_model_id", "model_name",
                    "prediction_home_margin", "source", "source_model_name",
                ]
                keys = [c for c in keys if c in preds.columns]
                preds["combo_label"] = [
                    f"C{int(c)}" for c in preds["portfolio_combo"]
                ]
                preds = (
                    preds.groupby(keys, dropna=False)["combo_label"]
                    .agg(lambda x: ", ".join(dict.fromkeys(map(str, x))))
                    .reset_index()
                    .rename(columns={"combo_label": "combo_memberships"})
                )

            summary_rows = []
            if len(detail):
                for _, g in detail.groupby("game_join_key", sort=False):
                    first = g.iloc[0]
                    home = str(first["home"]); away = str(first["away"])
                    edge = pd.to_numeric(g["edge_home"], errors="coerce")
                    qualifies = g["qualifies"].fillna(False).astype(bool)
                    mean_home = int((edge > 0).sum())
                    mean_away = int((edge < 0).sum())
                    bet_home = int((qualifies & edge.gt(0)).sum())
                    bet_away = int((qualifies & edge.lt(0)).sum())
                    scorable = g["availability_state"].astype(str).eq("SCORABLE")
                    passes = int((scorable & ~qualifies).sum())
                    insufficient = int((~scorable).sum())
                    combo_means = pd.to_numeric(
                        g.loc[scorable, "consensus_home_margin"], errors="coerce"
                    ).dropna()
                    portfolio_mean = float(combo_means.mean()) if len(combo_means) else np.nan
                    portfolio_sd = float(combo_means.std(ddof=1)) if len(combo_means) >= 2 else np.nan
                    portfolio_combos_used = int(len(combo_means))
                    if bet_home > bet_away:
                        committee = home
                    elif bet_away > bet_home:
                        committee = away
                    elif bet_home + bet_away:
                        committee = "SPLIT"
                    else:
                        committee = "NO BET"
                    home_ranks = ", ".join(
                        f"C{int(c)}"
                        for c in g.loc[qualifies & edge.gt(0), "portfolio_combo"].tolist()
                    )
                    away_ranks = ", ".join(
                        f"C{int(c)}"
                        for c in g.loc[qualifies & edge.lt(0), "portfolio_combo"].tolist()
                    )
                    summary_rows.append({
                        "away": away, "home": home,
                        "market_home_margin": first.get("market_home_margin", np.nan),
                        "selected_combos": int(len(g)),
                        "mean_favors_home": mean_home, "mean_favors_away": mean_away,
                        "bet_home": bet_home, "bet_away": bet_away,
                        "passes": passes, "insufficient": insufficient,
                        "portfolio_mean_home_margin": portfolio_mean,
                        "portfolio_sd_across_combos": portfolio_sd,
                        "portfolio_combos_used": portfolio_combos_used,
                        "committee_direction": committee,
                        "home_bet_ranks": home_ranks, "away_bet_ranks": away_ranks,
                    })
            summary = pd.DataFrame(summary_rows)
            return {"detail": detail, "summary": summary, "predictions": preds}
        return await asyncio.to_thread(compute)

    @reactive.effect
    @reactive.event(input.apply_strategy_current)
    def start_strategy_current():
        s = strategy.get()
        combos = list(s.get("combinations", []))
        if not combos:
            ui.notification_show("Choose one or more finalist combinations on Page 3 first.", type="error")
            return
        strategy_current_task(
            combos,
            int(input.current_season()),
            int(input.current_week()),
            float(s.get("primary_k", DEFAULT_K)),
            int(s.get("min_available_models", 4)),
        )

    def strategy_current_result():
        if strategy_current_task.status() != "success":
            return None
        return strategy_current_task.result()

    @reactive.effect
    def finish_patrick_recommended():
        state = patrick_state.get()
        phase = str(state.get("phase", "idle"))

        if phase == "searching":
            status = auto_task.status()
            if status == "error":
                patrick_state.set({"phase": "error", "message": "Patrick's recommended combination search failed."})
                return
            if status != "success":
                return
            r = auto_result()
            top = r.get("top", pd.DataFrame()).copy() if r is not None else pd.DataFrame()
            if top.empty:
                patrick_state.set({"phase": "error", "message": "The recommended search returned no eligible finalist combinations."})
                return

            top = top.head(PATRICK_FINALISTS).copy()
            combos = []
            for i, row in enumerate(top.itertuples(index=False), start=1):
                rank = int(getattr(row, "search_rank", i))
                raw_ids = str(getattr(row, "model_ids", ""))
                ids = [x for x in raw_ids.split("|") if x]
                if ids:
                    combos.append({"rank": rank, "model_ids": ids})
            if not combos:
                patrick_state.set({"phase": "error", "message": "The recommended finalist combinations could not be resolved."})
                return

            periods = tuple((y, w) for y, w in HISTORICAL_PERIODS if y in set(HISTORICAL_SEASONS))
            discovery_periods = periods[:-PATRICK_HOLDOUT_WEEKS]
            holdout_periods = periods[-PATRICK_HOLDOUT_WEEKS:]
            committee_analysis = analyze_finalist_portfolio(
                DATA, combos, discovery_periods, holdout_periods,
                min_available_models=PATRICK_MIN_AVAILABLE,
                thresholds=K_GRID,
                combo_min_bets=PATRICK_MIN_SEARCH_BETS,
                meta_min_bets=PATRICK_MIN_SEARCH_BETS,
                overlap_threshold=PATRICK_OVERLAP_THRESHOLD,
                min_meta_communities=PATRICK_META_MIN_COMMUNITIES,
                standard_price=-110,
            )
            combos = list(committee_analysis.get("combinations", combos))[:PATRICK_FINALISTS]
            # Hard-cap the one-click preset to its documented top-12 contract.
            union_ids = list(dict.fromkeys(mid for c in combos for mid in c["model_ids"]))
            strategy.set({
                "source": "patrick_recommended",
                "label": f"Patrick's recommended portfolio: {len(combos)} combos with auto-k + diversified META",
                "model_ids": union_ids,
                "combinations": combos,
                "primary_k": PATRICK_K,
                "min_available_models": PATRICK_MIN_AVAILABLE,
                "committee_analysis": committee_analysis,
                    "discovery_periods": tuple(discovery_periods),
                "holdout_periods": tuple(holdout_periods),
            })
            ui.update_selectize(
                "auto_portfolio_ranks",
                selected=[str(c["rank"]) for c in combos],
            )
            ui.update_numeric("auto_pick_rank", value=1)
            patrick_state.set({
                "phase": "scoring",
                "message": f"Selected top {len(combos)} combinations; scoring the current week…",
            })
            strategy_current_task(
                combos,
                int(input.current_season()),
                int(input.current_week()),
                PATRICK_K,
                PATRICK_MIN_AVAILABLE,
            )
            return

        if phase == "scoring":
            status = strategy_current_task.status()
            if status == "error":
                patrick_state.set({"phase": "error", "message": "The recommended finalists were selected, but current-week scoring failed."})
            elif status == "success":
                result = strategy_current_result()
                games = len(result.get("summary", pd.DataFrame())) if result else 0
                patrick_state.set({
                    "phase": "complete",
                    "message": f"Complete: top {len(strategy.get().get('combinations', []))} combinations auto-tuned across k = 0.25–2.00, overlap-adjusted, and applied to {games} current games.",
                })

    def _active_committee_analysis() -> dict:
        a = strategy.get().get("committee_analysis", {})
        return a if isinstance(a, dict) else {}

    def _selected_meta_k(method: str = "Diversified META") -> float:
        a = _active_committee_analysis()
        d = a.get("meta_selected", pd.DataFrame()) if a else pd.DataFrame()
        if isinstance(d, pd.DataFrame) and len(d) and "method" in d.columns:
            q = d[d["method"].astype(str).eq(method)]
            if len(q):
                v = pd.to_numeric(pd.Series([q.iloc[0].get("selected_k")]), errors="coerce").iloc[0]
                if np.isfinite(v):
                    return float(v)
        return float(strategy.get().get("primary_k", DEFAULT_K))

    def _diversified_meta_current(g: pd.DataFrame) -> dict:
        if g is None or g.empty:
            return {}
        scorable = g[g["availability_state"].astype(str).eq("SCORABLE")].copy()
        if scorable.empty:
            return {}
        if "community" not in scorable.columns:
            scorable["community"] = pd.to_numeric(scorable.get("portfolio_combo"), errors="coerce")
        units = []
        raw_unit_vars = []
        for cid, z in scorable.groupby("community", dropna=False):
            zz = z.copy()
            means_s = pd.to_numeric(zz["consensus_home_margin"], errors="coerce")
            sds_s = pd.to_numeric(zz["model_sd"], errors="coerce")
            counts_s = pd.to_numeric(zz.get("available_models", 1), errors="coerce").fillna(1).clip(lower=1)
            ok = means_s.notna() & sds_s.notna()
            means = means_s[ok].to_numpy(dtype=float)
            sds = sds_s[ok].to_numpy(dtype=float)
            counts = counts_s[ok].to_numpy(dtype=float)
            if not len(means):
                continue
            cmean = float(np.mean(means))
            mean_uncertainty = float(np.mean(np.square(sds) / counts)) if len(sds) else np.nan
            between_combo = float(np.var(means, ddof=1)) if len(means) >= 2 else 0.0
            cvar_mean = mean_uncertainty + between_combo if np.isfinite(mean_uncertainty) else np.nan
            raw_within = float(np.mean(np.square(sds))) if len(sds) else np.nan
            raw_cvar = raw_within + between_combo if np.isfinite(raw_within) else np.nan
            if np.isfinite(cvar_mean):
                units.append((cid, cmean, cvar_mean))
                raw_unit_vars.append(raw_cvar)
        if not units:
            return {}
        means = np.array([x[1] for x in units], dtype=float)
        vars_ = np.array([x[2] for x in units], dtype=float)
        meta_mean = float(np.mean(means))
        within = float(np.mean(vars_))
        between = float(np.var(means, ddof=1)) if len(means) >= 2 else 0.0
        meta_sd = float(np.sqrt(max(0.0, within + between)))
        raw_within = float(np.nanmean(np.array(raw_unit_vars, dtype=float))) if raw_unit_vars else np.nan
        raw_total_sd = float(np.sqrt(max(0.0, raw_within + between))) if np.isfinite(raw_within) else np.nan
        market = pd.to_numeric(pd.Series([g.iloc[0].get("market_home_margin")]), errors="coerce").iloc[0]
        edge = meta_mean - float(market) if np.isfinite(market) else np.nan
        if np.isfinite(edge) and np.isfinite(meta_sd):
            signal = abs(edge) / meta_sd if meta_sd > 1e-12 else (np.inf if abs(edge) > 1e-12 else 0.0)
        else:
            signal = np.nan
        meta_k = _selected_meta_k("Diversified META")
        min_comm = int(_active_committee_analysis().get("min_meta_communities", PATRICK_META_MIN_COMMUNITIES))
        qualifies = bool(len(units) >= min_comm and (np.isfinite(signal) or np.isinf(signal)) and signal >= meta_k)
        home = str(g.iloc[0].get("home", "")); away = str(g.iloc[0].get("away", ""))
        side = home if np.isfinite(edge) and edge > 0 else (away if np.isfinite(edge) and edge < 0 else "")
        return {
            "independent_communities": int(len(units)),
            "meta_mean_home_margin": meta_mean,
            "meta_total_sd": meta_sd,
            "meta_consensus_sd": meta_sd,
            "meta_raw_total_sd": raw_total_sd,
            "meta_between_community_sd": float(np.sqrt(max(0.0, between))),
            "meta_edge_home": edge,
            "meta_signal_sd": signal,
            "meta_k": meta_k,
            "meta_qualifies": qualifies,
            "meta_bet_side": side if qualifies else "",
        }

    def _summarize_portfolio_detail(detail: pd.DataFrame) -> pd.DataFrame:
        summary_rows = []
        if detail is None or detail.empty:
            return pd.DataFrame()
        for game_key, g in detail.groupby("game_join_key", sort=False):
            first = g.iloc[0]
            home = str(first["home"]); away = str(first["away"])
            edge = pd.to_numeric(g["edge_home"], errors="coerce")
            qualifies = g["qualifies"].fillna(False).astype(bool)
            mean_home = int((edge > 0).sum())
            mean_away = int((edge < 0).sum())
            bet_home = int((qualifies & edge.gt(0)).sum())
            bet_away = int((qualifies & edge.lt(0)).sum())
            scorable = g["availability_state"].astype(str).eq("SCORABLE")
            passes = int((scorable & ~qualifies).sum())
            insufficient = int((~scorable).sum())
            combo_means = pd.to_numeric(
                g.loc[scorable, "consensus_home_margin"], errors="coerce"
            ).dropna()
            portfolio_mean = float(combo_means.mean()) if len(combo_means) else np.nan
            portfolio_sd = float(combo_means.std(ddof=1)) if len(combo_means) >= 2 else np.nan
            portfolio_combos_used = int(len(combo_means))
            meta_now = _diversified_meta_current(g)
            if bet_home > bet_away:
                committee = home
            elif bet_away > bet_home:
                committee = away
            elif bet_home + bet_away:
                committee = "SPLIT"
            else:
                committee = "NO BET"
            home_ranks = ", ".join(
                f"C{int(c)}"
                for c in g.loc[qualifies & edge.gt(0), "portfolio_combo"].tolist()
            )
            away_ranks = ", ".join(
                f"C{int(c)}"
                for c in g.loc[qualifies & edge.lt(0), "portfolio_combo"].tolist()
            )
            summary_rows.append({
                "game_join_key": game_key,
                "away": away, "home": home,
                "pt_market_home_margin": first.get("pt_market_home_margin", first.get("market_home_margin", np.nan)),
                "market_home_margin": first.get("market_home_margin", np.nan),
                "line_overridden": bool(first.get("line_overridden", False)),
                "selected_combos": int(len(g)),
                "mean_favors_home": mean_home, "mean_favors_away": mean_away,
                "bet_home": bet_home, "bet_away": bet_away,
                "passes": passes, "insufficient": insufficient,
                "portfolio_mean_home_margin": portfolio_mean,
                "portfolio_sd_across_combos": portfolio_sd,
                "portfolio_combos_used": portfolio_combos_used,
                **meta_now,
                "committee_direction": committee,
                "home_bet_ranks": home_ranks, "away_bet_ranks": away_ranks,
            })
        return pd.DataFrame(summary_rows)

    def strategy_current_view_result():
        """Current portfolio re-scored against any session-only alternate lines."""
        raw = strategy_current_result()
        if raw is None:
            return None
        detail = raw.get("detail", pd.DataFrame()).copy()
        preds = raw.get("predictions", pd.DataFrame()).copy()
        if detail.empty:
            return {"detail": detail, "summary": pd.DataFrame(), "predictions": preds}

        detail["pt_market_home_margin"] = pd.to_numeric(
            detail["market_home_margin"], errors="coerce"
        )
        detail["line_overridden"] = False
        overrides = dict(line_overrides.get() or {})
        for game_key, market_home_margin in overrides.items():
            mask = detail["game_join_key"].astype(str).eq(str(game_key))
            if mask.any() and np.isfinite(float(market_home_margin)):
                detail.loc[mask, "market_home_margin"] = float(market_home_margin)
                detail.loc[mask, "line_overridden"] = True

        mu = pd.to_numeric(detail["consensus_home_margin"], errors="coerce").to_numpy(dtype=float)
        market = pd.to_numeric(detail["market_home_margin"], errors="coerce").to_numpy(dtype=float)
        sd = pd.to_numeric(detail["model_sd"], errors="coerce").to_numpy(dtype=float)
        edge = mu - market
        signal = np.full(len(detail), np.nan, dtype=float)
        finite = np.isfinite(edge) & np.isfinite(sd)
        positive_sd = finite & (sd > 1e-12)
        signal[positive_sd] = np.abs(edge[positive_sd]) / sd[positive_sd]
        zero_sd = finite & ~positive_sd
        signal[zero_sd & (np.abs(edge) > 1e-12)] = np.inf
        signal[zero_sd & (np.abs(edge) <= 1e-12)] = 0.0

        detail["edge_home"] = edge
        detail["signal_sd"] = signal
        scorable = detail["availability_state"].astype(str).eq("SCORABLE").to_numpy()
        fallback_k = float(strategy.get().get("primary_k", DEFAULT_K))
        if "primary_k" in detail.columns:
            k_vec = pd.to_numeric(detail["primary_k"], errors="coerce").fillna(fallback_k).to_numpy(dtype=float)
        else:
            k_vec = np.full(len(detail), fallback_k, dtype=float)
        qualifies = scorable & (np.isfinite(signal) | np.isinf(signal)) & (signal >= k_vec)
        detail["qualifies"] = qualifies
        home = detail["home"].astype(str).to_numpy()
        away = detail["away"].astype(str).to_numpy()
        side = np.where(edge > 0, home, np.where(edge < 0, away, ""))
        detail["bet_side"] = np.where(qualifies, side, "")

        summary = _summarize_portfolio_detail(detail)
        return {"detail": detail, "summary": summary, "predictions": preds}

    @reactive.effect
    def sync_line_override_games():
        raw = strategy_current_result()
        if raw is None:
            ui.update_select("line_override_game", choices={"": "Apply the portfolio first"}, selected="")
            return
        d = raw.get("detail", pd.DataFrame())
        if d.empty:
            ui.update_select("line_override_game", choices={"": "No scorable games"}, selected="")
            return
        games = d[["game_join_key", "away", "home"]].drop_duplicates("game_join_key")
        choices = {
            str(r.game_join_key): f"{r.away} @ {r.home}"
            for r in games.itertuples(index=False)
        }
        selected = next(iter(choices), "")
        ui.update_select("line_override_game", choices=choices, selected=selected)

    @reactive.effect
    def sync_line_override_value():
        key = str(input.line_override_game() or "")
        raw = strategy_current_result()
        if not key or raw is None:
            return
        d = raw.get("detail", pd.DataFrame())
        g = d[d["game_join_key"].astype(str).eq(key)] if len(d) else pd.DataFrame()
        if g.empty:
            return
        first = g.iloc[0]
        original_margin = pd.to_numeric(pd.Series([first.get("market_home_margin")]), errors="coerce").iloc[0]
        if not np.isfinite(original_margin):
            return
        used_margin = float((line_overrides.get() or {}).get(key, float(original_margin)))
        home = str(first.get("home", "Home"))
        ui.update_numeric(
            "line_override_value",
            label=f"Available line for {home} (home team)",
            value=round(-used_margin, 2),
        )

    @render.text
    def line_override_prompt():
        key = str(input.line_override_game() or "")
        raw = strategy_current_result()
        if not key or raw is None:
            return "Apply the portfolio first, then select a game."
        d = raw.get("detail", pd.DataFrame())
        g = d[d["game_join_key"].astype(str).eq(key)] if len(d) else pd.DataFrame()
        if g.empty:
            return "Choose a game."
        first = g.iloc[0]
        away = str(first["away"]); home = str(first["home"])
        original = pd.to_numeric(pd.Series([first.get("market_home_margin")]), errors="coerce").iloc[0]
        if not np.isfinite(original):
            return f"Enter the available point spread for {home}."
        used = float((line_overrides.get() or {}).get(key, float(original)))
        return (
            f"PredictionTracker: {_spread_label(away, home, original)} · "
            f"currently scored at: {_spread_label(away, home, used)}"
        )

    @reactive.effect
    @reactive.event(input.apply_line_override)
    def apply_line_override_value():
        key = str(input.line_override_game() or "")
        if not key:
            ui.notification_show("Apply the portfolio and choose a game first.", type="error")
            return
        try:
            home_team_spread = float(input.line_override_value())
        except Exception:
            ui.notification_show("Enter a valid sportsbook spread.", type="error")
            return
        if not np.isfinite(home_team_spread):
            ui.notification_show("Enter a finite sportsbook spread.", type="error")
            return
        overrides = dict(line_overrides.get() or {})
        # Sportsbook home-team spread is the negative of expected home margin:
        # Virginia -3 -> internal home margin +3; home +3 -> internal margin -3.
        overrides[key] = -home_team_spread
        line_overrides.set(overrides)
        ui.notification_show("Alternate line applied to this game.", type="message", duration=4)

    @reactive.effect
    @reactive.event(input.reset_line_override)
    def reset_line_override_value():
        key = str(input.line_override_game() or "")
        overrides = dict(line_overrides.get() or {})
        if key in overrides:
            overrides.pop(key, None)
            line_overrides.set(overrides)
            ui.notification_show("This game was reset to the PredictionTracker line.", type="message", duration=4)

    @reactive.effect
    @reactive.event(input.clear_line_overrides)
    def clear_line_override_values():
        line_overrides.set({})
        ui.notification_show("All games were reset to PredictionTracker lines.", type="message", duration=4)

    @render.text
    def line_override_status():
        n = len(line_overrides.get() or {})
        if n == 0:
            return "No alternate lines active. All games use PredictionTracker's market line."
        return f"{n} alternate line{'s' if n != 1 else ''} active in this session. Tables below are re-scored automatically."

    @render.text
    def line_override_game_result():
        key = str(input.line_override_game() or "")
        r = strategy_current_view_result()
        if not key or r is None:
            return ""
        d = r.get("summary", pd.DataFrame())
        g = d[d["game_join_key"].astype(str).eq(key)] if len(d) else pd.DataFrame()
        if g.empty:
            return ""
        row = g.iloc[0]
        away = str(row["away"]); home = str(row["home"])
        line = _spread_label(away, home, row["market_home_margin"])
        return (
            f"At {line}: {int(row['bet_home'])} combinations BET {home}, "
            f"{int(row['bet_away'])} BET {away}, {int(row['passes'])} PASS"
            + (f", {int(row['insufficient'])} insufficient." if int(row['insufficient']) else ".")
        )

    @render.text
    def strategy_current_status():
        s = strategy_current_task.status()
        if s == "initial":
            return "Refresh PredictionTracker on Page 2, choose finalists on Page 3, then apply the portfolio here."
        if s == "running":
            return "Scoring each selected finalist combination independently…"
        if s == "success":
            r = strategy_current_result()
            n = len(r.get("detail", pd.DataFrame())) if r else 0
            return f"Portfolio scored successfully ({n:,} game × combination rows)."
        if s == "error":
            return "Could not apply the finalist portfolio. Refresh Page 2 first if the current file is missing."
        return str(s)

    @reactive.effect
    def show_strategy_current_error():
        if strategy_current_task.status() == "error":
            try:
                strategy_current_task.result()
            except Exception as exc:
                ui.notification_show(f"Upcoming-predictions error: {exc}", type="error", duration=12)

    @render.text
    def strategy_games_n():
        r = strategy_current_view_result()
        if r is None:
            return "—"
        d = r.get("summary", pd.DataFrame())
        return f"{len(d):,}"

    @render.text
    def committee_community_n():
        a = _active_committee_analysis()
        return str(int(a.get("overlap_summary", {}).get("communities", 0))) if a else "—"

    @render.text
    def committee_mean_overlap():
        a = _active_committee_analysis()
        v = a.get("overlap_summary", {}).get("mean_pairwise_jaccard", np.nan) if a else np.nan
        return f"{100*float(v):.1f}%" if np.isfinite(v) else "—"

    @render.text
    def committee_meta_k():
        a = _active_committee_analysis()
        return f"{_selected_meta_k('Diversified META'):.2f} SD" if a else "—"

    @render.text
    def committee_holdout_ats():
        a = _active_committee_analysis()
        d = a.get("meta_summary", pd.DataFrame()) if a else pd.DataFrame()
        if not isinstance(d, pd.DataFrame) or d.empty:
            return "—"
        q = d[d["method"].astype(str).eq("Diversified META") & d["period"].astype(str).eq("Holdout")]
        if q.empty:
            return "—"
        ats = pd.to_numeric(pd.Series([q.iloc[0].get("ats_pct")]), errors="coerce").iloc[0]
        bets = int(q.iloc[0].get("bets", 0) or 0)
        return f"{100*float(ats):.1f}% ({bets} bets)" if np.isfinite(ats) else f"— ({bets} bets)"

    @render.text
    def committee_holdout_window():
        periods = tuple(strategy.get().get("holdout_periods", ()) or ())
        if not periods:
            return "No holdout selected."
        wanted = set((int(y), int(w)) for y, w in periods)
        d = _HISTORICAL_GRADED_DATA.copy()
        yy = pd.to_numeric(d["season"], errors="coerce")
        ww = pd.to_numeric(d["week"], errors="coerce")
        mask = pd.Series([
            (int(y), int(w)) in wanted if pd.notna(y) and pd.notna(w) else False
            for y, w in zip(yy, ww)
        ], index=d.index)
        q = d.loc[mask]
        games = int(q["game_key"].astype(str).nunique()) if "game_key" in q.columns else 0
        first, last = periods[0], periods[-1]
        return (
            f"{int(first[0])} W{int(first[1])}–{int(last[0])} W{int(last[1])} "
            f"· {len(periods)} completed weeks · {games:,} graded games"
        )

    @render.data_frame
    def committee_meta_backtest_table():
        a = _active_committee_analysis()
        d = a.get("meta_summary", pd.DataFrame()).copy() if a else pd.DataFrame()
        if not isinstance(d, pd.DataFrame) or d.empty:
            return render.DataGrid(pd.DataFrame())
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low"])
        keep = [c for c in ["method", "period", "selected_k", "scorable_games", "bets", "wins", "losses", "ats_pct", "roi", "wilson_low", "units"] if c in d.columns]
        d = d[keep].rename(columns={
            "method": "META method", "period": "Period", "selected_k": "Frozen k",
            "scorable_games": "Scorable games", "bets": "Bets", "wins": "Wins", "losses": "Losses",
            "ats_pct": "ATS %", "roi": "ROI %", "wilson_low": "Wilson LB %", "units": "Units",
        })
        return render.DataGrid(d, filters=False, height="260px")

    @render.data_frame
    def committee_meta_spread_table():
        a = _active_committee_analysis()
        d = a.get("meta_spread_scale", pd.DataFrame()).copy() if a else pd.DataFrame()
        if not isinstance(d, pd.DataFrame) or d.empty:
            return render.DataGrid(pd.DataFrame())
        d = _pct_frame(d, ["ats_pct", "roi", "wilson_low", "favorite_ats_pct", "underdog_ats_pct"])
        for c in ["forecast_mae", "market_mae", "delta_mae_vs_market", "forecast_bias", "mean_abs_edge"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        keep = [c for c in [
            "period", "line_bucket", "scorable_games", "forecast_mae", "market_mae",
            "delta_mae_vs_market", "forecast_bias", "bets", "ats_pct", "roi",
            "wilson_low", "favorite_bets", "favorite_ats_pct", "underdog_bets",
            "underdog_ats_pct", "mean_abs_edge",
        ] if c in d.columns]
        d = d[keep].rename(columns={
            "period": "Period", "line_bucket": "|Market spread|",
            "scorable_games": "Scorable games", "forecast_mae": "META MAE",
            "market_mae": "Market MAE", "delta_mae_vs_market": "ΔMAE vs market",
            "forecast_bias": "META bias", "bets": "META bets", "ats_pct": "ATS %",
            "roi": "ROI %", "wilson_low": "Wilson LB %",
            "favorite_bets": "Fav bets", "favorite_ats_pct": "Fav ATS %",
            "underdog_bets": "Dog bets", "underdog_ats_pct": "Dog ATS %",
            "mean_abs_edge": "Mean bet edge (pts)",
        })
        return render.DataGrid(d, filters=False, height="390px")

    @render.data_frame
    def committee_current_spread_context_table():
        r = strategy_current_view_result()
        a = _active_committee_analysis()
        hist = a.get("meta_spread_scale", pd.DataFrame()).copy() if a else pd.DataFrame()
        if r is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            return render.DataGrid(pd.DataFrame())
        cur = r.get("summary", pd.DataFrame()).copy()
        if cur.empty:
            return render.DataGrid(pd.DataFrame())
        rows = []
        for x in cur.itertuples(index=False):
            market = pd.to_numeric(pd.Series([getattr(x, "market_home_margin", np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(market):
                continue
            bucket = meta_spread_bucket_label(float(market))
            qd = hist[(hist["period"].astype(str).eq("Discovery")) & (hist["line_bucket"].astype(str).eq(bucket))]
            qh = hist[(hist["period"].astype(str).eq("Holdout")) & (hist["line_bucket"].astype(str).eq(bucket))]
            dr = qd.iloc[0] if len(qd) else pd.Series(dtype=object)
            hr = qh.iloc[0] if len(qh) else pd.Series(dtype=object)
            edge = pd.to_numeric(pd.Series([getattr(x, "meta_edge_home", np.nan)]), errors="coerce").iloc[0]
            qualifies = bool(getattr(x, "meta_qualifies", False))
            if qualifies and np.isfinite(edge) and abs(float(market)) > 1e-12:
                side_type = "Favorite" if float(edge) * float(market) > 0 else "Underdog"
            elif qualifies:
                side_type = "Pick'em"
            else:
                side_type = "—"
            if side_type == "Favorite":
                disc_side_bets = dr.get("favorite_bets", np.nan); disc_side_ats = dr.get("favorite_ats_pct", np.nan)
                hold_side_bets = hr.get("favorite_bets", np.nan); hold_side_ats = hr.get("favorite_ats_pct", np.nan)
            elif side_type == "Underdog":
                disc_side_bets = dr.get("underdog_bets", np.nan); disc_side_ats = dr.get("underdog_ats_pct", np.nan)
                hold_side_bets = hr.get("underdog_bets", np.nan); hold_side_ats = hr.get("underdog_ats_pct", np.nan)
            else:
                disc_side_bets = np.nan; disc_side_ats = np.nan; hold_side_bets = np.nan; hold_side_ats = np.nan
            home = str(getattr(x, "home", "")); away = str(getattr(x, "away", ""))
            rows.append({
                "Game": f"{away} @ {home}",
                "Line used": _spread_label(away, home, float(market)),
                "|Spread|": round(abs(float(market)), 1),
                "Regime": bucket,
                "META": (f"BET {getattr(x, 'meta_bet_side', '')}" if qualifies else "PASS"),
                "Bet type": side_type,
                "Discovery games": int(dr.get("scorable_games", 0) or 0),
                "Discovery bets": int(dr.get("bets", 0) or 0),
                "Discovery ATS %": dr.get("ats_pct", np.nan),
                "Discovery ΔMAE": dr.get("delta_mae_vs_market", np.nan),
                "Same-side disc bets": disc_side_bets,
                "Same-side disc ATS %": disc_side_ats,
                "Holdout games": int(hr.get("scorable_games", 0) or 0),
                "Holdout bets": int(hr.get("bets", 0) or 0),
                "Holdout ATS %": hr.get("ats_pct", np.nan),
                "Holdout ΔMAE": hr.get("delta_mae_vs_market", np.nan),
                "Same-side hold bets": hold_side_bets,
                "Same-side hold ATS %": hold_side_ats,
            })
        d = pd.DataFrame(rows)
        if d.empty:
            return render.DataGrid(d)
        for c in ["Discovery ATS %", "Same-side disc ATS %", "Holdout ATS %", "Same-side hold ATS %"]:
            d[c] = 100 * pd.to_numeric(d[c], errors="coerce")
        for c in ["Discovery ΔMAE", "Holdout ΔMAE"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        for c in ["Same-side disc bets", "Same-side hold bets"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
        d = d.sort_values(["|Spread|", "Game"], ascending=[False, True]).reset_index(drop=True)
        return render.DataGrid(d, filters=False, height="340px")

    @render.data_frame
    def committee_combo_k_table():
        a = _active_committee_analysis()
        d = a.get("combo_k_selected", pd.DataFrame()).copy() if a else pd.DataFrame()
        if not isinstance(d, pd.DataFrame) or d.empty:
            return render.DataGrid(pd.DataFrame())
        d["Combo"] = "C" + pd.to_numeric(d["portfolio_combo"], errors="coerce").astype("Int64").astype(str)
        d = _pct_frame(d, ["discovery_ats_pct", "discovery_roi", "discovery_wilson_low", "neighbor_wilson_floor", "holdout_ats_pct", "holdout_roi", "holdout_wilson_low"])
        keep = [c for c in ["Combo", "search_rank", "community", "selected_k", "discovery_bets", "discovery_ats_pct", "discovery_roi", "discovery_wilson_low", "neighbor_wilson_floor", "holdout_bets", "holdout_ats_pct", "holdout_roi", "holdout_wilson_low"] if c in d.columns]
        d = d[keep].rename(columns={
            "search_rank": "Search rank", "community": "Community", "selected_k": "Auto k",
            "discovery_bets": "Discovery bets", "discovery_ats_pct": "Discovery ATS %",
            "discovery_roi": "Discovery ROI %", "discovery_wilson_low": "Discovery Wilson LB %",
            "neighbor_wilson_floor": "Neighbor-k Wilson floor %",
            "holdout_bets": "Holdout bets", "holdout_ats_pct": "Holdout ATS %",
            "holdout_roi": "Holdout ROI %", "holdout_wilson_low": "Holdout Wilson LB %",
        })
        return render.DataGrid(d, filters=False, height="360px")

    @render.data_frame
    def committee_overlap_table():
        a = _active_committee_analysis()
        d = a.get("community_table", pd.DataFrame()).copy() if a else pd.DataFrame()
        if not isinstance(d, pd.DataFrame) or d.empty:
            return render.DataGrid(pd.DataFrame())
        d["Models"] = d["model_ids"].astype(str).apply(lambda x: ", ".join(MODEL_NAME_MAP.get(mid, mid) for mid in x.split("|") if mid))
        d["Overlap to representative %"] = 100 * pd.to_numeric(d["jaccard_to_representative"], errors="coerce")
        keep = ["combo", "search_rank", "community", "combo_size", "Overlap to representative %", "Models"]
        return render.DataGrid(d[keep].rename(columns={
            "combo": "Combo", "search_rank": "Search rank", "community": "Community", "combo_size": "N",
        }), filters=True, height="360px")

    @render.data_frame
    def strategy_combo_summary_table():
        r = strategy_current_view_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("summary", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d["Game"] = d["away"].astype(str) + " @ " + d["home"].astype(str)
        d["PT market"] = [
            _spread_label(a, h, m)
            for a, h, m in zip(d["away"], d["home"], d["pt_market_home_margin"])
        ]
        d["Line used"] = [
            _spread_label(a, h, m)
            for a, h, m in zip(d["away"], d["home"], d["market_home_margin"])
        ]
        d["Mean direction"] = [
            f"{h}: {int(nh)} | {a}: {int(na)}"
            for a, h, nh, na in zip(d["away"], d["home"], d["mean_favors_home"], d["mean_favors_away"])
        ]
        d["Bet direction"] = [
            f"{h}: {int(bh)} | {a}: {int(ba)}"
            for a, h, bh, ba in zip(d["away"], d["home"], d["bet_home"], d["bet_away"])
        ]
        d["Portfolio mean ± SD"] = [
            (
                f"{_spread_label(a, h, m)} ± {float(sd):.2f}"
                if np.isfinite(pd.to_numeric(pd.Series([m]), errors="coerce").iloc[0])
                and np.isfinite(pd.to_numeric(pd.Series([sd]), errors="coerce").iloc[0])
                else (f"{_spread_label(a, h, m)} ± —" if np.isfinite(pd.to_numeric(pd.Series([m]), errors="coerce").iloc[0]) else "—")
            )
            for a, h, m, sd in zip(
                d["away"], d["home"], d["portfolio_mean_home_margin"], d["portfolio_sd_across_combos"]
            )
        ]
        if "meta_mean_home_margin" in d.columns:
            d["Diversified META ± SD"] = [
                (
                    f"{_spread_label(a, h, m)} ± {float(sd):.2f}"
                    if np.isfinite(pd.to_numeric(pd.Series([m]), errors="coerce").iloc[0])
                    and np.isfinite(pd.to_numeric(pd.Series([sd]), errors="coerce").iloc[0])
                    else "—"
                )
                for a, h, m, sd in zip(d["away"], d["home"], d["meta_mean_home_margin"], d["meta_total_sd"])
            ]
            d["META decision"] = [
                (f"BET {side}" if bool(q) and str(side) else "PASS")
                for q, side in zip(d["meta_qualifies"].fillna(False), d["meta_bet_side"].fillna(""))
            ]
        else:
            d["Diversified META ± SD"] = "—"
            d["META decision"] = "—"
        keep = [
            "Game", "PT market", "Line used", "Portfolio mean ± SD", "Diversified META ± SD",
            "independent_communities", "meta_signal_sd", "meta_k", "META decision", "portfolio_combos_used",
            "selected_combos", "Mean direction", "Bet direction",
            "passes", "insufficient", "committee_direction", "home_bet_ranks", "away_bet_ranks",
        ]
        keep = [c for c in keep if c in d.columns]
        return render.DataGrid(d[keep].rename(columns={
            "portfolio_combos_used": "Portfolio combos used", "selected_combos": "Combos", "passes": "Passes", "insufficient": "Insufficient",
            "independent_communities": "Independent communities", "meta_signal_sd": "META signal (SD)", "meta_k": "META k",
            "committee_direction": "Combo consensus", "home_bet_ranks": "Home bet combos",
            "away_bet_ranks": "Away bet combos",
        }), filters=True, height="430px")

    @render.data_frame
    def strategy_combo_detail_table():
        r = strategy_current_view_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("detail", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d["Game"] = d["away"].astype(str) + " @ " + d["home"].astype(str)
        d["PT market"] = [
            _spread_label(a, h, m) for a, h, m in zip(d["away"], d["home"], d["pt_market_home_margin"])
        ]
        d["Line used"] = [
            _spread_label(a, h, m) for a, h, m in zip(d["away"], d["home"], d["market_home_margin"])
        ]
        d["Expected spread"] = [
            _spread_label(a, h, m) for a, h, m in zip(d["away"], d["home"], d["consensus_home_margin"])
        ]
        d["Mean ± SD"] = [
            (f"{_spread_label(a, h, m)} ± {float(sd):.2f}" if np.isfinite(pd.to_numeric(pd.Series([sd]), errors='coerce').iloc[0]) else f"{_spread_label(a, h, m)} ± —")
            for a, h, m, sd in zip(d["away"], d["home"], d["consensus_home_margin"], d["model_sd"])
        ]
        d["k×SD decision band"] = [
            (f"{_spread_label(a, h, lo)} to {_spread_label(a, h, hi)}"
             if np.isfinite(pd.to_numeric(pd.Series([lo]), errors='coerce').iloc[0]) and np.isfinite(pd.to_numeric(pd.Series([hi]), errors='coerce').iloc[0])
             else "—")
            for a, h, lo, hi in zip(d["away"], d["home"], d["decision_lower"], d["decision_upper"])
        ]
        q = d["qualifies"].fillna(False).astype(bool)
        scorable = d["availability_state"].astype(str).eq("SCORABLE")
        d["Decision"] = np.where(q, "BET " + d["bet_side"].astype(str), np.where(scorable, "PASS", "INSUFFICIENT"))
        keep = [
            "Game", "portfolio_combo", "combo_rank", "community", "primary_k", "combo_models", "PT market", "Line used", "Expected spread", "Mean ± SD",
            "k×SD decision band", "signal_sd", "available_models", "combo_size", "Decision",
        ]
        out = d[keep].rename(columns={
            "portfolio_combo": "Combo", "combo_rank": "Search rank", "community": "Community", "primary_k": "Auto k",
            "combo_models": "Models", "signal_sd": "Signal (SD)",
            "available_models": "Models available", "combo_size": "Combo size",
        })
        return render.DataGrid(out.sort_values(["Game", "Combo"]), filters=True, height="650px")

    @render.data_frame
    def strategy_model_predictions_table():
        r = strategy_current_view_result()
        if r is None:
            return render.DataGrid(pd.DataFrame())
        d = r.get("predictions", pd.DataFrame()).copy()
        if d.empty:
            return render.DataGrid(d)
        d["Game"] = d["away"].astype(str) + " @ " + d["home"].astype(str)
        keep = ["Game", "model_name", "prediction_home_margin", "combo_memberships", "source"]
        d = d[keep].rename(columns={
            "model_name": "Model", "prediction_home_margin": "Projected home margin",
            "combo_memberships": "Used by combos", "source": "Source",
        })
        return render.DataGrid(d, filters=True, height="520px")


    # ------------------------------------------------------------------
    # Page 5: forecast hierarchy plots
    # ------------------------------------------------------------------
    @reactive.effect
    def update_forecast_plot_games():
        r = strategy_current_view_result()
        if r is None:
            ui.update_select("plot_game", choices={"": "Apply a portfolio on Page 4 first"}, selected="")
            return
        d = r.get("summary", pd.DataFrame()).copy()
        if d.empty:
            ui.update_select("plot_game", choices={"": "No scored games"}, selected="")
            return
        choices = {
            str(row.game_join_key): f"{row.away} @ {row.home}"
            for row in d.itertuples(index=False)
        }
        current = str(input.plot_game() or "")
        selected = current if current in choices else next(iter(choices))
        ui.update_select("plot_game", choices=choices, selected=selected)

    def _forecast_plot_inputs():
        key = str(input.plot_game() or "")
        view = strategy_current_view_result()
        up = upcoming_result()
        if not key or view is None or up is None:
            return None
        summary = view.get("summary", pd.DataFrame()).copy()
        detail = view.get("detail", pd.DataFrame()).copy()
        all_preds = up.get("predictions", pd.DataFrame()).copy()
        if summary.empty or detail.empty or all_preds.empty:
            return None
        sr = summary[summary["game_join_key"].astype(str).eq(key)]
        dg = detail[detail["game_join_key"].astype(str).eq(key)].copy()
        if sr.empty or dg.empty:
            return None
        row = sr.iloc[0]
        away = str(row["away"]); home = str(row["home"])
        ig = all_preds[
            all_preds["away"].astype(str).eq(away)
            & all_preds["home"].astype(str).eq(home)
        ].copy()
        if len(ig):
            ig = ig.drop_duplicates("canonical_model_id", keep="first")
        return row, dg, ig

    @render.text
    def forecast_plot_status():
        z = _forecast_plot_inputs()
        if z is None:
            return "Refresh Page 2 and apply a finalist portfolio on Page 4 first."
        row, dg, ig = z
        selected = set(map(str, strategy.get().get("model_ids", [])))
        used_now = int(ig["canonical_model_id"].astype(str).isin(selected).sum()) if len(ig) else 0
        market = _spread_label(str(row["away"]), str(row["home"]), row["market_home_margin"])
        meta_margin = row.get("meta_mean_home_margin", row.get("portfolio_mean_home_margin", np.nan))
        meta = _spread_label(str(row["away"]), str(row["home"]), meta_margin)
        sd = pd.to_numeric(pd.Series([row.get("meta_total_sd", row.get("portfolio_sd_across_combos"))]), errors="coerce").iloc[0]
        sd_text = f" ± {float(sd):.2f}" if np.isfinite(sd) else ""
        return (
            f"{len(ig)} individual current models ({used_now} used by the finalist portfolio) · "
            f"{int(row['portfolio_combos_used'])} scorable combinations · diversified META {meta}{sd_text} · line used {market}."
        )

    @render.plot
    def forecast_plot():
        z = _forecast_plot_inputs()
        if z is None:
            return None
        row, dg, ig = z
        s = strategy.get()
        return build_forecast_plot(
            str(input.plot_style() or "hierarchy"),
            ig,
            dg,
            away=str(row["away"]),
            home=str(row["home"]),
            selected_model_ids=s.get("model_ids", []),
            meta_mean=float(row.get("meta_mean_home_margin", row.get("portfolio_mean_home_margin", np.nan))),
            meta_sd=float(row.get("meta_total_sd", row.get("portfolio_sd_across_combos", np.nan))) if np.isfinite(pd.to_numeric(pd.Series([row.get("meta_total_sd", row.get("portfolio_sd_across_combos"))]), errors="coerce").iloc[0]) else np.nan,
            market_home_margin=float(row["market_home_margin"]),
            pt_market_home_margin=float(row["pt_market_home_margin"]),
            k=_selected_meta_k("Diversified META"),
        )


app = App(app_ui, server)
