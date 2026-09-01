#!/usr/bin/env python3
"""Workbook-first Tableau Public extraction utilities for Andrew Percival's CFB Picker.

The CFB Picker dashboard is a Tableau Public workbook. Dashboard CSV exports are
not a reliable historical API: a dashboard export can expose only one sheet and
filters may hide the underlying prediction table. This module therefore:

1. downloads the public workbook package,
2. inventories the packaged TWB/TWBX and embedded data sources,
3. reads CSV/XLSX/Hyper extracts when available,
4. falls back to CSV exports for worksheet names discovered from the TWB,
5. normalizes plausible game/model prediction tables into a long format.

It intentionally writes discovery diagnostics even when parsing yields zero
prediction rows so a source-layout change is inspectable rather than silent.
"""
from __future__ import annotations

import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlencode
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

WORKBOOK_NAME = "CFBPicker"
DEFAULT_VIEW = "Standings"
DEFAULT_VIEW_CANDIDATES = ["Games", "List", "Standings", "Predictions", "Prediction", "Game Predictions", "GamePredictions", "Details", "Game Details"]
PUBLIC_BASE = "https://public.tableau.com"
WORKBOOK_URL = f"{PUBLIC_BASE}/workbooks/{WORKBOOK_NAME}.twb"
VIEW_BASE = f"{PUBLIC_BASE}/views/{WORKBOOK_NAME}"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/csv;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

KNOWN_PICKER_NAMES = [
    "FEI", "SP+", "FPI", "ESPN FPI", "KFord", "KFord Ratings", "Kelley Ford",
    "Massey Ratings", "Massey", "Harville", "David Harville", "TeamRankings",
    "TeamRankings.com", "Sagarin: Predictor", "Sagarin Predictor",
    "Sagarin: Golden", "Sagarin Golden Mean", "Sagarin", "Sagarin Ratings",
    "Sagarin: Recent", "Sagarin Recent", "Slate Index", "Slate Fluker",
]

MODEL_ALIAS_GROUPS = {
    "FEI": ["FEI"],
    "SP+": ["SP+", "SP Plus", "SPPlus", "Bill Connelly SP+"],
    "FPI": ["FPI", "ESPN FPI"],
    "KFord": ["KFord", "KFord Ratings", "Kelley Ford", "Kelley Ford Ratings"],
    "Massey Ratings": ["Massey Ratings", "Massey"],
    "Harville": ["Harville", "David Harville"],
    "TeamRankings": ["TeamRankings", "TeamRankings.com", "Team Rankings"],
    "Sagarin: Predictor": ["Sagarin: Predictor", "Sagarin Predictor", "Sagarin Pred"],
    "Sagarin: Golden": ["Sagarin: Golden", "Sagarin Golden Mean", "Sagarin Golden"],
    "Sagarin": ["Sagarin", "Sagarin Ratings", "Sagarin Rating"],
    "Sagarin: Recent": ["Sagarin: Recent", "Sagarin Recent"],
    "Slate Fluker": ["Slate Fluker", "Slate Index"],
}

TEAM_ALIASES = {
    "uconn": "connecticut",
    "ucf": "central florida",
    "usc": "southern california",
    "ole miss": "mississippi",
    "miami fl": "miami florida",
    "miami florida": "miami florida",
    "miami": "miami florida",
    "nc state": "north carolina state",
    "app state": "appalachian state",
    "appalachian st": "appalachian state",
    "pitt": "pittsburgh",
    "smu": "southern methodist",
    "utsa": "texas san antonio",
    "utep": "texas el paso",
    "ul monroe": "louisiana monroe",
    "ulm": "louisiana monroe",
    "louisiana lafayette": "louisiana",
    "ul lafayette": "louisiana",
    "utsa roadrunners": "texas san antonio",
}

META_HINTS = {
    "season", "year", "week", "wk", "date", "game", "matchup", "away", "road", "visitor",
    "home", "host", "team", "model", "picker", "system", "prediction", "projection", "spread",
    "line", "score", "result", "actual", "ats", "mae", "error", "rank", "record", "wins", "losses",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def norm_text(value) -> str:
    s = str(value if value is not None else "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9+]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).replace("+", "plus"))


def slug_team(value) -> str:
    s = norm_text(value).replace("+", " plus ")
    # Strip common ranking/record decorations without stripping meaningful state abbreviations.
    s = re.sub(r"^#\d+\s+", "", s)
    s = re.sub(r"\s+\(\d+[-–]\d+\)\s*$", "", s)
    s = s.replace(" university", "").replace(" state university", " state")
    s = re.sub(r"\b(st\.?|state)\b", " state ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return TEAM_ALIASES.get(s, s)


def canonical_picker_name(name: str) -> str:
    key = compact(name)
    for canonical, aliases in MODEL_ALIAS_GROUPS.items():
        if key in {compact(x) for x in aliases}:
            return canonical
    return str(name).strip()


def _get(session: requests.Session, url: str, *, timeout: int = 60) -> requests.Response:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def download_workbook(dest: Path, *, timeout: int = 90) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        s.headers.update(BROWSER_HEADERS)
        r = _get(s, WORKBOOK_URL, timeout=timeout)
    dest.write_bytes(r.content)
    return {
        "url": WORKBOOK_URL,
        "status_code": int(r.status_code),
        "content_type": r.headers.get("content-type", ""),
        "content_disposition": r.headers.get("content-disposition", ""),
        "bytes": len(r.content),
        "zip_signature": bool(r.content[:2] == b"PK"),
        "saved_to": str(dest),
    }


def _read_twb_names(twb_path: Path) -> tuple[list[str], list[str]]:
    worksheets: list[str] = []
    dashboards: list[str] = []
    try:
        root = ET.parse(twb_path).getroot()
        for e in root.iter():
            tag = e.tag.split("}")[-1]
            if tag == "worksheet" and e.attrib.get("name"):
                worksheets.append(e.attrib["name"])
            elif tag == "dashboard" and e.attrib.get("name"):
                dashboards.append(e.attrib["name"])
    except Exception:
        pass
    return list(dict.fromkeys(worksheets)), list(dict.fromkeys(dashboards))


def unpack_workbook(workbook_path: Path, extract_dir: Path) -> dict:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    raw = workbook_path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extractall(extract_dir)
    else:
        twb = extract_dir / f"{WORKBOOK_NAME}.twb"
        twb.write_bytes(raw)

    files = sorted(p for p in extract_dir.rglob("*") if p.is_file())
    twbs = [p for p in files if p.suffix.lower() == ".twb"]
    worksheets: list[str] = []
    dashboards: list[str] = []
    for p in twbs:
        w, d = _read_twb_names(p)
        worksheets.extend(w)
        dashboards.extend(d)
    return {
        "files": [str(p.relative_to(extract_dir)) for p in files],
        "twb_files": [str(p.relative_to(extract_dir)) for p in twbs],
        "hyper_files": [str(p.relative_to(extract_dir)) for p in files if p.suffix.lower() == ".hyper"],
        "csv_files": [str(p.relative_to(extract_dir)) for p in files if p.suffix.lower() == ".csv"],
        "xlsx_files": [str(p.relative_to(extract_dir)) for p in files if p.suffix.lower() in {".xlsx", ".xls"}],
        "worksheets": list(dict.fromkeys(worksheets)),
        "dashboards": list(dict.fromkeys(dashboards)),
    }


def _stringify_hyper_name(x) -> str:
    try:
        return str(x).replace('"', "")
    except Exception:
        return str(x)


def read_hyper_tables(path: Path, *, max_rows: int | None = None) -> tuple[list[tuple[str, pd.DataFrame]], dict]:
    diagnostics = {"path": str(path), "available": False, "error": None, "tables": []}
    try:
        from tableauhyperapi import HyperProcess, Connection, Telemetry  # type: ignore
    except Exception as e:
        diagnostics["error"] = (
            "tableauhyperapi is not installed. Install the release requirements "
            "(`python -m pip install -r requirements.txt`) and rerun. " + str(e)
        )
        return [], diagnostics

    tables: list[tuple[str, pd.DataFrame]] = []
    try:
        with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hp:
            with Connection(endpoint=hp.endpoint, database=str(path)) as conn:
                schemas = list(conn.catalog.get_schema_names())
                for schema in schemas:
                    for table in conn.catalog.get_table_names(schema):
                        tdef = conn.catalog.get_table_definition(table)
                        columns = [_stringify_hyper_name(c.name) for c in tdef.columns]
                        query = f"SELECT * FROM {table}"
                        if max_rows is not None:
                            query += f" LIMIT {int(max_rows)}"
                        rows = conn.execute_list_query(query)
                        df = pd.DataFrame(rows, columns=columns)
                        name = str(table)
                        diagnostics["tables"].append({"table": name, "rows": int(len(df)), "columns": columns})
                        tables.append((name, df))
        diagnostics["available"] = True
    except Exception as e:
        diagnostics["error"] = f"Hyper read failed: {type(e).__name__}: {e}"
    return tables, diagnostics


def read_packaged_tables(extract_dir: Path) -> tuple[list[tuple[str, pd.DataFrame]], list[dict]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    diagnostics: list[dict] = []
    for p in sorted(extract_dir.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        rel = str(p.relative_to(extract_dir))
        if suffix == ".csv":
            try:
                df = pd.read_csv(p, low_memory=False)
                tables.append((rel, df))
                diagnostics.append({"source": rel, "type": "csv", "rows": len(df), "columns": list(map(str, df.columns))})
            except Exception as e:
                diagnostics.append({"source": rel, "type": "csv", "error": str(e)})
        elif suffix in {".xlsx", ".xls"}:
            try:
                sheets = pd.read_excel(p, sheet_name=None)
                for sheet, df in sheets.items():
                    name = f"{rel}::{sheet}"
                    tables.append((name, df))
                    diagnostics.append({"source": name, "type": "excel", "rows": len(df), "columns": list(map(str, df.columns))})
            except Exception as e:
                diagnostics.append({"source": rel, "type": "excel", "error": str(e)})
        elif suffix == ".hyper":
            ht, hd = read_hyper_tables(p)
            diagnostics.append({"source": rel, "type": "hyper", **hd})
            for name, df in ht:
                tables.append((f"{rel}::{name}", df))
    return tables, diagnostics


def expected_games_view_titles(season: int | None = None, week: int | None = None) -> list[str]:
    """Return display-title candidates used by the live CFB Picker game dashboard.

    Tableau can change the *displayed* dashboard title without changing the workbook
    name.  In 2026 the live page has been observed with a title such as
    ``Games 2026: Wk 1-2``.  We therefore treat the title as a discovery hint, never
    as a single hard-coded source of truth.
    """
    if season is None:
        return []
    season = int(season)
    out = [f"Games {season}"]
    if week is not None:
        week = int(week)
        out = [
            f"Games {season}: Wk 1-{week}",
            f"Games {season}: Wk {week}",
            f"Games {season} Wk 1-{week}",
            f"Games {season} Wk {week}",
            *out,
        ]
    return list(dict.fromkeys(out))


def tableau_route_variants(name: str) -> list[str]:
    """Generate conservative Tableau route variants for one visible sheet title."""
    raw = str(name).strip()
    if not raw:
        return []
    variants = [raw]
    # Tableau Public often publishes routes with spaces/colons removed while the
    # visible dashboard caption retains them (e.g. Games 2026: Wk 1-2).
    variants.extend([
        re.sub(r"[\s:]+", "", raw),
        re.sub(r"[^A-Za-z0-9+_-]+", "", raw),
        re.sub(r"[^A-Za-z0-9]+", "", raw),
    ])
    return list(dict.fromkeys(v for v in variants if v))


def tableau_view_candidates(
    worksheet_names: Iterable[str],
    dashboard_names: Iterable[str] = (),
    *,
    target_season: int | None = None,
    target_week: int | None = None,
) -> list[str]:
    """Build ordered live-view candidates without assuming Standings/List names."""
    discovered = [*list(dashboard_names), *list(worksheet_names)]
    gameish = [x for x in discovered if re.search(r"games?|list|prediction|detail", str(x), re.I)]
    return list(dict.fromkeys([
        *expected_games_view_titles(target_season, target_week),
        *gameish,
        *DEFAULT_VIEW_CANDIDATES,
        *discovered,
    ]))


def _tableau_export_params(target_season: int | None, target_week: int | None) -> dict[str, str]:
    params = {
        ":showVizHome": "no",
        ":showTabs": "false",
        ":toolbar": "no",
        "Perspective": "Forward",
        "Show Line": "Yes",
        "Vs Line X": "Close",
    }
    if target_season is not None:
        params["Year"] = str(int(target_season))
    if target_week is not None:
        params["Week"] = str(int(target_week))
    # Current-week pages contain unfinished games. Historical Tableau views simply
    # ignore this value when the field/caption differs.
    if target_season is not None:
        params["Game Complete"] = "No"
    return params


def export_discovered_worksheets(
    worksheet_names: Iterable[str],
    dashboard_names: Iterable[str] = (),
    *,
    target_season: int | None = None,
    target_week: int | None = None,
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict]]:
    """Try TWB-discovered and title-derived Tableau CSV routes.

    A successful HTTP 200 is not enough: Tableau can return its HTML app shell for
    an invalid route.  We only retain responses that pandas can parse as a
    non-empty CSV.  Diagnostics preserve every attempted route and final URL.
    """
    names = tableau_view_candidates(
        worksheet_names,
        dashboard_names,
        target_season=target_season,
        target_week=target_week,
    )
    params = _tableau_export_params(target_season, target_week)
    query = urlencode(params)
    tables: list[tuple[str, pd.DataFrame]] = []
    diagnostics: list[dict] = []
    attempted_routes: set[str] = set()

    with requests.Session() as sess:
        sess.headers.update(BROWSER_HEADERS)
        for display_name in names:
            for route in tableau_route_variants(display_name):
                route_key = str(route).strip().casefold()
                if not route_key or route_key in attempted_routes:
                    continue
                attempted_routes.add(route_key)
                safe = quote(route, safe="")
                url = f"{VIEW_BASE}/{safe}.csv?{query}"
                try:
                    r = sess.get(url, timeout=45, allow_redirects=True)
                    info = {
                        "display_name": str(display_name),
                        "route_candidate": route,
                        "url": url,
                        "final_url": r.url,
                        "status": int(r.status_code),
                        "bytes": len(r.content),
                        "content_type": r.headers.get("content-type", ""),
                    }
                    if r.status_code == 200 and len(r.content) > 0:
                        head = r.content[:300].lstrip().lower()
                        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
                            info["rejected"] = "html_app_shell"
                        else:
                            try:
                                df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
                                info.update({"rows": len(df), "columns": list(map(str, df.columns))})
                                if len(df) > 0:
                                    # Filtered Tableau exports can omit Year/Week because
                                    # they are constant. Preserve the requested context so
                                    # downstream current-week filtering still works.
                                    if target_season is not None and _pick_col(df, ["season", "season year", "year"]) is None:
                                        df["Season"] = int(target_season)
                                    # Do NOT synthesize Week when it is absent. A live title
                                    # such as "Games 2026: Wk 1-2" can be cumulative. The
                                    # current refresher resolves those rows against the
                                    # canonical game map and then keeps the requested week.
                                    source_name = f"tableau_csv::{display_name}::{route}"
                                    tables.append((source_name, df))
                            except Exception as e:
                                info["parse_error"] = str(e)
                    diagnostics.append(info)
                except Exception as e:
                    diagnostics.append({
                        "display_name": str(display_name),
                        "route_candidate": route,
                        "url": url,
                        "error": f"{type(e).__name__}: {e}",
                    })
    return tables, diagnostics


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c in df.columns:
        k = compact(c)
        if k and k not in out:
            out[k] = str(c)
    return out


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cmap = _column_map(df)
    for c in candidates:
        key = compact(c)
        if key in cmap:
            return cmap[key]
    # conservative partial match for sufficiently descriptive aliases
    for c in candidates:
        key = compact(c)
        if len(key) < 5:
            continue
        for k, original in cmap.items():
            if key in k or k in key:
                return original
    return None


def _numeric(series: pd.Series) -> pd.Series:
    # Tableau exports occasionally attach % or Unicode minus signs.
    s = series.astype(str).str.replace("−", "-", regex=False).str.replace("–", "-", regex=False)
    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    s = s.str.replace(r"^PK$", "", regex=True)
    return pd.to_numeric(s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False), errors="coerce")


def _season_from_value(series: pd.Series) -> pd.Series:
    n = _numeric(series)
    return n.where(n.between(2000, 2100))


def _week_from_value(series: pd.Series) -> pd.Series:
    n = _numeric(series)
    return n.where(n.between(0, 30))


def _split_matchup(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    away = pd.Series(index=series.index, dtype="object")
    home = pd.Series(index=series.index, dtype="object")
    patterns = [r"\s+@\s+", r"\s+at\s+", r"\s+vs\.?\s+"]
    for i, value in series.astype(str).items():
        parts = None
        for p in patterns:
            m = re.split(p, value, maxsplit=1, flags=re.I)
            if len(m) == 2:
                parts = m
                break
        if parts:
            away.at[i] = parts[0].strip()
            home.at[i] = parts[1].strip()
    return away, home


def discover_picker_values(tables: list[tuple[str, pd.DataFrame]]) -> list[str]:
    vals: list[str] = []
    for _, df in tables:
        if df.empty:
            continue
        model_col = _pick_col(df, ["model", "model name", "picker", "system", "rating system", "projection model"])
        if model_col:
            x = df[model_col].dropna().astype(str).str.strip()
            vals.extend([v for v in x.unique().tolist() if v and len(v) <= 80])
    # Retain known names if they appear as wide columns.
    all_cols = [str(c) for _, df in tables for c in df.columns]
    for known in KNOWN_PICKER_NAMES:
        if any(compact(c) == compact(known) for c in all_cols):
            vals.append(known)
    return list(dict.fromkeys(vals))


@dataclass
class ParsedTable:
    source_table: str
    frame: pd.DataFrame
    parser: str
    notes: dict


def _base_game_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "season": _pick_col(df, ["season", "season year", "year"]),
        "week": _pick_col(df, ["week", "week number", "wk"]),
        "away": _pick_col(df, ["away", "away team", "road", "road team", "visitor", "visitor team"]),
        "home": _pick_col(df, ["home", "home team", "host", "host team"]),
        "matchup": _pick_col(df, ["matchup", "game", "game name", "teams"]),
        "date": _pick_col(df, ["date", "game date", "kickoff date"]),
        "market": _pick_col(df, ["closing line", "close line", "closing spread", "market line", "vegas line", "spread", "line"]),
        "actual_home": _pick_col(df, ["actual home margin", "home margin", "final margin", "actual margin"]),
        "home_final": _pick_col(df, ["home final", "home final score", "home score", "final home score"]),
        "away_final": _pick_col(df, ["away final", "away final score", "away score", "final away score"]),
    }


def _materialize_game_fields(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if cols["season"]:
        out["season"] = _season_from_value(df[cols["season"]])
    elif cols["date"]:
        dt = pd.to_datetime(df[cols["date"]], errors="coerce")
        out["season"] = dt.dt.year
    else:
        out["season"] = np.nan
    out["week"] = _week_from_value(df[cols["week"]]) if cols["week"] else np.nan
    if cols["away"] and cols["home"]:
        out["away"] = df[cols["away"]].astype(str).str.strip()
        out["home"] = df[cols["home"]].astype(str).str.strip()
    elif cols["matchup"]:
        away, home = _split_matchup(df[cols["matchup"]])
        out["away"] = away
        out["home"] = home
    else:
        out["away"] = np.nan
        out["home"] = np.nan
    out["market_home_margin_close"] = _numeric(df[cols["market"]]) if cols["market"] else np.nan
    if cols["actual_home"]:
        out["actual_home_margin"] = _numeric(df[cols["actual_home"]])
    elif cols["home_final"] and cols["away_final"]:
        out["actual_home_margin"] = _numeric(df[cols["home_final"]]) - _numeric(df[cols["away_final"]])
    else:
        out["actual_home_margin"] = np.nan
    return out


def parse_long_table(name: str, df: pd.DataFrame) -> ParsedTable | None:
    if df.empty:
        return None
    cols = _base_game_columns(df)
    model_col = _pick_col(df, ["model", "model name", "picker", "system", "rating system", "projection model"])
    pred_margin = _pick_col(df, [
        "prediction home margin", "predicted home margin", "projected home margin", "model home margin",
        "prediction margin", "predicted margin", "projected margin", "model margin", "prediction", "projection",
    ])
    home_pred = _pick_col(df, ["predicted home score", "home predicted score", "projected home score", "home projection", "home prediction"])
    away_pred = _pick_col(df, ["predicted away score", "away predicted score", "projected away score", "away projection", "away prediction"])
    has_game = (cols["away"] and cols["home"]) or cols["matchup"]
    if not model_col or not has_game or (not pred_margin and not (home_pred and away_pred)):
        return None
    base = _materialize_game_fields(df, cols)
    base["picker"] = df[model_col].astype(str).str.strip()
    # Conservative fuzzy matching can map a generic "Prediction" column to both
    # home_pred and away_pred. Never subtract a column from itself; in that case
    # fall back to the direct margin field below.
    if home_pred == away_pred:
        home_pred = None
        away_pred = None
    if home_pred and away_pred:
        base["prediction_raw"] = _numeric(df[home_pred]) - _numeric(df[away_pred])
        basis = "home_score_minus_away_score"
        base["orientation_certainty"] = "home_margin"
    else:
        base["prediction_raw"] = _numeric(df[pred_margin])
        basis = f"direct:{pred_margin}"
        # Explicit home-captioned fields are safe; generic Prediction is audited later via PT overlap.
        base["orientation_certainty"] = "home_margin" if "home" in norm_text(pred_margin) else "infer_from_overlap"
    base["source_table"] = name
    base = base.replace({"nan": np.nan, "None": np.nan})
    base = base.dropna(subset=["away", "home", "picker", "prediction_raw"])
    if base.empty:
        return None
    return ParsedTable(name, base, "long", {"prediction_basis": basis, "model_column": model_col})


def parse_wide_table(name: str, df: pd.DataFrame, picker_values: Iterable[str]) -> ParsedTable | None:
    if df.empty:
        return None
    cols = _base_game_columns(df)
    has_game = (cols["away"] and cols["home"]) or cols["matchup"]
    if not has_game:
        return None
    known_norm = {compact(x): canonical_picker_name(x) for x in [*KNOWN_PICKER_NAMES, *list(picker_values)]}
    model_columns: list[tuple[str, str]] = []
    for c in df.columns:
        k = compact(c)
        if k in known_norm:
            model_columns.append((str(c), known_norm[k]))
            continue
        # Tableau extracts sometimes caption wide fields as "SP+ Prediction"
        # or "FEI Projection" rather than only the model name.
        for model_key, model_name in known_norm.items():
            suffix = k[len(model_key):] if k.startswith(model_key) else ""
            if suffix in {"prediction", "projection", "pred", "margin", "spread", "projectedmargin", "predictedmargin"}:
                model_columns.append((str(c), model_name))
                break
    if not model_columns:
        return None
    base = _materialize_game_fields(df, cols)
    rows = []
    for col, picker in model_columns:
        z = base.copy()
        z["picker"] = picker
        z["prediction_raw"] = _numeric(df[col])
        z["orientation_certainty"] = "infer_from_overlap"
        z["source_table"] = name
        rows.append(z)
    out = pd.concat(rows, ignore_index=True)
    out = out.replace({"nan": np.nan, "None": np.nan}).dropna(subset=["away", "home", "picker", "prediction_raw"])
    if out.empty:
        return None
    return ParsedTable(name, out, "wide", {"model_columns": [x[0] for x in model_columns]})


def parse_prediction_tables(tables: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list[dict], list[str]]:
    picker_values = discover_picker_values(tables)
    parsed: list[pd.DataFrame] = []
    diag: list[dict] = []
    for name, df in tables:
        entry = {"source_table": name, "rows": int(len(df)), "columns": list(map(str, df.columns))}
        p = parse_long_table(name, df)
        if p is None:
            p = parse_wide_table(name, df, picker_values)
        if p is not None:
            entry.update({"parser": p.parser, "parsed_rows": int(len(p.frame)), **p.notes})
            parsed.append(p.frame)
        else:
            entry.update({"parser": None, "parsed_rows": 0})
        diag.append(entry)
    if not parsed:
        return pd.DataFrame(), diag, picker_values
    out = pd.concat(parsed, ignore_index=True, sort=False)
    out["picker"] = out["picker"].map(canonical_picker_name)
    out["away_slug"] = out["away"].map(slug_team)
    out["home_slug"] = out["home"].map(slug_team)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["prediction_raw"] = pd.to_numeric(out["prediction_raw"], errors="coerce")
    # De-duplicate identical views of the same underlying data.
    subset = ["season", "week", "away_slug", "home_slug", "picker", "prediction_raw"]
    out = out.sort_values(["source_table"]).drop_duplicates(subset=subset, keep="first")
    return out.reset_index(drop=True), diag, picker_values


def collect_cfbpicker_tables(
    cache_dir: Path,
    *,
    force_download: bool = False,
    target_season: int | None = None,
    target_week: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    workbook = cache_dir / "CFBPicker.twbx"
    download_meta = None
    inventory = {"files": [], "twb_files": [], "hyper_files": [], "csv_files": [], "xlsx_files": [], "worksheets": [], "dashboards": []}
    packaged_tables: list[tuple[str, pd.DataFrame]] = []
    packaged_diag: list[dict] = []
    try:
        if force_download or not workbook.exists() or workbook.stat().st_size < 1000:
            download_meta = download_workbook(workbook)
        else:
            download_meta = {"saved_to": str(workbook), "bytes": workbook.stat().st_size, "cached": True, "zip_signature": workbook.read_bytes()[:2] == b"PK"}
        extract_dir = cache_dir / "workbook_unpacked"
        inventory = unpack_workbook(workbook, extract_dir)
        packaged_tables, packaged_diag = read_packaged_tables(extract_dir)
    except Exception as e:
        download_meta = {"url": WORKBOOK_URL, "error": f"{type(e).__name__}: {e}"}
        packaged_diag.append({"type": "workbook", "error": download_meta["error"]})

    # Worksheet exports are fallback/supplemental. Dashboard exports are often
    # incomplete, so use TWB-discovered worksheet names plus a short set of
    # defensible guesses solely as a fallback if workbook download is blocked.
    worksheet_names = list(inventory.get("worksheets", []))
    dashboard_names = list(inventory.get("dashboards", []))
    csv_tables, csv_diag = export_discovered_worksheets(
        worksheet_names,
        dashboard_names,
        target_season=target_season,
        target_week=target_week,
    )
    all_tables = packaged_tables + csv_tables
    parsed, parse_diag, picker_values = parse_prediction_tables(all_tables)
    diagnostics = {
        "created_at_utc": utc_now(),
        "workbook_download": download_meta,
        "workbook_inventory": inventory,
        "packaged_table_diagnostics": packaged_diag,
        "target_season": target_season,
        "target_week": target_week,
        "expected_games_view_titles": expected_games_view_titles(target_season, target_week),
        "worksheet_export_diagnostics": csv_diag,
        "parse_diagnostics": parse_diag,
        "discovered_picker_names": picker_values,
        "parsed_prediction_rows": int(len(parsed)),
    }
    return parsed, diagnostics


def write_discovery_diagnostics(root: Path, diagnostics: dict, parsed: pd.DataFrame | None = None) -> None:
    derived = root / "data" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "cfbpicker_tableau_discovery.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    rows = []
    for d in diagnostics.get("parse_diagnostics", []):
        rows.append({
            "source_table": d.get("source_table"),
            "rows": d.get("rows"),
            "parsed_rows": d.get("parsed_rows"),
            "parser": d.get("parser"),
            "columns": " | ".join(map(str, d.get("columns", []))),
        })
    if rows:
        pd.DataFrame(rows).to_csv(derived / "cfbpicker_tableau_columns.csv", index=False)
    if parsed is not None and len(parsed):
        parsed.to_csv(derived / "cfbpicker_parsed_predictions_raw.csv", index=False)
