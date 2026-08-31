#!/usr/bin/env python3
"""
Fetch and normalize public College Football Prediction Tracker data.

Published sources:
  Current predictions:
    https://www.thepredictiontracker.com/ncaapredictions.csv
  Current prediction page / update timestamp:
    https://www.thepredictiontracker.com/predncaa.html
  Season performance:
    https://www.thepredictiontracker.com/ncaaresults.php
  Historical season archives:
    https://www.thepredictiontracker.com/ncaa{YEAR}.csv

The script is conservative:
- one request per requested source;
- retry/backoff for temporary failures;
- validates before replacing production files;
- never overwrites a valid production file with a failed/empty download;
- cache-busts the live current CSV/page and creates a timestamped current snapshot when content changes;
- cross-checks the current CSV slate against the live HTML page when the page can be parsed;
- supports stale-data fallback for scheduled website builds.

Usage:
  python scripts/scrape_predictiontracker.py --root . --seasons 2025

Use --strict for a run that must fail if any requested source fails.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.thepredictiontracker.com"
CURRENT_CSV_URL = f"{BASE_URL}/ncaapredictions.csv"
CURRENT_PAGE_URL = f"{BASE_URL}/predncaa.html"
RESULTS_URL = f"{BASE_URL}/ncaaresults.php"
ARCHIVE_INDEX_URL = f"{BASE_URL}/ncaaarchive.html"

USER_AGENT = (
    "manulbets-web-ncaaf-consensus/0.3 "
    "(research dashboard; low-frequency public-data retrieval)"
)

MISSING_VALUES = ["", ".", "NA", "N/A", "null", "None"]


@dataclass
class SourceRecord:
    name: str
    url: str
    status: str
    fetched_at_utc: str
    http_status: int | None = None
    bytes: int | None = None
    sha256: str | None = None
    canonical_sha256: str | None = None
    rows: int | None = None
    columns: int | None = None
    changed: bool | None = None
    production_path: str | None = None
    snapshot_path: str | None = None
    published_update: str | None = None
    message: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.strftime("%Y%m%dT%H%M%SZ")


def iso_utc(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.isoformat().replace("+00:00", "Z")


def cache_busted_url(url: str, moment: datetime | None = None) -> str:
    """Return a URL with a unique query token so current-week caches revalidate."""
    moment = moment or utc_now()
    token = moment.strftime("%Y%m%d%H%M%S%f")
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_ncaaf_refresh={token}"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/csv,text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            # PredictionTracker/CDN caching can lag the visible HTML page.
            # Current-week refreshes must ask intermediaries to revalidate.
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_bytes(
    session: requests.Session,
    url: str,
    timeout_seconds: int = 45,
    *,
    force_revalidate: bool = False,
) -> tuple[bytes, requests.Response]:
    headers = None
    if force_revalidate:
        headers = {
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        }
    response = session.get(url, timeout=timeout_seconds, headers=headers)
    response.raise_for_status()
    content = response.content
    if not content:
        raise ValueError(f"Empty response from {url}")
    return content, response


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, na_rep="")
    return buffer.getvalue().encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_if_changed(path: Path, content: bytes) -> bool:
    new_hash = sha256_bytes(content)
    old_hash = file_sha256(path)
    if old_hash == new_hash:
        return False
    atomic_write(path, content)
    return True


def clean_column_name(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def flatten_columns(columns: Iterable[Any]) -> list[str]:
    flattened: list[str] = []
    seen: dict[str, int] = {}

    for column in columns:
        if isinstance(column, tuple):
            parts = [
                str(part)
                for part in column
                if str(part).lower() not in {"nan", "none", ""}
            ]
            name = "_".join(parts)
        else:
            name = str(column)

        name = clean_column_name(name)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        flattened.append(name)

    return flattened


def read_source_csv(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(content),
        na_values=MISSING_VALUES,
        keep_default_na=True,
    )
    frame.columns = [clean_column_name(column) for column in frame.columns]
    frame = frame.loc[
        :,
        ~frame.columns.str.startswith("unnamed"),
    ].copy()
    return frame


def normalize_team_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("home", "road"):
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return frame


def coerce_prediction_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_names = {
        "line",
        "lineopen",
        "linemidweek",
        "week",
        "actual",
        "vscore",
        "hscore",
        "total",
        "phcover",
        "phwin",
    }

    for column in frame.columns:
        if column.startswith("line") or column in numeric_names:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def normalize_current(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_source_csv(content)
    required = {"line", "road", "home"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Current prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    frame = normalize_team_columns(frame)
    frame = coerce_prediction_numeric_columns(frame)

    initial_rows = len(frame)
    frame = frame.dropna(subset=["home", "road", "line"]).copy()
    incomplete_rows_removed = initial_rows - len(frame)

    before_exact = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    exact_duplicates_removed = before_exact - len(frame)

    # Same matchup and market line should not have conflicting rows.
    key = ["road", "home", "line"]
    conflicts = (
        frame.groupby(key, dropna=False)
        .size()
        .reset_index(name="rows")
        .query("rows > 1")
    )
    if not conflicts.empty:
        examples = conflicts.head(5).to_dict(orient="records")
        raise ValueError(
            "Conflicting duplicate current-game rows remain after exact "
            f"deduplication. Examples: {examples}"
        )

    model_columns = [
        column
        for column in frame.columns
        if column.startswith("line") and column != "line"
    ]
    if len(model_columns) < 3:
        raise ValueError(
            f"Only {len(model_columns)} model columns were found; expected at least 3."
        )

    nonmissing_models = frame[model_columns].notna().sum(axis=1)
    if int(nonmissing_models.max()) < 3:
        raise ValueError("No current game has at least three model predictions.")

    frame = frame.sort_values(["home", "road"], kind="stable").reset_index(drop=True)

    report = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "model_columns": len(model_columns),
        "minimum_models_per_game": int(nonmissing_models.min()),
        "median_models_per_game": float(nonmissing_models.median()),
        "maximum_models_per_game": int(nonmissing_models.max()),
        "incomplete_rows_removed": incomplete_rows_removed,
        "exact_duplicates_removed": exact_duplicates_removed,
    }
    return frame, report


def normalize_archive(
    content: bytes,
    season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_source_csv(content)
    required = {"home", "road", "line", "week"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Archive {season} is missing required columns: "
            + ", ".join(missing)
        )

    frame = normalize_team_columns(frame)
    frame = coerce_prediction_numeric_columns(frame)

    if "actual" not in frame.columns:
        if {"hscore", "vscore"}.issubset(frame.columns):
            frame["actual"] = frame["hscore"] - frame["vscore"]
        else:
            raise ValueError(
                f"Archive {season} lacks actual and hscore/vscore."
            )

    frame["season"] = season
    frame = frame.dropna(
        subset=["home", "road", "line", "week", "actual"]
    ).copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame = frame.drop_duplicates().reset_index(drop=True)

    first_columns = [
        column
        for column in (
            "season",
            "week",
            "home",
            "road",
            "line",
            "actual",
            "hscore",
            "vscore",
        )
        if column in frame.columns
    ]
    remaining = [column for column in frame.columns if column not in first_columns]
    frame = frame[first_columns + remaining]

    report = {
        "season": season,
        "rows": len(frame),
        "columns": len(frame.columns),
        "model_columns": len(
            [
                column
                for column in frame.columns
                if column.startswith("line") and column != "line"
            ]
        ),
        "minimum_week": int(frame["week"].min()),
        "maximum_week": int(frame["week"].max()),
    }
    return frame, report


def parse_prediction_page_update(html_content: bytes) -> str | None:
    text = BeautifulSoup(html_content, "html.parser").get_text(" ", strip=True)
    match = re.search(
        r"Updated:\s*(.+?)(?=\s+Home\s+Visitor|\s+Home\s+Road|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def parse_prediction_page_summary(html_content: bytes) -> pd.DataFrame:
    """Best-effort parser for the visible current-game summary on predncaa.html.

    PredictionTracker renders the board as fixed-width/preformatted text.  The
    summary rows begin with Home / Visitor followed by Opening, Updated, and
    Midweek lines.  We deliberately require those first three numeric columns,
    which keeps the later raw-model matrix from being mistaken for the summary.

    This is a *validation* parser, not the production model-data parser.  If the
    site's markup changes and no rows can be recognized, callers record
    validation_unavailable rather than replacing the CSV with guessed HTML data.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    blocks = [x.get_text("\n", strip=False) for x in soup.find_all("pre")]
    if not blocks:
        blocks = [soup.get_text("\n", strip=False)]

    row_re = re.compile(
        r"^\s*(?P<home>\S(?:.*?\S)?)\s{2,}"
        r"(?P<road>\S(?:.*?\S)?)\s{2,}"
        r"(?P<lineopen>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<line>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<linemidweek>[+-]?\d+(?:\.\d+)?)(?:\s+|$)"
    )

    rows = []
    for block in blocks:
        if not (
            re.search(r"Opening\s+Updated\s+Midweek", block, re.I)
            or (
                re.search(r"Opening\s+line", block, re.I)
                and re.search(r"Updated\s+line", block, re.I)
                and re.search(r"Midweek\s+line", block, re.I)
            )
        ):
            continue
        for line in block.splitlines():
            m = row_re.match(line)
            if not m:
                continue
            g = m.groupdict()
            rows.append({
                "home": re.sub(r"\s+", " ", g["home"]).strip(),
                "road": re.sub(r"\s+", " ", g["road"]).strip(),
                "lineopen": float(g["lineopen"]),
                "line": float(g["line"]),
                "linemidweek": float(g["linemidweek"]),
            })

    if not rows:
        return pd.DataFrame(columns=["home", "road", "lineopen", "line", "linemidweek"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _team_slug(value: Any) -> str:
    x = str(value or "").lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def validate_current_csv_against_page(
    normalized: pd.DataFrame,
    html_content: bytes,
) -> dict[str, Any]:
    """Detect the failure mode where the HTML page is fresh but CSV is stale."""
    page = parse_prediction_page_summary(html_content)
    if len(page) < 5:
        return {
            "status": "validation_unavailable",
            "page_games": (int(len(page)) if len(page) else None),
            "csv_games": int(len(normalized)),
            "matched_games": None,
            "match_rate": None,
            "line_match_rate": None,
            "message": "Could not parse the visible PredictionTracker summary table; CSV was validated structurally only.",
        }

    csv = normalized[[c for c in ["home", "road", "line", "lineopen", "linemidweek"] if c in normalized.columns]].copy()
    for d in (page, csv):
        d["_game_key"] = [
            _team_slug(h) + "__" + _team_slug(r)
            for h, r in zip(d["home"], d["road"])
        ]
    page = page.drop_duplicates("_game_key")
    csv = csv.drop_duplicates("_game_key")

    page_keys = set(page["_game_key"])
    csv_keys = set(csv["_game_key"])
    matched = page_keys & csv_keys
    denom = max(1, len(page_keys))
    match_rate = len(matched) / denom

    line_matches = []
    if matched and "line" in csv.columns:
        p_line = page.set_index("_game_key")["line"]
        c_line = pd.to_numeric(csv.set_index("_game_key")["line"], errors="coerce")
        for key in matched:
            a = p_line.get(key)
            b = c_line.get(key)
            if pd.notna(a) and pd.notna(b):
                line_matches.append(abs(float(a) - float(b)) < 0.01)
    line_match_rate = (
        sum(line_matches) / len(line_matches) if line_matches else None
    )

    # A current board should substantially agree with the board users see on
    # predncaa.html.  A stale prior-week CSV typically has almost no overlap.
    severe_count_gap = abs(len(page_keys) - len(csv_keys)) > max(3, int(round(0.20 * len(page_keys))))
    bad_overlap = match_rate < 0.80
    bad_lines = line_match_rate is not None and line_match_rate < 0.80
    status = "ok" if not (severe_count_gap or bad_overlap or bad_lines) else "mismatch"

    out = {
        "status": status,
        "page_games": int(len(page_keys)),
        "csv_games": int(len(csv_keys)),
        "matched_games": int(len(matched)),
        "match_rate": float(match_rate),
        "line_match_rate": (float(line_match_rate) if line_match_rate is not None else None),
        "severe_count_gap": bool(severe_count_gap),
    }
    if status != "ok":
        page_only = sorted(page_keys - csv_keys)[:5]
        csv_only = sorted(csv_keys - page_keys)[:5]
        out["message"] = (
            "Current CSV does not match the visible PredictionTracker page. "
            f"page_games={len(page_keys)}, csv_games={len(csv_keys)}, "
            f"match_rate={match_rate:.1%}, line_match_rate="
            f"{line_match_rate if line_match_rate is not None else 'NA'}. "
            f"Examples page-only={page_only}; csv-only={csv_only}"
        )
    else:
        out["message"] = "Current CSV agrees with the visible PredictionTracker board."
    return out


def parse_results_html(
    content: bytes,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    html_text = content.decode("utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html_text))

    selected: pd.DataFrame | None = None
    for candidate in tables:
        candidate = candidate.copy()
        candidate.columns = flatten_columns(candidate.columns)
        column_set = set(candidate.columns)

        has_system = any(
            column == "system" or column.endswith("_system")
            for column in column_set
        )
        has_mse = any(
            "mean_square_error" in column
            for column in column_set
        )
        if has_system and has_mse and len(candidate) >= 5:
            selected = candidate
            break

    if selected is None:
        raise ValueError(
            "Could not identify the NCAA season totals table in results HTML."
        )

    rename: dict[str, str] = {}
    for column in selected.columns:
        if column == "system" or column.endswith("_system"):
            rename[column] = "system"
        elif "pct_correct" in column:
            rename[column] = "pct_correct"
        elif "against_spread" in column:
            rename[column] = "ats_pct"
        elif "absolute_error" in column:
            rename[column] = "absolute_error"
        elif column.endswith("bias") or column == "bias":
            rename[column] = "bias"
        elif "mean_square_error" in column:
            rename[column] = "mse"
        elif column.endswith("games") or column == "games":
            rename[column] = "games"
        elif column.endswith("suw") or column == "suw":
            rename[column] = "suw"
        elif column.endswith("sul") or column == "sul":
            rename[column] = "sul"
        elif column.endswith("atsw") or column == "atsw":
            rename[column] = "atsw"
        elif column.endswith("atsl") or column == "atsl":
            rename[column] = "atsl"
        elif column.endswith("rank") or column == "rank":
            rename[column] = "rank"

    selected = selected.rename(columns=rename)
    selected = selected.loc[
        :,
        ~selected.columns.duplicated(),
    ].copy()

    required = {"system", "mse", "games"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(
            "Parsed season totals table is missing: "
            + ", ".join(missing)
        )

    selected["system"] = (
        selected["system"]
        .astype("string")
        .str.strip()
    )
    selected = selected[
        selected["system"].notna()
        & selected["system"].ne("")
    ].copy()

    for column in (
        "rank",
        "pct_correct",
        "ats_pct",
        "absolute_error",
        "bias",
        "mse",
        "games",
        "suw",
        "sul",
        "atsw",
        "atsl",
    ):
        if column in selected.columns:
            selected[column] = pd.to_numeric(
                selected[column],
                errors="coerce",
            )

    page_text = BeautifulSoup(html_text, "html.parser").get_text(
        " ",
        strip=True,
    )
    season_match = re.search(
        r"(\d{4})\s+Season Totals",
        page_text,
        flags=re.IGNORECASE,
    )
    through_match = re.search(
        r"Through\s+(\d{4}-\d{2}-\d{2})",
        page_text,
        flags=re.IGNORECASE,
    )
    season = int(season_match.group(1)) if season_match else None
    through_date = through_match.group(1) if through_match else None

    selected.insert(0, "season", season)
    selected.insert(1, "through_date", through_date)
    selected = selected.reset_index(drop=True)

    report = {
        "season": season,
        "through_date": through_date,
        "rows": len(selected),
        "columns": len(selected.columns),
    }
    return selected, report


def update_production_and_snapshot(
    normalized: pd.DataFrame,
    raw_content: bytes,
    production_path: Path,
    raw_path: Path,
    snapshot_directory: Path | None,
    snapshot_prefix: str,
    run_time: datetime,
) -> tuple[bool, str | None]:
    canonical = canonical_csv_bytes(normalized)
    old_hash = file_sha256(production_path)
    new_hash = sha256_bytes(canonical)
    changed = old_hash != new_hash

    # Raw source is useful for reproducibility and source-parser debugging.
    write_if_changed(raw_path, raw_content)

    snapshot_path: Path | None = None
    if changed:
        atomic_write(production_path, canonical)
        if snapshot_directory is not None:
            snapshot_directory.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_directory / (
                f"{utc_stamp(run_time)}_{snapshot_prefix}.csv"
            )
            atomic_write(snapshot_path, canonical)

    return changed, str(snapshot_path) if snapshot_path else None


def combine_archives(
    archive_frames: list[pd.DataFrame],
    output_path: Path,
) -> bool:
    if not archive_frames:
        return False

    combined = pd.concat(
        archive_frames,
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(
        ["season", "week", "home", "road"],
        kind="stable",
    ).reset_index(drop=True)
    return write_if_changed(
        output_path,
        canonical_csv_bytes(combined),
    )


def process_current(
    session: requests.Session,
    root: Path,
    run_time: datetime,
) -> tuple[SourceRecord, pd.DataFrame | None]:
    fetched_at = iso_utc(run_time)
    record = SourceRecord(
        name="current_predictions",
        url=CURRENT_CSV_URL,
        status="error",
        fetched_at_utc=fetched_at,
    )

    try:
        # Fetch the visible page and CSV with the SAME unique cache-busting token.
        # This avoids the observed state where predncaa.html advances to the new
        # week while an intermediary continues serving the prior ncaapredictions.csv.
        token_time = run_time
        page_content = None
        published_update = None
        try:
            page_content, _ = fetch_bytes(
                session,
                cache_busted_url(CURRENT_PAGE_URL, token_time),
                force_revalidate=True,
            )
            published_update = parse_prediction_page_update(page_content)
            write_if_changed(
                root / "data/raw/predictiontracker/predncaa.html",
                page_content,
            )
        except Exception as exc:
            logging.warning(
                "Could not fetch/parse current prediction page metadata: %s",
                exc,
            )

        content, response = fetch_bytes(
            session,
            cache_busted_url(CURRENT_CSV_URL, token_time),
            force_revalidate=True,
        )
        normalized, report = normalize_current(content)

        validation = None
        if page_content is not None:
            validation = validate_current_csv_against_page(normalized, page_content)
            report["page_validation"] = validation
            if validation.get("status") == "mismatch":
                raise ValueError(validation.get("message") or "PredictionTracker CSV/page mismatch")
        else:
            report["page_validation"] = {
                "status": "page_fetch_failed",
                "message": "Visible page could not be fetched; CSV structural validation only.",
            }

        production_path = root / "data/current/ncaapredictions.csv"
        raw_path = (
            root
            / "data/raw/predictiontracker/current_ncaapredictions.csv"
        )
        snapshot_directory = (
            root
            / "data/snapshots/predictiontracker/current"
        )
        changed, snapshot_path = update_production_and_snapshot(
            normalized=normalized,
            raw_content=content,
            production_path=production_path,
            raw_path=raw_path,
            snapshot_directory=snapshot_directory,
            snapshot_prefix="ncaapredictions",
            run_time=run_time,
        )

        record.status = "ok"
        record.http_status = response.status_code
        record.bytes = len(content)
        record.sha256 = sha256_bytes(content)
        record.canonical_sha256 = sha256_bytes(canonical_csv_bytes(normalized))
        report["canonical_sha256"] = record.canonical_sha256
        record.rows = report["rows"]
        record.columns = report["columns"]
        record.changed = changed
        record.production_path = str(production_path.relative_to(root))
        record.snapshot_path = (
            str(Path(snapshot_path).relative_to(root))
            if snapshot_path
            else None
        )
        record.published_update = published_update
        record.message = json.dumps(report, sort_keys=True)
        return record, normalized

    except Exception as exc:
        record.message = f"{type(exc).__name__}: {exc}"
        return record, None


def process_archive(
    session: requests.Session,
    root: Path,
    season: int,
    run_time: datetime,
) -> tuple[SourceRecord, pd.DataFrame | None]:
    url = f"{BASE_URL}/ncaa{season}.csv"
    record = SourceRecord(
        name=f"archive_{season}",
        url=url,
        status="error",
        fetched_at_utc=iso_utc(run_time),
    )

    try:
        content, response = fetch_bytes(session, url)
        normalized, report = normalize_archive(content, season)

        production_path = (
            root / f"data/historical/ncaa{season}.csv"
        )
        raw_path = (
            root
            / f"data/raw/predictiontracker/archive_ncaa{season}.csv"
        )
        changed, _ = update_production_and_snapshot(
            normalized=normalized,
            raw_content=content,
            production_path=production_path,
            raw_path=raw_path,
            snapshot_directory=None,
            snapshot_prefix=f"ncaa{season}",
            run_time=run_time,
        )

        record.status = "ok"
        record.http_status = response.status_code
        record.bytes = len(content)
        record.sha256 = sha256_bytes(content)
        record.rows = report["rows"]
        record.columns = report["columns"]
        record.changed = changed
        record.production_path = str(production_path.relative_to(root))
        record.message = json.dumps(report, sort_keys=True)
        return record, normalized

    except Exception as exc:
        record.message = f"{type(exc).__name__}: {exc}"
        return record, None


def process_results(
    session: requests.Session,
    root: Path,
    run_time: datetime,
) -> tuple[SourceRecord, pd.DataFrame | None]:
    record = SourceRecord(
        name="season_performance",
        url=RESULTS_URL,
        status="error",
        fetched_at_utc=iso_utc(run_time),
    )

    try:
        content, response = fetch_bytes(session, RESULTS_URL)
        normalized, report = parse_results_html(content)

        production_path = (
            root
            / "data/reference/predictiontracker_season_totals.csv"
        )
        raw_path = (
            root
            / "data/raw/predictiontracker/ncaaresults.html"
        )
        changed = write_if_changed(
            production_path,
            canonical_csv_bytes(normalized),
        )
        write_if_changed(raw_path, content)

        record.status = "ok"
        record.http_status = response.status_code
        record.bytes = len(content)
        record.sha256 = sha256_bytes(content)
        record.rows = report["rows"]
        record.columns = report["columns"]
        record.changed = changed
        record.production_path = str(production_path.relative_to(root))
        record.message = json.dumps(report, sort_keys=True)
        return record, normalized

    except Exception as exc:
        record.message = f"{type(exc).__name__}: {exc}"
        return record, None


def existing_production_is_usable(root: Path) -> bool:
    current = root / "data/current/ncaapredictions.csv"
    history = root / "data/historical/ncaa2025.csv"
    return current.exists() and history.exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Prediction Tracker current, archive, and results data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2025],
        help="Archive seasons to fetch.",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
    )
    parser.add_argument(
        "--skip-results",
        action="store_true",
    )
    parser.add_argument(
        "--skip-archives",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero if any requested source fails.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Pause between source requests.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_time = utc_now()
    session = create_session()
    records: list[SourceRecord] = []
    archive_frames: list[pd.DataFrame] = []

    if not args.skip_current:
        logging.info("Fetching current Prediction Tracker CSV")
        record, _ = process_current(session, root, run_time)
        records.append(record)
        logging.info("%s: %s", record.name, record.status)
        time.sleep(args.delay_seconds)

    if not args.skip_results:
        logging.info("Fetching season performance table")
        record, _ = process_results(session, root, run_time)
        records.append(record)
        logging.info("%s: %s", record.name, record.status)
        time.sleep(args.delay_seconds)

    if not args.skip_archives:
        for index, season in enumerate(sorted(set(args.seasons))):
            logging.info("Fetching Prediction Tracker archive %s", season)
            record, frame = process_archive(
                session,
                root,
                season,
                run_time,
            )
            records.append(record)
            logging.info("%s: %s", record.name, record.status)
            if frame is not None:
                archive_frames.append(frame)
            if index < len(set(args.seasons)) - 1:
                time.sleep(args.delay_seconds)

        if archive_frames:
            combine_archives(
                archive_frames,
                root / "data/historical/ncaa_history.csv",
            )

    failures = [record for record in records if record.status != "ok"]
    manifest = {
        "run_started_at_utc": iso_utc(run_time),
        "run_finished_at_utc": iso_utc(),
        "overall_status": "ok" if not failures else "stale_fallback",
        "requested_archive_seasons": sorted(set(args.seasons)),
        "records": [asdict(record) for record in records],
    }

    manifest_path = (
        root
        / "data/derived/predictiontracker_source_status.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        manifest_path,
        json.dumps(
            manifest,
            indent=2,
            sort_keys=False,
        ).encode("utf-8"),
    )

    if failures:
        logging.error(
            "%s requested source(s) failed. See %s",
            len(failures),
            manifest_path,
        )
        for failure in failures:
            logging.error("%s: %s", failure.name, failure.message)

        if args.strict or not existing_production_is_usable(root):
            return 2

        logging.warning(
            "Continuing with previously committed production data."
        )

    print(f"Prediction Tracker source status: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
