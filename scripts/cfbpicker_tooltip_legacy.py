#!/usr/bin/env python3
"""
Scrape CFB Picker game-level history from the narrow Tableau L# worksheet.

Why this works
--------------
The full List dashboard is server-rendered as image tiles. When the view is
restricted to one year, week, and picker, Tableau switches to a small
client-rendered worksheet. Game headers are available as DOM cells and each
prediction mark exposes a structured Tableau tooltip after a click.

No OCR is used.

Outputs
-------
Per-page checkpoints:
  data/raw/cfbpicker/tooltips/year=YYYY/picker=<key>/week=WW.csv

Combined:
  data/cfbpicker/cfbpicker_history_long.csv
  data/cfbpicker/cfbpicker_history_wide.csv
  data/cfbpicker/cfbpicker_model_diagnostics.csv
  data/cfbpicker/cfbpicker_model_map.csv
  data/cfbpicker/discovered_pickers_YYYY.csv
  data/cfbpicker/candidate_pickers_YYYY.txt
  data/cfbpicker/collectable_pickers_YYYY.txt

Examples
--------
Discover the forward-looking picker list:
  python scripts/scrape_cfbpicker_history.py \
    --root . --year 2025 --discover-only \
    --headed --system-chrome

Test one model:
  python scripts/scrape_cfbpicker_history.py \
    --root . --year 2025 --weeks 1-22 \
    --pickers "CFB Geek" \
    --headed --system-chrome --strict

Scrape every genuine forward model, including models also hosted by
Prediction Tracker (needed for overlap/deduplication analysis):
  python scripts/scrape_cfbpicker_history.py \
    --root . --year 2025 --weeks 1-22 \
    --all-models --headed --system-chrome --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from html import unescape
from html.parser import HTMLParser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd

BASE = "https://public.tableau.com/views/CFBPicker"
LIST_ROUTE = "L"
STANDINGS_ROUTE = "Standings"

PSEUDO_OR_AGGREGATE = {
    "CFP Resume Ranks": "resume/outcome ranking rather than a spread model",
    "Line: MCS": "market-line pseudo-model",
    "Line: Open": "market-line pseudo-model",
    "Metrics Consensus": "aggregate of component models",
    "Metrics Consensus ^": "aggregate/variant",
    "Metrics Consensus +": "aggregate/variant",
}

# Models already represented in the Prediction Tracker historical matrix.
KNOWN_PREDICTIONTRACKER_EQUIVALENTS = {
    "Congrove": "linecong",
    "Dokter Entropy": "linedokter",
    "ESPN FPI": "lineespn",
    "Harville": "lineharville",
    "Keeper": "linekeep",
    "Massey": "linemass",
    "Moore Ratings": "linemoore",
    "PiRate": "linepiratings",
    "PiRate: Bias": "linepibias",
    "PiRate: Mean": "linepimean",
    "Sagarin": "linesag",
    "Sagarin: Golden": "linesaggm",
    "Sagarin: Predictor": "linesagpred",
    "Sagarin: Recent": "linesagr",
    "TeamRankings": "lineteamrank",
}


@dataclass
class PageRecord:
    year: int
    week: int
    picker: str
    status: str
    url: str
    rows_detected: int = 0
    rows_extracted: int = 0
    checkpoint: str | None = None
    message: str | None = None


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_key(value: str) -> str:
    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def model_key(value: str) -> str:
    return "linecfbp_" + safe_key(value)


def team_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"\bstate\b", "st", text)
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    aliases = {
        "miamifl": "miami",
        "miamiflorida": "miami",
        "miamioh": "miamiohio",
        "southerncal": "usc",
        "southerncalifornia": "usc",
        "connecticut": "uconn",
        "massachusetts": "umass",
        "samhoustonst": "samhouston",
        "westernkentuckyst": "westernkentucky",
    }
    return aliases.get(text, text)


def parse_int_spec(values: Iterable[str]) -> list[int]:
    result: set[int] = set()
    for raw in values:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            match = re.fullmatch(r"(\d+)-(\d+)", piece)
            if match:
                start, end = map(int, match.groups())
                if end < start:
                    start, end = end, start
                result.update(range(start, end + 1))
            else:
                result.add(int(piece))
    return sorted(result)


def parse_picker_args(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for raw in values:
        # Preserve commas inside no current picker names; comma is a safe
        # command-line separator here.
        for piece in str(raw).split(","):
            piece = piece.strip()
            if piece:
                output.append(piece)
    return list(dict.fromkeys(output))


def build_list_url(year: int, week: int, picker: str) -> str:
    params = {
        ":showVizHome": "no",
        ":showTabs": "false",
        ":toolbar": "no",
        "Year": str(year),
        "Week": str(week),
        "Picker": picker,
        "Perspective": "Forward",
        "Game Complete": "Yes",
        "Show Incomplete": "No",
        "Show Line": "Yes",
        "Vs Line X": "Close",
    }
    return f"{BASE}/{LIST_ROUTE}?{urlencode(params)}"


def build_standings_url(year: int) -> str:
    params = {
        ":showVizHome": "no",
        ":showTabs": "true",
        ":toolbar": "no",
        "Year": str(year),
        "Perspective": "Forward",
        "Game Complete": "Yes",
        "Vs Line X": "Close",
    }
    return f"{BASE}/{STANDINGS_ROUTE}?{urlencode(params)}"


def parse_length_prefixed(content: bytes) -> list[Any]:
    text = content.decode("utf-8", errors="replace")
    objects: list[Any] = []
    position = 0

    while position < len(text):
        match = re.match(r"(\d+);", text[position:])
        if not match:
            break
        length = int(match.group(1))
        start = position + len(match.group(0))
        end = start + length
        if end > len(text):
            break
        try:
            objects.append(json.loads(text[start:end]))
        except json.JSONDecodeError:
            break
        position = end

    return objects


def parse_bootstrap_objects(responses: list[Any]) -> list[Any]:
    """Parse captured Tableau command/bootstrap responses into JSON objects.

    Tableau responses are observed in two formats: ordinary JSON bodies and
    Tableau's length-prefixed ``<n>;<json>`` stream.  The Embedding-API
    collectors capture both, so normalize them here before picker-domain
    discovery.  This adapter is the same logic used by the proven multi-season
    inventory collector.
    """
    objects: list[Any] = []
    for response in responses:
        try:
            body = response.body()
        except Exception:
            continue
        if not isinstance(body, (bytes, bytearray)):
            continue
        stripped = bytes(body).lstrip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            try:
                objects.append(json.loads(bytes(body).decode("utf-8", "replace")))
            except json.JSONDecodeError:
                continue
        else:
            objects.extend(parse_length_prefixed(bytes(body)))
    return objects


def walk(node: Any) -> Iterable[Any]:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def picker_items_from_objects(objects: list[Any]) -> list[str]:
    candidates: list[str] = []

    for root in objects:
        for node in walk(root):
            if not isinstance(node, dict):
                continue

            title = str(node.get("title", ""))
            function_name = str(node.get("fn", ""))
            item_set = node.get("dataHighlighterItemSet")

            is_picker_highlighter = (
                "Highlight Picker" in title
                or "PICKER:nk" in function_name
            )
            if not is_picker_highlighter or not isinstance(item_set, dict):
                continue

            items = item_set.get("dataHighlighterItems", [])
            if not isinstance(items, list):
                continue

            for item in items:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    if text:
                        candidates.append(text)

    return list(dict.fromkeys(candidates))


def classify_picker(picker: str) -> tuple[str, str | None]:
    if picker in PSEUDO_OR_AGGREGATE:
        return "exclude", PSEUDO_OR_AGGREGATE[picker]
    if picker in KNOWN_PREDICTIONTRACKER_EQUIVALENTS:
        return (
            "existing_predictiontracker",
            KNOWN_PREDICTIONTRACKER_EQUIVALENTS[picker],
        )
    return "incremental_candidate", None


def discover_pickers(
    context: Any,
    year: int,
    wait_seconds: int,
) -> tuple[list[str], str]:
    page = context.new_page()
    page.set_default_timeout(20_000)
    bootstrap_responses: list[Any] = []

    def on_response(response: Any) -> None:
        lower = response.url.lower()
        if (
            "startsession/viewing" in lower
            or "bootstrapsession" in lower
        ):
            bootstrap_responses.append(response)

    page.on("response", on_response)
    url = build_standings_url(year)
    print(f"Discovering pickers: {url}")
    response = page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=90_000,
    )
    if response is not None and response.status >= 400:
        raise RuntimeError(f"Standings returned HTTP {response.status}")

    page.wait_for_timeout(wait_seconds * 1000)

    objects: list[Any] = []
    for item in bootstrap_responses:
        try:
            body = item.body()
        except Exception:
            continue
        stripped = body.lstrip()
        if stripped.startswith(b"{"):
            try:
                objects.append(json.loads(body.decode("utf-8", "replace")))
            except json.JSONDecodeError:
                pass
        else:
            objects.extend(parse_length_prefixed(body))

    pickers = picker_items_from_objects(objects)
    page.close()

    if not pickers:
        raise RuntimeError(
            "The Tableau Picker highlighter domain was not found."
        )
    return pickers, url


def write_picker_outputs(
    root: Path,
    year: int,
    pickers: list[str],
) -> tuple[Path, Path]:
    output_dir = root / "data/cfbpicker"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    candidates: list[str] = []
    collectable: list[str] = []
    for picker in pickers:
        category, mapping = classify_picker(picker)
        rows.append(
            {
                "year": year,
                "picker": picker,
                "model_key": model_key(picker),
                "category": category,
                "mapping_or_reason": mapping,
                "collect": category != "exclude",
            }
        )
        if category == "incremental_candidate":
            candidates.append(picker)
        if category != "exclude":
            collectable.append(picker)

    discovery_path = output_dir / f"discovered_pickers_{year}.csv"
    pd.DataFrame(rows).to_csv(discovery_path, index=False)

    candidate_path = output_dir / f"candidate_pickers_{year}.txt"
    candidate_path.write_text(
        "\n".join(candidates) + "\n",
        encoding="utf-8",
    )

    collectable_path = output_dir / f"collectable_pickers_{year}.txt"
    collectable_path.write_text(
        "\n".join(collectable) + "\n",
        encoding="utf-8",
    )
    return candidate_path, collectable_path


def collect_header_rows(zone: Any, canvas_box: dict[str, float]) -> list[dict[str, Any]]:
    # Scope header cells to the same Tableau viz zone as the mark canvas.
    # Using page.locator() can mix headers from hidden/adjacent worksheets.
    locator = zone.locator(".tab-vizHeaderWrapper")
    cells: list[dict[str, Any]] = []
    top = canvas_box["y"]
    bottom = top + canvas_box["height"]

    for index in range(locator.count()):
        item = locator.nth(index)
        try:
            box = item.bounding_box()
            if not box:
                continue
            center_y = box["y"] + box["height"] / 2
            if not (top <= center_y <= bottom):
                continue
            text = item.inner_text(timeout=500).strip()
            if not text:
                continue
            cells.append(
                {
                    "id": item.get_attribute("id"),
                    "text": text,
                    "x": box["x"],
                    "center_y": center_y,
                }
            )
        except Exception:
            continue

    grouped: dict[float, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(round(cell["center_y"], 1), []).append(cell)

    rows: list[dict[str, Any]] = []
    for center_y, row_cells in sorted(grouped.items()):
        row_cells = sorted(row_cells, key=lambda item: item["x"])
        rows.append(
            {
                "center_y": center_y,
                "header_values": [item["text"] for item in row_cells],
                "anchor_id": next(
                    (
                        item["id"]
                        for item in row_cells
                        if item.get("id")
                    ),
                    None,
                ),
            }
        )
    return rows


def visible_tooltip_text(page: Any) -> str:
    locator = page.locator(".tab-tooltipContent:visible")
    count = locator.count()
    if count == 0:
        return ""
    try:
        return locator.last.inner_text(timeout=1_000).strip()
    except Exception:
        return ""


class _TooltipHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "td":
            self.parts.append("\t")
        elif tag in {"tr", "div", "p"}:
            self.parts.append("\n")


def tooltip_html_to_text(html: str) -> str:
    parser = _TooltipHTMLTextParser()
    parser.feed(html)
    parser.close()
    raw = unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in raw.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def tooltip_text_from_response(response: Any) -> str:
    """Read Tableau's tooltip directly from its fresh command response.

    Reading the network response avoids stale tooltip DOM nodes, which can
    remain visible after a prior selection and caused duplicate-game errors.
    """
    body = response.body().decode("utf-8", errors="replace")
    payload = json.loads(body)
    command_results = (
        payload
        .get("vqlCmdResponse", {})
        .get("cmdResultList", [])
    )
    for result in command_results:
        command_return = result.get("commandReturn", {})
        encoded = command_return.get("tooltipText")
        if not encoded:
            continue
        tooltip_payload = json.loads(encoded)
        html = tooltip_payload.get("htmlTooltip", "")
        if html:
            return tooltip_html_to_text(html)
    return ""


def click_and_read_tooltip_response(
    page: Any,
    x: float,
    y: float,
    timeout_ms: int = 6_000,
) -> str:
    """Click one Tableau mark and parse that click's response body."""
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and "render-tooltip-server" in response.url
        ),
        timeout=timeout_ms,
    ) as response_info:
        page.mouse.move(x, y)
        page.wait_for_timeout(50)
        page.mouse.down()
        page.wait_for_timeout(40)
        page.mouse.up()
    return tooltip_text_from_response(response_info.value)


def wait_for_tooltip(
    page: Any,
    picker: str,
    prior_text: str,
    timeout_ms: int = 4_000,
) -> str:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        text = visible_tooltip_text(page)
        complete = (
            "Picker:" in text
            and picker in text
            and "Pick:" in text
        )
        # Never accept the prior row's tooltip. The former fallback did so
        # after a missed click and created duplicate games on larger weeks.
        if complete and (not prior_text or text != prior_text):
            return text
        page.wait_for_timeout(150)
    return ""


def labeled_team_margin(text: str, label: str) -> tuple[str | None, float | None]:
    match = re.search(
        rf"^\s*{re.escape(label)}:\s*(.*?)\s+([+-]?\d+(?:\.\d+)?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return None, None
    return match.group(1).strip(), float(match.group(2))


def parse_tooltip(text: str) -> dict[str, Any]:
    picker_match = re.search(
        r"^\s*Picker:\s*(.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    person_match = re.search(
        r"^\s*Person:\s*(.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    twitter_match = re.search(
        r"^\s*Twitter:\s*(.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )

    line_team, line_value = labeled_team_margin(text, "Line")
    pick_team, pick_value = labeled_team_margin(text, "Pick")
    result_team, result_value = labeled_team_margin(text, "Res")

    score_matches = re.findall(
        r"^\s*(\d+)\s+(.+?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    # The final two numeric-leading lines are away and home scores.
    score_matches = score_matches[-2:]

    if (
        picker_match is None
        or pick_team is None
        or pick_value is None
        or len(score_matches) != 2
    ):
        raise ValueError(f"Could not parse tooltip:\n{text}")

    away_score, away = score_matches[0]
    home_score, home = score_matches[1]

    return {
        "picker": picker_match.group(1).strip(),
        "person": (
            person_match.group(1).strip()
            if person_match else None
        ),
        "twitter": (
            twitter_match.group(1).strip()
            if twitter_match else None
        ),
        "line_team": line_team,
        "line_value": line_value,
        "pick_team": pick_team,
        "pick_value": pick_value,
        "result_team": result_team,
        "result_value": result_value,
        "away": away.strip(),
        "home": home.strip(),
        "away_pts": int(away_score),
        "home_pts": int(home_score),
        "tooltip_text": text,
    }


def favorite_to_home_margin(
    team: str | None,
    value: float | None,
    away: str,
    home: str,
) -> float | None:
    if value is None:
        return None
    if abs(value) < 1e-12:
        return 0.0
    if team is None:
        raise ValueError("A nonzero margin has no associated team.")

    favorite = team_key(team)
    if favorite == team_key(home):
        return abs(float(value))
    if favorite == team_key(away):
        return -abs(float(value))
    raise ValueError(
        f"Tooltip team {team!r} did not match away={away!r}, home={home!r}."
    )


def tooltip_matches_header(
    parsed: dict[str, Any],
    header_values: list[str],
) -> bool:
    """Check that the clicked tooltip belongs to the intended header row.

    The first two displayed header cells are Away and Home. Exact matching is
    allowed for short team names such as SMU, UCF, TCU, BYU, USC, LSU, and UNLV.
    Prefix matching is used only when Tableau visibly truncates a label with
    dots, for example ``Jacksonvil..`` for Jacksonville State.
    """
    if len(header_values) < 2:
        return False

    def cell_matches(displayed: str, full_name: str) -> bool:
        displayed_text = str(displayed).strip()

        def raw_name_key(value: str) -> str:
            # Do not apply aliases here. A displayed truncation such as
            # ``Massachu..`` must be compared with ``Massachusetts``, not
            # with its canonical alias ``umass``.
            return re.sub(r"[^a-z0-9]", "", str(value).lower())

        displayed_alias_key = team_key(displayed_text.replace("..", ""))
        full_alias_key = team_key(full_name)

        if not displayed_alias_key or not full_alias_key:
            return False

        # Exact aliases and short names are always valid.
        if displayed_alias_key == full_alias_key:
            return True

        visibly_truncated = (
            ".." in displayed_text
            or displayed_text.endswith("…")
            or displayed_text.endswith(".")
        )
        if visibly_truncated:
            displayed_raw = raw_name_key(
                displayed_text
                .replace("..", "")
                .removesuffix("…")
                .removesuffix(".")
            )
            full_raw = raw_name_key(full_name)
            return (
                len(displayed_raw) >= 3
                and full_raw.startswith(displayed_raw)
            )

        return False

    return (
        cell_matches(header_values[0], parsed["away"])
        and cell_matches(header_values[1], parsed["home"])
    )


def checkpoint_path(root: Path, year: int, picker: str, week: int) -> Path:
    return (
        root
        / "data/raw/cfbpicker/tooltips"
        / f"year={year}"
        / f"picker={safe_key(picker)}"
        / f"week={week:02d}.csv"
    )


def scrape_page(
    context: Any,
    root: Path,
    year: int,
    week: int,
    picker: str,
    wait_seconds: int,
    force: bool,
    screenshot_errors: bool,
) -> PageRecord:
    url = build_list_url(year, week, picker)
    output_path = checkpoint_path(root, year, picker, week)

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        return PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="skipped_existing",
            url=url,
            rows_detected=len(existing),
            rows_extracted=len(existing),
            checkpoint=str(output_path.relative_to(root)),
        )

    page = context.new_page()
    page.set_default_timeout(20_000)

    try:
        response = page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=90_000,
        )
        if response is not None and response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")

        page.wait_for_timeout(wait_seconds * 1000)

        # Tableau leaves prior tooltips in an overlay. On larger weeks that
        # overlay can sit over a later canvas row and intercept the physical
        # click. Keep it visible for diagnostics but never let it receive
        # pointer events.
        try:
            page.add_style_tag(
                content=(
                    ".tab-tooltip, .tab-tooltipContent, "
                    "[class*='tooltip' i] { "
                    "pointer-events: none !important; }"
                )
            )
        except Exception:
            pass

        view = page.locator(".tab-tvView").first
        canvas = view.locator("canvas.tabCanvas").first
        canvas_box = canvas.bounding_box()
        zone = canvas.locator(
            "xpath=ancestor::div[contains(@class, 'tabZone-viz')][1]"
        ).first
        if zone.count() == 0:
            zone = page.locator("body")

        if not canvas_box:
            body = page.locator("body").inner_text(timeout=5_000)
            if re.search(r"No data|No records|No matching", body, re.I):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(output_path, index=False)
                return PageRecord(
                    year=year,
                    week=week,
                    picker=picker,
                    status="no_data",
                    url=url,
                    checkpoint=str(output_path.relative_to(root)),
                )
            raise RuntimeError("Could not locate the L# mark canvas.")

        rows = collect_header_rows(zone, canvas_box)
        if not rows:
            # A legitimately empty week can produce a nearly blank view.
            body = page.locator("body").inner_text(timeout=5_000)
            if "Picker" in body or "Games" in body:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(output_path, index=False)
                return PageRecord(
                    year=year,
                    week=week,
                    picker=picker,
                    status="no_data",
                    url=url,
                    checkpoint=str(output_path.relative_to(root)),
                )
            raise RuntimeError("No data rows were detected.")

        extracted: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=1):
            anchor_id = row.get("anchor_id")
            if anchor_id:
                try:
                    page.locator(f"#{anchor_id}").scroll_into_view_if_needed()
                    page.wait_for_timeout(100)
                except Exception:
                    pass

            # Recompute the relevant row center and canvas position after any
            # page/internal scrolling.
            current_y = row["center_y"]
            if anchor_id:
                try:
                    anchor_box = page.locator(f"#{anchor_id}").bounding_box()
                    if anchor_box:
                        current_y = (
                            anchor_box["y"] + anchor_box["height"] / 2
                        )
                except Exception:
                    pass

            current_canvas_box = canvas.bounding_box()
            if not current_canvas_box:
                raise RuntimeError(
                    f"Canvas disappeared before row {row_number}."
                )

            y = current_y
            tooltip = ""
            parsed = None
            attempt_notes: list[str] = []

            # Read the fresh render-tooltip-server response generated by this
            # click instead of reading Tableau's persistent tooltip DOM. This
            # makes it impossible to accept the previous row accidentally.
            x_fractions = (0.50, 0.30, 0.70, 0.15, 0.85)
            y_offsets = (0.0, -2.0, 2.0, -5.0, 5.0)
            for x_fraction in x_fractions:
                if parsed is not None:
                    break
                for y_offset in y_offsets:
                    try:
                        page.keyboard.press("Escape")
                    except Exception:
                        pass

                    # Move outside the viz first so that a hover event cannot
                    # be confused with the click we are about to capture.
                    page.mouse.move(
                        max(1.0, current_canvas_box["x"] - 12.0),
                        max(1.0, current_canvas_box["y"] - 8.0),
                    )
                    page.wait_for_timeout(100)

                    x = (
                        current_canvas_box["x"]
                        + current_canvas_box["width"] * x_fraction
                    )
                    try:
                        candidate = click_and_read_tooltip_response(
                            page,
                            x=x,
                            y=y + y_offset,
                        )
                    except Exception as exc:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            f"{type(exc).__name__}"
                        )
                        continue

                    if not candidate:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            "empty tooltip response"
                        )
                        continue

                    try:
                        candidate_parsed = parse_tooltip(candidate)
                    except Exception as exc:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            f"parse {type(exc).__name__}"
                        )
                        continue

                    if not tooltip_matches_header(
                        candidate_parsed,
                        row["header_values"],
                    ):
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            f"got {candidate_parsed['away']} at "
                            f"{candidate_parsed['home']}"
                        )
                        continue

                    tooltip = candidate
                    parsed = candidate_parsed
                    break

            if parsed is None:
                detail = "; ".join(attempt_notes[-10:])
                raise RuntimeError(
                    f"No matching tooltip response for row {row_number}: "
                    f"{row['header_values']}. Attempts: {detail}"
                )
            market_home_margin = favorite_to_home_margin(
                parsed["line_team"],
                parsed["line_value"],
                parsed["away"],
                parsed["home"],
            )
            prediction_home_margin = favorite_to_home_margin(
                parsed["pick_team"],
                parsed["pick_value"],
                parsed["away"],
                parsed["home"],
            )
            result_home_margin = favorite_to_home_margin(
                parsed["result_team"],
                parsed["result_value"],
                parsed["away"],
                parsed["home"],
            )
            actual_home_margin = parsed["home_pts"] - parsed["away_pts"]

            if (
                result_home_margin is not None
                and abs(result_home_margin - actual_home_margin) > 0.01
            ):
                raise ValueError(
                    "Result tooltip disagreed with scores for "
                    f"{parsed['away']} at {parsed['home']}."
                )

            extracted.append(
                {
                    "year": year,
                    "week": week,
                    "picker": picker,
                    "model_key": model_key(picker),
                    "person": parsed["person"],
                    "twitter": parsed["twitter"],
                    "away": parsed["away"],
                    "home": parsed["home"],
                    "away_pts": parsed["away_pts"],
                    "home_pts": parsed["home_pts"],
                    "actual_home_margin": actual_home_margin,
                    "market_home_margin_close": market_home_margin,
                    "prediction_home_margin": prediction_home_margin,
                    "prediction_error_home": (
                        prediction_home_margin - actual_home_margin
                    ),
                    "absolute_error": abs(
                        prediction_home_margin - actual_home_margin
                    ),
                    "source_url": url,
                    "scraped_at_utc": iso_utc(),
                    "row_number": row_number,
                    "header_values": " | ".join(row["header_values"]),
                    "tooltip_text": tooltip.replace("\n", "\\n"),
                }
            )
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass

        frame = pd.DataFrame(extracted)
        duplicate_games = frame.duplicated(
            ["year", "week", "picker", "away", "home"],
            keep=False,
        )
        if duplicate_games.any():
            examples = frame.loc[
                duplicate_games,
                ["away", "home"],
            ].to_dict(orient="records")
            raise ValueError(f"Duplicate games detected: {examples[:5]}")

        if len(frame) != len(rows):
            raise ValueError(
                f"Detected {len(rows)} rows but extracted {len(frame)}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

        return PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="ok",
            url=url,
            rows_detected=len(rows),
            rows_extracted=len(frame),
            checkpoint=str(output_path.relative_to(root)),
        )

    except Exception as exc:
        if screenshot_errors:
            error_dir = (
                root
                / "data/raw/cfbpicker/errors"
                / f"year={year}"
                / f"picker={safe_key(picker)}"
            )
            error_dir.mkdir(parents=True, exist_ok=True)
            try:
                page.screenshot(
                    path=str(error_dir / f"week={week:02d}.png"),
                    full_page=True,
                )
                (error_dir / f"week={week:02d}.html").write_text(
                    page.content(),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="error",
            url=url,
            message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        page.close()


def read_checkpoints(
    root: Path,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    base = root / "data/raw/cfbpicker/tooltips"
    if years is None:
        year_dirs = sorted(
            path
            for path in base.glob("year=*")
            if re.fullmatch(r"year=\d{4}", path.name)
        )
    else:
        year_dirs = [base / f"year={int(year)}" for year in years]

    paths: list[Path] = []
    for year_dir in year_dirs:
        paths.extend(sorted(year_dir.glob("picker=*/week=*.csv")))

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(
            ["year", "week", "picker", "away", "home"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def rebuild_outputs(
    root: Path,
    years: Iterable[int] | None = None,
) -> dict[str, Any]:
    # Global outputs always include every collected season unless a caller
    # deliberately supplies a year subset. This prevents one season run from
    # overwriting the previously collected history.
    long_data = read_checkpoints(root, years=years)
    output_dir = root / "data/cfbpicker"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "cfbpicker_history_long.csv"
    wide_path = output_dir / "cfbpicker_history_wide.csv"
    diagnostics_path = output_dir / "cfbpicker_model_diagnostics.csv"
    model_map_path = output_dir / "cfbpicker_model_map.csv"
    coverage_path = output_dir / "cfbpicker_model_season_coverage.csv"

    if long_data.empty:
        pd.DataFrame().to_csv(long_path, index=False)
        pd.DataFrame().to_csv(wide_path, index=False)
        pd.DataFrame().to_csv(diagnostics_path, index=False)
        pd.DataFrame().to_csv(model_map_path, index=False)
        pd.DataFrame().to_csv(coverage_path, index=False)
        return {"rows": 0, "games": 0, "models": 0, "seasons": 0}

    long_data["year"] = pd.to_numeric(
        long_data["year"], errors="raise"
    ).astype(int)
    long_data["week"] = pd.to_numeric(
        long_data["week"], errors="raise"
    ).astype(int)
    long_data["home_key"] = long_data["home"].map(team_key)
    long_data["road_key"] = long_data["away"].map(team_key)
    long_data["game_key"] = (
        long_data["year"].astype(str)
        + "_w"
        + long_data["week"].astype(str)
        + "_"
        + long_data["road_key"]
        + "_"
        + long_data["home_key"]
    )
    long_data = long_data.sort_values(
        ["year", "week", "away", "home", "picker"],
        kind="stable",
    ).reset_index(drop=True)
    long_data.to_csv(long_path, index=False)

    game_fields = (
        long_data.drop_duplicates("game_key")
        .loc[
            :,
            [
                "game_key",
                "year",
                "week",
                "away",
                "home",
                "away_pts",
                "home_pts",
                "actual_home_margin",
                "market_home_margin_close",
            ],
        ]
    )
    predictions = long_data.pivot(
        index="game_key",
        columns="model_key",
        values="prediction_home_margin",
    ).reset_index()
    predictions.columns.name = None
    wide = game_fields.merge(predictions, on="game_key", how="left")
    wide.to_csv(wide_path, index=False)

    rows: list[dict[str, Any]] = []
    total_games = long_data["game_key"].nunique()
    for (picker, key), group in long_data.groupby(
        ["picker", "model_key"],
        dropna=False,
    ):
        edge = (
            group["prediction_home_margin"]
            - group["market_home_margin_close"]
        )
        ats_margin = np.sign(edge) * (
            group["actual_home_margin"]
            - group["market_home_margin_close"]
        )
        wins = int((ats_margin > 0).sum())
        losses = int((ats_margin < 0).sum())
        pushes = int((ats_margin == 0).sum())
        decisions = wins + losses

        rows.append(
            {
                "picker": picker,
                "model_key": key,
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "seasons": int(group["year"].nunique()),
                "predictions": len(group),
                "unique_games": group["game_key"].nunique(),
                "coverage": (
                    group["game_key"].nunique() / total_games
                    if total_games else np.nan
                ),
                "bias_home_margin": group["prediction_error_home"].mean(),
                "mae": group["absolute_error"].mean(),
                "rmse": math.sqrt(
                    np.mean(np.square(group["prediction_error_home"]))
                ),
                "ats_wins": wins,
                "ats_losses": losses,
                "ats_pushes": pushes,
                "ats_decisions": decisions,
                "ats_rate": wins / decisions if decisions else np.nan,
                "mean_abs_market_edge": edge.abs().mean(),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values(
        ["mae", "ats_rate"],
        ascending=[True, False],
    )
    diagnostics.to_csv(diagnostics_path, index=False)

    model_map = (
        long_data.loc[
            :,
            ["year", "picker", "model_key", "person", "twitter"],
        ]
        .drop_duplicates()
        .sort_values(["picker", "year"])
    )
    model_map.to_csv(model_map_path, index=False)

    season_coverage = (
        long_data.groupby(
            ["year", "picker", "model_key"],
            dropna=False,
        )
        .agg(
            predictions=("game_key", "size"),
            unique_games=("game_key", "nunique"),
            first_week=("week", "min"),
            last_week=("week", "max"),
        )
        .reset_index()
        .sort_values(["year", "picker"])
    )
    season_coverage.to_csv(coverage_path, index=False)

    return {
        "rows": len(long_data),
        "games": int(long_data["game_key"].nunique()),
        "models": int(long_data["model_key"].nunique()),
        "seasons": int(long_data["year"].nunique()),
        "years": sorted(long_data["year"].unique().tolist()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--weeks", nargs="*", default=["1-22"])
    parser.add_argument("--pickers", nargs="*", default=[])
    parser.add_argument("--picker-file", type=Path)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Collect every genuine forward model, including models also "
            "available through Prediction Tracker. Aggregate/line rows stay excluded."
        ),
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--system-chrome", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument(
        "--no-error-screenshots",
        action="store_true",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    weeks = parse_int_spec(args.weeks)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run: python -m pip install playwright",
            file=sys.stderr,
        )
        return 2

    with sync_playwright() as playwright:
        launch: dict[str, Any] = {
            "headless": not args.headed,
        }
        if args.system_chrome:
            launch["channel"] = "chrome"

        browser = playwright.chromium.launch(**launch)
        context = browser.new_context(
            viewport={"width": 1500, "height": 1200},
            locale="en-US",
        )

        discovered, discovery_url = discover_pickers(
            context=context,
            year=args.year,
            wait_seconds=max(8, args.wait_seconds),
        )
        candidate_path, collectable_path = write_picker_outputs(
            root=root,
            year=args.year,
            pickers=discovered,
        )
        print(
            f"Discovered {len(discovered)} forward picker values. "
            f"Incremental file: {candidate_path}; "
            f"all-model file: {collectable_path}"
        )

        if args.discover_only:
            browser.close()
            return 0

        selected = parse_picker_args(args.pickers)
        if args.picker_file:
            picker_file = args.picker_file
            if not picker_file.is_absolute():
                picker_file = root / picker_file
            selected.extend(
                line.strip()
                for line in picker_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        if args.all_candidates:
            selected.extend(
                picker
                for picker in discovered
                if classify_picker(picker)[0] == "incremental_candidate"
            )
        if args.all_models:
            selected.extend(
                picker
                for picker in discovered
                if classify_picker(picker)[0] != "exclude"
            )
        selected = list(dict.fromkeys(selected))

        if not selected:
            browser.close()
            raise SystemExit(
                "No pickers selected. Use --pickers, --picker-file, "
                "--all-candidates, or --all-models."
            )

        unknown = sorted(set(selected) - set(discovered))
        if unknown:
            print(
                "WARNING: selected picker values not present in discovery: "
                + ", ".join(unknown),
                file=sys.stderr,
            )

        records: list[PageRecord] = []
        pages_run = 0

        for picker in selected:
            for week in weeks:
                if args.max_pages is not None and pages_run >= args.max_pages:
                    break

                print(
                    f"Scraping year={args.year} week={week} "
                    f"picker={picker}"
                )
                record = scrape_page(
                    context=context,
                    root=root,
                    year=args.year,
                    week=week,
                    picker=picker,
                    wait_seconds=args.wait_seconds,
                    force=args.force,
                    screenshot_errors=not args.no_error_screenshots,
                )
                records.append(record)
                pages_run += 1
                print(
                    f"  {record.status}: "
                    f"{record.rows_extracted}/{record.rows_detected}"
                )
                if record.message:
                    print(f"  {record.message}", file=sys.stderr)

                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)

            if args.max_pages is not None and pages_run >= args.max_pages:
                break

        browser.close()

    summary = rebuild_outputs(root=root, years=None)
    failures = [record for record in records if record.status == "error"]

    manifest = {
        "run_at_utc": iso_utc(),
        "year": args.year,
        "weeks": weeks,
        "selected_pickers": selected,
        "discovered_picker_count": len(discovered),
        "discovery_url": discovery_url,
        "pages": [asdict(record) for record in records],
        "combined": summary,
    }
    manifest_path = (
        root
        / f"data/cfbpicker/cfbpicker_tooltip_scrape_status_{args.year}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Status: {manifest_path}")

    if failures and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
