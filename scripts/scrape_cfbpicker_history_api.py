#!/usr/bin/env python3
"""
Scrape CFB Picker seasons using Tableau Embedding API v3 (v3.5.33 transport).

The public workbook ignores historical Year URL parameters. This collector
loads the workbook normally, then applies the visible categorical quick filter
whose exact caption is ``Year `` (with a trailing space). Week and Picker are
applied in the same live workbook session before the proven tooltip extractor
runs.

This is a collection-only script. It imports normalization, tooltip parsing,
checkpoint rebuilding, and model classification from the existing
``scripts/scrape_cfbpicker_history.py``. No consensus code is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import importlib.util
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd

TABLEAU_BASE = "https://public.tableau.com/views/CFBPicker"
TABLEAU_API = (
    "https://public.tableau.com/javascripts/api/"
    "tableau.embedding.3.latest.min.js"
)


def load_history_module(root: Path) -> Any:
    # v3.5.33 vendors the proven tooltip helpers under a stable name so the
    # API collector no longer depends on whichever importer owns
    # scrape_cfbpicker_history.py in the modern pipeline.
    path = Path(__file__).with_name("cfbpicker_tooltip_legacy.py")
    if not path.exists():
        raise FileNotFoundError(f"Missing CFB Picker tooltip helper: {path}")
    spec = importlib.util.spec_from_file_location(
        "cfbpicker_history_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextlib.contextmanager
def local_server(directory: Path):
    port = free_port()

    def handler(*args: Any, **kwargs: Any) -> QuietHandler:
        return QuietHandler(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def embedding_html() -> str:
    template = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CFB Picker API collector</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: auto; }
    #host, tableau-viz { width: 1700px; height: 1250px; }
  </style>
</head>
<body>
<div id="host"></div>
<script type="module">
  import {
    TableauViz,
    TableauEventType,
    FilterUpdateType
  } from "__TABLEAU_API__";

  const query = new URLSearchParams(window.location.search);
  const viewName = query.get("view") || "Standings";
  const requestedYear = Number(query.get("year"));
  const requestedWeek = query.get("week");
  const requestedPicker = query.get("picker");

  window.__cfbp = {
    status: "initializing",
    viewName,
    requestedYear,
    requestedWeek,
    requestedPicker,
    attempts: [],
    errors: []
  };

  function serialize(value) {
    if (value === null || value === undefined) return value;
    if (["string", "number", "boolean"].includes(typeof value)) return value;
    if (Array.isArray(value)) return value.map(serialize);
    const output = {};
    try {
      for (const key of Object.keys(value)) {
        try { output[key] = serialize(value[key]); }
        catch (error) { output[key] = `[unreadable: ${error}]`; }
      }
      return output;
    } catch (error) {
      return String(value);
    }
  }

  async function filterSummary(sheet) {
    try {
      const filters = await sheet.getFiltersAsync();
      return filters.map((filter) => ({
        fieldName: filter.fieldName,
        fieldId: filter.fieldId,
        filterType: filter.filterType,
        worksheetName: filter.worksheetName,
        appliedValues: serialize(filter.appliedValues),
        minValue: serialize(filter.minValue),
        maxValue: serialize(filter.maxValue),
        isAllSelected: filter.isAllSelected
      }));
    } catch (error) {
      return [{error: String(error)}];
    }
  }

  function targets(activeSheet) {
    const output = [activeSheet];
    if (Array.isArray(activeSheet.worksheets)) {
      output.push(...activeSheet.worksheets);
    }
    return output;
  }

  async function applyCategorical(activeSheet, fieldName, value) {
    const errors = [];
    for (const target of targets(activeSheet)) {
      try {
        const result = await target.applyFilterAsync(
          fieldName,
          [String(value)],
          FilterUpdateType.Replace
        );
        window.__cfbp.attempts.push({
          target: target.name,
          fieldName,
          value: String(value),
          result: serialize(result)
        });
        return true;
      } catch (error) {
        errors.push(`${target.name}: ${error}`);
      }
    }
    window.__cfbp.attempts.push({fieldName, value, errors});
    return false;
  }

  function formattedValues(filters, fieldName) {
    const output = [];
    for (const filter of filters) {
      if (filter.fieldName !== fieldName) continue;
      for (const value of (filter.appliedValues || [])) {
        output.push(String(
          value?._formattedValue ?? value?._nativeValue ?? value?._value
        ));
      }
    }
    return output;
  }

  async function collectAllFilters(activeSheet) {
    const output = [];
    for (const target of targets(activeSheet)) {
      const filters = await filterSummary(target);
      for (const filter of filters) {
        output.push({...filter, targetName: target.name});
      }
    }
    return output;
  }

  const source = new URL(`__TABLEAU_BASE__/${viewName}`);
  source.searchParams.set(":showVizHome", "no");
  source.searchParams.set("Perspective", "Forward");
  source.searchParams.set("Show Incomplete", "No");
  source.searchParams.set("Show Line", "Yes");
  source.searchParams.set("Vs Line X", "Close");

  const viz = new TableauViz();
  viz.src = source.toString();
  viz.toolbar = "hidden";
  viz.hideTabs = false;
  viz.width = "1700px";
  viz.height = "1250px";

  viz.addEventListener(TableauEventType.VizLoadError, (event) => {
    window.__cfbp.status = "viz_load_error";
    window.__cfbp.errors.push(String(event?.detail || event));
  });

  viz.addEventListener(TableauEventType.FirstInteractive, async () => {
    try {
      const workbook = viz.workbook;
      const activeSheet = workbook.activeSheet;
      window.__cfbp.workbookName = workbook.name;
      window.__cfbp.activeSheet = {
        name: activeSheet.name,
        sheetType: activeSheet.sheetType,
        worksheetNames: Array.isArray(activeSheet.worksheets)
          ? activeSheet.worksheets.map((item) => item.name)
          : []
      };
      window.__cfbp.status = "interactive";

      // Give Playwright time to clear bootstrap responses before commands fire.
      await new Promise((resolve) => setTimeout(resolve, 1200));

      const yearApplied = await applyCategorical(
        activeSheet, "Year ", requestedYear
      );
      await new Promise((resolve) => setTimeout(resolve, 800));

      let weekApplied = true;
      if (requestedWeek !== null && requestedWeek !== "") {
        weekApplied = await applyCategorical(
          activeSheet, "Week ", requestedWeek
        );
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      let pickerApplied = true;
      if (requestedPicker !== null && requestedPicker !== "") {
        pickerApplied = await applyCategorical(
          activeSheet, "Picker", requestedPicker
        );
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      // This is already present as a URL/default filter in the working 2025
      // collector. Apply it through the API when the sheet exposes it.
      await applyCategorical(activeSheet, "Game Complete", "Yes");
      await new Promise((resolve) => setTimeout(resolve, 2500));

      const filters = await collectAllFilters(activeSheet);
      const yearValues = formattedValues(filters, "Year ");
      const weekValues = formattedValues(filters, "Week ");
      const pickerValues = formattedValues(filters, "Picker");

      const yearVerified = yearValues.includes(String(requestedYear));
      const weekVerified = requestedWeek === null || requestedWeek === "" ||
        weekValues.includes(String(requestedWeek));
      const pickerVerified = requestedPicker === null || requestedPicker === "" ||
        pickerValues.includes(String(requestedPicker));

      window.__cfbp.filtersAfter = filters;
      window.__cfbp.yearValues = yearValues;
      window.__cfbp.weekValues = weekValues;
      window.__cfbp.pickerValues = pickerValues;
      window.__cfbp.yearVerified = yearVerified;
      window.__cfbp.weekVerified = weekVerified;
      window.__cfbp.pickerVerified = pickerVerified;
      window.__cfbp.applied = {yearApplied, weekApplied, pickerApplied};
      window.__cfbp.status = (
        yearApplied && weekApplied && pickerApplied &&
        yearVerified && weekVerified && pickerVerified
      ) ? "ready" : "filter_verification_failed";
    } catch (error) {
      window.__cfbp.status = "error";
      window.__cfbp.errors.push(
        `${error?.name || "Error"}: ${error?.message || error}`
      );
    }
  });

  document.getElementById("host").appendChild(viz);
  window.__tableauViz = viz;
</script>
</body>
</html>
'''
    return (
        template.replace("__TABLEAU_API__", TABLEAU_API)
        .replace("__TABLEAU_BASE__", TABLEAU_BASE)
    )


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
        for piece in str(raw).split(","):
            piece = piece.strip()
            if piece:
                output.append(piece)
    return list(dict.fromkeys(output))


def tableau_frame(page: Any, route: str) -> Any:
    needle = f"/views/CFBPicker/{route}".lower()
    for frame in page.frames:
        if needle in frame.url.lower():
            return frame
    raise RuntimeError(f"Could not locate Tableau frame for {route}.")


def open_filtered_view(
    context: Any,
    server_url: str,
    route: str,
    year: int,
    week: int | None,
    picker: str | None,
    wait_seconds: int,
) -> tuple[Any, Any, list[Any], dict[str, Any], str]:
    query: dict[str, str] = {
        "view": route,
        "year": str(year),
    }
    if week is not None:
        query["week"] = str(week)
    if picker is not None:
        query["picker"] = picker
    local_url = f"{server_url}?{urlencode(query)}"

    page = context.new_page()
    page.set_default_timeout(20_000)
    post_interactive_responses: list[Any] = []
    capture = {"enabled": False}

    def on_response(response: Any) -> None:
        if not capture["enabled"]:
            return
        lower = response.url.lower()
        if (
            "startsession/viewing" in lower
            or "bootstrapsession" in lower
            or "/commands/" in lower
        ):
            post_interactive_responses.append(response)

    page.on("response", on_response)
    page.goto(local_url, wait_until="domcontentloaded", timeout=90_000)

    deadline = time.monotonic() + 120
    interactive_seen = False
    state: dict[str, Any] = {}
    terminal = {
        "ready",
        "filter_verification_failed",
        "viz_load_error",
        "error",
    }
    while time.monotonic() < deadline:
        state = page.evaluate("() => window.__cfbp || {}")
        status = state.get("status")
        if status == "interactive" and not interactive_seen:
            post_interactive_responses.clear()
            capture["enabled"] = True
            interactive_seen = True
        if status in terminal:
            break
        page.wait_for_timeout(250)

    if state.get("status") != "ready":
        raise RuntimeError(
            "Tableau API filters were not verified: "
            + json.dumps(state, default=str)[:4000]
        )

    page.wait_for_timeout(max(1, wait_seconds) * 1000)
    frame = tableau_frame(page, route)
    return page, frame, post_interactive_responses, state, local_url


def discover_pickers_api(
    context: Any,
    server_url: str,
    history: Any,
    year: int,
    wait_seconds: int,
) -> tuple[list[str], str, dict[str, Any]]:
    page = None
    try:
        page, frame, responses, state, local_url = open_filtered_view(
            context=context,
            server_url=server_url,
            route="Standings",
            year=year,
            week=None,
            picker=None,
            wait_seconds=wait_seconds,
        )
        objects = history.parse_bootstrap_objects(responses)
        pickers = history.picker_items_from_objects(objects)

        # A filter command may be followed by several presentation responses.
        # Retain the last occurrence order and remove pseudo UI text.
        pickers = [
            value.strip()
            for value in pickers
            if value and value.strip() not in {"(All)", "All"}
        ]
        pickers = list(dict.fromkeys(pickers))

        if not pickers:
            error_dir = Path("data/cfbpicker")
            body = frame.locator("body").inner_text(timeout=10_000)
            raise RuntimeError(
                "Picker domain was not found after applying historical year. "
                f"Frame text preview: {body[:1000]}"
            )
        return pickers, local_url, state
    finally:
        if page is not None:
            page.close()


def scrape_page_api(
    context: Any,
    server_url: str,
    root: Path,
    history: Any,
    year: int,
    week: int,
    picker: str,
    wait_seconds: int,
    force: bool,
    screenshot_errors: bool,
) -> Any:
    output_path = history.checkpoint_path(root, year, picker, week)
    source_url = f"{TABLEAU_BASE}/L"

    if output_path.exists() and not force:
        try:
            existing = pd.read_csv(output_path)
            count = len(existing)
        except pd.errors.EmptyDataError:
            count = 0
        return history.PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="skipped_existing",
            url=source_url,
            rows_detected=count,
            rows_extracted=count,
            checkpoint=str(output_path.relative_to(root)),
        )

    page = None
    try:
        page, frame, _, state, local_url = open_filtered_view(
            context=context,
            server_url=server_url,
            route="L",
            year=year,
            week=week,
            picker=picker,
            wait_seconds=wait_seconds,
        )

        try:
            frame.add_style_tag(
                content=(
                    ".tab-tooltip, .tab-tooltipContent, "
                    "[class*='tooltip' i] { "
                    "pointer-events: none !important; }"
                )
            )
        except Exception:
            pass

        view = frame.locator(".tab-tvView").first
        canvas = view.locator("canvas.tabCanvas").first
        canvas_box = canvas.bounding_box()
        zone = canvas.locator(
            "xpath=ancestor::div[contains(@class, 'tabZone-viz')][1]"
        ).first
        if zone.count() == 0:
            zone = frame.locator("body")

        if not canvas_box:
            body = frame.locator("body").inner_text(timeout=5_000)
            if re.search(r"No data|No records|No matching", body, re.I):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(output_path, index=False)
                return history.PageRecord(
                    year=year,
                    week=week,
                    picker=picker,
                    status="no_data",
                    url=source_url,
                    checkpoint=str(output_path.relative_to(root)),
                )
            raise RuntimeError("Could not locate the API-filtered L# canvas.")

        rows = history.collect_header_rows(zone, canvas_box)
        if not rows:
            body = frame.locator("body").inner_text(timeout=5_000)
            if "Picker" in body or "Games" in body:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(output_path, index=False)
                return history.PageRecord(
                    year=year,
                    week=week,
                    picker=picker,
                    status="no_data",
                    url=source_url,
                    checkpoint=str(output_path.relative_to(root)),
                )
            raise RuntimeError("No API-filtered data rows were detected.")

        extracted: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=1):
            anchor_id = row.get("anchor_id")
            if anchor_id:
                try:
                    frame.locator(f"#{anchor_id}").scroll_into_view_if_needed()
                    page.wait_for_timeout(100)
                except Exception:
                    pass

            current_y = row["center_y"]
            if anchor_id:
                try:
                    anchor_box = frame.locator(f"#{anchor_id}").bounding_box()
                    if anchor_box:
                        current_y = anchor_box["y"] + anchor_box["height"] / 2
                except Exception:
                    pass

            current_canvas_box = canvas.bounding_box()
            if not current_canvas_box:
                raise RuntimeError(f"Canvas disappeared before row {row_number}.")

            parsed = None
            tooltip = ""
            attempt_notes: list[str] = []
            for x_fraction in (0.50, 0.30, 0.70, 0.15, 0.85):
                if parsed is not None:
                    break
                for y_offset in (0.0, -2.0, 2.0, -5.0, 5.0):
                    try:
                        page.keyboard.press("Escape")
                    except Exception:
                        pass
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
                        candidate = history.click_and_read_tooltip_response(
                            page, x=x, y=current_y + y_offset
                        )
                    except Exception as exc:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            f"{type(exc).__name__}"
                        )
                        continue
                    if not candidate:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: empty"
                        )
                        continue
                    try:
                        candidate_parsed = history.parse_tooltip(candidate)
                    except Exception as exc:
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: "
                            f"parse {type(exc).__name__}"
                        )
                        continue
                    if not history.tooltip_matches_header(
                        candidate_parsed, row["header_values"]
                    ):
                        attempt_notes.append(
                            f"x={x_fraction:.2f},dy={y_offset:+.1f}: got "
                            f"{candidate_parsed['away']} at "
                            f"{candidate_parsed['home']}"
                        )
                        continue
                    tooltip = candidate
                    parsed = candidate_parsed
                    break

            if parsed is None:
                raise RuntimeError(
                    f"No matching tooltip response for row {row_number}: "
                    f"{row['header_values']}. Attempts: "
                    + "; ".join(attempt_notes[-10:])
                )

            market_home_margin = history.favorite_to_home_margin(
                parsed["line_team"],
                parsed["line_value"],
                parsed["away"],
                parsed["home"],
            )
            prediction_home_margin = history.favorite_to_home_margin(
                parsed["pick_team"],
                parsed["pick_value"],
                parsed["away"],
                parsed["home"],
            )
            result_home_margin = history.favorite_to_home_margin(
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
                    "model_key": history.model_key(picker),
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
                    "source_url": source_url,
                    "filter_method": "tableau_embedding_api",
                    "scraped_at_utc": history.iso_utc(),
                    "row_number": row_number,
                    "header_values": " | ".join(row["header_values"]),
                    "tooltip_text": tooltip.replace("\n", "\\n"),
                }
            )
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass

        data = pd.DataFrame(extracted)
        duplicate_games = data.duplicated(
            ["year", "week", "picker", "away", "home"], keep=False
        )
        if duplicate_games.any():
            examples = data.loc[
                duplicate_games, ["away", "home"]
            ].to_dict(orient="records")
            raise ValueError(f"Duplicate games detected: {examples[:5]}")
        if len(data) != len(rows):
            raise ValueError(
                f"Detected {len(rows)} rows but extracted {len(data)}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        return history.PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="ok",
            url=source_url,
            rows_detected=len(rows),
            rows_extracted=len(data),
            checkpoint=str(output_path.relative_to(root)),
        )

    except Exception as exc:
        if screenshot_errors and page is not None:
            error_dir = (
                root
                / "data/raw/cfbpicker/errors_api"
                / f"year={year}"
                / f"picker={history.safe_key(picker)}"
            )
            error_dir.mkdir(parents=True, exist_ok=True)
            try:
                page.screenshot(
                    path=str(error_dir / f"week={week:02d}.png"),
                    full_page=True,
                )
                (error_dir / f"week={week:02d}_state.json").write_text(
                    json.dumps(
                        page.evaluate("() => window.__cfbp || {}"),
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return history.PageRecord(
            year=year,
            week=week,
            picker=picker,
            status="error",
            url=source_url,
            message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if page is not None:
            page.close()


def run_one_year(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    history = load_history_module(root)
    weeks = parse_int_spec(args.weeks)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="cfbpicker_api_collector_") as tmp:
        temp_dir = Path(tmp)
        (temp_dir / "index.html").write_text(
            embedding_html(), encoding="utf-8"
        )

        with local_server(temp_dir) as server_url:
            with sync_playwright() as playwright:
                launch: dict[str, Any] = {"headless": not args.headed}
                if args.system_chrome:
                    launch["channel"] = "chrome"
                browser = playwright.chromium.launch(**launch)
                context = browser.new_context(
                    viewport={"width": 1750, "height": 1300},
                    locale="en-US",
                )

                print(f"Discovering API-filtered pickers for {args.year}")
                discovered, discovery_url, discovery_state = discover_pickers_api(
                    context=context,
                    server_url=server_url,
                    history=history,
                    year=args.year,
                    wait_seconds=max(4, args.wait_seconds),
                )
                candidate_path, collectable_path = history.write_picker_outputs(
                    root=root, year=args.year, pickers=discovered
                )
                print(
                    f"Discovered {len(discovered)} picker values. "
                    f"Incremental: {candidate_path}; all models: {collectable_path}"
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
                        if history.classify_picker(picker)[0]
                        == "incremental_candidate"
                    )
                if args.all_models:
                    selected.extend(
                        picker
                        for picker in discovered
                        if history.classify_picker(picker)[0] != "exclude"
                    )
                selected = list(dict.fromkeys(selected))
                if not selected:
                    browser.close()
                    raise SystemExit(
                        "No pickers selected. Use --pickers, --picker-file, "
                        "--all-candidates, or --all-models."
                    )

                records: list[Any] = []
                pages_run = 0
                for picker in selected:
                    for week in weeks:
                        if (
                            args.max_pages is not None
                            and pages_run >= args.max_pages
                        ):
                            break
                        print(
                            f"Scraping API year={args.year} week={week} "
                            f"picker={picker}"
                        )
                        record = scrape_page_api(
                            context=context,
                            server_url=server_url,
                            root=root,
                            history=history,
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
                        if args.delay_seconds:
                            time.sleep(args.delay_seconds)
                    if (
                        args.max_pages is not None
                        and pages_run >= args.max_pages
                    ):
                        break
                browser.close()

    summary = history.rebuild_outputs(root=root, years=None)
    failures = [record for record in records if record.status == "error"]
    manifest = {
        "run_at_utc": history.iso_utc(),
        "filter_method": "tableau_embedding_api",
        "year": args.year,
        "weeks": weeks,
        "selected_pickers": selected,
        "discovered_picker_count": len(discovered),
        "discovery_url": discovery_url,
        "discovery_state": discovery_state,
        "pages": [asdict(record) for record in records],
        "combined": summary,
    }
    manifest_path = (
        root
        / f"data/cfbpicker/cfbpicker_api_scrape_status_{args.year}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Status: {manifest_path}")
    return 2 if failures and args.strict else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--weeks", nargs="*", default=["1-22"])
    parser.add_argument("--pickers", nargs="*", default=[])
    parser.add_argument("--picker-file", type=Path)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--system-chrome", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--no-error-screenshots", action="store_true")
    args = parser.parse_args()
    return run_one_year(args)


if __name__ == "__main__":
    raise SystemExit(main())
