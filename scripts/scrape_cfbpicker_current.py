#!/usr/bin/env python3
"""Refresh the current CFB Picker model predictions into a canonical long cache.

Direct Tableau Public extraction is attempted first. A season/week-tagged GitHub
mirror is available as a Connect Cloud fallback, paralleling PredictionTracker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cfbpicker_tableau import BROWSER_HEADERS, collect_cfbpicker_tables, write_discovery_diagnostics
from scrape_cfbpicker_history import map_picker_models, match_games

GITHUB_BASE = "https://raw.githubusercontent.com/manulbets-web/ncaaf-consensus-lab/main"
MIRROR_CSV_URL = f"{GITHUB_BASE}/data/current/cfbpicker_current_long.csv"
MIRROR_META_URL = f"{GITHUB_BASE}/data/current/cfbpicker_mirror_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _orientation_sign_map(root: Path) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    path = root / "data" / "derived" / "cfbpicker_orientation_audit.csv"
    exact: dict[tuple[str, str], float] = {}
    picker: dict[str, float] = {}
    if not path.exists():
        return exact, picker
    x = pd.read_csv(path, low_memory=False)
    if x.empty or "picker" not in x or "sign" not in x:
        return exact, picker
    x["sign"] = pd.to_numeric(x["sign"], errors="coerce")
    x = x.dropna(subset=["sign"])
    for r in x.itertuples(index=False):
        exact[(str(getattr(r, "source_table", "")), str(r.picker))] = float(r.sign)
    for p, g in x.groupby("picker"):
        vals = pd.to_numeric(g["sign"], errors="coerce").dropna()
        if len(vals):
            picker[str(p)] = float(1.0 if vals.mean() >= 0 else -1.0)
    return exact, picker


def orient_current(parsed: pd.DataFrame, root: Path) -> pd.DataFrame:
    exact, by_picker = _orientation_sign_map(root)
    z = parsed.copy()
    signs = []
    methods = []
    for r in z.itertuples(index=False):
        if str(getattr(r, "orientation_certainty", "")) == "home_margin":
            signs.append(1.0); methods.append("explicit_home_margin"); continue
        key = (str(getattr(r, "source_table", "")), str(r.picker))
        if key in exact:
            signs.append(exact[key]); methods.append("historical_source_picker_audit"); continue
        if str(r.picker) in by_picker:
            signs.append(by_picker[str(r.picker)]); methods.append("historical_picker_audit"); continue
        signs.append(np.nan); methods.append("unverified")
    z["prediction_sign"] = signs
    z["orientation_method"] = methods
    game_match_sign = pd.to_numeric(
        z.get("game_match_sign", pd.Series(1.0, index=z.index)), errors="coerce"
    ).fillna(1.0)
    z["prediction_home_margin"] = (
        pd.to_numeric(z["prediction_raw"], errors="coerce")
        * game_match_sign
        * z["prediction_sign"]
    )
    return z


def load_canonical(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    derived = root / "data" / "derived"
    pred = pd.read_csv(derived / "model_game_predictions.csv", low_memory=False)
    reg_path = derived / "model_registry.csv"
    reg = pd.read_csv(reg_path, low_memory=False) if reg_path.exists() else pd.DataFrame()
    return pred, reg


def write_current(root: Path, frame: pd.DataFrame, *, season: int, week: int, transport: str) -> dict:
    cur = root / "data" / "current"
    cur.mkdir(parents=True, exist_ok=True)
    out = cur / "cfbpicker_current_long.csv"
    cols = [
        "season", "week", "away", "home", "market_home_margin_close",
        "canonical_model_id", "model_name", "prediction_home_margin", "picker",
        "source_table", "orientation_method",
    ]
    for c in cols:
        if c not in frame:
            frame[c] = np.nan
    frame[cols].to_csv(out, index=False)
    raw = out.read_bytes()
    meta = {
        "created_at_utc": utc_now(), "season": int(season), "week": int(week),
        "rows": int(len(frame)), "games": int(frame[["away", "home"]].drop_duplicates().shape[0]) if len(frame) else 0,
        "models": int(frame["canonical_model_id"].nunique()) if len(frame) else 0,
        "canonical_sha256": sha256_bytes(raw), "transport": transport,
    }
    (cur / "cfbpicker_mirror_status.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    snap = root / "data" / "snapshots" / "cfbpicker" / "current"
    snap.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (snap / f"season_{season}_week_{week:02d}_{stamp}_{meta['canonical_sha256'][:12]}.csv").write_bytes(raw)
    return meta


def github_fallback(root: Path, *, season: int, week: int) -> dict:
    with requests.Session() as s:
        s.headers.update(BROWSER_HEADERS)
        meta_r = s.get(MIRROR_META_URL, timeout=30); meta_r.raise_for_status()
        csv_r = s.get(MIRROR_CSV_URL, timeout=30); csv_r.raise_for_status()
    meta = meta_r.json()
    if int(meta.get("season", -1)) != int(season) or int(meta.get("week", -1)) != int(week):
        raise RuntimeError(f"CFB Picker GitHub mirror is season/week {meta.get('season')}/{meta.get('week')}, requested {season}/{week}.")
    actual = sha256_bytes(csv_r.content)
    expected = str(meta.get("canonical_sha256") or "")
    if expected and actual != expected:
        raise RuntimeError("CFB Picker GitHub mirror SHA-256 mismatch.")
    cur = root / "data" / "current"
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "cfbpicker_current_long.csv").write_bytes(csv_r.content)
    (cur / "cfbpicker_mirror_status.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "transport": "github_mirror"}


def direct_refresh(
    root: Path,
    *,
    season: int,
    week: int,
    mapping_file: Path | None = None,
    headed: bool = False,
    system_chrome: bool = False,
) -> dict:
    """Use the proven Embedding API + L# tooltip extractor for fresh current rows."""
    api_script = Path(__file__).with_name("scrape_cfbpicker_current_api.py")
    if not api_script.exists():
        raise FileNotFoundError(f"Missing current API collector: {api_script}")
    cmd = [
        sys.executable, str(api_script), "--root", str(root),
        "--season", str(int(season)), "--week", str(int(week)), "--strict",
    ]
    if mapping_file is not None:
        cmd += ["--mapping-file", str(mapping_file)]
    if headed:
        cmd.append("--headed")
    if system_chrome:
        cmd.append("--system-chrome")
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            "Embedding API current collector failed. "
            + (proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}")[-5000:]
        )
    out = root / "data" / "current" / "cfbpicker_current_long.csv"
    if not out.exists():
        raise RuntimeError("Embedding API collector returned success but did not write cfbpicker_current_long.csv.")
    frame = pd.read_csv(out, low_memory=False)
    if frame.empty:
        raise RuntimeError("Embedding API current collector returned zero rows.")
    meta = write_current(root, frame, season=season, week=week, transport="tableau_embedding_api_tooltip")
    meta["collector_stdout_tail"] = proc.stdout[-2500:]
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--mapping-file", type=Path, default=None)  # accepted for current_week.py compatibility
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force-download", action="store_true")  # compatibility; API is always fresh
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--system-chrome", action="store_true")
    ap.add_argument("--no-github-fallback", action="store_true")
    ap.add_argument(
        "--from-cache", action="store_true",
        help="Validate and publish the existing API cache without rerunning Tableau.",
    )
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if args.from_cache:
        cache = root / "data/current/cfbpicker_current_long.csv"
        try:
            if not cache.exists():
                raise FileNotFoundError(f"Existing CFB Picker cache not found: {cache}")
            frame = pd.read_csv(cache, low_memory=False)
            if frame.empty:
                raise RuntimeError("Existing CFB Picker cache contains zero rows.")
            required = {"season", "week", "canonical_model_id", "prediction_home_margin"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise RuntimeError("Existing CFB Picker cache is missing: " + ", ".join(missing))
            seasons = set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int))
            weeks = set(pd.to_numeric(frame["week"], errors="coerce").dropna().astype(int))
            if seasons != {int(args.season)} or weeks != {int(args.week)}:
                raise RuntimeError(
                    f"Existing cache is tagged season/week {sorted(seasons)}/{sorted(weeks)}, "
                    f"requested {int(args.season)}/{int(args.week)}."
                )
            meta = write_current(
                root, frame, season=args.season, week=args.week,
                transport="tableau_embedding_api_tooltip_existing_cache",
            )
            print(json.dumps({"status": "ok", "from_cache": True, **meta}, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({
                "status": "error", "from_cache": True,
                "error": f"{type(exc).__name__}: {exc}",
            }, indent=2))
            return 2 if args.strict else 0
    # Current-week refresh is freshness-sensitive. --quick/--force-download are
    # retained for compatibility but the Embedding API always opens a fresh session.
    try:
        meta = direct_refresh(
            root, season=args.season, week=args.week, mapping_file=args.mapping_file,
            headed=args.headed, system_chrome=args.system_chrome,
        )
        print(json.dumps({"status": "ok", **meta}, indent=2))
        return 0
    except Exception as direct_error:
        if args.no_github_fallback:
            print(json.dumps({"status": "error", "direct_error": f"{type(direct_error).__name__}: {direct_error}"}, indent=2))
            return 2
        try:
            meta = github_fallback(root, season=args.season, week=args.week)
            print(json.dumps({"status": "ok", "direct_error": f"{type(direct_error).__name__}: {direct_error}", **meta}, indent=2))
            return 0
        except Exception as mirror_error:
            print(json.dumps({
                "status": "error",
                "direct_error": f"{type(direct_error).__name__}: {direct_error}",
                "mirror_error": f"{type(mirror_error).__name__}: {mirror_error}",
            }, indent=2))
            return 2 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
