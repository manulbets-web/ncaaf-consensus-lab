#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh cached kickoff dates for the current PredictionTracker CFB slate.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when no fresh schedule could be fetched and no cache exists.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root / "strategy_lab"))
    from current_week import refresh_current_game_schedule  # noqa: E402

    status = refresh_current_game_schedule(root, season=int(args.season))
    print(json.dumps(status, indent=2))
    if args.strict and status.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
