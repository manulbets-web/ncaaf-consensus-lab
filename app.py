"""Posit Connect Cloud entrypoint for NCAAF Consensus Lab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "strategy_lab"
sys.path.insert(0, str(LAB))
os.environ.setdefault("NCAAF_CLOUD_MODE", "1")

from strategy_lab.app import app  # noqa: E402,F401
