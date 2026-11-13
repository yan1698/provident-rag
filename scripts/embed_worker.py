"""Run the Day 2 embedding worker from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.embed.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
