"""Offline stage arithmetic. No model, no torch, no database."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_backend = ROOT / "backend"
if _backend.is_dir():
    sys.path.insert(0, str(_backend))
elif (ROOT / "app").is_dir():
    sys.path.insert(0, str(ROOT))

from app.search.stages import (  # noqa: E402
    add_stage,
    leaf_stages,
    leaf_total_ms,
    note_meta,
    reset_stages,
    snapshot_meta,
    snapshot_stages,
)


def main() -> None:
    reset_stages()
    add_stage("bm25_ms", time.perf_counter() - 0.02)
    add_stage("bm25_ms", time.perf_counter() - 0.02)
    summed = snapshot_stages()["bm25_ms"]
    if summed < 30:
        raise SystemExit(f"{_TEST_MATERIALS['text_001']}{summed}")

    reset_stages()
    note_meta("device", "x" * 200)
    stored = snapshot_meta()["device"]
    if not isinstance(stored, str) or len(stored) != 80:
        raise SystemExit(f"{_TEST_MATERIALS['text_002']}{stored!r}")

    stages = {"vector_ms": 100, "embed_ms": 40, "ann_ms": 60, "bm25_ms": 10}
    if sum(stages.values()) != 210:
        raise SystemExit(_TEST_MATERIALS["text_005"])
    leaves = leaf_stages(stages)
    if "embed_ms" in leaves or "ann_ms" in leaves:
        raise SystemExit(f"{_TEST_MATERIALS['text_003']}{leaves}")
    if leaf_total_ms(stages) != 110:
        raise SystemExit(f"FAIL: leaf total {leaf_total_ms(stages)} != 110")

    waiting = {
        "vector_ms": 100,
        "embed_ms": 40,
        "ann_ms": 60,
        "rerank_wait_ms": 20,
        "rerank_ms": 80,
    }
    if leaf_total_ms(waiting) != 200:
        raise SystemExit(f"{_TEST_MATERIALS['text_004']}{leaf_total_ms(waiting)}")
    print("stage diagnostics ok")


if __name__ == "__main__":
    main()
