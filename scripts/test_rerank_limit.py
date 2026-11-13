"""Cross-Encoder slot and agent stop helpers. No torch, no database."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.budget import next_failure_streak, past_deadline  # noqa: E402
from app.rerank.limit import InflightGate  # noqa: E402


def _assert(cond: bool, detail: object) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {detail}")


def main() -> None:
    streak, tripped = next_failure_streak(0, {"error": "x"}, limit=2)
    _assert(streak == 1 and not tripped, (streak, tripped))
    streak, tripped = next_failure_streak(streak, {"error": "y"}, limit=2)
    _assert(streak == 2 and tripped, (streak, tripped))
    streak, tripped = next_failure_streak(streak, {"count": 1}, limit=2)
    _assert(streak == 0 and not tripped, (streak, tripped))
    streak, tripped = next_failure_streak(5, {"error": "x"}, limit=0)
    _assert(streak == 6 and not tripped, (streak, tripped))
    _assert(past_deadline(None, 10) is False, "none")
    _assert(past_deadline(5, 5) is True, "equal")
    _assert(past_deadline(6, 5) is False, "before")

    gate = InflightGate()
    _assert(gate.acquire(0) is False, _TEST_MATERIALS["text_001"])
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def _first() -> None:
        gate.acquire(1)
        order.append("first")
        entered.set()
        release.wait(2)
        gate.release(True)
        order.append("released")

    def _second() -> None:
        entered.wait(2)
        gate.acquire(1)
        order.append("second")
        gate.release(True)

    threads = [threading.Thread(target=_first), threading.Thread(target=_second)]
    threads[0].start()
    if not entered.wait(2):
        raise SystemExit(_TEST_MATERIALS["text_003"])
    threads[1].start()
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
    _assert(order[0] == "first" and "second" in order and "released" in order, order)

    print(_TEST_MATERIALS["text_002"])


if __name__ == "__main__":
    main()
