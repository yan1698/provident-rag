"""Process-local cap on Cross-Encoder calls. No torch import."""

from __future__ import annotations

import threading
from contextlib import contextmanager


class InflightGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sem: threading.BoundedSemaphore | None = None
        self._size = 0

    def acquire(self, limit: int) -> bool:
        """Block until a slot is free. ``limit <= 0`` does not block and returns False."""
        size = int(limit)
        if size <= 0:
            return False
        with self._lock:
            if self._sem is None or self._size != size:
                self._sem = threading.BoundedSemaphore(size)
                self._size = size
            sem = self._sem
        sem.acquire()
        return True

    def release(self, held: bool) -> None:
        if not held or self._sem is None:
            return
        self._sem.release()


GATE = InflightGate()


@contextmanager
def rerank_slot(limit: int):
    held = GATE.acquire(limit)
    try:
        yield
    finally:
        GATE.release(held)
