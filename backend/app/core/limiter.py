"""Fixed-window counter. Store must speak Redis SET/INCR/TTL/EXPIRE."""

from __future__ import annotations

from typing import Protocol


class Counter(Protocol):
    def set(self, name: str, value: int, ex: int | None = None, nx: bool = False) -> bool: ...
    def incr(self, name: str) -> int: ...
    def ttl(self, name: str) -> int: ...
    def expire(self, name: str, time: int) -> bool: ...


def consume(store: Counter, key: str, limit: int, window: int) -> tuple[bool, int]:
    """Increment ``key`` inside a fixed window.

    Returns ``(allowed, retry_after_seconds)``. ``limit <= 0`` means unlimited.
    """
    if limit <= 0:
        return True, 0
    window = max(int(window), 1)
    store.set(key, 0, ex=window, nx=True)
    count = int(store.incr(key))
    ttl = int(store.ttl(key))
    if ttl < 0:
        store.expire(key, window)
        ttl = window
    if count > limit:
        return False, max(ttl, 1)
    return True, 0
