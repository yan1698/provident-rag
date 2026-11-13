"""Day 14: Redis JSON cache for retrieval, with graceful fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import redis

from app.core.config import Settings, get_settings

_HITS = "provident:cache:hits"
_MISSES = "provident:cache:misses"


def _client(settings: Settings | None = None) -> redis.Redis | None:
    settings = settings or get_settings()
    try:
        return redis.from_url(settings.redis_connection_url, socket_connect_timeout=1)
    except Exception:  # noqa: BLE001
        return None


def make_key(kind: str, params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    return f"provident:v1:{kind}:{digest}"


def cache_get(key: str, settings: Settings | None = None) -> Any | None:
    client = _client(settings)
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set(key: str, value: Any, *, ttl: int, settings: Settings | None = None) -> None:
    client = _client(settings)
    if client is None:
        return
    try:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        client.setex(key, max(1, ttl), payload)
    except Exception:  # noqa: BLE001
        return


def _incr(name: str, settings: Settings | None = None) -> None:
    client = _client(settings)
    if client is None:
        return
    try:
        client.incr(name)
    except Exception:  # noqa: BLE001
        return


def cached_payload(
    kind: str,
    params: dict[str, Any],
    factory: Callable[[], dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return (payload, cached). Payload never includes a stale ``cached`` flag."""
    settings = settings or get_settings()
    from app.search.stages import note_meta

    if not settings.search_cache_enabled:
        note_meta("search_cache_status", "disabled")
        value = dict(factory())
        value.pop("cached", None)
        return value, False

    key = make_key(kind, params)
    hit = cache_get(key, settings)
    if isinstance(hit, dict):
        note_meta("search_cache_status", "hit")
        hit = dict(hit)
        hit.pop("cached", None)
        _incr(_HITS, settings)
        return hit, True

    note_meta("search_cache_status", "miss")
    value = dict(factory())
    value.pop("cached", None)
    cache_set(key, value, ttl=settings.search_cache_ttl, settings=settings)
    _incr(_MISSES, settings)
    return value, False


def cache_clear_search(settings: Settings | None = None) -> int:
    """Drop all ``provident:v1:*`` search keys. Best-effort; returns deleted count."""
    client = _client(settings)
    if client is None:
        return 0
    deleted = 0
    try:
        for key in client.scan_iter(match="provident:v1:*", count=200):
            deleted += int(client.delete(key) or 0)
    except Exception:  # noqa: BLE001
        return deleted
    return deleted


def cache_stats(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    hits = 0
    misses = 0
    client = _client(settings)
    if client is not None:
        try:
            raw_hits = client.get(_HITS)
            raw_misses = client.get(_MISSES)
            hits = int(raw_hits or 0)
            misses = int(raw_misses or 0)
        except Exception:  # noqa: BLE001
            pass
    total = hits + misses
    return {
        "enabled": settings.search_cache_enabled,
        "ttl": settings.search_cache_ttl,
        "hits": hits,
        "misses": misses,
        "hit_rate": None if total == 0 else hits / total,
    }
