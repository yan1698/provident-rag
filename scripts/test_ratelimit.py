"""P0-2: fixed-window counter + client IP parsing; optional live 429 via a fake IP."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.limiter import consume

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
# RFC 5737 documentation IP — must not be the operator's real address
_PROBE_IP = "203.0.113.99"


class MemoryCounter:
    def __init__(self) -> None:
        self.n: dict[str, int] = {}
        self.expires: dict[str, int] = {}

    def set(self, name: str, value: int, ex: int | None = None, nx: bool = False) -> bool:
        if nx and name in self.n:
            return False
        self.n[name] = int(value)
        if ex is not None:
            self.expires[name] = int(ex)
        return True

    def incr(self, name: str) -> int:
        self.n[name] = self.n.get(name, 0) + 1
        return self.n[name]

    def ttl(self, name: str) -> int:
        if name not in self.n:
            return -2
        return int(self.expires.get(name, -1))

    def expire(self, name: str, time: int) -> bool:
        if name not in self.n:
            return False
        self.expires[name] = int(time)
        return True


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _client_ip(headers: dict[str, str], host: str | None) -> str:
    real = (headers.get("x-real-ip") or "").strip()
    if real:
        return real.split(",")[0].strip() or "unknown"
    forwarded = (headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return host or "unknown"


def main() -> None:
    store = MemoryCounter()
    key = _TEST_MATERIALS["text_003"]
    allowed = []
    last_retry = 0
    for _ in range(5):
        ok, retry = consume(store, key, limit=3, window=60)
        allowed.append(ok)
        last_retry = retry
    _assert(allowed == [True, True, True, False, False], allowed)
    _assert(last_retry == 60, last_retry)
    _assert(store.n[key] == 5, store.n[key])

    other = MemoryCounter()
    ok, retry = consume(other, "k", limit=0, window=60)
    _assert(ok and retry == 0, (ok, retry))

    _assert(_client_ip({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "127.0.0.1") == "1.2.3.4", "xff")
    _assert(_client_ip({"x-real-ip": "8.8.8.8"}, "127.0.0.1") == "8.8.8.8", "real-ip")
    _assert(_client_ip({}, "127.0.0.1") == "127.0.0.1", "direct")
    print(_TEST_MATERIALS["text_004"])
    _live_login_limit()


def _live_login_limit() -> None:
    if os.environ.get("SKIP_LIVE_RATELIMIT") == "1":
        print("live 429 skipped")
        return
    statuses = []
    detail = ""
    retry_after = ""
    try:
        for _ in range(6):
            req = Request(
                f"{API_BASE}/auth/login",
                data=json.dumps({"username": "rl_probe", "password": "wrong-password"}).encode(),
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("X-Forwarded-For", _PROBE_IP)
            try:
                with urlopen(req, timeout=10) as resp:
                    statuses.append(resp.status)
            except HTTPError as exc:
                statuses.append(exc.code)
                if exc.code == 429:
                    detail = exc.read().decode("utf-8", errors="replace")
                    retry_after = exc.headers.get("Retry-After") or ""
    except URLError as exc:
        print(f"live 429 skipped ({exc.reason})")
        return
    _assert(statuses[:5] == [401] * 5, f"{_TEST_MATERIALS['text_001']}{statuses}")
    _assert(statuses[5] == 429, f"{_TEST_MATERIALS['text_002']}{statuses}")
    _assert(_TEST_MATERIALS["text_005"] in detail, detail)
    _assert(retry_after.isdigit(), f"Retry-After={retry_after!r}")
    print("live login 429 ok")


if __name__ == "__main__":
    main()
