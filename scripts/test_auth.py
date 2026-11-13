"""Multi-user login + content isolation."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def _request(
    url: str,
    data: bytes | None = None,
    *,
    method: str | None = None,
    token: str = "",
    timeout: int = 30,
) -> tuple[int, dict]:
    method = method or ("GET" if data is None else "POST")
    req = Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            payload = json.loads(body) if body else {}
            return resp.status, payload
    except HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw.decode("utf-8", errors="replace")}
        return exc.code, payload
    except URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _json(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def main() -> None:
    status, root = _request(f"{BASE}/")
    _assert(status == 200, f"root HTTP {status}")
    print(f"api {root.get('version')} auth_enabled={root.get('auth_enabled')}")

    status, health = _request(f"{BASE}/health")
    _assert(status == 200, f"/health should stay public, got {status}: {health}")

    stamp = str(int(time.time()))
    alice = f"alice_{stamp}"
    bob = f"bob_{stamp}"
    password = "secret12"

    status, created = _request(
        f"{BASE}/auth/register",
        _json({"username": alice, "password": password, "display_name": "Alice"}),
    )
    _assert(status == 201, f"register alice HTTP {status}: {created}")
    _assert(created.get("access_token"), f"register missing token: {created}")
    alice_token = created["access_token"]
    _assert(created.get("user", {}).get("username") == alice, f"alice username: {created}")
    _assert(created.get("eval_gate") is False, f"{_TEST_MATERIALS['text_001']}{created}")

    status, dup = _request(
        f"{BASE}/auth/register",
        _json({"username": alice, "password": password}),
    )
    _assert(status == 409, f"{_TEST_MATERIALS['text_002']}{status}: {dup}")

    status, login = _request(
        f"{BASE}/auth/login",
        _json({"username": alice, "password": password}),
    )
    _assert(status == 200 and login.get("access_token"), f"login alice HTTP {status}: {login}")
    alice_token = login["access_token"]

    status, bad = _request(
        f"{BASE}/auth/login",
        _json({"username": alice, "password": "wrong-password"}),
    )
    _assert(status == 401, f"{_TEST_MATERIALS['text_003']}{status}: {bad}")

    status, me = _request(f"{BASE}/auth/me", token=alice_token)
    _assert(status == 200 and me.get("user", {}).get("username") == alice, f"/auth/me {status}: {me}")

    status, metrics = _request(f"{BASE}/metrics", token=alice_token)
    _assert(
        status == 200 and isinstance((metrics.get("llm") or {}).get("calls"), int), f"/metrics llm {status}: {metrics}"
    )

    status, bob_created = _request(
        f"{BASE}/auth/register",
        _json({"username": bob, "password": password}),
    )
    _assert(status == 201, f"register bob HTTP {status}: {bob_created}")
    bob_token = bob_created["access_token"]

    if not root.get("auth_enabled"):
        print(_TEST_MATERIALS["text_009"])
        print(_TEST_MATERIALS["text_010"])
        return

    status, denied = _request(f"{BASE}/documents")
    _assert(status == 401, f"{_TEST_MATERIALS['text_004']}{status}: {denied}")

    status, eval_denied = _request(f"{BASE}/eval/sets", token=alice_token)
    _assert(status == 403, f"{_TEST_MATERIALS['text_005']}{status}: {eval_denied}")

    status, alice_docs = _request(f"{BASE}/documents", token=alice_token)
    _assert(status == 200, f"alice documents HTTP {status}: {alice_docs}")
    alice_ids = {item.get("doc_id") for item in alice_docs.get("results") or []}

    status, bob_docs = _request(f"{BASE}/documents", token=bob_token)
    _assert(status == 200, f"bob documents HTTP {status}: {bob_docs}")
    bob_ids = {item.get("doc_id") for item in bob_docs.get("results") or []}
    _assert(not (alice_ids & bob_ids), f"{_TEST_MATERIALS['text_006']}{alice_ids & bob_ids}")

    status, alice_traces = _request(f"{BASE}/traces", token=alice_token)
    status_b, bob_traces = _request(f"{BASE}/traces", token=bob_token)
    _assert(status == 200 and status_b == 200, _TEST_MATERIALS["text_008"])
    alice_t = {item.get("id") for item in alice_traces.get("results") or []}
    bob_t = {item.get("id") for item in bob_traces.get("results") or []}
    _assert(not (alice_t & bob_t), f"{_TEST_MATERIALS['text_007']}{alice_t & bob_t}")

    print("auth + isolation ok")


if __name__ == "__main__":
    main()
