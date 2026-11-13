"""P0-3 / P0-4: another user cannot extract or see another account's documents."""

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
    content_type: str = "application/json",
    timeout: int = 60,
) -> tuple[int, dict]:
    method = method or ("GET" if data is None else "POST")
    req = Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
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


def _register(username: str, password: str) -> str:
    status, payload = _request(
        f"{BASE}/auth/register",
        json.dumps({"username": username, "password": password, "display_name": username}).encode(),
    )
    _assert(status == 201 and payload.get("access_token"), f"register {username} HTTP {status}: {payload}")
    return payload["access_token"]


def _ingest(token: str, filename: str, text: str) -> str:
    import uuid

    boundary = uuid.uuid4().hex
    body = (
        (f"--{boundary}{_TEST_MATERIALS['text_008']}{filename}{_TEST_MATERIALS['text_009']}").encode()
        + text.encode("utf-8")
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, payload = _request(
        f"{BASE}/ingest",
        body,
        token=token,
        content_type=f"{_TEST_MATERIALS['text_007']}{boundary}",
    )
    _assert(status == 200 and payload.get("doc_id"), f"ingest HTTP {status}: {payload}")
    return str(payload["doc_id"])


def main() -> None:
    status, root = _request(f"{BASE}/")
    _assert(status == 200, f"root HTTP {status}")
    if not root.get("auth_enabled"):
        print(_TEST_MATERIALS["text_011"])
        return
    print(f"api {root.get('version')}")

    stamp = str(int(time.time()))
    alice = _register(f"iso_a_{stamp}", "secret12")
    bob = _register(f"iso_b_{stamp}", "secret12")
    doc_id = _ingest(alice, f"iso-{stamp}.md", _TEST_MATERIALS["text_010"])

    status, catalog = _request(f"{BASE}/graph/catalog", token=alice)
    _assert(status == 200, f"alice catalog HTTP {status}: {catalog}")
    row = next((item for item in catalog.get("documents") or [] if item.get("doc_id") == doc_id), None)
    _assert(row is not None, f"alice catalog missing {doc_id}: {catalog}")
    _assert(row.get("extracted") is False, f"{_TEST_MATERIALS['text_001']}{row}")

    status, stolen = _request(
        f"{BASE}/graph",
        json.dumps({"extract": True, "doc_ids": [doc_id]}).encode(),
        token=bob,
    )
    _assert(status == 404, f"{_TEST_MATERIALS['text_002']}{status}: {stolen}")
    _assert(stolen.get("detail") == _TEST_MATERIALS["text_012"], stolen)

    status, catalog = _request(f"{BASE}/graph/catalog", token=alice)
    _assert(status == 200, f"{_TEST_MATERIALS['text_003']}{status}")
    row = next((item for item in catalog.get("documents") or [] if item.get("doc_id") == doc_id), None)
    _assert(row is not None and row.get("extracted") is False, f"{_TEST_MATERIALS['text_004']}{row}")

    status, pending = _request(f"{BASE}/embed", token=alice)
    _assert(status == 200, f"alice embed HTTP {status}: {pending}")
    alice_pending = int(pending.get("pending") or 0)
    status, bob_embed = _request(f"{BASE}/embed", data=b"{}", token=bob)
    _assert(status == 200, f"bob embed HTTP {status}: {bob_embed}")
    status, after = _request(f"{BASE}/embed", token=alice)
    _assert(status == 200, f"{_TEST_MATERIALS['text_005']}{status}: {after}")
    _assert(int(after.get("pending") or 0) == alice_pending, f"{_TEST_MATERIALS['text_006']}{after}")

    print("tenant isolation ok")


if __name__ == "__main__":
    main()
