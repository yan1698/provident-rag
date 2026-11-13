"""HTTP helpers for acceptance scripts and run_eval (stdlib only)."""

from __future__ import annotations

import json
import os
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def is_rate_limited(status: int, detail: str) -> bool:
    """Kimi 免费额度 RPM 很低；限速会被后端包成 503，需要看 detail 才能区分。"""
    if status == 429:
        return True
    return status == 503 and ("429" in detail or "rate_limit" in detail)


def request_json(
    url: str,
    data: bytes | None = None,
    *,
    method: str | None = None,
    token: str = "",
    timeout: int = 240,
    headers: dict[str, str] | None = None,
    retries: int = 0,
    backoff: float = 30.0,
) -> dict:
    req_method = method or ("GET" if data is None else "POST")
    attempt = 0
    while True:
        req = Request(url, data=data, method=req_method)
        extra = headers or {}
        if not any(key.lower() == "content-type" for key in extra):
            req.add_header("Content-Type", "application/json")
        for key, value in extra.items():
            req.add_header(key, value)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if attempt < retries and is_rate_limited(exc.code, detail):
                wait = backoff * (attempt + 1)
                print(f"  rate limited, retrying in {wait:.0f}s ...", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            raise SystemExit(f"HTTP {exc.code} {url}: {detail}") from exc
        except URLError as exc:
            raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc


def get_json(
    url: str,
    *,
    token: str = "",
    timeout: int = 240,
    retries: int = 0,
    backoff: float = 30.0,
) -> dict:
    return request_json(url, token=token, timeout=timeout, retries=retries, backoff=backoff)


def post_json(
    url: str,
    payload: dict,
    *,
    token: str = "",
    timeout: int = 240,
    retries: int = 0,
    backoff: float = 30.0,
) -> dict:
    return request_json(
        url,
        data=json.dumps(payload).encode("utf-8"),
        token=token,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )


def post_file(
    filename: str,
    payload: bytes,
    content_type: str,
    *,
    base: str | None = None,
    token: str = "",
    timeout: int = 240,
    retries: int = 0,
    backoff: float = 30.0,
) -> dict:
    boundary = uuid.uuid4().hex
    disp = f'Content-Disposition: form-data; name="file"; filename="{filename}"'
    body = (
        f"--{boundary}\r\n{disp}\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8")
        + payload
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    return request_json(
        f"{base or API_BASE}/ingest",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        token=token,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )


def raw_get(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> tuple[int, dict[str, str], bytes]:
    """Return status, lower-cased headers, body (for CORS / static asset checks)."""
    req = Request(url, method="GET")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urlopen(req, timeout=timeout) as resp:
            header_map = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, header_map, resp.read()
    except HTTPError as exc:
        header_map = {k.lower(): v for k, v in exc.headers.items()}
        return exc.code, header_map, exc.read()
    except URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc
