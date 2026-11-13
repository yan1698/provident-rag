"""HS256 JWT using the standard library (header.payload.signature)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from uuid import UUID


class TokenError(Exception):
    """Invalid or expired access token."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_access_token(
    *,
    user_id: UUID,
    username: str,
    secret: str,
    expire_hours: int = 24,
) -> str:
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + max(1, expire_hours) * 3600,
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing = f"{header}.{body}".encode()
    signature = _b64encode(hmac.new(secret.encode("utf-8"), signing, hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def decode_access_token(token: str, *, secret: str) -> dict[str, Any]:
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise TokenError("malformed token")
    header_b64, body_b64, signature_b64 = parts
    expected = _b64encode(
        hmac.new(
            secret.encode("utf-8"),
            f"{header_b64}.{body_b64}".encode(),
            hashlib.sha256,
        ).digest()
    )
    if not hmac.compare_digest(expected, signature_b64):
        raise TokenError("invalid signature")
    try:
        payload = json.loads(_b64decode(body_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("invalid payload") from exc
    if not isinstance(payload, dict):
        raise TokenError("invalid payload")
    exp = payload.get("exp")
    try:
        if int(exp) < int(time.time()):
            raise TokenError("expired")
    except (TypeError, ValueError) as exc:
        raise TokenError("invalid expiry") from exc
    return payload
