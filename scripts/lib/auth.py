"""Login helpers for scripts when AUTH_ENABLED=true."""

from __future__ import annotations

import os
from getpass import getpass

from lib.http import API_BASE, post_json


def resolve_bearer_token(
    *,
    token: str = "",
    username: str = "",
    password: str = "",
    auth_enabled: bool = True,
    base: str | None = None,
    default_username: str | None = None,
) -> str:
    """Return JWT for API calls, or empty string when auth is off."""
    api = base or API_BASE
    token = (token or os.environ.get("PROVIDENT_TOKEN") or "").strip()
    if token:
        return token

    username = (username or os.environ.get("PROVIDENT_USERNAME") or default_username or "").strip()
    if username:
        password = password or os.environ.get("PROVIDENT_PASSWORD") or ""
        if not password:
            password = getpass(f"Password for {username}: ")
        login = post_json(f"{api}/auth/login", {"username": username, "password": password})
        access = login.get("access_token")
        if not access:
            raise SystemExit(f"login failed: {login}")
        who = (login.get("user") or {}).get("username") or username
        print(f"logged in as {who}", flush=True)
        return str(access)

    if auth_enabled:
        raise SystemExit(
            "AUTH_ENABLED=true：trace 会按账号隔离。请提供登录态之一：\n"
            "  --token <jwt>  或  $env:PROVIDENT_TOKEN\n"
            "  --username <用户> --password <密码>  或  $env:PROVIDENT_USERNAME / PROVIDENT_PASSWORD\n"
            "（评测管理权限通过 users.role=admin 显式授予）"
        )
    return ""
