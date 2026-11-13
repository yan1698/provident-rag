"""Fail closed on production misconfig. Development may still warn."""

from __future__ import annotations

import logging
from typing import Any

WEAK_JWT_SECRETS = frozenset({"", "dev-only-change-me", "change-me-in-production", "secret"})
WEAK_POSTGRES_PASSWORDS = frozenset({"", "rag", "postgres", "password", "changeme"})

logger = logging.getLogger("provident.startup")


def is_production(settings: Any) -> bool:
    return str(getattr(settings, "app_env", "") or "").strip().lower() == "production"


def is_weak_jwt(secret: str | None) -> bool:
    return (secret or "").strip() in WEAK_JWT_SECRETS


def is_weak_postgres_password(password: str | None) -> bool:
    return (password or "").strip() in WEAK_POSTGRES_PASSWORDS


def _cors_origins(settings: Any) -> list[str]:
    listed = getattr(settings, "cors_origin_list", None)
    if listed is not None:
        return [str(item).strip() for item in listed if str(item).strip()]
    raw = str(getattr(settings, "cors_origins", "") or "").strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect_production_problems(settings: Any) -> list[str]:
    """Return problems that must block APP_ENV=production. Empty when not production."""
    if not is_production(settings):
        return []
    problems: list[str] = []
    if not bool(getattr(settings, "auth_enabled", True)):
        problems.append("AUTH_ENABLED 不能为 false")
    if is_weak_jwt(getattr(settings, "jwt_secret", None)):
        problems.append("JWT_SECRET 必须替换默认值")
    if bool(getattr(settings, "allow_llm_stub", False)):
        problems.append("ALLOW_LLM_STUB 不能为 true")
    if is_weak_postgres_password(getattr(settings, "postgres_password", None)):
        problems.append("POSTGRES_PASSWORD 必须替换默认值")
    if "*" in _cors_origins(settings):
        problems.append("CORS_ORIGINS 不能为 *")
    return problems


def assert_production_ready(settings: Any) -> None:
    problems = collect_production_problems(settings)
    if problems:
        raise RuntimeError("生产环境配置校验失败：" + "；".join(problems))


def warn_if_insecure_defaults(settings: Any) -> None:
    if is_production(settings):
        return
    notes: list[str] = []
    if is_weak_jwt(getattr(settings, "jwt_secret", None)):
        notes.append("JWT_SECRET 仍是默认值")
    if is_weak_postgres_password(getattr(settings, "postgres_password", None)):
        notes.append("POSTGRES_PASSWORD 仍是默认值")
    if "*" in _cors_origins(settings):
        notes.append("CORS_ORIGINS=*")
    if notes:
        logger.warning(
            "开发环境使用不安全默认配置：%s。公网暴露时请运行 python scripts/rotate_secrets.py",
            "；".join(notes),
        )
