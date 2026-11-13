"""HTTP rate limits on paid and abuse-prone routes. Redis-backed; fail-open."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.core.config import Settings, get_settings
from app.core.limiter import consume
from app.db.session import get_db
from app.models.user import User


def client_ip(request: Request) -> str:
    real = (request.headers.get("x-real-ip") or "").strip()
    if real:
        return real.split(",")[0].strip() or "unknown"
    forwarded = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _redis(settings: Settings):
    import redis

    try:
        return redis.from_url(
            settings.redis_connection_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    except Exception:  # noqa: BLE001 — fail-open when Redis is unreachable
        return None


def enforce(settings: Settings, scope: str, identity: str, limit: int) -> None:
    if not settings.rate_limit_enabled or limit <= 0:
        return
    client = _redis(settings)
    if client is None:
        return
    key = f"provident:rl:{scope}:{identity}"
    try:
        allowed, retry_after = consume(
            client,
            key,
            limit,
            settings.rate_limit_window_seconds,
        )
    except Exception:  # noqa: BLE001 — Redis blip must not take the API down
        return
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail=f"请求过于频繁，请 {retry_after} 秒后再试",
        headers={"Retry-After": str(retry_after)},
    )


def _identity(request: Request, user: User | None) -> str:
    if user is not None:
        return f"user:{user.id}"
    return f"ip:{client_ip(request)}"


def rate_limit_register(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    enforce(settings, "register", f"ip:{client_ip(request)}", settings.rate_limit_auth_per_minute)


def rate_limit_login(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    enforce(settings, "login", f"ip:{client_ip(request)}", settings.rate_limit_auth_per_minute)


def rate_limit_query(
    request: Request,
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> None:
    enforce(settings, "query", _identity(request, user), settings.rate_limit_query_per_minute)


def rate_limit_jobs(scope: str):
    """Per-route bucket for ingest / embed / graph (same quota, separate counters)."""

    def _dep(
        request: Request,
        settings: Settings = Depends(get_settings),
        user: User | None = Depends(get_current_user),
    ) -> None:
        enforce(settings, scope, _identity(request, user), settings.rate_limit_jobs_per_minute)

    return _dep


def rate_limit_agent_run(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> None:
    # A retry of a known key must reach create_agent_run, which returns the
    # original run. Admission still rejects a new key while a turn is active.
    if idempotency_key:
        enforce(settings, "agent", _identity(request, user), settings.rate_limit_agent_per_minute)
        return
    if settings.rate_limit_enabled and settings.rate_limit_agent_concurrency > 0:
        from app.agent.service import count_running_agent_runs

        running = count_running_agent_runs(
            db,
            owner_id=current_owner(user),
            stale_seconds=settings.agent_running_stale_seconds,
        )
        if running >= settings.rate_limit_agent_concurrency:
            raise HTTPException(
                status_code=409,
                detail="已有任务正在排队或执行，请等当前任务结束后再提交",
            )
    enforce(settings, "agent", _identity(request, user), settings.rate_limit_agent_per_minute)
