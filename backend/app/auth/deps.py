from __future__ import annotations

from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.eval_gate import role_is_admin, user_matches_eval_gate
from app.auth.tokens import TokenError, decode_access_token
from app.core.config import Settings, get_settings
from app.core.startup import is_production, is_weak_jwt
from app.db.session import get_db
from app.models.user import User

_bearer = HTTPBearer(auto_error=False)


def ensure_jwt_secret(settings: Settings) -> None:
    _reject_weak_secret(settings)


def owner_id(user: User | None) -> UUID | None:
    return None if user is None else user.id


def eval_gate_username(settings: Settings) -> str:
    return (settings.eval_gate_username or "").strip().casefold()


def user_has_eval_gate(user: User | None, settings: Settings) -> bool:
    """Regression gate: admin role first; env pin remains a one-release fallback."""
    if user is None:
        return False
    if role_is_admin(getattr(user, "role", None)):
        return True
    return user_matches_eval_gate(
        user_id=user.id,
        username=user.username,
        auth_enabled=settings.auth_enabled,
        gate_user_id=settings.eval_gate_user_id,
        gate_username=settings.eval_gate_username,
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Return the logged-in user, or None when AUTH_ENABLED=false.

    When auth is on, missing/invalid tokens raise 401.
    """
    if not settings.auth_enabled:
        return None
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=401, detail="未登录")
    _reject_weak_secret(settings)
    try:
        payload = decode_access_token(creds.credentials, secret=settings.jwt_secret)
        user_id = UUID(str(payload.get("sub") or ""))
    except (TokenError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="账号不可用")
    return user


def require_eval_gate_user(
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Allow /eval/* without login when AUTH_ENABLED=false (scripts/CI). Otherwise developer only."""
    if not settings.auth_enabled:
        return None
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    if not user_has_eval_gate(user, settings):
        raise HTTPException(status_code=403, detail="仅开发者账号可访问回归闸门")
    return user


def _reject_weak_secret(settings: Settings) -> None:
    if is_production(settings) and is_weak_jwt(settings.jwt_secret):
        raise HTTPException(status_code=503, detail="生产环境必须配置 JWT_SECRET")
