from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.deps import ensure_jwt_secret, get_current_user, user_has_eval_gate
from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import create_access_token
from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_login, rate_limit_register
from app.db.session import get_db
from app.models.user import User

router = APIRouter(tags=["auth"])

_optional_bearer = HTTPBearer(auto_error=False)
_USERNAME_RE = re.compile(r"^[\w\u4e00-\u9fff]{2,32}$")


class RegisterIn(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=64)


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=32)
    password: str = Field(..., min_length=1, max_length=128)


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    display_name: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
    eval_gate: bool = False


class AuthStatusOut(BaseModel):
    auth_enabled: bool
    user: UserOut | None = None
    eval_gate: bool = False


def normalize_username(raw: str) -> str:
    value = (raw or "").strip().casefold()
    if not _USERNAME_RE.match(value):
        raise ValueError("用户名为 2–32 个字符，可用字母、数字、下划线或汉字")
    return value


def to_user_out(row: User) -> UserOut:
    return UserOut(
        id=row.id,
        username=row.username,
        display_name=row.display_name or row.username,
        created_at=row.created_at,
    )


def _issue(user: User, settings: Settings) -> TokenOut:
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        secret=settings.jwt_secret,
        expire_hours=settings.jwt_expire_hours,
    )
    return TokenOut(
        access_token=token,
        user=to_user_out(user),
        eval_gate=user_has_eval_gate(user, settings),
    )


@router.get("/auth/status", response_model=AuthStatusOut)
def auth_status(settings: Settings = Depends(get_settings)) -> AuthStatusOut:
    return AuthStatusOut(auth_enabled=settings.auth_enabled, user=None, eval_gate=False)


@router.post("/auth/register", response_model=TokenOut, status_code=201)
def auth_register(
    body: RegisterIn,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit_register),
) -> TokenOut:
    try:
        username = normalize_username(body.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_jwt_secret(settings)
    display = (body.display_name or "").strip() or username
    row = User(
        username=username,
        display_name=display,
        password_hash=hash_password(body.password),
        role="user",
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="用户名已被使用") from exc
    db.refresh(row)
    return _issue(row, settings)


@router.post("/auth/login", response_model=TokenOut)
def auth_login(
    body: LoginIn,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit_login),
) -> TokenOut:
    try:
        username = normalize_username(body.username)
    except ValueError:
        username = (body.username or "").strip().casefold()
    row = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if row is None or not row.is_active or not verify_password(body.password, row.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码不正确")
    ensure_jwt_secret(settings)
    row.last_login_at = datetime.now(timezone.utc)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _issue(row, settings)


@router.get("/auth/me", response_model=AuthStatusOut)
def auth_me(
    creds: HTTPAuthorizationCredentials | None = Depends(_optional_bearer),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthStatusOut:
    if not settings.auth_enabled:
        return AuthStatusOut(auth_enabled=False, user=None)
    user = get_current_user(creds, db, settings)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    return AuthStatusOut(
        auth_enabled=True,
        user=to_user_out(user),
        eval_gate=user_has_eval_gate(user, settings),
    )
