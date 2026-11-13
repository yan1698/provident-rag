"""Who may open the regression gate. No FastAPI / ORM — scripts can import this."""

from __future__ import annotations

from uuid import UUID

ADMIN_ROLE = "admin"


def role_is_admin(role: str | None) -> bool:
    return (role or "").strip().casefold() == ADMIN_ROLE


def user_matches_eval_gate(
    *,
    user_id: UUID | None,
    username: str | None,
    auth_enabled: bool,
    gate_user_id: str = "",
    gate_username: str = "",
) -> bool:
    """True when this login is the configured maintainer.

    If ``gate_user_id`` is set (even if invalid), username is ignored.
    An invalid UUID matches nobody. Username is only a fallback when the id is empty.
    """
    if not auth_enabled or user_id is None:
        return False
    pinned = (gate_user_id or "").strip()
    if pinned:
        try:
            return user_id == UUID(pinned)
        except ValueError:
            return False
    allowed = (gate_username or "").strip().casefold()
    if not allowed:
        return False
    return (username or "").strip().casefold() == allowed
