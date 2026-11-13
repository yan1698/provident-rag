"""Add users.role. First real revision after the baseline stamp.

Revision ID: 0003userrole
Revises: 0002chunkidx
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from uuid import UUID

from alembic import op
from sqlalchemy import text

revision: str = "0003userrole"
down_revision: str | None = "0002chunkidx"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(16) NOT NULL DEFAULT 'user'")
    pinned = (os.environ.get("EVAL_GATE_USER_ID") or "").strip()
    username = (os.environ.get("EVAL_GATE_USERNAME") or "").strip()
    conn = op.get_bind()
    if pinned:
        try:
            UUID(pinned)
        except ValueError:
            pinned = ""
    if pinned:
        conn.execute(
            text("UPDATE users SET role = 'admin' WHERE id = CAST(:id AS uuid)"),
            {"id": pinned},
        )
    elif username:
        conn.execute(
            text("UPDATE users SET role = 'admin' WHERE lower(username) = lower(:u)"),
            {"u": username},
        )


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS role")
