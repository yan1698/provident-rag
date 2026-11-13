"""Optional query diagnostics. Does not change golden score columns. Revises 0009idempotency."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010tracediag"
down_revision: str | None = "0009idempotency"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE query_traces ADD COLUMN IF NOT EXISTS diagnostics JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE query_traces DROP COLUMN IF EXISTS diagnostics")
