"""Client idempotency keys for agent runs. Revises 0008collections."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009idempotency"
down_revision: str | None = "0008collections"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_idempotency (
            id UUID PRIMARY KEY,
            owner_key VARCHAR(64) NOT NULL,
            session_key VARCHAR(64) NOT NULL,
            idempotency_key VARCHAR(80) NOT NULL,
            run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            goal TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_idempotency
        ON agent_idempotency (owner_key, session_key, idempotency_key)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_idempotency")
