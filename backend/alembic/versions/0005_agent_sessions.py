"""Session table for multi-turn Agent. Revises 0004llmcalls."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005agentsess"
down_revision: str | None = "0004llmcalls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id          UUID PRIMARY KEY,
            user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
            title       TEXT NOT NULL DEFAULT '',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id ON agent_sessions (user_id)")
    op.execute(
        "ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS session_id UUID "
        "REFERENCES agent_sessions(id) ON DELETE SET NULL"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_session_id ON agent_runs (session_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_agent_runs_session_id")
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS session_id")
    op.execute("DROP TABLE IF EXISTS agent_sessions")
