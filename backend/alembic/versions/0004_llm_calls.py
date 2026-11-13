"""Persist every metered LLM call.

Revision ID: 0004llmcalls
Revises: 0003userrole
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004llmcalls"
down_revision: str | None = "0003userrole"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id               UUID PRIMARY KEY,
            user_id          UUID REFERENCES users(id) ON DELETE SET NULL,
            run_id           UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
            trace_id         UUID REFERENCES query_traces(id) ON DELETE SET NULL,
            provider         VARCHAR(32) NOT NULL DEFAULT '',
            model            VARCHAR(128) NOT NULL DEFAULT '',
            purpose          VARCHAR(32) NOT NULL DEFAULT '',
            prompt_tokens    INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms       INTEGER NOT NULL DEFAULT 0,
            status           VARCHAR(16) NOT NULL DEFAULT 'ok',
            request_id       VARCHAR(64) NOT NULL DEFAULT '',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at ON llm_calls (created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_user_id ON llm_calls (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_trace_id ON llm_calls (trace_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_purpose ON llm_calls (purpose)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_calls")
