"""Session checkpoint, memory items, corpus index version. Revises 0005agentsess."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006memstate"
down_revision: str | None = "0005agentsess"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS rolled_back BOOLEAN NOT NULL DEFAULT FALSE
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_session_state (
            session_id       UUID PRIMARY KEY REFERENCES agent_sessions(id) ON DELETE CASCADE,
            revision         INTEGER NOT NULL DEFAULT 0,
            slots            JSONB NOT NULL DEFAULT '{}'::jsonb,
            catalog_digest   JSONB NOT NULL DEFAULT '[]'::jsonb,
            gaps             JSONB NOT NULL DEFAULT '[]'::jsonb,
            checkpoint_json  JSONB,
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_session_events (
            id              UUID PRIMARY KEY,
            session_id      UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
            run_id          UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
            revision_after  INTEGER NOT NULL DEFAULT 0,
            type            VARCHAR(32) NOT NULL DEFAULT '',
            payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_session_events_session ON agent_session_events (session_id, created_at)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            id                UUID PRIMARY KEY,
            owner_id          UUID REFERENCES users(id) ON DELETE CASCADE,
            session_id        UUID REFERENCES agent_sessions(id) ON DELETE SET NULL,
            kind              VARCHAR(32) NOT NULL,
            key               TEXT NOT NULL DEFAULT '',
            payload           JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_event_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
            source_run_id     UUID,
            version           INTEGER NOT NULL DEFAULT 1,
            rule_version      INTEGER NOT NULL DEFAULT 1,
            stale             BOOLEAN NOT NULL DEFAULT FALSE,
            retracted         BOOLEAN NOT NULL DEFAULT FALSE,
            stale_ingest_id   TEXT NOT NULL DEFAULT '',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_owner_kind ON memory_items (owner_id, kind)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_index_state (
            owner_key                 VARCHAR(64) PRIMARY KEY,
            chunk_generation          INTEGER NOT NULL DEFAULT 0,
            committed_index_version   INTEGER NOT NULL DEFAULT 0,
            updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS corpus_index_state")
    op.execute("DROP TABLE IF EXISTS memory_items")
    op.execute("DROP INDEX IF EXISTS idx_agent_session_events_session")
    op.execute("DROP TABLE IF EXISTS agent_session_events")
    op.execute("DROP TABLE IF EXISTS agent_session_state")
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS rolled_back")
