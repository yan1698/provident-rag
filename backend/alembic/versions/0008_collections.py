"""Collections, document cards, and retrieval-scope logs. Revises 0007ingestprep."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008collections"
down_revision: str | None = "0007ingestprep"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS collections (
            id UUID PRIMARY KEY,
            owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_collections_owner ON collections (owner_id)")
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS collection_id UUID REFERENCES collections(id) ON DELETE SET NULL
        """
    )
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS graph_collection_id UUID
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection_id)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_cards (
            doc_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            display_title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            topics JSONB NOT NULL DEFAULT '[]',
            doc_type TEXT NOT NULL DEFAULT 'other',
            filename_usable BOOLEAN NOT NULL DEFAULT true,
            status TEXT NOT NULL DEFAULT 'ready',
            assign_source TEXT NOT NULL DEFAULT '',
            assign_score DOUBLE PRECISION NOT NULL DEFAULT 0,
            assign_gap DOUBLE PRECISION NOT NULL DEFAULT 0,
            assign_version INTEGER NOT NULL DEFAULT 1,
            mixed BOOLEAN NOT NULL DEFAULT false,
            batch_id UUID,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS assign_batches (
            id UUID PRIMARY KEY,
            owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
            snapshot JSONB NOT NULL DEFAULT '[]',
            undone_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS assignment_log (
            id UUID PRIMARY KEY,
            owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
            doc_id UUID,
            source TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_assignment_log_owner ON assignment_log (owner_id, created_at)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS scope_logs (
            id UUID PRIMARY KEY,
            owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
            mode TEXT NOT NULL DEFAULT 'soft',
            widened BOOLEAN NOT NULL DEFAULT false,
            query TEXT NOT NULL DEFAULT '',
            doc_id UUID,
            doc_collection_id UUID,
            scope_key TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_scope_logs_owner ON scope_logs (owner_id, created_at)")
    op.execute(
        """
        ALTER TABLE agent_sessions
        ADD COLUMN IF NOT EXISTS pinned_scope JSONB
        """
    )
    op.execute(
        """
        ALTER TABLE agent_runs
        ADD COLUMN IF NOT EXISTS scope JSONB
        """
    )
    op.execute(
        """
        ALTER TABLE graph_entities
        ADD COLUMN IF NOT EXISTS collection_id UUID
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_graph_entities_collection ON graph_entities (collection_id)")


def downgrade() -> None:
    op.execute("ALTER TABLE graph_entities DROP COLUMN IF EXISTS collection_id")
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS scope")
    op.execute("ALTER TABLE agent_sessions DROP COLUMN IF EXISTS pinned_scope")
    op.execute("DROP TABLE IF EXISTS scope_logs")
    op.execute("DROP TABLE IF EXISTS assignment_log")
    op.execute("DROP TABLE IF EXISTS assign_batches")
    op.execute("DROP TABLE IF EXISTS document_cards")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS graph_collection_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS collection_id")
    op.execute("DROP TABLE IF EXISTS collections")
