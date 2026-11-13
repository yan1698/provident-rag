"""Chunk section path and document ingest version. Revises 0006memstate."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007ingestprep"
down_revision: str | None = "0006memstate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS ingest_version INTEGER NOT NULL DEFAULT 1
        """
    )
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS section_path TEXT NOT NULL DEFAULT ''
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS section_path")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS ingest_version")
