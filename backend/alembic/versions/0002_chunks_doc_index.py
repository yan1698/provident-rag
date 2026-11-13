"""Index chunks by document. Missing on volumes created before init.sql had it.

Revision ID: 0002chunkidx
Revises: 0001baseline
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002chunkidx"
down_revision: str | None = "0001baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_doc_id")
