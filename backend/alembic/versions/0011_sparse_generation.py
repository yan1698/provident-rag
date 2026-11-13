"""Transactional BM25 corpus generation, independent of embedding progress."""

from __future__ import annotations

import uuid

from alembic import op
from sqlalchemy import text

revision = "0011sparsegen"
down_revision = "0010tracediag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    schema = bind.scalar(text("SELECT current_schema()"))
    quoted = bind.dialect.identifier_preparer.quote_schema(schema)
    # Installing triggers and their state is one DDL transaction. Never bootstrap
    # this from a query or reuse the Agent's embedding-progress counter.
    op.execute(f"""CREATE TABLE {quoted}.sparse_index_state (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        epoch uuid NOT NULL, generation bigint NOT NULL DEFAULT 0,
        protocol integer NOT NULL CHECK (protocol = 1))""")
    bind.execute(text(f"INSERT INTO {quoted}.sparse_index_state VALUES (true, :epoch, 0, 1)"), {"epoch": uuid.uuid4()})
    op.execute(f"""CREATE FUNCTION {quoted}.bump_sparse_generation() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        UPDATE {quoted}.sparse_index_state SET generation = generation + 1 WHERE singleton;
        IF NOT FOUND THEN RAISE EXCEPTION 'BM25 generation state missing'; END IF;
        RETURN NULL;
        END $$""")
    for table in ("chunks", "documents"):
        op.execute(f"""CREATE TRIGGER sparse_{table}_write
            AFTER INSERT OR DELETE OR TRUNCATE ON {quoted}.{table}
            FOR EACH STATEMENT EXECUTE FUNCTION {quoted}.bump_sparse_generation()""")
    columns = {
        "chunks": ("id", "doc_id", "page", "start_offset", "end_offset", "text", "created_at"),
        "documents": ("id", "owner_id", "collection_id"),
    }
    for table, names in columns.items():
        old = ",".join(f"OLD.{name}" for name in names)
        new = ",".join(f"NEW.{name}" for name in names)
        op.execute(f"""CREATE TRIGGER sparse_{table}_update
            AFTER UPDATE OF {",".join(names)} ON {quoted}.{table} FOR EACH ROW
            WHEN (ROW({old}) IS DISTINCT FROM ROW({new}))
            EXECUTE FUNCTION {quoted}.bump_sparse_generation()""")


def downgrade() -> None:
    bind = op.get_bind()
    schema = bind.dialect.identifier_preparer.quote_schema(bind.scalar(text("SELECT current_schema()")))
    for table in ("chunks", "documents"):
        for suffix in ("write", "update"):
            op.execute(f"DROP TRIGGER sparse_{table}_{suffix} ON {schema}.{table}")
    op.execute(f"DROP FUNCTION {schema}.bump_sparse_generation()")
    op.execute(f"DROP TABLE {schema}.sparse_index_state")
