"""Fail-closed stamp for committed PostgreSQL corpus snapshots (migration 0011)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.search.stages import stage_timer


class Uncacheable(Exception):
    """Fixed, non-sensitive reason; callers must run the legacy path."""


@dataclass(frozen=True)
class CorpusStamp:
    identity: tuple
    epoch: str
    generation: int


_STAMP_SQL = text("""
    SELECT s.epoch, s.generation, s.protocol, current_user AS role,
           current_database() AS database, s.tableoid AS state_oid,
           c.oid AS chunks_oid, d.oid AS documents_oid,
           current_setting('transaction_isolation') AS isolation,
           current_setting('session_replication_role') AS replication_role,
           txid_current_if_assigned() AS xid,
           c.relrowsecurity OR d.relrowsecurity AS rls,
           (SELECT count(*) FROM pg_trigger t
            WHERE t.tgenabled = 'O' AND NOT t.tgisinternal
              AND t.tgfoid = to_regprocedure('bump_sparse_generation()')
              AND ((t.tgrelid = c.oid AND t.tgname IN ('sparse_chunks_write', 'sparse_chunks_update'))
                OR (t.tgrelid = d.oid AND t.tgname IN ('sparse_documents_write', 'sparse_documents_update')))) AS guards,
           c.relnamespace AS chunk_schema, d.relnamespace AS doc_schema,
           (SELECT relnamespace FROM pg_class WHERE oid = s.tableoid) AS state_schema
    FROM sparse_index_state s
    JOIN pg_class c ON c.oid = to_regclass('chunks')
    JOIN pg_class d ON d.oid = to_regclass('documents')
    WHERE s.singleton
""")


def read_stamp(db) -> CorpusStamp:
    """No commit/rollback of the caller transaction; errors use a connection savepoint.

    Only read-only-so-far READ COMMITTED transactions qualify. A writer's own
    uncommitted version must never escape into a process-global snapshot.
    """
    if isinstance(db, Session):
        if db.new or db.dirty or db.deleted:
            raise Uncacheable("pending_orm_writes")
        conn = db.connection()
    else:
        conn = db
    if conn.dialect.name != "postgresql":
        raise Uncacheable("unsupported_database")
    with stage_timer("sparse_version_check_ms"):
        try:
            # Connection savepoint avoids Session.begin_nested's implicit flush.
            with conn.begin_nested():
                row = conn.execute(_STAMP_SQL).mappings().one()
        except Exception as exc:
            raise Uncacheable("version_unavailable") from exc
    if row["protocol"] != 1 or row["guards"] != 4:
        raise Uncacheable("version_guards_missing")
    if row["isolation"] != "read committed" or row["xid"] is not None:
        raise Uncacheable("transaction_not_committed_reader")
    if row["replication_role"] != "origin" or row["rls"]:
        raise Uncacheable("unsupported_visibility")
    if not row["chunk_schema"] == row["doc_schema"] == row["state_schema"]:
        raise Uncacheable("shadowed_tables")
    return CorpusStamp(
        (conn.engine, row["database"], row["role"], row["state_oid"], row["chunks_oid"], row["documents_oid"]),
        str(row["epoch"]),
        int(row["generation"]),
    )
