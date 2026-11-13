"""P1-1: live DB is at head; a throwaway database can upgrade from empty."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = "rag_alembic_test"
EXPECTED = (
    "users",
    "documents",
    "chunks",
    "query_traces",
    "agent_runs",
    "query_tags",
    "query_trace_tags",
    "query_feedback",
    "graph_entities",
    "graph_edges",
    "graph_mentions",
    "llm_calls",
    "agent_sessions",
    "agent_session_state",
    "agent_session_events",
    "memory_items",
    "corpus_index_state",
    "sparse_index_state",
    "collections",
    "document_cards",
    "assign_batches",
    "assignment_log",
    "scope_logs",
    "alembic_version",
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "exec", "-T", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _psql(sql: str, *, database: str = "rag") -> str:
    ran = _compose("postgres", "psql", "-U", "rag", "-d", database, "-t", "-A", "-c", sql)
    _assert(ran.returncode == 0, f"psql failed: {(ran.stderr or ran.stdout).strip()}")
    return (ran.stdout or "").strip()


def main() -> None:
    current = _compose("api", "python", "-m", "app.db.migrate", "current")
    _assert(current.returncode == 0, f"alembic current failed: {(current.stderr or current.stdout).strip()}")
    out = (current.stdout or "") + (current.stderr or "")
    _assert("0011sparsegen" in out, f"{_TEST_MATERIALS['text_001']}{out}")

    tables = _psql("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1").splitlines()
    missing = [name for name in EXPECTED if name not in tables]
    _assert(not missing, f"live missing tables: {missing}")
    has_chunk_idx = _psql("SELECT 1 FROM pg_indexes WHERE tablename = 'chunks' AND indexname = 'idx_chunks_doc_id'")
    _assert(has_chunk_idx == "1", _TEST_MATERIALS["text_003"])
    has_role = _psql("SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'role'")
    _assert(has_role == "1", _TEST_MATERIALS["text_004"])
    admins = _psql("SELECT count(*) FROM users WHERE role = 'admin'")
    _assert(admins.isdigit() and int(admins) >= 1, _TEST_MATERIALS["text_005"])
    has_session_col = _psql(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'agent_runs' AND column_name = 'session_id'"
    )
    _assert(has_session_col == "1", _TEST_MATERIALS["text_006"])
    has_rolled_back = _psql(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'agent_runs' AND column_name = 'rolled_back'"
    )
    _assert(has_rolled_back == "1", _TEST_MATERIALS["text_007"])
    has_ingest_version = _psql(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'documents' AND column_name = 'ingest_version'"
    )
    _assert(has_ingest_version == "1", _TEST_MATERIALS["text_008"])

    _psql(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
    )
    _psql(f"DROP DATABASE IF EXISTS {TEST_DB}")
    _psql(f"CREATE DATABASE {TEST_DB}")
    try:
        upgraded = _compose(
            "api",
            "python",
            "-m",
            "app.db.migrate",
            "upgrade",
            "--database",
            TEST_DB,
        )
        _assert(
            upgraded.returncode == 0,
            f"empty upgrade failed: {(upgraded.stderr or upgraded.stdout).strip()}",
        )
        created = _psql(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1",
            database=TEST_DB,
        ).splitlines()
        missing = [name for name in EXPECTED if name not in created]
        _assert(not missing, f"{_TEST_MATERIALS['text_002']}{missing}")
        version = _psql("SELECT version_num FROM alembic_version", database=TEST_DB)
        _assert(version == "0011sparsegen", f"empty upgrade version: {version}")
        has_diagnostics = _psql(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'query_traces' AND column_name = 'diagnostics'",
            database=TEST_DB,
        )
        _assert(has_diagnostics == "1", _TEST_MATERIALS["text_010"])
        has_session_col = _psql(
            "SELECT 1 FROM information_schema.columns WHERE table_name = 'agent_runs' AND column_name = 'session_id'",
            database=TEST_DB,
        )
        _assert(has_session_col == "1", _TEST_MATERIALS["text_011"])
        empty_role = _psql(
            "SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'role'",
            database=TEST_DB,
        )
        _assert(empty_role == "1", _TEST_MATERIALS["text_012"])
        hnsw = _psql(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding_cosine'",
            database=TEST_DB,
        )
        _assert(hnsw == "1", _TEST_MATERIALS["text_013"])
        has_section = _psql(
            "SELECT 1 FROM information_schema.columns WHERE table_name = 'chunks' AND column_name = 'section_path'",
            database=TEST_DB,
        )
        _assert(has_section == "1", _TEST_MATERIALS["text_014"])
    finally:
        _psql(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
        )
        _psql(f"DROP DATABASE IF EXISTS {TEST_DB}")

    print(_TEST_MATERIALS["text_009"])


if __name__ == "__main__":
    main()
