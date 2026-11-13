"""Per-owner ingest generation vs committed embedding version."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session


def owner_key(owner_id: UUID | None) -> str:
    return str(owner_id) if owner_id is not None else "anon"


def _row(db: Session, key: str):
    from app.models.agent import CorpusIndexState

    row = db.get(CorpusIndexState, key)
    if row is None:
        row = CorpusIndexState(
            owner_key=key,
            chunk_generation=0,
            committed_index_version=0,
        )
        db.add(row)
        db.flush()
    return row


def bump_chunk_generation(db: Session, owner_id: UUID | None) -> int:
    row = _row(db, owner_key(owner_id))
    row.chunk_generation = int(row.chunk_generation or 0) + 1
    row.updated_at = datetime.now(timezone.utc)
    db.add(row)
    return int(row.chunk_generation)


def commit_index_version(db: Session, owner_id: UUID | None) -> int:
    row = _row(db, owner_key(owner_id))
    row.committed_index_version = int(row.chunk_generation or 0)
    row.updated_at = datetime.now(timezone.utc)
    db.add(row)
    return int(row.committed_index_version)


def retrieve_index_stamp(db: Session, owner_id: UUID | None) -> dict[str, int | bool]:
    """index_version = searchable embed; committed_index_version = ingest generation."""
    row = _row(db, owner_key(owner_id))
    generation = int(row.chunk_generation or 0)
    searchable = int(row.committed_index_version or 0)
    return {
        "index_version": searchable,
        "committed_index_version": generation,
        "chunk_generation": generation,
        "search_executed": True,
        "downstream_ok": True,
        "index_ready": generation == searchable,
    }
