from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.embed.model import encode_texts
from app.models import Chunk, Document


@dataclass
class EmbedRunStats:
    pending: int
    embedded: int
    skipped: int
    copied: int
    batches: int


def count_pending(db: Session, *, doc_id: UUID | None = None, owner_id: UUID | None = None) -> int:
    stmt = select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
    if doc_id is not None:
        stmt = stmt.where(Chunk.doc_id == doc_id)
    elif owner_id is not None:
        stmt = stmt.join(Document, Chunk.doc_id == Document.id).where(Document.owner_id == owner_id)
    return int(db.scalar(stmt) or 0)


def run_embedding_pipeline(
    db: Session,
    *,
    settings: Settings | None = None,
    doc_id: UUID | None = None,
    owner_id: UUID | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> EmbedRunStats:
    settings = settings or get_settings()
    pending_total = count_pending(db, doc_id=doc_id, owner_id=owner_id)
    if pending_total == 0:
        if not dry_run:
            try:
                from app.corpus.index_state import commit_index_version

                commit_index_version(db, owner_id)
                db.commit()
            except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
                if "corpus_index_state" not in str(exc) and "does not exist" not in str(exc).lower():
                    raise
                db.rollback()
        return EmbedRunStats(pending=0, embedded=0, skipped=0, copied=0, batches=0)

    to_process = _fetch_pending(db, doc_id=doc_id, owner_id=owner_id, limit=limit)
    hash_to_vector = _embedded_hash_map(
        db,
        hashes={chunk.chunk_hash for chunk in to_process if chunk.chunk_hash},
        owner_id=owner_id,
    )
    skipped = 0
    embedded = 0
    copied = 0
    batches = 0

    batch_size = settings.embedding_batch_size
    for start in range(0, len(to_process), batch_size):
        batch = to_process[start : start + batch_size]
        encode_queue: list[Chunk] = []
        now = datetime.now(timezone.utc)
        changed = False

        for chunk in batch:
            if chunk.chunk_hash in hash_to_vector:
                if dry_run:
                    copied += 1
                else:
                    chunk.embedding = hash_to_vector[chunk.chunk_hash]
                    chunk.embedded_at = now
                    copied += 1
                    changed = True
                continue
            encode_queue.append(chunk)

        if encode_queue:
            batches += 1
            if dry_run:
                embedded += len(encode_queue)
                for chunk in encode_queue:
                    hash_to_vector.setdefault(chunk.chunk_hash, [])
                continue

            vectors = encode_texts(
                [_embed_input(c, settings) for c in encode_queue],
                settings=settings,
            )
            for chunk, vector in zip(encode_queue, vectors, strict=True):
                chunk.embedding = vector
                chunk.embedded_at = now
                hash_to_vector[chunk.chunk_hash] = vector
            embedded += len(encode_queue)
            changed = True

        if not dry_run and changed:
            db.commit()

    if not dry_run:
        try:
            from app.corpus.index_state import commit_index_version

            commit_index_version(db, owner_id)
            db.commit()
        except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
            if "corpus_index_state" not in str(exc) and "does not exist" not in str(exc).lower():
                raise
            db.rollback()

    return EmbedRunStats(
        pending=pending_total,
        embedded=embedded,
        skipped=skipped,
        copied=copied,
        batches=batches,
    )


def _scope_chunks(stmt, *, doc_id: UUID | None = None, owner_id: UUID | None = None):
    if doc_id is not None:
        return stmt.where(Chunk.doc_id == doc_id)
    if owner_id is not None:
        return stmt.join(Document, Chunk.doc_id == Document.id).where(Document.owner_id == owner_id)
    return stmt


_HASH_IN_BATCH = 500


def _embedded_hash_map(
    db: Session,
    *,
    hashes: set[str] | None = None,
    owner_id: UUID | None = None,
) -> dict[str, list[float]]:
    """Load vectors for the given hashes only. ``hashes=None`` is the full-table fallback."""
    mapping: dict[str, list[float]] = {}
    if hashes is not None and not hashes:
        return mapping

    def _load(stmt):
        scoped = _scope_chunks(stmt, owner_id=owner_id)
        for chunk_hash, embedding in db.execute(scoped).all():
            if chunk_hash not in mapping and embedding is not None:
                mapping[chunk_hash] = list(embedding)

    base = select(Chunk.chunk_hash, Chunk.embedding).where(Chunk.embedding.is_not(None))
    if hashes is None:
        _load(base)
        return mapping
    wanted = list(hashes)
    for start in range(0, len(wanted), _HASH_IN_BATCH):
        batch = wanted[start : start + _HASH_IN_BATCH]
        _load(base.where(Chunk.chunk_hash.in_(batch)))
    return mapping


def _embed_input(chunk: Chunk, settings: Settings) -> str:
    """Section path is a short prefix so a chunk is not embedded without its heading."""
    body = chunk.text or ""
    path = (chunk.section_path or "").strip()
    if path:
        body = f"{path[:160]}\n{body}"
    prefix = settings.embedding_passage_prefix or ""
    return f"{prefix}{body}" if prefix else body


def _fetch_pending(
    db: Session,
    *,
    doc_id: UUID | None,
    owner_id: UUID | None = None,
    limit: int | None,
) -> list[Chunk]:
    stmt = _scope_chunks(
        select(Chunk).where(Chunk.embedding.is_(None)).order_by(Chunk.created_at, Chunk.start_offset),
        doc_id=doc_id,
        owner_id=owner_id,
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())
