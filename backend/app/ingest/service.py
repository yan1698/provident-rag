from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, joinedload

from app.core.config import Settings
from app.ingest.chunker import ChunkDraft, chunk_pages
from app.ingest.parser import CONTENT_TYPE_BY_SUFFIX, detect_suffix, parse_document
from app.models import Chunk, Document

PREVIEW_CHARS = 80
# Bump when cleaning or chunk boundaries change so a re-upload rebuilds chunks.
INGEST_VERSION = 3


@dataclass
class ChunkView:
    id: uuid.UUID
    page: int
    start: int
    end: int
    token_count: int
    hash: str
    preview: str


@dataclass
class IngestResult:
    doc_id: uuid.UUID
    filename: str
    content_type: str | None
    byte_size: int
    file_hash: str
    reused: bool
    chunk_count: int
    token_stats: dict[str, int]
    chunks: list[ChunkView]


class IngestError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def ingest_file(
    db: Session,
    *,
    filename: str,
    data: bytes,
    content_type: str | None,
    settings: Settings,
    owner_id: uuid.UUID | None = None,
) -> IngestResult:
    safe_name = Path(filename or "unnamed").name
    if not data:
        raise IngestError(400, "empty file")
    if len(data) > settings.ingest_max_bytes:
        raise IngestError(
            413,
            f"file too large: {len(data)} bytes (max {settings.ingest_max_bytes})",
        )

    file_hash = hashlib.sha256(data).hexdigest()
    existing_q = db.query(Document).options(joinedload(Document.chunks)).filter(Document.file_hash == file_hash)
    if owner_id is None:
        existing_q = existing_q.filter(Document.owner_id.is_(None))
    else:
        existing_q = existing_q.filter(Document.owner_id == owner_id)
    existing = existing_q.one_or_none()
    if existing is not None and int(existing.ingest_version or 1) == INGEST_VERSION:
        return _to_result(existing, reused=True)

    try:
        pages = parse_document(safe_name, data)
    except ValueError as exc:
        raise IngestError(400, str(exc)) from exc

    drafts = chunk_pages(
        pages,
        min_tokens=settings.chunk_min_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        filename=safe_name,
    )
    if not drafts:
        raise IngestError(422, "document produced no chunks")

    suffix = detect_suffix(safe_name)
    resolved_type = content_type or CONTENT_TYPE_BY_SUFFIX.get(suffix)
    rebuilt = existing is not None
    if existing is None:
        document = Document(
            owner_id=owner_id,
            filename=safe_name,
            content_type=resolved_type,
            byte_size=len(data),
            file_hash=file_hash,
            ingest_version=INGEST_VERSION,
            chunks=[_chunk_from_draft_unbound(draft) for draft in drafts],
        )
        db.add(document)
    else:
        existing.filename = safe_name
        existing.content_type = resolved_type
        existing.byte_size = len(data)
        existing.ingest_version = INGEST_VERSION
        existing.graph_extracted_at = None
        existing.chunks.clear()
        db.flush()
        for draft in drafts:
            existing.chunks.append(_chunk_from_draft_unbound(draft))
        document = existing
    db.commit()
    if rebuilt:
        try:
            from app.graph.service import prune_orphans

            prune_orphans(db, owner_id=owner_id)
            db.commit()
        except Exception as exc:  # noqa: BLE001 — graph tables may be absent in a partial schema
            if "graph_" not in str(exc) and "does not exist" not in str(exc).lower():
                raise
            db.rollback()
    try:
        from app.corpus.index_state import bump_chunk_generation
        from app.memory.store import stale_on_ingest

        bump_chunk_generation(db, owner_id)
        stale_on_ingest(db, owner_id, ingest_id=str(document.id))
        db.commit()
    except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
        if (
            "corpus_index_state" not in str(exc)
            and "memory_items" not in str(exc)
            and "does not exist" not in str(exc).lower()
        ):
            raise
        db.rollback()
    try:
        from app.corpus.cards import refresh_card

        refresh_card(db, document)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — schema 0008 not applied yet
        if (
            "document_cards" not in str(exc)
            and "collections" not in str(exc)
            and "does not exist" not in str(exc).lower()
        ):
            raise
        db.rollback()
    document = db.query(Document).options(joinedload(Document.chunks)).filter(Document.id == document.id).one()
    from app.cache.service import cache_clear_search

    cache_clear_search(settings)
    return _to_result(document, reused=False)


def _chunk_from_draft_unbound(draft: ChunkDraft) -> Chunk:
    return Chunk(
        page=draft.page,
        start_offset=draft.start_offset,
        end_offset=draft.end_offset,
        text=draft.text,
        section_path=draft.section_path,
        token_count=draft.token_count,
        chunk_hash=draft.chunk_hash,
    )


def _to_result(document: Document, *, reused: bool) -> IngestResult:
    ordered = sorted(document.chunks, key=lambda c: (c.page, c.start_offset))
    chunks = [
        ChunkView(
            id=c.id,
            page=c.page,
            start=c.start_offset,
            end=c.end_offset,
            token_count=c.token_count,
            hash=c.chunk_hash,
            preview=(c.text[:PREVIEW_CHARS] + ("…" if len(c.text) > PREVIEW_CHARS else "")),
        )
        for c in ordered
    ]
    counts = [c.token_count for c in ordered]
    token_stats = {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": round(sum(counts) / len(counts)) if counts else 0,
        "last": counts[-1] if counts else 0,
    }
    return IngestResult(
        doc_id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        byte_size=document.byte_size,
        file_hash=document.file_hash,
        reused=reused,
        chunk_count=len(chunks),
        token_stats=token_stats,
        chunks=chunks,
    )
