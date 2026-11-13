from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.cache.service import cache_clear_search
from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_jobs
from app.db.session import get_db
from app.graph.service import prune_orphans
from app.ingest.service import IngestError, ingest_file
from app.models import Chunk, Document
from app.models.user import User

router = APIRouter(tags=["ingest"])


def _document_section_rows(db, user, doc_id):
    from app.agent.store import SqlAgentStore

    doc = db.get(Document, doc_id)
    uid = current_owner(user)
    if doc is None or (uid is not None and doc.owner_id != uid):
        raise HTTPException(status_code=404, detail="document not found")
    return SqlAgentStore(db, owner_id=uid).load_doc_chunks(str(doc_id))


@router.get("/documents/{doc_id}/sections")
def document_sections(doc_id: uuid.UUID, db: Session = Depends(get_db), user: User | None = Depends(get_current_user)):
    from app.corpus.sections import section_records

    return {"doc_id": str(doc_id), "sections": section_records(str(doc_id), _document_section_rows(db, user, doc_id))}


@router.get("/documents/{doc_id}/sections/{section_id}")
def document_section(
    doc_id: uuid.UUID,
    section_id: str,
    descendants: bool = True,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
):
    from dataclasses import asdict
    from app.corpus.sections import section_chunk_ids, section_records

    rows = _document_section_rows(db, user, doc_id)
    sections = section_records(str(doc_id), rows)
    selected = next((s for s in sections if s["id"] == section_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="section not found")
    wanted = set(section_chunk_ids(sections, section_id, descendants=descendants))
    return {"doc_id": str(doc_id), "section": selected, "chunks": [asdict(r) for r in rows if r.id in wanted]}


class ChunkOut(BaseModel):
    id: uuid.UUID
    page: int
    start: int
    end: int
    token_count: int
    hash: str
    preview: str


class TokenStats(BaseModel):
    min: int
    max: int
    mean: int
    last: int


class IngestOut(BaseModel):
    doc_id: uuid.UUID
    filename: str
    content_type: str | None
    byte_size: int
    file_hash: str
    reused: bool = Field(description="True when the same file hash was already ingested")
    chunk_count: int
    token_stats: TokenStats
    chunks: list[ChunkOut]


class DocumentListItem(BaseModel):
    doc_id: uuid.UUID
    filename: str
    content_type: str | None
    byte_size: int
    chunk_count: int
    embedded_count: int
    created_at: datetime
    display_title: str = ""
    collection_id: uuid.UUID | None = None
    collection_name: str = ""
    doc_type: str = ""
    doc_type_label: str = ""
    card_status: str = ""
    filename_usable: bool = True
    mixed: bool = False


class DocumentListOut(BaseModel):
    count: int
    chunks: int
    embedded_chunks: int
    results: list[DocumentListItem]


def _attach_cards(db: Session, results: list[DocumentListItem]) -> None:
    if not results:
        return
    from app.corpus.cards import DOC_TYPE_LABELS
    from app.models.collection import Collection, DocumentCard

    ids = [item.doc_id for item in results]
    cards = {
        row.doc_id: row for row in db.execute(select(DocumentCard).where(DocumentCard.doc_id.in_(ids))).scalars().all()
    }
    missing = [item.doc_id for item in results if item.doc_id not in cards]
    if missing:
        from sqlalchemy.orm import selectinload

        docs = (
            db.execute(select(Document).where(Document.id.in_(missing)).options(selectinload(Document.chunks)))
            .scalars()
            .all()
        )
        from app.corpus.cards import refresh_card

        for doc in docs:
            try:
                refresh_card(db, doc)
            except Exception:
                db.rollback()
                break
        else:
            db.commit()
            cards = {
                row.doc_id: row
                for row in db.execute(select(DocumentCard).where(DocumentCard.doc_id.in_(ids))).scalars().all()
            }
            placed = dict(db.execute(select(Document.id, Document.collection_id).where(Document.id.in_(ids))).all())
            for item in results:
                item.collection_id = placed.get(item.doc_id)
    collection_ids = {item.collection_id for item in results if item.collection_id}
    names = {}
    if collection_ids:
        names = {
            row.id: row.name
            for row in db.execute(select(Collection).where(Collection.id.in_(collection_ids))).scalars().all()
        }
    for item in results:
        card = cards.get(item.doc_id)
        item.display_title = (card.display_title if card else "") or item.filename
        item.collection_name = names.get(item.collection_id, "")
        if card is None:
            item.card_status = "pending"
            continue
        item.doc_type = card.doc_type or ""
        item.doc_type_label = DOC_TYPE_LABELS.get(card.doc_type, "其他")
        item.card_status = card.status or "ready"
        item.filename_usable = bool(card.filename_usable)
        item.mixed = bool(card.mixed)


@router.get("/documents", response_model=DocumentListOut)
def documents_list(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> DocumentListOut:
    stmt = (
        select(
            Document.id,
            Document.filename,
            Document.content_type,
            Document.byte_size,
            Document.created_at,
            func.count(Chunk.id),
            func.count(Chunk.embedded_at),
            Document.collection_id,
        )
        .outerjoin(Chunk, Chunk.doc_id == Document.id)
        .group_by(Document.id, Document.collection_id)
        .order_by(Document.created_at.desc())
    )
    uid = current_owner(user)
    if uid is not None:
        stmt = stmt.where(Document.owner_id == uid)
    rows = db.execute(stmt).all()
    results = [
        DocumentListItem(
            doc_id=row[0],
            filename=row[1],
            content_type=row[2],
            byte_size=int(row[3] or 0),
            created_at=row[4],
            chunk_count=int(row[5] or 0),
            embedded_count=int(row[6] or 0),
            collection_id=row[7] if len(row) > 7 else None,
        )
        for row in rows
    ]
    _attach_cards(db, results)
    return DocumentListOut(
        count=len(results),
        chunks=sum(item.chunk_count for item in results),
        embedded_chunks=sum(item.embedded_count for item in results),
        results=results,
    )


class DocumentChunkOut(BaseModel):
    index: int
    chunk_id: uuid.UUID
    page: int
    start: int
    end: int
    token_count: int
    embedded: bool
    text: str
    chunk_hash: str = ""


class DocumentDetailOut(BaseModel):
    doc_id: uuid.UUID
    filename: str
    content_type: str | None
    byte_size: int
    file_hash: str
    chunk_count: int
    embedded_count: int
    created_at: datetime
    chunks: list[DocumentChunkOut]
    display_title: str = ""
    collection_id: uuid.UUID | None = None
    collection_name: str = ""
    summary: str = ""
    doc_type_label: str = ""
    card_status: str = ""


@router.get("/documents/{doc_id}", response_model=DocumentDetailOut)
def documents_get(
    doc_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
) -> DocumentDetailOut:
    doc = db.get(Document, doc_id)
    uid = current_owner(user)
    if doc is None or (uid is not None and doc.owner_id != uid):
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
    rows = (
        db.execute(select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.page, Chunk.start_offset)).scalars().all()
    )
    chunks = [
        DocumentChunkOut(
            index=i,
            chunk_id=row.id,
            page=row.page,
            start=row.start_offset,
            end=row.end_offset,
            token_count=row.token_count,
            embedded=row.embedded_at is not None,
            text=row.text or "",
            chunk_hash=row.chunk_hash or "",
        )
        for i, row in enumerate(rows, start=1)
    ]
    detail = DocumentDetailOut(
        doc_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        byte_size=doc.byte_size,
        file_hash=doc.file_hash or "",
        chunk_count=len(chunks),
        embedded_count=sum(1 for item in chunks if item.embedded),
        created_at=doc.created_at,
        chunks=chunks,
    )
    listing = [
        DocumentListItem(
            doc_id=doc.id,
            filename=doc.filename,
            content_type=doc.content_type,
            byte_size=doc.byte_size,
            chunk_count=len(chunks),
            embedded_count=detail.embedded_count,
            created_at=doc.created_at,
            collection_id=doc.collection_id,
        )
    ]
    _attach_cards(db, listing)
    card_row = listing[0]
    detail.display_title = card_row.display_title
    detail.collection_id = card_row.collection_id
    detail.collection_name = card_row.collection_name
    detail.doc_type_label = card_row.doc_type_label
    detail.card_status = card_row.card_status
    from app.models.collection import DocumentCard

    card = db.get(DocumentCard, doc.id)
    detail.summary = card.summary if card is not None else ""
    return detail


class DocumentDeleteOut(BaseModel):
    doc_id: uuid.UUID
    filename: str
    chunk_count: int
    deleted: bool = True


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteOut)
def documents_delete(
    doc_id: uuid.UUID,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> DocumentDeleteOut:
    doc = db.get(Document, doc_id)
    uid = current_owner(user)
    if doc is None or (uid is not None and doc.owner_id != uid):
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
    chunk_count = db.execute(select(func.count()).select_from(Chunk).where(Chunk.doc_id == doc_id)).scalar_one()
    filename = doc.filename
    db.delete(doc)
    db.commit()
    prune_orphans(db, owner_id=uid)
    db.commit()
    cache_clear_search(settings)
    return DocumentDeleteOut(
        doc_id=doc_id,
        filename=filename,
        chunk_count=int(chunk_count or 0),
        deleted=True,
    )


@router.post("/ingest", response_model=IngestOut)
async def ingest(
    file: UploadFile = File(..., description="PDF / TXT / Markdown"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
    _: None = Depends(rate_limit_jobs("ingest")),
) -> Any:
    data = await file.read()
    try:
        result = ingest_file(
            db,
            filename=file.filename or "unnamed",
            data=data,
            content_type=file.content_type,
            settings=settings,
            owner_id=current_owner(user),
        )
    except IngestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return result
