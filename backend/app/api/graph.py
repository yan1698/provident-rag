from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_jobs
from app.db.session import get_db
from app.graph.jobs import graph_job_state, start_graph_job
from app.graph.service import count_pending, document_graph, entity_neighborhood, graph_counts
from app.models import Document
from app.models.user import User

router = APIRouter(tags=["graph"])


class GraphStatusOut(BaseModel):
    enabled: bool
    running: bool
    started: bool = False
    documents: int
    pending_docs: int
    built_docs: int
    entities: int
    edges: int
    mentions: int
    stale_mentions: int = 0
    pending_extract: int = 0
    extract_enabled: bool = False
    skipped_chunks: int = 0
    extract_calls: int = 0
    extract_failed: int = 0
    extract_empty: int = 0
    extract_dropped: int = 0
    concepts: int = 0
    source: str = "rules"
    error: str | None = None
    note: str | None = None


def _status(db: Session, *, started: bool = False, owner_id=None, settings: Settings) -> GraphStatusOut:
    from app.llm import router_llm_available

    job = graph_job_state(owner_id=owner_id)
    counts = graph_counts(db, owner_id=owner_id)
    pending_extract = int(counts.get("pending_extract") or 0)
    extract_on = router_llm_available(settings)
    if not extract_on:
        pending_extract = 0
    return GraphStatusOut(
        enabled=bool(settings.graph_enabled),
        extract_enabled=extract_on,
        running=bool(job["running"]),
        started=started,
        documents=counts["documents"],
        pending_docs=counts["pending_docs"],
        built_docs=counts["built_docs"],
        entities=counts["entities"],
        edges=counts["edges"],
        mentions=counts["mentions"],
        stale_mentions=counts["stale_mentions"],
        pending_extract=pending_extract,
        skipped_chunks=int(job.get("skipped_chunks") or 0),
        extract_calls=int(job.get("extract_calls") or 0),
        extract_failed=int(job.get("extract_failed") or 0),
        extract_empty=int(job.get("extract_empty") or 0),
        extract_dropped=int(job.get("extract_dropped") or 0),
        concepts=int(counts.get("concepts") or 0),
        source=str(job.get("source") or "rules"),
        error=job["error"] if isinstance(job.get("error"), str) else None,
        note=job["note"] if isinstance(job.get("note"), str) else None,
    )


@router.get("/graph", response_model=GraphStatusOut)
def graph_status(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> GraphStatusOut:
    uid = current_owner(user)
    return _status(db, owner_id=uid, settings=settings)


class GraphStartIn(BaseModel):
    force: bool = False
    extract: bool = False
    doc_ids: list[UUID] = Field(default_factory=list)


@router.post("/graph", response_model=GraphStatusOut)
def graph_start(
    payload: GraphStartIn | None = None,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit_jobs("graph")),
) -> GraphStatusOut:
    uid = current_owner(user)
    started = False
    force = bool(payload and payload.force)
    doc_ids = list(payload.doc_ids or []) if payload else []
    if settings.graph_enabled:
        pending = count_pending(db, owner_id=uid)
        counts = graph_counts(db, owner_id=uid)
        rebuild = counts["stale_mentions"] > 0 and pending == 0
        want_extract = bool(payload and payload.extract) and bool(doc_ids)
        if doc_ids:
            if uid is not None:
                owned = set(
                    db.scalars(
                        select(Document.id).where(
                            Document.id.in_(doc_ids),
                            Document.owner_id == uid,
                        )
                    ).all()
                )
                if any(item not in owned for item in doc_ids):
                    raise HTTPException(status_code=404, detail="文档不存在")
                doc_ids = list(owned)
            started = start_graph_job(
                all_docs=False,
                extract=want_extract,
                doc_ids=doc_ids,
                owner_id=uid,
            )
        elif (pending > 0 or rebuild or force) and not want_extract:
            started = start_graph_job(all_docs=rebuild or force, extract=False, owner_id=uid)
    return _status(db, started=started, owner_id=uid, settings=settings)


class GraphDocOut(BaseModel):
    doc_id: str
    filename: str
    skeleton: bool = False
    extracted: bool = False
    chars: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    skip: str = ""
    atx: bool = False


class GraphCatalogOut(BaseModel):
    documents: list[GraphDocOut]
    stale: bool = False


@router.get("/graph/catalog", response_model=GraphCatalogOut)
def graph_catalog(
    collection_id: UUID | None = None,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> GraphCatalogOut:
    from app.graph.service import list_graph_docs

    uid = current_owner(user)
    rows = list_graph_docs(db, owner_id=uid, settings=settings, collection_id=collection_id)
    stale = False
    if collection_id is not None:
        owned = db.execute(select(Document).where(Document.collection_id == collection_id)).scalars().all()
        if uid is not None:
            owned = [doc for doc in owned if doc.owner_id == uid]
        stale = any(
            doc.graph_extracted_at is not None and doc.graph_collection_id != doc.collection_id for doc in owned
        )
    return GraphCatalogOut(documents=[GraphDocOut(**row) for row in rows], stale=stale)


class GraphMentionOut(BaseModel):
    chunk_id: str
    quote: str
    start_char: int
    end_char: int
    doc_id: str = ""


class GraphEntityOut(BaseModel):
    id: str
    name: str
    kind: str
    mentions: list[GraphMentionOut]


class GraphEdgeOut(BaseModel):
    id: str
    relation: str
    src_id: str
    dst_id: str
    src_name: str
    dst_name: str


class DocumentGraphOut(BaseModel):
    doc_id: str
    filename: str
    entities: list[GraphEntityOut]
    edges: list[GraphEdgeOut]
    source: str = "rules"


@router.get("/graph/documents/{doc_id}", response_model=DocumentGraphOut)
def graph_document(
    doc_id: UUID,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> DocumentGraphOut:
    if not settings.graph_enabled:
        raise HTTPException(status_code=404, detail="graph disabled")
    payload = document_graph(db, doc_id, owner_id=current_owner(user))
    if payload is None:
        raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
    return payload


class NeighborhoodOut(BaseModel):
    seed_id: str
    hops: int
    entities: list[GraphEntityOut]
    edges: list[GraphEdgeOut]
    source: str = "mixed"


@router.get("/graph/neighborhood", response_model=NeighborhoodOut)
def graph_neighborhood(
    entity_id: UUID | None = None,
    name: str = "",
    hops: int = 2,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> NeighborhoodOut:
    if not settings.graph_enabled:
        raise HTTPException(status_code=404, detail="graph disabled")
    if entity_id is None and not str(name).strip():
        raise HTTPException(status_code=400, detail="entity_id or name is required")
    payload = entity_neighborhood(
        db,
        entity_id=entity_id,
        name=name,
        hops=hops,
        owner_id=current_owner(user),
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="entity not found")
    return payload


class RelatedConceptOut(BaseModel):
    id: str = ""
    name: str
    kind: str


class RelatedEdgeOut(BaseModel):
    relation: str
    src_name: str
    dst_name: str
    label: str = ""


class RelatedOut(BaseModel):
    concepts: list[RelatedConceptOut]
    edges: list[RelatedEdgeOut]


@router.get("/graph/related", response_model=RelatedOut)
def graph_related(
    chunk_ids: str = "",
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> RelatedOut:
    if not settings.graph_enabled:
        return RelatedOut(concepts=[], edges=[])
    from app.graph.walk import related_concepts

    ids = [piece.strip() for piece in (chunk_ids or "").split(",") if piece.strip()]
    payload = related_concepts(db, ids, owner_id=current_owner(user))
    return RelatedOut(
        concepts=[RelatedConceptOut(**item) for item in payload.get("concepts") or []],
        edges=[RelatedEdgeOut(**item) for item in payload.get("edges") or []],
    )
