"""Graph pipeline: pending docs → rules drafts → upsert → prune."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from app.graph.order import entity_doc_pos, entity_sort_key
from app.graph.rules import build_document_graph
from app.graph.schema import (
    KINDS_ALL,
    KINDS_EXTRACT,
    RELATIONS_ALL,
    DraftEdge,
    DraftEntity,
    GraphDrafts,
    is_markdown_filename,
    norm_name,
    quote_in_chunk,
)
from app.models import Chunk, Document, GraphEdge, GraphEntity, GraphMention


@dataclass
class GraphRunStats:
    documents: int = 0
    built: int = 0
    skipped: int = 0
    entities: int = 0
    edges: int = 0
    mentions: int = 0
    skipped_chunks: int = 0
    extract_calls: int = 0
    extract_failed: int = 0
    extract_empty: int = 0
    extract_dropped: int = 0
    note: str = ""
    source: str = "rules"


def _chunked_filenames(db: Session, *, owner_id: UUID | None = None) -> list[tuple[UUID, str]]:
    stmt = select(Document.id, Document.filename).where(
        Document.id.in_(select(Chunk.doc_id).distinct()),
    )
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    return [(row[0], row[1]) for row in db.execute(stmt).all()]


def _built_doc_ids(db: Session, *, owner_id: UUID | None = None) -> set[UUID]:
    if owner_id is None:
        return {row[0] for row in db.execute(select(GraphMention.doc_id).distinct()).all()}
    return {
        row[0]
        for row in db.execute(
            select(GraphMention.doc_id)
            .join(Document, Document.id == GraphMention.doc_id)
            .where(Document.owner_id == owner_id)
            .distinct()
        ).all()
    }


def count_pending(db: Session, *, owner_id: UUID | None = None) -> int:
    built = _built_doc_ids(db, owner_id=owner_id)
    return sum(1 for doc_id, _name in _chunked_filenames(db, owner_id=owner_id) if doc_id not in built)


def count_pending_extract(db: Session, *, owner_id: UUID | None = None) -> int:
    """Documents with a skeleton graph that have not finished an LLM extract pass."""
    built = _built_doc_ids(db, owner_id=owner_id)
    stmt = select(Document.id, Document.filename).where(
        Document.id.in_(select(Chunk.doc_id).distinct()),
        Document.graph_extracted_at.is_(None),
    )
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    return sum(1 for doc_id, _name in db.execute(stmt).all() if doc_id in built)


def count_stale_mentions(db: Session, *, owner_id: UUID | None = None) -> int:
    stmt = (
        select(func.count())
        .select_from(GraphMention)
        .outerjoin(Chunk, Chunk.id == GraphMention.chunk_id)
        .where(or_(Chunk.id.is_(None), Chunk.chunk_hash != GraphMention.chunk_hash))
    )
    if owner_id is not None:
        stmt = stmt.join(Document, Document.id == GraphMention.doc_id).where(Document.owner_id == owner_id)
    return int(db.scalar(stmt) or 0)


def graph_counts(db: Session, *, owner_id: UUID | None = None) -> dict[str, int]:
    ent = select(func.count(GraphEntity.id))
    edge = select(func.count(GraphEdge.id))
    men = select(func.count(GraphMention.id))
    if owner_id is not None:
        ent = ent.where(GraphEntity.owner_id == owner_id)
        edge = edge.where(GraphEdge.owner_id == owner_id)
        men = men.join(Document, Document.id == GraphMention.doc_id).where(Document.owner_id == owner_id)
    pending = count_pending(db, owner_id=owner_id)
    total_docs = sum(1 for _doc_id, _name in _chunked_filenames(db, owner_id=owner_id))
    concepts = select(func.count(GraphEntity.id)).where(GraphEntity.kind.in_(tuple(KINDS_EXTRACT)))
    if owner_id is not None:
        concepts = concepts.where(GraphEntity.owner_id == owner_id)
    return {
        "documents": total_docs,
        "pending_docs": pending,
        "built_docs": max(total_docs - pending, 0),
        "entities": int(db.scalar(ent) or 0),
        "edges": int(db.scalar(edge) or 0),
        "mentions": int(db.scalar(men) or 0),
        "stale_mentions": count_stale_mentions(db, owner_id=owner_id),
        "pending_extract": count_pending_extract(db, owner_id=owner_id),
        "concepts": int(db.scalar(concepts) or 0),
    }


def _mention_dict(item: GraphMention) -> dict:
    return {
        "chunk_id": str(item.chunk_id),
        "doc_id": str(item.doc_id),
        "quote": item.quote,
        "start_char": item.start_char,
        "end_char": item.end_char,
    }


def document_graph(db: Session, doc_id: UUID, *, owner_id: UUID | None = None) -> dict | None:
    """Entities and edges evidenced in one document. None if the doc is missing / not owned."""
    doc = db.get(Document, doc_id)
    if doc is None or (owner_id is not None and doc.owner_id != owner_id):
        return None
    mentions = list(db.execute(select(GraphMention).where(GraphMention.doc_id == doc_id)).scalars().all())
    entity_ids = {row.entity_id for row in mentions}
    entities = []
    if entity_ids:
        entities = list(db.execute(select(GraphEntity).where(GraphEntity.id.in_(entity_ids))).scalars().all())
    by_id = {row.id: row for row in entities}
    mentions_by_entity: dict[UUID, list[GraphMention]] = {}
    for row in mentions:
        mentions_by_entity.setdefault(row.entity_id, []).append(row)
    chunk_rows = list(db.execute(select(Chunk).where(Chunk.doc_id == doc_id)).scalars().all())
    chunk_pos = {row.id: (int(row.page or 0), int(row.start_offset or 0)) for row in chunk_rows}
    edge_ids = {row.edge_id for row in mentions if row.edge_id}
    edges = []
    if edge_ids:
        edges = list(db.execute(select(GraphEdge).where(GraphEdge.id.in_(edge_ids))).scalars().all())
    entity_out = _serialize_entities(entities, mentions_by_entity, chunk_pos)
    edge_out = _serialize_edges(edges, by_id)
    return {
        "doc_id": str(doc.id),
        "filename": doc.filename,
        "entities": entity_out,
        "edges": edge_out,
        "source": _doc_source(edges, entities),
    }


def entity_neighborhood(
    db: Session,
    *,
    entity_id: UUID | None = None,
    name: str = "",
    hops: int = 2,
    owner_id: UUID | None = None,
) -> dict | None:
    """Undirected neighborhood around one entity. None if the seed is missing / not owned."""
    from app.graph.query import neighbor_entity_ids
    from app.graph.schema import norm_name

    seed = None
    if entity_id is not None:
        seed = db.get(GraphEntity, entity_id)
    elif name.strip():
        key = norm_name(name)
        stmt = select(GraphEntity).where(GraphEntity.norm_name == key)
        if owner_id is None:
            stmt = stmt.where(GraphEntity.owner_id.is_(None))
        else:
            stmt = stmt.where(GraphEntity.owner_id == owner_id)
        seed = db.execute(stmt).scalars().first()
    if seed is None:
        return None
    if owner_id is not None and seed.owner_id != owner_id:
        return None
    hops = max(1, min(int(hops or 2), 3))
    ids = neighbor_entity_ids(db, [seed.id], hops=hops, owner_id=owner_id)[:80]
    if seed.id not in ids:
        ids = [seed.id, *ids]
    entities = list(db.execute(select(GraphEntity).where(GraphEntity.id.in_(ids))).scalars().all())
    by_id = {row.id: row for row in entities}
    kept = set(by_id)
    edges = list(
        db.execute(select(GraphEdge).where(GraphEdge.src_id.in_(kept), GraphEdge.dst_id.in_(kept))).scalars().all()
    )
    if owner_id is None:
        edges = [row for row in edges if row.owner_id is None]
    else:
        edges = [row for row in edges if row.owner_id == owner_id]
    mentions = list(db.execute(select(GraphMention).where(GraphMention.entity_id.in_(kept))).scalars().all())
    mentions_by_entity: dict[UUID, list[GraphMention]] = {}
    for row in mentions:
        mentions_by_entity.setdefault(row.entity_id, []).append(row)
    chunk_ids = {row.chunk_id for row in mentions if row.chunk_id}
    chunk_pos: dict[UUID, tuple[int, int]] = {}
    if chunk_ids:
        chunk_rows = list(db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids))).scalars().all())
        chunk_pos = {row.id: (int(row.page or 0), int(row.start_offset or 0)) for row in chunk_rows}
    return {
        "seed_id": str(seed.id),
        "hops": hops,
        "entities": _serialize_entities(entities, mentions_by_entity, chunk_pos),
        "edges": _serialize_edges(edges, by_id),
        "source": _doc_source(edges, entities),
    }


def _serialize_entities(
    entities: list[GraphEntity],
    mentions_by_entity: dict[UUID, list[GraphMention]],
    chunk_pos: dict[UUID, tuple[int, int]] | None = None,
) -> list[dict]:
    chunk_pos = chunk_pos or {}
    entity_out = []
    for ent in sorted(
        entities,
        key=lambda item: entity_sort_key(item, mentions_by_entity.get(item.id) or [], chunk_pos),
    ):
        rows = mentions_by_entity.get(ent.id) or []
        preferred = [item for item in rows if item.edge_id is None] or rows
        pos = entity_doc_pos(kind=ent.kind, mentions=preferred, chunk_pos=chunk_pos)
        entity_out.append(
            {
                "id": str(ent.id),
                "name": ent.name,
                "kind": ent.kind,
                "doc_pos": {"page": pos[1], "offset": pos[2], "start_char": pos[3]},
                "mentions": [_mention_dict(item) for item in preferred],
            }
        )
    return entity_out


def _serialize_edges(edges: list[GraphEdge], by_id: dict[UUID, GraphEntity]) -> list[dict]:
    edge_out = []
    for edge in edges:
        src = by_id.get(edge.src_id)
        dst = by_id.get(edge.dst_id)
        if src is None or dst is None:
            continue
        edge_out.append(
            {
                "id": str(edge.id),
                "relation": edge.relation,
                "src_id": str(edge.src_id),
                "dst_id": str(edge.dst_id),
                "src_name": src.name,
                "dst_name": dst.name,
            }
        )
    return edge_out


def _doc_source(edges, entities) -> str:
    kinds = {item.kind for item in entities}
    sources = {item.source for item in edges}
    if kinds & KINDS_EXTRACT or "llm" in sources:
        return "mixed" if "rules" in sources or (kinds & {"document", "doc_section"}) else "llm"
    return "rules"


def prune_orphans(db: Session, *, owner_id: UUID | None = None) -> None:
    _recompute_counts(db, owner_id=owner_id)
    edge_q = delete(GraphEdge).where(GraphEdge.evidence_count <= 0)
    ent_q = delete(GraphEntity).where(GraphEntity.mention_count <= 0)
    if owner_id is not None:
        edge_q = edge_q.where(GraphEdge.owner_id == owner_id)
        ent_q = ent_q.where(GraphEntity.owner_id == owner_id)
    db.execute(edge_q)
    db.execute(ent_q)
    db.flush()


def run_graph_pipeline(
    db: Session,
    *,
    owner_id: UUID | None = None,
    all_docs: bool = False,
    extract: bool = False,
    doc_ids: list | None = None,
    settings=None,
) -> GraphRunStats:
    from app.core.config import get_settings
    from app.llm import router_llm_available

    settings = settings or get_settings()
    prune_orphans(db, owner_id=owner_id)
    stats = GraphRunStats()
    extract_ok = bool(extract) and router_llm_available(settings) and bool(doc_ids)
    docs = _load_docs(
        db,
        owner_id=owner_id,
        pending_only=not all_docs and not doc_ids,
        include_extract=extract_ok,
        doc_ids=doc_ids or None,
    )
    stats.documents = len(docs)
    notes: list[str] = []
    sources: set[str] = set()
    empty_extract_docs = 0
    wrote_concepts = 0
    for document in docs:
        chunks = list(document.chunks)
        if not chunks:
            stats.skipped += 1
            continue
        if getattr(document, "graph_extracted_at", None) is not None:
            stats.skipped += 1
            continue
        _clear_doc_mentions(db, document.id)
        drafts = GraphDrafts()
        drafts.extend(build_document_graph(document, chunks))
        if extract_ok:
            from app.graph.extract import NO_KEY_NOTE, SKIP_SHORT_NOTE, extract_document, llm_wrote_entities
            from app.graph.resolve import resolve_drafts

            extracted = extract_document(document, chunks, settings, force=False)
            stats.skipped_chunks += extracted.skipped
            stats.extract_calls += extracted.calls
            stats.extract_failed += extracted.failed
            stats.extract_empty += extracted.empty
            stats.extract_dropped += extracted.dropped
            if extracted.note:
                notes.append(extracted.note)
            resolved = resolve_drafts(extracted)
            llm_wrote = llm_wrote_entities(extracted)
            if SKIP_SHORT_NOTE in (extracted.note or ""):
                document.graph_extracted_at = datetime.now(timezone.utc)
            elif llm_wrote:
                wrote_concepts += 1
                document.graph_extracted_at = datetime.now(timezone.utc)
            elif NO_KEY_NOTE in (extracted.note or ""):
                pass
            else:
                empty_extract_docs += 1
            drafts.extend(resolved)
        elif drafts.note:
            notes.append(drafts.note)
        sources.add(drafts.source)
        upsert_drafts(
            db,
            drafts,
            owner_id=document.owner_id,
            doc_id=document.id,
            collection_id=getattr(document, "collection_id", None),
        )
        document.graph_collection_id = getattr(document, "collection_id", None)
        stats.built += 1
        db.commit()
    if extract_ok and empty_extract_docs and not wrote_concepts:
        notes.append(f"抽取未写入概念（{empty_extract_docs} 篇）")
    prune_orphans(db, owner_id=owner_id)
    counts = graph_counts(db, owner_id=owner_id)
    stats.entities = counts["entities"]
    stats.edges = counts["edges"]
    stats.mentions = counts["mentions"]
    stats.note = "；".join(dict.fromkeys(notes))
    if "llm" in sources and "rules" in sources:
        stats.source = "mixed"
    elif "llm" in sources:
        stats.source = "llm"
    db.commit()
    return stats


def upsert_drafts(
    db: Session,
    drafts,
    *,
    owner_id: UUID | None,
    doc_id: UUID,
    collection_id: UUID | None = None,
) -> None:
    chunk_map = _chunk_map(db, doc_id)
    entity_ids: dict[tuple[str, str], UUID] = {}
    for item in drafts.entities:
        stored = _upsert_entity(
            db,
            item,
            owner_id=owner_id,
            chunk_map=chunk_map,
            doc_id=doc_id,
            collection_id=collection_id,
        )
        if stored is not None:
            entity_ids[(item.kind, norm_name(item.name))] = stored
    for item in drafts.edges:
        _upsert_edge(db, item, owner_id=owner_id, entity_ids=entity_ids, chunk_map=chunk_map, doc_id=doc_id)
    _recompute_counts(db, owner_id=owner_id)
    db.flush()


def list_graph_docs(
    db: Session, *, owner_id: UUID | None = None, settings=None, collection_id: UUID | None = None
) -> list[dict]:
    """Per-document skeleton / extract status plus a token ceiling for the UI."""
    from app.graph.extract import estimate_extract

    stmt = select(Document).options(selectinload(Document.chunks)).order_by(Document.created_at, Document.filename)
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    if collection_id is not None:
        stmt = stmt.where(Document.collection_id == collection_id)
    rows = list(db.execute(stmt).scalars().unique().all())
    built = {row[0] for row in db.execute(select(GraphMention.doc_id).distinct()).all()}
    out = []
    for doc in rows:
        chunks = list(doc.chunks)
        est = estimate_extract(doc, chunks, settings)
        out.append(
            {
                "doc_id": str(doc.id),
                "filename": doc.filename,
                "skeleton": doc.id in built,
                "extracted": bool(doc.graph_extracted_at),
                "chars": int(est["chars"]),
                "calls": int(est["calls"]),
                "input_tokens": int(est["input_tokens"]),
                "output_tokens": int(est["output_tokens"]),
                "total_tokens": int(est["total_tokens"]),
                "skip": str(est.get("skip") or ""),
                "atx": is_markdown_filename(doc.filename),
            }
        )
    return out


def _load_docs(
    db: Session,
    *,
    owner_id: UUID | None,
    pending_only: bool,
    include_extract: bool = False,
    doc_ids: list | None = None,
) -> list[Document]:
    stmt = select(Document).options(selectinload(Document.chunks)).order_by(Document.created_at, Document.filename)
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    rows = list(db.execute(stmt).scalars().unique().all())
    if doc_ids:
        wanted = {item if isinstance(item, UUID) else UUID(str(item)) for item in doc_ids}
        return [doc for doc in rows if doc.id in wanted and doc.chunks]
    if not pending_only:
        return [doc for doc in rows if doc.chunks and (not include_extract or doc.graph_extracted_at is None)]
    built = {row[0] for row in db.execute(select(GraphMention.doc_id).distinct()).all()}
    if owner_id is not None:
        built = {
            row[0]
            for row in db.execute(
                select(GraphMention.doc_id)
                .join(Document, Document.id == GraphMention.doc_id)
                .where(Document.owner_id == owner_id)
                .distinct()
            ).all()
        }
    extracted_pending: set[UUID] = set()
    if include_extract:
        extracted_pending = {doc.id for doc in rows if doc.graph_extracted_at is None and doc.chunks}
    out = []
    for doc in rows:
        if not doc.chunks:
            continue
        if doc.id not in built:
            out.append(doc)
            continue
        if include_extract and doc.id in extracted_pending:
            out.append(doc)
    return out


def _clear_doc_mentions(db: Session, doc_id: UUID) -> None:
    db.execute(delete(GraphMention).where(GraphMention.doc_id == doc_id))
    db.flush()


def _chunk_map(db: Session, doc_id: UUID) -> dict[str, Chunk]:
    rows = db.execute(select(Chunk).where(Chunk.doc_id == doc_id)).scalars().all()
    return {str(row.id): row for row in rows}


def _upsert_entity(
    db: Session,
    item: DraftEntity,
    *,
    owner_id: UUID | None,
    chunk_map: dict[str, Chunk],
    doc_id: UUID,
    collection_id: UUID | None = None,
) -> UUID | None:
    chunk = chunk_map.get(item.chunk_id)
    if chunk is None:
        return None
    span = quote_in_chunk(item.quote, chunk.text or "")
    if span is None:
        return None
    key = norm_name(item.name)
    if not key:
        return None
    kind = item.kind if item.kind in KINDS_ALL else "concept"
    row = _get_entity(db, owner_id=owner_id, kind=kind, key=key, collection_id=collection_id)
    if row is None:
        row = GraphEntity(
            owner_id=owner_id,
            collection_id=collection_id,
            name=item.name,
            norm_name=key,
            kind=kind,
            aliases=list(item.aliases or []),
            mention_count=0,
        )
        db.add(row)
        db.flush()
    elif item.aliases:
        merged = list(dict.fromkeys([*(row.aliases or []), *item.aliases]))
        row.aliases = merged
    _add_mention(
        db,
        entity_id=row.id,
        edge_id=None,
        doc_id=doc_id,
        chunk=chunk,
        quote=(chunk.text or "")[span[0] : span[1]],
        start=span[0],
        end=span[1],
    )
    return row.id


def _upsert_edge(
    db: Session,
    item: DraftEdge,
    *,
    owner_id: UUID | None,
    entity_ids: dict[tuple[str, str], UUID],
    chunk_map: dict[str, Chunk],
    doc_id: UUID,
) -> None:
    if item.relation not in RELATIONS_ALL:
        return
    chunk = chunk_map.get(item.chunk_id)
    if chunk is None:
        return
    span = quote_in_chunk(item.quote, chunk.text or "")
    if span is None:
        return
    src = entity_ids.get((item.src_kind, norm_name(item.src_name)))
    dst = entity_ids.get((item.dst_kind, norm_name(item.dst_name)))
    if src is None:
        src_ent = _get_entity(db, owner_id=owner_id, kind=item.src_kind, key=norm_name(item.src_name))
        src = None if src_ent is None else src_ent.id
    if dst is None:
        dst_ent = _get_entity(db, owner_id=owner_id, kind=item.dst_kind, key=norm_name(item.dst_name))
        dst = None if dst_ent is None else dst_ent.id
    if src is None or dst is None or src == dst:
        return
    row = _get_edge(db, owner_id=owner_id, src_id=src, dst_id=dst, relation=item.relation)
    if row is None:
        row = GraphEdge(
            owner_id=owner_id,
            src_id=src,
            dst_id=dst,
            relation=item.relation,
            confidence=1.0,
            evidence_count=0,
            source=item.source or "rules",
        )
        db.add(row)
        db.flush()
    _add_mention(
        db,
        entity_id=src,
        edge_id=row.id,
        doc_id=doc_id,
        chunk=chunk,
        quote=(chunk.text or "")[span[0] : span[1]],
        start=span[0],
        end=span[1],
    )


def _get_entity(
    db: Session,
    *,
    owner_id: UUID | None,
    kind: str,
    key: str,
    collection_id: UUID | None = None,
) -> GraphEntity | None:
    stmt = select(GraphEntity).where(GraphEntity.kind == kind, GraphEntity.norm_name == key)
    if owner_id is None:
        stmt = stmt.where(GraphEntity.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEntity.owner_id == owner_id)
    if collection_id is None:
        stmt = stmt.where(GraphEntity.collection_id.is_(None))
    else:
        stmt = stmt.where(GraphEntity.collection_id == collection_id)
    return db.execute(stmt).scalars().first()


def _get_edge(
    db: Session,
    *,
    owner_id: UUID | None,
    src_id: UUID,
    dst_id: UUID,
    relation: str,
) -> GraphEdge | None:
    stmt = select(GraphEdge).where(
        GraphEdge.src_id == src_id,
        GraphEdge.dst_id == dst_id,
        GraphEdge.relation == relation,
    )
    if owner_id is None:
        stmt = stmt.where(GraphEdge.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEdge.owner_id == owner_id)
    return db.execute(stmt).scalars().first()


def _add_mention(
    db: Session,
    *,
    entity_id: UUID,
    edge_id: UUID | None,
    doc_id: UUID,
    chunk: Chunk,
    quote: str,
    start: int,
    end: int,
) -> None:
    if edge_id is None:
        stmt = select(GraphMention.id).where(
            GraphMention.entity_id == entity_id,
            GraphMention.chunk_id == chunk.id,
            GraphMention.edge_id.is_(None),
        )
    else:
        stmt = select(GraphMention.id).where(
            GraphMention.entity_id == entity_id,
            GraphMention.chunk_id == chunk.id,
            GraphMention.edge_id == edge_id,
        )
    if db.execute(stmt).first():
        return
    db.add(
        GraphMention(
            edge_id=edge_id,
            entity_id=entity_id,
            doc_id=doc_id,
            chunk_id=chunk.id,
            chunk_hash=chunk.chunk_hash or "",
            quote=quote,
            start_char=start,
            end_char=end,
        )
    )


def _recompute_counts(db: Session, *, owner_id: UUID | None = None) -> None:
    mention_sub = (
        select(GraphMention.entity_id, func.count(GraphMention.id).label("n"))
        .group_by(GraphMention.entity_id)
        .subquery()
    )
    db.execute(
        update(GraphEntity).values(
            mention_count=func.coalesce(
                select(mention_sub.c.n).where(mention_sub.c.entity_id == GraphEntity.id).scalar_subquery(),
                0,
            )
        )
    )
    edge_sub = (
        select(GraphMention.edge_id, func.count(GraphMention.id).label("n"))
        .where(GraphMention.edge_id.is_not(None))
        .group_by(GraphMention.edge_id)
        .subquery()
    )
    db.execute(
        update(GraphEdge).values(
            evidence_count=func.coalesce(
                select(edge_sub.c.n).where(edge_sub.c.edge_id == GraphEdge.id).scalar_subquery(),
                0,
            )
        )
    )
    db.flush()
