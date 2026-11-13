"""Collections, assignment, inbox metrics, and scope logs."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.corpus.cards import ASSIGN_FLOOR, ASSIGN_GAP
from app.corpus.relevance import coverage, jaccard, lexical_keys
from app.corpus.scope import Scope
from app.models import Document
from app.models.collection import (
    AssignBatch,
    AssignmentLog,
    Collection,
    DocumentCard,
    ScopeLog,
)

INBOX_WATERMARK = 0.4


def _owner(stmt, owner_id):
    if owner_id is None:
        return stmt.where(Collection.owner_id.is_(None))
    return stmt.where(Collection.owner_id == owner_id)


def list_collections(db: Session, owner_id) -> list[Collection]:
    stmt = select(Collection).order_by(Collection.created_at, Collection.name)
    if owner_id is None:
        stmt = stmt.where(Collection.owner_id.is_(None))
    else:
        stmt = stmt.where(Collection.owner_id == owner_id)
    return list(db.execute(stmt).scalars().all())


def collection_counts(db: Session, owner_id) -> dict[str, int]:
    stmt = select(Document.collection_id, func.count()).group_by(Document.collection_id)
    if owner_id is None:
        stmt = stmt.where(Document.owner_id.is_(None))
    else:
        stmt = stmt.where(Document.owner_id == owner_id)
    counts: dict[str, int] = {}
    for collection_id, count in db.execute(stmt).all():
        counts[str(collection_id) if collection_id else ""] = int(count or 0)
    return counts


def create_collection(db: Session, *, owner_id, name: str, description: str = "") -> Collection:
    label = (name or "").strip()
    if not label:
        raise ValueError("库名不能为空")
    row = Collection(owner_id=owner_id, name=label[:80], description=(description or "").strip()[:240])
    db.add(row)
    db.flush()
    rescore_against(db, owner_id, row)
    return row


def profile_text(collection: Collection, cards: list[DocumentCard]) -> str:
    topics = []
    for card in cards:
        topics.extend(str(item) for item in (card.topics or []))
    return " ".join([collection.name, collection.description or "", *topics[:40]])


def score_assignment(card_text: str, profile: str) -> float:
    left = lexical_keys(card_text)
    right = lexical_keys(profile)
    if not left or not right:
        return 0.0
    return max(coverage(left, right), jaccard(left, right))


def auto_assign(db: Session, document: Document, card: DocumentCard) -> None:
    """Place a card into an existing library when the lead is clear. Never creates one."""
    if card.assign_source == "user":
        return
    collections = list_collections(db, document.owner_id)
    if not collections or not (card.topics or card.display_title):
        card.assign_source = ""
        card.assign_score = 0.0
        card.assign_gap = 0.0
        document.collection_id = None
        return
    card_text = " ".join([card.display_title, card.summary, " ".join(card.topics or [])])
    ranked = _rank(db, collections, card_text)
    top_score, top = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    gap = top_score - second
    card.assign_score = round(top_score, 4)
    card.assign_gap = round(gap, 4)
    clear = top_score >= ASSIGN_FLOOR and (len(ranked) == 1 or gap >= ASSIGN_GAP)
    if clear:
        document.collection_id = top.id
        card.assign_source = "auto"
        card.mixed = False
    else:
        document.collection_id = None
        card.assign_source = ""
        card.mixed = len(ranked) > 1 and top_score >= ASSIGN_FLOOR and gap < ASSIGN_GAP


def _rank(db: Session, collections: list[Collection], card_text: str) -> list[tuple[float, Collection]]:
    ranked = []
    for collection in collections:
        member_ids = list(
            db.execute(select(Document.id).where(Document.collection_id == collection.id)).scalars().all()
        )
        cards = []
        if member_ids:
            cards = list(db.execute(select(DocumentCard).where(DocumentCard.doc_id.in_(member_ids))).scalars().all())
        ranked.append((score_assignment(card_text, profile_text(collection, cards)), collection))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def assign_document(
    db: Session,
    document: Document,
    *,
    collection_id: uuid.UUID | None,
    source: str,
    batch_id: uuid.UUID | None = None,
) -> DocumentCard:
    card = db.get(DocumentCard, document.id)
    if card is None:
        from app.corpus.cards import refresh_card

        card = refresh_card(db, document)
    if source != "user" and card.assign_source == "user":
        return card
    previous = document.collection_id
    document.collection_id = collection_id
    if source in {"user", "batch"}:
        card.assign_source = "user"
    elif not card.assign_source:
        card.assign_source = "auto"
    card.assign_version = int(card.assign_version or 1) + 1
    card.batch_id = batch_id
    card.mixed = False
    card.updated_at = datetime.now(timezone.utc)
    if previous != collection_id:
        db.add(AssignmentLog(owner_id=document.owner_id, doc_id=document.id, source=source))
        _stale_membership(db, document.owner_id, str(document.id))
    db.flush()
    return card


def _stale_membership(db: Session, owner_id, ingest_id: str) -> None:
    try:
        from app.memory.store import stale_on_ingest

        stale_on_ingest(db, owner_id, ingest_id=ingest_id)
    except Exception as exc:  # noqa: BLE001 — membership change still stands if memory tables are absent
        if "memory_items" not in str(exc) and "does not exist" not in str(exc).lower():
            raise


def rescore_against(db: Session, owner_id, collection: Collection) -> int:
    """Score the inbox and marginal automatic cards against one library."""
    stmt = select(Document)
    if owner_id is None:
        stmt = stmt.where(Document.owner_id.is_(None))
    else:
        stmt = stmt.where(Document.owner_id == owner_id)
    moved = 0
    profile = profile_text(collection, [])
    members = list(
        db.execute(
            select(DocumentCard)
            .join(Document, Document.id == DocumentCard.doc_id)
            .where(Document.collection_id == collection.id)
        )
        .scalars()
        .all()
    )
    profile = profile_text(collection, members)
    for document in db.execute(stmt).scalars().all():
        card = db.get(DocumentCard, document.id)
        if card is None or card.assign_source == "user":
            continue
        marginal = document.collection_id is None or float(card.assign_gap or 0) < ASSIGN_GAP
        if not marginal:
            continue
        card_text = " ".join([card.display_title, card.summary, " ".join(card.topics or [])])
        score = score_assignment(card_text, profile)
        current = float(card.assign_score or 0)
        if document.collection_id == collection.id:
            continue
        if score >= ASSIGN_FLOOR and score >= current + ASSIGN_GAP:
            document.collection_id = collection.id
            card.assign_source = "auto"
            card.assign_score = round(score, 4)
            card.assign_version = int(card.assign_version or 1) + 1
            moved += 1
            db.add(AssignmentLog(owner_id=owner_id, doc_id=document.id, source="auto"))
    if moved:
        _stale_membership(db, owner_id, str(collection.id))
    db.flush()
    return moved


def log_scope_search(
    db: Session, owner_id, scope: Scope, *, query: str, widened_docs: list[tuple] | None = None
) -> None:
    if scope.mode != "soft":
        return
    db.add(
        ScopeLog(
            owner_id=owner_id,
            mode="soft",
            widened=bool(widened_docs),
            query=(query or "")[:400],
            scope_key=scope.key(),
        )
    )
    for doc_id, collection_id in widened_docs or []:
        if collection_id is None:
            continue
        if str(collection_id) in scope.collection_ids:
            continue
        db.add(
            ScopeLog(
                owner_id=owner_id,
                mode="soft",
                widened=True,
                query=(query or "")[:400],
                doc_id=doc_id,
                doc_collection_id=collection_id,
                scope_key=scope.key(),
            )
        )
    db.flush()


def overview(db: Session, owner_id) -> dict:
    counts = collection_counts(db, owner_id)
    total = sum(counts.values())
    inbox = counts.get("", 0)
    collections = []
    for row in list_collections(db, owner_id):
        collections.append(
            {
                "id": str(row.id),
                "name": row.name,
                "description": row.description or "",
                "doc_count": counts.get(str(row.id), 0),
            }
        )
    since = datetime.now(timezone.utc) - timedelta(days=7)
    soft_stmt = (
        select(func.count()).select_from(ScopeLog).where(ScopeLog.created_at >= since, ScopeLog.doc_id.is_(None))
    )
    wide_stmt = (
        select(func.count())
        .select_from(ScopeLog)
        .where(ScopeLog.created_at >= since, ScopeLog.widened.is_(True), ScopeLog.doc_id.is_(None))
    )
    move_stmt = (
        select(func.count())
        .select_from(AssignmentLog)
        .where(AssignmentLog.created_at >= since, AssignmentLog.source == "user")
    )
    if owner_id is None:
        soft_stmt = soft_stmt.where(ScopeLog.owner_id.is_(None))
        wide_stmt = wide_stmt.where(ScopeLog.owner_id.is_(None))
        move_stmt = move_stmt.where(AssignmentLog.owner_id.is_(None))
    else:
        soft_stmt = soft_stmt.where(ScopeLog.owner_id == owner_id)
        wide_stmt = wide_stmt.where(ScopeLog.owner_id == owner_id)
        move_stmt = move_stmt.where(AssignmentLog.owner_id == owner_id)
    soft_n = int(db.execute(soft_stmt).scalar_one() or 0)
    wide_n = int(db.execute(wide_stmt).scalar_one() or 0)
    batch = _open_batch(db, owner_id)
    return {
        "collections": collections,
        "total": total,
        "inbox_count": inbox,
        "inbox_ratio": round(inbox / total, 4) if total else 0.0,
        "watermark": INBOX_WATERMARK,
        "fallback_rate": round(wide_n / soft_n, 4) if soft_n else 0.0,
        "fallback_7d": wide_n,
        "moves_7d": int(db.execute(move_stmt).scalar_one() or 0),
        "suggestions": _suggestions(db, owner_id),
        "open_batch_id": str(batch.id) if batch is not None else None,
    }


def _open_batch(db: Session, owner_id) -> AssignBatch | None:
    stmt = select(AssignBatch).where(AssignBatch.undone_at.is_(None)).order_by(AssignBatch.created_at.desc())
    if owner_id is None:
        stmt = stmt.where(AssignBatch.owner_id.is_(None))
    else:
        stmt = stmt.where(AssignBatch.owner_id == owner_id)
    return db.execute(stmt).scalars().first()


def _suggestions(db: Session, owner_id) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=30)
    stmt = select(ScopeLog).where(
        ScopeLog.created_at >= since, ScopeLog.doc_id.is_not(None), ScopeLog.widened.is_(True)
    )
    if owner_id is None:
        stmt = stmt.where(ScopeLog.owner_id.is_(None))
    else:
        stmt = stmt.where(ScopeLog.owner_id == owner_id)
    grouped: dict[str, dict] = {}
    for row in db.execute(stmt).scalars().all():
        key = str(row.doc_id)
        bucket = grouped.setdefault(key, {"doc_id": key, "count": 0, "collection_id": str(row.doc_collection_id or "")})
        bucket["count"] += 1
    out = [item for item in grouped.values() if item["count"] >= 2]
    out.sort(key=lambda item: item["count"], reverse=True)
    return out[:8]


def preview_organize(db: Session, owner_id) -> list[dict]:
    collections = list_collections(db, owner_id)
    if not collections:
        return []
    stmt = select(Document).where(Document.collection_id.is_(None))
    if owner_id is None:
        stmt = stmt.where(Document.owner_id.is_(None))
    else:
        stmt = stmt.where(Document.owner_id == owner_id)
    groups: dict[str, dict] = {}
    for document in db.execute(stmt).scalars().all():
        card = db.get(DocumentCard, document.id)
        if card is None:
            continue
        card_text = " ".join([card.display_title, card.summary, " ".join(card.topics or [])])
        ranked = _rank(db, collections, card_text)
        if not ranked or ranked[0][0] < ASSIGN_FLOOR:
            continue
        chosen = ranked[0][1]
        bucket = groups.setdefault(
            str(chosen.id),
            {"collection_id": str(chosen.id), "name": chosen.name, "documents": []},
        )
        bucket["documents"].append({"doc_id": str(document.id), "title": card.display_title or document.filename})
    return list(groups.values())


def accept_organize(db: Session, owner_id, assignments: list[dict]) -> AssignBatch:
    batch = AssignBatch(owner_id=owner_id, snapshot=[])
    db.add(batch)
    db.flush()
    snapshot = []
    for item in assignments:
        doc = db.get(Document, uuid.UUID(str(item["doc_id"])))
        if doc is None or (owner_id is not None and doc.owner_id != owner_id):
            continue
        card = db.get(DocumentCard, doc.id)
        snapshot.append(
            {
                "doc_id": str(doc.id),
                "collection_id": str(doc.collection_id) if doc.collection_id else None,
                "assign_source": card.assign_source if card else "",
                "version": int(card.assign_version or 1) if card else 1,
            }
        )
        target = item.get("collection_id")
        assign_document(
            db,
            doc,
            collection_id=uuid.UUID(str(target)) if target else None,
            source="batch",
            batch_id=batch.id,
        )
    batch.snapshot = snapshot
    db.flush()
    return batch


def undo_batch(db: Session, owner_id, batch_id: uuid.UUID) -> int:
    batch = db.get(AssignBatch, batch_id)
    if batch is None or batch.undone_at is not None:
        raise ValueError("批次不存在或已撤销")
    if owner_id is not None and batch.owner_id != owner_id:
        raise ValueError("批次不存在或已撤销")
    restored = 0
    for item in batch.snapshot or []:
        doc = db.get(Document, uuid.UUID(str(item["doc_id"])))
        card = db.get(DocumentCard, doc.id) if doc is not None else None
        if doc is None or card is None:
            continue
        if card.batch_id != batch.id or int(card.assign_version or 1) != int(item.get("version") or 0) + 1:
            continue
        previous = item.get("collection_id")
        doc.collection_id = uuid.UUID(previous) if previous else None
        card.assign_source = str(item.get("assign_source") or "")
        card.batch_id = None
        card.assign_version = int(card.assign_version or 1) + 1
        restored += 1
    batch.undone_at = datetime.now(timezone.utc)
    if restored:
        _stale_membership(db, owner_id, str(batch.id))
    db.flush()
    return restored
