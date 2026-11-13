"""Rule router. One lexical pass over library names and card topics."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.classify import has_strong_synthesis
from app.corpus.library import list_collections, profile_text, score_assignment
from app.corpus.scope import Scope
from app.models import Document
from app.models.collection import DocumentCard


def resolve_scope(
    db: Session,
    owner_id,
    goal: str,
    *,
    mode: str | None = None,
    collection_ids: list[str] | None = None,
    pinned: dict | None = None,
) -> Scope:
    requested = (mode or "auto").strip().lower()
    collections = {str(row.id): row for row in list_collections(db, owner_id)}
    if requested in {"all", "soft", "hard"}:
        ids = [item for item in (collection_ids or []) if item in collections]
        labels = tuple(collections[item].name for item in ids)
        if requested == "hard" and len(ids) == 1:
            return Scope(mode="hard", collection_ids=(ids[0],), labels=labels)
        if requested == "soft" and ids:
            ordered = tuple(sorted(ids))
            labels = tuple(collections[item].name for item in ordered)
            return Scope(mode="soft", collection_ids=ordered, labels=labels)
        return Scope(mode="all")
    if pinned:
        scope = Scope.from_dict(pinned)
        if scope.mode != "all":
            return scope
    if not collections:
        return Scope(mode="all")
    return _route(db, list(collections.values()), goal)


def _route(db: Session, collections, goal: str) -> Scope:
    ranked = []
    for collection in collections:
        member_ids = list(
            db.execute(select(Document.id).where(Document.collection_id == collection.id)).scalars().all()
        )
        cards = []
        if member_ids:
            cards = list(db.execute(select(DocumentCard).where(DocumentCard.doc_id.in_(member_ids))).scalars().all())
        ranked.append((score_assignment(goal or "", profile_text(collection, cards)), collection))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < 0.55:
        return Scope(mode="all")
    top_score, top = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    compare = has_strong_synthesis(goal or "") and len(ranked) > 1 and ranked[1][0] >= 0.35
    if compare:
        chosen = [top, ranked[1][1]]
        ids = tuple(sorted(str(item.id) for item in chosen))
        labels = tuple(next(item.name for item in chosen if str(item.id) == key) for key in ids)
        return Scope(mode="soft", collection_ids=ids, labels=labels)
    if len(ranked) == 1 or top_score - second >= 0.15:
        return Scope(mode="soft", collection_ids=(str(top.id),), labels=(top.name,))
    return Scope(mode="all")
