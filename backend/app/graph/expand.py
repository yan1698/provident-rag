"""G2: expand retrieval along grounded graph edges. Ranking only, never a veto."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.generate.prompt import EvidenceItem


def attach_graph_candidates(
    hits: list[EvidenceItem],
    candidates: list[EvidenceItem],
    *,
    scores: dict[str, float] | None = None,
    min_score: float = 0.02,
    max_chars: int = 6000,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """Keep every hit. Add graph-role chunks that pass the rerank floor and fit the budget."""
    if not hits:
        return [], {"graph_added": 0, "graph_dropped": 0, "note": ""}
    hit_ids = {item.chunk_id for item in hits}
    floor = max(0.0, float(min_score))
    ranked = [
        item
        for item in candidates
        if item.chunk_id not in hit_ids and float((scores or {}).get(item.chunk_id, item.score or 0.0) or 0.0) >= floor
    ]
    ranked.sort(key=lambda item: -float((scores or {}).get(item.chunk_id, item.score or 0.0) or 0.0))
    used = sum(len(item.text or "") for item in hits)
    budget = max(used, int(max_chars))
    kept: list[EvidenceItem] = []
    dropped = 0
    for item in ranked:
        extra = len(item.text or "")
        if used + extra > budget:
            dropped += 1
            continue
        used += extra
        score = float((scores or {}).get(item.chunk_id, item.score or 0.0) or 0.0)
        kept.append(replace(item, role="graph", score=score, index=0))
    combined = list(hits) + kept
    out: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in combined:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        out.append(replace(item, index=len(out) + 1))
    note = ""
    if dropped:
        note = f"图谱扩展截断：保留命中块，丢弃 {dropped} 个关系邻居"
    elif kept:
        note = f"图谱扩展：补了 {len(kept)} 个关系邻居"
    return out, {"graph_added": len(kept), "graph_dropped": dropped, "note": note}


def expand_graph(
    db,
    evidence: list[EvidenceItem],
    *,
    goal: str,
    settings,
    owner_id=None,
    scope=None,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    from uuid import UUID

    from sqlalchemy import select

    from app.graph.query import neighbor_entity_ids, seed_entities_for_chunks
    from app.models import Chunk, GraphMention

    empty = {"graph_added": 0, "graph_dropped": 0, "note": ""}
    if not bool(getattr(settings, "graph_expand_enabled", False)):
        return list(evidence), empty
    if not bool(getattr(settings, "graph_enabled", True)):
        return list(evidence), empty
    hits = [item for item in evidence if getattr(item, "role", "hit") == "hit"]
    if not hits:
        return list(evidence), empty
    chunk_ids: list[UUID] = []
    for item in hits:
        try:
            chunk_ids.append(UUID(str(item.chunk_id)))
        except ValueError:
            continue
    seeds = seed_entities_for_chunks(db, chunk_ids, owner_id=owner_id)
    if not seeds:
        return list(evidence), empty
    hops = max(1, min(int(getattr(settings, "graph_expand_hops", 2) or 2), 3))
    neighbor_ids = neighbor_entity_ids(db, seeds, hops=hops, owner_id=owner_id)
    extra_ids = [item for item in neighbor_ids if item not in set(seeds)] or neighbor_ids
    mention_rows = list(
        db.execute(select(GraphMention).where(GraphMention.entity_id.in_(extra_ids or neighbor_ids))).scalars().all()
    )
    already = {item.chunk_id for item in evidence}
    want: list[UUID] = []
    for row in mention_rows:
        cid = str(row.chunk_id)
        if cid in already:
            continue
        want.append(row.chunk_id)
    if not want:
        return list(evidence), empty
    chunks = list(db.execute(select(Chunk).where(Chunk.id.in_(set(want)))).scalars().all())
    if owner_id is not None:
        from app.models import Document

        owned = {
            row[0]
            for row in db.execute(
                select(Document.id).where(
                    Document.id.in_({item.doc_id for item in chunks}),
                    Document.owner_id == owner_id,
                )
            ).all()
        }
        chunks = [item for item in chunks if item.doc_id in owned]
    if scope is not None and getattr(scope, "mode", "all") != "all" and chunks:
        from app.models import Document

        placed = {
            row[0]: row[1]
            for row in db.execute(
                select(Document.id, Document.collection_id).where(Document.id.in_({item.doc_id for item in chunks}))
            ).all()
        }
        chunks = [item for item in chunks if scope.allows(placed.get(item.doc_id))]
    candidates = [
        EvidenceItem(
            index=0,
            chunk_id=str(row.id),
            doc_id=str(row.doc_id),
            page=int(row.page or 1),
            text=(row.text or "").strip(),
            chunk_hash=row.chunk_hash or "",
            role="graph",
            start_offset=int(row.start_offset or 0),
        )
        for row in chunks
        if (row.text or "").strip()
    ]
    if not candidates:
        return list(evidence), empty
    scores = _rerank_texts(goal, candidates, settings)
    floor = float(getattr(settings, "agent_retrieve_min_score", 0.02) or 0.0)
    max_chars = max(800, int(getattr(settings, "qa_max_context_chars", 6000) or 6000))
    expanded, note = attach_graph_candidates(
        evidence,
        candidates,
        scores=scores,
        min_score=floor,
        max_chars=max_chars,
    )
    return expanded, note


def _rerank_texts(goal: str, items: list[EvidenceItem], settings) -> dict[str, float]:
    from app.rerank.model import rerank_scores

    passages = [item.text for item in items]
    try:
        raw = rerank_scores(goal or "", passages, settings=settings)
    except Exception:  # noqa: BLE001 — expand must not take down retrieve
        raw = [0.0] * len(items)
    return {item.chunk_id: float(score) for item, score in zip(items, raw)}
