"""Attach same-document neighbors so a hit is not an isolated fragment.

Lookup is by document order (page, start_offset), never by offset arithmetic.
Hits are never dropped to make room; leftover budget goes to ±radius neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.generate.prompt import EvidenceItem
from app.models import Chunk


@dataclass(frozen=True)
class NeighborRow:
    chunk_id: str
    doc_id: str
    page: int
    start_offset: int
    text: str
    chunk_hash: str = ""


def expand_neighbors(
    db: Session,
    evidence: list[EvidenceItem],
    *,
    settings: Settings | None = None,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    radius = 1
    max_chars = 6000
    if settings is not None:
        radius = max(0, int(getattr(settings, "agent_neighbor_radius", 1) or 0))
        max_chars = max(800, int(getattr(settings, "qa_max_context_chars", 6000) or 6000))
    if radius <= 0 or not evidence:
        return list(evidence), _empty_note()
    doc_ids: list[UUID] = []
    for item in evidence:
        try:
            doc_ids.append(UUID(str(item.doc_id)))
        except ValueError:
            continue
    if not doc_ids:
        return list(evidence), _empty_note()
    rows = list(
        db.execute(
            select(Chunk).where(Chunk.doc_id.in_(set(doc_ids))).order_by(Chunk.doc_id, Chunk.page, Chunk.start_offset)
        )
        .scalars()
        .all()
    )
    corpus = [
        NeighborRow(
            chunk_id=str(row.id),
            doc_id=str(row.doc_id),
            page=int(row.page or 1),
            start_offset=int(row.start_offset or 0),
            text=(row.text or "").strip(),
            chunk_hash=row.chunk_hash or "",
        )
        for row in rows
    ]
    return expand_from_rows(evidence, corpus, radius=radius, max_chars=max_chars)


def expand_from_rows(
    hits: list[EvidenceItem],
    corpus: list[NeighborRow],
    *,
    radius: int = 1,
    max_chars: int = 6000,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    if not hits or radius <= 0:
        return list(hits), _empty_note()

    by_doc: dict[str, list[NeighborRow]] = {}
    for row in corpus:
        by_doc.setdefault(row.doc_id, []).append(row)
    for rows in by_doc.values():
        rows.sort(key=lambda item: (item.page, item.start_offset))

    hit_ids = {item.chunk_id for item in hits}
    candidates: list[tuple[int, str, NeighborRow]] = []
    seen_nb: set[str] = set()
    for item in hits:
        rows = by_doc.get(item.doc_id) or []
        pos = next((i for i, row in enumerate(rows) if row.chunk_id == item.chunk_id), None)
        if pos is None:
            continue
        for dist in range(1, radius + 1):
            for idx in (pos - dist, pos + dist):
                if idx < 0 or idx >= len(rows):
                    continue
                row = rows[idx]
                if row.chunk_id in hit_ids or row.chunk_id in seen_nb:
                    continue
                seen_nb.add(row.chunk_id)
                candidates.append((dist, item.chunk_id, row))
    candidates.sort(key=lambda item: (item[0], item[2].page, item[2].start_offset))

    hit_chars = sum(len(item.text or "") for item in hits)
    budget = max(hit_chars, int(max_chars))
    used = hit_chars
    kept_nb: list[EvidenceItem] = []
    dropped = 0
    for _dist, parent_id, row in candidates:
        extra = len(row.text or "")
        if used + extra > budget:
            dropped += 1
            continue
        used += extra
        kept_nb.append(
            EvidenceItem(
                index=0,
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                page=row.page,
                text=row.text,
                chunk_hash=row.chunk_hash,
                role="neighbor",
                start_offset=row.start_offset,
                parent_chunk_id=parent_id,
            )
        )

    combined = list(hits) + kept_nb
    combined.sort(
        key=lambda item: (
            item.doc_id,
            item.page,
            item.start_offset,
            0 if getattr(item, "role", "hit") != "neighbor" else 1,
        )
    )
    out: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in combined:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        out.append(replace(item, index=len(out) + 1))

    note = ""
    if dropped:
        note = f"邻域扩展截断：保留命中块，丢弃 {dropped} 个邻居"
    elif kept_nb:
        note = f"邻域扩展：补了 {len(kept_nb)} 个相邻块"
    return out, {"neighbors_added": len(kept_nb), "neighbors_dropped": dropped, "note": note}


def _empty_note() -> dict[str, Any]:
    return {"neighbors_added": 0, "neighbors_dropped": 0, "note": ""}
