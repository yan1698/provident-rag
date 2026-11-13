"""Corpus section catalog: shared Markdown heading state machine.

Used by gold labeling and Agent survey. Do not give Agent a private heading regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
# Same ATX rule, but captures hash count for part_of stacks (graph + future survey).
HEADING_LEVEL_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MIN_SECTION_SHARE = 0.3


def iter_headings(text: str) -> tuple[tuple[int, str, int], ...]:
    """Yield (level, title, start) in document order. Level is the ATX hash count."""
    rows: list[tuple[int, str, int]] = []
    for match in HEADING_LEVEL_RE.finditer(text or ""):
        title = match.group(2).strip()
        if not title:
            continue
        rows.append((len(match.group(1)), title, match.start()))
    return tuple(rows)


@dataclass
class CatalogDocument:
    doc_id: str
    filename: str
    sections: list[str] = field(default_factory=list)
    preview: str = ""
    section_entries: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogBuild:
    documents: tuple[CatalogDocument, ...]
    truncated: bool = False

    @property
    def section_count(self) -> int:
        return sum(len(item.sections) for item in self.documents)


def section_shares(text: str, current: str) -> tuple[dict[str, float], str]:
    """Walk one chunk: how much of it belongs to each heading, plus the trailing section."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return ({current: 1.0} if current else {}), current

    shares: dict[str, float] = {}
    cursor = 0
    section = current
    for match in matches:
        span = match.start() - cursor
        if span > 0 and section:
            shares[section] = shares.get(section, 0.0) + span
        section = match.group(1).strip()
        cursor = match.start()
    tail = len(text) - cursor
    if tail > 0:
        shares[section] = shares.get(section, 0.0) + tail

    total = sum(shares.values()) or 1.0
    return {key: value / total for key, value in shares.items()}, section


def dominant_sections(shares: dict[str, float]) -> tuple[str, ...]:
    if not shares:
        return ()
    ordered = sorted(shares.items(), key=lambda item: item[1], reverse=True)
    keep = [ordered[0][0]]
    keep.extend(name for name, share in ordered[1:] if share >= MIN_SECTION_SHARE)
    return tuple(keep)


def label_chunk_sections(texts: list[str]) -> list[tuple[str, ...]]:
    """Label ordered chunks in one document with the headings they cover."""
    out: list[tuple[str, ...]] = []
    section = ""
    for text in texts:
        shares, section = section_shares(text or "", section)
        out.append(dominant_sections(shares))
    return out


def headings_in_order(texts: list[str], *, fallback: str = "") -> list[str]:
    """Unique headings in first-seen order. No ATX heading → filename fallback."""
    ordered: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in HEADING_RE.finditer(text or ""):
            title = match.group(1).strip()
            if title and title not in seen:
                seen.add(title)
                ordered.append(title)
    if ordered:
        return ordered
    name = fallback.strip()
    return [name] if name else []


def build_catalog(
    items: list[tuple[str, str, list[str]]],
    *,
    max_sections: int = 80,
    goal: str = "",
    max_sections_per_doc: int | None = None,
) -> CatalogBuild:
    """Build a catalog from (doc_id, filename, chunk_texts). Skip docs with no chunks.

    When ``goal`` is set, documents are ordered by lexical overlap so a long
    first-uploaded handbook cannot push later, on-topic files out of the budget.
    """
    from app.corpus.relevance import score_document

    cap = max(1, int(max_sections))
    per_doc = max_sections_per_doc
    if per_doc is None:
        per_doc = max(8, min(24, cap))
    prepared: list[tuple[float, int, str, str, list[str], str]] = []
    for index, (doc_id, filename, texts) in enumerate(items):
        if not texts:
            continue
        sections = headings_in_order(texts, fallback=filename)
        if not sections:
            continue
        preview = "\n".join(piece for piece in texts[:4] if piece)[:1200]
        score = score_document(goal, filename, sections, preview) if (goal or "").strip() else 0.0
        prepared.append((score, index, doc_id, filename, sections, preview))
    if (goal or "").strip():
        prepared.sort(key=lambda row: (-row[0], row[1]))

    remaining = cap
    documents: list[CatalogDocument] = []
    truncated = False
    for _score, _index, doc_id, filename, sections, preview in prepared:
        if remaining <= 0:
            truncated = True
            break
        take = min(len(sections), remaining, per_doc)
        if take < len(sections):
            truncated = True
        documents.append(
            CatalogDocument(
                doc_id=doc_id,
                filename=filename,
                sections=sections[:take],
                preview=preview,
            )
        )
        remaining -= take
    return CatalogBuild(documents=tuple(documents), truncated=truncated)


def catalog_from_digest(rows: list | None) -> list[CatalogDocument]:
    """Rehydrate a previous run's catalog without scanning chunk text."""
    documents: list[CatalogDocument] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        sections = [str(title).strip() for title in (item.get("sections") or []) if str(title).strip()]
        if not filename and not sections:
            continue
        documents.append(
            CatalogDocument(
                doc_id=str(item.get("doc_id") or ""),
                filename=filename,
                sections=sections,
                preview=str(item.get("preview") or ""),
                section_entries=list(item.get("section_entries") or []),
            )
        )
    return documents


def catalog_digest_payload(documents: list[CatalogDocument] | tuple[CatalogDocument, ...]) -> list[dict]:
    return [
        {
            "doc_id": item.doc_id,
            "filename": item.filename,
            "sections": list(item.sections),
            "preview": (getattr(item, "preview", "") or "")[:200],
            "section_entries": list(item.section_entries),
        }
        for item in documents
    ]


def load_corpus_catalog(
    db: Session,
    *,
    max_sections: int = 80,
    goal: str = "",
    owner_id=None,
    scope=None,
    target_document_id=None,
) -> CatalogBuild:
    """Scan the current full corpus. No vector search."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from app.models import Document

    stmt = select(Document).options(selectinload(Document.chunks)).order_by(Document.created_at, Document.filename)
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    if target_document_id:
        from uuid import UUID

        stmt = stmt.where(Document.id == UUID(str(target_document_id)))
    rows = db.execute(stmt).scalars().unique().all()
    if scope is not None:
        rows = [doc for doc in rows if scope.allows(doc.collection_id)]
    items: list[tuple[str, str, list[str]]] = []
    for doc in rows:
        chunks = sorted(doc.chunks, key=lambda chunk: (chunk.start_offset, chunk.page))
        items.append((str(doc.id), doc.filename, [chunk.text or "" for chunk in chunks]))
    built = build_catalog(items, max_sections=max_sections, goal=goal or "")
    from app.corpus.sections import section_records

    by_id = {str(doc.id): doc for doc in rows}
    for item in built.documents:
        item.section_entries = section_records(item.doc_id, list(by_id[item.doc_id].chunks))
    return built


def catalog_observation(built: CatalogBuild) -> dict:
    documents = [
        {
            "doc_id": item.doc_id,
            "filename": item.filename,
            "sections": list(item.sections),
            "section_entries": list(item.section_entries),
        }
        for item in built.documents
    ]
    sections = [{"filename": item.filename, "title": title} for item in built.documents for title in item.sections]
    return {
        "documents": documents,
        "section_count": built.section_count,
        "sections": sections,
        "truncated": built.truncated,
    }
