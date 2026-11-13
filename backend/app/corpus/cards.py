"""Document cards from headings and the opening sentence. No model call."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.corpus.catalog import headings_in_order
from app.corpus.names import filename_stem, filename_usable
from app.graph.schema import first_sentence
from app.models import Document
from app.models.collection import DocumentCard

DOC_TYPES = ("handbook", "spec", "paper", "note", "transcript", "other")
DOC_TYPE_LABELS = {
    "handbook": "手册",
    "spec": "规范",
    "paper": "论文",
    "note": "笔记",
    "transcript": "记录",
    "other": "其他",
}
ASSIGN_FLOOR = 0.55
ASSIGN_GAP = 0.15


def build_card_fields(document: Document) -> dict:
    chunks = sorted(document.chunks or [], key=lambda row: (row.page, row.start_offset))
    texts = [row.text or "" for row in chunks]
    headings = [title for title in headings_in_order(texts, fallback="") if title and title != document.filename]
    blob = "\n".join(texts)
    usable = filename_usable(document.filename or "")
    opening = first_sentence(texts[0] if texts else "", limit=120)
    if usable:
        title = filename_stem(document.filename or "") or (document.filename or "未命名文档")
    elif headings:
        title = headings[0][:40]
    elif opening:
        title = opening[:40]
    else:
        title = f"未命名文档 {(document.file_hash or '')[:6]}".strip()
    topics = []
    for heading in headings:
        if heading in blob and heading not in topics:
            topics.append(heading[:12] if len(heading) > 12 else heading)
        if len(topics) >= 6:
            break
    chars = len(blob.strip())
    if chars < 800:
        doc_type = "note"
    elif len(headings) >= 2:
        doc_type = "handbook"
    else:
        doc_type = "other"
    summary = opening if opening and opening in blob else ""
    return {
        "display_title": title[:200],
        "summary": summary[:120],
        "topics": topics,
        "doc_type": doc_type,
        "filename_usable": usable,
    }


def refresh_card(db: Session, document: Document) -> DocumentCard:
    """Write the filing card. User-locked collection is left in place."""
    from app.corpus.library import auto_assign

    fields = build_card_fields(document)
    card = db.get(DocumentCard, document.id)
    if card is None:
        card = DocumentCard(doc_id=document.id, assign_version=1)
        db.add(card)
    card.display_title = fields["display_title"]
    card.summary = fields["summary"]
    card.topics = fields["topics"]
    card.doc_type = fields["doc_type"]
    card.filename_usable = fields["filename_usable"]
    card.status = "ready"
    card.updated_at = datetime.now(timezone.utc)
    if card.assign_source != "user":
        auto_assign(db, document, card)
    db.flush()
    return card
