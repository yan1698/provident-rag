"""Day 10: 把 golden queries 上的 gold 标注解析成语料里真实的 chunk 集合。

标注有两种写法，可混用：

- ``gold_sections``：Markdown 章节标题（可省略 ``#`` 前缀，支持子串匹配）。
  一个 chunk 若有不少于 ``MIN_SECTION_SHARE`` 的文本落在该章节内，就算该章节的 gold chunk。
- ``gold_chunk_hashes``：直接写 ``chunk_hash``，支持只写前缀（便于人工标注）。

章节归属完全由 chunk 文本推导：chunk 是按段落拼出来的，段落间距与原文可能差几个空白字符，
所以这里不依赖 ``start_offset`` 做区间运算，只按文档顺序维护"当前章节"状态机。
状态机与 Agent survey 共用 ``app.corpus.catalog``，禁止再写一份 heading regex。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from app.corpus.catalog import dominant_sections, label_chunk_sections, section_shares
from app.eval.golden import GoldenQuery, GoldenSet, corpus_filenames

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# SQLAlchemy 与 app.models 只在下面两个查库函数里用到，放进函数体，
# 好让 resolve_gold / LabeledChunk 这些纯标注逻辑在无 DB 的 CI 里也能 import。

_NORMALIZE_RE = re.compile(r"[\s#*`_.、,，:：]+")
_MIN_HASH_PREFIX = 6


@dataclass(frozen=True)
class GoldChunk:
    chunk_id: str
    chunk_hash: str
    doc_id: str
    page: int
    section: str
    preview: str


@dataclass(frozen=True)
class GoldResolution:
    chunks: tuple[GoldChunk, ...]
    missing_labels: tuple[str, ...]

    @property
    def hashes(self) -> set[str]:
        return {item.chunk_hash for item in self.chunks}


@dataclass(frozen=True)
class LabeledChunk:
    chunk_id: str
    chunk_hash: str
    doc_id: str
    page: int
    sections: tuple[str, ...]
    preview: str
    text: str = ""

    @property
    def section(self) -> str:
        return self.sections[0] if self.sections else ""


def corpus_chunks(db: Session, corpus: str = "", *, owner_id: UUID | None = None) -> list[LabeledChunk]:
    """按文档顺序读取 chunk，并给每个 chunk 打上它覆盖到的章节标题。"""
    from sqlalchemy import select

    from app.models import Chunk, Document

    stmt = select(Chunk.id, Chunk.doc_id, Chunk.page, Chunk.start_offset, Chunk.text, Chunk.chunk_hash).order_by(
        Chunk.doc_id, Chunk.start_offset
    )
    doc_ids = _corpus_doc_ids(db, corpus, owner_id=owner_id)
    if doc_ids:
        stmt = stmt.where(Chunk.doc_id.in_(doc_ids))
    elif owner_id is not None:
        owned = select(Document.id).where(Document.owner_id == owner_id)
        stmt = stmt.where(Chunk.doc_id.in_(owned))

    labeled: list[LabeledChunk] = []
    current_doc: UUID | None = None
    section = ""
    for row in db.execute(stmt).all():
        if row.doc_id != current_doc:
            current_doc = row.doc_id
            section = ""
        shares, section = section_shares(row.text or "", section)
        labeled.append(
            LabeledChunk(
                chunk_id=str(row.id),
                chunk_hash=row.chunk_hash or "",
                doc_id=str(row.doc_id),
                page=row.page,
                sections=dominant_sections(shares),
                preview=preview(row.text or ""),
                text=row.text or "",
            )
        )
    return labeled


def label_sections(texts: list[str]) -> list[tuple[str, ...]]:
    """给同一文档内按顺序排列的 chunk 文本标注章节，供离线标注工具复用。"""
    return label_chunk_sections(texts)


def resolve_gold(spec: GoldenQuery, chunks: list[LabeledChunk]) -> GoldResolution:
    matched: dict[str, GoldChunk] = {}
    missing: list[str] = []

    for raw_hash in spec.gold_chunk_hashes:
        prefix = raw_hash.strip().lower()
        if len(prefix) < _MIN_HASH_PREFIX:
            missing.append(raw_hash)
            continue
        hits = [c for c in chunks if c.chunk_hash.lower().startswith(prefix)]
        if not hits:
            missing.append(raw_hash)
        for chunk in hits:
            matched.setdefault(chunk.chunk_hash, _to_gold(chunk, chunk.section))

    for label in spec.gold_sections:
        wanted = _normalize(label)
        if not wanted:
            missing.append(label)
            continue
        hits = [(c, s) for c in chunks for s in c.sections if wanted in _normalize(s)]
        if not hits:
            missing.append(label)
        for chunk, section in hits:
            matched.setdefault(chunk.chunk_hash, _to_gold(chunk, section))

    ordered = tuple(sorted(matched.values(), key=lambda item: (item.doc_id, item.page, item.chunk_hash)))
    return GoldResolution(chunks=ordered, missing_labels=tuple(missing))


def resolve_golden_set(db: Session, golden: GoldenSet, *, owner_id: UUID | None = None) -> dict[str, GoldResolution]:
    chunks = corpus_chunks(db, golden.corpus, owner_id=owner_id)
    return {spec.id: resolve_gold(spec, chunks) for spec in golden.queries if spec.has_gold}


def _corpus_doc_ids(db: Session, corpus: str, *, owner_id: UUID | None = None) -> list[UUID]:
    """golden.corpus 指向源文件时，只在这些文档内找 gold chunk；找不到则退回全库。"""
    names = list(corpus_filenames(corpus))
    if not names:
        return []
    from sqlalchemy import select

    from app.models import Document

    stmt = select(Document.id).where(Document.filename.in_(names))
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    rows = db.execute(stmt).scalars().all()
    return list(rows)


def _to_gold(chunk: LabeledChunk, section: str) -> GoldChunk:
    return GoldChunk(
        chunk_id=chunk.chunk_id,
        chunk_hash=chunk.chunk_hash,
        doc_id=chunk.doc_id,
        page=chunk.page,
        section=section,
        preview=chunk.preview,
    )


def preview(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).lower()
