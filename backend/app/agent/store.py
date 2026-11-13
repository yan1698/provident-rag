"""Corpus / graph access for agent tools. Tools talk to this, not to SQLAlchemy.

``owner_id`` is a constructor argument, not a method parameter — forgetting it
used to silently scan the whole table. A Fake store lets tools run without Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.generate.prompt import EvidenceItem

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.search.rerank import RerankHit


@dataclass(frozen=True)
class ChunkRow:
    id: str
    doc_id: str
    page: int
    text: str
    chunk_hash: str
    start_offset: int = 0
    section_path: str = ""


class AgentStore(Protocol):
    owner_id: Any

    def search(self, query: str, *, settings: Settings) -> list[RerankHit]: ...

    def load_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRow]: ...

    def load_doc_chunks(self, doc_id: str) -> list[ChunkRow]: ...

    def load_section_chunks(self, doc_id: str, section_id: str, *, descendants: bool = True) -> list[ChunkRow]: ...

    def load_catalog(self, *, goal: str, max_sections: int): ...

    def expand_neighbors(
        self, evidence: list[EvidenceItem], *, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]: ...

    def expand_graph(
        self, evidence: list[EvidenceItem], *, goal: str, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]: ...

    def graph_counts(self) -> dict[str, int]: ...

    def graph_walk(self, subjects: list[str], *, hops: int) -> dict[str, Any]: ...


class SqlAgentStore:
    def __init__(self, db: Session, owner_id=None):
        self.session = db
        self.owner_id = owner_id

    def search(self, query: str, *, settings: Settings) -> list[RerankHit]:
        from app.corpus.scope import Scope
        from app.search.rerank import rerank_search

        scope = getattr(self, "scope", None)
        self.scope_note = ""
        hits = rerank_search(self.session, query, settings=settings, owner_id=self.owner_id, scope=scope)
        target = getattr(self, "target_document_id", "")
        if target:
            return [hit for hit in hits if str(hit.doc_id) == target]
        if scope is None or scope.mode != "soft":
            return hits
        if hits:
            self._log_scope(scope, query, [])
            return hits
        wide = rerank_search(self.session, query, settings=settings, owner_id=self.owner_id, scope=Scope())
        outside = self._outside(scope, wide)
        self._log_scope(scope, query, outside)
        if outside:
            self.scope_note = "该范围没有，已在全部文档中找到"
        return wide

    def _outside(self, scope, hits) -> list[tuple]:
        from sqlalchemy import select

        from app.models import Document

        doc_ids = {hit.doc_id for hit in hits}
        if not doc_ids:
            return []
        rows = self.session.execute(select(Document.id, Document.collection_id).where(Document.id.in_(doc_ids))).all()
        found = []
        for doc_id, collection_id in rows:
            if collection_id is None or scope.allows(collection_id):
                continue
            found.append((doc_id, collection_id))
        return found

    def _log_scope(self, scope, query: str, outside: list[tuple]) -> None:
        try:
            from app.corpus.library import log_scope_search

            with self.session.begin_nested():
                log_scope_search(self.session, self.owner_id, scope, query=query, widened_docs=outside)
        except Exception:  # noqa: BLE001 — a scope log must not fail retrieval
            return

    def load_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        from sqlalchemy import select

        from app.models import Chunk

        ids: list[UUID] = []
        for value in chunk_ids:
            try:
                ids.append(UUID(str(value)))
            except (TypeError, ValueError):
                continue
        if not ids:
            return {}
        from app.models import Document

        stmt = (
            select(
                Chunk.id, Chunk.doc_id, Chunk.page, Chunk.text, Chunk.chunk_hash, Chunk.start_offset, Chunk.section_path
            )
            .join(Document)
            .where(Chunk.id.in_(ids))
        )
        if self.owner_id is not None:
            stmt = stmt.where(Document.owner_id == self.owner_id)
        scope = getattr(self, "scope", None)
        if scope is not None and scope.mode == "hard":
            allowed_docs = self._readable_doc_ids(scope)
            stmt = stmt.where(Chunk.doc_id.in_(allowed_docs))
        target = getattr(self, "target_document_id", "")
        if target:
            stmt = stmt.where(Chunk.doc_id == UUID(target))
        rows = self.session.execute(stmt).all()
        return {
            str(row.id): ChunkRow(
                id=str(row.id),
                doc_id=str(row.doc_id),
                page=int(row.page or 1),
                text=row.text or "",
                chunk_hash=row.chunk_hash or "",
                start_offset=int(row.start_offset or 0),
                section_path=row.section_path or "",
            )
            for row in rows
        }

    def load_doc_chunks(self, doc_id: str) -> list[ChunkRow]:
        from sqlalchemy import select

        from app.models import Chunk

        target = getattr(self, "target_document_id", "")
        if target and str(doc_id) != target:
            return []
        try:
            uid = UUID(str(doc_id))
        except (TypeError, ValueError):
            return []
        from app.models import Document

        stmt = select(Chunk).join(Document).where(Chunk.doc_id == uid)
        if self.owner_id is not None:
            stmt = stmt.where(Document.owner_id == self.owner_id)
        scope = getattr(self, "scope", None)
        if scope is not None and scope.mode == "hard":
            stmt = stmt.where(Chunk.doc_id.in_(self._readable_doc_ids(scope)))
        rows = self.session.execute(stmt.order_by(Chunk.start_offset, Chunk.page)).scalars().all()
        return [
            ChunkRow(
                id=str(row.id),
                doc_id=str(row.doc_id),
                page=int(row.page or 1),
                text=row.text or "",
                chunk_hash=row.chunk_hash or "",
                start_offset=int(row.start_offset or 0),
                section_path=row.section_path or "",
            )
            for row in rows
        ]

    def _readable_doc_ids(self, scope):
        from sqlalchemy import select
        from app.models import Document

        stmt = select(Document.id, Document.collection_id)
        if self.owner_id is not None:
            stmt = stmt.where(Document.owner_id == self.owner_id)
        return [row.id for row in self.session.execute(stmt) if scope.allows(row.collection_id)]

    def load_section_chunks(self, doc_id: str, section_id: str, *, descendants: bool = True) -> list[ChunkRow]:
        from app.corpus.sections import section_chunk_ids, section_records

        rows = self.load_doc_chunks(doc_id)
        wanted = set(section_chunk_ids(section_records(doc_id, rows), section_id, descendants=descendants))
        return [row for row in rows if row.id in wanted]

    def load_catalog(self, *, goal: str, max_sections: int):
        from app.corpus.catalog import load_corpus_catalog

        catalog = load_corpus_catalog(
            self.session,
            max_sections=max_sections,
            goal=goal,
            owner_id=self.owner_id,
            scope=getattr(self, "scope", None),
            target_document_id=getattr(self, "target_document_id", None),
        )
        target = getattr(self, "target_document_id", "")
        if target:
            from dataclasses import replace

            catalog = replace(catalog, documents=tuple(doc for doc in catalog.documents if doc.doc_id == target))
        return catalog

    def expand_neighbors(
        self, evidence: list[EvidenceItem], *, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        from app.agent.neighbors import expand_neighbors

        return expand_neighbors(self.session, evidence, settings=settings)

    def expand_graph(
        self, evidence: list[EvidenceItem], *, goal: str, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        from app.graph.expand import expand_graph

        if getattr(self, "target_document_id", ""):
            return list(evidence), {"skipped": "explicit_document_target"}
        return expand_graph(
            self.session,
            evidence,
            goal=goal,
            settings=settings,
            owner_id=self.owner_id,
            scope=getattr(self, "scope", None),
        )

    def graph_counts(self) -> dict[str, int]:
        from app.graph.service import graph_counts

        return graph_counts(self.session, owner_id=self.owner_id)

    def graph_walk(self, subjects: list[str], *, hops: int) -> dict[str, Any]:
        from app.graph.walk import graph_walk_compare

        if getattr(self, "target_document_id", ""):
            return {"nodes": [], "edges": [], "skipped": "explicit_document_target"}
        return graph_walk_compare(
            self.session, subjects, owner_id=self.owner_id, hops=hops, scope=getattr(self, "scope", None)
        )


class FakeAgentStore:
    """In-memory store for tool unit tests. No Postgres, no models."""

    def __init__(self, chunks: list[ChunkRow] | None = None, *, catalog=None, owner_id=None):
        self.session = None
        self.owner_id = owner_id
        self._chunks = list(chunks or [])
        self._catalog = catalog
        self.searches: list[str] = []
        self.load_catalog_calls = 0

    def search(self, query: str, *, settings: Settings) -> list[Any]:
        from types import SimpleNamespace
        from uuid import uuid4

        self.searches.append(query)
        hits = []
        needle = (query or "").casefold()
        for row in self._chunks:
            if needle and needle not in (row.text or "").casefold() and needle not in (row.doc_id or ""):
                # still return the first chunk so factoid tools have *some* hit shape
                continue
            hits.append(
                SimpleNamespace(
                    chunk_id=UUID(row.id) if _is_uuid(row.id) else uuid4(),
                    doc_id=UUID(row.doc_id) if _is_uuid(row.doc_id) else uuid4(),
                    page=row.page,
                    snippet=(row.text or "")[:200],
                    score=1.0,
                )
            )
        if not hits and self._chunks:
            row = self._chunks[0]
            hits.append(
                SimpleNamespace(
                    chunk_id=UUID(row.id) if _is_uuid(row.id) else uuid4(),
                    doc_id=UUID(row.doc_id) if _is_uuid(row.doc_id) else uuid4(),
                    page=row.page,
                    snippet=(row.text or "")[:200],
                    score=0.5,
                )
            )
        return hits

    def load_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        wanted = {str(item) for item in chunk_ids}
        return {row.id: row for row in self._chunks if row.id in wanted}

    def load_doc_chunks(self, doc_id: str) -> list[ChunkRow]:
        return [row for row in self._chunks if row.doc_id == str(doc_id)]

    def load_section_chunks(self, doc_id: str, section_id: str, *, descendants: bool = True) -> list[ChunkRow]:
        from app.corpus.sections import section_chunk_ids, section_records

        rows = self.load_doc_chunks(doc_id)
        wanted = set(section_chunk_ids(section_records(doc_id, rows), section_id, descendants=descendants))
        return [row for row in rows if row.id in wanted]

    def load_catalog(self, *, goal: str, max_sections: int):
        self.load_catalog_calls += 1
        if self._catalog is not None:
            return self._catalog
        from app.corpus.catalog import build_catalog

        grouped: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        for row in self._chunks:
            grouped.setdefault(row.doc_id, []).append(row.text)
            names.setdefault(row.doc_id, f"{row.doc_id}.md")
        items = [(doc_id, names[doc_id], texts) for doc_id, texts in grouped.items()]
        return build_catalog(items, max_sections=max_sections, goal=goal or "")

    def expand_neighbors(
        self, evidence: list[EvidenceItem], *, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        return evidence, {"neighbors_added": 0, "neighbors_dropped": 0, "note": "fake"}

    def expand_graph(
        self, evidence: list[EvidenceItem], *, goal: str, settings: Settings
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        return evidence, {"graph_added": 0, "graph_dropped": 0, "note": "fake"}

    def graph_counts(self) -> dict[str, int]:
        return {"entities": 0, "edges": 0}

    def graph_walk(self, subjects: list[str], *, hops: int) -> dict[str, Any]:
        return {"skipped": True, "reason": "fake store", "dimensions": [], "edges": []}


def _is_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (TypeError, ValueError):
        return False
