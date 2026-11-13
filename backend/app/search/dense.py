from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.embed.model import encode_query
from app.search.stages import observe_search

SNIPPET_CHARS = 200


@dataclass
class DenseHit:
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    start: int
    end: int


@observe_search("dense", "vector_ms")
def dense_search(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    owner_id: UUID | None = None,
    scope=None,
    settings: Settings | None = None,
) -> list[DenseHit]:
    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.dense_top_k
    k = max(1, min(k, settings.dense_max_top_k))

    q = query.strip()
    if not q:
        return []

    from app.search.stages import add_stage

    started = time.perf_counter()
    query_vector = encode_query(q, settings=settings)
    add_stage("embed_ms", started)
    vector_literal = _vector_literal(query_vector)
    owner_sql, extra = _owner_clause(owner_id)
    from app.corpus.scope import scope_sql

    scope_filter, scope_extra = scope_sql(scope)
    extra.update(scope_extra)
    owner_sql = f"{owner_sql}\n            {scope_filter}"

    started = time.perf_counter()
    rows = (
        db.execute(
            text(
                f"""
            SELECT
                id,
                doc_id,
                page,
                start_offset,
                end_offset,
                text,
                1 - (embedding <=> CAST(:query_vec AS vector)) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            {owner_sql}
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :top_k
            """
            ),
            {"query_vec": vector_literal, "top_k": k, **extra},
        )
        .mappings()
        .all()
    )
    add_stage("ann_ms", started)

    hits: list[DenseHit] = []
    for row in rows:
        content = row["text"] or ""
        snippet = content[:SNIPPET_CHARS]
        if len(content) > SNIPPET_CHARS:
            snippet += "…"
        hits.append(
            DenseHit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                page=int(row["page"]),
                snippet=snippet,
                score=round(float(row["score"]), 6),
                start=int(row["start_offset"]),
                end=int(row["end_offset"]),
            )
        )
    return hits


def _owner_clause(owner_id: UUID | None) -> tuple[str, dict[str, str]]:
    if owner_id is None:
        return "", {}
    return (
        "AND doc_id IN (SELECT id FROM documents WHERE owner_id = CAST(:owner_id AS uuid))",
        {"owner_id": str(owner_id)},
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"
