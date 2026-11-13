from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import Chunk
from app.rerank.model import rerank_scores
from app.search.hybrid import hybrid_search
from app.search.stages import observe_search, record_candidates, stage_timer


@dataclass
class RerankHit:
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    rrf_score: float
    dense_score: float | None
    sparse_score: float | None
    dense_rank: int | None
    sparse_rank: int | None
    start: int
    end: int


@observe_search()
def rerank_search(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    candidates: int | None = None,
    owner_id: UUID | None = None,
    scope=None,
    settings: Settings | None = None,
) -> list[RerankHit]:
    """Dense+BM25(RRF) 的 Top-N 候选，用 Cross-Encoder 对 (query, 全文) 精排，取 Top-K。

    候选阶段只依赖排名/近似相关性，snippet 也被截断到 200 字符；Cross-Encoder
    需要完整语义，因此这里回表取 chunk 全文，而不是复用 hybrid 阶段的 snippet。
    """
    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.rerank_top_k
    k = max(1, min(k, settings.rerank_max_top_k))
    candidate_pool = candidates if candidates is not None else settings.rerank_candidates

    q = query.strip()
    if not q:
        return []

    hybrid_hits = hybrid_search(db, q, top_k=candidate_pool, owner_id=owner_id, scope=scope, settings=settings)
    record_candidates("rerank_input", hybrid_hits)
    if not hybrid_hits:
        record_candidates("rerank_output", [])
        return []

    chunk_ids = [hit.chunk_id for hit in hybrid_hits]
    with stage_timer("rerank_fetch_ms"):
        rows = db.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids))).all()
    text_by_id = {row.id: row.text or "" for row in rows}

    passages = [text_by_id.get(hit.chunk_id, hit.snippet) for hit in hybrid_hits]
    from app.search.stages import add_stage, note_meta

    note_meta("rerank_candidates", len(passages))

    started = time.perf_counter()
    scores = rerank_scores(q, passages, settings=settings)
    add_stage("rerank_call_ms", started)

    ranked = sorted(zip(hybrid_hits, scores), key=lambda item: item[1], reverse=True)
    record_candidates("rerank_output", ranked, scored=True)

    hits: list[RerankHit] = []
    for hit, score in ranked[:k]:
        hits.append(
            RerankHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                page=hit.page,
                snippet=hit.snippet,
                score=round(float(score), 6),
                rrf_score=hit.score,
                dense_score=hit.dense_score,
                sparse_score=hit.sparse_score,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
                start=hit.start,
                end=hit.end,
            )
        )
    return hits
