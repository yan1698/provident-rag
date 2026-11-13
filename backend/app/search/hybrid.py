from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.search.dense import dense_search
from app.search.sparse import sparse_search
from app.search.stages import observe_search, record_candidates


@dataclass
class HybridHit:
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    dense_score: float | None
    sparse_score: float | None
    dense_rank: int | None
    sparse_rank: int | None
    start: int
    end: int


@observe_search()
def hybrid_search(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    dense_candidates: int | None = None,
    sparse_candidates: int | None = None,
    rrf_k: int | None = None,
    owner_id: UUID | None = None,
    scope=None,
    settings: Settings | None = None,
) -> list[HybridHit]:
    """Dense Top-N + BM25 Top-N，用 Reciprocal Rank Fusion 合并为 Top-K。

    RRF(chunk) = sum_over_lists( 1 / (rrf_k + rank_in_list) )
    只按名次计分，天然屏蔽两路检索分数量纲不同的问题（cosine 相似度 vs BM25 分数）。
    """
    settings = settings or get_settings()
    top_k = top_k if top_k is not None else settings.hybrid_top_k
    dense_candidates = dense_candidates if dense_candidates is not None else settings.hybrid_dense_candidates
    sparse_candidates = sparse_candidates if sparse_candidates is not None else settings.hybrid_sparse_candidates
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    q = query.strip()
    if not q:
        return []

    from app.search.stages import add_stage

    dense_hits = dense_search(db, q, top_k=dense_candidates, owner_id=owner_id, scope=scope, settings=settings)
    sparse_hits = sparse_search(db, q, top_k=sparse_candidates, owner_id=owner_id, scope=scope, settings=settings)
    started = time.perf_counter()

    merged: dict[UUID, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        entry = merged.setdefault(
            hit.chunk_id,
            {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "page": hit.page,
                "snippet": hit.snippet,
                "start": hit.start,
                "end": hit.end,
                "rrf": 0.0,
                "dense_score": None,
                "sparse_score": None,
                "dense_rank": None,
                "sparse_rank": None,
            },
        )
        entry["rrf"] += 1.0 / (rrf_k + rank)
        entry["dense_score"] = hit.score
        entry["dense_rank"] = rank

    for rank, hit in enumerate(sparse_hits, start=1):
        entry = merged.setdefault(
            hit.chunk_id,
            {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "page": hit.page,
                "snippet": hit.snippet,
                "start": hit.start,
                "end": hit.end,
                "rrf": 0.0,
                "dense_score": None,
                "sparse_score": None,
                "dense_rank": None,
                "sparse_rank": None,
            },
        )
        entry["rrf"] += 1.0 / (rrf_k + rank)
        entry["sparse_score"] = hit.score
        entry["sparse_rank"] = rank

    ranked = sorted(merged.values(), key=lambda e: e["rrf"], reverse=True)
    add_stage("rrf_ms", started)
    record_candidates("rrf", ranked)
    ranked = ranked[:top_k]

    return [
        HybridHit(
            chunk_id=e["chunk_id"],
            doc_id=e["doc_id"],
            page=e["page"],
            snippet=e["snippet"],
            score=round(e["rrf"], 6),
            dense_score=e["dense_score"],
            sparse_score=e["sparse_score"],
            dense_rank=e["dense_rank"],
            sparse_rank=e["sparse_rank"],
            start=e["start"],
            end=e["end"],
        )
        for e in ranked
    ]
