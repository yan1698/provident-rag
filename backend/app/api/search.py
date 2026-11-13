from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner
from app.cache.service import cached_payload
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.user import User
from app.search.dense import DenseHit, dense_search
from app.search.hybrid import HybridHit, hybrid_search
from app.search.rerank import RerankHit, rerank_search
from app.search.sparse import SparseHit, sparse_search

router = APIRouter(tags=["search"])


class DenseSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language query")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of hits to return")


class DenseSearchHit(BaseModel):
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    start: int
    end: int


class DenseSearchResponse(BaseModel):
    query: str
    top_k: int
    count: int
    results: list[DenseSearchHit]
    cached: bool = False


class SparseSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language query")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of hits to return")


class SparseSearchHit(BaseModel):
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    start: int
    end: int


class SparseSearchResponse(BaseModel):
    query: str
    top_k: int
    count: int
    results: list[SparseSearchHit]
    cached: bool = False


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language query")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Final RRF hits to return")
    dense_candidates: int | None = Field(
        default=None, ge=1, le=200, description="Dense candidate pool size before fusion"
    )
    sparse_candidates: int | None = Field(
        default=None, ge=1, le=200, description="BM25 candidate pool size before fusion"
    )


class HybridSearchHit(BaseModel):
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


class HybridSearchResponse(BaseModel):
    query: str
    top_k: int
    dense_candidates: int
    sparse_candidates: int
    count: int
    results: list[HybridSearchHit]
    cached: bool = False


class RerankSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language query")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Final reranked hits to return")
    candidates: int | None = Field(
        default=None, ge=1, le=200, description="Hybrid (RRF) candidate pool size before reranking"
    )


class RerankSearchHit(BaseModel):
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


class RerankSearchResponse(BaseModel):
    query: str
    top_k: int
    candidates: int
    count: int
    results: list[RerankSearchHit]
    cached: bool = False


def _to_dense_hit(hit: DenseHit) -> DenseSearchHit:
    return DenseSearchHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        page=hit.page,
        snippet=hit.snippet,
        score=hit.score,
        start=hit.start,
        end=hit.end,
    )


def _to_sparse_hit(hit: SparseHit) -> SparseSearchHit:
    return SparseSearchHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        page=hit.page,
        snippet=hit.snippet,
        score=hit.score,
        start=hit.start,
        end=hit.end,
    )


def _to_hybrid_hit(hit: HybridHit) -> HybridSearchHit:
    return HybridSearchHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        page=hit.page,
        snippet=hit.snippet,
        score=hit.score,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        start=hit.start,
        end=hit.end,
    )


def _to_rerank_hit(hit: RerankHit) -> RerankSearchHit:
    return RerankSearchHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        page=hit.page,
        snippet=hit.snippet,
        score=hit.score,
        rrf_score=hit.rrf_score,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        start=hit.start,
        end=hit.end,
    )


@router.post("/search/dense", response_model=DenseSearchResponse)
def search_dense(
    body: DenseSearchRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> DenseSearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    effective_top_k = body.top_k if body.top_k is not None else settings.dense_top_k
    uid = current_owner(user)

    def _build() -> dict:
        hits = dense_search(db, query, top_k=effective_top_k, owner_id=uid, settings=settings)
        return DenseSearchResponse(
            query=query,
            top_k=effective_top_k,
            count=len(hits),
            results=[_to_dense_hit(h) for h in hits],
        ).model_dump(mode="json")

    payload, cached = cached_payload(
        "dense",
        {"query": query, "top_k": effective_top_k, "owner_id": str(uid) if uid else ""},
        _build,
        settings=settings,
    )
    payload["cached"] = cached
    return DenseSearchResponse.model_validate(payload)


@router.post("/search/sparse", response_model=SparseSearchResponse)
def search_sparse(
    body: SparseSearchRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> SparseSearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    effective_top_k = body.top_k if body.top_k is not None else settings.sparse_top_k
    uid = current_owner(user)

    def _build() -> dict:
        hits = sparse_search(db, query, top_k=effective_top_k, owner_id=uid, settings=settings)
        return SparseSearchResponse(
            query=query,
            top_k=effective_top_k,
            count=len(hits),
            results=[_to_sparse_hit(h) for h in hits],
        ).model_dump(mode="json")

    payload, cached = cached_payload(
        "sparse",
        {"query": query, "top_k": effective_top_k, "owner_id": str(uid) if uid else ""},
        _build,
        settings=settings,
    )
    payload["cached"] = cached
    return SparseSearchResponse.model_validate(payload)


@router.post("/search/hybrid", response_model=HybridSearchResponse)
def search_hybrid(
    body: HybridSearchRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> HybridSearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    effective_top_k = body.top_k if body.top_k is not None else settings.hybrid_top_k
    dense_candidates = body.dense_candidates if body.dense_candidates is not None else settings.hybrid_dense_candidates
    sparse_candidates = (
        body.sparse_candidates if body.sparse_candidates is not None else settings.hybrid_sparse_candidates
    )
    uid = current_owner(user)

    def _build() -> dict:
        hits = hybrid_search(
            db,
            query,
            top_k=effective_top_k,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            owner_id=uid,
            settings=settings,
        )
        return HybridSearchResponse(
            query=query,
            top_k=effective_top_k,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            count=len(hits),
            results=[_to_hybrid_hit(h) for h in hits],
        ).model_dump(mode="json")

    payload, cached = cached_payload(
        "hybrid",
        {
            "query": query,
            "top_k": effective_top_k,
            "dense_candidates": dense_candidates,
            "sparse_candidates": sparse_candidates,
            "owner_id": str(uid) if uid else "",
        },
        _build,
        settings=settings,
    )
    payload["cached"] = cached
    return HybridSearchResponse.model_validate(payload)


@router.post("/search/rerank", response_model=RerankSearchResponse)
def search_rerank(
    body: RerankSearchRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: User | None = Depends(get_current_user),
) -> RerankSearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    effective_top_k = body.top_k if body.top_k is not None else settings.rerank_top_k
    candidates = body.candidates if body.candidates is not None else settings.rerank_candidates
    uid = current_owner(user)

    def _build() -> dict:
        hits = rerank_search(
            db,
            query,
            top_k=effective_top_k,
            candidates=candidates,
            owner_id=uid,
            settings=settings,
        )
        return RerankSearchResponse(
            query=query,
            top_k=effective_top_k,
            candidates=candidates,
            count=len(hits),
            results=[_to_rerank_hit(h) for h in hits],
        ).model_dump(mode="json")

    payload, cached = cached_payload(
        "rerank",
        {
            "query": query,
            "top_k": effective_top_k,
            "candidates": candidates,
            "owner_id": str(uid) if uid else "",
        },
        _build,
        settings=settings,
    )
    payload["cached"] = cached
    return RerankSearchResponse.model_validate(payload)
