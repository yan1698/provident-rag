from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.eval.gold import GoldResolution, corpus_chunks, resolve_gold
from app.eval.golden import GoldenSet
from app.eval.metrics import EvalReport, QueryScore, aggregate, score_query
from app.models import Chunk
from app.models.trace import QueryTrace


def latest_traces_for_golden(db: Session, golden: GoldenSet, *, owner_id: UUID | None = None) -> dict[str, QueryTrace]:
    """Pick the newest trace per golden query (eval_id first, else exact query text)."""
    found: dict[str, QueryTrace] = {}
    eval_ids = [item.id for item in golden.queries]

    if eval_ids:
        stmt = select(QueryTrace).where(QueryTrace.eval_id.in_(eval_ids)).order_by(QueryTrace.created_at.desc())
        if owner_id is not None:
            stmt = stmt.where(QueryTrace.user_id == owner_id)
        rows = db.execute(stmt).scalars().all()
        for row in rows:
            if row.eval_id and row.eval_id not in found:
                found[row.eval_id] = row

    for spec in golden.queries:
        if spec.id in found:
            continue
        stmt = select(QueryTrace).where(QueryTrace.query == spec.query).order_by(QueryTrace.created_at.desc()).limit(1)
        if owner_id is not None:
            stmt = stmt.where(QueryTrace.user_id == owner_id)
        row = db.execute(stmt).scalar_one_or_none()
        if row is not None:
            found[spec.id] = row
    return found


def report_from_db(db: Session, golden: GoldenSet, *, owner_id: UUID | None = None) -> EvalReport:
    traces = latest_traces_for_golden(db, golden, owner_id=owner_id)
    return score_traces(db, golden, traces, owner_id=owner_id)


def score_traces(
    db: Session,
    golden: GoldenSet,
    traces: dict[str, QueryTrace],
    *,
    verification_by_id: dict[str, dict[str, Any]] | None = None,
    owner_id: UUID | None = None,
) -> EvalReport:
    """Score golden queries against a trace map. verification_by_id overrides stored verification."""
    chunks = corpus_chunks(db, golden.corpus, owner_id=owner_id)
    gold = {spec.id: resolve_gold(spec, chunks) for spec in golden.queries if spec.has_gold}
    # corpus 声明的文件一份都没入库时 corpus_chunks 会退回全库，此时集合覆盖一切、
    # 语料外证据恒为 0。这是刻意的：认不出语料就别亮红灯。
    corpus_hashes = frozenset(item.chunk_hash for item in chunks if item.chunk_hash)
    empty = GoldResolution(chunks=(), missing_labels=())
    overrides = verification_by_id or {}
    scores: list[QueryScore] = []
    for spec in golden.queries:
        resolution = gold.get(spec.id, empty)
        row = traces.get(spec.id)
        if row is None:
            scores.append(
                score_query(
                    spec,
                    trace_id=None,
                    gold_hashes=resolution.hashes,
                    gold_missing_labels=resolution.missing_labels,
                )
            )
            continue
        stored = row.verification if isinstance(row.verification, dict) else None
        verification = overrides.get(spec.id, stored)
        scores.append(
            score_query(
                spec,
                trace_id=str(row.id),
                answer=row.answer or "",
                evidence=with_chunk_hashes(db, list(row.evidence or [])),
                citations=list(row.citations or []),
                verification=verification,
                gold_hashes=resolution.hashes,
                gold_missing_labels=resolution.missing_labels,
                corpus_hashes=corpus_hashes,
            )
        )
    return aggregate(golden, scores)


def with_chunk_hashes(db: Session, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Day 8/9 的 traces 没有 chunk_hash，按 chunk_id 回表补齐，让旧 trace 也能算检索指标。"""
    missing = [item.get("chunk_id") for item in evidence if not item.get("chunk_hash")]
    chunk_ids: list[UUID] = []
    for value in missing:
        try:
            chunk_ids.append(UUID(str(value)))
        except (TypeError, ValueError):
            continue
    if not chunk_ids:
        return evidence

    rows = db.execute(select(Chunk.id, Chunk.chunk_hash).where(Chunk.id.in_(chunk_ids))).all()
    hash_by_id = {str(row.id): row.chunk_hash or "" for row in rows}
    if not hash_by_id:
        return evidence

    patched: list[dict[str, Any]] = []
    for item in evidence:
        if item.get("chunk_hash"):
            patched.append(item)
            continue
        patched.append({**item, "chunk_hash": hash_by_id.get(str(item.get("chunk_id")), "")})
    return patched
