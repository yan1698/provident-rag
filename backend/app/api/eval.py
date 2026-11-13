from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, owner_id as current_owner, require_eval_gate_user
from app.db.session import get_db
from app.eval.baseline import (
    compare_metrics,
    parse_baseline,
    precomputed_metrics,
    report_metrics,
    score_baseline_from_db,
)
from app.eval.gold import corpus_chunks, resolve_gold
from app.eval.golden import DEFAULT_EVAL_SET, GoldenSet, list_golden_sets, load_golden
from app.eval.replay import replay_snapshot
from app.eval.service import report_from_db
from app.eval.snapshot import corpus_snapshot
from app.models.user import User

router = APIRouter(tags=["eval"])


EvalSetQuery = Query(
    default=None,
    max_length=64,
    pattern=r"^[A-Za-z0-9._-]+$",
    description="Versioned eval set name (default: golden-agent-notes-v1)",
)


def _load_golden_or_404(eval_set: str | None = None) -> GoldenSet:
    try:
        return load_golden(name=eval_set or None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class QueryScoreOut(BaseModel):
    eval_id: str
    query: str
    expect: str
    trace_id: str | None
    coverage_rate: float | None
    hallucination_rate: float | None
    citation_count: int
    term_hits: list[str]
    term_misses: list[str]
    term_hit: bool | None
    refusal_ok: bool | None
    gold_count: int
    retrieved_count: int
    gold_hit_count: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    first_hit_rank: int | None
    gold_missing_labels: list[str]


class EvalSummaryOut(BaseModel):
    eval_set: str
    query_count: int
    traced_count: int
    missing: list[str]
    avg_coverage_rate: float | None
    avg_hallucination_rate: float | None
    term_hit_rate: float | None
    citation_rate: float | None
    refusal_ok_rate: float | None
    gold_labeled_count: int
    retrieval_scored_count: int
    avg_recall_at_k: float | None
    mrr: float | None
    results: list[QueryScoreOut]


class GoldChunkOut(BaseModel):
    chunk_id: str
    chunk_hash: str
    page: int
    section: str
    preview: str


class GoldQueryOut(BaseModel):
    eval_id: str
    query: str
    gold_sections: list[str]
    gold_chunk_hashes: list[str]
    gold_count: int
    missing_labels: list[str]
    chunks: list[GoldChunkOut]


class GoldOut(BaseModel):
    eval_set: str
    corpus: str
    corpus_chunk_count: int
    sections: list[str]
    labeled_count: int
    results: list[GoldQueryOut]


class EvalSetBriefOut(BaseModel):
    name: str
    query_count: int
    corpus: str


class EvalSetsOut(BaseModel):
    default: str
    results: list[EvalSetBriefOut]


@router.get("/eval/sets", response_model=EvalSetsOut)
def eval_sets(user: User | None = Depends(require_eval_gate_user)) -> EvalSetsOut:
    results: list[EvalSetBriefOut] = []
    for name in list_golden_sets():
        try:
            golden = load_golden(name=name)
        except (FileNotFoundError, ValueError):
            continue
        results.append(EvalSetBriefOut(name=golden.name, query_count=len(golden.queries), corpus=golden.corpus))
    return EvalSetsOut(default=DEFAULT_EVAL_SET, results=results)


@router.get("/eval/summary", response_model=EvalSummaryOut)
def eval_summary(
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    eval_set: str | None = EvalSetQuery,
) -> EvalSummaryOut:
    """Legacy metrics subset of ``GET /eval/report``. Prefer report; this endpoint stays for old scripts."""
    golden = _load_golden_or_404(eval_set)
    report = report_from_db(db, golden, owner_id=current_owner(user))
    return EvalSummaryOut(
        eval_set=report.eval_set,
        query_count=report.query_count,
        traced_count=report.traced_count,
        missing=report.missing,
        avg_coverage_rate=report.avg_coverage_rate,
        avg_hallucination_rate=report.avg_hallucination_rate,
        term_hit_rate=report.term_hit_rate,
        citation_rate=report.citation_rate,
        refusal_ok_rate=report.refusal_ok_rate,
        gold_labeled_count=report.gold_labeled_count,
        retrieval_scored_count=report.retrieval_scored_count,
        avg_recall_at_k=report.avg_recall_at_k,
        mrr=report.mrr,
        results=[
            QueryScoreOut(
                eval_id=item.eval_id,
                query=item.query,
                expect=item.expect,
                trace_id=item.trace_id,
                coverage_rate=item.coverage_rate,
                hallucination_rate=item.hallucination_rate,
                citation_count=item.citation_count,
                term_hits=item.term_hits,
                term_misses=item.term_misses,
                term_hit=item.term_hit,
                refusal_ok=item.refusal_ok,
                gold_count=item.gold_count,
                retrieved_count=item.retrieved_count,
                gold_hit_count=item.gold_hit_count,
                recall_at_k=item.recall_at_k,
                reciprocal_rank=item.reciprocal_rank,
                first_hit_rank=item.first_hit_rank,
                gold_missing_labels=item.gold_missing_labels,
            )
            for item in report.results
        ],
    )


@router.get("/eval/gold", response_model=GoldOut)
def eval_gold(
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    eval_set: str | None = EvalSetQuery,
) -> GoldOut:
    """展示 golden 标注解析成了哪些 chunk，用于人工核对章节标签是否写错。"""
    golden = _load_golden_or_404(eval_set)
    chunks = corpus_chunks(db, golden.corpus, owner_id=current_owner(user))
    sections: list[str] = []
    for chunk in chunks:
        for section in chunk.sections:
            if section and section not in sections:
                sections.append(section)

    results: list[GoldQueryOut] = []
    for spec in golden.queries:
        resolution = resolve_gold(spec, chunks)
        results.append(
            GoldQueryOut(
                eval_id=spec.id,
                query=spec.query,
                gold_sections=list(spec.gold_sections),
                gold_chunk_hashes=list(spec.gold_chunk_hashes),
                gold_count=len(resolution.chunks),
                missing_labels=list(resolution.missing_labels),
                chunks=[
                    GoldChunkOut(
                        chunk_id=item.chunk_id,
                        chunk_hash=item.chunk_hash,
                        page=item.page,
                        section=item.section,
                        preview=item.preview,
                    )
                    for item in resolution.chunks
                ],
            )
        )

    return GoldOut(
        eval_set=golden.name,
        corpus=golden.corpus,
        corpus_chunk_count=len(chunks),
        sections=sections,
        labeled_count=len([item for item in results if item.gold_count > 0]),
        results=results,
    )


class ReplayRequest(BaseModel):
    reverify: bool = False


@router.get("/eval/report")
def eval_report(
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    reverify: bool = False,
    eval_set: str | None = EvalSetQuery,
) -> dict[str, Any]:
    """Freeze current golden traces into a one-shot report. Does not call the generator LLM."""
    golden = _load_golden_or_404(eval_set)
    return replay_snapshot(db, golden, reverify=reverify, owner_id=current_owner(user))


@router.post("/eval/replay")
def eval_replay(
    body: ReplayRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    eval_set: str | None = EvalSetQuery,
) -> dict[str, Any]:
    """Replay stored traces. ``reverify=true`` re-runs the local Cross-Encoder, still no LLM."""
    golden = _load_golden_or_404(eval_set)
    return replay_snapshot(db, golden, reverify=body.reverify, owner_id=current_owner(user))


@router.post("/eval/compare")
def eval_compare(
    body: dict[str, Any],
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
    reverify: bool = False,
    eval_set: str | None = EvalSetQuery,
) -> dict[str, Any]:
    """Score a Dify (or any) baseline with the same judge and diff against Provident."""
    golden = _load_golden_or_404(eval_set)
    uid = current_owner(user)
    baseline = parse_baseline(body)
    ours = replay_snapshot(db, golden, reverify=False, owner_id=uid)
    scored = bool(baseline.records)
    if scored:
        need_verify = reverify or any(item.verification is None for item in baseline.records.values())
        theirs_report = score_baseline_from_db(db, golden, baseline, reverify=need_verify, owner_id=uid)
        theirs_metrics = report_metrics(theirs_report)
        baseline_results = [asdict(item) for item in theirs_report.results]
    else:
        theirs_metrics = precomputed_metrics(baseline)
        theirs_metrics.setdefault("eval_set", baseline.eval_set)
        baseline_results = []

    comparison = compare_metrics(
        {**ours["metrics"], "eval_set": ours["eval_set"]},
        theirs_metrics,
        ours_corpus=corpus_snapshot(db, golden, owner_id=uid),
        baseline=baseline,
    )
    return {
        "ours": {
            "eval_set": ours["eval_set"],
            "metrics": ours["metrics"],
            "passed": ours["passed"],
            "gate_failures": ours["gate_failures"],
        },
        "baseline": {
            "system": baseline.system,
            "name": baseline.name,
            "eval_set": baseline.eval_set,
            "metrics": theirs_metrics,
            "scored": scored,
        },
        "comparison": comparison,
        "baseline_results": baseline_results,
    }


@router.get("/eval/graph")
def eval_graph(
    db: Session = Depends(get_db),
    user: User | None = Depends(require_eval_gate_user),
) -> dict[str, Any]:
    """Graph-only metrics. Never merged into golden.json gates."""
    from app.eval.graph import graph_eval_payload

    return graph_eval_payload(db, owner_id=current_owner(user))
