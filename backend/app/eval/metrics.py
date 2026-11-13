from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.eval.golden import GoldenQuery, GoldenSet


@dataclass
class QueryScore:
    eval_id: str
    query: str
    expect: str
    trace_id: str | None
    coverage_rate: float | None
    hallucination_rate: float | None
    citation_count: int
    term_hits: list[str] = field(default_factory=list)
    term_misses: list[str] = field(default_factory=list)
    term_hit: bool | None = None
    refusal_ok: bool | None = None
    gold_count: int = 0
    retrieved_count: int = 0
    gold_hit_count: int = 0
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    first_hit_rank: int | None = None
    gold_missing_labels: list[str] = field(default_factory=list)
    off_corpus_evidence_count: int = 0
    off_corpus_evidence_rate: float | None = None


@dataclass
class EvalReport:
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
    avg_off_corpus_evidence_rate: float | None
    results: list[QueryScore]


def evidence_blob(evidence: list[dict[str, Any]] | None) -> str:
    parts = []
    for item in evidence or []:
        parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


def term_overlap(blob: str, terms: tuple[str, ...]) -> tuple[list[str], list[str]]:
    haystack = blob.lower()
    hits: list[str] = []
    misses: list[str] = []
    for term in terms:
        if term.lower() in haystack:
            hits.append(term)
        else:
            misses.append(term)
    return hits, misses


def evidence_hashes(evidence: list[dict[str, Any]] | None) -> list[str]:
    """按检索名次返回 evidence 的 chunk_hash（缺失的位置留空串，保持名次对齐）。"""
    return [str(item.get("chunk_hash") or "") for item in evidence or []]


def off_corpus_evidence(ranked_hashes: list[str], corpus_hashes: set[str]) -> tuple[int, int]:
    """返回 (来自 golden 语料之外的证据条数, 能判定归属的证据条数)。

    对不上 chunk_hash 的证据（旧 trace、外部 baseline）既不算命中也不算越界，
    两侧都不计，避免拿「认不出来」当「跑偏了」。
    """
    known = 0
    outside = 0
    for chunk_hash in ranked_hashes:
        if not chunk_hash:
            continue
        known += 1
        if chunk_hash not in corpus_hashes:
            outside += 1
    return outside, known


def retrieval_hits(ranked_hashes: list[str], gold_hashes: set[str]) -> tuple[int, int | None]:
    """返回 (命中的 gold 数, 第一条命中的名次)；名次从 1 开始。"""
    seen: set[str] = set()
    first_rank: int | None = None
    for position, chunk_hash in enumerate(ranked_hashes, start=1):
        if not chunk_hash or chunk_hash not in gold_hashes or chunk_hash in seen:
            continue
        seen.add(chunk_hash)
        if first_rank is None:
            first_rank = position
    return len(seen), first_rank


def looks_like_refusal(answer: str, verification: dict[str, Any] | None) -> bool:
    if "现有证据不足以回答" in (answer or ""):
        return True
    claims = (verification or {}).get("claims") or []
    return any(item.get("status") == "insufficient" for item in claims)


def score_query(
    spec: GoldenQuery,
    *,
    trace_id: str | None,
    answer: str = "",
    evidence: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
    gold_hashes: Iterable[str] | None = None,
    gold_missing_labels: Iterable[str] = (),
    corpus_hashes: Iterable[str] | None = None,
) -> QueryScore:
    coverage = None if verification is None else verification.get("coverage_rate")
    hallucination = None if verification is None else verification.get("hallucination_rate")
    citation_count = len(citations or [])
    hits: list[str] = []
    misses: list[str] = []
    term_hit: bool | None = None
    refusal_ok: bool | None = None

    if spec.expect == "grounded" and spec.relevant_terms:
        hits, misses = term_overlap(evidence_blob(evidence), spec.relevant_terms)
        needed = max(1, (len(spec.relevant_terms) + 1) // 2)
        term_hit = len(hits) >= needed
    elif spec.expect == "refusal":
        refusal_ok = looks_like_refusal(answer, verification)

    gold = {h for h in (gold_hashes or ()) if h}
    ranked = evidence_hashes(evidence)
    hit_count, first_rank = retrieval_hits(ranked, gold) if gold else (0, None)
    recall_at_k = None
    reciprocal_rank = None
    if gold and trace_id:
        recall_at_k = round(hit_count / len(gold), 6)
        reciprocal_rank = round(1.0 / first_rank, 6) if first_rank else 0.0

    off_corpus_count = 0
    off_corpus_rate = None
    if corpus_hashes is not None and trace_id:
        known_corpus = {h for h in corpus_hashes if h}
        off_corpus_count, identified = off_corpus_evidence(ranked, known_corpus)
        off_corpus_rate = round(off_corpus_count / identified, 6) if identified else None

    return QueryScore(
        eval_id=spec.id,
        query=spec.query,
        expect=spec.expect,
        trace_id=trace_id,
        coverage_rate=None if coverage is None else float(coverage),
        hallucination_rate=None if hallucination is None else float(hallucination),
        citation_count=citation_count,
        term_hits=hits,
        term_misses=misses,
        term_hit=term_hit,
        refusal_ok=refusal_ok,
        gold_count=len(gold),
        retrieved_count=len(ranked),
        gold_hit_count=hit_count,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        first_hit_rank=first_rank,
        gold_missing_labels=[label for label in gold_missing_labels],
        off_corpus_evidence_count=off_corpus_count,
        off_corpus_evidence_rate=off_corpus_rate,
    )


def aggregate(golden: GoldenSet, scores: list[QueryScore]) -> EvalReport:
    traced = [item for item in scores if item.trace_id]
    missing = [item.eval_id for item in scores if not item.trace_id]
    coverages = [item.coverage_rate for item in traced if item.coverage_rate is not None]
    hallucinations = [item.hallucination_rate for item in traced if item.hallucination_rate is not None]
    term_flags = [item.term_hit for item in traced if item.term_hit is not None]
    refusal_flags = [item.refusal_ok for item in traced if item.refusal_ok is not None]
    cited = [item for item in traced if item.expect == "grounded"]
    retrieval = [item for item in traced if item.gold_count > 0 and item.recall_at_k is not None]
    # 只统计 grounded 题：拒答题本来就该在语料里找不到东西，它的证据来自哪儿不说明问题。
    off_corpus = [float(item.off_corpus_evidence_rate) for item in cited if item.off_corpus_evidence_rate is not None]

    return EvalReport(
        eval_set=golden.name,
        query_count=len(golden.queries),
        traced_count=len(traced),
        missing=missing,
        avg_coverage_rate=_mean(coverages),
        avg_hallucination_rate=_mean(hallucinations),
        term_hit_rate=_mean([1.0 if flag else 0.0 for flag in term_flags]),
        citation_rate=_mean([1.0 if item.citation_count > 0 else 0.0 for item in cited]) if cited else None,
        refusal_ok_rate=_mean([1.0 if flag else 0.0 for flag in refusal_flags]),
        gold_labeled_count=len([item for item in scores if item.gold_count > 0]),
        retrieval_scored_count=len(retrieval),
        avg_recall_at_k=_mean([float(item.recall_at_k) for item in retrieval]),
        mrr=_mean([float(item.reciprocal_rank or 0.0) for item in retrieval]),
        avg_off_corpus_evidence_rate=_mean(off_corpus),
        results=scores,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)
